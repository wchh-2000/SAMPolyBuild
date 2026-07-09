"""Model-independent lifecycle state for iterative polygon refinement."""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import inspect
from typing import Any, Callable
from uuid import uuid4

import cv2
import numpy as np

from .tile_manager import PolygonRecord


class RefineSessionState(str, Enum):
    """Allowed lifecycle states for one polygon refinement session."""

    IDLE = "idle"
    RUNNING = "running"
    ACCEPTED = "accepted"
    CANCELLED = "cancelled"


PromptCallback = Callable[[list[tuple[int, int]], list[tuple[int, int]]], dict]


@dataclass
class RefineSession:
    """Keep iterative Prompt changes isolated until the user accepts them."""

    session_id: str
    tile_id: str
    polygon_id: str
    original_polygon: PolygonRecord
    current_polygon: PolygonRecord
    positive_points: list[tuple[int, int]] = field(default_factory=list)
    negative_points: list[tuple[int, int]] = field(default_factory=list)
    click_order: list[dict[str, Any]] = field(default_factory=list)
    history: list[PolygonRecord] = field(default_factory=list)
    current_mask: Any = None
    is_modified: bool = False
    state: RefineSessionState = RefineSessionState.IDLE

    @classmethod
    def start(cls, polygon: PolygonRecord) -> "RefineSession":
        """Start a running session from an independently copied polygon."""
        if not isinstance(polygon, PolygonRecord):
            raise TypeError("RefineSession requires a PolygonRecord.")
        original = deepcopy(polygon)
        return cls(
            session_id=f"refine_{uuid4().hex}",
            tile_id=polygon.tile_id,
            polygon_id=polygon.id,
            original_polygon=original,
            current_polygon=deepcopy(original),
            state=RefineSessionState.RUNNING,
        )

    def add_positive_point(
        self,
        point: tuple[int, int],
        prompt_callback: PromptCallback,
    ) -> PolygonRecord:
        """Run Prompt with all accumulated positive points and update memory only."""
        return self._apply_click(point, is_positive=True, prompt_callback=prompt_callback)

    def add_negative_point(
        self,
        point: tuple[int, int],
        prompt_callback: PromptCallback,
    ) -> PolygonRecord:
        """Run Prompt with all accumulated negative clicks and update memory only."""
        return self._apply_click(point, is_positive=False, prompt_callback=prompt_callback)

    def can_accept_negative_point(
        self,
        point: tuple[int, int],
        allow_outside_negative_points: bool = False,
    ) -> bool:
        """Allow negative points only inside the current preview mask or polygon by default."""
        self._require_running()
        if allow_outside_negative_points:
            return True
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return False
        normalized = (int(point[0]), int(point[1]))
        if self.current_mask is not None:
            return _point_inside_mask(self.current_mask, normalized)
        if self.current_polygon is None:
            return False
        return _point_inside_polygon(self.current_polygon, normalized)

    def get_prompt_points_and_labels(self) -> tuple[list[list[int]], list[int]]:
        """Return ordered prompt clicks and labels in exact user-click sequence."""
        points = [
            [int(item["point"][0]), int(item["point"][1])]
            for item in self.click_order
        ]
        labels = [int(item["label"]) for item in self.click_order]
        return points, labels

    def accept(self) -> PolygonRecord:
        """Finish the session and return the accepted polygon for TileResult replacement."""
        self._require_running()
        self.state = RefineSessionState.ACCEPTED
        return deepcopy(self.current_polygon)

    def cancel(self) -> PolygonRecord:
        """Finish the session and return the untouched original polygon."""
        self._require_running()
        self.current_polygon = deepcopy(self.original_polygon)
        self.state = RefineSessionState.CANCELLED
        return deepcopy(self.original_polygon)

    def _polygon_from_prediction(
        self,
        prediction: dict,
        click_order: list[dict[str, Any]],
    ) -> PolygonRecord:
        """Convert one adapter prediction to a canonical, validated PolygonRecord."""
        if not isinstance(prediction, dict):
            raise ValueError("Prompt prediction must be a JSON-style object.")
        missing = [
            name for name in ("points_tile", "score", "bbox_tile")
            if name not in prediction
        ]
        if missing:
            raise ValueError(f"Prompt prediction is missing fields: {missing}")
        click_history = deepcopy(self.original_polygon.click_history)
        click_history.extend(deepcopy(item) for item in click_order)
        metadata = deepcopy(self.original_polygon.metadata)
        metadata.update(deepcopy(prediction.get("metadata") or {}))
        metadata["refine_session_id"] = self.session_id
        return PolygonRecord(
            id=self.original_polygon.id,
            tile_id=self.original_polygon.tile_id,
            status=self.original_polygon.status,
            source="refine",
            score=prediction["score"],
            points_tile=prediction["points_tile"],
            bbox_tile=prediction["bbox_tile"],
            points_global=[],
            click_history=click_history,
            metadata=metadata,
        )

    def _apply_click(
        self,
        point: tuple[int, int],
        is_positive: bool,
        prompt_callback: PromptCallback,
    ) -> PolygonRecord:
        """Run Prompt with ordered positive/negative points and update memory only."""
        self._require_running()
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Refine point must be an [x, y] pair.")
        normalized_point = (int(point[0]), int(point[1]))
        candidate_positive = [*self.positive_points]
        candidate_negative = [*self.negative_points]
        candidate_click_order = [deepcopy(item) for item in self.click_order]
        label = 1 if is_positive else 0
        click_type = "positive" if is_positive else "negative"
        if is_positive:
            candidate_positive.append(normalized_point)
        else:
            candidate_negative.append(normalized_point)
        candidate_click_order.append(
            {
                "type": click_type,
                "point": [float(normalized_point[0]), float(normalized_point[1])],
                "label": label,
                "mode": "refine",
                "session_id": self.session_id,
            }
        )
        prompt_points = [
            [int(item["point"][0]), int(item["point"][1])]
            for item in candidate_click_order
        ]
        prompt_labels = [int(item["label"]) for item in candidate_click_order]
        prediction = self._call_prompt_callback(
            prompt_callback,
            candidate_positive,
            candidate_negative,
            prompt_points,
            prompt_labels,
            candidate_click_order,
        )
        candidate = self._polygon_from_prediction(prediction, candidate_click_order)

        self.positive_points = candidate_positive
        self.negative_points = candidate_negative
        self.click_order = candidate_click_order
        self.current_mask = _extract_prediction_mask(prediction)
        self.current_polygon = candidate
        self.history.append(deepcopy(candidate))
        self.is_modified = True
        return deepcopy(candidate)

    def _call_prompt_callback(
        self,
        prompt_callback: PromptCallback,
        positive_points: list[tuple[int, int]],
        negative_points: list[tuple[int, int]],
        prompt_points: list[list[int]],
        prompt_labels: list[int],
        click_order: list[dict[str, Any]],
    ) -> dict:
        """Pass ordered prompt metadata when the callback can accept it."""
        signature = inspect.signature(prompt_callback)
        kwargs = {}
        for name, value in (
            ("prompt_points", prompt_points),
            ("prompt_labels", prompt_labels),
            ("click_order", click_order),
        ):
            if name in signature.parameters:
                kwargs[name] = value
        return prompt_callback(positive_points, negative_points, **kwargs)

    def _require_running(self) -> None:
        """Reject invalid lifecycle transitions with an explicit error."""
        if self.state is not RefineSessionState.RUNNING:
            raise RuntimeError(f"RefineSession is not running: state={self.state.value}")


def _extract_prediction_mask(prediction: dict) -> np.ndarray | None:
    """Convert one in-memory prediction mask into a stable 2D uint8 array when available."""
    mask = prediction.get("mask") if isinstance(prediction, dict) else None
    if mask is None:
        return None
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        mask_array = mask_array[0]
    if mask_array.ndim != 2:
        return None
    return (mask_array > 0).astype(np.uint8)


def _point_inside_mask(mask, point: tuple[int, int]) -> bool:
    """Return True when one tile-space point falls on a foreground mask pixel."""
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        mask_array = mask_array[0]
    if mask_array.ndim != 2:
        return False
    x, y = int(point[0]), int(point[1])
    if not (0 <= x < mask_array.shape[1] and 0 <= y < mask_array.shape[0]):
        return False
    return bool(mask_array[y, x] > 0)


def _point_inside_polygon(polygon: PolygonRecord, point: tuple[int, int]) -> bool:
    """Return True when one tile-space point lies inside or on the polygon boundary."""
    if polygon is None or not polygon.points_tile:
        return False
    contour = np.asarray(polygon.points_tile, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0
