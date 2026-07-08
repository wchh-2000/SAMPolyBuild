"""State container for Prompt-first tile review sessions."""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
from uuid import uuid4

from .tile_manager import PolygonRecord


PromptCallback = Callable[[list[tuple[int, int]], list[tuple[int, int]]], dict]


class PromptReviewSessionState(str, Enum):
    """Lifecycle states for one Prompt-first building extraction session."""

    IDLE = "idle"
    RUNNING = "running"
    ACCEPTED = "accepted"
    CANCELLED = "cancelled"


@dataclass
class PromptReviewSession:
    """Track one pending Prompt polygon before it is committed to TileResult."""

    session_id: str
    tile_id: str
    bbox: list[float]
    bbox_source: str
    selected_auto_polygon_id: str | None = None
    selected_auto_score: float | None = None
    selected_auto_distance: float | None = None
    positive_points: list[tuple[int, int]] = field(default_factory=list)
    negative_points: list[tuple[int, int]] = field(default_factory=list)
    history: list[PolygonRecord] = field(default_factory=list)
    current_polygon: PolygonRecord | None = None
    current_prediction: dict | None = None
    is_modified: bool = False
    state: PromptReviewSessionState = PromptReviewSessionState.IDLE

    @classmethod
    def start(
        cls,
        tile_id: str,
        bbox: list[float],
        bbox_source: str,
        selected_auto_polygon_id: str | None = None,
        selected_auto_score: float | None = None,
        selected_auto_distance: float | None = None,
    ) -> "PromptReviewSession":
        """Start one running session with a fixed bbox provider decision."""
        if not bbox or len(bbox) != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2].")
        return cls(
            session_id=f"prompt_{uuid4().hex}",
            tile_id=str(tile_id).strip(),
            bbox=[float(value) for value in bbox],
            bbox_source=str(bbox_source).strip(),
            selected_auto_polygon_id=selected_auto_polygon_id,
            selected_auto_score=(
                float(selected_auto_score)
                if selected_auto_score is not None else None
            ),
            selected_auto_distance=(
                float(selected_auto_distance)
                if selected_auto_distance is not None else None
            ),
            state=PromptReviewSessionState.RUNNING,
        )

    def add_positive_point(
        self,
        point: tuple[int, int],
        prompt_callback: PromptCallback,
    ) -> PolygonRecord:
        """Append one positive point and refresh the preview polygon."""
        return self._apply_click(point, is_positive=True, prompt_callback=prompt_callback)

    def add_negative_point(
        self,
        point: tuple[int, int],
        prompt_callback: PromptCallback,
    ) -> PolygonRecord:
        """Append one negative point and refresh the preview polygon."""
        return self._apply_click(point, is_positive=False, prompt_callback=prompt_callback)

    def accept(self, polygon_id: str, source_name: str = "prompt") -> PolygonRecord:
        """Finalize the latest preview polygon as one canonical saved polygon."""
        self._require_running()
        if self.current_prediction is None or self.current_polygon is None:
            raise RuntimeError("Prompt session has no current polygon to accept.")
        self.state = PromptReviewSessionState.ACCEPTED
        accepted = deepcopy(self.current_polygon)
        accepted.id = str(polygon_id).strip()
        accepted.source = str(source_name).strip()
        return accepted

    def cancel(self) -> None:
        """Discard the in-memory preview without mutating any TileResult."""
        self._require_running()
        self.state = PromptReviewSessionState.CANCELLED

    def _apply_click(
        self,
        point: tuple[int, int],
        is_positive: bool,
        prompt_callback: PromptCallback,
    ) -> PolygonRecord:
        """Run Prompt with accumulated points and keep only valid preview results."""
        self._require_running()
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Prompt point must be an [x, y] pair.")
        normalized = (int(point[0]), int(point[1]))
        candidate_positive = [*self.positive_points]
        candidate_negative = [*self.negative_points]
        if is_positive:
            candidate_positive.append(normalized)
        else:
            candidate_negative.append(normalized)
        prediction = prompt_callback(candidate_positive, candidate_negative)
        candidate = self._polygon_from_prediction(prediction, candidate_positive, candidate_negative)

        self.positive_points = candidate_positive
        self.negative_points = candidate_negative
        self.current_prediction = deepcopy(prediction)
        self.current_polygon = candidate
        self.history.append(deepcopy(candidate))
        self.is_modified = True
        return deepcopy(candidate)

    def _polygon_from_prediction(
        self,
        prediction: dict,
        positive_points: list[tuple[int, int]],
        negative_points: list[tuple[int, int]],
    ) -> PolygonRecord:
        """Normalize one Prompt callback result into a preview polygon."""
        if not isinstance(prediction, dict):
            raise ValueError("Prompt prediction must be a JSON-style object.")
        missing = [
            name for name in ("points_tile", "score", "bbox_tile")
            if name not in prediction
        ]
        if missing:
            raise ValueError(f"Prompt prediction is missing fields: {missing}")

        click_history = [
            {
                "type": "positive",
                "point": [float(x), float(y)],
                "mode": "prompt_first",
                "session_id": self.session_id,
            }
            for x, y in positive_points
        ] + [
            {
                "type": "negative",
                "point": [float(x), float(y)],
                "mode": "prompt_first",
                "session_id": self.session_id,
            }
            for x, y in negative_points
        ]
        metadata = deepcopy(prediction.get("metadata") or {})
        metadata.update(
            {
                "created_by": "prompt_first_review",
                "bbox_source": self.bbox_source,
                "selected_auto_polygon_id": self.selected_auto_polygon_id,
                "selected_auto_score": self.selected_auto_score,
                "selected_auto_distance": self.selected_auto_distance,
                "prompt_version": len(self.history) + 1,
                "prompt_session_id": self.session_id,
            }
        )
        return PolygonRecord(
            id=f"{self.tile_id}__prompt_preview__",
            tile_id=self.tile_id,
            status="active",
            source="prompt",
            score=prediction["score"],
            points_tile=prediction["points_tile"],
            bbox_tile=prediction["bbox_tile"],
            points_global=[],
            click_history=click_history,
            metadata=metadata,
        )

    def _require_running(self) -> None:
        """Reject invalid state transitions with an explicit lifecycle error."""
        if self.state is not PromptReviewSessionState.RUNNING:
            raise RuntimeError(f"PromptReviewSession is not running: state={self.state.value}")
