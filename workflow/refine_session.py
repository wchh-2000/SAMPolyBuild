"""Model-independent lifecycle state for iterative polygon refinement."""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
from uuid import uuid4

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
    history: list[PolygonRecord] = field(default_factory=list)
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
        self._require_running()
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Refine point must be an [x, y] pair.")
        normalized_point = (int(point[0]), int(point[1]))
        candidate_points = [*self.positive_points, normalized_point]
        prediction = prompt_callback(candidate_points, list(self.negative_points))
        candidate = self._polygon_from_prediction(prediction, candidate_points)

        # Mutate session state only after prediction validation succeeds.
        self.positive_points = candidate_points
        self.current_polygon = candidate
        self.history.append(deepcopy(candidate))
        self.is_modified = True
        return deepcopy(candidate)

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
        points: list[tuple[int, int]],
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
        click_history.extend(
            {
                "type": "positive",
                "point": [float(x), float(y)],
                "mode": "refine",
                "session_id": self.session_id,
            }
            for x, y in points
        )
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

    def _require_running(self) -> None:
        """Reject invalid lifecycle transitions with an explicit error."""
        if self.state is not RefineSessionState.RUNNING:
            raise RuntimeError(f"RefineSession is not running: state={self.state.value}")
