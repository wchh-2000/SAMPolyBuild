"""Coordinate conversion primitives for tile-based processing.

The foundation intentionally supports translation only. Rotation, scaling, CRS
conversion, and affine georeferencing belong to later workflow stages.
"""

import math
from collections.abc import Sequence


def _coordinate_pair(value: Sequence[float], name: str) -> tuple[float, float]:
    """Validate and normalize a JSON-compatible two-dimensional coordinate."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numeric values.")
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly two numeric values.") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{name} coordinates must be finite.")
    return x, y


def tile_to_global(point: Sequence[float], offset: Sequence[float]) -> list[float]:
    """Translate one tile-local ``[x, y]`` point into global pixel coordinates."""
    tile_x, tile_y = _coordinate_pair(point, "point")
    offset_x, offset_y = _coordinate_pair(offset, "offset")
    return [tile_x + offset_x, tile_y + offset_y]


def global_to_tile(point: Sequence[float], offset: Sequence[float]) -> list[float]:
    """Translate one global ``[x, y]`` point into tile-local pixel coordinates."""
    global_x, global_y = _coordinate_pair(point, "point")
    offset_x, offset_y = _coordinate_pair(offset, "offset")
    return [global_x - offset_x, global_y - offset_y]
