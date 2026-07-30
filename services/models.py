import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class PromptPoint:
    x: float
    y: float
    label: int

    def __post_init__(self):
        x = float(self.x)
        y = float(self.y)
        label_value = float(self.label)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Prompt coordinates must be finite numbers.")
        if not math.isfinite(label_value) or label_value not in (0.0, 1.0):
            raise ValueError("Prompt label must be 0 (negative) or 1 (positive).")
        label = int(label_value)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "label", label)


@dataclass
class PromptServiceConfig:
    checkpoint: str
    model_config: str
    gpu: int = 0
    device: Optional[str] = None
    bbox_size: int = 256
    bbox_mode: str = "auto"
    auto_min_score: float = 0.0
    include_bbox_center_prompt: bool = True
    crop_padding_ratio: float = 0.1


@dataclass
class InferenceRequest:
    image_path: str
    prompts: List[PromptPoint]
    bbox: Optional[List[float]] = None
    bbox_mode: Optional[str] = None
    bbox_size: Optional[int] = None
    auto_results: Optional[str] = None
    auto_image_id: Optional[str] = None
    auto_min_score: Optional[float] = None


@dataclass
class InferenceResponse:
    success: bool
    polygon: List[List[float]] = field(default_factory=list)
    bbox: Optional[List[int]] = None
    score: Optional[float] = None
    message: str = ""
    model_score: Optional[float] = None
    bbox_source: Optional[str] = None

    def to_dict(self):
        return asdict(self)
