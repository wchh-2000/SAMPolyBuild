import math
import threading
from pathlib import Path
from typing import List, Tuple

from .models import InferenceRequest, InferenceResponse, PromptPoint, PromptServiceConfig
from tools.prompt_inference_utils import (
    build_prompt_predictor,
    infer_polygon_with_prompts,
    load_prompt_settings,
    load_rgb_image,
    normalize_bbox,
    normalize_prompts,
    select_bbox,
)


class PromptInferenceService:
    def __init__(self, config: PromptServiceConfig):
        self.config = config
        self.settings = None
        self.predictor = None
        self.model = None
        self.device = None
        self._initialized = False
        self._inference_lock = threading.Lock()

    @property
    def initialized(self):
        return self._initialized

    def initialize(self) -> None:
        if self._initialized:
            return
        self._validate_service_config()
        self.settings = load_prompt_settings(
            self.config.model_config,
            checkpoint=self.config.checkpoint,
        )
        self.predictor, self.device = build_prompt_predictor(
            self.settings,
            gpu=self.config.gpu,
            device=self.config.device,
        )
        self.model = self.predictor.model
        self._initialized = True

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        if not self._initialized or self.predictor is None:
            return self._failure("PromptInferenceService is not initialized.")

        try:
            prompts = self._validate_request_prompts(request.prompts)
            image = load_rgb_image(request.image_path)
            image_height, image_width = image.shape[:2]
            normalized_prompts = normalize_prompts(prompts, image_width, image_height)
            bbox, bbox_source, bbox_message = self._resolve_bbox(
                request,
                normalized_prompts,
                image_width,
                image_height,
            )

            with self._inference_lock:
                prediction = infer_polygon_with_prompts(
                    predictor=self.predictor,
                    image=image,
                    bbox=bbox,
                    prompts=normalized_prompts,
                    include_bbox_center_prompt=self.config.include_bbox_center_prompt,
                    crop_margin=self.config.crop_padding_ratio,
                    multi_mask=bool(getattr(self.settings, "multi_mask", True)),
                    max_distance=int(getattr(self.settings, "max_distance", 10)),
                )

            polygon = [
                [float(x), float(y)]
                for x, y in prediction["polygon"]
            ]
            message = "ok" if not bbox_message else f"ok; {bbox_message}"
            return InferenceResponse(
                success=True,
                polygon=polygon,
                bbox=[int(value) for value in bbox],
                score=float(prediction["score"]),
                model_score=float(prediction["model_score"]),
                bbox_source=bbox_source,
                message=message,
            )
        except Exception as exc:
            return self._failure(f"{type(exc).__name__}: {exc}")

    def close(self) -> None:
        self.predictor = None
        self.model = None
        self.settings = None
        self.device = None
        self._initialized = False

    def _validate_service_config(self) -> None:
        if not Path(self.config.model_config).is_file():
            raise FileNotFoundError(f"Prompt config not found: {self.config.model_config}")
        if not Path(self.config.checkpoint).is_file():
            raise FileNotFoundError(f"Prompt checkpoint not found: {self.config.checkpoint}")
        if self.config.bbox_mode not in ("auto", "fixed"):
            raise ValueError("bbox_mode must be 'auto' or 'fixed'.")
        if self.config.bbox_size <= 0:
            raise ValueError("bbox_size must be greater than zero.")
        if not math.isfinite(float(self.config.crop_padding_ratio)) or self.config.crop_padding_ratio < 0:
            raise ValueError("crop_padding_ratio must be greater than or equal to zero.")

    def _validate_request_prompts(self, prompts: List[PromptPoint]) -> List[PromptPoint]:
        if not prompts:
            raise ValueError("At least one prompt point is required.")
        normalized = []
        for prompt in prompts:
            if isinstance(prompt, PromptPoint):
                normalized.append(prompt)
            elif isinstance(prompt, dict):
                normalized.append(PromptPoint(**prompt))
            else:
                raise TypeError("Each prompt must be a PromptPoint or a compatible dictionary.")
        if not any(prompt.label == 1 for prompt in normalized):
            raise ValueError("At least one positive prompt point is required.")
        return normalized

    def _resolve_bbox(
        self,
        request: InferenceRequest,
        prompts: List[Tuple[float, float, int]],
        image_width: int,
        image_height: int,
    ):
        if request.bbox is not None:
            bbox = normalize_bbox(request.bbox, image_width, image_height)
            return bbox, "request", ""

        first_positive = next((x, y) for x, y, label in prompts if label == 1)
        mode = request.bbox_mode or self.config.bbox_mode
        bbox_size = request.bbox_size if request.bbox_size is not None else self.config.bbox_size
        auto_min_score = (
            request.auto_min_score
            if request.auto_min_score is not None
            else self.config.auto_min_score
        )
        bbox, source, info = select_bbox(
            mode=mode,
            click=first_positive,
            image_width=image_width,
            image_height=image_height,
            bbox_size=bbox_size,
            auto_results=request.auto_results,
            auto_image_id=request.auto_image_id,
            auto_min_score=auto_min_score,
            image_path=request.image_path,
        )
        fallback_message = ""
        if info.get("fallback_used"):
            fallback_message = f"bbox fallback to fixed: {info.get('fallback_reason', 'unknown reason')}"
        return bbox, source, fallback_message

    @staticmethod
    def _failure(message):
        return InferenceResponse(
            success=False,
            polygon=[],
            bbox=None,
            score=None,
            model_score=None,
            bbox_source=None,
            message=message,
        )
