from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as functional
from monai.visualize.class_activation_maps import GradCAMpp
from PIL import Image

from spina_bifida.config import InferenceThresholdSettings
from spina_bifida.inference.artifacts import (
    InvalidRoi,
    clamp_roi,
    image_asset,
    pad_roi,
    render_heatmap,
    render_padded_heatmap,
)
from spina_bifida.inference.contracts import (
    AnalysisResult,
    AssessmentCategory,
    AssessmentResult,
    AssetType,
    FrameResult,
    GeneratedAsset,
    InputFrame,
    ModelMetadata,
    PlaneResult,
    PlaneType,
    QualityResult,
    RoiResult,
)
from spina_bifida.inference.model_loader import ModelBundle


class InferenceService:
    def __init__(
        self,
        models: ModelBundle,
        thresholds: InferenceThresholdSettings | None = None,
    ) -> None:
        self.models = models
        self.thresholds = thresholds or InferenceThresholdSettings()

    def analyze(self, frames: Sequence[InputFrame]) -> AnalysisResult:
        if not 1 <= len(frames) <= 2:
            raise ValueError("inference requires one or two frames")
        results = tuple(self._analyze_frame(frame) for frame in frames)
        warnings = tuple(
            dict.fromkeys(warning for result in results for warning in result.warnings),
        )
        return AnalysisResult(
            frames=results,
            warnings=warnings,
            model=ModelMetadata(
                version=self.models.version,
                device=self.models.device.type,
                service="spina-bifida-inference",
            ),
            thresholds=self.thresholds,
        )

    def _analyze_frame(self, frame: InputFrame) -> FrameResult:
        with Image.open(frame.path) as opened:
            original = opened.convert("RGB")
        original_asset = image_asset(AssetType.ORIGINAL, original)
        detection = self._detect(original)
        if detection is None:
            return self._rejected_frame(frame.frame_id, original_asset, "no_detection")

        box, plane_score, plane_class = detection
        plane_type = {1: PlaneType.SAGITTAL, 2: PlaneType.AXIAL}.get(
            plane_class,
            PlaneType.UNKNOWN,
        )
        try:
            bounded_box = clamp_roi(box, original.width, original.height)
        except InvalidRoi:
            return self._rejected_frame(frame.frame_id, original_asset, "invalid_roi")
        if plane_type == PlaneType.UNKNOWN:
            return self._rejected_frame(
                frame.frame_id,
                original_asset,
                "unknown_plane",
                plane_score=plane_score,
            )

        cropped = original.crop(bounded_box)
        display_box = pad_roi(bounded_box, original.width, original.height)
        display_roi = original.crop(display_box)
        tensor = self._prepare(cropped).to(self.models.device)
        quality_model, assessment_model = self._classifiers(plane_type)
        quality_score = self._predict(tensor, quality_model)
        assessment_score = self._predict(tensor, assessment_model)
        exact_heatmap = self._heatmap(tensor, assessment_model, cropped.size)
        display_heatmap = render_padded_heatmap(
            original,
            bounded_box,
            display_box,
            exact_heatmap,
        )
        quality_accepted = quality_score >= self.thresholds.quality
        warnings = () if quality_accepted else ("quality_below_threshold",)
        category = (
            AssessmentCategory.ATTENTION_REQUIRED
            if assessment_score >= self.thresholds.assessment
            else AssessmentCategory.NO_EXPRESSED_SIGNAL
        )
        return FrameResult(
            frame_id=frame.frame_id,
            analyzable=True,
            quality=QualityResult(score=quality_score, accepted=quality_accepted),
            plane=PlaneResult(type=plane_type, score=plane_score),
            roi=RoiResult(box=bounded_box, display_box=display_box),
            assessment=AssessmentResult(category=category, score=assessment_score),
            explanations=("assessment_regions",),
            warnings=warnings,
            assets=(
                original_asset,
                image_asset(AssetType.ROI, display_roi),
                image_asset(AssetType.HEATMAP, display_heatmap),
            ),
        )

    def _detect(
        self,
        image: Image.Image,
    ) -> tuple[tuple[float, float, float, float], float, int] | None:
        prediction = self.models.yolo.predict(
            image,
            verbose=False,
            conf=self.thresholds.plane_detection,
            device=str(self.models.device),
        )[0]
        if prediction.boxes.shape[0] == 0:
            return None
        values = prediction.boxes.data[0].detach().cpu().tolist()
        left, top, right, bottom, confidence, plane_class = values
        return (
            (float(left), float(top), float(right), float(bottom)),
            max(0.0, min(1.0, float(confidence))),
            int(plane_class),
        )

    def _classifiers(self, plane: PlaneType) -> tuple[Any, Any]:
        if plane == PlaneType.AXIAL:
            return self.models.axial_quality, self.models.axial_pathology
        return self.models.sagittal_quality, self.models.sagittal_pathology

    def _prepare(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.expand_dims(array, axis=2)
        array = array.transpose(1, 0, 2).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        minimum = tensor.amin()
        maximum = tensor.amax()
        denominator = maximum - minimum
        if float(denominator) > 1e-8:
            tensor = (tensor - minimum) / denominator
        else:
            tensor = torch.zeros_like(tensor)
        return cast(
            torch.Tensor,
            functional.interpolate(tensor, size=(255, 255), mode="area"),
        )

    def _predict(self, tensor: torch.Tensor, model: Any) -> float:
        model_input = tensor.permute(0, 1, 3, 2) if self.models.version == "v2" else tensor
        with torch.inference_mode():
            output = model(model_input)
            score = torch.sigmoid(output[0][1]).item()
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def _heatmap(tensor: torch.Tensor, model: Any, output_size: tuple[int, int]) -> Image.Image:
        grad_cam = GradCAMpp(nn_module=model, target_layers="class_layers.relu")
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            cam = grad_cam(x=tensor)
        return render_heatmap(tensor, cam, output_size)

    @staticmethod
    def _rejected_frame(
        frame_id: str,
        original_asset: GeneratedAsset,
        warning: str,
        *,
        plane_score: float = 0.0,
    ) -> FrameResult:
        return FrameResult(
            frame_id=frame_id,
            analyzable=False,
            quality=QualityResult(score=0.0, accepted=False),
            plane=PlaneResult(type=PlaneType.UNKNOWN, score=plane_score),
            roi=None,
            assessment=None,
            explanations=(),
            warnings=(warning,),
            assets=(original_asset,),
        )
