from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from spina_bifida.config import InferenceThresholdSettings
from spina_bifida.inference.artifacts import (
    InvalidRoi,
    clamp_roi,
    normalize_array,
    pad_roi,
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
from spina_bifida.inference.model_loader import CudaUnavailable, resolve_device
from spina_bifida.inference.processor import InferenceService


def make_frame(*, frame_id: str = "frame-1") -> FrameResult:
    return FrameResult(
        frame_id=frame_id,
        analyzable=True,
        quality=QualityResult(score=0.94, accepted=True),
        plane=PlaneResult(type=PlaneType.SAGITTAL, score=0.91),
        roi=RoiResult(
            box=(10, 20, 110, 120),
            display_box=(5, 15, 115, 125),
        ),
        assessment=AssessmentResult(
            category=AssessmentCategory.ATTENTION_REQUIRED,
            score=0.82,
        ),
        explanations=("quality_regions", "assessment_regions"),
        assets=(
            GeneratedAsset(AssetType.ORIGINAL, 320, 240, "a" * 64),
            GeneratedAsset(AssetType.ROI, 100, 100, "b" * 64),
            GeneratedAsset(AssetType.HEATMAP, 100, 100, "c" * 64),
        ),
    )


def make_result(*frames: FrameResult) -> AnalysisResult:
    return AnalysisResult(
        frames=frames or (make_frame(),),
        warnings=(),
        model=ModelMetadata(
            version="v1",
            device="cpu",
            service="spina-bifida-inference",
        ),
    )


def test_analysis_result_orders_clinical_fields() -> None:
    payload = make_result().to_dict()
    frame = payload["frames"][0]

    assert list(frame)[:6] == [
        "analyzable",
        "quality",
        "plane",
        "roi",
        "assessment",
        "explanations",
    ]
    assert frame["assessment"]["category"] in {
        "no_expressed_signal",
        "attention_required",
    }
    assert frame["roi"] == {
        "box": [10, 20, 110, 120],
        "display_box": [5, 15, 115, 125],
    }
    assert 0.0 <= frame["assessment"]["score"] <= 1.0


def test_result_serializes_one_or_two_frames_and_machine_metadata() -> None:
    payload = make_result(make_frame(), make_frame(frame_id="frame-2")).to_dict()

    assert [frame["frame_id"] for frame in payload["frames"]] == ["frame-1", "frame-2"]
    assert payload["model"] == {
        "version": "v1",
        "device": "cpu",
        "service": "spina-bifida-inference",
    }
    assert payload["thresholds"] == {
        "plane_detection": 0.1,
        "quality": 0.5,
        "assessment": 0.5,
        "warning_margin": 0.1,
    }
    assert payload["warnings"] == []


def test_unknown_plane_no_roi_and_rejected_quality_have_explicit_nulls() -> None:
    rejected = FrameResult(
        frame_id="rejected",
        analyzable=False,
        quality=QualityResult(score=0.12, accepted=False),
        plane=PlaneResult(type=PlaneType.UNKNOWN, score=0.0),
        roi=None,
        assessment=None,
        explanations=(),
        warnings=("no_detection",),
        assets=(GeneratedAsset(AssetType.ORIGINAL, 32, 24, "d" * 64),),
    )

    payload = make_result(rejected).to_dict()["frames"][0]

    assert payload["plane"] == {"type": "unknown", "score": 0.0}
    assert payload["roi"] is None
    assert payload["assessment"] is None
    assert payload["warnings"] == ["no_detection"]


def test_asset_descriptors_never_serialize_binary_content() -> None:
    asset = GeneratedAsset(
        AssetType.HEATMAP,
        10,
        20,
        "e" * 64,
        content=b"private-image-content",
    )

    assert asset.to_dict() == {
        "kind": "heatmap",
        "width": 10,
        "height": 20,
        "checksum": "e" * 64,
        "media_type": "image/png",
    }
    assert "private-image-content" not in str(asset.to_dict())


def test_input_frame_keeps_internal_path_out_of_result_contract() -> None:
    input_frame = InputFrame("frame-1", Path("/private/data/frame.png"))

    assert input_frame.frame_id == "frame-1"
    assert "private/data" not in str(make_result().to_dict())


def test_scores_must_be_probabilities() -> None:
    with pytest.raises(ValueError):
        QualityResult(score=1.1, accepted=True)
    with pytest.raises(ValueError):
        AssessmentResult(AssessmentCategory.NO_EXPRESSED_SIGNAL, -0.1)


def test_roi_requires_positive_area() -> None:
    with pytest.raises(ValueError):
        RoiResult(
            box=(10, 10, 10, 20),
            display_box=(9, 9, 11, 21),
        )


def test_roi_display_box_must_contain_exact_box() -> None:
    with pytest.raises(ValueError):
        RoiResult(
            box=(10, 10, 20, 20),
            display_box=(11, 9, 21, 21),
        )


def test_cpu_device_never_queries_cuda() -> None:
    with patch(
        "spina_bifida.inference.model_loader.torch.cuda.is_available",
        side_effect=AssertionError("CUDA must not be initialized"),
    ):
        device = resolve_device("cpu")

    assert device.type == "cpu"


@pytest.mark.parametrize(("available", "expected"), [(True, "cuda"), (False, "cpu")])
def test_auto_device_uses_cuda_only_when_available(available: bool, expected: str) -> None:
    with patch(
        "spina_bifida.inference.model_loader.torch.cuda.is_available",
        return_value=available,
    ):
        device = resolve_device("auto")

    assert device.type == expected


def test_explicit_cuda_fails_closed_when_unavailable() -> None:
    with (
        patch(
            "spina_bifida.inference.model_loader.torch.cuda.is_available",
            return_value=False,
        ),
        pytest.raises(CudaUnavailable),
    ):
        resolve_device("cuda")


def test_roi_is_clamped_with_legacy_pillow_rounding() -> None:
    assert clamp_roi((-5.2, 2.6, 99.7, 120.4), 80, 100) == (0, 3, 80, 100)
    with pytest.raises(InvalidRoi):
        clamp_roi((5.0, 5.0, 5.4, 5.4), 100, 100)


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((20, 20, 80, 80), (14, 14, 86, 86)),
        ((0, 0, 50, 50), (0, 0, 55, 55)),
        ((50, 50, 100, 100), (45, 45, 100, 100)),
    ],
)
def test_pad_roi_adds_ten_percent_and_clamps(
    box: tuple[int, int, int, int],
    expected: tuple[int, int, int, int],
) -> None:
    assert pad_roi(box, 100, 100) == expected


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((10, 10, 11, 11), (9, 9, 12, 12)),
        ((10, 10, 15, 15), (9, 9, 16, 16)),
        ((0, 0, 1, 5), (0, 0, 2, 6)),
        ((0, 25, 5, 26), (0, 24, 6, 27)),
        ((25, 25, 30, 30), (24, 24, 30, 30)),
    ],
)
def test_pad_roi_rounds_tiny_fractional_padding_up_before_clamping(
    box: tuple[int, int, int, int],
    expected: tuple[int, int, int, int],
) -> None:
    assert pad_roi(box, 30, 30) == expected


def test_render_padded_heatmap_keeps_display_context_and_exact_box_offset() -> None:
    original = Image.new("RGB", (100, 100), (10, 20, 30))
    exact_heatmap = Image.new("RGB", (60, 60), (200, 100, 50))

    rendered = render_padded_heatmap(
        original,
        exact_box=(20, 20, 80, 80),
        display_box=(14, 14, 86, 86),
        exact_heatmap=exact_heatmap,
    )

    assert rendered.size == (72, 72)
    assert rendered.getpixel((0, 0)) == (10, 20, 30)
    assert rendered.getpixel((5, 5)) == (10, 20, 30)
    assert rendered.getpixel((6, 6)) == (200, 100, 50)
    assert rendered.getpixel((65, 65)) == (200, 100, 50)
    assert rendered.getpixel((66, 66)) == (10, 20, 30)


def test_render_padded_heatmap_matches_edge_display_roi_dimensions() -> None:
    original = Image.new("RGB", (100, 100), (10, 20, 30))
    exact_heatmap = Image.new("RGB", (50, 50), (200, 100, 50))

    rendered = render_padded_heatmap(
        original,
        exact_box=(0, 0, 50, 50),
        display_box=(0, 0, 55, 55),
        exact_heatmap=exact_heatmap,
    )

    assert rendered.size == (55, 55)
    assert rendered.getpixel((0, 0)) == (200, 100, 50)
    assert rendered.getpixel((49, 49)) == (200, 100, 50)
    assert rendered.getpixel((50, 50)) == (10, 20, 30)


def test_analysis_keeps_exact_crop_for_models_while_assets_include_context(
    tmp_path: Path,
) -> None:
    source = Image.new("RGB", (100, 100), (10, 20, 30))
    source.paste((70, 80, 90), (20, 20, 80, 80))
    frame_path = tmp_path / "frame.png"
    source.save(frame_path)

    class CapturingService(InferenceService):
        def __init__(self) -> None:
            super().__init__(SimpleNamespace(device=torch.device("cpu"), version="v1"))
            self.prepared_crop: Image.Image | None = None
            self.prepared_tensor = torch.ones((1, 3, 2, 2))
            self.predict_inputs: list[torch.Tensor] = []
            self.heatmap_output_sizes: list[tuple[int, int]] = []

        def _detect(
            self,
            image: Image.Image,
        ) -> tuple[tuple[float, float, float, float], float, int] | None:
            return ((20.0, 20.0, 80.0, 80.0), 0.91, 1)

        def _prepare(self, image: Image.Image) -> torch.Tensor:
            self.prepared_crop = image.copy()
            return self.prepared_tensor

        def _classifiers(self, plane: PlaneType) -> tuple[object, object]:
            return object(), object()

        def _predict(self, tensor: torch.Tensor, model: object) -> float:
            self.predict_inputs.append(tensor)
            return 0.75

        def _heatmap(
            self,
            tensor: torch.Tensor,
            model: object,
            output_size: tuple[int, int],
        ) -> Image.Image:
            self.heatmap_output_sizes.append(output_size)
            return Image.new("RGB", output_size, (200, 100, 50))

    service = CapturingService()
    result = service.analyze([InputFrame("frame-1", frame_path)])
    frame = result.frames[0]
    assets = {asset.kind: asset for asset in frame.assets}
    roi_image = Image.open(BytesIO(assets[AssetType.ROI].content)).convert("RGB")
    heatmap_image = Image.open(BytesIO(assets[AssetType.HEATMAP].content)).convert("RGB")

    assert service.prepared_crop is not None
    assert service.prepared_crop.size == (60, 60)
    assert service.prepared_crop.getpixel((0, 0)) == (70, 80, 90)
    assert service.predict_inputs == [service.prepared_tensor, service.prepared_tensor]
    assert service.heatmap_output_sizes == [(60, 60)]
    assert frame.roi is not None and frame.roi.box == (20, 20, 80, 80)
    assert frame.quality.score == 0.75
    assert frame.assessment is not None and frame.assessment.score == 0.75
    assert roi_image.size == (72, 72)
    assert heatmap_image.size == (72, 72)
    assert roi_image.getpixel((0, 0)) == (10, 20, 30)
    assert heatmap_image.getpixel((0, 0)) == (10, 20, 30)
    assert heatmap_image.getpixel((6, 6)) == (200, 100, 50)


def test_analysis_uses_configured_quality_and_assessment_thresholds(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", (32, 32), (10, 20, 30)).save(frame_path)

    class ThresholdService(InferenceService):
        def __init__(self) -> None:
            super().__init__(
                SimpleNamespace(device=torch.device("cpu"), version="v2"),
                InferenceThresholdSettings(
                    plane_detection=0.23,
                    quality=0.7,
                    assessment=0.8,
                    warning_margin=0.08,
                ),
            )
            self.scores = iter((0.65, 0.75))

        def _detect(
            self,
            image: Image.Image,
        ) -> tuple[tuple[float, float, float, float], float, int] | None:
            return ((0.0, 0.0, 32.0, 32.0), 0.91, 1)

        def _classifiers(self, plane: PlaneType) -> tuple[object, object]:
            return object(), object()

        def _predict(self, tensor: torch.Tensor, model: object) -> float:
            return next(self.scores)

        def _heatmap(
            self,
            tensor: torch.Tensor,
            model: object,
            output_size: tuple[int, int],
        ) -> Image.Image:
            return Image.new("RGB", output_size, (200, 100, 50))

    result = ThresholdService().analyze([InputFrame("frame-1", frame_path)])
    frame = result.frames[0]

    assert frame.quality.accepted is False
    assert frame.assessment is not None
    assert frame.assessment.category is AssessmentCategory.NO_EXPRESSED_SIGNAL
    assert result.to_dict()["thresholds"] == {
        "plane_detection": 0.23,
        "quality": 0.7,
        "assessment": 0.8,
        "warning_margin": 0.08,
    }


def test_detector_uses_configured_confidence_threshold() -> None:
    class RecordingYolo:
        def __init__(self) -> None:
            self.confidence: float | None = None

        def predict(
            self,
            image: Image.Image,
            *,
            verbose: bool,
            conf: float,
            device: str,
        ) -> list[SimpleNamespace]:
            self.confidence = conf
            return [SimpleNamespace(boxes=SimpleNamespace(shape=(0, 6)))]

    yolo = RecordingYolo()
    service = InferenceService(
        SimpleNamespace(yolo=yolo, device=torch.device("cpu"), version="v2"),
        InferenceThresholdSettings(plane_detection=0.23),
    )

    assert service._detect(Image.new("RGB", (16, 16))) is None
    assert yolo.confidence == 0.23


def test_constant_and_non_finite_heatmap_normalization_is_safe() -> None:
    values = np.array([[np.nan, np.inf], [5.0, 5.0]], dtype=np.float32)

    normalized = normalize_array(values)

    assert np.isfinite(normalized).all()
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
