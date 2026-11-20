from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from spina_bifida.config import InferenceThresholdSettings


class PlaneType(StrEnum):
    AXIAL = "axial"
    SAGITTAL = "sagittal"
    UNKNOWN = "unknown"


class AssessmentCategory(StrEnum):
    NO_EXPRESSED_SIGNAL = "no_expressed_signal"
    ATTENTION_REQUIRED = "attention_required"


class AssetType(StrEnum):
    ORIGINAL = "original"
    ROI = "roi"
    HEATMAP = "heatmap"


def _validate_probability(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError("score must be between zero and one")


@dataclass(frozen=True, slots=True)
class InputFrame:
    frame_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class GeneratedAsset:
    kind: AssetType
    width: int
    height: int
    checksum: str
    content: bytes = field(default=b"", repr=False, compare=False)
    media_type: str = "image/png"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("asset dimensions must be positive")
        if len(self.checksum) != 64:
            raise ValueError("asset checksum must be SHA-256")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "width": self.width,
            "height": self.height,
            "checksum": self.checksum,
            "media_type": self.media_type,
        }


@dataclass(frozen=True, slots=True)
class QualityResult:
    score: float
    accepted: bool

    def __post_init__(self) -> None:
        _validate_probability(self.score)

    def to_dict(self) -> dict[str, object]:
        return {"score": self.score, "accepted": self.accepted}


@dataclass(frozen=True, slots=True)
class PlaneResult:
    type: PlaneType
    score: float

    def __post_init__(self) -> None:
        _validate_probability(self.score)

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type.value, "score": self.score}


@dataclass(frozen=True, slots=True)
class RoiResult:
    box: tuple[int, int, int, int]
    display_box: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        for name, box in (("ROI", self.box), ("ROI display box", self.display_box)):
            left, top, right, bottom = box
            if left < 0 or top < 0 or right <= left or bottom <= top:
                raise ValueError(
                    f"{name} must have positive area and non-negative coordinates",
                )
        left, top, right, bottom = self.box
        display_left, display_top, display_right, display_bottom = self.display_box
        if (
            display_left > left
            or display_top > top
            or display_right < right
            or display_bottom < bottom
        ):
            raise ValueError("ROI display box must contain the exact ROI box")

    def to_dict(self) -> dict[str, object]:
        return {
            "box": list(self.box),
            "display_box": list(self.display_box),
        }


@dataclass(frozen=True, slots=True)
class AssessmentResult:
    category: AssessmentCategory
    score: float

    def __post_init__(self) -> None:
        _validate_probability(self.score)

    def to_dict(self) -> dict[str, object]:
        return {"category": self.category.value, "score": self.score}


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame_id: str
    analyzable: bool
    quality: QualityResult
    plane: PlaneResult
    roi: RoiResult | None
    assessment: AssessmentResult | None
    explanations: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    assets: tuple[GeneratedAsset, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "analyzable": self.analyzable,
            "quality": self.quality.to_dict(),
            "plane": self.plane.to_dict(),
            "roi": self.roi.to_dict() if self.roi else None,
            "assessment": self.assessment.to_dict() if self.assessment else None,
            "explanations": list(self.explanations),
            "frame_id": self.frame_id,
            "warnings": list(self.warnings),
            "assets": [asset.to_dict() for asset in self.assets],
        }


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    version: str
    device: str
    service: str

    def to_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "device": self.device,
            "service": self.service,
        }


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    frames: tuple[FrameResult, ...]
    warnings: tuple[str, ...]
    model: ModelMetadata
    thresholds: InferenceThresholdSettings = field(default_factory=InferenceThresholdSettings)

    def __post_init__(self) -> None:
        if not 1 <= len(self.frames) <= 2:
            raise ValueError("analysis requires one or two frame results")

    def to_dict(self) -> dict[str, object]:
        return {
            "frames": [frame.to_dict() for frame in self.frames],
            "warnings": list(self.warnings),
            "model": self.model.to_dict(),
            "thresholds": self.thresholds.model_dump(),
        }
