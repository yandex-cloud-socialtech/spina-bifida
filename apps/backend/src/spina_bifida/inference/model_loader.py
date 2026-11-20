from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

type DeviceMode = Literal["auto", "cpu", "cuda"]
type ModelVersion = Literal["v1", "v2"]

MODEL_FILENAMES = (
    "best_object_detection.pt",
    "axial_quality.pt",
    "axial_pathology.pt",
    "sagittal_quality.pt",
    "sagittal_pathology.pt",
)


class CudaUnavailable(RuntimeError):
    pass


class ModelFilesMissing(FileNotFoundError):
    def __init__(self, missing: list[Path]) -> None:
        super().__init__("required model files are missing")
        self.missing = tuple(missing)


def resolve_device(requested: DeviceMode) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise CudaUnavailable("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(slots=True)
class ModelBundle:
    yolo: Any
    axial_quality: Any
    axial_pathology: Any
    sagittal_quality: Any
    sagittal_pathology: Any
    version: ModelVersion
    device: torch.device

    @classmethod
    def load(
        cls,
        models_root: Path,
        version: ModelVersion,
        device: torch.device,
    ) -> ModelBundle:
        version_root = models_root / version
        paths = {name: version_root / name for name in MODEL_FILENAMES}
        missing = [path for path in paths.values() if not path.is_file()]
        if missing:
            raise ModelFilesMissing(missing)

        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        from monai.networks.nets.densenet import DenseNet121, DenseNet201
        from ultralytics import YOLO

        constructor = DenseNet121 if version == "v1" else DenseNet201

        def load_classifier(filename: str) -> Any:
            model = constructor(
                spatial_dims=2,
                in_channels=3,
                out_channels=2,
                pretrained=False,
            )
            state = torch.load(paths[filename], map_location=device, weights_only=True)
            model.load_state_dict(state, strict=True)
            return model.to(device).eval()

        yolo = YOLO(str(paths["best_object_detection.pt"]))
        yolo.to(device)
        return cls(
            yolo=yolo,
            axial_quality=load_classifier("axial_quality.pt"),
            axial_pathology=load_classifier("axial_pathology.pt"),
            sagittal_quality=load_classifier("sagittal_quality.pt"),
            sagittal_pathology=load_classifier("sagittal_pathology.pt"),
            version=version,
            device=device,
        )
