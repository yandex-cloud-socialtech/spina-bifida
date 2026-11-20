from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from spina_bifida.config import InferenceThresholdSettings
from spina_bifida.inference.model_loader import ModelBundle, ModelVersion
from spina_bifida.inference.processor import InferenceService

SUPPORTED_MODEL_VERSIONS: tuple[ModelVersion, ...] = ("v1", "v2")


class UnsupportedModelVersion(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class InferenceRegistry:
    services: Mapping[ModelVersion, InferenceService]

    @classmethod
    def load(
        cls,
        models_root: Path,
        device: torch.device,
        thresholds: InferenceThresholdSettings | None = None,
    ) -> InferenceRegistry:
        return cls(
            {
                version: InferenceService(
                    ModelBundle.load(models_root, version, device),
                    thresholds,
                )
                for version in SUPPORTED_MODEL_VERSIONS
            },
        )

    def for_version(self, version: str) -> InferenceService:
        if version not in self.services:
            raise UnsupportedModelVersion(version)
        return self.services[cast(ModelVersion, version)]
