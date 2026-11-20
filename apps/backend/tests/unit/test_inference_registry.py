from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from spina_bifida.inference.model_loader import (
    ModelBundle,
    ModelFilesMissing,
    ModelVersion,
)
from spina_bifida.inference.registry import InferenceRegistry, UnsupportedModelVersion


def test_registry_eagerly_loads_both_supported_bundles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded: list[ModelVersion] = []

    def fake_load(
        models_root: Path,
        version: ModelVersion,
        device: torch.device,
    ) -> ModelBundle:
        assert models_root == tmp_path
        assert device == torch.device("cpu")
        loaded.append(version)
        return Mock(spec=ModelBundle, version=version, device=device)

    monkeypatch.setattr(ModelBundle, "load", staticmethod(fake_load))

    registry = InferenceRegistry.load(tmp_path, torch.device("cpu"))

    assert loaded == ["v1", "v2"]
    assert registry.for_version("v1").models.version == "v1"
    assert registry.for_version("v2").models.version == "v2"


def test_registry_fails_startup_when_either_bundle_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "v2" / "axial_quality.pt"

    def fake_load(
        models_root: Path,
        version: ModelVersion,
        device: torch.device,
    ) -> ModelBundle:
        if version == "v2":
            raise ModelFilesMissing([missing])
        return Mock(spec=ModelBundle, version=version, device=device)

    monkeypatch.setattr(ModelBundle, "load", staticmethod(fake_load))

    with pytest.raises(ModelFilesMissing) as caught:
        InferenceRegistry.load(tmp_path, torch.device("cpu"))

    assert caught.value.missing == (missing,)


def test_registry_rejects_an_unpersistable_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_load(
        models_root: Path,
        version: ModelVersion,
        device: torch.device,
    ) -> ModelBundle:
        return Mock(spec=ModelBundle, version=version, device=device)

    monkeypatch.setattr(ModelBundle, "load", staticmethod(fake_load))
    registry = InferenceRegistry.load(tmp_path, torch.device("cpu"))

    with pytest.raises(UnsupportedModelVersion, match="v3"):
        registry.for_version("v3")
