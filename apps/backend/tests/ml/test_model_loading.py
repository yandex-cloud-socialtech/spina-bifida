from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path

import pytest
import torch

from spina_bifida.inference.model_loader import MODEL_FILENAMES
from spina_bifida.inference.registry import InferenceRegistry

pytestmark = pytest.mark.ml

REPOSITORY_ROOT = Path(__file__).parents[4]


@pytest.fixture(scope="module")
def models_root() -> Path:
    configured = os.environ.get("SPINA_BIFIDA_MODELS_ROOT")
    if not configured:
        pytest.skip("SPINA_BIFIDA_MODELS_ROOT is required for real-weight tests")
    root = Path(configured)
    if not root.is_dir():
        pytest.fail(f"Model root does not exist: {root}")
    return root


def test_release_manifest_covers_and_verifies_all_ten_weights(models_root: Path) -> None:
    manifest = REPOSITORY_ROOT / "deploy" / "rewrite" / "models" / "manifest.sha256"
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        checksum, relative_path = line.split(maxsplit=1)
        entries[relative_path] = checksum

    expected = {f"{version}/{name}" for version in ("v1", "v2") for name in MODEL_FILENAMES}
    assert set(entries) == expected
    for relative_path, expected_checksum in entries.items():
        digest = hashlib.sha256((models_root / relative_path).read_bytes()).hexdigest()
        assert digest == expected_checksum, relative_path


def test_registry_eagerly_loads_every_bundle_without_pretrained_downloads(
    models_root: Path,
) -> None:
    registry = InferenceRegistry.load(models_root, torch.device("cpu"))

    for persisted_version in ("v1", "v2"):
        bundle = registry.for_version(persisted_version).models
        assert bundle.device.type == "cpu"
        assert bundle.version == persisted_version
        assert bundle.yolo is not None
        assert all(
            model.training is False
            for model in (
                bundle.axial_quality,
                bundle.axial_pathology,
                bundle.sagittal_quality,
                bundle.sagittal_pathology,
            )
        )
    del registry
    gc.collect()
