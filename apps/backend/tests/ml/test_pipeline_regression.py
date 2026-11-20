from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from spina_bifida.inference.artifacts import image_asset, pad_roi, render_padded_heatmap
from spina_bifida.inference.contracts import AssetType, InputFrame
from spina_bifida.inference.model_loader import ModelBundle
from spina_bifida.inference.processor import InferenceService

pytestmark = pytest.mark.ml

BACKEND_ROOT = Path(__file__).parents[2]
REPOSITORY_ROOT = BACKEND_ROOT.parents[1]


@pytest.fixture(scope="module")
def models_root() -> Path:
    configured = os.environ.get("SPINA_BIFIDA_MODELS_ROOT")
    if not configured:
        pytest.skip("SPINA_BIFIDA_MODELS_ROOT is required for real-weight tests")
    return Path(configured)


@pytest.fixture(scope="module")
def expected() -> dict[str, Any]:
    fixture = BACKEND_ROOT / "tests" / "fixtures" / "expected_results.json"
    return json.loads(fixture.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_v1_four_example_regression(
    models_root: Path,
    expected: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = InferenceService(ModelBundle.load(models_root, "v1", torch.device("cpu")))
    tolerance = float(expected["score_tolerance"])
    examples_root = REPOSITORY_ROOT / "apps" / "frontend" / "public" / "examples"
    exact_heatmaps: list[Image.Image] = []

    def capture_exact_heatmap(
        original: Image.Image,
        exact_box: tuple[int, int, int, int],
        display_box: tuple[int, int, int, int],
        exact_heatmap: Image.Image,
    ) -> Image.Image:
        exact_heatmaps.append(exact_heatmap.copy())
        return render_padded_heatmap(original, exact_box, display_box, exact_heatmap)

    monkeypatch.setattr(
        "spina_bifida.inference.processor.render_padded_heatmap",
        capture_exact_heatmap,
    )

    for filename, baseline in expected["examples"].items():
        frame_path = examples_root / filename
        result = service.analyze([InputFrame(Path(filename).stem, frame_path)])
        frame = result.frames[0]
        assert frame.analyzable is True
        assert frame.plane.type.value == baseline["plane"]
        assert frame.roi is not None and list(frame.roi.box) == baseline["roi"]
        assert frame.quality.score == pytest.approx(baseline["quality_score"], abs=tolerance)
        assert frame.assessment is not None
        assert frame.assessment.category.value == baseline["assessment_category"]
        assert frame.assessment.score == pytest.approx(
            baseline["assessment_score"],
            abs=tolerance,
        )
        with Image.open(frame_path) as opened:
            original = opened.convert("RGB")
            exact_roi = original.crop(tuple(baseline["roi"]))
            assert frame.roi.display_box == pad_roi(
                tuple(baseline["roi"]),
                *original.size,
            )
        assert image_asset(AssetType.ROI, exact_roi).checksum == baseline["exact_roi_checksum"]
        assert image_asset(AssetType.HEATMAP, exact_heatmaps[-1]).checksum == baseline[
            "exact_heatmap_checksum"
        ]
        assets = {asset.kind.value: asset for asset in frame.assets}
        assert [assets["original"].width, assets["original"].height] == list(original.size)
        display_left, display_top, display_right, display_bottom = frame.roi.display_box
        assert [assets["roi"].width, assets["roi"].height] == [
            display_right - display_left,
            display_bottom - display_top,
        ]
        for kind, expected_asset in baseline["assets"].items():
            assert [assets[kind].width, assets[kind].height] == expected_asset["dimensions"]
            if "checksum" in expected_asset:
                assert assets[kind].checksum == expected_asset["checksum"]
