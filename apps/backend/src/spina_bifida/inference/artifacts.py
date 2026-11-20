from __future__ import annotations

import hashlib
from io import BytesIO
from math import ceil

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from spina_bifida.inference.contracts import AssetType, GeneratedAsset


class InvalidRoi(ValueError):
    pass


def clamp_roi(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    # Pillow rounds float crop coordinates to the nearest integer. Preserve that
    # legacy orientation/crop behavior, then clamp it to the decoded image.
    left = max(0, min(width, round(box[0])))
    top = max(0, min(height, round(box[1])))
    right = max(0, min(width, round(box[2])))
    bottom = max(0, min(height, round(box[3])))
    if right <= left or bottom <= top:
        raise InvalidRoi("detected ROI has zero area after bounds clamping")
    return left, top, right, bottom


def pad_roi(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    fraction: float = 0.10,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    horizontal_padding = ceil((right - left) * fraction)
    vertical_padding = ceil((bottom - top) * fraction)
    return (
        max(0, left - horizontal_padding),
        max(0, top - vertical_padding),
        min(width, right + horizontal_padding),
        min(height, bottom + vertical_padding),
    )


def normalize_array(values: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    finite = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    shifted = finite - float(finite.min())
    maximum = float(shifted.max())
    if maximum <= epsilon:
        return np.zeros_like(shifted, dtype=np.float32)
    return shifted / maximum


def render_heatmap(
    input_tensor: torch.Tensor,
    cam_tensor: torch.Tensor,
    output_size: tuple[int, int],
) -> Image.Image:
    source = input_tensor.squeeze(0).detach().cpu().numpy()[0]
    cam = cam_tensor[0].detach().cpu().numpy()[0]
    source_normalized = normalize_array(source)
    cam_normalized = normalize_array(cam)

    colored = colormaps.get_cmap("RdBu")(cam_normalized)[..., :3]
    colored_image = Image.fromarray((colored * 255).astype(np.uint8), mode="RGB")
    grayscale = Image.fromarray((source_normalized * 255).astype(np.uint8), mode="L")
    source_image = Image.merge("RGB", (grayscale, grayscale, grayscale))
    blended = Image.blend(source_image, colored_image, alpha=0.5)
    return (
        blended.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        .rotate(90)
        .resize(output_size)
    )


def render_padded_heatmap(
    original: Image.Image,
    exact_box: tuple[int, int, int, int],
    display_box: tuple[int, int, int, int],
    exact_heatmap: Image.Image,
) -> Image.Image:
    display_heatmap = original.crop(display_box)
    offset = (exact_box[0] - display_box[0], exact_box[1] - display_box[1])
    display_heatmap.paste(exact_heatmap, offset)
    return display_heatmap


def image_asset(kind: AssetType, image: Image.Image) -> GeneratedAsset:
    normalized = image.convert("RGB")
    output = BytesIO()
    normalized.save(output, format="PNG")
    content = output.getvalue()
    return GeneratedAsset(
        kind=kind,
        width=normalized.width,
        height=normalized.height,
        checksum=hashlib.sha256(content).hexdigest(),
        content=content,
    )
