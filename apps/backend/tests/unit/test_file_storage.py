from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from werkzeug.datastructures import FileStorage

from spina_bifida.storage.files import FileStore, UploadRejected, UploadValidator

FIXTURES = Path(__file__).parents[1] / "fixtures" / "images"


def image_upload(
    name: str,
    *,
    filename: str | None = None,
    content_type: str | None = None,
) -> FileStorage:
    data = (FIXTURES / name).read_bytes()
    suffix = Path(name).suffix.lower()
    inferred_type = "image/png" if suffix == ".png" else "image/jpeg"
    return FileStorage(
        stream=BytesIO(data),
        filename=filename or name,
        content_type=content_type or inferred_type,
    )


@pytest.fixture
def upload_validator() -> UploadValidator:
    return UploadValidator(max_request_bytes=100_000, max_image_pixels=10_000)


@pytest.mark.parametrize(
    ("fixture_name", "code"),
    [
        ("corrupt.jpg", "image_decode_failed"),
        ("oversized.png", "image_dimensions_exceeded"),
    ],
)
def test_invalid_image_has_stable_error(
    upload_validator: UploadValidator,
    fixture_name: str,
    code: str,
) -> None:
    with pytest.raises(UploadRejected) as caught:
        upload_validator.validate(image_upload(fixture_name))

    assert caught.value.code == code


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [("scan.gif", "image/jpeg"), ("scan.jpg", "application/octet-stream")],
)
def test_unsupported_extension_or_mime_is_415(
    upload_validator: UploadValidator,
    filename: str,
    content_type: str,
) -> None:
    with pytest.raises(UploadRejected) as caught:
        upload_validator.validate(
            image_upload("valid.jpg", filename=filename, content_type=content_type),
        )

    assert caught.value.code == "unsupported_format"
    assert caught.value.status == 415
    assert caught.value.params["allowed"] == ["image/jpeg", "image/png"]


def test_mime_spoofing_is_rejected(upload_validator: UploadValidator) -> None:
    with pytest.raises(UploadRejected) as caught:
        upload_validator.validate(
            image_upload("valid.png", filename="scan.jpg", content_type="image/jpeg"),
        )

    assert caught.value.code == "mime_mismatch"
    assert caught.value.status == 415


def test_max_byte_size_is_checked_before_decode() -> None:
    validator = UploadValidator(max_request_bytes=10, max_image_pixels=10_000)

    with pytest.raises(UploadRejected) as caught:
        validator.validate(image_upload("valid.jpg"))

    assert caught.value.code == "image_too_large"


def test_decompression_bomb_warning_is_rejected() -> None:
    validator = UploadValidator(max_request_bytes=100_000, max_image_pixels=500)

    with pytest.raises(UploadRejected) as caught:
        validator.validate(image_upload("valid.png"))

    assert caught.value.code == "image_dimensions_exceeded"


@pytest.mark.parametrize("fixture_name", ["valid.jpg", "valid.png"])
def test_valid_images_are_detached_rgb(
    upload_validator: UploadValidator,
    fixture_name: str,
) -> None:
    validated = upload_validator.validate(image_upload(fixture_name))

    assert validated.image.mode == "RGB"
    assert validated.image.size == (32, 24)
    assert len(validated.checksum) == 64


def test_normalization_strips_exif_and_uses_server_filename(tmp_path: Path) -> None:
    validator = UploadValidator(max_request_bytes=100_000, max_image_pixels=10_000)
    file_store = FileStore(tmp_path)
    validated = validator.validate(
        image_upload("with_exif.jpg", filename="../../with_exif.jpg"),
    )

    stored = file_store.save_normalized("job-id", "frame-id", validated.image)
    with Image.open(stored.path) as reopened:
        assert reopened.getexif() == {}
        assert reopened.mode == "RGB"
    assert "with_exif" not in stored.path.name
    assert stored.path.suffix == ".png"
    assert stored.path.resolve().is_relative_to(tmp_path.resolve())


def test_store_refuses_paths_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    outside = tmp_path / "outside.png"
    outside.write_bytes((FIXTURES / "valid.png").read_bytes())
    file_store = FileStore(root)

    with pytest.raises(FileNotFoundError):
        file_store.open_path(outside)
