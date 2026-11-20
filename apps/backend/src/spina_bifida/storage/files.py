from __future__ import annotations

import hashlib
import shutil
import warnings
from contextlib import suppress
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import IO, BinaryIO
from uuid import NAMESPACE_URL, uuid4, uuid5

from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy import select
from werkzeug.datastructures import FileStorage

from spina_bifida.analyses.models import AnalysisAsset, AnalysisJob, JobState
from spina_bifida.extensions import db
from spina_bifida.inference.contracts import GeneratedAsset

ALLOWED_MIME_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    image: Image.Image
    checksum: str
    mime_type: str


@dataclass(frozen=True, slots=True)
class StoredFile:
    path: Path
    checksum: str


class UploadRejected(ValueError):
    def __init__(
        self,
        code: str,
        status: int = 422,
        params: dict[str, object] | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.status = status
        self.params = params or {}


class AssetNotFound(FileNotFoundError):
    pass


class UploadValidator:
    def __init__(self, *, max_request_bytes: int, max_image_pixels: int) -> None:
        self.max_request_bytes = max_request_bytes
        self.max_image_pixels = max_image_pixels

    def validate(self, file: FileStorage) -> ValidatedImage:
        filename = file.filename or ""
        extension = Path(filename).suffix.casefold()
        mime_type = file.mimetype.casefold()
        if extension not in ALLOWED_EXTENSIONS or mime_type not in ALLOWED_MIME_TYPES:
            raise UploadRejected(
                "unsupported_format",
                415,
                {"allowed": list(ALLOWED_MIME_TYPES)},
            )

        raw = file.stream.read(self.max_request_bytes + 1)
        if len(raw) > self.max_request_bytes:
            raise UploadRejected(
                "image_too_large",
                params={"max_bytes": self.max_request_bytes},
            )
        if not raw:
            raise UploadRejected("image_decode_failed")

        previous_max_pixels = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = self.max_image_pixels
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(raw)) as probe:
                    detected_format = probe.format
                    if detected_format != ALLOWED_MIME_TYPES[mime_type]:
                        raise UploadRejected("mime_mismatch", 415)
                    probe.verify()
                with Image.open(BytesIO(raw)) as decoded:
                    decoded.load()
                    if decoded.width * decoded.height > self.max_image_pixels:
                        raise UploadRejected(
                            "image_dimensions_exceeded",
                            params={"max_pixels": self.max_image_pixels},
                        )
                    normalized = ImageOps.exif_transpose(decoded).convert("RGB").copy()
        except UploadRejected:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            raise UploadRejected(
                "image_dimensions_exceeded",
                params={"max_pixels": self.max_image_pixels},
            ) from error
        except (OSError, SyntaxError, UnidentifiedImageError) as error:
            raise UploadRejected("image_decode_failed") from error
        finally:
            Image.MAX_IMAGE_PIXELS = previous_max_pixels

        checksum = hashlib.sha256()
        checksum.update(normalized.mode.encode("ascii"))
        checksum.update(normalized.width.to_bytes(8, "big"))
        checksum.update(normalized.height.to_bytes(8, "big"))
        checksum.update(normalized.tobytes())
        return ValidatedImage(
            image=normalized,
            checksum=checksum.hexdigest(),
            mime_type=mime_type,
        )


class FileStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def save_normalized(
        self,
        job_id: str,
        frame_id: str,
        image: Image.Image,
    ) -> StoredFile:
        job_directory = self._job_directory(job_id)
        job_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        name = f"{uuid5(NAMESPACE_URL, frame_id)}-{uuid4()}.png"
        destination = (job_directory / name).resolve()
        self._require_inside_root(destination)

        output = BytesIO()
        image.convert("RGB").save(output, format="PNG")
        content = output.getvalue()
        return self._write_content(destination, content)

    def save_normalized_stream(
        self,
        job_id: str,
        frame_id: str,
        stream: IO[bytes],
    ) -> StoredFile:
        job_directory = self._job_directory(job_id)
        job_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        name = f"{uuid5(NAMESPACE_URL, frame_id)}-{uuid4()}.png"
        destination = (job_directory / name).resolve()
        self._require_inside_root(destination)

        temporary = destination.with_suffix(".tmp")
        checksum = hashlib.sha256()
        stream.seek(0)
        try:
            with temporary.open("xb") as output:
                while chunk := stream.read(1024 * 1024):
                    output.write(chunk)
                    checksum.update(chunk)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return StoredFile(destination, checksum.hexdigest())

    def save_generated(
        self,
        job_id: str,
        frame_id: str,
        asset: GeneratedAsset,
    ) -> StoredFile:
        if not asset.content:
            raise ValueError("generated asset content is empty")
        job_directory = self._job_directory(job_id)
        job_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination = (
            job_directory / f"{uuid5(NAMESPACE_URL, frame_id)}-{asset.kind.value}-{uuid4()}.png"
        ).resolve()
        self._require_inside_root(destination)
        return self._write_content(destination, asset.content)

    def delete_asset(self, path: Path) -> None:
        resolved = path.resolve()
        self._require_inside_root(resolved)
        with suppress(FileNotFoundError):
            resolved.unlink()

    @staticmethod
    def _write_content(destination: Path, content: bytes) -> StoredFile:
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
        return StoredFile(destination, hashlib.sha256(content).hexdigest())

    def open_asset(
        self,
        owner_id: str,
        job_id: str,
        asset_id: str,
        *,
        allow_admin: bool = False,
    ) -> BinaryIO:
        statement = (
            select(AnalysisAsset)
            .join(AnalysisJob)
            .where(
                AnalysisAsset.id == asset_id,
                AnalysisAsset.job_id == job_id,
                AnalysisJob.state != JobState.DELETED,
            )
        )
        if not allow_admin:
            statement = statement.where(AnalysisJob.owner_id == owner_id)
        asset = db.session.scalar(statement)
        if asset is None:
            raise AssetNotFound(asset_id)
        return self.open_path(Path(asset.local_path))

    def open_path(self, path: Path) -> BinaryIO:
        resolved = path.resolve()
        self._require_inside_root(resolved)
        try:
            return resolved.open("rb")
        except OSError as error:
            raise AssetNotFound(str(resolved)) from error

    def delete_job(self, job_id: str) -> None:
        directory = self._job_directory(job_id)
        self._require_inside_root(directory)
        if directory.is_dir():
            shutil.rmtree(directory)

    def _job_directory(self, job_id: str) -> Path:
        directory_name = str(uuid5(NAMESPACE_URL, f"analysis-job:{job_id}"))
        return (self.root / directory_name).resolve()

    def _require_inside_root(self, path: Path) -> None:
        if path == self.root or not path.is_relative_to(self.root):
            raise AssetNotFound(str(path))
