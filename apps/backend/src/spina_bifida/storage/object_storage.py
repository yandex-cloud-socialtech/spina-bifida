from __future__ import annotations

from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]

from spina_bifida.config import ObjectStorageSettings


class S3ObjectStorage:
    def __init__(self, settings: ObjectStorageSettings) -> None:
        if (
            not settings.endpoint
            or not settings.bucket
            or settings.access_key is None
            or settings.secret_key is None
        ):
            raise ValueError("object storage credentials are incomplete")
        self.bucket = settings.bucket
        self.client: Any = boto3.client(
            "s3",
            endpoint_url=settings.endpoint,
            aws_access_key_id=settings.access_key.get_secret_value(),
            aws_secret_access_key=settings.secret_key.get_secret_value(),
            region_name="ru-central1",
        )

    def put_file(self, local_path: Path, object_key: str, checksum: str) -> None:
        self.client.upload_file(
            str(local_path),
            self.bucket,
            object_key,
            ExtraArgs={"Metadata": {"sha256": checksum}},
        )
        metadata = self.client.head_object(Bucket=self.bucket, Key=object_key).get("Metadata", {})
        if metadata.get("sha256") != checksum:
            raise OSError("uploaded object checksum metadata mismatch")
