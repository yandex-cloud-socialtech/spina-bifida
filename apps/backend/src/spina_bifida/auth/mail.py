from __future__ import annotations

from typing import Any

import boto3  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]
from flask import current_app

from spina_bifida.config import PostboxSettings
from spina_bifida.observability.audit import audit


class MailUnavailable(RuntimeError):
    pass


class PostboxMailer:
    def __init__(self, settings: PostboxSettings | None = None, client: Any = None) -> None:
        self.settings: PostboxSettings = settings or current_app.extensions["settings"].postbox
        self._client = client

    def require_configured(self) -> None:
        if not self.settings.enabled:
            raise MailUnavailable

    def send(self, recipient: str, subject: str, text: str) -> None:
        self.require_configured()
        assert self.settings.access_key is not None
        assert self.settings.secret_key is not None
        try:
            client = self._client or boto3.client(
                "sesv2",
                endpoint_url=self.settings.endpoint,
                region_name=self.settings.region,
                aws_access_key_id=self.settings.access_key.get_secret_value(),
                aws_secret_access_key=self.settings.secret_key.get_secret_value(),
                config=Config(
                    connect_timeout=5, read_timeout=10, retries={"total_max_attempts": 1}
                ),
            )
            client.send_email(
                FromEmailAddress=self.settings.sender,
                Destination={"ToAddresses": [recipient]},
                Content={
                    "Simple": {
                        "Subject": {"Data": subject, "Charset": "UTF-8"},
                        "Body": {"Text": {"Data": text, "Charset": "UTF-8"}},
                    }
                },
            )
        except Exception as error:
            # Provider exceptions may include addresses, signed URLs or credentials.
            audit("auth_mail_failed", None)
            raise MailUnavailable from error
