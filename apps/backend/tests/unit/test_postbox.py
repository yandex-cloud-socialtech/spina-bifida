import boto3
import pytest
from botocore.stub import Stubber
from pydantic import SecretStr, ValidationError

from spina_bifida.auth.mail import MailUnavailable, PostboxMailer
from spina_bifida.config import PostboxSettings


def settings(**extra):
    data = dict(
        enabled=True,
        sender="noreply@example.test",
        public_url="https://app.example.test",
        access_key=SecretStr("access-key"),
        secret_key=SecretStr("secret-key"),
    )
    return PostboxSettings(**(data | extra))


def test_postbox_uses_utf8_sesv2_contract():
    client = boto3.client(
        "sesv2",
        region_name="ru-central1",
        endpoint_url="https://postbox.cloud.yandex.net",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    with Stubber(client) as stubber:
        stubber.add_response(
            "send_email",
            {"MessageId": "test-message"},
            {
                "FromEmailAddress": "noreply@example.test",
                "Destination": {"ToAddresses": ["doctor@example.test"]},
                "Content": {
                    "Simple": {
                        "Subject": {"Data": "Подтверждение", "Charset": "UTF-8"},
                        "Body": {"Text": {"Data": "Текст письма", "Charset": "UTF-8"}},
                    }
                },
            },
        )
        PostboxMailer(settings(), client).send(
            "doctor@example.test", "Подтверждение", "Текст письма"
        )
        stubber.assert_no_pending_responses()


def test_postbox_error_does_not_log_provider_exception(monkeypatch):
    events = []
    monkeypatch.setattr("spina_bifida.auth.mail.audit", lambda *args: events.append(args))

    class FailingClient:
        def send_email(self, **kwargs):
            raise RuntimeError("secret-address@example.test secret-key signed-token")

    with pytest.raises(MailUnavailable):
        PostboxMailer(settings(), FailingClient()).send("doctor@example.test", "Subject", "Body")
    assert events == [("auth_mail_failed", None)]


def test_postbox_secret_values_are_excluded_from_serialization():
    value = settings()
    assert "secret-key" not in repr(value)
    assert "access-key" not in repr(value)
    assert "secret-key" not in value.model_dump_json()


@pytest.mark.parametrize(
    "origin",
    [
        "http://app.example.test",
        "https://app.example.test/?token=private",
        "https://user:password@app.example.test",
        "https://app.example.test/#private",
    ],
)
def test_postbox_links_require_a_fixed_https_origin(origin):
    with pytest.raises(ValidationError):
        settings(public_url=origin)
