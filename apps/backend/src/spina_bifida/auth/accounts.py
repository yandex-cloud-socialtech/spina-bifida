from __future__ import annotations

import secrets
from datetime import timedelta
from hashlib import sha256
from urllib.parse import urlencode

from sqlalchemy import delete, select, update

from spina_bifida.auth.mail import PostboxMailer
from spina_bifida.auth.models import AuthToken, User, UserRole, UserStatus, utc_now
from spina_bifida.auth.service import (
    AuthService,
    DuplicateEmail,
    InvalidUserField,
    _password_hasher,
)
from spina_bifida.extensions import db
from spina_bifida.observability.audit import audit


class InvalidToken(ValueError):
    pass


class InvalidCurrentPassword(ValueError):
    pass


class AccountService:
    def __init__(self, mailer: PostboxMailer | None = None) -> None:
        self.mailer = mailer or PostboxMailer()
        self.auth = AuthService()

    def register(self, email: str, password: str, locale: str) -> None:
        self.mailer.require_configured()
        normalized = self.auth._normalize_email(email)
        self.auth._locale(locale)
        self._validate_password(password)
        user = db.session.scalar(select(User).where(User.email == normalized))
        if user is None:
            # Reserve the normalized email before sending; public registration
            # always creates an ordinary, unverified account.
            try:
                user = self.auth.create_user(
                    normalized,
                    password,
                    UserRole.USER,
                    locale,
                    status=UserStatus.PENDING_EMAIL,
                )
            except DuplicateEmail:
                user = db.session.scalar(select(User).where(User.email == normalized))
        if user is not None and user.status == UserStatus.PENDING_EMAIL:
            # Bind the latest verification link to the password in this request.
            # A prior unverified signup must not retain an attacker-chosen password.
            db.session.execute(
                update(User)
                .where(
                    User.id == user.id,
                    User.status == UserStatus.PENDING_EMAIL,
                )
                .values(
                    password_hash=_password_hasher.hash(password),
                    password_version=User.password_version + 1,
                )
            )
            db.session.refresh(user)
            if user.status == UserStatus.PENDING_EMAIL:
                self._send_link(user, "verify_email", 24 * 60)
        audit("registration_requested", None)

    def request_reset(self, email: str) -> None:
        self.mailer.require_configured()
        normalized = self.auth._normalize_email(email)
        user = db.session.scalar(select(User).where(User.email == normalized))
        if user is not None and user.status == UserStatus.ACTIVE:
            self._send_link(user, "password_reset", 30)
        audit("password_reset_requested", None)

    def _send_link(self, user: User, purpose: str, minutes: int) -> None:
        raw = secrets.token_urlsafe(32)
        now = utc_now()
        # Delete expired tokens; retain only the latest link for this purpose.
        db.session.execute(delete(AuthToken).where(AuthToken.expires_at < now))
        db.session.execute(
            delete(AuthToken).where(
                AuthToken.user_id == user.id,
                AuthToken.purpose == purpose,
            )
        )
        db.session.add(
            AuthToken(
                token_hash=sha256(raw.encode()).hexdigest(),
                user_id=user.id,
                purpose=purpose,
                password_version=user.password_version,
                expires_at=now + timedelta(minutes=minutes),
            )
        )
        path = "verify-email" if purpose == "verify_email" else "reset-password"
        # Fragment stays out of reverse-proxy/access logs and Referer headers.
        url = (
            self.mailer.settings.public_url.rstrip("/")
            + "/"
            + path
            + "#"
            + urlencode(
                {"token": raw},
            )
        )
        if purpose == "verify_email":
            subject = "Spina Bifida — подтвердите электронную почту / Verify your email"
            text = (
                f"Подтвердите адрес в течение 24 часов: {url}\n"
                "После подтверждения дождитесь одобрения администратора.\n"
                "Если вы не регистрировались, проигнорируйте письмо.\n\n"
                "Verify your email within 24 hours using the link above. "
                "Administrator approval is required before signing in."
            )
        else:
            subject = "Spina Bifida — восстановление пароля / Reset your password"
            text = (
                f"Для установки нового пароля перейдите по ссылке в течение 30 минут: {url}\n"
                "Если вы не запрашивали сброс, проигнорируйте письмо.\n\n"
                "Use the link above within 30 minutes to reset your password. "
                "Ignore this email if you did not request a reset."
            )
        # Release SQLite's writer lock before network I/O, so the ML worker
        # can still commit progress if the mail provider is slow or unavailable.
        db.session.commit()
        self.mailer.send(user.email, subject, text)

    def _consume(self, raw: str, purpose: str) -> User:
        if not 20 <= len(raw) <= 256:
            raise InvalidToken
        # Conditional UPDATE consumes a token once even across concurrent requests.
        token = db.session.execute(
            update(AuthToken)
            .where(
                AuthToken.token_hash == sha256(raw.encode()).hexdigest(),
                AuthToken.purpose == purpose,
                AuthToken.used_at.is_(None),
                AuthToken.expires_at > utc_now(),
            )
            .values(used_at=utc_now())
            .returning(
                AuthToken.user_id,
                AuthToken.password_version,
            )
        ).first()
        if token is None:
            db.session.rollback()
            raise InvalidToken
        user = db.session.get(User, token.user_id)
        status = UserStatus.PENDING_EMAIL if purpose == "verify_email" else UserStatus.ACTIVE
        if user is None or user.password_version != token.password_version or user.status != status:
            db.session.rollback()
            raise InvalidToken
        return user

    def verify_email(self, raw: str) -> None:
        user = self._consume(raw, "verify_email")
        user.email_verified_at = utc_now()
        user.status = UserStatus.PENDING
        db.session.commit()
        audit("email_verified", user.id)

    def moderate(self, user_id: str, approved: bool, actor_id: str) -> User:
        user = self.auth._required(user_id)
        if user.status != UserStatus.PENDING or user.email_verified_at is None:
            raise InvalidUserField("status")
        status = UserStatus.ACTIVE if approved else UserStatus.REJECTED
        text = (
            (
                "Ваша регистрация одобрена. Теперь вы можете войти в сервис.\n"
                "Your registration has been approved. You can now sign in."
            )
            if approved
            else (
                "Ваша заявка на регистрацию отклонена администратором.\n"
                "Your registration request has been declined by the administrator."
            )
        )
        try:
            # The supported single API process sends before taking a DB writer
            # lock; provider failure keeps the request pending and retryable.
            self.mailer.send(user.email, "Spina Bifida — результат модерации / Registration", text)
            changed = db.session.execute(
                update(User)
                .where(
                    User.id == user.id,
                    User.status == UserStatus.PENDING,
                )
                .values(status=status)
                .returning(User.id)
            ).scalar_one_or_none()
            if changed is None:
                raise InvalidUserField("status")
            db.session.commit()
        except Exception:
            db.session.rollback()
            raise
        db.session.refresh(user)
        audit("registration_moderated", actor_id, target_user_id=user.id, status=str(status))
        return user

    def reset_password(self, raw: str, password: str) -> None:
        self._validate_password(password)
        user = self._consume(raw, "password_reset")
        self._set_password(user, password)

    def change_password(self, user: User, current_password: str, password: str) -> None:
        self._validate_password(password)
        if not self.auth._verify(user.password_hash, current_password):
            raise InvalidCurrentPassword
        self._set_password(user, password)

    @staticmethod
    def _validate_password(password: str) -> None:
        if not 12 <= len(password) <= 256:
            raise InvalidUserField("password")

    @staticmethod
    def _set_password(user: User, password: str) -> None:
        changed = db.session.execute(
            update(User)
            .where(
                User.id == user.id,
                User.password_version == user.password_version,
                User.status == UserStatus.ACTIVE,
            )
            .values(
                password_hash=_password_hasher.hash(password),
                password_version=User.password_version + 1,
            )
            .returning(User.id)
        ).scalar_one_or_none()
        if changed is None:
            db.session.rollback()
            raise InvalidToken
        db.session.execute(delete(AuthToken).where(AuthToken.user_id == user.id))
        db.session.commit()
        audit("password_changed", user.id)
