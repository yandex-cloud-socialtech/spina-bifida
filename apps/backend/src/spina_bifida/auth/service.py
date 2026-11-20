from __future__ import annotations

from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from argon2.low_level import Type
from flask_sqlalchemy.session import Session as FlaskSession
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, scoped_session

from spina_bifida.auth.models import SupportedLocale, User, UserRole, UserStatus, utc_now
from spina_bifida.extensions import db
from spina_bifida.observability.audit import audit

type SessionType = Session | scoped_session[Session] | scoped_session[FlaskSession]


class DuplicateEmail(ValueError):
    pass


class InvalidUserField(ValueError):
    pass


class UserNotFound(LookupError):
    pass


class LastActiveAdmin(ValueError):
    pass


class SelfBlockForbidden(ValueError):
    pass


_password_hasher = PasswordHasher(type=Type.ID)
_dummy_password_hash = _password_hasher.hash("nonexistent-account-password")


class AuthService:
    def __init__(self, session: SessionType | None = None) -> None:
        self.session: SessionType = session or db.session

    def create_user(
        self,
        email: str,
        password: str,
        role: str,
        locale: str,
        *,
        actor_id: str | None = None,
        status: UserStatus = UserStatus.ACTIVE,
    ) -> User:
        normalized_email = self._normalize_email(email)
        validated_role = self._role(role)
        validated_locale = self._locale(locale)
        if not 12 <= len(password) <= 256:
            raise InvalidUserField("password")

        user = User(
            email=normalized_email,
            password_hash=_password_hasher.hash(password),
            role=validated_role,
            status=status,
            email_verified_at=utc_now() if status == UserStatus.ACTIVE else None,
            locale=validated_locale,
        )
        self.session.add(user)
        try:
            self.session.commit()
        except IntegrityError as error:
            self.session.rollback()
            raise DuplicateEmail(normalized_email) from error
        audit(
            "user_created",
            actor_id,
            target_user_id=user.id,
            role=str(user.role),
            locale=str(user.locale),
        )
        return user

    def authenticate(self, email: str, password: str) -> User | None:
        normalized_email = email.strip().casefold()
        user = self.session.scalar(select(User).where(User.email == normalized_email))
        candidate_hash = user.password_hash if user is not None else _dummy_password_hash
        verified = self._verify(candidate_hash, password)
        if user is None or user.status != UserStatus.ACTIVE or not verified:
            audit("login_failed", None)
            return None
        audit("login_succeeded", user.id)
        return user

    def list_users(self) -> list[User]:
        return list(
            self.session.scalars(
                select(User)
                .where(User.status != UserStatus.DELETED)
                .order_by(User.created_at, User.id),
            )
        )

    def get_active_user(self, user_id: str) -> User | None:
        user = self.session.get(User, user_id)
        if user is None or user.status != UserStatus.ACTIVE:
            return None
        return user

    def block_user(
        self,
        user_id: str,
        *,
        blocked: bool,
        actor_id: str | None = None,
    ) -> User:
        user = self._required(user_id)
        if blocked and user_id == actor_id:
            raise SelfBlockForbidden(user_id)
        if user.status not in (UserStatus.ACTIVE, UserStatus.BLOCKED):
            raise InvalidUserField("status")
        if blocked and self._is_last_active_admin(user):
            raise LastActiveAdmin(user_id)
        if user.status != UserStatus.DELETED:
            user.status = UserStatus.BLOCKED if blocked else UserStatus.ACTIVE
            self.session.commit()
            audit(
                "user_status_changed",
                actor_id,
                target_user_id=user.id,
                status=str(user.status),
            )
        return user

    def update_user(
        self,
        user_id: str,
        *,
        role: str | None = None,
        locale: str | None = None,
        blocked: bool | None = None,
        actor_id: str | None = None,
    ) -> User:
        user = self._required(user_id)
        if blocked and user_id == actor_id:
            raise SelfBlockForbidden(user_id)
        new_role = self._role(role) if role is not None else None
        if new_role != UserRole.ADMIN and role is not None and self._is_last_active_admin(user):
            raise LastActiveAdmin(user_id)
        if new_role is not None:
            user.role = new_role
        if locale is not None:
            user.locale = self._locale(locale)
        if blocked is not None:
            if user.status not in (UserStatus.ACTIVE, UserStatus.BLOCKED):
                raise InvalidUserField("status")
            if blocked and self._is_last_active_admin(user):
                raise LastActiveAdmin(user_id)
            user.status = UserStatus.BLOCKED if blocked else UserStatus.ACTIVE
        self.session.commit()
        audit(
            "user_updated",
            actor_id,
            target_user_id=user.id,
            role=str(user.role),
            locale=str(user.locale),
            status=str(user.status),
        )
        return user

    def delete_user(self, user_id: str, *, actor_id: str | None = None) -> User:
        user = self._required(user_id)
        if self._is_last_active_admin(user):
            raise LastActiveAdmin(user_id)
        user.email = f"deleted-{uuid4()}@invalid.local"
        user.password_hash = ""
        user.status = UserStatus.DELETED
        self.session.commit()
        audit("user_deleted", actor_id, target_user_id=user.id)
        return user

    def _required(self, user_id: str) -> User:
        user = self.session.get(User, user_id)
        if user is None or user.status == UserStatus.DELETED:
            raise UserNotFound(user_id)
        return user

    def _is_last_active_admin(self, user: User) -> bool:
        if user.role != UserRole.ADMIN or user.status != UserStatus.ACTIVE:
            return False
        count = self.session.scalar(
            select(func.count())
            .select_from(User)
            .where(
                User.role == UserRole.ADMIN,
                User.status == UserStatus.ACTIVE,
            ),
        )
        return count == 1

    @staticmethod
    def _normalize_email(email: str) -> str:
        normalized = email.strip().casefold()
        local, separator, domain = normalized.partition("@")
        if (
            not separator
            or not local
            or "." not in domain
            or "@" in domain
            or any(c.isspace() or ord(c) < 32 for c in normalized)
            or len(normalized) > 320
        ):
            raise InvalidUserField("email")
        return normalized

    @staticmethod
    def _role(role: str) -> UserRole:
        try:
            return UserRole(role)
        except ValueError as error:
            raise InvalidUserField("role") from error

    @staticmethod
    def _locale(locale: str) -> SupportedLocale:
        try:
            return SupportedLocale(locale)
        except ValueError as error:
            raise InvalidUserField("locale") from error

    @staticmethod
    def _verify(password_hash: str, password: str) -> bool:
        try:
            return _password_hasher.verify(password_hash, password)
        except (InvalidHashError, VerifyMismatchError):
            return False
