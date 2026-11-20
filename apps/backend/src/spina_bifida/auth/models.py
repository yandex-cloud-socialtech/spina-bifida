from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from spina_bifida.extensions import db

if TYPE_CHECKING:
    from spina_bifida.analyses.models import AnalysisJob, Feedback


def utc_now() -> datetime:
    return datetime.now(UTC)


class UserRole(StrEnum):
    USER = "user"
    ADMIN = "admin"


class UserStatus(StrEnum):
    PENDING_EMAIL = "pending_email"
    PENDING = "pending"
    REJECTED = "rejected"
    ACTIVE = "active"
    BLOCKED = "blocked"
    DELETED = "deleted"


class SupportedLocale(StrEnum):
    RU = "ru"
    EN = "en"
    KK = "kk"
    UZ = "uz"


class User(db.Model):  # type: ignore[misc, name-defined]
    __tablename__ = "users"
    __table_args__ = (Index("ix_users_email", "email", unique=True),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    locale: Mapped[str] = mapped_column(String(8), nullable=False, default="ru")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    email_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    password_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
    )

    jobs: Mapped[list[AnalysisJob]] = relationship(back_populates="owner")
    feedback: Mapped[list[Feedback]] = relationship(back_populates="author")


class AuthToken(db.Model):  # type: ignore[misc, name-defined]
    __tablename__ = "auth_tokens"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    purpose: Mapped[str] = mapped_column(String(24), nullable=False)
    password_version: Mapped[int] = mapped_column(Integer, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
