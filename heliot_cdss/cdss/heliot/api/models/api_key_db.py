from sqlalchemy import String, Boolean, DateTime, func, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db.session import Base

class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Tenant / client owner
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Public Prefix (Fast Lookup).
    key_prefix: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)

    # Token Hash
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False)

    # metadata
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Scopes List
    scopes: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    expires_at: Mapped["DateTime | None"] = mapped_column(DateTime(timezone=True), nullable=True)

    revoked_at: Mapped["DateTime | None"] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped["DateTime | None"] = mapped_column(DateTime(timezone=True), nullable=True)

    project = relationship("Project", back_populates="api_keys")