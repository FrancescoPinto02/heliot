from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError

from ..models.api_key_db import ApiKey
from ..models.project_db import Project
from ..security.pepper import PepperProvider, PepperNotFoundError, PepperConfigError
from ..security.tokens import (
    TokenParts,
    constant_time_equal,
    compute_token_hmac,
    generate_api_token,
    parse_api_token,
)


class ApiKeyError(Exception):
    """Base class for API key service errors."""


class ApiKeyNotFoundError(ApiKeyError):
    """Raised when an API key cannot be found (e.g., revoke by prefix)."""


class ApiKeyAlreadyRevokedError(ApiKeyError):
    """Raised when revoking an already revoked key."""


class ApiKeyCreateError(ApiKeyError):
    """Raised when API key creation fails due to invalid inputs."""


@dataclass(frozen=True, slots=True)
class ApiKeyCreate:
    """Input DTO for creating an API key."""
    project_id: int
    env: str
    name: str | None = None
    description: str | None = None
    scopes: Sequence[str] = ()
    expires_at: datetime | None = None
    pepper_version: int = 1
    # token generation params (can be kept default in most cases)
    prefix_len: int = 12
    secret_bytes: int = 32


@dataclass(frozen=True, slots=True)
class AuthContext:
    """Returned on successful verification."""
    project_id: int
    api_key_id: int
    scopes: tuple[str, ...]
    env: str
    key_prefix: str


class ApiKeyService:
    """
    Service layer for API keys.

    Args:
        db: SQLAlchemy Session
        pepper_provider: provider to resolve pepper bytes by version
    """

    def __init__(self, db: Session, pepper_provider: PepperProvider):
        self._db = db
        self._peppers = pepper_provider

    def create(self, data: ApiKeyCreate) -> tuple[str, ApiKey]:
        """
        Create a new API key for a project.

        Returns:
            (token_plain, api_key_row)

        Notes:
            - token_plain MUST be shown only once to the user
            - only prefix + hash are stored in DB
            - retries on rare unique-constraint collisions
        """
        if data.project_id <= 0:
            raise ApiKeyCreateError("project_id must be > 0")

        project = self._db.get(Project, data.project_id)
        if project is None:
            raise ApiKeyCreateError(f"Project id={data.project_id} not found")

        if data.expires_at is not None and data.expires_at.tzinfo is None:
            raise ApiKeyCreateError("expires_at must be timezone-aware (UTC recommended)")

        # Resolve pepper once (no need to do it per retry).
        try:
            pepper = self._peppers.get(data.pepper_version)
        except (PepperNotFoundError, PepperConfigError) as e:
            raise ApiKeyCreateError(f"Pepper configuration error: {e}")

        max_retries = 5

        for attempt in range(1, max_retries + 1):
            token_plain, prefix = generate_api_token(
                env=data.env,
                prefix_len=data.prefix_len,
                secret_bytes=data.secret_bytes,
            )
            token_hash = compute_token_hmac(token_plain=token_plain, pepper=pepper)

            api_key = ApiKey(
                project_id=data.project_id,
                env=data.env.strip().lower(),
                key_prefix=prefix,
                key_hash=token_hash,
                pepper_version=int(data.pepper_version),
                name=(data.name.strip() if data.name else None),
                description=(data.description.strip() if data.description else None),
                scopes=list(data.scopes) if data.scopes else [],
                is_active=True,
                expires_at=data.expires_at,
                revoked_at=None,
                last_used_at=None,
            )

            self._db.add(api_key)

            try:
                self._db.commit()
                self._db.refresh(api_key)
                return token_plain, api_key

            except IntegrityError:
                # Unique constraint violation: prefix collision (extremely rare) or
                # hash collision (practically impossible).
                self._db.rollback()

                if attempt == max_retries:
                    raise ApiKeyCreateError(
                        "Failed to create API key due to repeated unique constraint collisions"
                    )

                # Retry by generating a fresh token/prefix.
                continue

        raise ApiKeyCreateError("Failed to create API key")


    def verify(self, token_plain: str, *, update_last_used: bool = False) -> Optional[AuthContext]:
        """
        Verify a token and return an AuthContext on success, otherwise None.

        Verification checks:
          - token parsing & syntax
          - api key exists (prefix lookup)
          - key and project are active
          - not revoked
          - not expired
          - HMAC matches (constant-time compare)

        Args:
            token_plain: token received from client (Authorization header value)
            update_last_used: if True, updates last_used_at (write per successful auth)

        Returns:
            AuthContext if valid else None.
        """
        try:
            parts = parse_api_token(token_plain)
        except ValueError:
            return None

        # Fetch key by prefix and join project for a single round-trip.
        api_key: ApiKey | None = (
            self._db.query(ApiKey)
            .options(joinedload(ApiKey.project))
            .filter(ApiKey.key_prefix == parts.prefix)
            .one_or_none()
        )
        if api_key is None:
            return None

        # Fast state checks before crypto work (cheap rejects).
        if not api_key.is_active:
            return None
        if api_key.revoked_at is not None:
            return None

        now = datetime.now(timezone.utc)
        if api_key.expires_at is not None:
            # expires_at is stored timezone-aware; compare in UTC.
            if api_key.expires_at <= now:
                return None

        # Project state check
        if api_key.project is None:
            return None
        if not api_key.project.is_active:
            return None

        # Resolve pepper and compute HMAC
        try:
            pepper = self._peppers.get(int(api_key.pepper_version))
        except (PepperNotFoundError, PepperConfigError):
            # Treat missing/invalid pepper as auth failure (safe default).
            return None

        computed = compute_token_hmac(token_plain=parts.raw, pepper=pepper)
        if not constant_time_equal(computed, api_key.key_hash):
            return None

        # Optional write (may be disabled for performance/scalability reasons).
        if update_last_used:
            api_key.last_used_at = now
            self._db.add(api_key)
            self._db.commit()

        return AuthContext(
            project_id=api_key.project_id,
            api_key_id=api_key.id,
            scopes=tuple(api_key.scopes or []),
            env=api_key.env,
            key_prefix=api_key.key_prefix,
        )


    def revoke_by_prefix(self, prefix: str) -> ApiKey:
        """
        Revoke an API key by its public prefix.

        This keeps the record for audit/logging, but marks it as revoked.
        """
        prefix = (prefix or "").strip()
        if not prefix:
            raise ValueError("prefix is required")

        api_key: ApiKey | None = (
            self._db.query(ApiKey)
            .filter(ApiKey.key_prefix == prefix)
            .one_or_none()
        )
        if api_key is None:
            raise ApiKeyNotFoundError(f"API key prefix='{prefix}' not found")

        if api_key.revoked_at is not None:
            raise ApiKeyAlreadyRevokedError(f"API key prefix='{prefix}' already revoked")

        now = datetime.now(timezone.utc)
        api_key.revoked_at = now
        api_key.is_active = False

        self._db.add(api_key)
        self._db.commit()
        self._db.refresh(api_key)
        return api_key


    def list_by_project(self, project_id: int, *, include_inactive: bool = True) -> list[ApiKey]:
        """
        List API keys for a project (never includes token plaintext).

        Args:
            include_inactive: if False, returns only active & not revoked keys.
        """
        q = (
            self._db.query(ApiKey)
            .filter(ApiKey.project_id == project_id)
            .order_by(ApiKey.id.asc())
        )

        if not include_inactive:
            q = q.filter(ApiKey.is_active.is_(True)).filter(ApiKey.revoked_at.is_(None))

        return q.all()