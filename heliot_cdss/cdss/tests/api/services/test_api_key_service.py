from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import IntegrityError

class _Field:
    """Minimal SQLAlchemy-like field stub to support ==, .asc(), .is_()."""
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        return ("eq", self.name, other)

    def asc(self):
        return ("asc", self.name)

    def is_(self, other):
        return ("is", self.name, other)


fake_project_db = ModuleType("cdss.heliot.api.models.project_db")
fake_api_key_db = ModuleType("cdss.heliot.api.models.api_key_db")


class FakeProject:
    id = _Field("id")
    name = _Field("name")
    is_active = _Field("is_active")

    def __init__(self, *, id: int, is_active: bool = True):
        self.id = id
        self.is_active = is_active


class FakeApiKey:
    id = _Field("id")
    project_id = _Field("project_id")
    env = _Field("env")
    key_prefix = _Field("key_prefix")
    key_hash = _Field("key_hash")
    pepper_version = _Field("pepper_version")
    is_active = _Field("is_active")
    revoked_at = _Field("revoked_at")
    project = _Field("project")

    def __init__(self, **kwargs):
        # store any provided attributes (mimics SQLAlchemy model instance)
        for k, v in kwargs.items():
            setattr(self, k, v)

        # defaults used by service checks
        self.id = getattr(self, "id", 123)
        self.project = getattr(self, "project", None)


fake_project_db.Project = FakeProject
fake_api_key_db.ApiKey = FakeApiKey

sys.modules["cdss.heliot.api.models.project_db"] = fake_project_db
sys.modules["cdss.heliot.api.models.api_key_db"] = fake_api_key_db

from cdss.heliot.api.services.api_key_service import (
    ApiKeyService,
    ApiKeyCreate,
    AuthContext,
    ApiKeyCreateError,
    ApiKeyNotFoundError,
    ApiKeyAlreadyRevokedError,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _make_db_mock() -> Mock:
    db = Mock(name="db_session")
    db.get = Mock(name="get")
    db.add = Mock(name="add")
    db.commit = Mock(name="commit")
    db.refresh = Mock(name="refresh")
    db.query = Mock(name="query")
    return db

@pytest.fixture(autouse=True)
def _patch_joinedload(monkeypatch):
    """
    Avoid SQLAlchemy loader strategy evaluation in pure unit tests.
    """
    import cdss.heliot.api.services.api_key_service as m
    monkeypatch.setattr(m, "joinedload", lambda *args, **kwargs: object())


class _PepperProvider:
    def __init__(self, pepper: bytes | None = None, exc: Exception | None = None):
        self._pepper = pepper
        self._exc = exc

    def get(self, version: int) -> bytes:
        if self._exc:
            raise self._exc
        assert self._pepper is not None
        return self._pepper


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

class TestApiKeyServiceCreate:
    def test_rejects_non_positive_project_id(self, monkeypatch):
        """
        GOAL: create should reject project_id <= 0 with ApiKeyCreateError.
        """
        # Arrange
        db = _make_db_mock()
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        data = ApiKeyCreate(project_id=0, env="prod")

        # Act / Assert
        with pytest.raises(ApiKeyCreateError, match="project_id must be > 0"):
            svc.create(data)

        db.get.assert_not_called()
        db.add.assert_not_called()
        db.commit.assert_not_called()

    def test_rejects_missing_project(self):
        """
        GOAL: create should fail if the referenced project does not exist.
        """
        # Arrange
        db = _make_db_mock()
        db.get.return_value = None
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        data = ApiKeyCreate(project_id=1, env="prod")

        # Act / Assert
        with pytest.raises(ApiKeyCreateError, match=r"Project id=1 not found"):
            svc.create(data)

        db.get.assert_called_once()
        db.add.assert_not_called()

    def test_rejects_naive_expires_at(self):
        """
        GOAL: create should reject naive expires_at values to avoid timezone bugs.
        """
        # Arrange
        db = _make_db_mock()
        db.get.return_value = FakeProject(id=1, is_active=True)
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        naive = datetime(2030, 1, 1)  # tzinfo=None
        data = ApiKeyCreate(project_id=1, env="prod", expires_at=naive)

        # Act / Assert
        with pytest.raises(ApiKeyCreateError, match="expires_at must be timezone-aware"):
            svc.create(data)

        db.add.assert_not_called()
        db.commit.assert_not_called()

    def test_wraps_pepper_provider_errors_as_create_error(self, monkeypatch):
        """
        GOAL: create should wrap Pepper resolution errors into ApiKeyCreateError.
        """
        # Arrange
        db = _make_db_mock()
        db.get.return_value = FakeProject(id=1, is_active=True)

        from cdss.heliot.api.security.pepper import PepperNotFoundError

        peppers = _PepperProvider(exc=PepperNotFoundError("missing"))
        svc = ApiKeyService(db, peppers)

        data = ApiKeyCreate(project_id=1, env="prod")

        # Act / Assert
        with pytest.raises(ApiKeyCreateError, match=r"Pepper configuration error:"):
            svc.create(data)

        db.add.assert_not_called()
        db.commit.assert_not_called()

    def test_persists_key_and_returns_token_plain_on_success(self, monkeypatch):
        """
        GOAL: create should generate token, compute hash, persist ApiKey, and return (token_plain, api_key).
        """
        # Arrange
        db = _make_db_mock()
        db.get.return_value = FakeProject(id=1, is_active=True)

        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        # Patch token + HMAC functions for determinism
        import cdss.heliot.api.services.api_key_service as m

        monkeypatch.setattr(m, "generate_api_token", lambda env, prefix_len, secret_bytes: ("hl_sk_prod_PREF_secret", "PREF"))
        monkeypatch.setattr(m, "compute_token_hmac", lambda token_plain, pepper: b"H" * 32)

        data = ApiKeyCreate(
            project_id=1,
            env="prod",
            name="  My Key  ",
            description="  Desc  ",
            scopes=("a", "b"),
            pepper_version=1,
        )

        # Act
        token_plain, api_key = svc.create(data)

        # Assert
        assert token_plain == "hl_sk_prod_PREF_secret"
        assert api_key.project_id == 1
        assert api_key.key_prefix == "PREF"
        assert api_key.key_hash == b"H" * 32
        assert api_key.pepper_version == 1
        assert api_key.name == "My Key"
        assert api_key.description == "Desc"
        assert api_key.scopes == ["a", "b"]
        assert api_key.is_active is True
        assert api_key.revoked_at is None
        assert api_key.last_used_at is None

        db.add.assert_called_once_with(api_key)
        db.commit.assert_called_once_with()
        db.refresh.assert_called_once_with(api_key)

    def test_retries_on_integrity_error_then_succeeds(self, monkeypatch):
        """
        GOAL: create should rollback and retry on IntegrityError collisions, then succeed and return the new key.
        """
        # Arrange
        db = _make_db_mock()
        db.get.return_value = FakeProject(id=1, is_active=True)

        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        import cdss.heliot.api.services.api_key_service as m

        # Deterministic "token generation" across attempts
        tokens = [
            ("hl_sk_prod_PREF1_secret", "PREF1"),
            ("hl_sk_prod_PREF2_secret", "PREF2"),
        ]
        gen_mock = Mock(side_effect=tokens)
        monkeypatch.setattr(m, "generate_api_token", gen_mock)

        # Deterministic HMAC output
        monkeypatch.setattr(m, "compute_token_hmac", lambda token_plain, pepper: b"H" * 32)

        # First commit fails (collision), second succeeds
        db.commit.side_effect = [
            IntegrityError("stmt", "params", Exception("orig")),
            None,
        ]

        data = ApiKeyCreate(project_id=1, env="prod", pepper_version=1)

        # Act
        token_plain, api_key = svc.create(data)

        # Assert
        assert token_plain == "hl_sk_prod_PREF2_secret"
        assert api_key.key_prefix == "PREF2"
        assert api_key.key_hash == b"H" * 32

        assert gen_mock.call_count == 2
        assert db.rollback.call_count == 1
        assert db.commit.call_count == 2
        db.refresh.assert_called_once_with(api_key)

    def test_raises_after_max_retries_on_repeated_integrity_error(self, monkeypatch):
        """
        GOAL: create should raise ApiKeyCreateError after max retries if IntegrityError keeps happening.
        """
        # Arrange
        db = _make_db_mock()
        db.get.return_value = FakeProject(id=1, is_active=True)

        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        import cdss.heliot.api.services.api_key_service as m

        # Token generation always returns *something*; the failure is forced by commit.
        monkeypatch.setattr(m, "generate_api_token",
                            lambda env, prefix_len, secret_bytes: ("hl_sk_prod_PREF_secret", "PREF"))
        monkeypatch.setattr(m, "compute_token_hmac", lambda token_plain, pepper: b"H" * 32)

        db.commit.side_effect = IntegrityError("stmt", "params", Exception("orig"))

        data = ApiKeyCreate(project_id=1, env="prod", pepper_version=1)

        # Act / Assert
        with pytest.raises(ApiKeyCreateError, match="repeated unique constraint collisions"):
            svc.create(data)

        assert db.rollback.call_count == 5
        assert db.commit.call_count == 5
        db.refresh.assert_not_called()


class TestApiKeyServiceVerify:
    def test_returns_none_on_parse_error(self, monkeypatch):
        """
        GOAL: verify should return None when token parsing fails.
        """
        # Arrange
        db = _make_db_mock()
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        import cdss.heliot.api.services.api_key_service as m  # ADJUST ME
        monkeypatch.setattr(m, "parse_api_token", lambda token_plain: (_ for _ in ()).throw(ValueError("bad")))

        # Act
        result = svc.verify("bad-token")

        # Assert
        assert result is None
        db.query.assert_not_called()

    def test_returns_none_when_key_not_found_by_prefix(self, monkeypatch):
        """
        GOAL: verify should return None when no ApiKey exists for the parsed prefix.
        """
        # Arrange
        db = _make_db_mock()
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        import cdss.heliot.api.services.api_key_service as m  # ADJUST ME
        parts = SimpleNamespace(env="prod", prefix="PREF", raw="token-raw")
        monkeypatch.setattr(m, "parse_api_token", lambda _: parts)

        q = Mock()
        db.query.return_value = q
        q.options.return_value = q
        q.filter.return_value = q
        q.one_or_none.return_value = None

        # Act
        result = svc.verify("token")

        # Assert
        assert result is None

    def test_returns_none_on_fast_state_rejects(self, monkeypatch):
        """
        GOAL: verify should reject inactive or revoked keys before doing crypto work.
        """
        # Arrange
        db = _make_db_mock()
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        import cdss.heliot.api.services.api_key_service as m  # ADJUST ME
        parts = SimpleNamespace(env="prod", prefix="PREF", raw="token-raw")
        monkeypatch.setattr(m, "parse_api_token", lambda _: parts)

        # Build an ApiKey row
        api_key = FakeApiKey(
            id=10,
            project_id=1,
            key_prefix="PREF",
            key_hash=b"H" * 32,
            pepper_version=1,
            scopes=["s1"],
            is_active=False,
            revoked_at=None,
            expires_at=None,
            project=FakeProject(id=1, is_active=True),
        )

        q = Mock()
        db.query.return_value = q
        q.options.return_value = q
        q.filter.return_value = q
        q.one_or_none.return_value = api_key

        compute_mock = Mock()
        monkeypatch.setattr(m, "compute_token_hmac", compute_mock)

        # Act
        result = svc.verify("token")

        # Assert
        assert result is None
        compute_mock.assert_not_called()

    def test_returns_none_when_expired(self, monkeypatch):
        """
        GOAL: verify should return None when the key is expired.
        """
        # Arrange
        db = _make_db_mock()
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        import cdss.heliot.api.services.api_key_service as m  # ADJUST ME
        parts = SimpleNamespace(env="prod", prefix="PREF", raw="token-raw")
        monkeypatch.setattr(m, "parse_api_token", lambda _: parts)

        fixed_now = datetime(2030, 1, 1, tzinfo=timezone.utc)

        class _FakeDateTime:
            @staticmethod
            def now(tz=None):
                return fixed_now

        monkeypatch.setattr(m, "datetime", _FakeDateTime)

        api_key = FakeApiKey(
            id=10,
            project_id=1,
            key_prefix="PREF",
            key_hash=b"H" * 32,
            pepper_version=1,
            scopes=["s1"],
            is_active=True,
            revoked_at=None,
            expires_at=fixed_now,  # <= now => expired
            project=FakeProject(id=1, is_active=True),
        )

        q = Mock()
        db.query.return_value = q
        q.options.return_value = q
        q.filter.return_value = q
        q.one_or_none.return_value = api_key

        # Act
        result = svc.verify("token")

        # Assert
        assert result is None

    def test_returns_none_when_project_inactive_or_missing(self, monkeypatch):
        """
        GOAL: verify should fail when the associated project is missing or inactive.
        """
        # Arrange
        db = _make_db_mock()
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        import cdss.heliot.api.services.api_key_service as m  # ADJUST ME
        parts = SimpleNamespace(env="prod", prefix="PREF", raw="token-raw")
        monkeypatch.setattr(m, "parse_api_token", lambda _: parts)

        q = Mock()
        db.query.return_value = q
        q.options.return_value = q
        q.filter.return_value = q

        api_key_missing_project = FakeApiKey(
            id=10, project_id=1, key_prefix="PREF", key_hash=b"H" * 32, pepper_version=1,
            scopes=["s1"], is_active=True, revoked_at=None, expires_at=None, project=None
        )
        q.one_or_none.return_value = api_key_missing_project

        # Act
        assert svc.verify("token") is None

        api_key_inactive_project = FakeApiKey(
            id=10, project_id=1, key_prefix="PREF", key_hash=b"H" * 32, pepper_version=1,
            scopes=["s1"], is_active=True, revoked_at=None, expires_at=None, project=FakeProject(id=1, is_active=False)
        )
        q.one_or_none.return_value = api_key_inactive_project

        # Act
        assert svc.verify("token") is None

    def test_returns_none_when_pepper_unavailable(self, monkeypatch):
        """
        GOAL: verify should treat missing/invalid pepper as authentication failure.
        """
        # Arrange
        db = _make_db_mock()

        from cdss.heliot.api.security.pepper import PepperConfigError  # noqa: E402
        peppers = _PepperProvider(exc=PepperConfigError("bad pepper"))

        svc = ApiKeyService(db, peppers)

        import cdss.heliot.api.services.api_key_service as m  # ADJUST ME
        parts = SimpleNamespace(env="prod", prefix="PREF", raw="token-raw")
        monkeypatch.setattr(m, "parse_api_token", lambda _: parts)

        api_key = FakeApiKey(
            id=10, project_id=1, key_prefix="PREF", key_hash=b"H" * 32, pepper_version=1,
            scopes=["s1"], is_active=True, revoked_at=None, expires_at=None, project=FakeProject(id=1, is_active=True)
        )

        q = Mock()
        db.query.return_value = q
        q.options.return_value = q
        q.filter.return_value = q
        q.one_or_none.return_value = api_key

        # Act
        result = svc.verify("token")

        # Assert
        assert result is None

    def test_returns_none_on_hmac_mismatch(self, monkeypatch):
        """
        GOAL: verify should return None when computed HMAC does not match stored hash.
        """
        # Arrange
        db = _make_db_mock()
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        import cdss.heliot.api.services.api_key_service as m  # ADJUST ME
        parts = SimpleNamespace(env="prod", prefix="PREF", raw="token-raw")
        monkeypatch.setattr(m, "parse_api_token", lambda _: parts)
        monkeypatch.setattr(m, "compute_token_hmac", lambda token_plain, pepper: b"A" * 32)
        monkeypatch.setattr(m, "constant_time_equal", lambda a, b: False)

        api_key = FakeApiKey(
            id=10, project_id=1, key_prefix="PREF", key_hash=b"B" * 32, pepper_version=1,
            scopes=["s1"], is_active=True, revoked_at=None, expires_at=None, project=FakeProject(id=1, is_active=True)
        )

        q = Mock()
        db.query.return_value = q
        q.options.return_value = q
        q.filter.return_value = q
        q.one_or_none.return_value = api_key

        # Act
        result = svc.verify("token")

        # Assert
        assert result is None

    def test_returns_auth_context_and_optionally_updates_last_used(self, monkeypatch):
        """
        GOAL: verify should return AuthContext on success and update last_used_at when requested.
        """
        # Arrange
        db = _make_db_mock()
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        import cdss.heliot.api.services.api_key_service as m  # ADJUST ME
        parts = SimpleNamespace(env="prod", prefix="PREF", raw="token-raw")
        monkeypatch.setattr(m, "parse_api_token", lambda _: parts)
        monkeypatch.setattr(m, "compute_token_hmac", lambda token_plain, pepper: b"H" * 32)
        monkeypatch.setattr(m, "constant_time_equal", lambda a, b: True)

        fixed_now = datetime(2030, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        class _FakeDateTime:
            @staticmethod
            def now(tz=None):
                return fixed_now

        monkeypatch.setattr(m, "datetime", _FakeDateTime)

        api_key = FakeApiKey(
            id=10,
            project_id=1,
            key_prefix="PREF",
            key_hash=b"H" * 32,
            pepper_version=1,
            scopes=["s1", "s2"],
            is_active=True,
            revoked_at=None,
            expires_at=fixed_now + timedelta(days=1),
            project=FakeProject(id=1, is_active=True),
            last_used_at=None,
        )

        q = Mock()
        db.query.return_value = q
        q.options.return_value = q
        q.filter.return_value = q
        q.one_or_none.return_value = api_key

        # Act
        ctx = svc.verify("token", update_last_used=True)

        # Assert
        assert isinstance(ctx, AuthContext)
        assert ctx.project_id == 1
        assert ctx.api_key_id == 10
        assert ctx.scopes == ("s1", "s2")
        assert ctx.env == "prod"
        assert ctx.key_prefix == "PREF"

        assert api_key.last_used_at == fixed_now
        db.add.assert_called_with(api_key)
        db.commit.assert_called()  # commit happens for update_last_used


class TestApiKeyServiceRevokeByPrefix:
    def test_rejects_blank_prefix(self):
        """
        GOAL: revoke_by_prefix should reject blank prefixes with ValueError.
        """
        # Arrange
        db = _make_db_mock()
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        # Act / Assert
        with pytest.raises(ValueError, match="prefix is required"):
            svc.revoke_by_prefix("   ")

    def test_raises_not_found_when_missing(self):
        """
        GOAL: revoke_by_prefix should raise ApiKeyNotFoundError when no key exists for prefix.
        """
        # Arrange
        db = _make_db_mock()
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        q = Mock()
        db.query.return_value = q
        q.filter.return_value = q
        q.one_or_none.return_value = None

        # Act / Assert
        with pytest.raises(ApiKeyNotFoundError, match=r"API key prefix='PREF' not found"):
            svc.revoke_by_prefix("PREF")

    def test_raises_already_revoked(self):
        """
        GOAL: revoke_by_prefix should raise ApiKeyAlreadyRevokedError when key is already revoked.
        """
        # Arrange
        db = _make_db_mock()
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        api_key = FakeApiKey(key_prefix="PREF", revoked_at=datetime(2030, 1, 1, tzinfo=timezone.utc), is_active=False)

        q = Mock()
        db.query.return_value = q
        q.filter.return_value = q
        q.one_or_none.return_value = api_key

        # Act / Assert
        with pytest.raises(ApiKeyAlreadyRevokedError, match=r"API key prefix='PREF' already revoked"):
            svc.revoke_by_prefix("PREF")

    def test_revokes_and_persists(self, monkeypatch):
        """
        GOAL: revoke_by_prefix should set revoked_at and deactivate the key, then persist changes.
        """
        # Arrange
        db = _make_db_mock()
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        import cdss.heliot.api.services.api_key_service as m  # ADJUST ME
        fixed_now = datetime(2030, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        class _FakeDateTime:
            @staticmethod
            def now(tz=None):
                return fixed_now

        monkeypatch.setattr(m, "datetime", _FakeDateTime)

        api_key = FakeApiKey(key_prefix="PREF", revoked_at=None, is_active=True)

        q = Mock()
        db.query.return_value = q
        q.filter.return_value = q
        q.one_or_none.return_value = api_key

        # Act
        result = svc.revoke_by_prefix("  PREF  ")

        # Assert
        assert result is api_key
        assert api_key.revoked_at == fixed_now
        assert api_key.is_active is False

        db.add.assert_called_once_with(api_key)
        db.commit.assert_called_once_with()
        db.refresh.assert_called_once_with(api_key)


class TestApiKeyServiceListByProject:
    def test_lists_keys_including_inactive_by_default(self):
        """
        GOAL: list_by_project should return query results ordered by id (default includes inactive).
        """
        # Arrange
        db = _make_db_mock()
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        q = Mock()
        db.query.return_value = q
        q.filter.return_value = q
        q.order_by.return_value = q
        q.all.return_value = ["k1", "k2"]

        # Act
        result = svc.list_by_project(1)

        # Assert
        assert result == ["k1", "k2"]
        q.all.assert_called_once_with()

    def test_lists_only_active_and_not_revoked_when_include_inactive_false(self):
        """
        GOAL: list_by_project should apply filters for active & not revoked when include_inactive is False.
        """
        # Arrange
        db = _make_db_mock()
        peppers = _PepperProvider(pepper=b"p" * 16)
        svc = ApiKeyService(db, peppers)

        q = Mock()
        db.query.return_value = q
        q.filter.return_value = q
        q.order_by.return_value = q
        q.all.return_value = []

        # Act
        result = svc.list_by_project(1, include_inactive=False)

        # Assert
        assert result == []
        assert q.filter.call_count >= 3  # project_id + active + revoked_at is None (don’t over-specify)
        q.all.assert_called_once_with()