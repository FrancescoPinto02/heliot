from __future__ import annotations

import sys
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, status

fake_db_session = ModuleType("cdss.heliot.api.db.session")
fake_db_session.get_db = lambda: None
fake_db_session.Base = object()

sys.modules["cdss.heliot.api.db.session"] = fake_db_session

fake_project_db = ModuleType("cdss.heliot.api.models.project_db")
fake_project_db.Project = type("Project", (), {})

fake_api_key_db = ModuleType("cdss.heliot.api.models.api_key_db")
fake_api_key_db.ApiKey = type("ApiKey", (), {})

sys.modules["cdss.heliot.api.models.project_db"] = fake_project_db
sys.modules["cdss.heliot.api.models.api_key_db"] = fake_api_key_db

import cdss.heliot.api.auth.deps as m


class TestRequireApiKey:
    def test_raises_401_when_authorization_header_is_missing(self):
        """
        GOAL: require_api_key should reject requests without an Authorization header.
        """
        # Arrange
        db = Mock(name="db_session")

        # Act / Assert
        with pytest.raises(HTTPException) as exc_info:
            m.require_api_key(authorization=None, db=db)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Missing Authorization header"

    @pytest.mark.parametrize(
        "authorization",
        [
            "Bearer",
            "Basic abc",
            "Token abc",
            "Bearer    ",
            "abc def ghi",
        ],
    )
    def test_raises_401_when_authorization_header_format_is_invalid(self, authorization: str):
        """
        GOAL: require_api_key should reject malformed Authorization headers that are not valid Bearer tokens.
        """
        # Arrange
        db = Mock(name="db_session")

        # Act / Assert
        with pytest.raises(HTTPException) as exc_info:
            m.require_api_key(authorization=authorization, db=db)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid Authorization header format"

    def test_raises_401_when_service_verification_fails(self, monkeypatch):
        """
        GOAL: require_api_key should return 401 when ApiKeyService.verify returns None.
        """
        # Arrange
        db = Mock(name="db_session")

        service_instance = Mock(name="api_key_service")
        service_instance.verify.return_value = None

        service_ctor = Mock(return_value=service_instance)
        pepper_provider_ctor = Mock(return_value=Mock(name="pepper_provider"))

        monkeypatch.setattr(m, "ApiKeyService", service_ctor)
        monkeypatch.setattr(m, "EnvPepperProvider", pepper_provider_ctor)

        authorization = "Bearer my-token"

        # Act / Assert
        with pytest.raises(HTTPException) as exc_info:
            m.require_api_key(authorization=authorization, db=db)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid API key"

        pepper_provider_ctor.assert_called_once_with()
        service_ctor.assert_called_once_with(db, pepper_provider_ctor.return_value)
        service_instance.verify.assert_called_once_with("my-token", update_last_used=False)

    def test_raises_401_when_authorization_header_is_empty(self):
        """
        GOAL: require_api_key should treat an empty Authorization header as missing.
        """
        # Arrange
        db = Mock(name="db_session")

        # Act / Assert
        with pytest.raises(HTTPException) as exc_info:
            m.require_api_key(authorization="", db=db)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Missing Authorization header"

    def test_returns_auth_context_when_service_verification_succeeds(self, monkeypatch):
        """
        GOAL: require_api_key should return the AuthContext produced by ApiKeyService.verify on success.
        """
        # Arrange
        db = Mock(name="db_session")

        auth_context = SimpleNamespace(
            project_id=1,
            api_key_id=10,
            scopes=("read", "write"),
            env="prod",
            key_prefix="pref123",
        )

        service_instance = Mock(name="api_key_service")
        service_instance.verify.return_value = auth_context

        service_ctor = Mock(return_value=service_instance)
        pepper_provider_ctor = Mock(return_value=Mock(name="pepper_provider"))

        monkeypatch.setattr(m, "ApiKeyService", service_ctor)
        monkeypatch.setattr(m, "EnvPepperProvider", pepper_provider_ctor)

        authorization = "Bearer valid-token"

        # Act
        result = m.require_api_key(authorization=authorization, db=db)

        # Assert
        assert result is auth_context
        pepper_provider_ctor.assert_called_once_with()
        service_ctor.assert_called_once_with(db, pepper_provider_ctor.return_value)
        service_instance.verify.assert_called_once_with("valid-token", update_last_used=False)

    def test_strips_token_before_verification(self, monkeypatch):
        """
        GOAL: require_api_key should strip surrounding whitespace from the Bearer token before verification.
        """
        # Arrange
        db = Mock(name="db_session")

        auth_context = SimpleNamespace(
            project_id=1,
            api_key_id=10,
            scopes=(),
            env="prod",
            key_prefix="pref123",
        )

        service_instance = Mock(name="api_key_service")
        service_instance.verify.return_value = auth_context

        service_ctor = Mock(return_value=service_instance)
        pepper_provider_ctor = Mock(return_value=Mock(name="pepper_provider"))

        monkeypatch.setattr(m, "ApiKeyService", service_ctor)
        monkeypatch.setattr(m, "EnvPepperProvider", pepper_provider_ctor)

        authorization = "Bearer    valid-token-with-spaces   "

        # Act
        result = m.require_api_key(authorization=authorization, db=db)

        # Assert
        assert result is auth_context
        service_instance.verify.assert_called_once_with("valid-token-with-spaces", update_last_used=False)