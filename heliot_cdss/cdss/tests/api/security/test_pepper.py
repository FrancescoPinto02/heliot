from __future__ import annotations

import base64
import pytest

from cdss.heliot.api.security.pepper import (
    PepperNotFoundError,
    PepperConfigError,
    StaticPepperProvider,
    EnvPepperProvider,
    _validate_pepper,
)


class TestValidatePepper:
    def test_rejects_non_bytes(self):
        """
        GOAL: _validate_pepper should reject non-bytes inputs to prevent misconfiguration.
        """
        # Arrange
        pepper = "not-bytes"

        # Act / Assert
        with pytest.raises(PepperConfigError, match="Pepper must be bytes"):
            _validate_pepper(pepper)  # type: ignore[arg-type]

    def test_rejects_too_short_bytes(self):
        """
        GOAL: _validate_pepper should reject peppers shorter than the minimum length.
        """
        # Arrange
        pepper = b"short-pepper"  # < 16 bytes

        # Act / Assert
        with pytest.raises(PepperConfigError, match="Pepper too short"):
            _validate_pepper(pepper)

    def test_accepts_bytearray_and_returns_bytes(self):
        """
        GOAL: _validate_pepper should accept bytearray and normalize it to immutable bytes.
        """
        # Arrange
        pepper = bytearray(b"x" * 16)

        # Act
        result = _validate_pepper(pepper)

        # Assert
        assert isinstance(result, bytes)
        assert result == b"x" * 16


class TestStaticPepperProvider:
    def test_returns_validated_pepper_for_version(self):
        """
        GOAL: StaticPepperProvider.get should return validated pepper bytes for an existing version.
        """
        # Arrange
        provider = StaticPepperProvider({1: b"a" * 16})

        # Act
        pepper = provider.get(1)

        # Assert
        assert pepper == b"a" * 16
        assert isinstance(pepper, bytes)

    def test_raises_not_found_for_missing_version(self):
        """
        GOAL: StaticPepperProvider.get should raise PepperNotFoundError when version is missing.
        """
        # Arrange
        provider = StaticPepperProvider({1: b"a" * 16})

        # Act / Assert
        with pytest.raises(PepperNotFoundError, match=r"Pepper version=2 not found"):
            provider.get(2)

    def test_raises_config_error_for_invalid_pepper(self):
        """
        GOAL: StaticPepperProvider.get should raise PepperConfigError when stored pepper is invalid.
        """
        # Arrange
        provider = StaticPepperProvider({1: b"too-short"})

        # Act / Assert
        with pytest.raises(PepperConfigError, match="Pepper too short"):
            provider.get(1)


class TestEnvPepperProvider:
    def test_raises_not_found_when_env_var_is_missing(self, monkeypatch):
        """
        GOAL: EnvPepperProvider.get should raise PepperNotFoundError when env var is not set.
        """
        # Arrange
        provider = EnvPepperProvider()
        monkeypatch.delenv("HELIOT_API_KEY_PEPPER_V1", raising=False)

        # Act / Assert
        with pytest.raises(PepperNotFoundError, match=r"HELIOT_API_KEY_PEPPER_V1 is not set"):
            provider.get(1)

    def test_raises_not_found_when_env_var_is_empty_or_whitespace(self, monkeypatch):
        """
        GOAL: EnvPepperProvider.get should treat empty/whitespace env values as missing.
        """
        # Arrange
        provider = EnvPepperProvider()
        monkeypatch.setenv("HELIOT_API_KEY_PEPPER_V1", "   ")

        # Act / Assert
        with pytest.raises(PepperNotFoundError, match=r"HELIOT_API_KEY_PEPPER_V1 is empty"):
            provider.get(1)

    def test_decodes_base64url_without_padding_when_prefer_base64_true(self, monkeypatch):
        """
        GOAL: EnvPepperProvider.get should accept base64url input with missing padding when prefer_base64 is True.
        """
        # Arrange
        provider = EnvPepperProvider(prefer_base64=True)
        raw = b"p" * 16
        encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")  # remove padding
        monkeypatch.setenv("HELIOT_API_KEY_PEPPER_V1", encoded)

        # Act
        pepper = provider.get(1)

        # Assert
        assert pepper == raw

    def test_falls_back_to_utf8_bytes_when_base64_decode_fails(self, monkeypatch):
        """
        GOAL: EnvPepperProvider.get should fall back to UTF-8 bytes when base64 decoding fails.
        """
        # Arrange
        provider = EnvPepperProvider(prefer_base64=True)
        # Not valid base64 -> should become raw utf-8 bytes.
        value = "non-ascii-🚫-long-enough"
        monkeypatch.setenv("HELIOT_API_KEY_PEPPER_V1", value)

        # Act
        pepper = provider.get(1)

        # Assert
        assert pepper == value.encode("utf-8")

    def test_uses_raw_utf8_when_prefer_base64_false(self, monkeypatch):
        """
        GOAL: EnvPepperProvider.get should use raw UTF-8 bytes when prefer_base64 is False.
        """
        # Arrange
        provider = EnvPepperProvider(prefer_base64=False)
        value = "raw-pepper-value-long-enough"
        monkeypatch.setenv("HELIOT_API_KEY_PEPPER_V1", value)

        # Act
        pepper = provider.get(1)

        # Assert
        assert pepper == value.encode("utf-8")

    def test_raises_config_error_when_decoded_pepper_is_too_short(self, monkeypatch):
        """
        GOAL: EnvPepperProvider.get should raise PepperConfigError when the resulting pepper is too short.
        """
        # Arrange
        provider = EnvPepperProvider(prefer_base64=True)
        raw = b"x" * 15  # invalid (<16)
        encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        monkeypatch.setenv("HELIOT_API_KEY_PEPPER_V1", encoded)

        # Act / Assert
        with pytest.raises(PepperConfigError, match="Pepper too short"):
            provider.get(1)

    def test_supports_custom_var_template(self, monkeypatch):
        """
        GOAL: EnvPepperProvider.get should read from the configured var_template.
        """
        # Arrange
        provider = EnvPepperProvider(var_template="MY_PEPPER_{version}", prefer_base64=False)
        monkeypatch.setenv("MY_PEPPER_7", "custom-template-pepper-long-enough")

        # Act
        pepper = provider.get(7)

        # Assert
        assert pepper == b"custom-template-pepper-long-enough"