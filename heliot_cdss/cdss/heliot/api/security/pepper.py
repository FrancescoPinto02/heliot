from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Dict, Protocol


class PepperNotFoundError(Exception):
    """Raised when a pepper for the requested version is not configured."""


class PepperConfigError(Exception):
    """Raised when a pepper exists but is invalid (wrong type/format/length)."""


class PepperProvider(Protocol):
    """Interface for providing pepper bytes by version."""

    def get(self, version: int) -> bytes:
        """Return pepper bytes for the given version, or raise PepperNotFoundError."""
        ...


def _validate_pepper(pepper: bytes) -> bytes:
    if not isinstance(pepper, (bytes, bytearray)):
        raise PepperConfigError("Pepper must be bytes")
    pepper_b = bytes(pepper)
    if len(pepper_b) < 16:
        # 16 bytes is a reasonable minimum; 32+ is better.
        raise PepperConfigError("Pepper too short (min 16 bytes)")
    return pepper_b


@dataclass(frozen=True, slots=True)
class StaticPepperProvider:
    """
    A simple in-memory provider (useful for tests/dev).

    Example:
        provider = StaticPepperProvider({1: b"supersecretpepper..."})
        pepper = provider.get(1)
    """
    peppers: Dict[int, bytes]

    def get(self, version: int) -> bytes:
        try:
            pepper = self.peppers[int(version)]
        except (KeyError, ValueError):
            raise PepperNotFoundError(f"Pepper version={version} not found")
        return _validate_pepper(pepper)


@dataclass(frozen=True, slots=True)
class EnvPepperProvider:
    """
    Reads peppers from environment variables.

    By default uses:
        HELIOT_API_KEY_PEPPER_V{N}

    Values are expected to be base64 (recommended) or raw text.

    If base64 decoding fails, it falls back to UTF-8 bytes of the string.
    """

    var_template: str = "HELIOT_API_KEY_PEPPER_V{version}"
    prefer_base64: bool = True

    def get(self, version: int) -> bytes:
        v = int(version)
        var_name = self.var_template.format(version=v)
        value = os.environ.get(var_name)
        if not value:
            raise PepperNotFoundError(f"{var_name} is not set")

        value = value.strip()
        if not value:
            raise PepperNotFoundError(f"{var_name} is empty")

        pepper_bytes: bytes

        if self.prefer_base64:
            # Accept base64 / base64url (with or without padding).
            try:
                padded = value + "=" * (-len(value) % 4)
                pepper_bytes = base64.urlsafe_b64decode(padded.encode("ascii"))
            except Exception:
                pepper_bytes = value.encode("utf-8")
        else:
            pepper_bytes = value.encode("utf-8")

        return _validate_pepper(pepper_bytes)