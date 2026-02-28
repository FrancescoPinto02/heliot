import base64
import hashlib
import hmac
from typing import Final


def b64url_encode(raw: bytes) -> str:
    """URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def b64url_decode(data: str) -> bytes:
    """Decode URL-safe base64 without padding."""
    padding = "=" * ((4 - (len(data) % 4)) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def hmac_sha256(key: bytes, msg: bytes) -> bytes:
    """Return raw 32-byte HMAC-SHA256 digest."""
    return hmac.new(key, msg, hashlib.sha256).digest()


def constant_time_equal(a: bytes, b: bytes) -> bool:
    """Constant-time comparison to avoid timing leaks."""
    return hmac.compare_digest(a, b)