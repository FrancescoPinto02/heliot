from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import hmac
import secrets
from typing import Final, Iterable


TOKEN_PRODUCT_PREFIX: Final[str] = "hl_sk"
TOKEN_SEPARATOR: Final[str] = "_"
ALLOWED_ENVS: Final[set[str]] = {"test", "prod"}


@dataclass(frozen=True, slots=True)
class TokenParts:
    """Parsed token components."""
    env: str
    prefix: str
    secret: str
    raw: str  # normalized (stripped) token string


def _b64url_no_pad(data: bytes) -> str:
    """Encode bytes to base64url without '=' padding."""
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def generate_api_token(env: str, prefix_len: int = 12, secret_bytes: int = 32) -> tuple[str, str]:
    """
    Generate a new API token and its public prefix.

    Args:
        env: Deployment environment string ("test" or "prod").
        prefix_len: Length of the public prefix.
        secret_bytes: Secret length in bytes.

    Returns:
        (token_plain, key_prefix)
    """
    env = env.strip().lower()
    if env not in ALLOWED_ENVS:
        raise ValueError(f"Invalid env '{env}'. Allowed: {sorted(ALLOWED_ENVS)}")

    if prefix_len < 8:
        raise ValueError("prefix_len too short; must be >= 8")
    if secret_bytes < 16:
        raise ValueError("secret_bytes too short; must be >= 16")

    # Prefix: use urlsafe chars and truncate; ensure it doesn't contain separator.
    prefix = ""
    while len(prefix) < prefix_len:
        chunk = secrets.token_urlsafe(16).replace("_", "").replace("-", "")
        prefix += chunk
    prefix = prefix[:prefix_len]

    # Secret: base64url without padding, derived from secure random bytes.
    secret = _b64url_no_pad(secrets.token_bytes(secret_bytes))

    token_plain = TOKEN_SEPARATOR.join([TOKEN_PRODUCT_PREFIX, env, prefix, secret])
    return token_plain, prefix


def parse_api_token(token: str) -> TokenParts:
    """
    Parse and validate the token format.
    This does NOT verify authenticity; it only validates syntax and extracts fields.

    Args:
        token: Token string.

    Returns:
        TokenParts(env, prefix, secret, raw)

    Raises:
        ValueError: If token is invalid.
    """
    if token is None:
        raise ValueError("Token is None")

    raw = token.strip()
    if not raw:
        raise ValueError("Token is empty")

    expected_prefix = f"{TOKEN_PRODUCT_PREFIX}{TOKEN_SEPARATOR}"
    if not raw.startswith(expected_prefix):
        raise ValueError("Malformed token: invalid product prefix")

    # Remove "hl_sk_"
    remainder = raw[len(expected_prefix):]

    # Split only first 2 separators
    # env, prefix, secret (secret may contain '_')
    try:
        env, prefix, secret = remainder.split(TOKEN_SEPARATOR, 2)
    except ValueError:
        raise ValueError("Malformed token: expected env, prefix and secret")

    env = env.strip().lower()

    if env not in ALLOWED_ENVS:
        raise ValueError("Malformed token: invalid env")

    if not prefix or len(prefix) < 8:
        raise ValueError("Malformed token: invalid prefix")

    if not secret or len(secret) < 20:
        raise ValueError("Malformed token: invalid secret")

    return TokenParts(
        env=env,
        prefix=prefix,
        secret=secret,
        raw=raw,
    )

def extract_prefix(token: str) -> str:
    """
    Extract the public prefix from a token without returning all parsed parts.

    Args:
        token: Token string.
    Returns:
        The public key prefix string.
    """
    parts = parse_api_token(token)
    return parts.prefix

def compute_token_hmac(token_plain: str, pepper: bytes) -> bytes:
    """
    Compute the HMAC-SHA256 of the full token string using a server-side pepper.

    Args:
        token_plain: The full token string as received/generated.
        pepper: Secret bytes stored server-side (NOT in the DB).

    Returns:
        32-byte HMAC digest.
    """
    if not isinstance(pepper, (bytes, bytearray)) or len(pepper) < 16:
        raise ValueError("pepper must be bytes and at least 16 bytes long")

    msg = token_plain.encode("utf-8")
    return hmac.new(bytes(pepper), msg, hashlib.sha256).digest()


def constant_time_equal(a: bytes, b: bytes) -> bool:
    """
    Compare two byte strings in constant time to mitigate timing attacks.

    Always use this for comparing stored hash vs computed hash.
    """
    if not isinstance(a, (bytes, bytearray)) or not isinstance(b, (bytes, bytearray)):
        return False
    return hmac.compare_digest(a, b)