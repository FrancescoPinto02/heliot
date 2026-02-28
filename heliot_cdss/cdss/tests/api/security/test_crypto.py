import base64
import hashlib
import hmac

import pytest

from cdss.heliot.api.security.crypto import b64url_decode, b64url_encode, hmac_sha256, constant_time_equal


@pytest.mark.parametrize(
    "raw",
    [
        b"",  # Empty input must be supported
        b"f",  # 1 byte -> padding would normally be "=="
        b"fo",  # 2 bytes -> padding would normally be "="
        b"foo",  # 3 bytes -> no padding
        b"\x00\xff\x10\x80",  # Binary bytes (non-ASCII) must be handled
        b"hello world",  # Typical text payload
    ],
)
def test_b64url_encode_matches_stdlib_urlsafe_without_padding(raw: bytes) -> None:
    # Goal: Ensure encoding is URL-safe base64 and padding is removed, matching stdlib behavior.

    # Arrange: compute expected urlsafe base64 then remove padding
    expected = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    # Act
    encoded = b64url_encode(raw)

    # Assert
    assert encoded == expected


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"f",
        b"fo",
        b"foo",
        b"foobar",
        b"\x00\xff\x10\x80",
    ],
)
def test_b64url_roundtrip_encode_decode(raw: bytes) -> None:
    # Goal: Verify encode/decode are inverse operations (lossless roundtrip).

    # Act
    encoded = b64url_encode(raw)
    decoded = b64url_decode(encoded)

    # Assert
    assert decoded == raw


@pytest.mark.parametrize(
    "raw",
    [
        b"f",      # would be "Zg=="
        b"fo",     # would be "Zm8="
        b"foo",    # would be "Zm9v"
        b"foob",   # would be "Zm9vYg=="
    ],
)
def test_b64url_decode_accepts_missing_padding(raw: bytes) -> None:
    # Goal: Ensure decoder correctly restores missing '=' padding.

    # Arrange: Create a standard urlsafe base64 string *with* padding
    padded = base64.urlsafe_b64encode(raw).decode("ascii")
    unpadded = padded.rstrip("=")

    # Act
    decoded = b64url_decode(unpadded)

    # Assert
    assert decoded == raw


@pytest.mark.parametrize(
    "raw",
    [
        b"\xfb\xef",  # produces '-' and '_' in urlsafe sometimes; good to exercise URL-safe alphabet
        b"\xff\xff\xff",
        b"\x00\x00\x00\x00\x01",
    ],
)
def test_b64url_encode_is_urlsafe_alphabet(raw: bytes) -> None:
    # Goal: Ensure the output contains only URL-safe base64 characters (no '+' or '/')

    encoded = b64url_encode(raw)

    assert "+" not in encoded
    assert "/" not in encoded
    assert "=" not in encoded


@pytest.mark.parametrize(
    "data",
    [
        "a b",      # whitespace not valid in strict base64 contexts
        "€",        # non-ASCII characters must fail at .encode("ascii")
    ],
)
def test_b64url_decode_rejects_invalid_input(data: str) -> None:
    # Goal: Validate that malformed inputs fail fast rather than producing silent corruption.

    # Act/Assert
    with pytest.raises((ValueError, UnicodeEncodeError)):
        b64url_decode(data)


def test_hmac_sha256_matches_stdlib_digest() -> None:
    # Goal: Verify HMAC-SHA256 is computed correctly by cross-checking against stdlib.

    # Arrange
    key = b"super-secret-key"
    msg = b"important message"
    expected = hmac.new(key, msg, hashlib.sha256).digest()

    # Act
    digest = hmac_sha256(key, msg)

    # Assert
    assert digest == expected


def test_hmac_sha256_has_expected_length() -> None:
    # Goal: Ensure the function returns the raw digest (32 bytes for SHA-256),

    digest = hmac_sha256(b"k", b"m")
    assert isinstance(digest, (bytes, bytearray))
    assert len(digest) == 32


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (b"", b"", True),                 # identical empties
        (b"abc", b"abc", True),           # identical payloads
        (b"abc", b"abd", False),          # single-byte difference
        (b"abc", b"ab", False),           # different length
        (b"\x00\x01", b"\x00\x01", True), # binary equality
        (b"\x00\x01", b"\x00\x02", False),
    ],
)
def test_constant_time_equal_correctness(a: bytes, b: bytes, expected: bool) -> None:
    # Goal: Verify functional correctness of constant-time comparison.

    assert constant_time_equal(a, b) is expected
