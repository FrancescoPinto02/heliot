import base64
import pytest

from cdss.heliot.api.security.tokens import (
    TOKEN_PRODUCT_PREFIX,
    TOKEN_SEPARATOR,
    generate_api_token,
    parse_api_token,
    extract_prefix,
    compute_token_hmac,
    constant_time_equal,
)


def _b64url_decode_no_pad(s: str) -> bytes:
    """Helper: decode base64url strings that may have no '=' padding."""
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode(s + pad)


class TestGenerateApiToken:
    def test_happy_path_returns_well_formed_token_and_prefix(self):
        """
        GOAL: Generating a token should return a well-formed token and a matching public prefix.
        """
        # Arrange
        env = "prod"
        prefix_len = 12
        secret_bytes = 32

        # Act
        token_plain, prefix = generate_api_token(env=env, prefix_len=prefix_len, secret_bytes=secret_bytes)

        # Assert
        parts = parse_api_token(token_plain)
        assert prefix == parts.prefix

    def test_rejects_invalid_env(self):
        """
        GOAL: Token generation must reject unknown deployment environments.
        """
        # Arrange
        invalid_env = "staging"

        # Act / Assert
        with pytest.raises(ValueError, match="Invalid env"):
            generate_api_token(env=invalid_env)


class TestParseApiToken:
    def test_accepts_secret_with_underscores(self):
        """
        GOAL: Parsing must allow secrets containing '_' by splitting only env and prefix first.
        """
        # Arrange
        secret = "secret_part1_part2_extra_long_value"
        token = f"{TOKEN_PRODUCT_PREFIX}_test_abcdEFGH1234_{secret}"

        # Act
        parts = parse_api_token(token)

        # Assert
        assert parts.env == "test"
        assert parts.prefix == "abcdEFGH1234"
        assert parts.secret == secret
        assert parts.raw == token

    def test_strips_and_normalizes_env_to_lowercase(self):
        """
        GOAL: Parsing should strip whitespace and normalize env case without altering raw token (except strip).
        """
        # Arrange
        token = f"  {TOKEN_PRODUCT_PREFIX}_PROD_abcdEFGH1234_thisisaverylongsecretvalue  "

        # Act
        parts = parse_api_token(token)

        # Assert
        assert parts.env == "prod"
        assert parts.raw == token.strip()

    def test_rejects_wrong_product_prefix(self):
        """
        GOAL: Parsing must reject tokens with an invalid product prefix to avoid misrouting credentials.
        """
        # Arrange
        token = "wrong_sk_prod_abcdEFGH1234_secretvalue_long_enough"

        # Act / Assert
        with pytest.raises(ValueError, match="invalid product prefix"):
            parse_api_token(token)

    def test_rejects_invalid_env_value(self):
        """
        GOAL: Parsing must enforce env allowlisting.
        """
        # Arrange
        token = f"{TOKEN_PRODUCT_PREFIX}_dev_abcdEFGH1234_secretvalue_long_enough"

        # Act / Assert
        with pytest.raises(ValueError, match="invalid env"):
            parse_api_token(token)

    def test_rejects_short_prefix(self):
        """
        GOAL: Parsing must reject prefixes shorter than the minimum to ensure consistent lookup security.
        """
        # Arrange
        token = f"{TOKEN_PRODUCT_PREFIX}_prod_short_secretvalue_long_enough"

        # Act / Assert
        with pytest.raises(ValueError, match="invalid prefix"):
            parse_api_token(token)

    def test_rejects_short_secret(self):
        """
        GOAL: Parsing must reject secrets below the minimum length to reduce weak-token risk.
        """
        # Arrange
        token = f"{TOKEN_PRODUCT_PREFIX}_prod_abcdEFGH1234_too_short"

        # Act / Assert
        with pytest.raises(ValueError, match="invalid secret"):
            parse_api_token(token)


class TestExtractPrefix:
    def test_returns_prefix_only(self):
        """
        GOAL: Extracting prefix should return only the public prefix and reuse the same validation rules as parsing.
        """
        # Arrange
        token = f"{TOKEN_PRODUCT_PREFIX}_test_abcdEFGH1234_secret_value_that_is_long_enough"

        # Act
        prefix = extract_prefix(token)

        # Assert
        assert prefix == "abcdEFGH1234"


class TestComputeTokenHmac:
    def test_returns_32_bytes_and_is_deterministic(self):
        """
        GOAL: HMAC computation must return a 32-byte digest and be deterministic for the same inputs.
        """
        # Arrange
        token_plain = f"{TOKEN_PRODUCT_PREFIX}_prod_abcdEFGH1234_secret_value_that_is_long_enough"
        pepper = b"super-secret-pepper-32bytes-min!!!!"  # >= 16 bytes

        # Act
        h1 = compute_token_hmac(token_plain, pepper)
        h2 = compute_token_hmac(token_plain, pepper)

        # Assert
        assert isinstance(h1, (bytes, bytearray))
        assert len(h1) == 32
        assert h1 == h2

    def test_rejects_non_bytes_or_too_short_pepper(self):
        """
        GOAL: HMAC computation must reject invalid peppers (type/length) to prevent weak server-side secrets.
        """
        # Arrange
        token_plain = f"{TOKEN_PRODUCT_PREFIX}_prod_abcdEFGH1234_secret_value_that_is_long_enough"

        # Act / Assert
        with pytest.raises(ValueError, match="pepper must be bytes"):
            compute_token_hmac(token_plain, pepper="not-bytes")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="at least 16 bytes"):
            compute_token_hmac(token_plain, pepper=b"too-short")


class TestConstantTimeEqual:
    def test_matches_equal_and_unequal_bytes_and_rejects_non_bytes(self):
        """
        GOAL: Constant-time comparison should behave correctly for bytes and safely return False for invalid types.
        """
        # Arrange
        a = b"\x01" * 32
        b_same = b"\x01" * 32
        b_diff = b"\x02" * 32

        # Act / Assert
        assert constant_time_equal(a, b_same) is True
        assert constant_time_equal(a, b_diff) is False
        assert constant_time_equal(a, "not-bytes") is False  # type: ignore[arg-type]