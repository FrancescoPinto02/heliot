from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from redis.exceptions import RedisError

from cdss.heliot.api.rate_limit.models import (
    RateLimitDecision,
    TokenBucketRateLimitPolicy,
)
from cdss.heliot.api.rate_limit.redis_token_bucket import (
    LUA_TOKEN_BUCKET_SCRIPT,
    RedisTokenBucketRateLimiter,
)
from cdss.heliot.api.rate_limit.service import RateLimitBackendError


class TestRedisTokenBucketRateLimiterInit:
    def test_registers_lua_script_on_init(self):
        """
        GOAL: the token bucket limiter should register its Lua script during initialization.
        """
        # Arrange
        client = Mock()
        script = AsyncMock()
        client.register_script.return_value = script

        # Act
        limiter = RedisTokenBucketRateLimiter(client, key_prefix="heliot:test")

        # Assert
        assert limiter._script is script
        client.register_script.assert_called_once_with(LUA_TOKEN_BUCKET_SCRIPT)


class TestRedisTokenBucketRateLimiterBuildKeys:
    def test_builds_tokens_and_timestamp_keys(self):
        """
        GOAL: _build_keys should generate the tokens and timestamp keys with the configured prefix.
        """
        # Arrange
        client = Mock()
        client.register_script.return_value = AsyncMock()
        limiter = RedisTokenBucketRateLimiter(client, key_prefix="heliot:test")

        # Act
        tokens_key, ts_key = limiter._build_keys("user-123")

        # Assert
        assert tokens_key == "heliot:test:user-123:token_bucket:tokens"
        assert ts_key == "heliot:test:user-123:token_bucket:ts"


class TestRedisTokenBucketRateLimiterCheckAndConsume:
    @pytest.mark.asyncio
    async def test_raises_value_error_when_identifier_is_empty(self):
        """
        GOAL: check_and_consume should reject empty identifiers.
        """
        # Arrange
        client = Mock()
        client.register_script.return_value = AsyncMock()
        limiter = RedisTokenBucketRateLimiter(client)
        policy = TokenBucketRateLimitPolicy(capacity=10, refill_rate_per_second=2.0)

        # Act / Assert
        with pytest.raises(ValueError, match="identifier is required"):
            await limiter.check_and_consume("", policy)

    @pytest.mark.asyncio
    async def test_returns_allowed_decision_when_request_is_allowed(self, monkeypatch):
        """
        GOAL: check_and_consume should return an allowed decision and floor the remaining token count.
        """
        # Arrange
        client = Mock()
        script = AsyncMock(return_value=[1, 4.9, 0])
        client.register_script.return_value = script
        limiter = RedisTokenBucketRateLimiter(client, key_prefix="heliot:test")
        policy = TokenBucketRateLimitPolicy(capacity=5, refill_rate_per_second=1.5)

        monkeypatch.setattr(
            "cdss.heliot.api.rate_limit.redis_token_bucket.time.time",
            lambda: 123.5,
        )

        # Act
        decision = await limiter.check_and_consume("user-123", policy)

        # Assert
        assert isinstance(decision, RateLimitDecision)
        assert decision.allowed is True
        assert decision.remaining == 4
        assert decision.retry_after_seconds == 0
        assert decision.limit == 5

        script.assert_awaited_once_with(
            keys=[
                "heliot:test:user-123:token_bucket:tokens",
                "heliot:test:user-123:token_bucket:ts",
            ],
            args=[5, 1.5, 123.5],
        )

    @pytest.mark.asyncio
    async def test_returns_denied_decision_when_request_is_rejected(self, monkeypatch):
        """
        GOAL: check_and_consume should return a denied decision with retry_after_seconds from Redis.
        """
        # Arrange
        client = Mock()
        script = AsyncMock(return_value=[0, 0.7, 2])
        client.register_script.return_value = script
        limiter = RedisTokenBucketRateLimiter(client)
        policy = TokenBucketRateLimitPolicy(capacity=5, refill_rate_per_second=1.0)

        monkeypatch.setattr(
            "cdss.heliot.api.rate_limit.redis_token_bucket.time.time",
            lambda: 100.0,
        )

        # Act
        decision = await limiter.check_and_consume("user-123", policy)

        # Assert
        assert decision.allowed is False
        assert decision.remaining == 0
        assert decision.retry_after_seconds == 2
        assert decision.limit == 5

    @pytest.mark.asyncio
    async def test_clamps_negative_remaining_to_zero_after_floor(self, monkeypatch):
        """
        GOAL: check_and_consume should clamp remaining tokens to zero when the script returns a negative value.
        """
        # Arrange
        client = Mock()
        script = AsyncMock(return_value=[0, -0.2, 1])
        client.register_script.return_value = script
        limiter = RedisTokenBucketRateLimiter(client)
        policy = TokenBucketRateLimitPolicy(capacity=5, refill_rate_per_second=1.0)

        monkeypatch.setattr(
            "cdss.heliot.api.rate_limit.redis_token_bucket.time.time",
            lambda: 100.0,
        )

        # Act
        decision = await limiter.check_and_consume("user-123", policy)

        # Assert
        assert decision.allowed is False
        assert decision.remaining == 0
        assert decision.retry_after_seconds == 1
        assert decision.limit == 5

    @pytest.mark.asyncio
    async def test_wraps_redis_error_as_backend_error(self, monkeypatch):
        """
        GOAL: check_and_consume should wrap RedisError into RateLimitBackendError.
        """
        # Arrange
        client = Mock()
        script = AsyncMock(side_effect=RedisError("boom"))
        client.register_script.return_value = script
        limiter = RedisTokenBucketRateLimiter(client)
        policy = TokenBucketRateLimitPolicy(capacity=5, refill_rate_per_second=1.0)

        monkeypatch.setattr(
            "cdss.heliot.api.rate_limit.redis_token_bucket.time.time",
            lambda: 100.0,
        )

        # Act / Assert
        with pytest.raises(RateLimitBackendError, match="Redis token bucket backend error: boom"):
            await limiter.check_and_consume("user-123", policy)