from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from redis.exceptions import RedisError

# Adjust imports to your real module paths
from cdss.heliot.api.rate_limit.models import (
    FixedWindowRateLimitPolicy,
    RateLimitDecision,
)
from cdss.heliot.api.rate_limit.redis_fixed_window import (
    LUA_RATE_LIMIT_SCRIPT,
    RedisFixedWindowRateLimiter,
)
from cdss.heliot.api.rate_limit.service import RateLimitBackendError


class TestRedisFixedWindowRateLimiterInit:
    def test_registers_lua_script_on_init(self):
        """
        GOAL: the rate limiter should register the Lua script during initialization.
        """
        # Arrange
        client = Mock()
        script = AsyncMock()
        client.register_script.return_value = script

        # Act
        limiter = RedisFixedWindowRateLimiter(client, key_prefix="heliot:test")

        # Assert
        assert limiter._script is script
        client.register_script.assert_called_once_with(LUA_RATE_LIMIT_SCRIPT)


class TestRedisFixedWindowRateLimiterBuildWindowKey:
    def test_builds_key_with_prefix_identifier_and_window_index(self):
        """
        GOAL: the window key should include prefix, identifier, and the computed fixed-window index.
        """
        # Arrange
        client = Mock()
        client.register_script.return_value = AsyncMock()
        limiter = RedisFixedWindowRateLimiter(client, key_prefix="heliot:test")
        policy = FixedWindowRateLimitPolicy(limit=10, window_seconds=60)

        # Act
        key = limiter._build_window_key("user-123", policy, now=125)

        # Assert
        assert key == "heliot:test:user-123:window:2"


class TestRedisFixedWindowRateLimiterCheckAndConsume:
    @pytest.mark.asyncio
    async def test_raises_value_error_when_identifier_is_empty(self):
        """
        GOAL: check_and_consume should reject empty identifiers.
        """
        # Arrange
        client = Mock()
        client.register_script.return_value = AsyncMock()
        limiter = RedisFixedWindowRateLimiter(client)
        policy = FixedWindowRateLimitPolicy(limit=10, window_seconds=60)

        # Act / Assert
        with pytest.raises(ValueError, match="identifier is required"):
            await limiter.check_and_consume("", policy)

    @pytest.mark.asyncio
    async def test_returns_allowed_decision_when_under_limit(self, monkeypatch):
        """
        GOAL: check_and_consume should allow the request and compute remaining quota when current count is within limit.
        """
        # Arrange
        client = Mock()
        script = AsyncMock(return_value=[3, 42])
        client.register_script.return_value = script
        limiter = RedisFixedWindowRateLimiter(client, key_prefix="heliot:test")
        policy = FixedWindowRateLimitPolicy(limit=10, window_seconds=60)

        monkeypatch.setattr(
            "cdss.heliot.api.rate_limit.redis_fixed_window.time.time",
            lambda: 125,
        )

        # Act
        decision = await limiter.check_and_consume("user-123", policy)

        # Assert
        assert isinstance(decision, RateLimitDecision)
        assert decision.allowed is True
        assert decision.remaining == 7
        assert decision.retry_after_seconds is None
        assert decision.limit == 10
        assert decision.window_seconds == 60

        script.assert_awaited_once_with(
            keys=["heliot:test:user-123:window:2"],
            args=[60],
        )

    @pytest.mark.asyncio
    async def test_returns_denied_decision_when_limit_is_exceeded(self, monkeypatch):
        """
        GOAL: check_and_consume should deny the request and expose retry_after_seconds when current count exceeds limit.
        """
        # Arrange
        client = Mock()
        script = AsyncMock(return_value=[11, 18])
        client.register_script.return_value = script
        limiter = RedisFixedWindowRateLimiter(client)
        policy = FixedWindowRateLimitPolicy(limit=10, window_seconds=60)

        monkeypatch.setattr(
            "cdss.heliot.api.rate_limit.redis_fixed_window.time.time",
            lambda: 125,
        )

        # Act
        decision = await limiter.check_and_consume("user-123", policy)

        # Assert
        assert decision.allowed is False
        assert decision.remaining == 0
        assert decision.retry_after_seconds == 18
        assert decision.limit == 10
        assert decision.window_seconds == 60

    @pytest.mark.asyncio
    async def test_falls_back_to_policy_window_when_redis_ttl_is_negative(self, monkeypatch):
        """
        GOAL: check_and_consume should fallback to policy.window_seconds when Redis returns a negative TTL.
        """
        # Arrange
        client = Mock()
        script = AsyncMock(return_value=[11, -1])
        client.register_script.return_value = script
        limiter = RedisFixedWindowRateLimiter(client)
        policy = FixedWindowRateLimitPolicy(limit=10, window_seconds=60)

        monkeypatch.setattr(
            "cdss.heliot.api.rate_limit.redis_fixed_window.time.time",
            lambda: 125,
        )

        # Act
        decision = await limiter.check_and_consume("user-123", policy)

        # Assert
        assert decision.allowed is False
        assert decision.remaining == 0
        assert decision.retry_after_seconds == 60

    @pytest.mark.asyncio
    async def test_wraps_redis_error_as_backend_error(self, monkeypatch):
        """
        GOAL: check_and_consume should wrap RedisError into RateLimitBackendError.
        """
        # Arrange
        client = Mock()
        script = AsyncMock(side_effect=RedisError("boom"))
        client.register_script.return_value = script
        limiter = RedisFixedWindowRateLimiter(client)
        policy = FixedWindowRateLimitPolicy(limit=10, window_seconds=60)

        monkeypatch.setattr(
            "cdss.heliot.api.rate_limit.redis_fixed_window.time.time",
            lambda: 125,
        )

        # Act / Assert
        with pytest.raises(RateLimitBackendError, match="Redis rate limit backend error: boom"):
            await limiter.check_and_consume("user-123", policy)