from __future__ import annotations

import time

import redis.asyncio as redis
from redis.exceptions import RedisError

from .models import RateLimitDecision, FixedWindowRateLimitPolicy
from .service import RateLimiter, RateLimitBackendError


# Lua script executed atomically by Redis
LUA_RATE_LIMIT_SCRIPT = """
local current = redis.call("INCR", KEYS[1])

if current == 1 then
    redis.call("EXPIRE", KEYS[1], ARGV[1])
end

local ttl = redis.call("TTL", KEYS[1])

return {current, ttl}
"""


class RedisFixedWindowRateLimiter(RateLimiter):

    def __init__(self, client: redis.Redis, key_prefix: str = "heliot:rate_limit"):
        self._client = client
        self._key_prefix = key_prefix
        self._script = self._client.register_script(LUA_RATE_LIMIT_SCRIPT)

    def _build_window_key(self, identifier: str, policy: FixedWindowRateLimitPolicy, now: int) -> str:
        window_index = now // policy.window_seconds
        return f"{self._key_prefix}:{identifier}:window:{window_index}"

    async def check_and_consume(self, identifier: str, policy: FixedWindowRateLimitPolicy) -> RateLimitDecision:
        if not identifier:
            raise ValueError("identifier is required")

        now = int(time.time())
        key = self._build_window_key(identifier, policy, now)

        try:
            result = await self._script(
                keys=[key],
                args=[policy.window_seconds],
            )

        except RedisError as e:
            raise RateLimitBackendError(f"Redis rate limit backend error: {e}") from e

        current_count = int(result[0])
        ttl = int(result[1])

        if ttl < 0:
            ttl = policy.window_seconds

        allowed = current_count <= policy.limit
        remaining = max(0, policy.limit - current_count)

        return RateLimitDecision(
            allowed=allowed,
            remaining=remaining,
            retry_after_seconds=ttl if not allowed else None,
            limit=policy.limit,
            window_seconds=policy.window_seconds,
        )