from __future__ import annotations

import logging
import math
import time

import redis.asyncio as redis
from redis.exceptions import RedisError

from .models import RateLimitDecision, TokenBucketRateLimitPolicy
from .service import RateLimiter, RateLimitBackendError

logger = logging.getLogger(__name__)

LUA_TOKEN_BUCKET_SCRIPT = """
local tokens_key = KEYS[1]
local ts_key = KEYS[2]

local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local tokens = tonumber(redis.call("GET", tokens_key))
local last_ts = tonumber(redis.call("GET", ts_key))

if tokens == nil then
    tokens = capacity
end

if last_ts == nil then
    last_ts = now
end

local elapsed = math.max(0, now - last_ts)
local refilled_tokens = math.min(capacity, tokens + (elapsed * refill_rate))

local allowed = 0
local remaining = refilled_tokens
local retry_after = 0

if refilled_tokens >= 1.0 then
    allowed = 1
    remaining = refilled_tokens - 1.0
else
    retry_after = math.ceil((1.0 - refilled_tokens) / refill_rate)
end

redis.call("SET", tokens_key, remaining)
redis.call("SET", ts_key, now)

-- TTL just prevents stale buckets from living forever.
local ttl = math.ceil(capacity / refill_rate * 2)
redis.call("EXPIRE", tokens_key, ttl)
redis.call("EXPIRE", ts_key, ttl)

return {allowed, remaining, retry_after}
"""


class RedisTokenBucketRateLimiter(RateLimiter):
    def __init__(self, client: redis.Redis, key_prefix: str = "heliot:rate_limit"):
        self._client = client
        self._key_prefix = key_prefix
        self._script = self._client.register_script(LUA_TOKEN_BUCKET_SCRIPT)

    def _build_keys(self, identifier: str) -> tuple[str, str]:
        base = f"{self._key_prefix}:{identifier}:token_bucket"
        return f"{base}:tokens", f"{base}:ts"

    async def check_and_consume(self, identifier: str, policy: TokenBucketRateLimitPolicy) -> RateLimitDecision:
        if not identifier:
            raise ValueError("identifier is required")

        now = time.time()
        tokens_key, ts_key = self._build_keys(identifier)

        try:
            result = await self._script(
                keys=[tokens_key, ts_key],
                args=[policy.capacity, policy.refill_rate_per_second, now],
            )
        except RedisError as e:
            raise RateLimitBackendError(f"Redis token bucket backend error: {e}") from e

        allowed = bool(int(result[0]))
        remaining_float = float(result[1])
        retry_after = int(result[2])

        # expose remaining as an integer for HTTP/header friendliness.
        remaining = max(0, math.floor(remaining_float))

        return RateLimitDecision(
            allowed=allowed,
            remaining=remaining,
            retry_after_seconds=retry_after if not allowed else 0,
            limit=policy.capacity,
        )