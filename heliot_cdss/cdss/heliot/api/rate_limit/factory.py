from __future__ import annotations

from fastapi import Request

from .models import BaseRateLimitPolicy, FixedWindowRateLimitPolicy
from .redis_fixed_window import RedisFixedWindowRateLimiter
from .service import RateLimiter


def build_rate_limit_policy(config: dict) -> BaseRateLimitPolicy:
    """
    Build the rate-limit policy from application config.
    """
    rate_limit_cfg = config.get("rate_limit", {})
    algorithm = rate_limit_cfg.get("algorithm", "fixed_window")

    if algorithm == "fixed_window":
        fixed_window_cfg = rate_limit_cfg.get("fixed_window", {})

        try:
            limit = int(fixed_window_cfg.get("limit", 60))
            window_seconds = int(fixed_window_cfg.get("window_seconds", 60))
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Invalid fixed_window rate-limit configuration: {e}") from e

        return FixedWindowRateLimitPolicy(
            limit=limit,
            window_seconds=window_seconds,
        )

    raise RuntimeError(f"Unsupported rate limit algorithm: {algorithm}")


def build_rate_limiter(request: Request, config: dict) -> RateLimiter:
    """
    Build the concrete rate limiter implementation from application config.
    """
    rate_limit_cfg = config.get("rate_limit", {})
    algorithm = rate_limit_cfg.get("algorithm", "fixed_window")

    redis_client = getattr(request.app.state, "redis", None)
    if redis_client is None:
        raise RuntimeError("Redis client is not initialized")

    if algorithm == "fixed_window":
        return RedisFixedWindowRateLimiter(
            client=redis_client,
            key_prefix="heliot:rate_limit",
        )

    raise RuntimeError(f"Unsupported rate limit algorithm: {algorithm}")