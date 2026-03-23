from __future__ import annotations

from fastapi import Request

from .models import BaseRateLimitPolicy, FixedWindowRateLimitPolicy, TokenBucketRateLimitPolicy
from .redis_fixed_window import RedisFixedWindowRateLimiter
from .redis_token_bucket import RedisTokenBucketRateLimiter
from .service import RateLimiter
from ..services.api_key_service import AuthContext


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

    if algorithm == "token_bucket":
        token_bucket_cfg = rate_limit_cfg.get("token_bucket", {})

        try:
            capacity = int(token_bucket_cfg.get("capacity", 10))
            refill_rate_per_second = float(
                token_bucket_cfg.get("refill_rate_per_second", 1.0)
            )
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Invalid token_bucket rate-limit configuration: {e}") from e

        return TokenBucketRateLimitPolicy(
            capacity=capacity,
            refill_rate_per_second=refill_rate_per_second,
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

    if algorithm == "token_bucket":
        return RedisTokenBucketRateLimiter(
            client=redis_client,
            key_prefix="heliot:rate_limit",
        )

    raise RuntimeError(f"Unsupported rate limit algorithm: {algorithm}")


def build_rate_limit_identifier(auth: AuthContext, config: dict) -> str:
    """
    Build the rate-limit identifier based on configured scope.

    Supported scopes:
    - api_key
    - project
    """
    rate_limit_cfg = config.get("rate_limit", {})
    scope = rate_limit_cfg.get("scope", "api_key")

    if scope == "api_key":
        return f"api_key:{auth.api_key_id}"

    if scope == "project":
        return f"project:{auth.project_id}"

    raise RuntimeError(f"Unsupported rate limit scope: {scope}")