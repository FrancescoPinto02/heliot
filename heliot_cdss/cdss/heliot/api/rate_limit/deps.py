from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, Request, Response, status

from ..auth.deps import require_api_key
from ..services.api_key_service import AuthContext
from .factory import build_rate_limit_policy, build_rate_limiter, build_rate_limit_identifier
from .service import RateLimitBackendError

logger = logging.getLogger(__name__)


def get_app_config(request: Request) -> dict:
    config = getattr(request.app.state, "config", None)
    if config is None:
        raise RuntimeError("Application config is not initialized")
    return config


async def require_rate_limit(request: Request, response: Response, auth: AuthContext = Depends(require_api_key)) -> AuthContext:
    config = get_app_config(request)

    policy = build_rate_limit_policy(config)
    limiter = build_rate_limiter(request, config)
    identifier = build_rate_limit_identifier(auth, config)

    try:
        decision = await limiter.check_and_consume(
            identifier=identifier,
            policy=policy,
        )
    except RateLimitBackendError as e:
        logger.warning("WARNING: Rate limiter backend unavailable, allowing request (fail-open): %s", e)
        return auth

    reset_seconds = decision.retry_after_seconds or 0

    response.headers["X-RateLimit-Limit"] = str(decision.limit)
    response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
    response.headers["X-RateLimit-Reset"] = str(reset_seconds)

    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "Retry-After": str(reset_seconds),
                "X-RateLimit-Limit": str(decision.limit),
                "X-RateLimit-Remaining": str(decision.remaining),
                "X-RateLimit-Reset": str(reset_seconds),
            },
        )

    return auth