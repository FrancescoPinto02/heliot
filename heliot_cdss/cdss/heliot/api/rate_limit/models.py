from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


class BaseRateLimitPolicy(ABC):
    """Marker base class for rate limit policies."""
    pass

@dataclass(frozen=True, slots=True)
class FixedWindowRateLimitPolicy(BaseRateLimitPolicy):
    """
    Fixed Window rate limit policy.

    Attributes:
        limit: Maximum number of requests allowed in the time window.
        window_seconds: Duration of the time window in seconds.
    """
    limit: int
    window_seconds: int

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError("limit must be > 0")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """
    Result of a rate limit check.

    Attributes:
        allowed: Whether the request is allowed.
        remaining: Number of requests remaining in the current window.
        retry_after_seconds: Seconds until the client can retry, if denied.
        limit: Applied limit.
        window_seconds: Applied time window.
    """
    allowed: bool
    remaining: int
    retry_after_seconds: int | None
    limit: int
    window_seconds: int