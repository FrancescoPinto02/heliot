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
class TokenBucketRateLimitPolicy(BaseRateLimitPolicy):
    capacity: int
    refill_rate_per_second: float

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if self.refill_rate_per_second <= 0:
            raise ValueError("refill_rate_per_second must be > 0")

@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """
    Result of a rate limit check.

    Attributes:
        allowed: Whether the request is allowed.
        remaining: Number of requests remaining.
        retry_after_seconds: Seconds until the client can retry, if denied.
        limit: Applied limit.
    """
    allowed: bool
    remaining: int
    retry_after_seconds: int | None
    limit: int