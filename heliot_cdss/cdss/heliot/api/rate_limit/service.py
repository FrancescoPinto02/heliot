from __future__ import annotations

from abc import ABC, abstractmethod

from .models import RateLimitDecision, BaseRateLimitPolicy


class RateLimitBackendError(Exception):
    """Raised when the rate limit backend is unavailable or misconfigured."""


class RateLimiter(ABC):
    """
    Abstract rate limiter.

    Implementations are responsible for checking whether a request identified
    by `identifier` is allowed under the given policy, and for consuming one
    request if allowed.
    """

    @abstractmethod
    def check_and_consume(self, identifier: str, policy: BaseRateLimitPolicy) -> RateLimitDecision:
        """
        Check whether a request is allowed and consume one unit if allowed.

        Args:
            identifier: Stable identifier for the subject being rate-limited.
            policy: Rate limit policy to apply.

        Returns:
            A RateLimitDecision describing whether the request is allowed.
        """
        raise NotImplementedError