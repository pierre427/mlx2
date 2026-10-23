"""
Per-domain async rate limiter.
Enforces a minimum gap between requests to the same domain with random jitter,
so we never appear to be hammering a site.
"""
import asyncio
import random
import time
from urllib.parse import urlparse


class DomainRateLimiter:
    def __init__(self, min_delay: float = 2.0, jitter: float = 2.0):
        # min_delay: minimum seconds between requests to same domain
        # jitter: max additional random seconds on top of min_delay
        self._min_delay = min_delay
        self._jitter = jitter
        self._last_request: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _domain(self, url: str) -> str:
        parsed = urlparse(url)
        # Treat www.example.com and example.com as the same domain
        host = parsed.netloc.lower().removeprefix("www.")
        return host

    def _get_lock(self, domain: str) -> asyncio.Lock:
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()
        return self._locks[domain]

    async def acquire(self, url: str) -> None:
        domain = self._domain(url)
        lock = self._get_lock(domain)
        async with lock:
            last = self._last_request.get(domain, 0.0)
            wait = self._min_delay + random.uniform(0, self._jitter) - (time.monotonic() - last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request[domain] = time.monotonic()


# Global singleton — shared across all fetcher calls
_limiter = DomainRateLimiter(min_delay=2.0, jitter=2.5)


async def throttle(url: str) -> None:
    await _limiter.acquire(url)
