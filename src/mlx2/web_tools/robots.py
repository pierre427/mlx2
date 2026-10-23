"""
Async robots.txt checker with per-domain caching.

Fetches /robots.txt for a domain once per process lifetime and caches the
parsed result.  Subsequent checks are in-memory only.

Design notes:
- We identify ourselves as Googlebot for the check because that's the agent
  most sites want to allow.  If we're blocked as Googlebot we're definitely
  blocked; if we're allowed as Googlebot we're very likely allowed.  Using a
  custom "mcp-web-tool" agent would cause most robots.txt files to fall back
  to the "*" catch-all, which is less informative.
- This is advisory only — the result is returned as a warning string that the
  LLM can decide to honour or override.  We never hard-block.
- robots.txt fetch uses a short timeout and fails open (allowed) so a missing
  or unreachable robots.txt never blocks legitimate fetches.
"""
from __future__ import annotations

import asyncio
import urllib.robotparser
from urllib.parse import urlparse

from .headers import HTTP_IMPERSONATE, navigation_headers
from .http_session import AsyncSession

# domain → (parser | None)
# None means we tried to fetch but got nothing — treat as "allow all"
_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}
_cache_locks: dict[str, asyncio.Lock] = {}

_CHECK_AGENTS = ["Googlebot", "Mozilla/5.0", "*"]
_FETCH_TIMEOUT = 8  # seconds


def _origin(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _robots_url(url: str) -> str:
    return _origin(url) + "/robots.txt"


def _get_lock(domain: str) -> asyncio.Lock:
    if domain not in _cache_locks:
        _cache_locks[domain] = asyncio.Lock()
    return _cache_locks[domain]


async def _fetch_robots(robots_url: str) -> str | None:
    """Fetch robots.txt with the same HTTP profile as page fetches."""
    try:
        from .network_guard import get
        r = await get(robots_url, headers=navigation_headers(robots_url), timeout=_FETCH_TIMEOUT)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None


def _parse_robots(text: str, robots_url: str) -> urllib.robotparser.RobotFileParser:
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    rp.parse(text.splitlines())
    return rp


async def _ensure_cached(url: str) -> urllib.robotparser.RobotFileParser | None:
    domain = urlparse(url).netloc.lower()
    if domain in _cache:
        return _cache[domain]

    async with _get_lock(domain):
        # Double-check after acquiring lock
        if domain in _cache:
            return _cache[domain]

        robots_url = _robots_url(url)
        text = await _fetch_robots(robots_url)
        if text:
            _cache[domain] = _parse_robots(text, robots_url)
        else:
            _cache[domain] = None  # fail open

    return _cache[domain]


async def check(url: str) -> tuple[bool, str]:
    """
    Check whether url is allowed by the domain's robots.txt.

    Returns:
        (allowed: bool, message: str)
        allowed=True  → no restriction found, proceed normally
        allowed=False → at least one agent rule disallows this path
        message       → human-readable explanation for the LLM
    """
    rp = await _ensure_cached(url)

    if rp is None:
        return True, "robots.txt unavailable (failed open)"

    for agent in _CHECK_AGENTS:
        if not rp.can_fetch(agent, url):
            return False, (
                f"robots.txt disallows '{url}' for agent '{agent}'. "
                "Proceeding anyway — you may decide to respect this restriction."
            )

    crawl_delay = rp.crawl_delay("*") or rp.crawl_delay("Googlebot")
    if crawl_delay and crawl_delay > 0:
        note = f" (robots.txt requests a {crawl_delay}s crawl-delay)"
    else:
        note = ""

    return True, f"robots.txt allows this URL{note}"
