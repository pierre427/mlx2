"""General web search via DDGS plus news search via Google News RSS.

``web_search`` must return direct web results.  Routing it through Google News
RSS made domain-qualified queries (for example, ``site:jobs.lever.co``) return
opaque ``news.google.com/rss/articles/...`` redirects instead of the requested
sites.  That is especially harmful to workflows that validate a URL before
fetching it.  General search therefore uses the already-required ``ddgs``
package, while ``news_search`` retains the Google News RSS implementation.
"""
from __future__ import annotations

import asyncio
import html as _html
import json as _json
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET

from .http_session import AsyncSession
from ddgs.ddgs import DDGS
from ddgs.http_client import HttpClient
import primp

from .fetcher import fetch
from .headers import HTTP_IMPERSONATE, SEARCH_IMPERSONATE, SEARCH_OS

_GNEWS = "https://news.google.com/rss/search"
_HL, _GL, _CEID = "en-US", "US", "US:en"        # language / country / edition
_REFERER = "https://news.google.com/"


class StableDDGS(DDGS):
    """Keep search requests on one UA/TLS pair (DDGS 9.14.4 internals)."""

    def _get_engines(self, category, selected_backend):
        engines = super()._get_engines(category, selected_backend)
        for engine in engines:
            if not isinstance(engine.http_client, HttpClient):
                raise RuntimeError(f"Unsupported search transport: {engine.name}")
            client = primp.Client(
                proxy=self._proxy,
                timeout=self._timeout,
                impersonate=SEARCH_IMPERSONATE,
                impersonate_os=SEARCH_OS,
                verify=self._verify if isinstance(self._verify, bool) else True,
                ca_cert_file=self._verify if isinstance(self._verify, str) else None,
            )
            # Preserve engine context without replacing identity headers.
            headers = {k: v for k, v in engine.headers_update.items()
                       if k.lower() not in {"user-agent", "sec-ch-ua", "sec-ch-ua-platform"}}
            if headers:
                client.headers_update(headers)
            engine.http_client.client = client
        return engines

# RSS is un-protected; Tier 1 suffices. Flip to force the browser tiers.
FORCE_BROWSER = os.environ.get("WEB_TOOLS_FORCE_BROWSER", "").lower() in (
    "1", "true", "yes")


def _rss_url(query: str) -> str:
    return (f"{_GNEWS}?q={urllib.parse.quote(query)}"
            f"&hl={_HL}&gl={_GL}&ceid={_CEID}")


def _strip_tags(s: str) -> str:
    return _html.unescape(re.sub(r"<[^>]+>", " ", s or "")).strip()


def _parse_rss(xml_text: str, max_results: int) -> list[dict[str, str]]:
    # Tolerate a browser-tier response that wraps the feed: start at <rss>.
    i = xml_text.find("<rss")
    if i > 0:
        xml_text = xml_text[i:]
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out: list[dict[str, str]] = []
    for item in root.iter("item"):
        src = item.find("source")
        out.append({
            "title": (item.findtext("title") or "").strip(),
            "url": (item.findtext("link") or "").strip(),
            "snippet": _strip_tags(item.findtext("description") or "")
                       or (item.findtext("title") or "").strip(),
            "date": (item.findtext("pubDate") or "").strip(),
            "source": (src.text or "").strip() if src is not None else "",
        })
        if len(out) >= max_results:
            break
    return out


_GNEWS_ART = re.compile(r"news\.google\.com/(?:rss/)?articles/([^?/]+)")


async def resolve_url(url: str) -> str | None:
    """Decode a Google News articles redirect to the real article URL via
    Google's batchexecute API. Returns None for non-Google-News URLs or on
    failure (caller falls back to the original URL). Google no longer does a
    plain HTTP redirect on these — the real URL is only obtainable this way."""
    m = _GNEWS_ART.search(url)
    if not m:
        return None
    aid = m.group(1)
    try:
        async with AsyncSession(impersonate=HTTP_IMPERSONATE) as s:
            r = await s.get(url, timeout=15)
            sig = re.search(r'data-n-a-sg="([^"]+)"', r.text)
            ts = re.search(r'data-n-a-ts="([^"]+)"', r.text)
            if not (sig and ts):
                return None
            inner = _json.dumps([
                "garturlreq",
                [["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1,
                  None, None, None, None, None, 0, 1],
                 "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0],
                aid, ts.group(1), sig.group(1)])
            data = "f.req=" + urllib.parse.quote(
                _json.dumps([[["Fbv4je", inner]]]))
            r2 = await s.post(
                "https://news.google.com/_/DotsSplashUi/data/batchexecute",
                data=data,
                headers={"content-type":
                         "application/x-www-form-urlencoded"}, timeout=15)
            seg = r2.text.split("garturlres")[-1] if "garturlres" in r2.text \
                else ""
            mm = re.search(r'(https?://[^"\\]+)', seg)
            return mm.group(1) if mm else None
    except Exception:
        return None


async def _gnews(query: str, max_results: int) -> list[dict[str, str]]:
    res = await fetch(_rss_url(query), referer=_REFERER,
                      force_browser=FORCE_BROWSER)
    return _parse_rss(res.html or "", max_results)


def _ddgs_text(
    query: str,
    max_results: int,
    region: str,
    timelimit: str | None = None,
    backend: str = "auto",
) -> list[dict[str, str]]:
    """Run blocking DDGS text search and normalize its result fields."""
    search_options = {
        "region": region,
        "safesearch": "moderate",
        "max_results": max_results,
        "backend": backend,
    }
    if timelimit:
        search_options["timelimit"] = timelimit
    hits = StableDDGS(timeout=10).text(query, **search_options)
    out: list[dict[str, str]] = []
    for hit in hits[:max_results]:
        url = str(hit.get("href") or hit.get("url") or "").strip()
        if not url:
            continue
        host = urllib.parse.urlparse(url).hostname or ""
        out.append({
            "title": str(hit.get("title") or "").strip(),
            "url": url,
            "snippet": str(hit.get("body") or hit.get("description") or "").strip(),
            "date": str(hit.get("date") or "").strip(),
            "source": str(hit.get("source") or host).strip(),
        })
    return out


# DDGS 'auto' includes engines that override the User-Agent independently or
# use a separate HTTP stack. Keep automatic search to the primp-backed engines
# we can bind to the same stable browser and OS profile.
_FALLBACK_BACKENDS = [
    "wikipedia", "grokipedia", "yahoo", "yandex", "startpage", "brave", "mojeek",
]


def _norm_url(url: str) -> str:
    """Normalise a result URL for cross-engine dedup. Ignores scheme (http vs
    https), host case, leading www, and trailing slash so the same page from
    two engines collapses to one hit."""
    try:
        p = urllib.parse.urlsplit(url)
        host = (p.netloc or "").lower().removeprefix("www.")
        path = p.path.rstrip("/") or "/"
        return urllib.parse.urlunsplit(("", host, path, p.query, ""))
    except Exception:
        return url.strip().lower().rstrip("/")


def _ddgs_multi(
    query: str,
    max_results: int,
    region: str,
    timelimit: str | None,
    backends: list[str],
) -> list[dict[str, str]]:
    """Query backends in order, merging deduped results, stopping once we have
    enough. Per-backend failures are swallowed so one dead engine can't sink
    the search."""
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    for backend in backends:
        if len(merged) >= max_results:
            break
        try:
            hits = _ddgs_text(query, max_results, region, timelimit, backend)
        except Exception:
            continue
        for hit in hits:
            key = _norm_url(hit["url"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
            if len(merged) >= max_results:
                break
    return merged[:max_results]


async def web_search(
    query: str,
    max_results: int = 10,
    region: str = "us-en",
    timelimit: str | None = None,
    backend: str = "auto",
) -> list[dict[str, str]]:
    """Search the general web. Returns direct result URLs.

    For backend='auto' (the default) this uses a bounded chain of engines with
    the same browser identity. An explicit backend must use that chain."""
    if backend and backend != "auto":
        if backend not in _FALLBACK_BACKENDS:
            raise ValueError(f"Unsupported backend for stable HTTP identity: {backend}")
        chain = [backend]
    else:
        chain = _FALLBACK_BACKENDS
    return await asyncio.wait_for(
        asyncio.to_thread(
            _ddgs_multi, query, max_results, region, timelimit, chain),
        timeout=45,
    )


async def news_search(
    query: str,
    max_results: int = 10,
    region: str = "us-en",
) -> list[dict[str, str]]:
    """Recent news via Google News (favours the last week). Returns
    [{title, url, snippet, date, source}]."""
    return await _gnews(f"{query} when:7d", max_results)
