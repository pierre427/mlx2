"""Public-web fetcher with pinned HTTP authority.

Page, robots and session-warming requests share the public-address transport.
Legacy browser tiers remain internal but fetch() refuses escalation because
browser subresources cannot yet enforce equivalent address pinning.
"""
from __future__ import annotations

import asyncio
import email.utils
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

from .http_session import AsyncSession, CURL_CFFI_AVAILABLE as _CURL_CFFI_AVAILABLE
from .network_guard import get as public_get, resolve

from .headers import HTTP_IMPERSONATE, navigation_headers
from .rate_limiter import throttle

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

import os as _os, sys as _sys
# Data dir for browser profiles. The service wrapper sets MCP_WEB_TOOL_DATA to
# a writable path; the DEFAULT must also be writable — /var/db needs root and
# silently breaks the browser tiers when the env var is missing.
if _sys.platform == "win32":
    _APP_DATA = Path(_os.environ.get("APPDATA", str(Path.home() / "AppData/Roaming"))) / "mcp_web_tool"
else:
    _APP_DATA = Path(_os.environ.get(
        "MLX2_WEB_DATA", str(Path.home() / "Library/Application Support/MLX2/web")))
CAMOUFOX_PROFILE_DIR = str(_APP_DATA / "camoufox_profile")
NODRIVER_PROFILE_DIR = str(_APP_DATA / "nodriver_profile")

# Persisted per-domain routing decisions. Circuit-breaker escalations
# (force_browser / force_nodriver) and Akamai detection are learned the hard
# way — by paying failed tier-1/tier-2 attempts. Without persistence every
# process restart re-pays that cost on known-hard domains. We store the learned
# routing per domain and re-seed new _DomainState objects from it, with a TTL so
# a domain that has since relaxed its protection gets re-probed cold (the
# "RecheckCold" behaviour) rather than escalated forever.
import json as _json
_ROUTING_STATE_PATH = _APP_DATA / "routing_state.json"
_ROUTING_TTL = 7 * 24 * 3600.0     # re-probe a domain cold after a week
_persisted_routing: dict[str, dict] = {}


def _load_routing_state() -> None:
    """Load persisted routing decisions, dropping entries past their TTL."""
    global _persisted_routing
    try:
        with open(_ROUTING_STATE_PATH, encoding="utf-8") as fh:
            raw = _json.load(fh)
    except (OSError, ValueError):
        return
    now = time.time()
    _persisted_routing = {
        dom: rec for dom, rec in raw.items()
        if isinstance(rec, dict) and (now - rec.get("saved_at", 0)) < _ROUTING_TTL
    }


def _save_routing_state() -> None:
    """Atomically write the routing table to disk (best-effort)."""
    try:
        _ROUTING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _ROUTING_STATE_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            _json.dump(_persisted_routing, fh)
        _os.replace(tmp, _ROUTING_STATE_PATH)
    except OSError:
        pass


def _persist_domain(key: str, st: "_DomainState") -> None:
    """Record a domain's learned routing to disk. Only called when a circuit
    trips or Akamai is detected, so writes are rare."""
    if not (st.force_browser or st.force_nodriver or st.is_akamai):
        return
    _persisted_routing[key] = {
        "force_browser": st.force_browser,
        "force_nodriver": st.force_nodriver,
        "is_akamai": st.is_akamai,
        "saved_at": time.time(),
    }
    _save_routing_state()


def _relax_persisted_domain(key: str, st: "_DomainState") -> None:
    """A solve-and-bounce restored the fast HTTP path for this domain — forget
    the forced browser escalation. The Akamai warm-up flag is cheap and still
    useful, so keep it if set; otherwise drop the domain entirely."""
    if st.is_akamai:
        _persisted_routing[key] = {
            "force_browser": False, "force_nodriver": False,
            "is_akamai": True, "saved_at": time.time(),
        }
    elif key in _persisted_routing:
        _persisted_routing.pop(key, None)
    else:
        return
    _save_routing_state()

_BOT_STATUSES = {429, 503}
_BOT_INDICATORS = [
    "cf-challenge",
    "cf_chl",
    "bot detected",
    "unusual traffic",
    "verify you are human",
    "verifying you are human",
    "ddos-guard",
    "enable javascript to continue",
    "please enable javascript",
    "your request has been blocked",
    "access to this page has been blocked",
    "this page is protected",
    "ray id",                    # Cloudflare error page signature
    "incident id",               # Akamai block page signature
]

# "captcha" needs care: the bare word appears in ordinary prose (e.g. the
# Wikipedia article on CAPTCHAs, or any page discussing bot mitigation), so a
# substring match on it alone false-positives a healthy 200 as a bot wall.
# Fire only on genuine challenge markers:
#   1. Vendor-specific widget/script tokens that appear only on a real rendered
#      challenge page, never in prose. Note the tokens are deliberately the
#      widget class / JS-global / script-URL forms, NOT the bare product names:
#      MediaWiki (Wikipedia) ships a config blob like
#      "wgConfirmEditCaptchaNeededForGenericEdit":"hcaptcha" in <head>, so the
#      bare words "hcaptcha"/"recaptcha" would false-positive every wiki page.
_CAPTCHA_VENDOR_INDICATORS = [
    "g-recaptcha",       # Google reCAPTCHA widget CSS class
    "grecaptcha",        # Google reCAPTCHA JS global
    "recaptcha/api.js",  # Google reCAPTCHA script URL
    "h-captcha",         # hCaptcha widget CSS class
    "js.hcaptcha.com",   # hCaptcha script host
    "cf-turnstile",      # Cloudflare Turnstile widget
    "px-captcha",        # PerimeterX
    "arkoselabs",        # Arkose Labs / FunCaptcha
    "funcaptcha",
]
#   2. The bare word "captcha", but only inside an imperative wall phrase. The
#      context words must be ones that do NOT appear in prose *about* captchas:
#      an encyclopedia entry says "a CAPTCHA is a challenge-response test used to
#      tell humans from robots", so proximity to "challenge"/"human"/"robot" is
#      NOT enough. A real wall instructs the visitor to act on the captcha —
#      "complete the captcha", "captcha to continue" — which prose never does.
_CAPTCHA_CONTEXT_RE = re.compile(
    r"(?:complete|solve|enter|type|fill\s+in|pass|verify|finish)\s+"
    r"(?:the\s+|this\s+|a\s+)?captcha"
    r"|captcha\s+(?:to\s+(?:continue|proceed|access|verify)"
    r"|below|verification\s+required|is\s+required|challenge)",
    re.IGNORECASE | re.DOTALL,
)
_CF_CHALLENGE_INDICATORS = [
    "just a moment", "checking your browser", "cf-browser-verification",
    "cf-turnstile",  # Cloudflare Turnstile managed challenge widget
]

# Akamai Bot Manager detection signals
_AKAMAI_RESPONSE_HEADERS = {"x-akamai-transformed", "x-akamai-request-id", "akamai-grn", "x-check-cacheable"}
_AKAMAI_BODY_SIGNALS = ["/_csi/", "/akam/", "ak.js", "akamai-js"]

_BACKOFF_BASE = 5.0
_BACKOFF_MAX = 120.0
_CIRCUIT_TRIP_THRESHOLD = 2

# Persistent Camoufox browser — kept alive across requests so the browser has
# warm TCP connections, a populated resource cache, and a visit history, all of
# which distinguish a real session from a cold-start bot launch.
_camoufox_browser = None   # BrowserContext returned by AsyncCamoufox.__aenter__
_camoufox_ctx = None       # AsyncCamoufox instance held open (never __aexit__'d)
_camoufox_init_lock = asyncio.Lock()

# Injected into nodriver (headless Chrome) before every page load to patch the
# signals that trivially distinguish headless Chrome from a real browser:
#   - navigator.connection.rtt = 0  (always 0 in headless, impossible on real connections)
#   - window.outerWidth/Height = 0  (headless default; real browsers = window chrome + viewport)
#   - navigator.deviceMemory        (Chrome caps real value at 8; containers often report 2)
#   - navigator.hardwareConcurrency (typical desktop is 8; containers report 2 or 4)
#   - window.chrome                 (absent in headless by default, present in headed Chrome)
_NODRIVER_STEALTH_SCRIPT = """
(function () {
    const def = (obj, prop, val) => {
        try { Object.defineProperty(obj, prop, {get: () => val, configurable: true}); }
        catch (_) {}
    };
    def(navigator, 'webdriver', undefined);
    def(navigator, 'deviceMemory', 8);
    def(navigator, 'hardwareConcurrency', 8);
    if (navigator.connection) {
        def(navigator.connection, 'rtt', Math.floor(Math.random() * 62) + 23);
        def(navigator.connection, 'downlink', 10);
        def(navigator.connection, 'effectiveType', '4g');
    }
    def(window, 'outerWidth',  1280);
    def(window, 'outerHeight', 800);
    if (!window.chrome) {
        window.chrome = {
            app: {isInstalled: false},
            runtime: {},
            loadTimes: function() { return {}; },
            csi: function() { return {}; },
        };
    }
    // Notifications API — DataDome queries Notification.permission on every load
    if (window.Notification) {
        try {
            Object.defineProperty(window.Notification, 'permission', {
                get: () => 'default', configurable: true
            });
        } catch (_) {}
    }
    // navigator.permissions.query — return realistic states for common probes
    if (navigator.permissions && navigator.permissions.query) {
        const _origQuery = navigator.permissions.query.bind(navigator.permissions);
        navigator.permissions.query = (desc) => {
            if (desc && desc.name === 'notifications') {
                return Promise.resolve({ state: 'prompt', onchange: null });
            }
            return _origQuery(desc);
        };
    }
    // Screen color depth (headless defaults to 24 but some builds report lower)
    def(screen, 'colorDepth', 24);
    def(screen, 'pixelDepth', 24);
})();
"""

# Injected into Camoufox pages via add_init_script.  Camoufox's C++ layer
# already handles canvas/WebGL/navigator/screen, so we only need JS-layer
# patches for APIs that C++ can't intercept.
_CAMOUFOX_EXTRA_SCRIPT = """
(function () {
    if (window.Notification) {
        try {
            Object.defineProperty(window.Notification, 'permission', {
                get: () => 'default', configurable: true
            });
        } catch (_) {}
    }
    if (navigator.permissions && navigator.permissions.query) {
        const _orig = navigator.permissions.query.bind(navigator.permissions);
        navigator.permissions.query = (desc) =>
            (desc && desc.name === 'notifications')
                ? Promise.resolve({ state: 'prompt', onchange: null })
                : _orig(desc);
    }
})();
"""

# JavaScript cookie-consent banner dismissal.  Tries exact-text matches first
# (safest), then falls back to contains-match.  Runs in both browser tiers.
_COOKIE_ACCEPT_JS = """
(function () {
    const exact = ['accept all', 'accept all cookies', 'allow all', 'allow all cookies',
                   'agree', 'i agree', 'got it'];
    const contains = ['accept all', 'allow all', 'accept cookies'];
    const isVisible = el => {
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0 && window.getComputedStyle(el).visibility !== 'hidden';
    };
    const candidates = [...document.querySelectorAll(
        'button, a[role="button"], input[type="button"], input[type="submit"]'
    )];
    for (const el of candidates) {
        const t = el.textContent.trim().toLowerCase();
        if (exact.includes(t) && isVisible(el)) { el.click(); return 'exact:' + t; }
    }
    for (const el of candidates) {
        const t = el.textContent.trim().toLowerCase();
        if (contains.some(p => t.includes(p)) && isVisible(el)) { el.click(); return 'contains:' + t; }
    }
    return null;
})()
"""


# ---------------------------------------------------------------------------
# Per-domain state
# ---------------------------------------------------------------------------

@dataclass
class _DomainState:
    session: AsyncSession = field(
        default_factory=lambda: AsyncSession(impersonate=HTTP_IMPERSONATE)
    )
    warmed: bool = False
    failures: int = 0            # Tier 1 consecutive bot-wall hits
    force_browser: bool = False  # Tier 1 circuit tripped → skip to Tier 2
    browser_failures: int = 0    # Tier 2 consecutive bot-wall hits
    force_nodriver: bool = False  # Tier 2 circuit tripped → skip to Tier 3
    backoff_until: float = 0.0
    is_akamai: bool = False  # Set after homepage warm detects Akamai signals
    # Solve-and-bounce: set True after a browser tier successfully clears a wall
    # and its clearance cookies are harvested into st.session. Arms one tier-1
    # re-probe with those cookies on the next fetch, to drop back to fast HTTP.
    bounce_pending: bool = False
    # Serialises concurrent fetches to this domain so tier-escalation counters
    # are never read-modify-written by two coroutines at once.
    fetch_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_states: dict[str, _DomainState] = {}
_load_routing_state()


def _domain_key(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _origin(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _state(url: str) -> _DomainState:
    key = _domain_key(url)
    if key not in _states:
        st = _DomainState()
        # Re-seed from disk so a restart skips the tiers this domain has
        # already forced us into (and remembers Akamai's warm-up dance).
        rec = _persisted_routing.get(key)
        if rec:
            st.force_browser = bool(rec.get("force_browser"))
            st.force_nodriver = bool(rec.get("force_nodriver"))
            st.is_akamai = bool(rec.get("is_akamai"))
        _states[key] = st
    return _states[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FetchResult(NamedTuple):
    url: str
    status: int
    html: str
    via_browser: bool
    content_type: str = ""       # response MIME type, when known
    raw: bytes = b""             # raw body bytes (only populated for PDFs)


async def fetch(
    url: str,
    *,
    referer: str | None = None,
    force_browser: bool = False,
    allow_browser: bool = True,
    timeout: int = 20,
) -> FetchResult:
    """Fetch a public URL using the guarded HTTP transport."""
    await asyncio.to_thread(resolve, url)
    # Browser engines make unpinned subresource requests. Until they provide
    # equivalent egress enforcement this public-web entry point fails closed.
    if force_browser:
        raise ValueError("Browser fetch is unavailable under the public-address guard")
    allow_browser = False
    st = _state(url)
    backoff_remaining = st.backoff_until - time.monotonic()
    if backoff_remaining > 0:
        await asyncio.sleep(backoff_remaining)

    await throttle(url)

    # fetch_lock serialises concurrent calls to the same domain so that
    # tier-escalation counters and circuit-breaker flags are never updated
    # by two coroutines simultaneously.
    async with st.fetch_lock:
        # Some bounded batch workflows prefer a fast negative result over a
        # browser escalation. Always make one lightweight attempt, regardless
        # of the domain circuit-breaker state, and return immediately.
        if not allow_browser:
            result = await _fetch_lightweight(url, referer=referer, timeout=timeout)
            return result or FetchResult(url, 0, "", False)

        key = _domain_key(url)

        # --- Tier 1: curl_cffi ---
        # Run tier-1 either normally (domain not forced to a browser) or as a
        # solve-and-bounce re-probe: a browser tier previously cleared this
        # domain's wall and its clearance cookies are now in st.session, so we
        # try fast HTTP once to see if it slips through. Note: cf_clearance is
        # bound to the solving browser's UA/TLS fingerprint — a Firefox (Camoufox)
        # solve won't validate on our Chrome-impersonating tier-1, so for strict
        # Cloudflare the re-probe simply fails and we fall back (one cheap
        # attempt). It succeeds for cookie/consent/DataDome-soft walls and for
        # Chrome-solved (nodriver) Cloudflare.
        normal_tier1 = not (force_browser or st.force_browser)
        bounce_tier1 = st.bounce_pending and not force_browser and not normal_tier1
        if normal_tier1 or bounce_tier1:
            st.bounce_pending = False
            result = await _fetch_lightweight(url, referer=referer, timeout=timeout)
            if result and _tier1_succeeded(result):
                st.failures = 0
                if bounce_tier1:                 # fast path restored — relax routing
                    st.force_browser = False
                    st.force_nodriver = False
                    _relax_persisted_domain(key, st)
                return result
            if normal_tier1:                     # only the normal path trips the circuit
                st.failures += 1
                if result and result.status == 429:
                    _apply_backoff(st)
                if st.failures >= _CIRCUIT_TRIP_THRESHOLD:
                    st.force_browser = True
                    _persist_domain(key, st)
            # a failed bounce just falls through to the browser tiers below

        # --- Tier 2: Camoufox ---
        if not st.force_nodriver:
            t2 = await _fetch_with_camoufox(url, referer=referer, timeout=timeout + 10)
            if t2.html and not _looks_like_bot_wall(t2):
                st.failures = 0
                st.browser_failures = 0
                st.bounce_pending = True        # cookies harvested — try fast path next time
                return t2
            st.browser_failures += 1
            if st.browser_failures >= _CIRCUIT_TRIP_THRESHOLD:
                st.force_nodriver = True
                _persist_domain(key, st)

        # --- Tier 3: nodriver ---
        t3 = await _fetch_with_nodriver(url, timeout=timeout + 20)
        if t3.html and not _looks_like_bot_wall(t3):
            st.failures = 0
            st.browser_failures = 0
            st.bounce_pending = True            # Chrome-solved cookies bounce cleanly to tier-1
        return t3


# ---------------------------------------------------------------------------
# Tier 1 — curl_cffi
# ---------------------------------------------------------------------------

def _looks_like_akamai(response) -> bool:
    """Heuristic: does this response look like it came from an Akamai-protected origin?"""
    headers_lower = {k.lower() for k in response.headers}
    if headers_lower & _AKAMAI_RESPONSE_HEADERS:
        return True
    body_sample = (response.text or "")[:5000].lower()
    return any(sig in body_sample for sig in _AKAMAI_BODY_SIGNALS)


async def _warm_akamai_extra(st: _DomainState, origin: str, homepage_html: str) -> None:
    """Visit one interior page to give Akamai's sensor script time to accumulate
    trust data before we hit the protected target URL."""
    from bs4 import BeautifulSoup
    interior_url = None
    try:
        soup = BeautifulSoup(homepage_html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if (href.startswith("/") and len(href) > 2
                    and "?" not in href and "#" not in href
                    and not href.startswith("//")):
                interior_url = origin + href
                break
    except Exception:
        pass
    if not interior_url:
        return
    await asyncio.sleep(random.uniform(2.0, 4.0))
    try:
        headers = navigation_headers(interior_url, referer=origin + "/")
        r = await public_get(
            interior_url, headers=headers, timeout=12,
            allow_redirects=True, max_redirects=5,
        )
        await asyncio.sleep(random.uniform(1.5, 3.0))
    except Exception:
        pass


async def _warm_session(st: _DomainState, url: str) -> None:
    """Visit the homepage once to build a cookie jar; detect and handle Akamai."""
    # Set warmed immediately — before any await — so concurrent callers that
    # check st.warmed before we return don't trigger a second warming request.
    st.warmed = True
    origin = _origin(url)
    homepage = origin + "/"
    if homepage.rstrip("/") == url.rstrip("/"):
        return
    headers = navigation_headers(homepage)
    try:
        r = await public_get(
            homepage,
            headers=headers,
            timeout=12,
            allow_redirects=True,
            max_redirects=5,
        )
        await asyncio.sleep(random.uniform(1.2, 2.8))

        # Akamai Bot Manager builds per-session trust across multiple page
        # visits.  If Akamai is detected, visit one interior page so the sensor
        # data collection script can accumulate a baseline before we hit the
        # protected target URL.
        if _looks_like_akamai(r):
            st.is_akamai = True
            _persist_domain(_domain_key(url), st)
            await _warm_akamai_extra(st, origin, r.text)
    except Exception:
        pass


def _parse_retry_after(header_value: str) -> float | None:
    """
    Parse a Retry-After header value.
    Accepts both integer-seconds ("60") and HTTP-date formats.
    Returns seconds to wait, or None if unparseable.
    """
    val = header_value.strip()
    # Integer seconds
    try:
        seconds = float(val)
        return max(0.0, min(seconds, _BACKOFF_MAX))
    except ValueError:
        pass
    # HTTP-date
    try:
        retry_time = email.utils.parsedate_to_datetime(val).timestamp()
        wait = retry_time - time.time()
        return max(0.0, min(wait, _BACKOFF_MAX))
    except Exception:
        return None


def _apply_backoff(st: _DomainState) -> None:
    # Only apply exponential backoff if Retry-After didn't already set a future timestamp.
    # _fetch_lightweight sets st.backoff_until from the Retry-After header when present;
    # calling this afterward would overwrite that server-specified value.
    if st.backoff_until > time.monotonic():
        return
    delay = min(_BACKOFF_BASE * (2 ** st.failures) + random.uniform(0, 3), _BACKOFF_MAX)
    st.backoff_until = time.monotonic() + delay


async def _fetch_lightweight(
    url: str,
    referer: str | None = None,
    timeout: int = 20,
) -> FetchResult | None:
    st = _state(url)

    if not st.warmed:
        await _warm_session(st, url)

    headers = navigation_headers(url, referer=referer)
    try:
        r = await public_get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
            max_redirects=10,
        )

        # Respect Retry-After when present on any rate-limit response
        if r.status_code == 429:
            retry_after = r.headers.get("retry-after", "")
            if retry_after:
                wait = _parse_retry_after(retry_after)
                if wait:
                    st.backoff_until = time.monotonic() + wait

        await asyncio.sleep(random.uniform(0.3, 0.9))

        # PDF detection: serve the raw bytes for downstream text extraction
        # instead of decoding a binary body as text (which yields garbage).
        ctype = (r.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        body = r.content or b""
        if ctype == "application/pdf" or body[:5] == b"%PDF-":
            return FetchResult(
                url=str(r.url),
                status=r.status_code,
                html="",
                via_browser=False,
                content_type="application/pdf",
                raw=bytes(body),
            )
        return FetchResult(
            url=str(r.url),
            status=r.status_code,
            html=r.text,
            via_browser=False,
            content_type=ctype,
        )
    except Exception:
        return None


def _tier1_succeeded(result: FetchResult) -> bool:
    """A tier-1 result is usable when it carries content (HTML or a decoded
    PDF) and doesn't look like a bot wall."""
    has_content = bool(result.html) or (
        result.content_type == "application/pdf" and bool(result.raw))
    return has_content and not _looks_like_bot_wall(result)


def _looks_like_bot_wall(result: FetchResult) -> bool:
    if result.status in _BOT_STATUSES:
        return True
    # Only scan the first 8 KB — bot-wall indicators appear in <head> or the
    # top of <body>.  Scanning a full 1–2 MB page causes false positives when
    # legitimate content happens to contain these substrings (e.g. "ray id" in
    # a product attribute, "your request has been blocked" in a help article).
    html_lower = result.html[:8192].lower()
    if any(ind in html_lower for ind in _BOT_INDICATORS):
        return True
    if any(ind in html_lower for ind in _CAPTCHA_VENDOR_INDICATORS):
        return True
    # Bare "captcha" only counts as a wall when challenge context sits nearby.
    if "captcha" in html_lower and _CAPTCHA_CONTEXT_RE.search(html_lower):
        return True
    return False


_BODY_OPEN_RE = re.compile(r"<body[^>]*>", re.IGNORECASE)
_TAG_RE       = re.compile(r"<[^>]+>")

def _body_text_length(html: str) -> int:
    """Word count of visible text in the first 10 KB of <body> content.
    Returns 0 for JS-shell pages (nearly empty body) so callers know to
    re-harvest after hydration.  Finding only the opening tag avoids the
    false-negative that occurs when </body> is outside the search window
    (i.e. on any real page whose body exceeds ~145 KB)."""
    m = _BODY_OPEN_RE.search(html[:30_000])
    if not m:
        return 0
    body_start = html[m.end():m.end() + 10_000]
    return len(_TAG_RE.sub(" ", body_start).split())


# ---------------------------------------------------------------------------
# Shared browser-tier helpers
# ---------------------------------------------------------------------------

def _inject_cookies(st: _DomainState, cookies: list[dict]) -> int:
    """Copy browser-harvested cookies into the domain's curl_cffi session so a
    later tier-1 request presents the clearance/session cookies the browser
    earned. Returns the number injected."""
    injected = 0
    for c in cookies:
        try:
            name = c.get("name")
            value = c.get("value")
            if not name or value is None:
                continue
            domain = (c.get("domain") or "").lstrip(".")
            path = c.get("path") or "/"
            if hasattr(st.session, "cookies"):
                st.session.cookies.set(name, str(value), domain=domain, path=path)
            else:
                scheme = "https"
                st.session.set_cookies(f"{scheme}://{domain}{path}", {name: str(value)})
            injected += 1
        except Exception:
            pass
    return injected


async def _harvest_camoufox_cookies(page, st: _DomainState) -> None:
    """Pull cookies from the Camoufox page's context into st.session."""
    try:
        cookies = await page.context.cookies()
    except Exception:
        return
    _inject_cookies(st, [dict(c) for c in (cookies or [])])


async def _harvest_nodriver_cookies(browser, st: _DomainState) -> None:
    """Pull cookies from the nodriver (Chrome) browser into st.session."""
    try:
        jar = await browser.cookies.get_all()
    except Exception:
        return
    norm = []
    for c in (jar or []):
        try:
            norm.append({
                "name": getattr(c, "name", None),
                "value": getattr(c, "value", None),
                "domain": getattr(c, "domain", "") or "",
                "path": getattr(c, "path", "/") or "/",
            })
        except Exception:
            pass
    _inject_cookies(st, norm)


def _bezier_curve(
    start: tuple[int, int],
    end: tuple[int, int],
    num_points: int = 30,
) -> list[tuple[int, int]]:
    """Cubic Bézier interpolation for natural mouse trajectories."""
    sx, sy = start
    ex, ey = end
    # Control points biased along the path with human-like lateral deviation
    cp1x = sx + (ex - sx) * random.uniform(0.2, 0.4) + random.randint(-80, 80)
    cp1y = sy + (ey - sy) * random.uniform(0.1, 0.3) + random.randint(-60, 60)
    cp2x = sx + (ex - sx) * random.uniform(0.6, 0.8) + random.randint(-80, 80)
    cp2y = sy + (ey - sy) * random.uniform(0.7, 0.9) + random.randint(-60, 60)
    points = []
    for i in range(num_points):
        t = i / max(num_points - 1, 1)
        u = 1 - t
        x = int(u**3*sx + 3*u**2*t*cp1x + 3*u*t**2*cp2x + t**3*ex)
        y = int(u**3*sy + 3*u**2*t*cp1y + 3*u*t**2*cp2y + t**3*ey)
        points.append((x, y))
    return points


async def _human_mouse_move(page) -> None:
    """2–4 Bézier-curve mouse sweeps across the viewport."""
    try:
        vs = page.viewport_size or {"width": 1280, "height": 800}
        w, h = vs["width"], vs["height"]
        cx = random.randint(w // 3, w * 2 // 3)
        cy = random.randint(h // 3, h * 2 // 3)
        for _ in range(random.randint(2, 4)):
            tx = random.randint(40, w - 40)
            ty = random.randint(40, h - 40)
            for px, py in _bezier_curve((cx, cy), (tx, ty), num_points=random.randint(20, 35)):
                await page.mouse.move(px, py)
                await asyncio.sleep(random.uniform(0.006, 0.022))
            cx, cy = tx, ty
            await asyncio.sleep(random.uniform(0.12, 0.45))
    except Exception:
        pass


async def _dismiss_cookie_banners(page) -> None:
    """
    Attempt to click the 'accept all' button on cookie/consent overlays.
    Uses CSS selector probes first (fast, no false-positives), then the
    shared JS approach.  Failures are silently ignored — a missed banner
    only affects content quality, never detection.
    """
    ACCEPT_SELECTORS = [
        "button[id*='accept' i]:not([id*='reject' i])",
        "button[class*='accept' i]:not([class*='reject' i])",
        "button[id*='agree' i]",
        "button[id*='allow-all' i]",
        "[aria-label*='Accept all' i]",
        "[data-testid*='accept' i]",
    ]
    try:
        for sel in ACCEPT_SELECTORS:
            try:
                btn = page.locator(sel).first
                if await btn.is_visible(timeout=800):
                    await btn.click(timeout=1500)
                    await asyncio.sleep(random.uniform(0.3, 0.7))
                    return
            except Exception:
                pass
        await page.evaluate(_COOKIE_ACCEPT_JS)
        await asyncio.sleep(random.uniform(0.2, 0.5))
    except Exception:
        pass


async def _human_scroll(page) -> None:
    """
    Simulate a short human scroll session.
    Fires scroll events that DataDome/PerimeterX watch for, without
    scrolling so far that lazy-loaded content shifts the DOM mid-harvest.
    """
    try:
        scroll_height = await page.evaluate("document.body.scrollHeight")
        viewport_height = await page.evaluate("window.innerHeight")
        if not scroll_height or not viewport_height or scroll_height <= viewport_height:
            return

        target = min(int(scroll_height * 0.55), 2800)
        pos = 0
        while pos < target:
            step = random.randint(180, 520)
            pos = min(pos + step, target)
            await page.evaluate(f"window.scrollBy(0, {step})")
            await asyncio.sleep(random.uniform(0.08, 0.35))

        # Brief pause at scroll position, then back to top
        await asyncio.sleep(random.uniform(0.4, 0.9))
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(random.uniform(0.15, 0.35))
    except Exception:
        pass


async def _nodriver_scroll(tab) -> None:
    """Scroll simulation for nodriver via tab.evaluate (no Playwright dependency)."""
    try:
        scroll_height = int(await tab.evaluate("document.body.scrollHeight") or 0)
        viewport_height = int(await tab.evaluate("window.innerHeight") or 0)
        if scroll_height <= viewport_height:
            return
        target = min(int(scroll_height * 0.55), 2800)
        pos = 0
        while pos < target:
            step = random.randint(180, 520)
            pos = min(pos + step, target)
            await tab.evaluate(f"window.scrollBy(0, {step})")
            await asyncio.sleep(random.uniform(0.08, 0.35))
        await asyncio.sleep(random.uniform(0.4, 0.9))
        await tab.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(random.uniform(0.15, 0.35))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tier 2 — Camoufox (C++-level Firefox fingerprint patches)
# ---------------------------------------------------------------------------

async def _ensure_camoufox():
    """Return the persistent T2 browser.

    Preference order:
    1. Camoufox — C++-patched Firefox (best anti-detection, Linux/macOS/Windows only)
    2. Playwright Chromium — stock headless Chrome via system binary (FreeBSD fallback)
    """
    global _camoufox_browser, _camoufox_ctx
    if _camoufox_browser is not None:
        return _camoufox_browser
    async with _camoufox_init_lock:
        if _camoufox_browser is not None:
            return _camoufox_browser
        Path(CAMOUFOX_PROFILE_DIR).mkdir(parents=True, exist_ok=True)
        try:
            from camoufox.async_api import AsyncCamoufox
            ctx = AsyncCamoufox(headless=True, os="windows", user_data_dir=CAMOUFOX_PROFILE_DIR)
            _camoufox_browser = await ctx.__aenter__()
            _camoufox_ctx = ctx
        except (ImportError, Exception):
            # Camoufox not available or failed — fall back to Playwright with system Chromium.
            import shutil as _shutil
            from playwright.async_api import async_playwright
            chrome = (
                _os.environ.get("CHROME_EXECUTABLE")
                or _shutil.which("google-chrome")
                or _shutil.which("chromium-browser")
                or _shutil.which("chromium")
                or _shutil.which("chrome")
                or "/usr/local/bin/chrome"
            )
            ctx = async_playwright()
            pw = await ctx.__aenter__()
            _camoufox_browser = await pw.chromium.launch(
                executable_path=chrome,
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--window-size=1280,800",
                ],
            )
            _camoufox_ctx = ctx
    return _camoufox_browser


async def _fetch_with_camoufox(
    url: str,
    referer: str | None = None,
    timeout: int = 30,
) -> FetchResult:
    global _camoufox_browser, _camoufox_ctx

    st = _state(url)
    status = 200
    html = ""
    final_url = url
    page = None

    # Acquire a page from the persistent browser; restart once if it has died.
    for _attempt in range(2):
        try:
            browser = await _ensure_camoufox()
            page = await browser.new_page()
            break
        except Exception:
            async with _camoufox_init_lock:
                if _camoufox_ctx is not None:
                    try:
                        await _camoufox_ctx.__aexit__(None, None, None)
                    except Exception:
                        pass
                _camoufox_browser = None
                _camoufox_ctx = None

    if page is None:
        return FetchResult(url=final_url, status=status, html=html, via_browser=True)

    try:
        # Inject JS-layer patches that Camoufox's C++ doesn't cover
        await page.add_init_script(_CAMOUFOX_EXTRA_SCRIPT)

        if referer:
            await page.set_extra_http_headers({"Referer": referer})

        def _on_response(response):
            nonlocal status
            try:
                if (response.request.resource_type == "document"
                        and response.request.is_navigation_request()):
                    status = response.status
            except Exception:
                pass

        page.on("response", _on_response)

        try:
            # Akamai: visit the homepage first within this browser session so
            # the sensor data collector accumulates trust before the target URL.
            if st.is_akamai:
                origin = _origin(url)
                try:
                    await page.goto(origin + "/", wait_until="domcontentloaded", timeout=15_000)
                    await asyncio.sleep(random.uniform(2.0, 4.0))
                except Exception:
                    pass

            await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)

            try:
                await page.wait_for_selector("body", timeout=5_000)
            except Exception:
                pass

            # Wait out Cloudflare challenge — up to 40s (10×4s) to cover
            # Turnstile proof-of-work puzzles which can take 10–30s to solve.
            cf_seen = False
            for _ in range(10):
                content_lower = (await page.content()).lower()
                if any(ind in content_lower for ind in _CF_CHALLENGE_INDICATORS):
                    await asyncio.sleep(4.0)
                    cf_seen = True
                else:
                    break

            if cf_seen:
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=8_000)
                except Exception:
                    pass

            # Wait for meaningful content element
            try:
                await page.wait_for_selector(
                    "main, article, [role='main'], #content, #main",
                    timeout=6_000,
                )
            except Exception:
                pass

            # Let JS-hydrated SPAs finish rendering — wait for network to go
            # quiet before harvesting so late-injected content is present.
            try:
                await page.wait_for_load_state("networkidle", timeout=8_000)
            except Exception:
                pass

            await _dismiss_cookie_banners(page)
            await _human_mouse_move(page)
            await _human_scroll(page)

            html = await page.content()
            final_url = page.url

            # Re-poll if body is nearly empty — SPA may still be hydrating
            for _ in range(2):
                if _body_text_length(html) >= 50:
                    break
                await asyncio.sleep(5.0)
                try:
                    html = await page.content()
                    final_url = page.url
                except Exception:
                    break

            # Harvest clearance/session cookies for the solve-and-bounce path.
            await _harvest_camoufox_cookies(page, st)
        except Exception:
            pass
    finally:
        try:
            await page.close()
        except Exception:
            pass

    return FetchResult(url=final_url, status=status, html=html, via_browser=True)


# ---------------------------------------------------------------------------
# Tier 3 — nodriver (CDP-minimal Chrome, no WebDriver protocol)
# ---------------------------------------------------------------------------

async def _fetch_with_nodriver(url: str, timeout: int = 40) -> FetchResult:
    import nodriver as uc
    from nodriver.cdp import page as _cdp_page

    Path(NODRIVER_PROFILE_DIR).mkdir(parents=True, exist_ok=True)

    st = _state(url)
    html = ""
    final_url = url
    browser = None

    try:
        import shutil as _shutil
        _chrome_path = (
            _os.environ.get("CHROME_EXECUTABLE")
            or _shutil.which("google-chrome")
            or _shutil.which("chromium-browser")
            or _shutil.which("chromium")
            or _shutil.which("chrome")
            or "/usr/local/bin/chrome"
        )
        browser = await uc.start(
            headless=True,
            browser_executable_path=_chrome_path,
            user_data_dir=NODRIVER_PROFILE_DIR,
            browser_args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--window-size=1280,800",
                "--disable-features=WebRTC,WebRTCHWEncoding,WebRTCHWDecoding",
                "--enforce-webrtc-ip-permission-check",
            ],
        )

        # Inject headless-signal patches on main_tab THEN navigate main_tab
        # directly.  browser.get(url) creates a NEW CDP target (new tab), so
        # any script registered on main_tab would never run on that target.
        # Navigating main_tab means the registered script fires in the same
        # document context that will be harvested.
        main_tab = browser.main_tab
        await main_tab.send(
            _cdp_page.add_script_to_evaluate_on_new_document(
                source=_NODRIVER_STEALTH_SCRIPT
            )
        )
        await asyncio.wait_for(main_tab.get(url), timeout=float(timeout))
        tab = main_tab

        # Wait out Cloudflare challenge — up to 40s for Turnstile puzzles
        for _ in range(10):
            content = await tab.get_content()
            if any(ind in content.lower() for ind in _CF_CHALLENGE_INDICATORS):
                await asyncio.sleep(4.0)
            else:
                break

        # Poll document.readyState before proceeding — avoids harvesting a
        # skeleton on JS-heavy pages or mid-navigation after CF resolution.
        for _ in range(10):
            try:
                state = await tab.evaluate("document.readyState")
                if state in ("complete", "interactive"):
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)

        # Wait for the browser's load event to have fired and for network
        # activity to settle — catches late-injecting SPAs that write content
        # after readyState:complete.
        for _ in range(8):
            try:
                load_done = await tab.evaluate(
                    "(performance.getEntriesByType('navigation')[0]?.loadEventEnd ?? 0) > 0"
                )
                if load_done:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)

        await asyncio.sleep(random.uniform(2.5, 4.0))

        # Dismiss cookie banners
        try:
            await tab.evaluate(_COOKIE_ACCEPT_JS)
            await asyncio.sleep(random.uniform(0.3, 0.6))
        except Exception:
            pass

        # Scroll simulation (fires the behavioural events bot detectors watch for)
        await _nodriver_scroll(tab)

        html = await tab.get_content()
        final_url = tab.url

        # Re-poll if body is nearly empty — SPA may still be hydrating
        for _ in range(2):
            if _body_text_length(html) >= 50:
                break
            await asyncio.sleep(5.0)
            try:
                html = await tab.get_content()
                final_url = tab.url
            except Exception:
                break

        # Harvest clearance/session cookies for the solve-and-bounce path.
        await _harvest_nodriver_cookies(browser, st)
    except Exception:
        pass
    finally:
        if browser:
            try:
                browser.stop()
            except Exception:
                pass

    return FetchResult(url=final_url, status=200, html=html, via_browser=True)
