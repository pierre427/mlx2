"""Request settings shared by the curl_cffi HTTP paths.

curl_cffi supplies the User-Agent, client hints, and navigation headers for
its impersonation profile. Supplying our own Chrome version or platform here
can make those headers disagree with its TLS and HTTP/2 fingerprint.
"""
from urllib.parse import urlparse


HTTP_IMPERSONATE = "chrome146"
SEARCH_IMPERSONATE = "chrome_146"  # primp uses a different profile spelling
SEARCH_OS = "macos"


def navigation_headers(url: str, referer: str | None = None) -> dict[str, str]:
    """Add navigation context without replacing profile-owned headers."""
    if not referer:
        return {}
    return {
        "referer": referer,
        "sec-fetch-site": _sec_fetch_site(url, referer),
    }


def _sec_fetch_site(url: str, referer: str) -> str:
    dest = urlparse(url)
    src = urlparse(referer)
    if dest.netloc == src.netloc:
        return "same-origin"
    dest_parts = dest.hostname.split(".") if dest.hostname else []
    src_parts = src.hostname.split(".") if src.hostname else []
    if len(dest_parts) >= 2 and len(src_parts) >= 2 and dest_parts[-2:] == src_parts[-2:]:
        return "same-site"
    return "cross-site"
