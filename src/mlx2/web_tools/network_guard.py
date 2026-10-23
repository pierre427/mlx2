"""Public-web transport with per-hop address validation and pinned connections.

The socket connects to an address from the checked DNS answer; HTTPS retains
hostname verification and SNI. No proxy, browser, or second DNS lookup can
expand that authority.
"""
from __future__ import annotations

import asyncio
import http.client
import ipaddress
import socket
import ssl
import time
from types import SimpleNamespace
from urllib.parse import urljoin, urlsplit

MAX_BODY = 20 * 1024 * 1024


def validate_url(url):
    parsed = urlsplit(url)
    host = (parsed.hostname or "").rstrip(".").lower()
    if parsed.scheme not in {"http", "https"} or not host or parsed.username is not None or parsed.password is not None:
        raise ValueError("Public web access requires an HTTP(S) URL without credentials")
    if any(ord(c) < 33 or ord(c) == 127 for c in url) or "%" in host:
        raise ValueError("Invalid public web URL")
    if host == "localhost" or host.endswith((".localhost", ".local")) or host == "local":
        raise ValueError("Private web destinations are not allowed")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        _public(address)
    parsed.port  # Validate port syntax before opening a socket.
    return parsed


def _public(address):
    mapped = getattr(address, "ipv4_mapped", None)
    if not address.is_global or address.is_multicast or (mapped is not None and not mapped.is_global):
        raise ValueError("Private or reserved web destinations are not allowed")


def resolve(url):
    parsed = validate_url(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    records = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    addresses = list(dict.fromkeys(record[4][0] for record in records))
    if not addresses:
        raise ValueError("Public web destination did not resolve")
    for address in addresses:
        _public(ipaddress.ip_address(address))
    return parsed, port, addresses


def _get(url, *, headers=None, timeout=20, allow_redirects=True, max_redirects=10):
    deadline = time.monotonic() + timeout
    for hop in range(max_redirects + 1):
        parsed, port, addresses = resolve(url)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Public web request timed out")
        # Connect directly to the validated address, preserving Host and TLS SNI.
        connection = http.client.HTTPConnection(parsed.hostname, port, timeout=remaining)
        def connect():
            stream = socket.create_connection((addresses[0], port), timeout=remaining)
            if parsed.scheme == "https":
                try:
                    stream = ssl.create_default_context().wrap_socket(stream, server_hostname=parsed.hostname)
                except BaseException:
                    stream.close()
                    raise
            connection.sock = stream
        connection.connect = connect
        request_headers = {k: v for k, v in (headers or {}).items()
                           if k.lower() not in {"host", "accept-encoding", "authorization", "cookie", "proxy-authorization"}}
        request_headers["Accept-Encoding"] = "identity"
        try:
            connection.request("GET", (parsed.path or "/") + (("?" + parsed.query) if parsed.query else ""), headers=request_headers)
            response = connection.getresponse()
            response_headers = {k.lower(): v for k, v in response.getheaders()}
            location = response_headers.get("location")
            if allow_redirects and response.status in {301, 302, 303, 307, 308} and location:
                if hop == max_redirects:
                    raise ValueError("Public web redirect limit exceeded")
                url = urljoin(url, location)
                continue
            body = response.read(MAX_BODY + 1)
            if len(body) > MAX_BODY:
                raise ValueError("Public web response exceeds 20 MiB")
            charset = response.headers.get_content_charset() or "utf-8"
            try:
                decoded = body.decode(charset, errors="replace")
            except LookupError:
                decoded = body.decode("utf-8", errors="replace")
            return SimpleNamespace(url=url, status_code=response.status, headers=response_headers, content=body, text=decoded)
        finally:
            connection.close()
    raise ValueError("Public web redirect limit exceeded")


async def get(url, **kwargs):
    return await asyncio.to_thread(_get, url, **kwargs)
