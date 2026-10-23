"""Public web authority tests; synthetic DNS/transport, never public network."""
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from types import SimpleNamespace

import pytest

from mlx2 import desktop_web
from mlx2.web_tools import network_guard as guard


@pytest.mark.parametrize('url', ['http://127.0.0.1/x', 'http://10.1.1.1/', 'http://192.168.1.1/',
    'http://169.254.169.254/', 'http://[::1]/', 'http://[::ffff:127.0.0.1]/',
    'http://localhost/', 'http://a.localhost/', 'http://machine.local/', 'http://224.0.0.1/',
    'http://0.0.0.0/', 'http://127.1/', 'http://2130706433/'])
def test_private_fetch_has_no_http_side_effect(url, monkeypatch):
    hits = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200); self.end_headers()
        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        with pytest.raises(ValueError):
            guard._get(url, timeout=1)
        assert hits == []
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_dns_mixed_public_private_answer_is_refused_before_connect(monkeypatch):
    monkeypatch.setattr(guard.socket, 'getaddrinfo', lambda *args, **kw: [
        (2,1,6,'',('93.184.216.34',80)), (2,1,6,'',('127.0.0.1',80))])
    monkeypatch.setattr(guard.socket, 'create_connection', lambda *a, **kw: pytest.fail('must not connect'))
    with pytest.raises(ValueError):
        guard._get('http://public.example/')


def test_redirect_target_checked_and_socket_pinned_to_checked_address(monkeypatch):
    dns_calls, connections = [], []
    def dns(host, port, **kw):
        dns_calls.append(host)
        return [(2,1,6,'',('93.184.216.34',port))] if len(dns_calls) == 1 else [(2,1,6,'',('127.0.0.1',port))]
    monkeypatch.setattr(guard.socket, 'getaddrinfo', dns)
    monkeypatch.setattr(guard.socket, 'create_connection', lambda address, **kw: connections.append(address) or object())
    class Connection:
        def __init__(self, *a, **kw): pass
        def request(self, *a, **kw): self.connect()
        def getresponse(self):
            return SimpleNamespace(status=302, getheaders=lambda: [('location', 'http://rebound.example/private')])
        def close(self): pass
    monkeypatch.setattr(guard.http.client, 'HTTPConnection', Connection)
    with pytest.raises(ValueError):
        guard._get('http://public.example/')
    assert connections == [('93.184.216.34', 80)]
    assert dns_calls == ['public.example', 'rebound.example']


def test_robots_warmups_and_target_use_same_guard(monkeypatch):
    from mlx2.web_tools import fetcher, robots
    calls = []
    async def deny(url, **kwargs):
        calls.append(url)
        raise ValueError('blocked destination')
    monkeypatch.setattr(guard, 'get', deny)
    monkeypatch.setattr(fetcher, 'public_get', deny)
    monkeypatch.setattr(fetcher.random, 'uniform', lambda *args: 0)
    async def exercise():
        await robots._fetch_robots('https://public.example/robots.txt')
        state = SimpleNamespace(warmed=False)
        await fetcher._warm_session(state, 'https://public.example/path')
        await fetcher._warm_akamai_extra(state, 'https://public.example', '<a href="/inside">inside</a>')
    asyncio.run(exercise())
    assert calls == ['https://public.example/robots.txt', 'https://public.example/', 'https://public.example/inside']


def test_browser_escalation_cannot_bypass_public_transport(monkeypatch):
    from mlx2.web_tools import fetcher
    monkeypatch.setattr(fetcher, 'resolve', lambda url: (None, 80, ['93.184.216.34']))
    async def lightweight(*args, **kwargs):
        return fetcher.FetchResult('http://public.example/', 403, 'blocked', False)
    async def browser(*args, **kwargs):
        pytest.fail('Unpinned browser must not run')
    async def no_throttle(*args): pass
    monkeypatch.setattr(fetcher, '_fetch_lightweight', lightweight)
    monkeypatch.setattr(fetcher, '_fetch_with_camoufox', browser)
    monkeypatch.setattr(fetcher, '_fetch_with_nodriver', browser)
    monkeypatch.setattr(fetcher, 'throttle', no_throttle)
    result = asyncio.run(fetcher.fetch('http://public.example/', allow_browser=True))
    assert result.status == 403 and not result.via_browser


def test_legacy_cached_private_redirect_is_refused(monkeypatch):
    from mlx2.web_tools import cache, robots
    async def robots_check(url): return True, 'synthetic'
    def resolve(url):
        guard.validate_url(url)
        return None, 80, ['93.184.216.34']
    monkeypatch.setattr(guard, 'resolve', resolve)
    monkeypatch.setattr(robots, 'check', robots_check)
    monkeypatch.setattr(cache, 'get', lambda url: {'content': 'private cached secret', 'metadata': {},
        'final_url': 'http://127.0.0.1/private', 'status': '200', 'via_browser': False})
    with pytest.raises(ValueError):
        asyncio.run(desktop_web.execute('fetch_page', {'url': 'https://public.example/'}))
