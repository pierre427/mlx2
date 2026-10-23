"""Built-in local web routing; synthetic providers, no external requests."""
import asyncio
from types import SimpleNamespace

import pytest

from mlx2 import desktop_web


@pytest.fixture(autouse=True)
def synthetic_dns(monkeypatch):
    # Providers are mocked in this file; DNS must remain synthetic too.
    from mlx2.web_tools import network_guard
    monkeypatch.setattr(network_guard, 'resolve', lambda url: (None, 443, ['93.184.216.34']))


@pytest.mark.parametrize('operation,arguments', [
    ('web_search', {'query': ''}), ('web_search', {'query': 'x', 'max_results': True}),
    ('web_search', {'query': 'x', 'region': 'invalid'}),
    ('fetch_page', {'url': 'file:///etc/passwd'}), ('fetch_page', {'url': 'https://user:secret@example.com'}),
    ('fetch_page', {'url': 'https://example.com', 'max_chars': 999}),
])
def test_invalid_requests_fail_before_network(operation, arguments):
    with pytest.raises(ValueError):
        desktop_web.validate(operation, arguments)


def test_native_dispatch_reuses_loop_and_shuts_down(monkeypatch):
    loops = []
    async def execute(operation, arguments):
        loops.append(asyncio.get_running_loop())
        return {'operation': operation, 'query': arguments.get('query')}
    monkeypatch.setattr(desktop_web, 'execute', execute)
    tools = desktop_web.WebTools()
    try:
        assert tools.dispatch({'command': 'web_search', 'query': 'synthetic'})['query'] == 'synthetic'
        tools.dispatch({'command': 'fetch_page', 'url': 'https://example.com'})
        assert loops[0] is loops[1]
    finally:
        tools.close()
    assert not tools.thread.is_alive()
    with pytest.raises(ValueError, match='closing'):
        tools.dispatch({'command': 'web_search', 'query': 'x'})


def test_shutdown_closes_owned_asyncsession_close_and_aclose_clients(monkeypatch):
    from mlx2.web_tools import fetcher
    closed = []

    class CurlSession:
        async def close(self):
            closed.append('close')

    class HttpxSession:
        async def aclose(self):
            closed.append('aclose')

    async def execute(*args):
        return {'results': []}

    monkeypatch.setattr(desktop_web, 'execute', execute)
    monkeypatch.setattr(fetcher, '_states', {'curl': SimpleNamespace(session=CurlSession()),
                                          'httpx': SimpleNamespace(session=HttpxSession())})
    monkeypatch.setattr(fetcher, '_camoufox_ctx', None)
    tools = desktop_web.WebTools()
    tools.dispatch({'command': 'web_search', 'query': 'synthetic'})
    tools.close()
    assert sorted(closed) == ['aclose', 'close']
    assert fetcher._states == {} and not tools.thread.is_alive()


@pytest.mark.parametrize('url,focus,expected', [
    ('https://example.com', 'content', 'content'),
    ('https://example.com#isinstance', None, 'isinstance'),
    ('https://example.com#section%20one', 'explicit', 'explicit'),
])
def test_fetch_uses_local_parser_cache_focus_and_bounded_result(monkeypatch, url, focus, expected):
    from mlx2.web_tools import cache, fetcher, parser, robots
    async def check(url):
        return True, 'synthetic robots'
    async def fetch(url, **options):
        assert options['allow_browser'] is False
        return SimpleNamespace(url=url, status=200, via_browser=False, content_type='text/html', raw=b'', html='<html>fixture</html>')
    stored, seen = [], []
    monkeypatch.setattr(robots, 'check', check)
    monkeypatch.setattr(fetcher, 'fetch', fetch)
    monkeypatch.setattr(cache, 'get', lambda url: None)
    monkeypatch.setattr(cache, 'put', lambda *args: stored.append(args))
    monkeypatch.setattr(cache, 'mark_seen', lambda urls: seen.extend(urls))
    monkeypatch.setattr(parser, 'html_to_markdown', lambda *args, **kw: 'full source content')
    monkeypatch.setattr(parser, 'extract_metadata', lambda *args, **kw: {'title': 'source'})
    def focused(content, query, maximum):
        assert query == expected
        return 'focused content'
    monkeypatch.setattr(parser, 'focus_markdown', focused)
    result = asyncio.run(desktop_web.execute('fetch_page', {'url': url, 'focus': focus, 'browser_fallback': False}))
    assert result['content'] == 'focused content' and result['reference_only']
    assert stored[0][4] == 'full source content' and seen == [url, url]
