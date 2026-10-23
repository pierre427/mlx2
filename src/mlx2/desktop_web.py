"""Built-in local web tools. No MCP connection, subprocess, or model is used."""
from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import re
import sys
import threading
from urllib.parse import unquote, urlsplit


def validate(operation, arguments):
    values = dict(arguments)
    if operation == 'web_search':
        query = values.get('query')
        if not isinstance(query, str) or not query.strip() or len(query.encode()) > 8192:
            raise ValueError('Search needs a nonempty query of at most 8192 bytes')
        maximum = values.get('max_results', 10)
        if type(maximum) is not int or not 1 <= maximum <= 20:
            raise ValueError('max_results must be 1–20')
        if not re.fullmatch('[a-z]{2}-[a-z]{2}', values.get('region', 'us-en')):
            raise ValueError('Use a region such as ca-en')
        if values.get('timelimit') not in (None, 'd', 'w', 'm', 'y'):
            raise ValueError('timelimit must be d, w, m, y, or null')
    elif operation == 'fetch_page':
        for key in ('url', 'referer'):
            url = values.get(key)
            if url is None and key == 'referer':
                continue
            if not isinstance(url, str) or len(url) > 8192:
                raise ValueError('Supply a bounded HTTP(S) URL')
            from .web_tools.network_guard import validate_url
            validate_url(url)
            parsed = urlsplit(url)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError('Supply an HTTP(S) URL without embedded credentials')
        maximum = values.get('max_chars', 50000)
        if type(maximum) is not int or not 1000 <= maximum <= 500000:
            raise ValueError('max_chars must be 1000–500000')
        if type(values.get('browser_fallback', True)) is not bool:
            raise ValueError('browser_fallback must be boolean')
        focus = values.get('focus')
        if focus is not None and (not isinstance(focus, str) or len(focus.encode()) > 8192):
            raise ValueError('focus must contain at most 8192 bytes')
    else:
        raise ValueError('Unknown built-in web operation')
    return values


async def execute(operation, arguments):
    from .web_tools import cache, robots
    if operation == 'web_search':
        from .web_tools.search import web_search
        results = await web_search(**arguments)
        cache.mark_seen([row.get('url') for row in results])
        return {'results': results, 'suggested_referer': 'https://duckduckgo.com/', 'provider': 'built-in local web tools'}
    from .web_tools.fetcher import fetch
    from .web_tools.parser import extract_metadata, focus_markdown, html_to_markdown
    from .web_tools.pdf import extract_pdf
    url = arguments['url']
    from .web_tools.network_guard import resolve
    await asyncio.to_thread(resolve, url)
    allowed, robots_message = await robots.check(url)
    hit = cache.get(url)
    if hit:
        content, metadata = hit['content'], hit['metadata']
        final_url, status, via_browser = hit['final_url'], hit['status'], hit['via_browser']
        await asyncio.to_thread(resolve, final_url)
    else:
        result = await fetch(url, referer=arguments.get('referer'), force_browser=False,
                             allow_browser=False)
        if result.content_type == 'application/pdf' and result.raw:
            content, metadata = extract_pdf(result.raw)
        else:
            content = html_to_markdown(result.html, url=result.url, max_chars=500000)
            metadata = extract_metadata(result.html, url=result.url)
        final_url, status, via_browser = result.url, result.status, result.via_browser
        if result.status == 200 and content.strip():
            cache.put(url, str(status), final_url, via_browser, content, metadata)
    cache.mark_seen([url, final_url])
    maximum = arguments.get('max_chars', 50000)
    focus = arguments.get('focus') or unquote(urlsplit(url).fragment).replace('-', ' ').strip()
    if focus:
        focused = focus_markdown(content, focus, maximum)
        if focused is not None:
            content = focused
    truncated = len(content) > maximum
    return {'url': final_url, 'status': str(status), 'content': content[:maximum], 'metadata': metadata,
            'via_browser': str(via_browser), 'robots': robots_message, 'robots_allowed': allowed,
            'cached': bool(hit), 'truncated': truncated, 'reference_only': True,
            'provider': 'built-in local web tools', 'network_scope': 'public', 'browser_fallback_available': False}


class WebTools:
    """One persistent event loop owns HTTP sessions, throttles, and browser state."""
    def __init__(self):
        self.lock = threading.Lock()
        self.loop = None
        self.thread = None
        self.closed = False

    def dispatch(self, request):
        operation = request['command']
        arguments = validate(operation, {k: v for k, v in request.items() if k != 'command'})
        with self.lock:
            if self.closed:
                raise ValueError('Built-in web tools are closing')
            if self.loop is None:
                self.loop = asyncio.new_event_loop()
                self.thread = threading.Thread(target=self.loop.run_forever, daemon=True, name='mlx2-local-web')
                self.thread.start()
            future = asyncio.run_coroutine_threadsafe(asyncio.wait_for(execute(operation, arguments), 60), self.loop)
        try:
            return future.result(timeout=65)
        except (TimeoutError, concurrent.futures.TimeoutError):
            future.cancel()
            raise ValueError('Built-in web request timed out; diagnose the source or change the request before retrying') from None
        except ImportError:
            raise ValueError('Built-in web dependencies are missing; install the desktop-web package extra') from None

    def close(self):
        with self.lock:
            self.closed = True
            if self.loop is not None:
                async def cleanup():
                    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    fetcher = sys.modules.get('mlx2.web_tools.fetcher')
                    if fetcher is not None:
                        async def close_session(session):
                            closer = getattr(session, 'aclose', None) or getattr(session, 'close', None)
                            if closer is not None:
                                result = closer()
                                if inspect.isawaitable(result):
                                    await result
                        await asyncio.gather(*(close_session(state.session) for state in list(fetcher._states.values())), return_exceptions=True)
                        fetcher._states.clear()
                        if fetcher._camoufox_ctx is not None:
                            await fetcher._camoufox_ctx.__aexit__(None, None, None)
                            fetcher._camoufox_ctx = fetcher._camoufox_browser = None
                    cache = sys.modules.get('mlx2.web_tools.cache')
                    if cache is not None and cache._conn is not None:
                        cache._conn.close()
                        cache._conn = None
                try:
                    asyncio.run_coroutine_threadsafe(cleanup(), self.loop).result(timeout=5)
                except (TimeoutError, concurrent.futures.TimeoutError):
                    pass
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.thread.join(timeout=5)
                if not self.thread.is_alive():
                    self.loop.close()
