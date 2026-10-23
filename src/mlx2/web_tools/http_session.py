"""Profile-matched async HTTP sessions on macOS and FreeBSD."""

from .headers import SEARCH_IMPERSONATE, SEARCH_OS

try:
    from curl_cffi.requests import AsyncSession  # type: ignore[no-redef]
    CURL_CFFI_AVAILABLE = True
except ImportError:
    import primp

    CURL_CFFI_AVAILABLE = False

    class AsyncSession:
        """curl_cffi-shaped adapter using primp's fixed Chrome/macOS profile."""

        def __init__(self, **_kwargs):
            self._client = primp.AsyncClient(
                impersonate=SEARCH_IMPERSONATE,
                impersonate_os=SEARCH_OS,
                follow_redirects=True,
                max_redirects=10,
            )

        async def get(self, url, *, allow_redirects=True, max_redirects=10, **kwargs):
            return await self._client.get(url, **kwargs)

        async def post(self, url, *, allow_redirects=True, max_redirects=10, **kwargs):
            return await self._client.post(url, **kwargs)

        def set_cookies(self, url, cookies):
            self._client.set_cookies(url, cookies)

        async def aclose(self):
            await self._client.__aexit__(None, None, None)

        async def __aenter__(self):
            await self._client.__aenter__()
            return self

        async def __aexit__(self, *_exc):
            return await self._client.__aexit__(*_exc)
