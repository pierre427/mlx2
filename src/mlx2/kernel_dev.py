"""Control a resident model's development kernels through its loopback admin API."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def _base_url(value):
    parsed = urllib.parse.urlsplit(value)
    try:
        local = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
    except (ValueError, TypeError):
        local = False
    if (parsed.scheme not in {"http", "https"} or not local
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        raise ValueError("--url must be a loopback HTTP(S) server URL without credentials or a path")
    try:
        _ = parsed.port
    except ValueError as error:
        raise ValueError("--url has an invalid port") from error
    return value.rstrip("/")


def _http(base, method, path, *, token, body=None, timeout=30):
    request = urllib.request.Request(
        _base_url(base) + path,
        data=None if body is None else json.dumps(body, allow_nan=False).encode("utf-8"),
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise SystemExit(f"{method} {path} failed with HTTP {error.code}: {detail}") from None
    except (OSError, urllib.error.URLError) as error:
        raise SystemExit(
            f"{method} {path} did not complete: {error}. "
            "Read status before retrying; a disconnected reload may still be running."
        ) from None


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m mlx2.kernel_dev",
        description="reload kernels or switch variants while a model's weights remain resident",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8285")
    parser.add_argument("--admin-token-file", required=True, type=Path)
    parser.add_argument("--timeout", type=float, help="HTTP timeout (default: drain timeout plus 120 seconds)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="show availability, execution generation and last reload receipt")
    reload = commands.add_parser("reload", help="drain, purge APCv2, replace kernels/settings, and resume")
    reload.add_argument("--modules", nargs="+", default=[])
    reload.add_argument("--settings", help='JSON list, e.g. [{"target":"package.kernel:MODE","value":"candidate"}]')
    reload.add_argument("--calls", help='JSON list, e.g. [{"target":"package.kernel:set_mode","args":["candidate"]}]')
    reload.add_argument("--file", "--request", dest="request_file", type=Path,
                        help="JSON request file; cannot be combined with operation flags")
    reload.add_argument("--drain-timeout-seconds", type=float)
    return parser


def _main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    from .http_security import load_secret_file

    try:
        base = _base_url(args.url)
        token = load_secret_file(args.admin_token_file, flag="--admin-token-file")
        body = None
        if args.command == "reload":
            if args.request_file is not None:
                if args.modules or args.settings is not None or args.calls is not None or args.drain_timeout_seconds is not None:
                    raise ValueError("--file cannot be combined with --modules, --settings, --calls or --drain-timeout-seconds")
                body = json.loads(args.request_file.read_text(encoding="utf-8"))
            else:
                body = {"modules": args.modules}
                if args.settings is not None:
                    body["settings"] = json.loads(args.settings)
                if args.calls is not None:
                    body["calls"] = json.loads(args.calls)
                if args.drain_timeout_seconds is not None:
                    body["drain_timeout_seconds"] = args.drain_timeout_seconds
            from .server import validate_kernel_reload_body

            body = validate_kernel_reload_body(body)
            # JSON has no NaN/Infinity, including nested setting/call values.
            json.dumps(body, allow_nan=False)
        timeout = args.timeout if args.timeout is not None else (
            body["drain_timeout_seconds"] + 120 if body is not None else 30
        )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("--timeout must be a positive finite number")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    result = _http(
        base, "POST" if body is not None else "GET",
        "/v1/admin/kernels/reload" if body is not None else "/v1/admin/kernels",
        token=token, body=body, timeout=timeout,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
