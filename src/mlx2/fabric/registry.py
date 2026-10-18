from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

from . import FabricError


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(raw, path)
    except BaseException:
        Path(raw).unlink(missing_ok=True)
        raise


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    try:
        waited = os.waitid(
            os.P_PID,
            pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
        if waited is not None:
            return False
    except (AttributeError, ChildProcessError, PermissionError):
        pass
    try:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            check=False,
            text=True,
        )
    except PermissionError:
        return True
    except OSError:
        return False
    if result.returncode != 0:
        return False
    status = result.stdout.strip()
    return bool(status) and not status.upper().startswith("Z")


def _loopback(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            and parsed.port is not None
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


def register_local_server(
    home: Path,
    *,
    port: int,
    pid: int,
    api_key_file: str | Path | None = None,
    admin_token_file: str | Path | None = None,
) -> Path:
    url = f"http://127.0.0.1:{port}"
    if not _loopback(url) or not 0 < port <= 65535:
        raise FabricError("local server must use a loopback URL")
    path = Path(home) / "local-servers" / f"{port}.json"
    _atomic_json(
        path,
        {
            "url": url,
            "pid": pid,
            "started_at": time.time(),
            "api_key_file": str(api_key_file) if api_key_file is not None else None,
            "admin_token_file": (
                str(admin_token_file) if admin_token_file is not None else None
            ),
        },
    )
    return path


def unregister_local_server(home: Path, port: int) -> None:
    (Path(home) / "local-servers" / f"{port}.json").unlink(missing_ok=True)


def list_local_servers(home: Path) -> list[dict]:
    root = Path(home) / "local-servers"
    result = []
    for path in sorted(root.glob("*.json")) if root.exists() else ():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(value, dict)
                or not _loopback(value.get("url", ""))
                or not isinstance(value.get("pid"), int)
            ):
                raise ValueError
            if not pid_alive(value["pid"]):
                path.unlink(missing_ok=True)
                continue
            result.append(value)
        except (OSError, ValueError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
    return result
