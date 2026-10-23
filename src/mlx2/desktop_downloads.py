"""Pinned Hugging Face acquisition with one app-wide downloader and throttle."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

from .desktop import atomic_json
from .http_security import load_secret_file


class DownloadError(ValueError):
    """Safe project-generated validation text, without provider secrets."""


class GlobalThrottle:
    def __init__(self, bytes_per_second=0):
        self.lock = threading.Lock()
        self.rate = bytes_per_second
        self.next_at = time.monotonic()

    def set_rate(self, rate):
        if type(rate) is not int or rate < 0:
            raise DownloadError("Download throttle must be a nonnegative byte count per second")
        with self.lock:
            self.rate = rate
            self.next_at = time.monotonic()

    def consume(self, size, cancel):
        with self.lock:
            if self.rate == 0:
                return
            now = time.monotonic()
            self.next_at = max(now, self.next_at) + size / self.rate
            delay = self.next_at - now
        if cancel.wait(delay):
            raise InterruptedError("Download cancelled")


def parse_hf_target(value):
    if not isinstance(value, str):
        raise TypeError("Enter a Hugging Face model URL or owner/repository")
    value = value.strip()
    revision = "main"
    if value.startswith("https://"):
        url = urllib.parse.urlsplit(value)
        if url.hostname != "huggingface.co" or url.port not in {None, 443} or url.username or url.password or url.query or url.fragment:
            raise DownloadError("Use an HTTPS huggingface.co model link")
        parts = url.path.strip("/").split("/")
        if len(parts) == 4 and parts[2] == "tree":
            revision = urllib.parse.unquote(parts[3])
            parts = parts[:2]
        if len(parts) != 2:
            raise DownloadError("Use a repository link, optionally with /tree/<revision>")
        value = "/".join(parts)
    elif "@" in value:
        value, revision = value.rsplit("@", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value) or any(p in {".", ".."} for p in value.split("/")):
        raise DownloadError("Model identifiers must be owner/repository")
    if not revision or len(revision) > 256:
        raise DownloadError("Invalid model revision")
    return value, revision


class _SafeHubRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise DownloadError("Download redirect must remain HTTPS")
        result = super().redirect_request(request, fp, code, msg, headers, newurl)
        origin = urllib.parse.urlsplit(request.full_url)
        if result and (parsed.hostname, parsed.port or 443) != (origin.hostname, origin.port or 443):
            result.remove_header("Authorization")
        return result


class Downloads:
    def __init__(self, controller):
        self.controller = controller
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=controller.preferences["download_parallelism"], thread_name_prefix="mlx2-download")
        self.throttle = GlobalThrottle(controller.preferences["download_bytes_per_second"])
        self.cancellations = {}
        self.lock = threading.RLock()
        self.progress = {}
        self.threads = []
        self.responses = {}
        self.closed = False

    def token(self):
        path = self.controller.preferences["hf_token_file"]
        return load_secret_file(path, flag="HF token file") if path else None

    def configure(self, old, new):
        if old["download_parallelism"] != new["download_parallelism"]:
            if self.cancellations:
                raise DownloadError("Wait for downloads to finish before changing parallelism")
            self.pool.shutdown(wait=False)
            self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=new["download_parallelism"], thread_name_prefix="mlx2-download")
        self.throttle.set_rate(new["download_bytes_per_second"])

    def start(self, target, *, adapter_after=False):
        repository, revision = parse_hf_target(target)
        spec = {"repository": repository, "requested_revision": revision, "adapter_after": bool(adapter_after), "root": str(self.controller.primary.library.root)}
        with self.lock:
            if self.closed:
                raise DownloadError("The app is closing; download was not queued")
            identity = self.controller.store.enqueue("download", spec)
            return self._launch(identity, spec)

    def _launch(self, identity, spec):
        cancel = threading.Event()
        with self.lock:
            if self.closed or identity in self.cancellations:
                raise DownloadError("Download is already active or the app is closing")
            self.cancellations[identity] = cancel
            self.progress[identity] = {"downloaded_bytes": 0, "total_bytes": None, "files_done": 0}
        thread = threading.Thread(target=self._run, args=(identity, spec, cancel), daemon=True, name="mlx2-hf-" + identity[:8])
        self.threads.append(thread)
        thread.start()
        return {"job_id": identity}

    def resume(self, identity):
        job = next((row for row in self.controller.store.jobs(1000) if row["id"] == identity and row["kind"] == "download"), None)
        if not job or job["state"] not in {"cancelled", "failed", "interrupted"}:
            raise DownloadError("Only an interrupted, failed or cancelled download can resume")
        return self._launch(identity, {**job["spec"], "resume": True})

    @staticmethod
    def _name(name):
        if not isinstance(name, str) or not name or "\\" in name or any(ord(c) < 32 or ord(c) == 127 for c in name):
            raise DownloadError("Unsafe repository filename")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name or name == "." or any(p == ".mlx2-acquisition.json" or p.endswith(".mlx2-partial") for p in relative.parts):
            raise DownloadError("Unsafe or reserved repository filename")
        return relative

    @staticmethod
    def _hash(path, cancel):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 << 20):
                if cancel.is_set():
                    raise InterruptedError("Download cancelled")
                digest.update(block)
        return digest.hexdigest()

    def _file(self, repository, sha, sibling, staging, token, cancel, identity, verified=None):
        from huggingface_hub import hf_hub_url
        name = sibling.rfilename
        relative = self._name(name)
        parent = staging
        if staging.is_symlink() or not staging.is_dir():
            raise DownloadError("Download staging directory changed")
        for component in relative.parts[:-1]:
            parent = parent / component
            if parent.is_symlink() or parent.exists() and not parent.is_dir():
                raise DownloadError("Download destinations must not be symlinks")
            parent.mkdir(mode=0o700, exist_ok=True)
        destination = parent / relative.name
        partial = destination.with_name(destination.name + ".mlx2-partial")
        if destination.is_symlink() or partial.is_symlink():
            raise DownloadError("Download destinations must not be symlinks")
        if destination.exists():
            if not destination.is_file() or not verified or verified.get("bytes") != sibling.size or destination.stat().st_size != sibling.size or self._hash(destination, cancel) != verified.get("sha256"):
                raise DownloadError("Existing staged file has no matching verified receipt")
            with self.lock:
                self.progress[identity]["files_done"] += 1
            return verified
        if partial.exists() and (not partial.is_file() or partial.stat().st_nlink != 1):
            raise DownloadError("Partial downloads must be private regular files")
        expected_size = sibling.size
        lfs = getattr(sibling, "lfs", None)
        expected_sha = getattr(lfs, "sha256", None) if lfs else None
        if isinstance(lfs, dict):
            expected_sha = lfs.get("sha256")
        headers = {"User-Agent": "mlx2-desktop"}
        if token:
            headers["Authorization"] = "Bearer " + token
        offset = partial.stat().st_size if partial.exists() else 0
        baseline = offset
        if offset > expected_size:
            raise DownloadError("Partial download is larger than pinned file metadata")
        if partial.exists() and offset == expected_size:
            # Cancellation can arrive after the last byte but before hashing
            # or publication. A range starting at EOF is unsatisfiable (416).
            if expected_sha:
                return self._finish_file(partial, destination, expected_size, expected_sha, cancel, identity, name)
            # Without a provider digest, fetch the pinned file again rather
            # than accepting unverified local bytes based only on their size.
            offset = 0
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(hf_hub_url(repository, name, revision=sha), headers=headers)
        opener = urllib.request.build_opener(_SafeHubRedirect())
        if cancel.is_set():
            raise InterruptedError("Download cancelled")
        with opener.open(request, timeout=15) as response:
            with self.lock:
                self.responses.setdefault(identity, []).append(response)
            try:
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or tuple(map(int, match.groups())) != (offset, expected_size - 1, expected_size):
                        raise DownloadError("Download resume range did not match pinned file metadata")
                elif response.status == 200:
                    offset = 0
                else:
                    raise DownloadError("Unexpected download response status")
                length = response.headers.get("Content-Length")
                if length is not None and (not length.isdecimal() or int(length) != expected_size - offset):
                    raise DownloadError("Download response length did not match pinned file metadata")
                flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | (os.O_APPEND if offset else os.O_TRUNC)
                fd = os.open(partial, flags, 0o600)
                with os.fdopen(fd, "ab" if offset else "wb") as stream:
                    written = offset
                    while True:
                        if cancel.is_set():
                            raise InterruptedError("Download cancelled")
                        block = response.read(min(64 << 10, expected_size - written + 1))
                        if not block:
                            break
                        if written + len(block) > expected_size:
                            raise DownloadError("Download body exceeds pinned file size")
                        self.throttle.consume(len(block), cancel)
                        growth = max(0, written + len(block) - max(written, baseline))
                        self.controller.disk.check(identity, growth)
                        stream.write(block)
                        self.controller.disk.consumed(identity, growth)
                        written += len(block)
                        with self.lock:
                            self.progress[identity]["downloaded_bytes"] += growth
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                with self.lock:
                    self.responses[identity].remove(response)
        return self._finish_file(partial, destination, expected_size, expected_sha, cancel, identity, name)

    def _finish_file(self, partial, destination, expected_size, expected_sha, cancel, identity, name):
        if cancel.is_set():
            raise InterruptedError("Download cancelled")
        if expected_size is not None and partial.stat().st_size != expected_size:
            raise DownloadError("Downloaded file size does not match HF metadata")
        digest = self._hash(partial, cancel)
        if expected_sha and digest != expected_sha:
            raise DownloadError("Downloaded weight digest does not match HF metadata")
        partial.replace(destination)
        with self.lock:
            self.progress[identity]["files_done"] += 1
        return {"path": name, "bytes": destination.stat().st_size, "sha256": digest}

    @staticmethod
    def _publish(staging, destination):
        # Exclusive atomic directory publication. Never replace even an empty
        # user-created destination in the check/rename race.
        import ctypes
        import sys
        libc = ctypes.CDLL(None, use_errno=True)
        if sys.platform == "darwin":
            rename = libc.renamex_np
            result = rename(os.fsencode(staging), os.fsencode(destination), 4)  # RENAME_EXCL
        elif hasattr(libc, "renameat2"):
            result = libc.renameat2(-100, os.fsencode(staging), -100, os.fsencode(destination), 1)
        else:
            raise DownloadError("This platform cannot publish a model without overwriting a destination")
        if result:
            code = ctypes.get_errno()
            raise OSError(code, "Exclusive model publication failed")

    @staticmethod
    def _abort(response):
        import socket
        try:
            raw = getattr(getattr(response, "fp", None), "raw", None)
            sock = getattr(raw, "_sock", None)
            if sock is not None:
                sock.shutdown(socket.SHUT_RDWR)
            response.close()
        except OSError:
            pass

    def _run(self, identity, spec, cancel):
        store = self.controller.store
        futures, files = [], []
        staging = None
        manifest = None
        verified = {}
        state, reason, result = "failed", "Download failed", None
        try:
            from huggingface_hub import HfApi
            token = self.token()
            root = Path(spec.get("root", self.controller.primary.library.root)).resolve(strict=True)
            if root != self.controller.primary.library.root.resolve(strict=True):
                raise DownloadError("The model library changed; restore its location before resuming")
            state_dir = root / ".download-state"
            if state_dir.is_symlink() or state_dir.exists() and not state_dir.is_dir():
                raise DownloadError("Download staging parent must be a real directory")
            staging = state_dir / ("hf-" + identity)
            if staging.is_symlink() or staging.exists() and not staging.is_dir():
                raise DownloadError("Download staging must be a real directory")
            old = None
            if spec.get("resume"):
                marker = staging / ".mlx2-acquisition.json"
                if marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 8_000_000:
                    raise DownloadError("A bounded acquisition receipt is required to resume")
                old = json.loads(marker.read_text())
                if old.get("repository") != spec["repository"] or old.get("job_id") != identity:
                    raise DownloadError("Resume receipt does not match this download")
            elif staging.exists():
                raise DownloadError("This evidence directory already exists; use Resume")
            store.update(identity, "running", "Resolving the exact Hugging Face revision")
            revision = old["resolved_revision"] if old else spec["requested_revision"]
            info = HfApi(token=token).model_info(spec["repository"], revision=revision, files_metadata=True, timeout=15)
            if cancel.is_set():
                raise InterruptedError("Download cancelled")
            sha = info.sha
            if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40,64}", sha) or old and sha != old["resolved_revision"]:
                raise DownloadError("HF did not return the pinned commit")
            names = [self._name(s.rfilename).as_posix() for s in info.siblings]
            if len(names) > 100000 or len(set(names)) != len(names):
                raise DownloadError("HF returned duplicate paths or too many files")
            name_set = set(names)
            if any(any(str(parent) in name_set for parent in PurePosixPath(name).parents if str(parent) != ".") for name in names):
                raise DownloadError("HF file and directory paths collide")
            if any(type(s.size) is not int or s.size < 0 for s in info.siblings):
                raise DownloadError("HF file sizes are unavailable; cannot reserve disk space safely")
            expected = [{"path": s.rfilename, "bytes": s.size} for s in info.siblings]
            if old and old.get("expected_files") != expected:
                raise DownloadError("Pinned file metadata changed; resume refused")
            total = sum(s.size for s in info.siblings)
            destination = root / spec["repository"].split("/", 1)[1]
            if destination.exists() or destination.is_symlink():
                raise DownloadError("A model folder with this name already exists; it will not be overwritten")
            verified = {f["path"]: f for f in old.get("files", [])} if old else {}
            existing = 0
            for name, sibling in zip(names, info.siblings, strict=True):
                path = staging.joinpath(*PurePosixPath(name).parts)
                for candidate in [path, path.with_name(path.name + ".mlx2-partial"), *path.parents]:
                    if candidate == root:
                        break
                    if candidate.is_symlink():
                        raise DownloadError("Download staging must not contain symlinks")
                if path.exists():
                    if name not in verified or not path.is_file() or path.stat().st_size != sibling.size:
                        raise DownloadError("Staged file has no matching verified receipt")
                    existing += sibling.size
                else:
                    partial = path.with_name(path.name + ".mlx2-partial")
                    if partial.exists():
                        if not partial.is_file() or partial.stat().st_nlink != 1 or partial.stat().st_size > sibling.size:
                            raise DownloadError("Invalid partial download")
                        existing += partial.stat().st_size
            self.controller.disk.reserve(identity, root, total - existing)
            state_dir.mkdir(mode=0o700, exist_ok=True)
            staging.mkdir(mode=0o700, exist_ok=True)
            self.progress[identity].update(downloaded_bytes=existing, total_bytes=total, total_files=len(info.siblings), commit=sha)
            manifest = {**spec, "job_id": identity, "resolved_revision": sha, "expected_files": expected, "files": list(verified.values())}
            atomic_json(staging / ".mlx2-acquisition.json", manifest)
            futures = [self.pool.submit(self._file, spec["repository"], sha, sibling, staging, token, cancel, identity, verified.get(sibling.rfilename)) for sibling in info.siblings]
            for future in concurrent.futures.as_completed(futures):
                files.append(future.result())
                manifest["files"] = list({**verified, **{item["path"]: item for item in files}}.values())
                atomic_json(staging / ".mlx2-acquisition.json", manifest)
                store.update(identity, "running", f"Downloaded {len(files)}/{len(futures)} files", dict(self.progress[identity]))
            if cancel.is_set():
                raise InterruptedError("Download cancelled")
            receipt = {**manifest, "completed_at": time.time()}
            atomic_json(staging / ".mlx2-acquisition.json", receipt)
            if cancel.is_set():
                raise InterruptedError("Download cancelled before model publication")
            self._publish(staging, destination)
            state, reason, result = "completed", "Download complete", {**receipt, "model_path": str(destination)}
            try:
                self.controller.primary.library.refresh()
            except Exception:  # noqa: BLE001 -- committed acquisition remains successful
                self.controller.event("download_library_refresh_failed", job_id=identity)
                reason = "Download complete; library refresh needs retrying"
            if spec["adapter_after"] and not cancel.is_set():
                try:
                    from .desktop import artifact_row
                    row = artifact_row(destination)
                    self.controller.helpers.dispatch({"command": "adapter_draft", "path": str(destination), "identity": row["identity"]})
                except Exception:  # noqa: BLE001 -- optional helper cannot undo a committed acquisition
                    self.controller.event("download_adapter_draft_unavailable", job_id=identity)
                    reason = "Download complete; adapter draft unavailable"
        except Exception as error:  # noqa: BLE001 - provider failures must leave durable safe state
            cancelled = cancel.is_set() or isinstance(error, InterruptedError)
            cancel.set()
            for future in futures:
                future.cancel()
            with self.lock:
                for response in list(self.responses.get(identity, [])):
                    self._abort(response)
            state = "cancelled" if cancelled else "failed"
            reason = "Download cancelled" if cancelled else str(error) if isinstance(error, DownloadError) else type(error).__name__
        finally:
            # Keep the job active and its reservation until every worker retires.
            settled = {}
            for future in futures:
                try:
                    item = future.result()
                    settled[item["path"]] = item
                except Exception:  # noqa: BLE001, S110 - original failure is retained above
                    pass
            if state != "completed" and manifest is not None and staging is not None and staging.is_dir():
                manifest["files"] = list({**verified, **settled}.values())
                atomic_json(staging / ".mlx2-acquisition.json", manifest)
            self.controller.disk.release(identity)
            with self.lock:
                self.cancellations.pop(identity, None)
                self.responses.pop(identity, None)
            store.update(identity, state, reason, result)

    def dispatch(self, request):
        if request["command"] == "download_start":
            return self.start(request["target"], adapter_after=request.get("adapter_after", False))
        if request["command"] == "download_resume":
            return self.resume(request["job_id"])
        if request["command"] == "download_cancel":
            with self.lock:
                event = self.cancellations.get(request["job_id"])
                if event:
                    event.set()
                    for response in list(self.responses.get(request["job_id"], [])):
                        self._abort(response)
            return {"cancel_requested": event is not None}
        if request["command"] == "hf_save_token":
            token = request["token"].strip()
            if not token or any(c.isspace() for c in token):
                raise DownloadError("HF token must be one nonempty token")
            path = self.controller.home / "hf-token"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(token)
            path.chmod(0o600)
            self.controller.configure_preferences({"hf_token_file": str(path)})
            return {"saved": True}
        raise DownloadError("Unknown download operation")

    def close(self):
        with self.lock:
            self.closed = True
            for event in self.cancellations.values():
                event.set()
            for responses in list(self.responses.values()):
                for response in list(responses):
                    self._abort(response)
        for thread in self.threads:
            thread.join(timeout=20)
        self.pool.shutdown(wait=True, cancel_futures=True)
