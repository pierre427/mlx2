"""Explicit, default-off CPU encode worker for a pinned tokenizers v1 build.

Templates and decoding stay on the caller's tokenizer. No MLX import, fork,
production dependency change, or automatic installation is performed here.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import sys
import threading
import time

SCHEMA = "mlx2.tokenizers-v1-worker.v1"
MAX_PACKET_BYTES = 32 * 1024 * 1024


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_files(manifest):
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported v1 worker manifest")
    for field in ("python", "extension", "wheel", "tokenizer", "config", "chat_template", "cpu_receipt"):
        entry = manifest[field]
        if _sha(entry["path"]) != entry["sha256"]:
            raise ValueError(f"v1 worker {field} identity mismatch")
    if manifest.get("tokenizers_version") != "1.0.0-rc.2":
        raise ValueError("unreviewed tokenizers v1 version")
    if not isinstance(manifest.get("canaries"), list) or not manifest["canaries"]:
        raise ValueError("missing exact-ID startup canaries")
    if manifest.get("qualification") != "cpu_exact_encode_candidate":
        raise ValueError("missing CPU encode candidate qualification")


class TokenizersV1Worker:
    """One serialized encode stream with explicit spawn, bounded IO and cleanup."""

    def __init__(self, manifest_path, *, model_path):
        self.manifest_path = str(Path(manifest_path).resolve())
        self.manifest = json.loads(Path(self.manifest_path).read_text())
        _verify_files(self.manifest)
        expected = Path(self.manifest["tokenizer"]["path"]).resolve().parent
        if Path(model_path).resolve() != expected:
            raise ValueError("v1 worker model path mismatch")
        self.minimum_chars = int(self.manifest.get("minimum_chars", 8192))
        self.timeout = float(self.manifest.get("timeout_seconds", 5.0))
        if self.minimum_chars < 0 or not 0 < self.timeout <= 60:
            raise ValueError("invalid v1 worker policy")
        self._lock = threading.RLock()
        self._pid = None
        self._input = self._output = None
        self._buffer = bytearray()
        self._closed = False
        self.counts = {"spawns": 0, "successful_encodes": 0, "encoded_tokens": 0,
                       "failures": 0, "restarts": 0, "ordinary_encodes": 0}
        self.startup_seconds = self.encode_seconds = 0.0
        atexit.register(self.close)

    def eligible(self, text, args, kwargs):
        return (isinstance(text, str) and len(text) >= self.minimum_chars and not args
                and set(kwargs) <= {"add_special_tokens"}
                and isinstance(kwargs.get("add_special_tokens", True), bool))

    def _packet(self, payload=None):
        deadline = time.monotonic() + self.timeout
        pending = memoryview(json.dumps(payload, ensure_ascii=False).encode() + b"\n") if payload is not None else memoryview(b"")
        if len(pending) > MAX_PACKET_BYTES:
            raise ValueError("v1 worker request exceeds packet bound")
        while True:
            if not pending and b"\n" in self._buffer:
                line, _, remaining = self._buffer.partition(b"\n")
                self._buffer = bytearray(remaining)
                return json.loads(line)
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("v1 worker IO timed out")
            readable, writable, _ = select.select([self._output], [self._input] if pending else [], [], left)
            if writable:
                pending = pending[os.write(self._input, pending):]
            if readable:
                chunk = os.read(self._output, 65536)
                if not chunk:
                    raise RuntimeError("v1 worker exited before response")
                self._buffer.extend(chunk)
                if len(self._buffer) > MAX_PACKET_BYTES:
                    raise ValueError("v1 worker response exceeds packet bound")

    def _start(self):
        if self._closed:
            raise RuntimeError("v1 worker is closed")
        if self._pid is not None:
            return
        if not hasattr(os, "posix_spawn"):
            raise RuntimeError("v1 worker requires posix_spawn; fork is forbidden")
        _verify_files(self.manifest)
        child_read, parent_write = os.pipe()
        parent_read, child_write = os.pipe()
        null = os.open(os.devnull, os.O_WRONLY)
        actions = [(os.POSIX_SPAWN_DUP2, child_read, 0), (os.POSIX_SPAWN_DUP2, child_write, 1),
                   (os.POSIX_SPAWN_DUP2, null, 2)]
        actions += [(os.POSIX_SPAWN_CLOSE, fd) for fd in (child_read, parent_write, parent_read, child_write, null) if fd > 2]
        executable = self.manifest["python"]["path"]
        argv = [executable, "-I", str(Path(__file__).resolve()), "--worker", self.manifest_path]
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", USE_TORCH="0", USE_TF="0", USE_FLAX="0")
        start = time.monotonic()
        try:
            self._pid = os.posix_spawn(executable, argv, env, file_actions=actions)
        except BaseException:
            os.close(parent_write)
            os.close(parent_read)
            raise
        finally:
            os.close(child_read)
            os.close(child_write)
            os.close(null)
        self._input, self._output = parent_write, parent_read
        os.set_blocking(self._input, False)
        os.set_blocking(self._output, False)
        self.counts["spawns"] += 1
        try:
            ready = self._packet()
            if ready.get("ready") is not True or ready.get("extension_sha256") != self.manifest["extension"]["sha256"]:
                raise RuntimeError(f"v1 worker startup refused: {ready}")
        except BaseException:
            self._stop()
            raise
        self._rss_bytes = ready.get("peak_rss_bytes")
        self.startup_seconds += time.monotonic() - start

    def start(self):
        """Warm the worker during model setup, outside request timing."""
        with self._lock:
            self._start()

    def encode(self, text, *, add_special_tokens=True):
        with self._lock:
            start = time.monotonic()
            try:
                self._start()
                packet = self._packet({"text": text, "add_special_tokens": add_special_tokens})
                ids = packet.get("ids")
                if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
                    raise RuntimeError(f"v1 worker invalid response: {packet.get('error', 'invalid IDs')}")
                self._rss_bytes = packet.get("peak_rss_bytes")
                self.counts["successful_encodes"] += 1
                self.counts["encoded_tokens"] += len(ids)
                return ids
            except BaseException:
                self.counts["failures"] += 1
                self._stop()
                raise
            finally:
                self.encode_seconds += time.monotonic() - start

    def _stop(self):
        pid, self._pid = self._pid, None
        for attr in ("_input", "_output"):
            fd = getattr(self, attr)
            if fd is not None:
                os.close(fd)
                setattr(self, attr, None)
        self._buffer.clear()
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    waited, _ = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    return
                if waited:
                    return
                time.sleep(.01)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass

    def restart(self):
        with self._lock:
            self._stop()
            self._closed = False
            atexit.unregister(self.close)
            atexit.register(self.close)
            self.counts["restarts"] += 1
            self._start()

    def close(self):
        with self._lock:
            self._stop()
            self._closed = True
            atexit.unregister(self.close)

    def record_ordinary(self):
        with self._lock:
            self.counts["ordinary_encodes"] += 1

    def status(self):
        with self._lock:
            return {"implemented": True, "qualified": "cpu_exact_encode_candidate",
                    "serving_qualified": False, "selected": True,
                    "observed_used": self.counts["successful_encodes"] > 0,
                    "manifest_path": self.manifest_path, "pid": self._pid,
                    "source_revision": self.manifest["source_revision"],
                    "source_patch_sha256": self.manifest["source_patch_sha256"],
                    "extension_sha256": self.manifest["extension"]["sha256"],
                    "wheel_sha256": self.manifest["wheel"]["sha256"],
                    "tokenizer_sha256": self.manifest["tokenizer"]["sha256"],
                    "minimum_chars": self.minimum_chars, "closed": self._closed,
                    "startup_seconds": self.startup_seconds, "encode_seconds": self.encode_seconds,
                    "worker_peak_rss_bytes": getattr(self, "_rss_bytes", None),
                    "counts": dict(self.counts)}


def _worker(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    _verify_files(manifest)
    import resource
    import tokenizers
    from tokenizers import Tokenizer
    extension = Path(tokenizers.__file__).parent / "tokenizers.abi3.so"
    if tokenizers.__version__ != manifest["tokenizers_version"] or _sha(extension) != manifest["extension"]["sha256"]:
        raise ValueError("installed tokenizers build identity mismatch")
    tokenizer = Tokenizer.from_file(manifest["tokenizer"]["path"])
    for canary in manifest["canaries"]:
        ids = tokenizer.encode(canary["text"], add_special_tokens=canary["add_special_tokens"]).ids
        if ids != canary["ids"]:
            raise ValueError("v1 worker startup canary token IDs differ")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({"ready": True, "extension_sha256": _sha(extension),
                      "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024}), flush=True)
    for line in sys.stdin:
        try:
            if len(line.encode()) > MAX_PACKET_BYTES:
                raise ValueError("request exceeds packet bound")
            packet = json.loads(line)
            ids = tokenizer.encode(packet["text"], add_special_tokens=packet["add_special_tokens"]).ids
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print(json.dumps({"ids": ids, "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024}), flush=True)
        except Exception as error:
            print(json.dumps({"error": str(error)}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("internal worker requires --worker MANIFEST")
    _worker(sys.argv[2])
