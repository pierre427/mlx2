#!/usr/bin/env python3
"""Remote EXL3 encode worker (Pierre 2026-11-09: use the M3 to help with conversions).

Holds no model data: a client (convert_bank.py --remote) sends one expert's three
(weight [I, O] f32, Hessian [I, I] f64) pairs, the worker runs exl3_encode.encode_many on
its Metal GPU (one expert at a time; transfers of the next request overlap the encode) and
returns trellis/suh/svh/weight per projection plus the encoder stats.

Wire format: 8-byte little-endian length + np.savez payload, both directions; frames larger
than MAX_FRAME are refused. Each connection opens with an auth frame carrying the shared token.
usage: exl3_worker.py [--host ADDRESS] [--port 8601]   (default host: $MLX2_EXL3_WORKER_HOST or 127.0.0.1)

Token (sweep 2026-11-09 EXL-8): set the same secret on worker and client, in MLX2_EXL3_WORKER_TOKEN
or in a file named by MLX2_EXL3_WORKER_TOKEN_FILE. The worker compares it in constant time and
drops a connection with a wrong token. Without a token it refuses to start unless --host is a
loopback address.
"""
import argparse, hmac, io, ipaddress, json, os, socket, socketserver, struct, sys, threading, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exl3_encode as E  # noqa: E402

GPU = threading.Lock()
MAX_FRAME = 2 << 30          # one DS4.1 expert request is ~0.3 GB
AUTH_FRAME = 64 << 10
AUTH_TIMEOUT_S = 10.0        # a peer that sends no auth frame cannot hold a handler thread


def load_token():
    tok = os.environ.get("MLX2_EXL3_WORKER_TOKEN", "")
    path = os.environ.get("MLX2_EXL3_WORKER_TOKEN_FILE", "")
    if not tok and path:
        tok = Path(path).read_text().strip()
    return tok


def is_loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def check_bind(host, token):
    if not token and not is_loopback(host):
        raise SystemExit(f"refusing to listen on {host} without a token: set MLX2_EXL3_WORKER_TOKEN "
                         "or MLX2_EXL3_WORKER_TOKEN_FILE (or bind a loopback address)")


def send_auth(sock, token):
    send_msg(sock, {"token": np.array(token)})


def check_auth(msg, token):
    if msg is None or "token" not in msg:
        return False
    if not token:
        return True
    return hmac.compare_digest(str(msg["token"]).encode(), token.encode())


def recv_msg(sock, limit=MAX_FRAME):
    head = _recv_exact(sock, 8)
    if not head:
        return None
    (n,) = struct.unpack("<Q", head)
    if n > limit:
        raise ConnectionError(f"frame of {n} bytes exceeds the {limit}-byte limit")
    return dict(np.load(io.BytesIO(_recv_exact(sock, n)), allow_pickle=False))


def send_msg(sock, arrays):
    buf = io.BytesIO(); np.savez(buf, **arrays); data = buf.getvalue()
    sock.sendall(struct.pack("<Q", len(data))); sock.sendall(data)


def _recv_exact(sock, n):
    chunks, got = [], 0
    while got < n:
        c = sock.recv(min(n - got, 1 << 24))
        if not c:
            if got == 0:
                return b""
            raise ConnectionError("short read")
        chunks.append(c); got += len(c)
    return b"".join(chunks)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.request.settimeout(getattr(self.server, "auth_timeout_s", AUTH_TIMEOUT_S))
        try:
            ok = check_auth(recv_msg(self.request, AUTH_FRAME), getattr(self.server, "token", ""))
        except (ConnectionError, ValueError, OSError):   # socket.timeout is an OSError
            ok = False
        if not ok:
            return  # wrong or missing token, or silent past the timeout: drop the connection
        self.request.settimeout(None)  # authenticated: requests and encodes may take long
        while True:
            msg = recv_msg(self.request)
            if msg is None:
                return
            meta = json.loads(str(msg["meta"]))
            items = [(msg["w0"], msg["h_in"]), (msg["w1"], msg["h_in"]), (msg["w2"], msg["h_mid"])]
            t0 = time.time()
            try:
                with GPU:
                    t1 = time.time()
                    enc = E.encode_many(items, meta["words"], seed=meta["seed"], probe=meta["probe"], backend="metal")
                out = {"meta": np.array(json.dumps({"ok": True, "wait_s": round(t1 - t0, 3), "encode_s": round(time.time() - t1, 3),
                                                    "host": socket.gethostname(),
                                                    "stats": [e["stats"] for e in enc]}, default=float))}
                lean = not meta.get("return_weight", True)
                checks = []
                for i, e in enumerate(enc):
                    out[f"trellis{i}"], out[f"suh{i}"], out[f"svh{i}"] = e["trellis"], e["suh"], e["svh"]
                    if lean:
                        # Slow links: check decode == encoder here instead of shipping the weight back.
                        from mlx2.runtime.models.exl3_reference import effective_weight
                        w = np.asarray(e["weight"])
                        weff = effective_weight(np.asarray(e["trellis"]).view(np.uint16), e["suh"], e["svh"], "mcg")
                        checks.append(bool(np.allclose(weff, w, rtol=0, atol=1e-6 * np.abs(w).max())))
                    else:
                        out[f"weight{i}"] = np.asarray(e["weight"], np.float32)
                if lean:
                    m = json.loads(str(out["meta"])); m["remote_decode_match"] = checks
                    out["meta"] = np.array(json.dumps(m, default=float))
            except Exception as exc:  # report, keep serving
                out = {"meta": np.array(json.dumps({"ok": False, "error": repr(exc)}))}
            send_msg(self.request, out)


class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("MLX2_EXL3_WORKER_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=8601)
    a = ap.parse_args()
    token = load_token()
    check_bind(a.host, token)
    print(f"exl3 worker on {a.host}:{a.port}" + (" (token required)" if token else " (loopback, no token)"), flush=True)
    srv = Server((a.host, a.port), Handler); srv.token = token
    srv.serve_forever()
