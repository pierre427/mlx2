#!/usr/bin/env python3
"""Remote EXL3 encode worker (Pierre 2026-11-09: use the M3 to help with conversions).

Holds no model data: a client (convert_bank.py --remote) sends one expert's three
(weight [I, O] f32, Hessian [I, I] f64) pairs, the worker runs exl3_encode.encode_many on
its Metal GPU (one expert at a time; transfers of the next request overlap the encode) and
returns trellis/suh/svh/weight per projection plus the encoder stats.

Wire format: 8-byte little-endian length + np.savez payload, both directions.
usage: exl3_worker.py [--host ADDRESS] [--port 8601]   (default host: $MLX2_EXL3_WORKER_HOST or 127.0.0.1)
"""
import argparse, io, json, socket, socketserver, struct, sys, threading, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exl3_encode as E  # noqa: E402

GPU = threading.Lock()


def recv_msg(sock):
    head = _recv_exact(sock, 8)
    if not head:
        return None
    (n,) = struct.unpack("<Q", head)
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
                for i, e in enumerate(enc):
                    out[f"trellis{i}"], out[f"suh{i}"], out[f"svh{i}"] = e["trellis"], e["suh"], e["svh"]
                    out[f"weight{i}"] = np.asarray(e["weight"], np.float32)
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
    print(f"exl3 worker on {a.host}:{a.port}", flush=True)
    Server((a.host, a.port), Handler).serve_forever()
