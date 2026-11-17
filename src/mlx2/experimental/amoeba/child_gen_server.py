"""Local on-policy generation backend (child process): a loopback ``/v1/completions`` over the ordinary decode.

scripts/amoeba_onpolicy_gen.py (the generation client: mid-document prompt windows, seeds, resume,
manifest) talks to an mlx2 server. For small models and CPU runs this child is that server: the
adapter's model with its own KV cache (``model.make_cache()``), ordinary decode, greedy or
temperature sampling with the request's seed, stop at EOS. It is the ordinary-reference route, not
a tuned serving route; the stage receipt says so. Big models use ``--gen-backend server`` (an mlx2
dev server on its best route, scripts/amoeba_gen_stage.py).

Requests are served one at a time (one model, one stream). Each response reports
``usage.completion_tokens``; every few seconds a ``gen`` progress event gives tokens per second.

    python -m mlx2.experimental.amoeba.child_gen_server --model M --port P [--cpu]
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .progress import child_event


class Generator:
    def __init__(self, model_path: str, cpu: bool):
        import mlx.core as mx
        if cpu:
            mx.set_default_device(mx.cpu)
        from mlx2.adapters.registry import inspect_model
        self.adapter = inspect_model(model_path).adapter_type(model_path)
        self.model, self.tok = self.adapter.model, self.adapter.tokenizer
        self.model.eval()
        t = getattr(self.tok, "_tokenizer", None) or self.tok
        self.t = t
        eos = getattr(self.tok, "eos_token_ids", None) or getattr(self.tok, "eos_token_id", None)
        self.eos = set(eos if isinstance(eos, (list, tuple, set)) else ([eos] if eos is not None else []))
        self.lock = threading.Lock()
        self.tokens = 0
        self.busy_s = 0.0
        self.fingerprint = self.adapter.identity.get("fingerprint")

    def encode(self, text):
        try:
            out = self.t.encode(text, add_special_tokens=False)
        except TypeError:
            out = self.t.encode(text)
        return list(getattr(out, "ids", out))

    def decode(self, ids):
        return self.t.decode(list(ids))

    def complete(self, prompt: str, max_tokens: int, temperature: float, seed):
        import mlx.core as mx
        ids = self.encode(prompt)
        with self.lock:
            t0 = time.time()
            make = getattr(self.model, "make_cache", None)
            cache = make() if callable(make) else None
            logits = self.model(mx.array([ids]), cache=cache)[:, -1, :].astype(mx.float32)
            key = mx.random.key(int(seed)) if seed is not None else None
            out = []
            for _ in range(max_tokens):
                if temperature and temperature > 0:
                    key, sub = mx.random.split(key) if key is not None else (None, None)
                    nxt = mx.random.categorical(logits / temperature, key=sub)
                else:
                    nxt = mx.argmax(logits, axis=-1)
                tid = int(nxt.item())
                if tid in self.eos:
                    break
                out.append(tid)
                if cache is None:
                    logits = self.model(mx.array([ids + out]))[:, -1, :].astype(mx.float32)
                else:
                    logits = self.model(mx.array([[tid]]), cache=cache)[:, -1, :].astype(mx.float32)
            dt = time.time() - t0
            self.tokens += len(out)
            self.busy_s += dt
        return self.decode(out), len(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    gen = Generator(a.model, a.cpu)

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok", "fingerprint": gen.fingerprint, "route": "ordinary-reference"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/completions":
                self._send(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n))
                text, k = gen.complete(body["prompt"], int(body.get("max_tokens", 64)),
                                       float(body.get("temperature", 0.0)), body.get("seed"))
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
                return
            self._send(200, {"choices": [{"text": text, "index": 0}], "usage": {"completion_tokens": k}})

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), H)

    def reporter():
        last = (0, 0.0)
        while True:
            time.sleep(5)
            tok, busy = gen.tokens, gen.busy_s
            if tok != last[0]:
                child_event("gen", tok_s=round((tok - last[0]) / max(1e-9, busy - last[1]), 1), tokens_total=tok)
                last = (tok, busy)
    threading.Thread(target=reporter, daemon=True).start()
    print(f"ready on 127.0.0.1:{a.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
