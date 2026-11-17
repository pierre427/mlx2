"""CPU test of amoeba_onpolicy_gen.py --window-tokens against a stub completion server (FN calibration c)."""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path.home() / "mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP"


class Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = json.dumps({"choices": [{"text": " CONT" + str(body.get("seed"))}], "usage": {"completion_tokens": 3}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.mark.skipif(not (MODEL / "tokenizer_config.json").exists(), reason="FN tokenizer not on disk")
def test_window_prompts(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    pre = "You are a careful reasoner. Answer the question below.\n"
    rows = [{"text": pre + " ".join(f"word{i}_{j}" for j in range(400)), "domain": "logic", "source": "s"} for i in range(5)]
    c = tmp_path / "c.jsonl"
    c.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "g"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_onpolicy_gen.py"), "--base-url",
                        f"http://127.0.0.1:{srv.server_port}", "--checkpoint", "x@y", "--corpus", str(c), "--out", str(out),
                        "--window-tokens", "96", "--tokenizer", str(MODEL), "--strip-prefix", "0.2", "--samples", "3"],
                       capture_output=True, text=True, cwd=ROOT, env={"HF_HUB_OFFLINE": "1", "PATH": "/usr/bin:/bin"})
    srv.shutdown()
    assert r.returncode == 0, r.stderr
    g = [json.loads(x) for x in (out / "gen.jsonl").read_text().splitlines()]
    assert len(g) == 20 and {x["mode"] for x in g} == {"greedy", "t0.7"}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    for x in g:
        assert x["text"].startswith(x["prompt"]) and "careful reasoner" not in x["prompt"]
        assert abs(len(tok.encode(x["prompt"], add_special_tokens=False)) - 96) <= 2
        assert x["doc_id"] and x["prompt_end_token"] >= 96
    ends = {x["doc_id"]: x["prompt_end_token"] for x in g}
    assert len(set(ends.values())) > 1                       # offsets differ across documents (seeded)
    assert json.loads((out / "manifest.json").read_text())["window_tokens"] == 96
    # resumable: a rerun over a partial file skips the rows it already has
    (out / "gen.jsonl").unlink()
    srv2 = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    r2 = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_onpolicy_gen.py"), "--base-url",
                         f"http://127.0.0.1:{srv2.server_port}", "--checkpoint", "x@y", "--corpus", str(c), "--out",
                         str(out), "--window-tokens", "96", "--tokenizer", str(MODEL), "--strip-prefix", "0.2",
                         "--samples", "3"], capture_output=True, text=True, cwd=ROOT,
                        env={"HF_HUB_OFFLINE": "1", "PATH": "/usr/bin:/bin"})
    srv2.shutdown()
    assert r2.returncode == 0 and "already done 20" in r2.stdout, r2.stdout + r2.stderr
    assert len((out / "gen.jsonl").read_text().split("\n")) - 1 == 20
