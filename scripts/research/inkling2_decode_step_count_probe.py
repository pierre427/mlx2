"""Count Inkling 2 served forwards per request against generated tokens (GPU).

Original mlx2 script (2026-11-23).  The served check of
``inkling2_attn_decode_kernel`` (inkattn-d006a66a4-serve) counted fewer
one-row decode forwards than generated tokens in both arms (about 107-112
for 128 tokens), inferred from per-layer kernel counters.  This probe
counts the forwards directly so the gap is either explained or confirmed.
Answer (2026-11-24 night run at 98883cc60): forwards equal generated tokens;
the gap was ``/v1/status`` staleness (the worker republishes it at most once
a second), so counters are now read with ``_fresh_status``:

* one loopback ``mlx2.server`` per arm (never port 8282, ``--max-lanes 1``,
  ``--execution-policy`` with only ``inkling2_attn_decode_kernel`` set, every
  other switch at its shipped default);
* a ``sitecustomize`` on the server's PYTHONPATH wraps
  ``mlx2.runtime.models.inkling2.language.Model.__call__`` and appends one
  JSON line per forward (input shape, whether logits were skipped, cache
  offset of the first layer when it has one) to ``forwards.jsonl``;
* per request (APCv2 cleared first): the summary prompt of the served check
  plus the two oracle prompts, greedy, ``reasoning_effort`` none; the
  request's token ids, usage, route receipt and the ``attn_decode`` /
  ``splitk`` counter deltas are stored next to the forwards logged between
  its start and end.

Per request the receipt reports forwards by row count, one-row forwards,
total rows fed, completion tokens and ``gap = completion_tokens -
one_row_forwards``, plus counter delta per one-row forward, so a
per-layer-kind normalization error shows up as a non-integer or a ratio
other than the layer count.  Sanity probe, no thermal hold.

Usage: python scripts/research/inkling2_decode_step_count_probe.py <model dir> <run dir>
       [--port 8398] [--max-tokens 128] [--arms off,on]
The caller owns the GPU (cpg_job.py lease -> scripts/run_with_gpu_locks.py).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inkling2_attn_decode_serve import ORACLES, _find, _fresh_status, _get, _post  # noqa: E402

SITECUSTOMIZE = '''
import builtins, json, os, sys, threading
_path = os.environ["MLX2_STEP_PROBE_LOG"]
_lock = threading.Lock()
_done = []


def _shape(value):
    return list(value.shape) if value is not None and hasattr(value, "shape") else None


def _wrap(cls, label):
    original = cls.__call__

    def call(self, inputs=None, cache=None, *args, **kwargs):
        source = inputs if inputs is not None else kwargs.get("input_ids")
        for key in ("inputs_embeds", "input_embeddings"):
            if source is None:
                source = kwargs.get(key)
        offset = None
        try:
            first = cache[0] if cache else None
            offset = getattr(first, "offset", None)
            if not isinstance(offset, int):
                offset = offset.tolist() if hasattr(offset, "tolist") else None
        except Exception:
            pass
        row = {"cls": label, "shape": _shape(source), "skip_logits": bool(kwargs.get("skip_logits")),
               "offset": offset, "cache_type": type(cache[0]).__name__ if cache else None}
        with _lock, open(_path, "a") as fh:
            fh.write(json.dumps(row) + "\\n")
        return original(self, inputs, cache, *args, **kwargs)

    cls.__call__ = call


_real_import = builtins.__import__


def _hook(name, *a, **k):
    module = _real_import(name, *a, **k)
    if not _done:
        language = sys.modules.get("mlx2.runtime.models.inkling2.language")
        if language is not None and hasattr(language, "Model") and hasattr(language, "TextModel"):
            _done.append(True)
            _wrap(language.Model, "Model")
            _wrap(language.TextModel, "TextModel")
            with open(_path, "a") as fh:
                fh.write(json.dumps({"hook": "installed"}) + "\\n")
    return module


builtins.__import__ = _hook
'''


def _lines(path):
    return path.read_text().splitlines() if path.exists() else []


def _summarize(rows, usage, token_ids, counters_before, counters_after):
    by_rows = collections.Counter()
    for row in rows:
        shape = row.get("shape") or []
        by_rows[shape[1] if len(shape) > 1 else None] += 1
    one_row = by_rows.get(1, 0)
    tokens = (usage or {}).get("completion_tokens")
    deltas = {}
    for key in ("attn_decode", "splitk"):
        before = dict(counters_before.get(key) or [])
        after = dict(counters_after.get(key) or [])
        for path, value in after.items():
            prior = before.get(path) or {}
            calls = (value or {}).get("calls", 0) - (prior or {}).get("calls", 0)
            declines = {r: n - (prior.get("declines") or {}).get(r, 0)
                        for r, n in ((value or {}).get("declines") or {}).items()}
            deltas[path] = {"calls": calls, "declines": declines,
                            "calls_plus_declines_per_one_row_forward":
                            None if not one_row else (calls + sum(declines.values())) / one_row}
    return {"forwards": len(rows), "forwards_by_rows": {str(k): v for k, v in by_rows.items()},
            "one_row_forwards": one_row,
            "rows_fed": sum((r.get("shape") or [0, 0])[1] for r in rows if r.get("shape")),
            "completion_tokens": tokens, "token_ids_len": len(token_ids),
            "gap": None if tokens is None else tokens - one_row,
            "first_offsets": [r.get("offset") for r in rows[:3]],
            "last_offsets": [r.get("offset") for r in rows[-3:]],
            "cache_types": sorted({str(r.get("cache_type")) for r in rows}),
            "counter_deltas": deltas}


def _counters(status):
    return {key: _find(status, key) for key in ("attn_decode", "splitk")}


def arm(model, run_dir, port, name, on, prompts, max_tokens):
    arm_dir = run_dir / name
    arm_dir.mkdir()
    hook = arm_dir / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(SITECUSTOMIZE)
    log_path = arm_dir / "forwards.jsonl"
    token = secrets.token_hex(16)
    token_file = arm_dir / "admin.token"
    token_file.write_text(token)
    os.chmod(token_file, 0o600)
    policy = arm_dir / "policy.json"
    policy.write_text(json.dumps({"inkling2_attn_decode_kernel": on}))
    log = open(arm_dir / "server.log", "w")
    cmd = [sys.executable, "-m", "mlx2.server", "--model", str(model), "--host", "127.0.0.1",
           "--port", str(port), "--max-lanes", "1", "--cache-bytes", str(8 << 30),
           "--admin-token-file", str(token_file), "--execution-policy", str(policy)]
    env = dict(os.environ, PYTHONPATH=f"{hook}:{ROOT / 'src'}", MLX2_STEP_PROBE_LOG=str(log_path))
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    result = {"command": cmd, "policy": json.loads(policy.read_text()), "requests": {}}
    try:
        deadline = time.monotonic() + 900
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"{name} server did not become healthy; see server.log")
            time.sleep(2)
        model_id = _get(port, "/v1/models")["data"][0]["id"]
        result["hook_installed"] = any('"hook"' in x for x in _lines(log_path))
        for label, prompt, pattern, limit in prompts:
            _post(port, "/v1/admin/apc/clear", {"reason": "step-count-probe"}, token)
            status, fresh_before = _fresh_status(port, token)
            before = _counters(status)
            start = len(_lines(log_path))
            body = {"model": model_id, "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0, "max_tokens": min(limit, max_tokens),
                    "reasoning_effort": "none", "logprobs": True}
            payload = _post(port, "/v1/chat/completions", body)
            logged = [json.loads(x) for x in _lines(log_path)[start:]]
            rows = [r for r in logged if r.get("cls") == "Model"]
            status, fresh_after = _fresh_status(port, token)
            after = _counters(status)
            choice = payload["choices"][0]
            message = choice.get("message") or {}
            token_ids = [int(r["id"]) for r in (choice.get("logprobs") or {}).get("content") or []]
            entry = _summarize(rows, payload.get("usage"), token_ids, before, after)
            entry["status_fresh"] = {"before": fresh_before, "after": fresh_after}
            entry["text_model_forwards"] = sum(r.get("cls") == "TextModel" for r in logged)
            entry.update(finish_reason=choice.get("finish_reason"), usage=payload.get("usage"),
                         token_ids=token_ids, route=(payload.get("mlx2") or {}).get("route"),
                         oracle_pass=None if pattern is None else bool(
                             re.search(pattern, message.get("content") or "")))
            (arm_dir / f"{label}.forwards.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in logged))
            result["requests"][label] = entry
            print(name, label, json.dumps({k: entry[k] for k in (
                "forwards_by_rows", "one_row_forwards", "completion_tokens", "gap")}), flush=True)
        result["checkpoint"] = _find(_get(port, "/v1/status", token), "checkpoint")[:4]
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        log.close()
        result["server_stopped"] = proc.poll() is not None
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--port", type=int, default=8398)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--arms", default="off,on")
    a = ap.parse_args()
    if a.port == 8282:
        raise SystemExit("port 8282 is reserved for release trees")
    if a.run_dir.exists() and any(a.run_dir.iterdir()):
        raise SystemExit(f"refusing a nonempty run directory {a.run_dir}")
    arms = [x for x in a.arms.split(",") if x]
    if not arms or set(arms) - {"off", "on"}:
        raise SystemExit("--arms takes off and/or on")
    a.run_dir.mkdir(parents=True, exist_ok=True)
    summary = ("Summarize the following document in two sentences.\n\n"
               + (ROOT / "docs/INKLING2-MINI.md").read_text()[:6000])
    prompts = [("summary", summary, None, a.max_tokens),
               *((label, text, pattern, 256) for label, text, pattern in ORACLES)]
    receipt = {"script": "scripts/research/inkling2_decode_step_count_probe.py",
               "mlx2": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
               "model_path": str(a.model), "kind": "sanity probe, no thermal hold", "arms": {}}
    for name in arms:
        receipt["arms"][name] = arm(a.model.resolve(), a.run_dir, a.port, name, name == "on",
                                    prompts, a.max_tokens)
        tmp = a.run_dir / "probe.json.tmp"
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        tmp.rename(a.run_dir / "probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
