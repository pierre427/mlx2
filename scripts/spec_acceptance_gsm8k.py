#!/usr/bin/env python3
"""GSM8K speculative-acceptance and accuracy harness (project standard).

Measures, per arm, on the GSM8K test split served through an mlx2 server:

* accuracy and invalid rate of the final numeric answer;
* speculative acceptance in vLLM's terms, from each request's speculation
  receipt (``verify_accept_hist``): mean acceptance length
  ``1 + accepted / verify_rounds`` (tokens committed per verify step, bonus
  included) and per-position acceptance ``p_k`` = share of verify rounds that
  accepted at least ``k`` drafted tokens;
* decode tok/s, completion tokens, and the route each request used.

Prompt format follows vLLM's ``tests/evals/gsm8k/gsm8k_eval.py`` completion
mode: ``--shots`` (default 5) worked examples from the start of the train
split as ``Question: ...\\nAnswer: ...`` blocks (calculator annotations
``<<...>>`` removed), then the test question; greedy; stops at the next
``Question:``.  The answer is the number after ``####``, else the last number.
``--chat`` sends the question as a chat message instead (reasoning content is
not scored).

Data: ``benchmarks/gsm8k/`` (openai/gsm8k ``main``, MIT licence): the full
test split and the first 8 train rows, converted unmodified from the pinned
parquet revision named in ``SOURCE.json``.  The receipt records the files'
sha256.  Nothing is downloaded.

Two modes:

* ``--base-url http://127.0.0.1:PORT --checkpoint NAME@sha256:FP`` measures a
  server the caller already runs (and owns the GPU for).
* ``--model PATH --arms arms.json --cpg-lock DIR --output DIR`` starts one
  loopback server per arm (``[{"name", "tree", "args", "env"}]``, as
  ``scripts/research/ripple_ab.py``), one at a time, under the caller's CPG
  lease and paired locks, and stops each server's process group in
  ``finally``.  Refuses a nonempty output directory.

Sanity and attribution numbers, no thermal hold: not publishable statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mlx2.spec-acceptance-gsm8k.v1"
DATASET = "openai/gsm8k"
DATA = ROOT / "benchmarks" / "gsm8k"
STOPS = ["Question:", "</s>", "<|im_end|>"]
_ANNOTATION = re.compile(r"<<[^>]*>>")
# A bare leading-dot decimal (".5") is one number, not the integer after it.
_NUMBER = re.compile(r"-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")


# ----------------------------------------------------------------- data --

def dataset_files(data=DATA):
    """``(source revision, {split: JSONL path})`` of the vendored GSM8K copy."""
    source = json.loads((Path(data) / "SOURCE.json").read_text())
    files = {"train": Path(data) / "train_few_shot.jsonl", "test": Path(data) / "test.jsonl"}
    for split, path in files.items():
        if sha256(path) != source["splits"][split]["output_sha256"]:
            raise ValueError(f"{path} does not match benchmarks/gsm8k/SOURCE.json")
    return source["revision"], files


def load_split(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def reference_answer(solution):
    return normalize(solution.split("####")[-1])


def normalize(text):
    if text is None:
        return None
    match = _NUMBER.search(text.replace("$", ""))
    if not match:
        return None
    value = match.group(0).replace(",", "").rstrip(".")
    if value.startswith("."):
        value = "0" + value
    elif value.startswith("-."):
        value = "-0" + value[1:]
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    if value == "-0":
        value = "0"
    return value.lstrip("+") or None


def extract_answer(text):
    """Number after the first ``####``, else the last number in the text."""
    if "####" in text:
        found = normalize(text.split("####", 1)[1])
        if found is not None:
            return found
    numbers = _NUMBER.findall(text.replace("$", ""))
    return normalize(numbers[-1]) if numbers else None


def few_shot_prefix(train, shots):
    blocks = []
    for item in train[:shots]:
        solution = _ANNOTATION.sub("", item["answer"]).strip()
        blocks.append(f"Question: {item['question']}\nAnswer: {solution}\n\n")
    return "".join(blocks)


# ------------------------------------------------------------ acceptance --

def acceptance_summary(receipts, max_position=16, draft_depth=None):
    """vLLM-style acceptance from per-request speculation receipts.

    Each receipt's ``verify_accept_hist`` maps accepted drafted tokens per
    verify round to a round count.  Requests without speculation rounds are
    counted in ``requests_without_rounds`` and contribute nothing.

    ``per_position`` runs to ``draft_depth`` when the caller knows it (routes
    share no depth field in their receipts), so never-accepted depths show 0
    instead of vanishing; otherwise it stops at the deepest accepted depth,
    and ``per_position_basis`` says which.
    """
    hist, proposed, without = {}, 0, 0
    for spec in receipts:
        rounds = (spec or {}).get("verify_accept_hist") or {}
        if not rounds:
            without += 1
            continue
        for accepted, count in rounds.items():
            hist[int(accepted)] = hist.get(int(accepted), 0) + int(count)
        proposed += int((spec or {}).get("proposed") or 0)
    total = sum(hist.values())
    if not total:
        return {"verify_rounds": 0, "requests_without_rounds": without}
    accepted = sum(k * v for k, v in hist.items())
    deepest = max(max(hist), 1)
    if draft_depth is not None and int(draft_depth) < deepest:
        raise ValueError(f"draft depth {draft_depth} is below an accepted depth of {deepest}")
    depth = int(draft_depth) if draft_depth is not None else deepest
    positions = {f"p{k}": sum(v for a, v in hist.items() if a >= k) / total
                 for k in range(1, min(depth, max_position) + 1)}
    return {"verify_rounds": total, "accepted_tokens": accepted, "proposed_tokens": proposed,
            "mean_acceptance_length": 1 + accepted / total,
            "draft_acceptance_rate": accepted / proposed if proposed else None,
            "per_position": positions,
            "per_position_basis": "draft_depth" if draft_depth is not None else "deepest_accepted",
            "accept_hist": {str(k): hist[k] for k in sorted(hist)},
            "requests_without_rounds": without}


def speculation_receipt(extra):
    """The request's verify-round receipt, whichever route produced it.

    External-draft and prompt-lookup lanes report it as ``speculation``
    (``verify_accept_hist`` plus ``proposed``).  Native self-MTP lanes leave
    ``speculation`` null and report the same histogram in ``mtp.stats``, with
    the proposal count as ``draft_proposed``; reading only ``speculation``
    scored every native-MTP request as having no verify rounds.
    """
    extra = extra if isinstance(extra, dict) else {}
    spec = extra.get("speculation")
    if isinstance(spec, dict) and spec.get("verify_accept_hist"):
        return spec
    mtp = extra.get("mtp")
    stats = mtp.get("stats") if isinstance(mtp, dict) else None
    if isinstance(stats, dict) and stats.get("verify_accept_hist"):
        return {"source": "mtp.stats", "verify_accept_hist": stats["verify_accept_hist"],
                "proposed": stats.get("draft_proposed")}
    return spec


def find(document, key):
    """First value of ``key`` anywhere in a JSON document."""
    if isinstance(document, dict):
        if key in document:
            return document[key]
        document = list(document.values())
    if isinstance(document, list):
        for value in document:
            found = find(value, key)
            if found is not None:
                return found
    return None


def median(values):
    values = sorted(v for v in values if v is not None)
    return values[len(values) // 2] if values else None


# --------------------------------------------------------------- serving --

class Client:
    def __init__(self, root, token=None):
        self.root, self.token = root.rstrip("/"), token

    def call(self, path, payload=None, *, admin=False, timeout=1800):
        headers = {"Content-Type": "application/json"}
        if admin and self.token:
            headers["Authorization"] = "Bearer " + self.token
        data = None if payload is None else json.dumps(payload).encode()
        with urlopen(Request(self.root + path, data=data, headers=headers), timeout=timeout) as response:
            body = response.read()
        return json.loads(body) if body.strip().startswith(b"{") else body.decode()


def run_questions(client, model_name, prefix, items, *, chat, max_tokens, log=None):
    rows = []
    for index, item in enumerate(items):
        start = time.monotonic()
        if chat:
            payload = {"model": model_name, "temperature": 0, "stream": False, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": item["question"]
                                     + "\nGive the final numeric answer after ####."}]}
            result = client.call("/v1/chat/completions", payload)
            choice = result["choices"][0]
            text = choice["message"].get("content") or ""
        else:
            payload = {"model": model_name, "temperature": 0, "stream": False, "max_tokens": max_tokens,
                       "stop": STOPS, "prompt": prefix + f"Question: {item['question']}\nAnswer:"}
            result = client.call("/v1/completions", payload)
            choice = result["choices"][0]
            text = choice.get("text") or ""
        elapsed = time.monotonic() - start
        usage = result.get("usage") or {}
        extra = result.get("mlx2") or {}
        ttft = find(result, "ttft_seconds")
        completion = usage.get("completion_tokens") or 0
        predicted, expected = extract_answer(text), reference_answer(item["answer"])
        row = {"index": index, "expected": expected, "predicted": predicted,
               "correct": predicted is not None and predicted == expected, "invalid": predicted is None,
               "finish_reason": choice.get("finish_reason"), "completion_tokens": completion,
               "prompt_tokens": usage.get("prompt_tokens"),
               "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "elapsed_seconds": elapsed, "ttft_seconds": ttft,
               "decode_tps": ((completion - 1) / (elapsed - ttft)
                              if ttft and completion > 1 and elapsed > ttft else None),
               "route": extra.get("route"), "speculation": speculation_receipt(extra), "text": text}
        rows.append(row)
        if log:
            log(row)
    return rows


def summarize(rows, draft_depth=None):
    count = len(rows)
    return {"questions": count,
            "accuracy": sum(r["correct"] for r in rows) / count if count else None,
            "invalid_rate": sum(r["invalid"] for r in rows) / count if count else None,
            "length_finishes": sum(r["finish_reason"] == "length" for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "decode_tps_median": median(r["decode_tps"] for r in rows),
            "routes": sorted({json.dumps(r["route"], sort_keys=True) for r in rows if r["route"]}),
            "acceptance": acceptance_summary((r["speculation"] for r in rows), draft_depth=draft_depth)}


def revision(tree):
    return {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=tree).strip(),
            "dirty_tracked": bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], text=True, cwd=tree).strip())}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path, document):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(document, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def compact(row):
    keep = {k: row.get(k) for k in ("index", "expected", "predicted", "correct", "completion_tokens",
                                     "decode_tps")}
    spec = row.get("speculation") or {}
    keep["accept_hist"] = spec.get("verify_accept_hist")
    return keep


# ------------------------------------------------------------------ main --

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--checkpoint", help="NAME@sha256:FINGERPRINT (base-url mode)")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--arms", type=Path)
    parser.add_argument("--cpg-lock", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=18297)
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--draft-depth", type=int,
                        help="route draft depth: per_position then covers every depth (arms JSON may set draft_depth)")
    args = parser.parse_args(argv)
    if bool(args.base_url) == bool(args.model):
        parser.error("give exactly one of --base-url or --model")
    if args.base_url and not args.checkpoint:
        parser.error("--base-url needs --checkpoint NAME@sha256:FINGERPRINT")
    if args.model and not (args.arms and args.cpg_lock):
        parser.error("--model needs --arms and --cpg-lock")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("refusing nonempty evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)

    data_revision, files = dataset_files()
    train, test = load_split(files["train"]), load_split(files["test"])
    items = test[args.offset:args.offset + args.num_questions]
    prefix = "" if args.chat else few_shot_prefix(train, args.shots)
    receipt = {"schema": SCHEMA, "qualified": False,
               "kind": "GSM8K acceptance and accuracy, greedy, no thermal hold",
               "dataset": {"name": DATASET, "revision": data_revision,
                           "files": {k: {"path": str(v), "sha256": sha256(v)} for k, v in files.items()},
                           "split": "test", "offset": args.offset, "questions": len(items)},
               "prompt": {"mode": "chat" if args.chat else f"completion {args.shots}-shot",
                          "max_tokens": args.max_tokens, "stops": None if args.chat else STOPS,
                          "temperature": 0},
               "driver_revision": revision(ROOT), "arms": []}
    path = args.output / "receipt.json"

    def progress(name):
        def log(row):
            if (row["index"] + 1) % 10 == 0:
                print(json.dumps({"arm": name, "done": row["index"] + 1}), flush=True)
        return log

    if args.base_url:
        receipt["checkpoint"] = args.checkpoint
        client = Client(args.base_url)
        status = client.call("/v1/status", timeout=30)
        rows = run_questions(client, args.checkpoint.split("@")[0], prefix, items, chat=args.chat,
                             max_tokens=args.max_tokens, log=progress("server"))
        arm = {"name": "server", "base_url": args.base_url, "startup_status_route": status.get("route_receipt"),
               "summary": summarize(rows, args.draft_depth), "rows": [compact(r) for r in rows]}
        receipt["arms"].append(arm)
        write_atomic(path, receipt)
        print(json.dumps({"arm": "server", **arm["summary"]}), flush=True)
        return 0

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    from qualification_gpu_ownership import require_qualification_lease
    from mlx2.adapters.registry import inspect_model

    owner = json.loads((args.cpg_lock / "owner.json").read_text())
    receipt["ownership"] = require_qualification_lease(
        task_id=owner["cpg_task"], generation=owner["cpg_generation"], cpg_owner_lock=args.cpg_lock)
    resolution = inspect_model(args.model)
    receipt["checkpoint"] = f"{args.model.name}@sha256:{resolution.artifact_fingerprint}"
    receipt["adapter"] = resolution.adapter_type.__name__
    receipt["swap_before"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
    token_file = args.output / "admin-token"
    token_file.write_text(os.urandom(24).hex())
    token_file.chmod(0o600)
    client = Client(f"http://127.0.0.1:{args.port}", token_file.read_text())
    try:
        for arm in json.loads(args.arms.read_text()):
            out = args.output / arm["name"]
            out.mkdir()
            tree = Path(arm["tree"]).resolve()
            command = [args.python, "-u", "-m", "mlx2.server", "--model", str(args.model),
                       "--host", "127.0.0.1", "--port", str(args.port), "--max-inflight", "1",
                       "--max-context", str(args.max_context), "--cache-bytes", str(2 << 30),
                       "--cache-dir", str(out / "cache"), "--admin-token-file", str(token_file),
                       *arm.get("args", [])]
            if "--max-lanes" not in arm.get("args", []):
                command += ["--max-lanes", "1"]
            log = (out / "server.log").open("w")
            env = {**os.environ, "PYTHONPATH": str(tree / "src"), **arm.get("env", {})}
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, env=env, cwd=tree)
            row = {"name": arm["name"], "tree": str(tree), "revision": revision(tree),
                   "args": arm.get("args", []), "env": arm.get("env", {}), "command": command}
            try:
                deadline = time.monotonic() + 900
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("server exited before readiness")
                    try:
                        status = client.call("/v1/status", timeout=10)
                        if status.get("healthy"):
                            break
                    except (URLError, TimeoutError, ConnectionError, OSError):
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError("server readiness timed out")
                    time.sleep(.5)
                row["startup_status"] = status
                rows = run_questions(client, args.model.name, prefix, items, chat=args.chat,
                                     max_tokens=args.max_tokens, log=progress(arm["name"]))
                time.sleep(1.5)
                row["final_status"] = client.call("/v1/status")
                (out / "metrics.prom").write_text(client.call("/metrics"))
                (out / "rows.json").write_text(json.dumps(rows, indent=1, default=str) + "\n")
                row["summary"] = summarize(rows, arm.get("draft_depth", args.draft_depth))
                row["rows"] = [compact(r) for r in rows]
                row["passed"] = True
            except BaseException as error:
                row["passed"] = False
                row["error"] = repr(error)
                raise
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait(timeout=10)
                log.close()
                row["server_stopped"] = server.poll() is not None
                receipt["arms"].append(row)
                receipt["swap_after"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"],
                                                                text=True).strip()
                write_atomic(path, receipt)
            print(json.dumps({"arm": row["name"], **{k: v for k, v in row["summary"].items()
                                                      if k != "routes"}}), flush=True)
    finally:
        token_file.unlink(missing_ok=True)
    return 0 if all(arm["passed"] for arm in receipt["arms"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
