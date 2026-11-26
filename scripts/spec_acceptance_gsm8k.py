#!/usr/bin/env python3
"""GSM8K speculative-acceptance and accuracy harness (project standard).

Measures, per arm, on the GSM8K test split served through an mlx2 server:

* accuracy and invalid rate of the final numeric answer;
* speculative acceptance in vLLM's terms, from each request's speculation
  receipt (``verify_accept_hist``): mean acceptance length
  ``1 + accepted / verify_rounds`` (tokens committed per verify step, bonus
  included) and per-position acceptance ``p_k`` = share of the verify rounds
  that drafted position ``k`` which accepted at least ``k`` drafted tokens
  (``verify_depth_hist``; receipts before 2026-11-18 lack it and fall back
  to all rounds, labelled in ``per_position_denominator``);
* decode tok/s, completion tokens, and the route each request used.

Prompt formats (``--format``):

* ``chat-fewshot`` (default since 2026-11-15): the ``--shots`` (default 5)
  worked examples from the start of the train split as alternating chat turns
  (user question, assistant solution ending ``#### N``; calculator annotations
  ``<<...>>`` removed), then the test question as the last user turn, through
  the model's own chat template.  Chat-trained models (e.g. Qwen3.8-27B) break
  on the raw completion format: they emit end-of-turn at once or misread the
  few-shot block, which voided several 2026-11-14/15 runs.
* ``chat-fewshot-annotated``: ``chat-fewshot`` with the train solutions'
  ``<<expression=result>>`` calculator annotations kept and every user turn
  asking for them (for a calculator triggered by the model's own marker).
* ``chat-annotated``: zero-shot; the question in prose and one instruction that
  describes the ``<<expression=result>>`` calculator format in words (one inline
  syntax example, no worked shots), to see whether the model converts prose into
  the marker format on its own (Pierre 2026-11-22).
* ``chat``: the test question alone as one chat message (zero-shot); this is
  what ``--chat`` selected before.
* ``completion``: vLLM's ``tests/evals/gsm8k/gsm8k_eval.py`` completion mode,
  ``Question: ...\\nAnswer: ...`` blocks then the test question, stopping at
  the next ``Question:``.  Kept for base models and for comparison with
  receipts written before 2026-11-15 (all of which used it unless they say
  ``chat``).  Numbers from different formats are not comparable; the receipt
  records the format.

The answer is the number after ``####``, else the last number.  Reasoning
content is not scored.

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


CHAT_INSTRUCTION = "\nGive the final numeric answer after ####."
# chat-fewshot-annotated: the shots keep GSM8K's calculator annotations and every user turn asks for
# them, so a calculator triggered by the model's own "<<expression=" marker can be measured.
ANNOTATED_INSTRUCTION = ("\nWrite each calculation as <<expression=result>> followed by the result, as in "
                         "the examples. Give the final numeric answer after ####.")

# chat-annotated: no worked examples; the format is described in words with one inline syntax example.
PROSE_MARKER_INSTRUCTION = ("\nWork through the problem step by step in plain sentences. Whenever you do a "
                            "calculation, write it as <<expression=result>>, with the arithmetic expression on "
                            "the left of the = sign and its value on the right, for example <<3*4=12>>, then "
                            "continue the sentence. Give the final numeric answer after ####.")


def few_shot_turns(train, shots, annotated=False):
    """The same worked examples as ``few_shot_prefix``, as chat turns (``annotated``: ``<<...>>`` kept)."""
    turns = []
    for item in train[:shots]:
        turns.append({"role": "user", "content": item["question"] + (ANNOTATED_INSTRUCTION if annotated else CHAT_INSTRUCTION)})
        turns.append({"role": "assistant", "content": (item["answer"] if annotated
                                                       else _ANNOTATION.sub("", item["answer"])).strip()})
    return turns


# ------------------------------------------------------------ acceptance --

def _histogram(value):
    """A ``{value: rounds}`` histogram from a dict or an index-is-value list.

    Most routes write dicts; Hy4-flash DFly2 writes lists indexed by value.
    """
    if isinstance(value, (list, tuple)):
        return {i: int(n) for i, n in enumerate(value) if int(n)}
    return {int(k): int(n) for k, n in (value or {}).items()}


def acceptance_summary(receipts, max_position=16, draft_depth=None):
    """vLLM-style acceptance from per-request speculation receipts.

    Each receipt's ``verify_accept_hist`` maps accepted drafted tokens per
    verify round to a round count.  Requests without speculation rounds are
    counted in ``requests_without_rounds`` and contribute nothing.

    ``per_position`` runs to ``draft_depth`` when the caller knows it, so
    never-accepted depths show 0 instead of vanishing; otherwise it stops at
    the deepest drafted (or, for old receipts, accepted) depth, and
    ``per_position_basis`` says which.

    Denominator (``per_position_denominator``).  ``p_k`` is accepted at
    position k over the rounds that drafted position k, as vLLM defines it
    (vllm#67610).  Receipts carry that count as ``verify_depth_hist``
    (deepest drafted position per round).  A round that drafted fewer than k
    tokens -- a parked or plain round, a shallow adaptive tile, a short
    prompt-lookup copy -- cannot accept position k, so dividing by every
    verify round reads deeper positions low.  Receipts written before
    2026-11-18 have no ``verify_depth_hist``; for them, or when any receipt
    in the arm lacks it, ``per_position`` keeps the old all-rounds
    denominator and says so.  ``per_position_per_round`` always holds the
    all-rounds figures, for comparison with older summaries.

    ``full_accept_fraction``: rounds that accepted their whole drafted depth
    over rounds that drafted anything (DBloom: mean acceptance length hides
    speedup stranded at the draft ceiling).  ``None`` when any receipt with
    rounds predates ``verify_full_accept_rounds`` (2026-11-19) or lacks a
    depth histogram.
    """
    hist, depths, proposed, without, legacy = {}, {}, 0, 0, 0
    full_rounds, full_missing = 0, 0
    for spec in receipts:
        rounds = _histogram((spec or {}).get("verify_accept_hist"))
        if not rounds:
            without += 1
            continue
        for accepted, count in rounds.items():
            hist[accepted] = hist.get(accepted, 0) + count
        drafted = _histogram((spec or {}).get("verify_depth_hist"))
        if sum(drafted.values()) != sum(rounds.values()):
            # Missing (old receipt) or not round-for-round with the accepts.
            legacy += 1
        else:
            for depth, count in drafted.items():
                depths[depth] = depths.get(depth, 0) + count
        proposed += int((spec or {}).get("proposed") or 0)
        full = (spec or {}).get("verify_full_accept_rounds")
        if full is None:
            full_missing += 1
        else:
            full_rounds += int(full)
    total = sum(hist.values())
    if not total:
        return {"verify_rounds": 0, "requests_without_rounds": without}
    accepted = sum(k * v for k, v in hist.items())
    deepest = max(max(hist), 1)
    use_drafted = not legacy and bool(depths)
    if use_drafted:
        deepest = max(deepest, max(depths))
    if draft_depth is not None and int(draft_depth) < deepest:
        kind = "a drafted" if use_drafted else "an accepted"
        raise ValueError(f"draft depth {draft_depth} is below {kind} depth of {deepest}")
    depth = int(draft_depth) if draft_depth is not None else deepest
    span = range(1, min(depth, max_position) + 1)

    def reached(k, table):
        return sum(v for a, v in table.items() if a >= k)

    per_round = {f"p{k}": reached(k, hist) / total for k in span}
    if use_drafted:
        for k in span:
            if reached(k, hist) > reached(k, depths):
                raise ValueError(f"more rounds accepted position {k} than drafted it")
        positions = {f"p{k}": (reached(k, hist) / reached(k, depths) if reached(k, depths) else None)
                     for k in span}
        denominator = "rounds_drafting_position"
    else:
        positions = per_round
        denominator = ("verify_rounds (receipts without verify_depth_hist: "
                       "under-reports positions some rounds did not draft)")
    return {"verify_rounds": total, "accepted_tokens": accepted, "proposed_tokens": proposed,
            "mean_acceptance_length": 1 + accepted / total,
            "draft_acceptance_rate": accepted / proposed if proposed else None,
            "per_position": positions,
            "per_position_basis": "draft_depth" if draft_depth is not None else (
                "deepest_drafted" if use_drafted else "deepest_accepted"),
            "per_position_denominator": denominator,
            "per_position_per_round": per_round,
            "drafted_at_position": ({f"p{k}": reached(k, depths) for k in span} if use_drafted else None),
            "full_accept_rounds": None if full_missing else full_rounds,
            "full_accept_fraction": (
                full_rounds / reached(1, depths)
                if use_drafted and not full_missing and reached(1, depths) else None
            ),
            "accept_hist": {str(k): hist[k] for k in sorted(hist)},
            "depth_hist": {str(k): depths[k] for k in sorted(depths)} if use_drafted else None,
            "requests_without_depth_hist": legacy,
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
    if isinstance(stats, dict) and stats.get("verify_accept_hist") is not None:
        return {"source": "mtp.stats", "verify_accept_hist": stats["verify_accept_hist"],
                "verify_depth_hist": stats.get("verify_depth_hist"),
                "verify_full_accept_rounds": stats.get("verify_full_accept_rounds"),
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

# Sampled runs: question with GSM8K test index i is sent seed + i.  Receipts
# without ``prompt.sampling.seed_law`` (before 2026-11-21) used seed + position
# in that run, which equals this law only for --offset 0 runs without --indices.
SEED_LAW = "seed_plus_test_index"


def sampling_fields(sampling, index):
    """Request fields for one question: empty (greedy) or the sampled settings
    with a deterministic per-question seed, so two arms draw from the same
    seeds question by question."""
    if not sampling:
        return {}
    fields = {"temperature": sampling["temperature"]}
    for name in ("top_p", "top_k"):
        if sampling.get(name) is not None:
            fields[name] = sampling[name]
    if sampling.get("seed") is not None:
        fields["seed"] = sampling["seed"] + index
    return fields


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


STOP_AT_HORIZON_H = 16


def stop_time(clock, now=None):
    """Epoch seconds of the next local ``HH:MM`` after ``now`` (``None`` for no clock).

    A clock more than ``STOP_AT_HORIZON_H`` hours ahead has just passed (a night job started at
    06:45 with ``--stop-at 06:30``): the deadline is ``now``, so nothing starts, rather than
    tomorrow's 06:30.
    """
    if not clock:
        return None
    import datetime as dt
    start = time.time() if now is None else now
    now = dt.datetime.fromtimestamp(start)
    hour, minute = (int(x) for x in clock.split(":"))
    at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if at <= now:
        at += dt.timedelta(days=1)
    return start if (at - now).total_seconds() > STOP_AT_HORIZON_H * 3600 else at.timestamp()


def served_model_name(status, checkpoint):
    name, _, fingerprint = checkpoint.partition("@")
    advertised = status.get("model")
    target = status.get("target_artifact")
    # Historical servers may expose a target+drafter or policy digest even as
    # their execution checkpoint. Keep their same-name fallback.
    if not target and (not advertised or advertised == name):
        return name
    requested = fingerprint.removeprefix("sha256:")
    candidates = [target, status.get("artifact"), status.get("base_artifact")]
    if not target:
        execution_checkpoint = (status.get("execution") or {}).get("checkpoint") or ""
        candidates.append(execution_checkpoint.rpartition("@")[2])
    if not requested or not any(
        isinstance(value, str) and requested == value.removeprefix("sha256:") for value in candidates
    ):
        raise ValueError(f"server model {advertised!r} does not identify the requested checkpoint {checkpoint!r}")
    return advertised or name


def run_questions(client, model_name, prefix, items, *, chat, max_tokens, log=None, sampling=None,
                  request_extra=None, chat_shots=None, instruction=CHAT_INSTRUCTION, deadline=None):
    """``sampling`` (optional) adds temperature/top_p/top_k and a per-question
    seed (``seed + test_index``, the law named ``SEED_LAW``) to every request;
    ``None`` keeps greedy.
    ``request_extra`` (an arm's ``request`` object, e.g. ``enable_thinking`` /
    ``reasoning_effort``) is merged into every request body.
    ``chat_shots`` (chat only) are worked-example turns sent before each
    question (``few_shot_turns``); ``None`` keeps the zero-shot chat prompt."""
    request_extra = dict(request_extra or {})
    reserved = {"model", "temperature", "top_p", "top_k", "seed", "stream", "max_tokens",
                "messages", "prompt", "stop"}
    if reserved & set(request_extra):
        raise ValueError(f"arm request may not set {sorted(reserved & set(request_extra))}")
    if not chat and request_extra:
        raise ValueError("arm request fields apply to --chat runs only")
    rows = []
    for index, item in enumerate(items):
        if deadline is not None and time.time() >= deadline:
            break                               # no new question after --stop-at; the summary counts rows
        start = time.monotonic()
        # Seed by the question's test index, not its position in this run, so an
        # --indices / --offset rerun draws the same seed for the same question.
        test_index = item.get("test_index")
        extra_fields = sampling_fields(sampling, index if test_index is None else test_index)
        if chat:
            payload = {"model": model_name, "temperature": 0, **extra_fields, **request_extra, "stream": False, "max_tokens": max_tokens,
                       "messages": list(chat_shots or [])
                                   + [{"role": "user", "content": item["question"] + instruction}]}
            result = client.call("/v1/chat/completions", payload)
            choice = result["choices"][0]
            text = choice["message"].get("content") or ""
            reasoning = choice["message"].get("reasoning_content") or choice["message"].get("reasoning") or ""
        else:
            payload = {"model": model_name, "temperature": 0, **extra_fields, **request_extra, "stream": False, "max_tokens": max_tokens,
                       "stop": STOPS, "prompt": prefix + f"Question: {item['question']}\nAnswer:"}
            result = client.call("/v1/completions", payload)
            choice = result["choices"][0]
            text = choice.get("text") or ""
            reasoning = ""
        elapsed = time.monotonic() - start
        usage = result.get("usage") or {}
        extra = result.get("mlx2") or {}
        ttft = find(result, "ttft_seconds")
        completion = usage.get("completion_tokens") or 0
        predicted, expected = extract_answer(text), reference_answer(item["answer"])
        row = {"index": index, "test_index": item.get("test_index"), "expected": expected, "predicted": predicted,
               "correct": predicted is not None and predicted == expected, "invalid": predicted is None,
               "finish_reason": choice.get("finish_reason"), "completion_tokens": completion,
               "prompt_tokens": usage.get("prompt_tokens"),
               "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "elapsed_seconds": elapsed, "ttft_seconds": ttft,
               "decode_tps": ((completion - 1) / (elapsed - ttft)
                              if ttft and completion > 1 and elapsed > ttft else None),
               "route": extra.get("route"), "speculation": speculation_receipt(extra), "text": text,
               "reasoning_chars": len(reasoning),
               # Calculator markers the model wrote, and the deterministic-tile counters of this request
               # (null when the server loaded no deterministic tiles).
               "markers": text.count("<<"), "reasoning_markers": reasoning.count("<<"),
               "det_tiles": (find(result, "amoeba_det_tiles_request") or {}).get("calculator")}
        if sampling:
            row["effective_sampling"] = find(result, "effective_sampling")
            row["sampling_filter"] = find(result, "sampling_filter")
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
            "completion_tokens_median": median(r["completion_tokens"] for r in rows),
            "reasoning_chars_median": median(r.get("reasoning_chars", 0) for r in rows),
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
    keep = {k: row.get(k) for k in ("index", "test_index", "expected", "predicted", "correct", "finish_reason",
                                     "completion_tokens", "decode_tps", "text", "route", "markers", "reasoning_markers",
                                     "det_tiles")}
    spec = row.get("speculation") or {}
    keep["accept_hist"] = spec.get("verify_accept_hist")
    keep["depth_hist"] = spec.get("verify_depth_hist")
    for name in ("effective_sampling", "sampling_filter"):
        if name in row:
            keep[name] = row[name]
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
    parser.add_argument("--indices", help="comma-separated test-set indices; overrides --offset/--num-questions")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="default 2048 for the chat formats (512 cut off hard problems mid-working), 256 for completion")
    parser.add_argument("--format", choices=("chat-fewshot", "chat-fewshot-annotated", "chat-annotated", "chat", "completion"), default=None,
                        help="prompt format (default chat-fewshot); see the module docstring")
    parser.add_argument("--chat", action="store_true", help="alias for --format chat (zero-shot)")
    parser.add_argument("--request", type=json.loads, default=None,
                        help='JSON merged into every request, e.g. \'{"enable_thinking": false}\'; in --model '
                             'mode an arm\'s own "request" fields override it (the receipt records the merge)')
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--stop-at", metavar="HH:MM",
                        help="ask no new question after this local time (next occurrence); arms not started are skipped")
    parser.add_argument("--continue-on-arm-failure", action="store_true",
                        help="--model mode: record a failed arm and go on to the next (exit 1 at the end)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="> 0 samples (sampled speculative acceptance); 0 keeps the greedy standard")
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=0,
                        help="sampled runs: test question i uses seed + i whatever --offset/--indices "
                             "selected it (receipt seed_law seed_plus_test_index; receipts "
                             "without seed_law used seed + position in the run)")
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

    deadline = stop_time(args.stop_at)
    sampling = None
    if args.temperature > 0:
        # Fields left unset take the server's resolved (vendor) defaults; each
        # row's receipt records the effective sampling the server used.
        sampling = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                    "seed": args.seed, "seed_law": SEED_LAW, "unset_fields": "server defaults"}
    elif args.top_p is not None or args.top_k is not None:
        parser.error("--top-p/--top-k need --temperature > 0")
    data_revision, files = dataset_files()
    train, test = load_split(files["train"]), load_split(files["test"])
    if args.indices is None:
        if args.offset < 0 or args.offset >= len(test) or args.num_questions < 1:
            parser.error("--offset and --num-questions must select at least one test question")
        picks = list(range(args.offset, min(args.offset + args.num_questions, len(test))))
    else:
        try:
            picks = [int(i) for i in args.indices.split(",")]
        except ValueError:
            parser.error("--indices must contain comma-separated integer test indices")
        if any(i < 0 or i >= len(test) for i in picks):
            parser.error(f"--indices must be between 0 and {len(test) - 1}")
    items = [{**test[i], "test_index": i} for i in picks]
    if args.chat and args.format not in (None, "chat"):
        parser.error("--chat conflicts with --format " + args.format)
    fmt = "chat" if args.chat else (args.format or "chat-fewshot")
    args.chat = fmt != "completion"
    if args.max_tokens is None:
        args.max_tokens = 2048 if args.chat else 256
    chat_shots = (few_shot_turns(train, args.shots, annotated=fmt == "chat-fewshot-annotated")
                  if fmt.startswith("chat-fewshot") else None)
    instruction = {"chat-fewshot-annotated": ANNOTATED_INSTRUCTION,
                   "chat-annotated": PROSE_MARKER_INSTRUCTION}.get(fmt, CHAT_INSTRUCTION)
    prefix = "" if args.chat else few_shot_prefix(train, args.shots)
    mode = {"chat-fewshot": f"chat {args.shots}-shot", "chat-fewshot-annotated": f"chat {args.shots}-shot, <<...>> kept",
            "chat-annotated": "chat zero-shot, <<...>> format described in prose",
            "chat": "chat", "completion": f"completion {args.shots}-shot"}[fmt]
    receipt = {"schema": SCHEMA, "qualified": False,
               "kind": "GSM8K acceptance and accuracy, " + ("sampled" if sampling else "greedy")
                       + ", no thermal hold",
               "dataset": {"name": DATASET, "revision": data_revision,
                           "files": {k: {"path": str(v), "sha256": sha256(v)} for k, v in files.items()},
                           "split": "test", "offset": args.offset, "questions": len(items),
                           **({"indices": picks} if args.indices else {})},
               "prompt": {"mode": mode, "format": fmt,
                          "max_tokens": args.max_tokens, "stops": None if args.chat else STOPS,
                          "temperature": args.temperature, **({"sampling": sampling} if sampling else {})},
               "driver_revision": revision(ROOT), "arms": [],
               **({"stop_at": {"clock": args.stop_at, "epoch": deadline}} if deadline else {})}
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
        rows = run_questions(client, served_model_name(status, args.checkpoint), prefix, items, chat=args.chat, chat_shots=chat_shots, instruction=instruction,
                             max_tokens=args.max_tokens, log=progress("server"), sampling=sampling,
                             request_extra=args.request, deadline=deadline)
        arm = {"name": "server", "base_url": args.base_url, "startup_status_route": status.get("route_receipt"),
               "request": args.request or {},
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
            if deadline is not None and time.time() >= deadline:
                receipt["arms"].append({"name": arm["name"], "passed": False, "skipped": f"--stop-at {args.stop_at}"})
                write_atomic(path, receipt)
                print(json.dumps({"arm": arm["name"], "skipped": "stop_at"}), flush=True)
                continue
            out = args.output / arm["name"]
            out.mkdir()
            # The top-level --request applies to every arm (2026-11-24: it was
            # silently ignored here, so enable_thinking=false never reached the
            # Ling-3.2-air night arms); an arm's own fields win.
            arm_request = {**(args.request or {}), **arm.get("request", {})}
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
                                      process_group=0, env=env, cwd=tree)
            row = {"name": arm["name"], "tree": str(tree), "revision": revision(tree),
                   "args": arm.get("args", []), "env": arm.get("env", {}), "request": arm_request,
                   "command": command}
            try:
                # Readiness has its own monotonic bound: ``deadline`` is the
                # --stop-at wall-clock epoch the question loop compares with
                # time.time(), and overwriting it stopped every arm at once.
                ready_by = time.monotonic() + 900
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("server exited before readiness")
                    try:
                        status = client.call("/v1/status", timeout=10)
                        if status.get("healthy"):
                            break
                    except (URLError, TimeoutError, ConnectionError, OSError):
                        pass
                    if time.monotonic() > ready_by:
                        raise RuntimeError("server readiness timed out")
                    time.sleep(.5)
                row["startup_status"] = status
                rows = run_questions(client, served_model_name(status, receipt["checkpoint"]), prefix, items, chat=args.chat, chat_shots=chat_shots, instruction=instruction,
                                     max_tokens=args.max_tokens, log=progress(arm["name"]), sampling=sampling,
                                     request_extra=arm_request, deadline=deadline)
                time.sleep(1.5)
                row["final_status"] = client.call("/v1/status")
                (out / "metrics.prom").write_text(client.call("/metrics"))
                (out / "rows.json").write_text(json.dumps(rows, indent=1, default=str) + "\n")
                row["summary"] = summarize(rows, arm.get("draft_depth", args.draft_depth))
                row["rows"] = [compact(r) for r in rows]
                row["stopped_at_deadline"] = len(rows) < len(items)
                row["passed"] = True
            except BaseException as error:
                row["passed"] = False
                row["error"] = repr(error)
                if not (args.continue_on_arm_failure and isinstance(error, Exception)):
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
            if "summary" in row:
                print(json.dumps({"arm": row["name"], **{k: v for k, v in row["summary"].items()
                                                          if k != "routes"}}), flush=True)
            else:
                print(json.dumps({"arm": row["name"], "error": row.get("error")}), flush=True)
    finally:
        token_file.unlink(missing_ok=True)
    return 0 if all(arm["passed"] for arm in receipt["arms"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
