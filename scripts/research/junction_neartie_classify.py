#!/usr/bin/env python3
"""Classify GSM8K A/B text differences under the AGENTS.md near-tie contract.

Original mlx2 script (2026-11-22).  Input: two arms of one
``scripts/spec_acceptance_gsm8k.py`` run (``--gsm8k-dir`` holding
``receipt.json`` and ``<arm>/rows.json``; the arm configs come from the
receipt, or from ``--arms`` in the driver's arms.json format).  For the
questions whose outputs differ it replays both arms on loopback servers
started with each arm's own args/env/tree, localizes the first divergent
generated token, reads one bounded logprob window at that position, and
classifies each question.

Replay (per arm, one server per phase, the driver's exact server command and
chat request bodies, greedy, the arm's ``request`` fields such as
``reasoning_effort``):

* phase ``trace``: every question of the original run in its original order
  (``--replay-order original``, default) -- so APCv2 sees the same prompt
  history and the junction arm restores where the run did -- or the
  ``--warmup-indices`` then the selected questions (``selected``).  Selected
  questions carry ``return_token_trace`` (generated token ids, reasoning
  included), so the first divergence is found on tokens, not on text.
* phase ``window``: the same sequence on a fresh server; each diverging
  question carries ``logprobs`` with ``top_logprobs`` K (default 10) and
  ``logprobs_start = logprobs_end = d + 1``: only the already-selected
  first-divergence position crosses to the host (the server's bounded
  window), with the original ``max_tokens`` so verify shapes are unchanged.
  The window's token trace must reproduce the trace phase through ``d``.
  (Any logprobs request makes the route build logprob rows every round; only
  the serialization is windowed.  The reproduction gate catches a token
  change that this or anything else causes.)

Selection (``--select``): ``any`` (default) takes questions whose content text
or completion-token count differs -- a reasoning-only divergence changes the
count but not the text -- and ``text`` only content-text differences.

Classification of one question (``classify_row``), ceiling 0.5 nat:

* ``no_divergence_on_replay``: the replayed token streams are equal;
* ``unclassified``: missing or unreproduced evidence, a length-only
  divergence, a selected token that is not its own window's top (greedy), or a
  cross token missing from a window that does not prove it is far;
* contract clauses (``contract``): ``two_way`` (same unordered top two, both
  selected-vs-alternate margins <= 0.5), ``exact_tie`` (one arm's margin is
  exactly 0 and the other arm selected one of its tied tokens with margin <=
  0.5), ``multi_way`` (each arm's selected token within 0.5 of the other
  arm's top log-probability, read from that arm's window);
* ``high_margin``: no clause holds and a selected token is more than 0.5 below
  the other arm's top -- read from the window, or proven by a window whose
  K-th entry is already more than 0.5 below its top while the token is absent;
* a clause holds, then the GSM8K functional oracle (the driver's
  ``extract_answer`` on each full replay continuation): both answers correct,
  or both the same answer -> ``near_tie_equivalent``; otherwise
  ``oracle_failed``.

Receipts name the checkpoint (``<artifact>@<fingerprint>``) and the mlx2
commit.  Needs ``--cpg-lock`` (the driver's qualification lease), refuses port
8282 and a nonempty ``--output``, stops each server's process group in
``finally``, and writes ``receipt.json`` atomically after every phase.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import spec_acceptance_gsm8k as driver
from hy4_context_ladder import stop_group

SCHEMA = "mlx2.junction-neartie-classification.v1"
CEILING = 0.5
EPSILON = 1e-9
# A greedy pick may sit a float hair below its window's top (the shared rule's
# tolerance, mlx2/near_tie.py).
TOP_TOLERANCE = 1e-6
VERDICTS = ("near_tie_equivalent", "high_margin", "oracle_failed", "unclassified",
            "no_divergence_on_replay")
RESERVED_PORT = 8282


# ------------------------------------------------------------ pure logic --

def first_divergence(a, b):
    """Index of the first differing token, ``len`` of the shorter stream when
    one is a strict prefix of the other, ``None`` when equal."""
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def select_questions(rows_a, rows_b, mode="any"):
    """Test indices whose outputs differ between the arms (paired by test index)."""
    by_b = {r["test_index"]: r for r in rows_b}
    picked = []
    for row in rows_a:
        other = by_b.get(row["test_index"])
        if other is None:
            continue
        text = row.get("text") != other.get("text")
        count = row.get("completion_tokens") != other.get("completion_tokens")
        if text or (mode == "any" and count):
            picked.append(row["test_index"])
    return picked


def window_from_entry(entry):
    """``{"selected", "selected_logprob", "top": [[id, lp], ...] desc}`` from
    one server logprob entry (``id``, ``logprob``, ``top_logprobs``)."""
    if not isinstance(entry, dict) or "id" not in entry or "logprob" not in entry:
        return None
    top = [[int(item["id"]), float(item["logprob"])] for item in entry.get("top_logprobs") or []]
    top.sort(key=lambda item: (-item[1], item[0]))
    return {"selected": int(entry["id"]), "selected_logprob": float(entry["logprob"]), "top": top}


def _lookup(window, token):
    for tid, lp in window["top"]:
        if tid == token:
            return lp
    return None


def margin(window):
    """Selected-vs-best-alternate margin (nat), ``None`` if unreadable.

    Negative when the selected token is not the window's top.
    """
    top = window.get("top") or []
    selected = window.get("selected")
    value = _lookup(window, selected)
    others = [lp for tid, lp in top if tid != selected]
    if value is None or not others or not all(math.isfinite(lp) for _, lp in top):
        return None
    return value - max(others)


def cross_gap(window, token, ceiling=CEILING):
    """How far ``token`` sits below ``window``'s top: ``("ok"|"far"|"missing", gap)``.

    A token absent from a top-K window has a log-probability at most the K-th
    entry's, so the window proves it far when its K-th entry is already more
    than ``ceiling`` below its top; otherwise the token is ``missing``.
    """
    top_lp = window["top"][0][1]
    value = _lookup(window, token)
    if value is not None:
        gap = top_lp - value
        return ("ok" if gap <= ceiling + EPSILON else "far"), gap
    floor_gap = top_lp - window["top"][-1][1]
    return ("far" if floor_gap > ceiling + EPSILON else "missing"), None


def contract(a, b, ceiling=CEILING):
    """Apply the three AGENTS.md clauses to two first-divergence windows.

    Returns ``{"verdict": "contract_pass"|"high_margin"|"unclassified",
    "clause", "reason", "margins", "cross"}``; the caller owns the prefix,
    reproduction and functional-oracle gates.
    """
    out = {"clause": None, "margins": None, "cross": None}
    if not a or not b or len(a.get("top") or []) < 2 or len(b.get("top") or []) < 2:
        return {**out, "verdict": "unclassified", "reason": "window shorter than two entries"}
    ma, mb = margin(a), margin(b)
    out["margins"] = {"a": ma, "b": mb}
    if ma is None or mb is None:
        return {**out, "verdict": "unclassified", "reason": "selected token missing from its own window"}
    if ma < -TOP_TOLERANCE or mb < -TOP_TOLERANCE:
        return {**out, "verdict": "unclassified", "reason": "selected token is not its own window's top"}
    if a["selected"] == b["selected"]:
        return {**out, "verdict": "unclassified", "reason": "windows select the same token (no divergence)"}
    top2_a = {t for t, _ in a["top"][:2]}
    top2_b = {t for t, _ in b["top"][:2]}
    if top2_a == top2_b and ma <= ceiling + EPSILON and mb <= ceiling + EPSILON:
        return {**out, "verdict": "contract_pass", "clause": "two_way",
                "reason": "same unordered top two, both margins within the ceiling"}
    for tied, other, other_margin in ((a, b, mb), (b, a, ma)):
        if margin(tied) == 0.0:
            best = tied["top"][0][1]
            tied_tokens = {t for t, lp in tied["top"] if lp == best}
            if other["selected"] in tied_tokens and other_margin <= ceiling + EPSILON:
                return {**out, "verdict": "contract_pass", "clause": "exact_tie",
                        "reason": "one arm's top is an exact tie and the other selected a tied token"}
    cross = {"b_selected_in_a": cross_gap(a, b["selected"], ceiling),
             "a_selected_in_b": cross_gap(b, a["selected"], ceiling)}
    out["cross"] = {k: {"status": s, "gap": g} for k, (s, g) in cross.items()}
    statuses = [s for s, _ in cross.values()]
    if statuses == ["ok", "ok"]:
        return {**out, "verdict": "contract_pass", "clause": "multi_way",
                "reason": "each selected token within the ceiling of the other arm's top"}
    if "far" in statuses:
        return {**out, "verdict": "high_margin",
                "reason": f"a selected token is more than {ceiling} nat below the other arm's top"}
    return {**out, "verdict": "unclassified",
            "reason": "a selected token is missing from the other arm's window, which does not prove it far"}


def functional_oracle(text_a, text_b, expected):
    """GSM8K oracle: both answers correct, or both continuations give the same answer."""
    answer_a, answer_b = driver.extract_answer(text_a or ""), driver.extract_answer(text_b or "")
    correct_a = answer_a is not None and answer_a == expected
    correct_b = answer_b is not None and answer_b == expected
    same = answer_a is not None and answer_a == answer_b
    return {"expected": expected, "a": answer_a, "b": answer_b, "a_correct": correct_a,
            "b_correct": correct_b, "same_answer": same, "passed": (correct_a and correct_b) or same}


def classify_row(trace_a, trace_b, windows, oracle, ceiling=CEILING):
    """One question's verdict from both arms' replay evidence.

    ``trace_*``: generated token ids of the trace phase.  ``windows``:
    ``{"a": {"window", "trace"}, "b": ...}`` from the window phase (or None).
    ``oracle``: ``functional_oracle`` of the trace phase's continuations.
    """
    d = first_divergence(trace_a, trace_b)
    row = {"first_divergence": d, "shared_prefix_tokens": d, "oracle": oracle}
    if d is None:
        return {**row, "verdict": "no_divergence_on_replay", "reason": "replayed token streams are equal"}
    if d >= len(trace_a) or d >= len(trace_b):
        return {**row, "verdict": "unclassified",
                "reason": "length-only divergence: one stream ended where the other continued"}
    row["tokens"] = {"a": trace_a[d], "b": trace_b[d]}
    if not windows or not windows.get("a") or not windows.get("b"):
        return {**row, "verdict": "unclassified", "reason": "no window evidence"}
    for side, trace in (("a", trace_a), ("b", trace_b)):
        evidence = windows[side]
        window, replay = evidence.get("window"), evidence.get("trace") or []
        if window is None:
            return {**row, "verdict": "unclassified", "reason": f"arm {side} returned no window"}
        if list(replay[: d + 1]) != list(trace[: d + 1]) or window["selected"] != trace[d]:
            return {**row, "verdict": "unclassified",
                    "reason": f"arm {side} did not reproduce its tokens through the divergence"}
    row["windows"] = {"a": windows["a"]["window"], "b": windows["b"]["window"]}
    result = contract(windows["a"]["window"], windows["b"]["window"], ceiling)
    row.update(clause=result["clause"], margins=result["margins"], cross=result["cross"])
    if result["verdict"] != "contract_pass":
        return {**row, "verdict": result["verdict"], "reason": result["reason"]}
    if not oracle["passed"]:
        return {**row, "verdict": "oracle_failed",
                "reason": f"{result['clause']} near tie, but the functional oracle failed"}
    return {**row, "verdict": "near_tie_equivalent", "reason": result["reason"]}


def count_verdicts(rows):
    """Verdict totals.  Only ``near_tie_equivalent`` passes: ``failures`` are
    contract or oracle failures, ``unexplained`` the unclassified rows (a
    failure under AGENTS.md until evidence classifies them), and
    ``not_reproduced`` the questions whose difference did not recur."""
    counts = dict.fromkeys(VERDICTS, 0)
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    clauses = {}
    for row in rows:
        if row["verdict"] == "near_tie_equivalent":
            clauses[row["clause"]] = clauses.get(row["clause"], 0) + 1
    return {**counts, "near_tie_clauses": clauses, "questions": len(rows),
            "failures": counts["high_margin"] + counts["oracle_failed"],
            "unexplained": counts["unclassified"],
            "not_reproduced": counts["no_divergence_on_replay"],
            "all_near_tie": bool(rows) and counts["near_tie_equivalent"] == len(rows)}


def replay_sequence(original_order, selected, warmups, mode):
    """Test indices to send, in order; selected ones carry evidence fields."""
    if mode == "original":
        last = max(original_order.index(i) for i in selected)
        return original_order[: last + 1]
    # A selected warm-up keeps its warm-up position (and carries evidence there).
    return list(warmups) + [i for i in selected if i not in warmups]


# --------------------------------------------------------------- serving --

class ArmServer:
    """One loopback ``mlx2.server`` with the driver's command for one arm."""

    def __init__(self, args, arm, out, token_file):
        self.args, self.arm, self.out, self.token_file = args, arm, out, token_file
        self.process = self.log = None

    def command(self):
        a = self.args
        command = [a.python, "-u", "-m", "mlx2.server", "--model", str(a.model),
                   "--host", "127.0.0.1", "--port", str(a.port), "--max-inflight", "1",
                   "--max-context", str(a.max_context), "--cache-bytes", str(2 << 30),
                   "--cache-dir", str(self.out / "cache"), "--admin-token-file", str(self.token_file),
                   *self.arm.get("args", [])]
        if "--max-lanes" not in self.arm.get("args", []):
            command += ["--max-lanes", "1"]
        return command

    def port_busy(self):
        import socket

        with socket.socket() as probe:
            return probe.connect_ex(("127.0.0.1", self.args.port)) == 0

    def __enter__(self):
        if self.port_busy():
            # A leftover server would answer for this arm.
            raise RuntimeError(f"port {self.args.port} is already serving; refusing to start")
        self.out.mkdir(parents=True)
        tree = Path(self.arm["tree"]).resolve()
        self.log = (self.out / "server.log").open("w")
        env = {**os.environ, "PYTHONPATH": str(tree / "src"), **self.arm.get("env", {})}
        self.process = subprocess.Popen(self.command(), stdout=self.log, stderr=subprocess.STDOUT,
                                        start_new_session=True, env=env, cwd=tree)
        self.client = driver.Client(f"http://127.0.0.1:{self.args.port}", self.token_file.read_text())
        try:
            self._wait_ready()
        except BaseException:
            self.__exit__(None, None, None)   # a failed start still stops the group
            raise
        return self

    def _wait_ready(self):
        ready_by = time.monotonic() + 900
        while True:
            if self.process.poll() is not None:
                raise RuntimeError("server exited before readiness")
            try:
                status = self.client.call("/v1/status", timeout=10)
                if status.get("healthy"):
                    if self.process.poll() is not None:
                        raise RuntimeError("our server exited; another process answered on the port")
                    self.status = status
                    return
            except (URLError, TimeoutError, ConnectionError, OSError):
                pass
            if time.monotonic() > ready_by:
                raise RuntimeError("server readiness timed out")
            time.sleep(0.5)

    def __exit__(self, *exc):
        self.stopped = stop_group(self.process)
        if self.log and not self.log.closed:
            self.log.close()
        if self.stopped != "stopped" and exc[0] is None:
            raise RuntimeError("server process group is still running after stop")
        return False


def request_body(model_name, item, chat_shots, instruction, request_extra, max_tokens, extra=None):
    """The driver's ``chat`` request body for one question, plus evidence fields."""
    return {"model": model_name, "temperature": 0, **(request_extra or {}), "stream": False,
            "max_tokens": max_tokens,
            "messages": list(chat_shots or []) + [{"role": "user", "content": item["question"] + instruction}],
            **(extra or {})}


def send(client, body):
    result = client.call("/v1/chat/completions", body)
    choice = result["choices"][0]
    extra = result.get("mlx2") or {}
    usage = result.get("usage") or {}
    trace = extra.get("token_trace") or {}
    content = (choice.get("logprobs") or {}).get("content") or []
    return {"text": choice["message"].get("content") or "",
            "reasoning_chars": len(choice["message"].get("reasoning_content") or ""),
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "route": extra.get("route"),
            "token_ids": trace.get("token_ids"), "execution_widths": trace.get("execution_widths"),
            "trace_truncated": trace.get("truncated"),
            "logprob_entries": len(content),
            "window": window_from_entry(content[0]) if len(content) == 1 else None}


def run_phase(args, arm, phase, out, token_file, sequence, selected, positions, context, checkpoint):
    """Replay ``sequence`` on a fresh server; evidence for selected indices only."""
    record = {"phase": phase, "requests": {}}
    context.setdefault("partial", {})[f"{arm['name']}-{phase}"] = record   # visible if the phase fails
    with ArmServer(args, arm, out, token_file) as server:
        record["command"] = server.command()
        record["startup_route"] = server.status.get("route_receipt")
        model_name = driver.served_model_name(server.status, checkpoint)
        for index in sequence:
            extra = None
            if index in selected:
                if phase == "trace":
                    extra = {"return_token_trace": True}
                elif index in positions:
                    p = positions[index] + 1
                    extra = {"return_token_trace": True, "logprobs": True, "top_logprobs": args.top_logprobs,
                             "logprobs_start": p, "logprobs_end": p}
            body = request_body(model_name, context["items"][index], context["shots"],
                                context["instruction"], arm.get("request"), context["max_tokens"], extra)
            result = send(server.client, body)
            if extra is None:
                result = {k: result[k] for k in ("cached_tokens", "completion_tokens", "route")}
            record["requests"][str(index)] = result
        time.sleep(1.5)
        apcv2 = (server.client.call("/v1/status") or {}).get("apcv2") or {}
        record["final_status_apcv2"] = {k: apcv2.get(k) for k in
                                        ("junction_hits", "interior_hits", "rolling_hits", "lifetime")}
    record["server_stopped"] = server.stopped
    return record


# ------------------------------------------------------------------ main --

def load_inputs(args):
    receipt = json.loads((args.gsm8k_dir / "receipt.json").read_text())
    arms_by_name = {a["name"]: a for a in (json.loads(args.arms.read_text()) if args.arms else receipt["arms"])}
    arms = []
    for name, rows_path in ((args.arm_a, args.rows_a), (args.arm_b, args.rows_b)):
        if name not in arms_by_name:
            raise SystemExit(f"refused: arm {name!r} not found")
        source = arms_by_name[name]
        arm = {"name": name, "tree": str(args.tree or source["tree"]), "args": list(source.get("args", [])),
               "env": dict(source.get("env", {})), "request": dict(source.get("request", {})),
               "original_revision": source.get("revision")}
        path = rows_path or (args.gsm8k_dir / name / "rows.json")
        arm["rows_path"] = str(path)
        arm["rows"] = json.loads(Path(path).read_text())
        arms.append(arm)
    return receipt, arms


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gsm8k-dir", type=Path, required=True, help="driver output (receipt.json, <arm>/rows.json)")
    parser.add_argument("--arm-a", required=True, help="reference arm name (e.g. without junctions)")
    parser.add_argument("--arm-b", required=True, help="candidate arm name (e.g. with junctions)")
    parser.add_argument("--arms", type=Path, help="arms.json overriding the receipt's arm configs")
    parser.add_argument("--rows-a", type=Path)
    parser.add_argument("--rows-b", type=Path)
    parser.add_argument("--tree", type=Path, help="mlx2 tree for both servers (default: each arm's)")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=18311)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--top-logprobs", type=int, default=10)
    parser.add_argument("--select", choices=("any", "text"), default="any")
    parser.add_argument("--indices", help="comma-separated test indices (overrides --select)")
    parser.add_argument("--replay-order", choices=("original", "selected"), default="original")
    parser.add_argument("--warmup-indices", help="--replay-order selected: default the run's first two questions")
    parser.add_argument("--ceiling", type=float, default=CEILING)
    args = parser.parse_args(argv)
    if args.port == RESERVED_PORT:
        parser.error("port 8282 is reserved for release trees")
    if not 2 <= args.top_logprobs <= 20:
        parser.error("--top-logprobs must be 2..20 (the contract reads a top two; the server caps at 20)")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("refusing nonempty evidence directory")
    receipt, arms = load_inputs(args)
    prompt = receipt.get("prompt") or {}
    fmt = prompt.get("format", "chat-fewshot")
    if fmt == "completion" or float(prompt.get("temperature") or 0) != 0 or prompt.get("sampling"):
        raise SystemExit("refused: greedy chat-format runs only")
    args.output.mkdir(parents=True, exist_ok=True)

    from qualification_gpu_ownership import require_qualification_lease
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.registry import inspect_model

    owner = json.loads((args.cpg_lock / "owner.json").read_text())
    ownership = require_qualification_lease(task_id=owner["cpg_task"], generation=owner["cpg_generation"],
                                            cpg_owner_lock=args.cpg_lock)
    checkpoint = f"{args.model.name}@sha256:{inspect_model(args.model).artifact_fingerprint}"
    if receipt.get("checkpoint") and receipt["checkpoint"] != checkpoint:
        raise SystemExit(f"refused: run checkpoint {receipt['checkpoint']} differs from {checkpoint}")

    _rev, files = driver.dataset_files()
    train, test = driver.load_split(files["train"]), driver.load_split(files["test"])
    shots = int(str(prompt.get("mode", "chat 5-shot")).split()[1].split("-")[0]) if fmt.startswith("chat-fewshot") else 0
    context = {"items": {i: {**test[i], "test_index": i} for i in range(len(test))},
               "shots": (driver.few_shot_turns(train, shots, annotated=fmt == "chat-fewshot-annotated")
                         if fmt.startswith("chat-fewshot") else None),
               "instruction": {"chat-fewshot-annotated": driver.ANNOTATED_INSTRUCTION,
                               "chat-annotated": driver.PROSE_MARKER_INSTRUCTION}.get(fmt, driver.CHAT_INSTRUCTION),
               "max_tokens": int(prompt["max_tokens"])}
    rows_a, rows_b = arms[0]["rows"], arms[1]["rows"]
    original_order = [r["test_index"] for r in rows_b]
    selected = ([int(i) for i in args.indices.split(",")] if args.indices
                else select_questions(rows_a, rows_b, args.select))
    unknown = sorted(set(selected) - set(original_order))
    if unknown:
        raise SystemExit(f"refused: indices {unknown} are not questions of the run")
    warmups = ([int(i) for i in args.warmup_indices.split(",")] if args.warmup_indices
               else original_order[:2])
    out = {"schema": SCHEMA, "qualified": False, "ceiling_nat": args.ceiling,
           "kind": "first-divergence near-tie classification, greedy replay, no thermal hold",
           "checkpoint": checkpoint, "source_revision": driver.revision(ROOT), "ownership": ownership,
           "input": {"gsm8k_dir": str(args.gsm8k_dir), "driver_revision": receipt.get("driver_revision"),
                     "format": fmt, "max_tokens": context["max_tokens"]},
           "selection": {"mode": "indices" if args.indices else args.select, "selected": selected,
                         "text_differs": len(select_questions(rows_a, rows_b, "text")),
                         "any_differs": len(select_questions(rows_a, rows_b, "any"))},
           "replay": {"order": args.replay_order, "warmups": warmups if args.replay_order == "selected" else None,
                      "top_logprobs": args.top_logprobs},
           "arms": [{k: v for k, v in arm.items() if k != "rows"} for arm in arms], "phases": {}}
    for arm in out["arms"]:
        arm["revision"] = driver.revision(Path(arm["tree"]))
        lane_args = arm["args"]
        arm["max_lanes"] = (int(lane_args[lane_args.index("--max-lanes") + 1])
                            if "--max-lanes" in lane_args else 1)
    width_one = all(arm["max_lanes"] == 1 for arm in out["arms"])
    out["width_scope"] = {
        "max_lanes": [arm["max_lanes"] for arm in out["arms"]], "lane_width_one": width_one,
        "note": ("AGENTS.md keeps width-one replay gates exact and allows near_tie_equivalent at "
                 "physical width > 1.  These arms differ by execution policy (prefill partition at "
                 "the junction), not by cold/warm replay of one route; whether the clause applies at "
                 "lane width one is the coordinator's call, so width-one rows are labelled.")
                if width_one else None}
    path = args.output / "receipt.json"
    driver.write_atomic(path, out)
    if not selected:
        out["counts"] = count_verdicts([])
        driver.write_atomic(path, out)
        print(json.dumps(out["counts"]))
        return 0
    sequence = replay_sequence(original_order, selected, warmups, args.replay_order)
    out["replay"]["sequence"] = sequence
    token_file = args.output / "admin-token"
    token_file.write_text(secrets.token_hex(24))
    token_file.chmod(0o600)
    sides = {"a": arms[0], "b": arms[1]}
    try:
        traces = {}
        for side, arm in sides.items():
            record = run_phase(args, arm, "trace", args.output / f"{side}-trace", token_file, sequence,
                               set(selected), {}, context, checkpoint)
            out["phases"][f"{side}-trace"] = record
            traces[side] = record["requests"]
            driver.write_atomic(path, out)
        positions = {}
        for index in selected:
            a = traces["a"][str(index)].get("token_ids") or []
            b = traces["b"][str(index)].get("token_ids") or []
            d = first_divergence(a, b)
            if d is not None and d < len(a) and d < len(b):
                positions[index] = d
        out["replay"]["window_positions"] = {str(k): v + 1 for k, v in positions.items()}
        windows = {}
        if positions:
            for side, arm in sides.items():
                record = run_phase(args, arm, "window", args.output / f"{side}-window", token_file, sequence,
                                   set(positions), positions, context, checkpoint)
                out["phases"][f"{side}-window"] = record
                windows[side] = record["requests"]
                driver.write_atomic(path, out)
    except BaseException as error:
        out["error"] = repr(error)
        # Every phase record so far, the failed one with the requests it finished.
        out["partial_phases"] = context.get("partial", {})
        driver.write_atomic(path, out)
        raise
    finally:
        token_file.unlink(missing_ok=True)
    originals = {side: {r["test_index"]: r for r in arm["rows"]} for side, arm in sides.items()}
    rows = []
    for index in selected:
        ta, tb = traces["a"][str(index)], traces["b"][str(index)]
        expected = driver.reference_answer(context["items"][index]["answer"])
        oracle = functional_oracle(ta["text"], tb["text"], expected)
        evidence = None
        if index in positions:
            evidence = {side: {"window": windows[side][str(index)]["window"],
                               "trace": windows[side][str(index)].get("token_ids")} for side in sides}
        row = classify_row(ta.get("token_ids") or [], tb.get("token_ids") or [], evidence, oracle, args.ceiling)
        d = row["first_divergence"]
        row.update({
            "test_index": index,
            "cached_tokens": {"a": ta["cached_tokens"], "b": tb["cached_tokens"],
                              "a_original": originals["a"].get(index, {}).get("cached_tokens"),
                              "b_original": originals["b"].get(index, {}).get("cached_tokens")},
            "replay_matches_original": {side: (traces[side][str(index)]["text"]
                                               == originals[side].get(index, {}).get("text")) for side in sides},
            "execution_width_at_divergence": {
                side: ((traces[side][str(index)].get("execution_widths") or [None] * (d + 1))[d]
                       if d is not None and d < len(traces[side][str(index)].get("execution_widths") or []) else None)
                for side in sides},
            "trace_truncated": {side: traces[side][str(index)].get("trace_truncated") for side in sides},
            "lane_width_one": width_one,
        })
        rows.append(row)
    out["rows"] = rows
    out["counts"] = count_verdicts(rows)
    driver.write_atomic(path, out)
    print(json.dumps(out["counts"]), flush=True)
    for row in rows:
        print(row["test_index"], row["verdict"], row.get("clause"), row["first_divergence"], row.get("reason"))
    return 0 if out["counts"]["all_near_tie"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
