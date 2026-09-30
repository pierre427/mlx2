#!/usr/bin/env python3
"""Bounded ragged batched prompt-lookup (PLD) qualification driver.

Scope: DIRECT-MODEL. One process, one model load, generators driven
directly. It is not HTTP serving, not a serving qualification receipt and
selects no route or default. Timings are not recorded.

Arms (fresh generators and caches each; identical pinned token IDs, greedy
sampling, stop tokens and per-lane output caps):

``ordinary_b1``     every lane alone in a width-1 ``BatchGenerator``. This is
                    the declared primary reference, fixed before any run.
``ordinary_bN``     all lanes in one width-N ``BatchGenerator``.
``pld_per_lane``    ``PromptLookupBatchGenerator`` with ``batched_verify=False``.
``pld_batched``     the same with ``batched_verify=True`` (segmented ragged
                    verify of plain/rotating KV lanes).
``pld_removal``     batched PLD where lane ``--remove-lane`` is removed at a
                    closed boundary (between polls) after it emitted
                    ``--remove-after`` tokens; survivors continue at lower
                    width (B2 -> B1 when N = 2).

Every comparison names its reference. PLD arms are judged against
``ordinary_b1``; the comparison with ``ordinary_bN`` and the
``ordinary_b1`` vs ``ordinary_bN`` geometry comparison are reported
separately and never swapped in after the fact. Per lane the driver records
token IDs and hash, logprob-row storage-bit digests (first
``--logprob-rows``), the final target cache ``state_digest`` (class, state
and ``meta_state``), the committed prompt boundary, and a greedy ordinary B1
continuation from the final cache. A missing digest is ``unavailable``,
never equal.

Coverage (all required for ``pass``): unequal prompt lengths and output
caps, a zero-proposal lane, proposals and a rejected suffix (partial
acceptance), rollback followed by appended tokens, batched verify at width
>= 2, and a closed-boundary removal whose survivor then ran at lower width.
A real workload that never proposes or rolls back is ``coverage_refused``,
not ``pass``. Any PLD lane that differs from its reference is a
``counterexample``.

Real runs need ``--i-own-the-gpu`` and explicit token prompts
(``--prompt-ids`` JSON list of lists) or the deterministic constructor
(``--construct``); the original text of the 2026-09-18 campaign prompts is
not recoverable here, so this is not a reproduction of that campaign.
``--tiny`` runs a deterministic random Muse-class model on CPU.

  PYTHONPATH=src .venv/bin/python scripts/qualify_ragged_pld.py --tiny --out /tmp/pld.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

# Full float32 matmuls unless the caller chose otherwise; must precede the
# first MLX operation (adapters pin the same value for real artifacts).
os.environ.setdefault("MLX_ENABLE_TF32", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SCHEMA = "mlx2.direct-model.ragged-pld.v1"
ARMS = ("ordinary_b1", "ordinary_bN", "pld_per_lane", "pld_batched", "pld_removal")
PRIMARY_REFERENCE = "ordinary_b1"
IDENTITY_FILES = (
    "scripts/qualify_ragged_pld.py",
    "scripts/paired_direct_ab.py",
    "src/mlx2/runtime/pld.py",
    "src/mlx2/runtime/generate.py",
    "src/mlx2/runtime/segmented_rotating_kv.py",
    "src/mlx2/runtime/models/cache.py",
)
MAX_LANES = 4
MAX_PROMPT_TOKENS = 16384
MAX_OUTPUT_TOKENS = 512
REPEAT_UNIT = "def area(width, height):\n    return width * height\n\n"
DISTINCT_UNIT = "Quartz vexing jumbled fog; my wry pixie bank."


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- models

def tiny_model():
    import mlx.core as mx
    from mlx2.adapters.muse_glimmer_config import ModelArgs
    from mlx2.runtime.models.muse_glimmer import Model

    mx.random.seed(8)
    model = Model(ModelArgs(
        hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=48, sliding_window=8, max_position_embeddings=1024,
    ))
    model.eval()
    mx.eval(model.parameters())
    return model


def tiny_prompts():
    """Unequal lengths: two repeated-structure lanes and one all-distinct lane."""
    repeat = [3, 4, 5, 6, 7] * 4 + [3, 4]
    looping = [9, 10, 11, 12] * 5 + [9]
    distinct = [20, 21, 22, 23, 24, 25, 26]
    return [repeat, looping, distinct], [24, 28, 2]


def constructed_prompts(tokenizer, lanes):
    """Deterministic token construction for real artifacts (not old prompts)."""
    def encode(text):
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))

    repeat = encode(REPEAT_UNIT * 6)
    distinct = encode(DISTINCT_UNIT)
    prompts = [repeat, repeat[: max(8, len(repeat) * 2 // 3)], distinct, repeat[: len(repeat) // 2]]
    caps = [64, 48, 3, 40]
    return prompts[:lanes], caps[:lanes]


# ---------------------------------------------------------------- run one arm

class Driver:
    def __init__(self, args):
        import mlx.core as mx

        self.args, self.mx = args, mx
        self.stops = ()
        self.identity = {}
        if args.tiny:
            mx.set_default_device(mx.cpu)
            self.model = tiny_model()
            self.prompts, self.caps = tiny_prompts()
            self.identity = {"model": "tiny-random-muse", "fingerprint": None}
            self.followup = [5, 6, 7]
            return
        from mlx2.adapters.registry import resolve_adapter

        cls = resolve_adapter(args.model, mtp=False, qualification_mode=True)
        # Constructed before any runtime module is imported (import-order guard).
        self.adapter = cls(args.model)
        from mlx2.serving import generation_stop_token_ids

        self.model = self.adapter.model
        self.stops = () if args.ignore_eos else generation_stop_token_ids(self.adapter)
        if args.prompt_ids:
            data = json.loads(Path(args.prompt_ids).read_text())
            self.prompts, self.caps = data["prompts"], data["max_tokens"]
            prompt_source = f"explicit token file sha256 {_sha(Path(args.prompt_ids).read_bytes())}"
        else:
            self.prompts, self.caps = constructed_prompts(self.adapter.tokenizer, args.lanes)
            prompt_source = "deterministic constructor (not the 2026-09-18 campaign prompts)"
        self.followup = list(self.prompts[0][:3])
        self.identity = {
            "model": str(args.model),
            "adapter": f"{cls.__module__}.{cls.__qualname__}",
            "adapter_sha256": _sha(Path(sys.modules[cls.__module__].__file__).read_bytes()),
            "fingerprint": self.adapter.identity.get("fingerprint"),
            "environment": dict(getattr(self.adapter, "environment", {}) or {}),
            "prompt_source": prompt_source,
        }

    def check_geometry(self):
        if not 2 <= len(self.prompts) <= MAX_LANES or len(self.caps) != len(self.prompts):
            raise SystemExit(f"refused: need 2..{MAX_LANES} lanes with one cap each")
        for prompt, cap in zip(self.prompts, self.caps):
            if not 2 <= len(prompt) <= MAX_PROMPT_TOKENS or not 1 <= cap <= MAX_OUTPUT_TOKENS:
                raise SystemExit("refused: prompt 2..16384 tokens and cap 1..512 per lane")
            if any(type(t) is not int or t < 0 for t in prompt):
                raise SystemExit("refused: prompts must be token id lists")
        if not 0 <= self.args.remove_lane < len(self.prompts):
            raise SystemExit("refused: --remove-lane out of range")

    def generator(self, arm, width):
        stops = [[t] for t in self.stops]
        if arm.startswith("ordinary"):
            from mlx2.runtime.generate import BatchGenerator

            return BatchGenerator(self.model, completion_batch_size=width, prefill_batch_size=1,
                                  prefill_step_size=self.args.prefill_step, stop_tokens=stops)
        from mlx2.runtime.pld import PromptLookupBatchGenerator

        policy = {**self.args.pld_policy, "batched_verify": arm != "pld_per_lane"}
        return PromptLookupBatchGenerator(self.model, completion_batch_size=width,
                                          prefill_step_size=self.args.prefill_step,
                                          stop_tokens=stops, prompt_lookup=policy)

    def run(self, arm, lanes, *, caches=None, all_tokens=None, caps=None, remove=None, prompts=None):
        """Drive one generator over ``lanes`` (indices); returns per-lane records."""
        from mlx2.runtime.sample_utils import LaneRNG
        from scripts.paired_direct_ab import state_digest

        mx, args = self.mx, self.args
        caps = caps or [self.caps[i] for i in lanes]
        prompts = prompts or [list(self.prompts[i]) for i in lanes]
        gen = self.generator(arm, len(lanes))
        records = {i: {"tokens": [], "logprob_rows": [], "widths": set(), "receipts": [],
                       "finish_reason": None, "final": None, "boundary": None} for i in lanes}
        failures, polls, deadline = [], 0, time.monotonic() + args.time_limit_s
        limit = sum(caps) * 4 + sum(len(p) for p in prompts) // max(1, args.prefill_step) + 64
        removed = None
        try:
            insert = {"max_tokens": caps}
            if caches is not None:
                insert.update(caches=caches, all_tokens=all_tokens)
            if arm.startswith("ordinary"):
                insert["lane_rngs"] = [LaneRNG(args.seed + i) for i in lanes]
            uids = gen.insert(prompts, **insert)
            by_uid = dict(zip(uids, lanes))
            live = set(uids)
            while live:
                polls += 1
                if polls > limit or time.monotonic() > deadline:
                    failures.append(f"bounded: stopped after {polls} polls")
                    break
                _, responses = gen.next()
                lost = gen.take_lane_failures()
                if lost:
                    failures.extend(str(f) for f in lost)
                    break
                for uid in list(live):
                    boundary = gen.pop_prompt_boundary(uid) if hasattr(gen, "pop_prompt_boundary") else None
                    if boundary is not None and records[by_uid[uid]]["boundary"] is None:
                        records[by_uid[uid]]["boundary"] = boundary
                for response in responses:
                    record = records[by_uid[response.uid]]
                    record["tokens"].append(int(response.token))
                    record["widths"].add(int(getattr(response, "execution_width", 1) or 1))
                    receipt = getattr(response, "speculative_receipt", None)
                    if receipt is not None and (not record["receipts"] or record["receipts"][-1] is not receipt):
                        record["receipts"].append(receipt)
                    row = getattr(response, "logprobs", None)
                    if row is not None and len(record["logprob_rows"]) < args.logprob_rows:
                        record["logprob_rows"].append(state_digest([row]))
                    if response.finish_reason:
                        record["finish_reason"] = response.finish_reason
                        record["final"] = response
                        live.discard(response.uid)
                if remove is not None and removed is None:
                    lane, after = remove
                    uid = next(u for u, i in by_uid.items() if i == lane)
                    if uid in live and len(records[lane]["tokens"]) >= after:
                        gen.remove([uid])  # between polls: a closed boundary
                        live.discard(uid)
                        removed = {"lane": lane, "after_tokens": len(records[lane]["tokens"])}
            stats = dict(getattr(gen, "scheduler_stats", {}) or {})
        finally:
            gen.close()
        out = {}
        for i, record in records.items():
            final = record["final"]
            boundary = record["boundary"]
            cache = getattr(final, "prompt_cache", None)
            out[i] = {
                "covered_tokens": covered_tokens(cache),
                "tokens": record["tokens"],
                "token_sha256": _sha(json.dumps(record["tokens"]).encode()),
                "finish_reason": record["finish_reason"],
                "execution_widths": sorted(record["widths"]),
                "logprob_rows": [d["sha256"] for d in record["logprob_rows"]],
                "logprob_rows_status": sorted({d["status"] for d in record["logprob_rows"]}) or ["unavailable"],
                "final_state": state_digest(getattr(final, "prompt_cache", None)),
                "boundary": None if boundary is None else {
                    "covered_tokens": boundary.get("covered_tokens") if isinstance(boundary, dict) else None,
                    "state": state_digest(boundary),
                },
                "rounds": _round_summary(record["receipts"]),
                "_final": final,
            }
        return out, stats, failures, removed

    def continuation(self, lane, record):
        """Greedy ordinary B1 continuation from the lane's final cache."""
        final = record.pop("_final", None)
        if not self.args.continuation_tokens:
            return {"status": "unavailable", "reason": "disabled (--continuation-tokens 0)"}
        covered = record["covered_tokens"]
        if final is None or getattr(final, "prompt_cache", None) is None or covered is None:
            return {"status": "unavailable", "reason": "no final cache or inconsistent offsets"}
        # Each route covers its own prefix of prompt + output (ordinary decode
        # may already hold the last token); feed exactly the uncovered rest.
        full = list(self.prompts[lane]) + list(record["tokens"])
        if not 0 < covered <= len(full):
            return {"status": "unavailable", "reason": f"covered {covered} outside 1..{len(full)}"}
        try:
            out, _stats, failures, _ = self.run(
                "ordinary_b1", [lane], caches=[final.prompt_cache], all_tokens=[full[:covered]],
                prompts=[full[covered:] + list(self.followup)], caps=[self.args.continuation_tokens])
        except Exception as error:  # noqa: BLE001 - recorded as unavailable, never exact
            return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"[:200]}
        if failures:
            return {"status": "unavailable", "reason": "; ".join(failures)[:200]}
        result = out[lane]
        result.pop("_final", None)
        return {"status": "complete", "tokens": result["tokens"], "final_state": result["final_state"]}


def covered_tokens(cache):
    """Tokens a final cache covers (one offset shared by every plane), or None."""
    if not cache:
        return None
    offsets = {int(entry.offset) for entry in cache if hasattr(entry, "offset")}
    return offsets.pop() if len(offsets) == 1 else None


def _round_summary(receipts):
    """Per-lane PLD rounds from receipts (one receipt object per round)."""
    rounds = [(int(r.get("round_proposed", 0)), int(r.get("round_accepted", 0))) for r in receipts]
    rejected = [i for i, (p, a) in enumerate(rounds) if p and a < p]
    return {
        "rounds": len(rounds),
        "proposed": sum(p for p, _ in rounds),
        "accepted": sum(a for _, a in rounds),
        "rejected_rounds": len(rejected),
        "partial_accept_rounds": sum(1 for p, a in rounds if 0 < a < p),
        "rollback_then_append": bool(rejected and rejected[0] < len(rounds) - 1),
    }


# ---------------------------------------------------------------- compare

def compare(arm, reference, lanes, results, *, prefix_only=(), continuation=True):
    """Token and storage-bit parity of ``arm`` against a named ``reference``.

    Token parity and bit parity are separate verdicts: equal tokens with
    different logprob or state bits is reported as bit divergence (different
    forward geometry), never as exact. Final states whose routes cover a
    different number of tokens are ``incomparable``, not equal and not a
    counterexample.
    """
    token_diffs, bit_diffs, incomparable = [], [], []
    for i in lanes:
        got, want = results[arm][i], results[reference][i]
        a, b = got["tokens"], want["tokens"]
        if i in prefix_only:
            if a != b[: len(a)]:
                token_diffs.append(f"lane {i}: removed-lane prefix differs")
            continue
        if a != b:
            index = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            token_diffs.append(f"lane {i}: tokens differ at {index}")
            continue
        rows = min(len(got["logprob_rows"]), len(want["logprob_rows"]))
        if rows == 0 or None in got["logprob_rows"][:rows] + want["logprob_rows"][:rows]:
            incomparable.append(f"lane {i}: logprob rows unavailable")
        elif got["logprob_rows"][:rows] != want["logprob_rows"][:rows]:
            first = next(k for k in range(rows) if got["logprob_rows"][k] != want["logprob_rows"][k])
            bit_diffs.append(f"lane {i}: logprob row bits differ from row {first}")
        g_state, w_state = got["final_state"], want["final_state"]
        if g_state["status"] != "complete" or w_state["status"] != "complete":
            incomparable.append(f"lane {i}: final state unavailable")
        elif got["covered_tokens"] != want["covered_tokens"]:
            incomparable.append(f"lane {i}: final caches cover {got['covered_tokens']} vs "
                                f"{want['covered_tokens']} tokens")
        elif g_state["sha256"] != w_state["sha256"]:
            bit_diffs.append(f"lane {i}: final state bits differ")
        if not continuation:
            continue
        g, w = got.get("continuation", {}), want.get("continuation", {})
        if g.get("status") != "complete" or w.get("status") != "complete":
            incomparable.append(f"lane {i}: continuation unavailable")
        elif g["tokens"] != w["tokens"]:
            token_diffs.append(f"lane {i}: continuation tokens differ")
        elif g["final_state"] != w["final_state"]:
            bit_diffs.append(f"lane {i}: continuation state bits differ")
    return {"arm": arm, "reference": reference, "tokens_exact": not token_diffs,
            "bits_exact": not token_diffs and not bit_diffs and not incomparable,
            "token_differences": token_diffs, "bit_differences": bit_diffs,
            "incomparable": incomparable}


def coverage(driver, results, stats, removed):
    batched, removal = results["pld_batched"], results["pld_removal"]
    lanes = range(len(driver.prompts))
    survivor = [i for i in lanes if removed and i != removed["lane"]]
    items = {
        "unequal_prompt_lengths": len({len(p) for p in driver.prompts}) > 1,
        "unequal_output_caps": len(set(driver.caps)) > 1,
        "zero_proposal_lane": any(batched[i]["rounds"]["proposed"] == 0 for i in lanes),
        "proposals": stats["pld_batched"].get("pld_proposed", 0) > 0,
        "rollback_counter": stats["pld_batched"].get("pld_rollbacks", 0) > 0,
        "rejected_suffix": any(batched[i]["rounds"]["rejected_rounds"] > 0 for i in lanes),
        "partial_acceptance": any(batched[i]["rounds"]["partial_accept_rounds"] > 0 for i in lanes),
        "rollback_then_append": any(batched[i]["rounds"]["rollback_then_append"] for i in lanes),
        "batched_width_ge2": stats["pld_batched"].get("pld_batched_max_width", 0) >= 2,
        "closed_boundary_removal": bool(removed),
        "survivor_ran_at_lower_width": any(
            len(removal[i]["execution_widths"]) > 1 and min(removal[i]["execution_widths"]) < len(driver.prompts)
            for i in survivor
        ),
    }
    return items


def run_all(args):
    driver = Driver(args)
    driver.check_geometry()
    lanes = list(range(len(driver.prompts)))
    results, stats, failures = {}, {}, {}
    removed = None
    for arm in ARMS:
        if arm == "ordinary_b1":
            merged, arm_stats, arm_failures = {}, [], []
            for i in lanes:
                out, st, fl, _ = driver.run(arm, [i])
                merged.update(out)
                arm_stats.append(st)
                arm_failures += fl
            results[arm], stats[arm], failures[arm] = merged, {"per_lane": arm_stats}, arm_failures
        else:
            remove = (args.remove_lane, args.remove_after) if arm == "pld_removal" else None
            out, st, fl, rem = driver.run(arm, lanes, remove=remove)
            results[arm], stats[arm], failures[arm] = out, st, fl
            if arm == "pld_removal":
                removed = rem
        # Finish this arm's continuations now and drop its final responses and
        # caches, so no arm's tensors are alive while the next one runs.
        for i in lanes:
            if arm == "pld_removal":
                results[arm][i].pop("_final", None)
            else:
                results[arm][i]["continuation"] = driver.continuation(i, results[arm][i])
    comparisons = [
        compare("ordinary_bN", PRIMARY_REFERENCE, lanes, results) | {"kind": "ordinary geometry (B1 vs BN)"},
        compare("pld_per_lane", PRIMARY_REFERENCE, lanes, results) | {"kind": "pld vs primary reference"},
        compare("pld_batched", PRIMARY_REFERENCE, lanes, results) | {"kind": "pld vs primary reference"},
        compare("pld_batched", "ordinary_bN", lanes, results) | {"kind": "pld vs batched ordinary (secondary)"},
        compare("pld_per_lane", "ordinary_bN", lanes, results) | {"kind": "pld vs batched ordinary (secondary)"},
    ]
    removal_cmp = compare("pld_removal", PRIMARY_REFERENCE, lanes, results,
                          prefix_only=(removed["lane"],) if removed else (), continuation=False)
    removal_cmp["kind"] = "membership survivor vs primary reference"
    comparisons.append(removal_cmp)
    cov = coverage(driver, results, stats, removed)
    primary = [c for c in comparisons if c["reference"] == PRIMARY_REFERENCE and c["arm"].startswith("pld")]
    pld_token_bad = [c for c in primary if not c["tokens_exact"]]
    pld_bits_bad = [c for c in primary if not c["bits_exact"]]
    engagement = [f for arm, fl in failures.items() for f in (f"{arm}: {x}" for x in fl)]
    for arm in ARMS:
        for i in lanes:
            rec = results[arm][i]
            if arm == "pld_removal" and removed and i == removed["lane"]:
                continue
            cap = driver.caps[i]
            if rec["finish_reason"] is None or (len(rec["tokens"]) < cap and rec["finish_reason"] != "stop"):
                engagement.append(f"{arm} lane {i}: early stop ({len(rec['tokens'])}/{cap}, {rec['finish_reason']})")
    if engagement:
        verdict = "refused"
    elif pld_token_bad:
        verdict = "counterexample"
    elif not all(cov.values()):
        verdict = "coverage_refused"
    elif pld_bits_bad:
        # Greedy tokens match the declared reference but logprob/state bits
        # do not (or cannot be compared): not exact, and not relabeled so.
        verdict = "token_exact_bits_diverge"
    else:
        verdict = "pass"
    from scripts.paired_direct_ab import mlx_identity, source_identity

    return {
        "schema": SCHEMA,
        "scope": ("direct-model ragged PLD (one process); not HTTP serving, not serving "
                  "qualification, no route or default selected; no timing"
                  + ("; TINY random CPU model" if args.tiny else "")),
        "verdict": verdict,
        "primary_reference": PRIMARY_REFERENCE,
        "ordinary_geometry": ("bit_exact" if comparisons[0]["bits_exact"] else
                              "token_exact_bits_diverge" if comparisons[0]["tokens_exact"] else
                              "tokens_diverge"),
        "comparisons": comparisons,
        "coverage": cov,
        "refusals": engagement,
        "removed": removed,
        "lanes": [{"prompt_tokens": len(p), "prompt_sha256": _sha(json.dumps(p).encode()), "max_tokens": c}
                  for p, c in zip(driver.prompts, driver.caps)],
        "results": {arm: {str(i): {k: v for k, v in r.items() if not k.startswith("_")}
                          for i, r in res.items()} for arm, res in results.items()},
        "scheduler_stats": stats,
        "protocol": {"arms": list(ARMS), "sampling": "greedy", "seed": args.seed,
                     "prefill_step": args.prefill_step, "pld_policy": args.pld_policy,
                     "logprob_rows": args.logprob_rows, "continuation_tokens": args.continuation_tokens,
                     "remove_lane": args.remove_lane, "remove_after": args.remove_after,
                     "time_limit_s": args.time_limit_s, "stop_tokens": list(driver.stops)},
        "identity": {"source": source_identity(), "files": {n: _sha((ROOT / n).read_bytes()) for n in IDENTITY_FILES},
                     "mlx": mlx_identity(driver.mx), "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32"),
                     **driver.identity},
    }


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiny", action="store_true", help="deterministic random CPU model; never Metal")
    ap.add_argument("--model")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--prompt-ids", help='JSON {"prompts": [[ids]...], "max_tokens": [..]}')
    ap.add_argument("--lanes", type=int, default=2, help="constructed prompts: 2..4 lanes")
    ap.add_argument("--pld-policy", type=json.loads, default={}, help="PLD policy JSON (batched_verify set per arm)")
    ap.add_argument("--prefill-step", type=int, default=None)
    ap.add_argument("--logprob-rows", type=int, default=16)
    ap.add_argument("--continuation-tokens", type=int, default=4)
    ap.add_argument("--remove-lane", type=int, default=0)
    ap.add_argument("--remove-after", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--time-limit-s", type=float, default=900.0)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--out", required=True)
    return ap


def resolve_args(ap, argv=None):
    a = ap.parse_args(argv)
    if a.tiny:
        if a.i_own_the_gpu or a.model or a.prompt_ids:
            ap.error("--tiny runs a random CPU model; drop --i-own-the-gpu/--model/--prompt-ids")
        a.prefill_step = 8 if a.prefill_step is None else a.prefill_step
    else:
        if not a.i_own_the_gpu:
            ap.error("refusing Metal execution without --i-own-the-gpu")
        if not a.model:
            ap.error("--model is required for a real run")
        a.prefill_step = 2048 if a.prefill_step is None else a.prefill_step
    if "batched_verify" in a.pld_policy:
        ap.error("batched_verify is set per arm; drop it from --pld-policy")
    if not math.isfinite(a.time_limit_s) or a.time_limit_s <= 0:
        ap.error("--time-limit-s must be finite and positive")
    if not 1 <= a.prefill_step <= MAX_PROMPT_TOKENS:
        ap.error(f"--prefill-step 1..{MAX_PROMPT_TOKENS}")
    if not 0 <= a.logprob_rows <= MAX_OUTPUT_TOKENS or not 0 <= a.continuation_tokens <= MAX_OUTPUT_TOKENS:
        ap.error(f"--logprob-rows and --continuation-tokens 0..{MAX_OUTPUT_TOKENS} (0 = explicitly unavailable)")
    if not 2 <= a.lanes <= MAX_LANES or not 1 <= a.remove_after <= MAX_OUTPUT_TOKENS:
        ap.error("invalid bounds")
    return a


def main(argv=None):
    ap = build_parser()
    args = resolve_args(ap, argv)
    record = run_all(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=1, default=sorted) + "\n")
    print(json.dumps({k: record[k] for k in ("verdict", "ordinary_geometry", "coverage", "refusals")}, indent=1))
    for c in record["comparisons"]:
        print(f"{c['arm']} vs {c['reference']} ({c['kind']}): tokens_exact={c['tokens_exact']} "
              f"bits_exact={c['bits_exact']} {(c['token_differences'] + c['bit_differences'] + c['incomparable'])[:3]}")
    return 0 if record["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
