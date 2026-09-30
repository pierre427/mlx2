#!/usr/bin/env python3
"""Bounded GDN rollback-retirement qualification driver (self-MTP, B2/B4).

Scope: DIRECT-MODEL, not HTTP serving, not serving qualification; no route,
default or runtime toggle is added. Timings are single-cohort diagnostics.

Two modes:

``--synthetic-oracle``
    Exact storage-bit oracle on the production ``qwen3_5.GatedDeltaNet``
    layer and ``ArraysCache`` (the recurrent plane every Qwen hybrid uses).
    Replays from a cold or re-chunked cache are NOT a valid exact reference:
    projection bits depend on the forward's row count. Every comparison here
    is geometry-matched: a ragged-trim batch against uniform trims of the
    same history per row; retirement on against off; ``extract(row)``
    against ``filter([row])``; and both accept implementations. It covers
    heterogeneous accepted lengths, short-after-wide verifies, retirement,
    rollback followed by appended decode steps, and lane extraction/removal,
    comparing whole-cache ``state_digest`` (state and ``meta_state``) and
    continuation output bits. Runs on CPU; ``--i-own-the-gpu`` runs it on
    the default (Metal) device.

model mode (``--tiny`` or ``--model``)
    One model load, a width-2 or width-4 self-MTP ``BatchGenerator``
    (policy recorded: num_draft 2, persistent, rate_gate false). Arm
    ``retire`` is the product; arm ``control`` replaces
    ``hybrid_speculative._retire_committed_rollbacks`` with a harness-local
    counter that retires nothing, restored in ``finally``. Fresh generator,
    caches and GC per arm; identical pinned token IDs, greedy sampling,
    stops and caps. Per lane: tokens, logprob-row storage bits, self-MTP
    receipt counters (proposed, accepted, verify histograms, observed
    widths), live rollback records and ``state_digest`` of the final cache,
    and an ordinary B1 continuation of the uncovered suffix. Per arm:
    active/cache/peak memory sampled before ``close``. Bounded: batch 2 or
    4, <= 16384 prompt tokens and <= 512 generated tokens per lane, finite
    polls and time. Refused on zero proposals, zero rejections, no batched
    width, no retirement on ``retire`` or retirement on ``control``.

  PYTHONPATH=src .venv/bin/python scripts/qualify_gdn_retirement.py --synthetic-oracle --out /tmp/o.json
  PYTHONPATH=src .venv/bin/python scripts/qualify_gdn_retirement.py --tiny --out /tmp/r.json
  PYTHONPATH=src .venv/bin/python scripts/qualify_gdn_retirement.py --i-own-the-gpu \\
      --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --batch 4 --context 1024 --gen 64 --out r.json
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

os.environ.setdefault("MLX_ENABLE_TF32", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SCHEMA = "mlx2.direct-model.gdn-retirement.v1"
SELF_MTP_POLICY = {"num_draft": 2, "persistent": True, "rate_gate": False, "prefill_step_size": 2048}
MAX_CONTEXT = 16384
MAX_GEN = 512
MAX_ROWS = 512
MAX_PREFILL_STEP = 16384
IDENTITY_FILES = (
    "scripts/qualify_gdn_retirement.py",
    "scripts/paired_direct_ab.py",
    "src/mlx2/runtime/hybrid_speculative.py",
    "src/mlx2/runtime/generate.py",
    "src/mlx2/runtime/models/cache.py",
    "src/mlx2/runtime/models/qwen3_5.py",
)
FILLER = "The keeper logged wind, swell and cloud each hour; ships passed at dusk. "


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bits(array):
    import mlx.core as mx
    import numpy as np

    view = {2: mx.uint16, 4: mx.uint32}.get(array.dtype.size)
    return np.array(array.view(view) if view else array).tobytes()


# ================================================================ synthetic oracle

def synthetic_oracle(seed=29454):
    """Geometry-matched exact comparisons; returns a list of case records."""
    import mlx.core as mx

    from mlx2.runtime.models import qwen3_5
    from mlx2.runtime.models.cache import ArraysCache
    from scripts.paired_direct_ab import state_digest

    mx.random.seed(seed)
    layer = qwen3_5.GatedDeltaNet(qwen3_5.TextModelArgs(
        hidden_size=32, linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=8, linear_conv_kernel_dim=4,
    ))
    mx.eval(layer.parameters())
    prefix, wide = mx.random.normal((2, 4, 32)), mx.random.normal((2, 5, 32))
    short, append = mx.random.normal((2, 2, 32)), mx.random.normal((2, 2, 32))
    probe = mx.random.normal((2, 1, 32))

    def history(trims, *, retire):
        cache = ArraysCache(2)
        layer(prefix, cache=cache)
        cache.start_speculation()
        for block, trim in zip((wide, short), trims):
            layer(block, cache=cache)
            (cache.trim_ragged if isinstance(trim, list) else cache.trim)(trim)
            if retire:
                cache.retire_rollbacks(keep=1)
        for step in range(append.shape[1]):  # rollback, then appended decode steps
            layer(append[:, step:step + 1], cache=cache)
            if retire:
                cache.retire_rollbacks(keep=1)
        mx.eval(cache.cache)
        return cache

    def row(cache, index):
        lane = cache.extract(index)
        out = layer(probe[index:index + 1], cache=lane)
        mx.eval(out)
        return {"state": state_digest([lane]), "continuation": _sha(_bits(out))}

    def whole(cache):
        out = layer(probe, cache=cache)
        mx.eval(out)
        return {"state": state_digest([cache]), "continuation": _sha(_bits(out))}

    cases = []

    def case(name, got, want, **extra):
        exact = got == want and all(d["state"]["status"] == "complete" for d in (got, want))
        cases.append({"case": name, "exact": exact, **extra})

    previous = qwen3_5._GDN_ARRAY_ACCEPT
    try:
        per_impl = {}
        for impl in (False, True):
            qwen3_5._GDN_ARRAY_ACCEPT = impl  # harness-local, restored below
            label = "per_row_fn" if impl else "rewind"
            ragged = [[3, 1], [1, 2]]  # rows keep (2, 1) and (4, 0) of wide/short
            rows = [(3, 1), (1, 2)]
            for index, uniform in enumerate(rows):
                case(f"{label}: heterogeneous accepted lengths + short-after-wide, row {index}",
                     row(history(ragged, retire=False), index),
                     row(history(list(uniform), retire=False), index),
                     reference="uniform trim of the same history at batch 2")
            retired, kept = history(ragged, retire=True), history(ragged, retire=False)
            case(f"{label}: retirement on vs off (whole batch)", whole(retired), whole(kept),
                 records={"retire": len(retired._rollbacks), "no_retire": len(kept._rollbacks)},
                 reference="same history without retirement")
            removal = history(ragged, retire=True)
            removal.filter([1])
            removed = layer(probe[1:2], cache=removal)
            mx.eval(removed)
            case(f"{label}: lane removal filter([1]) vs extract(1)",
                 {"state": state_digest([removal]), "continuation": _sha(_bits(removed))},
                 row(history(ragged, retire=True), 1), reference="extract(1) of the same history")
            per_impl[label] = whole(history(ragged, retire=True))
        case("accept implementations agree (rewind vs per_row_fn)", per_impl["rewind"], per_impl["per_row_fn"],
             reference="the other production accept implementation")
    finally:
        qwen3_5._GDN_ARRAY_ACCEPT = previous
    return cases


# ================================================================ model mode

def tiny_flash_next():
    """Deterministic tiny qwen4_exp (Flash-Next class) model with an MTP head."""
    import mlx.core as mx

    from mlx2.runtime.models.qwen4_exp import Model, ModelArgs

    mx.random.seed(43)
    text = dict(
        model_type="qwen4_exp_text", hidden_size=32, intermediate_size=0, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=64,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"], num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, hc_count=2, hc_lowrank=8,
        ple_layer_ids=[1], ple_embed_dim=32, ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=128, make_ngram_vocab_size_divisible_by=128, split_ngram_parts=1,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8, indexer_budget=8,
        indexer_compress_ratio=2, mtp_num_hidden_layers=1,
        rope_parameters={"type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.25},
    )
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=text))
    model.eval()
    mx.eval(model.parameters())
    return model


def covered_tokens(cache):
    offsets = set()
    for entry in cache or ():
        offset = getattr(entry, "offset", None)
        if isinstance(offset, int):
            offsets.add(offset)
    return offsets.pop() if len(offsets) == 1 else None


def live_records(cache):
    return sum(len(getattr(entry, "_rollbacks", ()) or ()) for entry in cache or ())


def batch_live_records(gen):
    """Rollback records held by the live self-MTP batch (host lengths only)."""
    state = getattr(getattr(gen, "_generation_batch", None), "state", None)
    caches = getattr(getattr(state, "caches", None), "target", None)
    return None if caches is None else live_records(caches)


class OwnerRefs:
    """Weak references to one arm's tensor owners (never strong ones).

    Tracks generators, final responses, their target and draft cache
    entries and every array in those caches' ``state``; ``alive`` runs GC
    and counts survivors. Objects that cannot be weakly referenced are
    counted as untrackable, which the driver treats as unverified.
    """

    def __init__(self):
        self.refs, self.unreferenceable = [], 0

    def track(self, obj):
        import weakref

        if obj is None:
            return
        try:
            self.refs.append(weakref.ref(obj))
        except TypeError:
            self.unreferenceable += 1

    def track_cache(self, cache):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        for entry in cache or ():
            self.track(entry)
            try:
                leaves = tree_flatten(entry.state)
            except Exception:  # noqa: BLE001 - the entry itself is still tracked
                continue
            for _, leaf in leaves:
                if isinstance(leaf, mx.array):
                    self.track(leaf)

    def track_response(self, response):
        if response is None:
            return
        self.track(response)
        self.track_cache(getattr(response, "prompt_cache", None))
        mtp_state = getattr(response, "mtp_state", None)
        if isinstance(mtp_state, tuple) and mtp_state:
            self.track_cache(mtp_state[0])
            self.track(mtp_state[1] if len(mtp_state) > 1 else None)

    def alive(self):
        import gc

        gc.collect()
        return {"tracked": len(self.refs), "alive": sum(ref() is not None for ref in self.refs),
                "unreferenceable": self.unreferenceable}


class RetirementDriver:
    def __init__(self, args):
        import mlx.core as mx

        self.args, self.mx = args, mx
        self.stops = ()
        if args.tiny:
            mx.set_default_device(mx.cpu)
            self.model = tiny_flash_next()
            self.prompts = [[1 + (i * 5 + j) % 60 for j in range(12 + 5 * i)] for i in range(args.batch)]
            self.identity = {"model": "tiny-random-qwen4_exp", "fingerprint": None}
            return
        from mlx2.adapters.registry import resolve_adapter

        cls = resolve_adapter(args.model, mtp=True, qualification_mode=True)
        # Constructed before any runtime module is imported (import-order guard).
        self.adapter = cls(args.model)
        from mlx2.serving import generation_stop_token_ids

        self.model = self.adapter.model
        self.stops = () if args.ignore_eos else generation_stop_token_ids(self.adapter)
        if args.prompt_ids:
            self.prompts = json.loads(Path(args.prompt_ids).read_text())
            source = f"explicit token file sha256 {_sha(Path(args.prompt_ids).read_bytes())}"
        else:
            tokenizer = self.adapter.tokenizer
            try:
                unit = list(tokenizer.encode(FILLER, add_special_tokens=False))
            except TypeError:
                unit = list(tokenizer.encode(FILLER))
            stream = unit * (args.context * args.batch // len(unit) + 2)
            # Disjoint, unequal slices: lane i gets context - 16 * i tokens.
            self.prompts, start = [], 0
            for i in range(args.batch):
                length = args.context - 16 * i
                self.prompts.append(stream[start:start + length])
                start += length
            source = "deterministic filler constructor"
        self.identity = {
            "model": str(args.model), "adapter": f"{cls.__module__}.{cls.__qualname__}",
            "adapter_sha256": _sha(Path(sys.modules[cls.__module__].__file__).read_bytes()),
            "fingerprint": self.adapter.identity.get("fingerprint"),
            "environment": dict(getattr(self.adapter, "environment", {}) or {}),
            "prompt_source": source,
        }

    def check_geometry(self):
        if len(self.prompts) != self.args.batch or self.args.batch not in (2, 4):
            raise SystemExit("refused: batch must be 2 or 4 with one prompt per lane")
        for prompt in self.prompts:
            if not 8 <= len(prompt) <= MAX_CONTEXT or any(type(t) is not int or t < 0 for t in prompt):
                raise SystemExit(f"refused: each prompt must be 8..{MAX_CONTEXT} token ids")

    def run_arm(self, arm):
        import gc

        from mlx2.runtime import hybrid_speculative as HS
        from mlx2.runtime.generate import BatchGenerator
        from mlx2.runtime.sample_utils import LaneRNG
        from scripts.paired_direct_ab import state_digest

        mx, args = self.mx, self.args
        tally = {"retire_calls": 0, "records_dropped": 0, "suppressed_calls": 0}
        original = HS._retire_committed_rollbacks

        def counted(caches):
            before = sum(len(getattr(c, "_rollbacks", ()) or ()) for c in caches)
            original(caches)
            after = sum(len(getattr(c, "_rollbacks", ()) or ()) for c in caches)
            tally["retire_calls"] += 1
            tally["records_dropped"] += before - after

        def suppressed(caches):
            tally["suppressed_calls"] += 1

        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        lanes = {i: {"tokens": [], "logprob_rows": [], "final": None, "finish_reason": None}
                 for i in range(len(self.prompts))}
        failures, polls, max_live = [], 0, None
        limit = args.gen * len(self.prompts) * 4 + sum(map(len, self.prompts)) // 256 + 256
        deadline = time.monotonic() + args.time_limit_s
        HS._retire_committed_rollbacks = counted if arm == "retire" else suppressed
        gen = None
        try:
            gen = BatchGenerator(self.model, completion_batch_size=len(self.prompts), prefill_batch_size=1,
                                 prefill_step_size=args.prefill_step, stop_tokens=[[t] for t in self.stops],
                                 self_mtp=dict(SELF_MTP_POLICY, prefill_step_size=args.prefill_step))
            started = time.perf_counter()
            uids = gen.insert([list(p) for p in self.prompts], max_tokens=[args.gen] * len(self.prompts),
                              lane_rngs=[LaneRNG(args.seed + i) for i in lanes],
                              self_mtp_configs=[{"sampling_temp": 0.0}] * len(self.prompts))
            by_uid = dict(zip(uids, lanes))
            live = set(uids)
            while live:
                polls += 1
                if polls > limit or time.monotonic() > deadline:
                    failures.append(f"bounded: stopped after {polls} polls")
                    break
                _, responses = gen.next()
                live_now = batch_live_records(gen)
                if live_now is not None:
                    max_live = live_now if max_live is None else max(max_live, live_now)
                lost = gen.take_lane_failures()
                if lost:
                    failures.extend(str(f) for f in lost)
                    break
                for response in responses:
                    lane = lanes[by_uid[response.uid]]
                    lane["tokens"].append(int(response.token))
                    row = getattr(response, "logprobs", None)
                    if len(lane["logprob_rows"]) < args.logprob_rows:
                        # None marks a row the route did not return or that
                        # could not be digested: unavailable, never equal.
                        lane["logprob_rows"].append(
                            None if row is None else state_digest([row])["sha256"])
                    if response.finish_reason:
                        lane["finish_reason"] = response.finish_reason
                        lane["final"] = response
                        live.discard(response.uid)
            mx.synchronize()
            elapsed = time.perf_counter() - started
            memory = {"active_bytes": mx.get_active_memory(), "cache_bytes": mx.get_cache_memory(),
                      "peak_bytes": mx.get_peak_memory(), "note": "sampled before generator close"}
        finally:
            HS._retire_committed_rollbacks = original
            if gen is not None:
                gen.close()
        out = {}
        for i, lane in lanes.items():
            final = lane["final"]
            receipt = getattr(final, "mtp_receipt", None) or {}
            stats = receipt.get("stats", {}) or {}
            cache = getattr(final, "prompt_cache", None)
            out[i] = {
                "tokens": lane["tokens"], "token_sha256": _sha(json.dumps(lane["tokens"]).encode()),
                "finish_reason": lane["finish_reason"], "logprob_rows": lane["logprob_rows"],
                "final_state": state_digest(cache), "covered_tokens": covered_tokens(cache),
                # A finished lane's detached cache has stopped speculating, so
                # this is 0 by construction; see max_live_rollback_records.
                "live_rollback_records_at_finish": live_records(cache),
                "mtp": {"route": receipt.get("route"), "observed_widths": receipt.get("observed_compute_widths"),
                        "num_draft": receipt.get("num_draft"),
                        "draft_cycles": stats.get("draft_cycles", 0), "draft_proposed": stats.get("draft_proposed", 0),
                        "draft_accepted": stats.get("draft_accepted", 0),
                        "verify_span_hist": stats.get("verify_span_hist"),
                        "verify_accept_hist": stats.get("verify_accept_hist")},
                "_final": final,
            }
        tally["max_live_rollback_records"] = max_live  # None: batch caches not observable
        owners = OwnerRefs()
        owners.track(gen)
        for record in out.values():
            owners.track_response(record["_final"])
        return {"lanes": out, "failures": failures, "retirement": tally, "memory": memory,
                "diagnostic_elapsed_s": elapsed, "polls": polls, "_owners": owners}

    def continuation(self, lane, record, owners=None):
        from mlx2.runtime.generate import BatchGenerator
        from scripts.paired_direct_ab import state_digest

        final = record.pop("_final", None)
        if not self.args.continuation_tokens:
            return {"status": "unavailable", "reason": "disabled (--continuation-tokens 0)"}
        covered = record["covered_tokens"]
        full = list(self.prompts[lane]) + record["tokens"]
        if final is None or getattr(final, "prompt_cache", None) is None or covered is None \
                or not 0 < covered <= len(full):
            return {"status": "unavailable", "reason": "no final cache or no single covered offset"}
        gen = BatchGenerator(self.model, completion_batch_size=1, prefill_batch_size=1,
                             prefill_step_size=self.args.prefill_step)
        if owners is not None:
            owners.track(gen)
        tokens = []
        try:
            (uid,) = gen.insert([full[covered:] + list(self.prompts[lane][:2])],
                                max_tokens=[self.args.continuation_tokens],
                                caches=[final.prompt_cache], all_tokens=[full[:covered]])
            last = None
            for _ in range(self.args.continuation_tokens * 8 + 64):
                _, responses = gen.next()
                for response in responses:
                    tokens.append(int(response.token))
                    if response.finish_reason:
                        last = response
                if last is not None:
                    break
            if last is None:
                return {"status": "unavailable", "reason": "continuation did not finish"}
            if owners is not None:
                owners.track_response(last)
            return {"status": "complete", "tokens": tokens, "final_state": state_digest(last.prompt_cache)}
        except Exception as error:  # noqa: BLE001 - unavailable, never exact
            return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"[:200]}
        finally:
            gen.close()


def model_mode(args):
    driver = RetirementDriver(args)
    driver.check_geometry()
    arms, isolation = {}, {}
    order = ("retire", "control")
    previous = None
    for arm in order:
        # The preceding arm keeps host scalars and digests only: every final
        # response, cache, array and generator it owned must be gone (after
        # GC) before this arm allocates, or its memory samples are shared.
        isolation[arm] = previous.alive() if previous is not None else None
        data = driver.run_arm(arm)
        previous = data.pop("_owners")
        for i, record in data["lanes"].items():
            record["continuation"] = driver.continuation(i, record, previous)
            assert "_final" not in record
        arms[arm] = data
        del data
    refusals, differences, incomparable = [], [], []
    for arm, check in isolation.items():
        if check is not None and (check["alive"] or check["unreferenceable"]):
            refusals.append(f"{arm}: preceding arm's tensor owners not released "
                            f"({check['alive']} alive, {check['unreferenceable']} untrackable)")
    for arm in order:
        refusals += [f"{arm}: {f}" for f in arms[arm]["failures"]]
        lanes = arms[arm]["lanes"].values()
        if sum(l["mtp"]["draft_proposed"] for l in lanes) <= 0:
            refusals.append(f"{arm}: no self-MTP proposals")
        if sum(l["mtp"]["draft_proposed"] - l["mtp"]["draft_accepted"] for l in lanes) <= 0:
            refusals.append(f"{arm}: no rejected proposals (no rollback)")
        if not any(max(l["mtp"]["observed_widths"] or [1]) >= 2 for l in lanes):
            refusals.append(f"{arm}: never ran batched (width >= 2)")
        for i, l in arms[arm]["lanes"].items():
            if l["finish_reason"] is None or (len(l["tokens"]) < args.gen and l["finish_reason"] != "stop"):
                refusals.append(f"{arm} lane {i}: early stop ({len(l['tokens'])}/{args.gen})")
    if arms["retire"]["retirement"]["records_dropped"] <= 0:
        refusals.append("retire: no rollback record was retired")
    if arms["control"]["retirement"]["retire_calls"] or arms["control"]["retirement"]["suppressed_calls"] <= 0:
        refusals.append("control: retirement ran or was never reached")
    for i in arms["retire"]["lanes"]:
        a, b = arms["retire"]["lanes"][i], arms["control"]["lanes"][i]
        if a["tokens"] != b["tokens"]:
            differences.append(f"lane {i}: tokens differ")
            continue
        rows = a["logprob_rows"] + b["logprob_rows"]
        if not rows or None in rows:
            reason = "disabled (--logprob-rows 0)" if not args.logprob_rows else "missing or undigestable"
            incomparable.append(f"lane {i}: logprob rows unavailable ({reason})")
        elif a["logprob_rows"] != b["logprob_rows"]:
            differences.append(f"lane {i}: logprob row bits differ")
        if "complete" not in (a["final_state"]["status"], b["final_state"]["status"]) or \
                a["final_state"]["status"] != b["final_state"]["status"]:
            incomparable.append(f"lane {i}: final state {a['final_state']['status']}/{b['final_state']['status']}")
        elif a["final_state"] != b["final_state"]:
            differences.append(f"lane {i}: final state digest differs")
        ca, cb = a["continuation"], b["continuation"]
        if ca.get("status") != "complete" or cb.get("status") != "complete":
            incomparable.append(f"lane {i}: continuation unavailable")
        elif ca != cb:
            differences.append(f"lane {i}: continuation differs")
    verdict = ("refused" if refusals else "counterexample" if differences
               else "exact_with_unavailable_parts" if incomparable else "pass")
    return driver, {
        "verdict": verdict, "refusals": refusals, "differences": differences, "incomparable": incomparable,
        "arm_isolation": isolation,
        "arms": {arm: {**data, "lanes": {str(i): r for i, r in data["lanes"].items()}}
                 for arm, data in arms.items()},
        "lanes": [{"prompt_tokens": len(p), "prompt_sha256": _sha(json.dumps(p).encode())} for p in driver.prompts],
        "protocol": {"batch": args.batch, "context": args.context, "gen": args.gen, "self_mtp": SELF_MTP_POLICY,
                     "prefill_step": args.prefill_step, "seed": args.seed, "sampling": "greedy",
                     "arm_order": list(order), "logprob_rows": args.logprob_rows,
                     "continuation_tokens": args.continuation_tokens, "time_limit_s": args.time_limit_s,
                     "stop_tokens": list(driver.stops),
                     "timing": "diagnostic_elapsed_s is one cohort per arm, not a controlled performance run"},
        "identity": dict(driver.identity),
    }


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synthetic-oracle", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="deterministic tiny Flash-Next-class CPU model")
    ap.add_argument("--model")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--prompt-ids", help="JSON list of per-lane token id lists")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--context", type=int, default=1024, help="constructed prompt tokens per lane (<= 16384)")
    ap.add_argument("--gen", type=int, default=None, help="generated tokens per lane (<= 512)")
    ap.add_argument("--prefill-step", type=int, default=2048)
    ap.add_argument("--logprob-rows", type=int, default=16)
    ap.add_argument("--continuation-tokens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time-limit-s", type=float, default=1800.0)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--out", required=True)
    return ap


def resolve_args(ap, argv=None):
    a = ap.parse_args(argv)
    if a.synthetic_oracle:
        if a.tiny or a.model:
            ap.error("--synthetic-oracle runs alone")
        return a
    if a.tiny:
        if a.i_own_the_gpu or a.model or a.prompt_ids:
            ap.error("--tiny runs a random CPU model; drop --i-own-the-gpu/--model/--prompt-ids")
        a.gen = 24 if a.gen is None else a.gen
        a.prefill_step = 8 if a.prefill_step == 2048 else a.prefill_step
    else:
        if not a.i_own_the_gpu:
            ap.error("refusing Metal execution without --i-own-the-gpu")
        if not a.model:
            ap.error("--model is required for a real run")
        a.gen = 64 if a.gen is None else a.gen
    if a.batch not in (2, 4):
        ap.error("--batch must be 2 or 4 (the authorized bounded cells)")
    if not 8 <= a.context <= MAX_CONTEXT or not 1 <= a.gen <= MAX_GEN:
        ap.error("--context 8..16384 and --gen 1..512 per lane")
    if not math.isfinite(a.time_limit_s) or a.time_limit_s <= 0:
        ap.error("--time-limit-s must be finite and positive")
    if not 1 <= a.prefill_step <= MAX_PREFILL_STEP:
        ap.error(f"--prefill-step 1..{MAX_PREFILL_STEP}")
    if not 0 <= a.logprob_rows <= MAX_ROWS or not 0 <= a.continuation_tokens <= MAX_ROWS:
        ap.error(f"--logprob-rows and --continuation-tokens 0..{MAX_ROWS} (0 = explicitly unavailable)")
    return a


def main(argv=None):
    ap = build_parser()
    args = resolve_args(ap, argv)
    import mlx.core as mx

    from scripts.paired_direct_ab import mlx_identity, source_identity

    common = {"source": source_identity(), "files": {n: _sha((ROOT / n).read_bytes()) for n in IDENTITY_FILES},
              "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32")}
    if args.synthetic_oracle:
        if not args.i_own_the_gpu:
            mx.set_default_device(mx.cpu)
        cases = synthetic_oracle()
        record = {
            "schema": SCHEMA, "mode": "synthetic_oracle",
            "scope": ("geometry-matched exact storage-bit oracle on qwen3_5.GatedDeltaNet + ArraysCache; "
                      f"device {mx.default_device()}; not a model or serving qualification"),
            "verdict": "pass" if all(c["exact"] for c in cases) else "counterexample",
            "cases": cases, "identity": {**common, "mlx": mlx_identity(mx)},
        }
    else:
        driver, body = model_mode(args)
        record = {"schema": SCHEMA, "mode": "model",
                  "scope": ("direct-model bounded self-MTP retirement A/B; not HTTP serving, not serving "
                            "qualification; no route, default or toggle added"
                            + ("; TINY random CPU model" if args.tiny else "")),
                  **body}
        record["identity"].update(common, mlx=mlx_identity(driver.mx))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    summary = {k: record.get(k) for k in ("mode", "verdict", "refusals", "differences", "incomparable")}
    if record["mode"] == "synthetic_oracle":
        summary["cases"] = [(c["case"], c["exact"]) for c in record["cases"]]
    print(json.dumps(summary, indent=1))
    return 0 if record["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
