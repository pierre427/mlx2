#!/usr/bin/env python3
"""Paired direct-model B1 A/B for implemented default-off mechanisms.

Scope: DIRECT-MODEL. One process loads the model once through its adapter and
drives the runtime generator itself; nothing here is HTTP serving evidence,
serving qualification, or a whole-model speed claim.

Mechanisms (one per cohort):

``qsdpa-tiling``
    Ordinary B1 ``BatchGenerator`` with quantized KV. Arm ``off`` runs the
    composed quantized SDPA untiled (budget 0), arm ``on`` with
    ``--budget-bytes``. Both arms use the same ``--kv-bits``/``--kv-group``
    codec, so the comparison is tiling only: an exact tiling result never
    qualifies approximate KV. Engagement: ``qsdpa_verify_metal.STATS
    ['composed_tiled_calls']`` rises on ``on`` and stays flat on ``off``.
    Tiling applies to composed prefill rows below the flash threshold (128
    for GQA), so choose ``--prompt-tokens`` with a final prefill chunk of
    35..127 rows (e.g. 16504 at ``--prefill-step 8192``).
``external-reclaim``
    B1 external-draft route (``adapter.create_external_batch`` with an
    explicit ``--policy``). Arm ``reclaim`` is the product: the pool clears
    when emitted responses cross a multiple of 256. Arm ``control`` is a
    harness-local suppression on that generator instance (never a product
    switch) that only counts the crossings it skipped. Engagement:
    ``external_allocator_reclaims`` > 0 on ``reclaim``, 0 on ``control``,
    and the control arm crossed the boundary too; a run too short to cross
    it is refused, and so is one whose draft never proposed (an ordinary
    fallback is not the external route). Each run records draft rounds,
    proposals and acceptances, the most responses one poll returned, and
    diagnostic active/cache/peak samples at every crossed 256-response
    quotient (same method on both arms; at least three pairs of 1024 tokens,
    the real default, are needed to see a plateau). Staged external-draft
    *prefill* reclaim is a separate candidate and is not exercised here.

Protocol: fresh generator and caches per run; identical pinned prompt token
IDs, ``LaneRNG(--seed)``, sampler, stop tokens and ``--max-tokens`` on both
arms; ``--warmups`` discarded rounds (both arms); ``--pairs`` measured pairs
alternating AB/BA. Each run records TTFT, decode time, tokens, the token-ID
hash, hashes of the first ``--logprob-rows`` full logprob rows where the
route returns them (host copies are symmetric across arms; the count
compared is recorded, and the ordinary route must return every requested
row), and MLX memory: peak (reset per run), plus active and cache sampled
in-run before the generator closes and the highest cache seen between polls.
A lane the generator drops, or a run that stops responding, is refused
rather than waited on. After timing, the final target cache and the draft
sidecar are hashed; differing hashes are a counterexample, and a missing
one is reported as ``unavailable`` (never as a passing state check).

Verdict: ``pass`` when every run engaged as required and every run's tokens
(and logprob rows where available) equal the first run's; ``counterexample``
when engaged but outputs differ; ``refused`` otherwise. Exit 0/1/1.

Real runs need ``--i-own-the-gpu``. ``--tiny`` runs random CPU models (the
harness and gates only; numbers are meaningless) and never touches Metal.

  PYTHONPATH=src .venv/bin/python scripts/paired_direct_ab.py --tiny \\
      --mechanism qsdpa-tiling --out /tmp/qsdpa-tiny.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARMS = {"qsdpa-tiling": ("off", "on"), "external-reclaim": ("reclaim", "control")}
IDENTITY_FILES = (
    "scripts/paired_direct_ab.py",
    "src/mlx2/runtime/models/base.py",
    "src/mlx2/runtime/models/qsdpa_verify_metal.py",
    "src/mlx2/runtime/generate.py",
    "src/mlx2/runtime/external_speculative.py",
    "src/mlx2/runtime/sample_utils.py",
    "src/mlx2/adapters/registry.py",
)
# Polls without any response before a run is declared stuck (prefill polls
# return nothing; a dropped lane never finishes).
IDLE_POLL_LIMIT = 4096
FILLER = (
    "The lighthouse keeper logged the weather every hour: wind, swell, cloud "
    "and the colour of the sea at dusk. Ships passed, some slowed, most did not. "
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_identity():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "src", "scripts")),
        "files": {name: _sha((ROOT / name).read_bytes()) for name in IDENTITY_FILES},
    }


def mlx_identity(mx):
    try:
        package = metadata.version("mlx")
    except metadata.PackageNotFoundError:
        package = None
    root = Path(mx.__file__).resolve().parent
    metallib = root / "lib" / "mlx.metallib"
    return {"version": mx.__version__, "package": package, "path": str(root),
            "metallib_sha256": _sha(metallib.read_bytes()) if metallib.exists() else None,
            "device": str(mx.default_device())}


# ---------------------------------------------------------------- models

def tiny_ordinary():
    import mlx.core as mx
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    mx.random.seed(7)
    model = TextModel(TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=64, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=4,
        mtp_num_hidden_layers=0, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=1024,
    ))
    model.eval()
    mx.eval(model.parameters())
    return model


def tiny_external():
    import mlx.core as mx
    from mlx2.adapters.muse_glimmer_config import ModelArgs
    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
    from mlx2.runtime.drafters.dflash2_config import DFlash2Config
    from mlx2.runtime.models.muse_glimmer import Model

    mx.random.seed(8)
    target = Model(ModelArgs(
        hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=128, sliding_window=8, max_position_embeddings=2048,
    ))
    draft = DFlash2DraftModel(DFlash2Config(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=128, num_target_layers=4, target_layer_ids=[0, 3],
        conv_kernel_size=2, conv_group_size=2, selector_rank=4,
        selector_top_k=4, block_size=4, mask_token_id=127,
        max_position_embeddings=2048, sliding_window=8,
        layer_types=["sliding_attention"] * 2,
    )).bind(target)
    target.eval()
    mx.eval(target.parameters(), draft.parameters())
    return target, draft


class Cohort:
    """One model load: how to build a fresh generator for one run."""

    def __init__(self, args):
        import mlx.core as mx

        self.args = args
        if args.tiny:
            mx.set_default_device(mx.cpu)
        self.mx = mx
        self.stops = ()
        if args.tiny:
            self.identity = {"model": "tiny-random", "fingerprint": None}
            if args.mechanism == "qsdpa-tiling":
                self.model = tiny_ordinary()
            else:
                self.model, self.draft = tiny_external()
            self.prompt = [1 + (i * 7) % 120 for i in range(args.prompt_tokens)]
            return
        from mlx2.adapters.registry import resolve_adapter
        from mlx2.serving import generation_stop_token_ids

        cls = resolve_adapter(args.model, mtp=False, qualification_mode=True)
        policy = None
        if args.mechanism == "external-reclaim":
            policy = json.loads(Path(args.policy).read_text())
            self.adapter = cls(args.model, execution_policy=policy)
            if getattr(self.adapter, "draft_model", None) is None:
                raise SystemExit("refused: the policy bound no external drafter")
        else:
            self.adapter = cls(args.model)
        self.model = self.adapter.model
        adapter_file = Path(sys.modules[cls.__module__].__file__)
        self.identity = {
            "adapter_sha256": _sha(adapter_file.read_bytes()),
            "model": str(args.model),
            "adapter": f"{cls.__module__}.{cls.__qualname__}",
            "fingerprint": self.adapter.identity.get("fingerprint"),
            "policy": policy,
            "policy_sha256": _sha(Path(args.policy).read_bytes()) if policy else None,
        }
        if not args.ignore_eos:
            self.stops = generation_stop_token_ids(self.adapter)
        self.prompt = self._prompt()

    def _prompt(self):
        args = self.args
        if args.prompt_ids:
            ids = json.loads(Path(args.prompt_ids).read_text())
            if not ids or not all(type(t) is int and t >= 0 for t in ids):
                raise SystemExit("refused: --prompt-ids must be a non-empty list of token ids")
            return ids
        tokenizer = self.adapter.tokenizer
        try:
            unit = list(tokenizer.encode(FILLER, add_special_tokens=False))
        except TypeError:
            unit = list(tokenizer.encode(FILLER))
        ids = (unit * (args.prompt_tokens // len(unit) + 1))[: args.prompt_tokens]
        return ids

    def generator(self):
        args = self.args
        stops = [[token] for token in self.stops]
        if args.mechanism == "qsdpa-tiling":
            from mlx2.runtime.generate import BatchGenerator

            return BatchGenerator(
                self.model, completion_batch_size=1, prefill_batch_size=1,
                prefill_step_size=args.prefill_step, kv_bits=args.kv_bits,
                kv_group_size=args.kv_group, stop_tokens=stops,
            )
        if args.tiny:
            from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

            return ExternalDraftBatchGenerator(
                self.model, draft_model=self.draft, binding="tiny",
                completion_batch_size=1, prefill_step_size=args.prefill_step,
                num_draft=2, stop_tokens=stops,
            )
        return self.adapter.create_external_batch(
            completion_batch_size=1, prefill_step_size=args.prefill_step,
            stop_tokens=stops,
        )


# ---------------------------------------------------------------- arms

def state_hash(obj):
    """sha256 over a cache/sidecar tree's arrays and scalars, or None.

    Each cache contributes its ``state`` (the arrays a snapshot would keep);
    other objects contribute their attributes. None means unavailable, never
    a passing state comparison.
    """
    if obj is None:
        return None
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    def plain(node):
        if isinstance(node, (list, tuple)):
            return [plain(item) for item in node]
        if isinstance(node, dict):
            return {str(k): plain(v) for k, v in sorted(node.items(), key=lambda kv: str(kv[0]))}
        if isinstance(node, mx.array) or node is None or isinstance(node, (int, float, str, bool)):
            return node
        if hasattr(node, "state"):
            return plain(node.state)
        if hasattr(node, "__dict__"):
            return plain(vars(node))
        return repr(node)

    try:
        digest = hashlib.sha256()
        for key, leaf in tree_flatten(plain(obj)):
            digest.update(key.encode())
            if isinstance(leaf, mx.array):
                digest.update(f"{leaf.dtype}{leaf.shape}".encode())
                if leaf.dtype == mx.bfloat16:
                    leaf = leaf.view(mx.uint16)
                digest.update(np.array(leaf).tobytes())
            else:
                digest.update(repr(leaf).encode())
        return digest.hexdigest()
    except Exception:  # noqa: BLE001 - unavailable is recorded, never passed
        return None


def _set_budget(value):
    from mlx2.runtime.models import base

    previous = base._QSDPA_SCORES_BUDGET
    base._QSDPA_SCORES_BUDGET = int(value)
    return previous


def _suppress_reclaim(batch, tally):
    """Harness-local control: count the crossings, never clear the pool."""
    from mlx2.runtime.generate import (
        ALLOCATOR_RECLAIM_MTP_TOKEN_INTERVAL,
        _crossed_counter_interval,
    )

    def suppressed(count):
        previous = batch._emitted_responses
        batch._emitted_responses = previous + int(count)
        if _crossed_counter_interval(
            previous, batch._emitted_responses, ALLOCATOR_RECLAIM_MTP_TOKEN_INTERVAL
        ):
            tally["suppressed_crossings"] += 1

    batch._reclaim_after_emission = suppressed


def run_arm(cohort, arm):
    """One fresh-cache B1 generation; returns its record."""
    import gc

    import numpy as np

    from mlx2.runtime.models import qsdpa_verify_metal as qvm
    from mlx2.runtime.sample_utils import LaneRNG

    mx, args = cohort.mx, cohort.args
    previous_budget = None
    if args.mechanism == "qsdpa-tiling":
        previous_budget = _set_budget(args.budget_bytes if arm == "on" else 0)
    tiled_before = qvm.STATS.get("composed_tiled_calls", 0)
    tiles_before = qvm.STATS.get("composed_tile_calls", 0)
    tally = {"suppressed_crossings": 0}
    failures, cache_high = [], 0
    gc.collect()  # the previous run's generator may sit in a reference cycle
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    batch = cohort.generator()
    from mlx2.runtime.generate import ALLOCATOR_RECLAIM_MTP_TOKEN_INTERVAL as INTERVAL

    tokens, rows, first = [], [], None
    boundary_samples, max_per_poll, final = [], 0, None
    try:
        if arm == "control":
            _suppress_reclaim(batch, tally)
        started = time.perf_counter()
        if args.mechanism == "qsdpa-tiling":
            (uid,) = batch.insert([list(cohort.prompt)], max_tokens=[args.max_tokens],
                                  lane_rngs=[LaneRNG(args.seed)])
        else:
            (uid,) = batch.insert([list(cohort.prompt)], max_tokens=[args.max_tokens],
                                  sampling_configs=[{"sampling_temp": 0.0}],
                                  lane_rngs=[LaneRNG(args.seed)])
        done = False
        idle = 0
        while not done:
            _, responses = batch.next()
            cache_high = max(cache_high, mx.get_cache_memory())
            lost = batch.take_lane_failures()
            if lost:
                failures.extend(str(f) for f in lost)
                break
            idle = 0 if responses else idle + 1
            before = len(tokens)
            if idle > IDLE_POLL_LIMIT:
                failures.append(f"no response in {IDLE_POLL_LIMIT} polls")
                break
            for response in responses:
                if response.uid != uid:
                    continue
                if first is None:
                    mx.synchronize()
                    first = time.perf_counter()
                tokens.append(int(response.token))
                logprobs = getattr(response, "logprobs", None)
                if logprobs is not None and len(rows) < args.logprob_rows:
                    rows.append(_sha(np.array(logprobs.astype(mx.float32)).tobytes()))
                if getattr(response, "finish_reason", None):
                    done = True
                    final = response
            max_per_poll = max(max_per_poll, len(tokens) - before)
            if args.mechanism == "external-reclaim" and len(tokens) // INTERVAL > before // INTERVAL:
                # Diagnostic pool samples at each crossed quotient, same method
                # on both arms; host counters only, no device sync.
                boundary_samples.append({
                    "emitted": len(tokens), "active_bytes": mx.get_active_memory(),
                    "cache_bytes": mx.get_cache_memory(), "peak_bytes": mx.get_peak_memory(),
                })
        mx.synchronize()
        ended = time.perf_counter()
        stats = dict(getattr(batch, "scheduler_stats", {}) or {})
        # In-run memory, sampled before close() returns the lane buffers.
        active_end, cache_end = mx.get_active_memory(), mx.get_cache_memory()
        peak_end = mx.get_peak_memory()
        # After timing: final target cache and draft sidecar state.
        target_state = state_hash(getattr(final, "prompt_cache", None))
        sidecar_state = state_hash(getattr(final, "cache_sidecar", None))
    finally:
        batch.close()
        if previous_budget is not None:
            _set_budget(previous_budget)
    record = {
        "arm": arm,
        "tokens": len(tokens),
        "token_sha256": _sha(json.dumps(tokens).encode()),
        "logprob_row_sha256": rows,
        "logprob_rows_compared": len(rows),
        "lane_failures": failures,
        "ttft_s": (first or ended) - started,
        "decode_s": ended - (first or ended),
        "decode_tok_s": (len(tokens) - 1) / (ended - first) if first and ended > first and len(tokens) > 1 else 0.0,
        "active_bytes": active_end,
        "cache_bytes": cache_end,
        "cache_high_bytes": max(cache_high, cache_end),
        "peak_bytes": peak_end,
        "memory_note": "in-run samples before generator close; diagnostic, not a performance claim",
        "boundary_samples": boundary_samples,
        "max_responses_per_poll": max_per_poll,
        "final_target_state_sha256": target_state,
        "final_draft_sidecar_sha256": sidecar_state,
        "counters": {
            "composed_tiled_calls": qvm.STATS.get("composed_tiled_calls", 0) - tiled_before,
            "composed_tiles": qvm.STATS.get("composed_tile_calls", 0) - tiles_before,
            "external_allocator_reclaims": stats.get("external_allocator_reclaims", 0),
            "suppressed_crossings": tally["suppressed_crossings"],
            "external_rounds": stats.get("external_rounds", 0),
            "proposed_tokens": stats.get("proposed_tokens", 0),
            "accepted_proposals": stats.get("accepted_proposals", 0),
        },
    }
    record["_tokens"] = tokens
    return record


def engagement_refusal(mechanism, record, logprob_rows=0):
    """Why a run did not exercise its arm as required, or None."""
    from mlx2.runtime.generate import ALLOCATOR_RECLAIM_MTP_TOKEN_INTERVAL

    c, arm = record["counters"], record["arm"]
    if record["lane_failures"]:
        return "lane failed: " + "; ".join(record["lane_failures"])[:200]
    if mechanism == "qsdpa-tiling":
        if record["logprob_rows_compared"] < min(logprob_rows, record["tokens"]):
            return "ordinary route returned fewer logprob rows than requested"
        if arm == "on" and c["composed_tiled_calls"] <= 0:
            return "tiling not engaged on the on arm"
        if arm == "off" and c["composed_tiled_calls"] != 0:
            return "tiling engaged on the off arm"
        return None
    if record["tokens"] < ALLOCATOR_RECLAIM_MTP_TOKEN_INTERVAL:
        return (f"{record['tokens']} tokens: too short to cross the "
                f"{ALLOCATOR_RECLAIM_MTP_TOKEN_INTERVAL}-response reclamation boundary")
    if c["external_rounds"] <= 0 or c["proposed_tokens"] <= 0:
        return "external draft never proposed (ordinary fallback, not the external route)"
    if arm == "reclaim" and c["external_allocator_reclaims"] <= 0:
        return "reclaim not engaged on the reclaim arm"
    if arm == "control" and (c["external_allocator_reclaims"] != 0 or c["suppressed_crossings"] <= 0):
        return "control arm reclaimed or never crossed the boundary"
    return None


def run_cohort(args):
    cohort = Cohort(args)
    first, second = ARMS[args.mechanism]
    for _ in range(args.warmups):
        for arm in (first, second):
            run_arm(cohort, arm)
    runs, refusals = [], []
    for pair in range(args.pairs):
        order = (first, second) if pair % 2 == 0 else (second, first)
        for arm in order:
            record = run_arm(cohort, arm)
            record["pair"] = pair
            reason = engagement_refusal(args.mechanism, record, args.logprob_rows)
            if reason:
                refusals.append(f"pair {pair} {arm}: {reason}")
            runs.append(record)
    reference = runs[0] if runs else None
    mismatches = []
    for record in runs[1:]:
        if record["_tokens"] != reference["_tokens"]:
            index = next((i for i, (a, b) in enumerate(zip(record["_tokens"], reference["_tokens"])) if a != b),
                         min(len(record["_tokens"]), len(reference["_tokens"])))
            mismatches.append(f"pair {record['pair']} {record['arm']}: tokens differ at {index}")
        elif record["logprob_row_sha256"] != reference["logprob_row_sha256"]:
            mismatches.append(f"pair {record['pair']} {record['arm']}: logprob rows differ")
        for key in ("final_target_state_sha256", "final_draft_sidecar_sha256"):
            if record[key] is not None and reference[key] is not None and record[key] != reference[key]:
                mismatches.append(f"pair {record['pair']} {record['arm']}: {key} differs")
    for record in runs:
        del record["_tokens"]
    summary = {}
    for arm in (first, second):
        mine = [r for r in runs if r["arm"] == arm]
        if mine:
            summary[arm] = {
                key: statistics.median(r[key] for r in mine)
                for key in ("ttft_s", "decode_tok_s", "peak_bytes", "active_bytes", "cache_bytes",
                            "cache_high_bytes")
            }
    verdict = "refused" if refusals or not runs else ("counterexample" if mismatches else "pass")
    return {
        "schema": "mlx2.direct-model.paired-ab.v1",
        "scope": ("direct-model B1 A/B (adapter + generator in one process); not HTTP "
                  "serving, not serving qualification, not a controlled performance run"
                  + ("; TINY random CPU models, numbers meaningless" if args.tiny else "")),
        "mechanism": args.mechanism,
        "verdict": verdict,
        "refusals": refusals,
        "mismatches": mismatches,
        "arms": [first, second],
        "state_comparison": {
            key: ("compared" if runs and all(r[key] is not None for r in runs) else "unavailable")
            for key in ("final_target_state_sha256", "final_draft_sidecar_sha256")
        },
        "protocol": {
            "pairs": args.pairs, "warmups_discarded": args.warmups, "order": "alternating AB/BA",
            "max_tokens": args.max_tokens, "seed": args.seed, "sampling": "greedy",
            "ignore_eos": bool(args.ignore_eos), "prefill_step": args.prefill_step,
            "kv_bits": args.kv_bits, "kv_group": args.kv_group,
            "budget_bytes": args.budget_bytes if args.mechanism == "qsdpa-tiling" else None,
            "logprob_rows": args.logprob_rows,
        },
        "prompt": {"tokens": len(cohort.prompt), "sha256": _sha(json.dumps(cohort.prompt).encode()),
                   "stop_tokens": list(cohort.stops)},
        "identity": {"source": source_identity(), "mlx": mlx_identity(cohort.mx), **cohort.identity},
        "runs": runs,
        "median_by_arm": summary,
    }


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mechanism", choices=sorted(ARMS), required=True)
    ap.add_argument("--model")
    ap.add_argument("--policy", help="external-draft policy JSON (external-reclaim)")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="random CPU models; never Metal")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--prompt-tokens", type=int)
    ap.add_argument("--prompt-ids", help="JSON list of pinned prompt token ids")
    ap.add_argument("--prefill-step", type=int)
    ap.add_argument("--kv-bits", type=int)
    ap.add_argument("--kv-group", type=int, default=64)
    ap.add_argument("--budget-bytes", type=int, help="on-arm score budget (16 MiB real, 1 KiB tiny)")
    ap.add_argument("--logprob-rows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ignore-eos", action="store_true", help="both arms; fixed-length rows")
    ap.add_argument("--out", required=True)
    return ap


def resolve_args(ap, argv=None):
    a = ap.parse_args(argv)
    qsdpa = a.mechanism == "qsdpa-tiling"
    if a.tiny:
        if a.i_own_the_gpu or a.model or a.policy:
            ap.error("--tiny runs random CPU models; drop --i-own-the-gpu/--model/--policy")
        defaults = (64, 8, 8, 4096, 1024) if qsdpa else (8, 300, None, 2048, None)
        a.ignore_eos = True
    else:
        if not a.i_own_the_gpu:
            ap.error("refusing Metal execution without --i-own-the-gpu")
        if not a.model:
            ap.error("--model is required for a real run")
        if not qsdpa and not a.policy:
            ap.error("external-reclaim needs an explicit --policy")
        defaults = (16504, 128, 8, 8192, 16 << 20) if qsdpa else (512, 1024, None, 2048, None)
    prompt, max_tokens, kv_bits, step, budget = defaults
    a.budget_bytes = budget if a.budget_bytes is None else a.budget_bytes
    a.prompt_tokens = prompt if a.prompt_tokens is None else a.prompt_tokens
    a.max_tokens = max_tokens if a.max_tokens is None else a.max_tokens
    a.prefill_step = step if a.prefill_step is None else a.prefill_step
    if qsdpa:
        a.kv_bits = kv_bits if a.kv_bits is None else a.kv_bits
        if a.kv_bits not in (4, 8):
            ap.error("qsdpa-tiling needs quantized KV: --kv-bits 4 or 8 (same codec on both arms)")
        if a.budget_bytes is None or a.budget_bytes <= 0:
            ap.error("--budget-bytes must be positive")
    elif a.kv_bits is not None:
        ap.error("--kv-bits applies to qsdpa-tiling only")
    if min(a.pairs, a.max_tokens, a.prompt_tokens, a.prefill_step) <= 0 or a.warmups < 0 or a.logprob_rows < 0:
        ap.error("--pairs/--max-tokens/--prompt-tokens/--prefill-step must be positive")
    return a


def main(argv=None):
    ap = build_parser()
    args = resolve_args(ap, argv)
    record = run_cohort(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("mechanism", "verdict", "refusals", "mismatches", "median_by_arm")},
                     indent=1))
    return 0 if record["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
