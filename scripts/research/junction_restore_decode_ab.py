#!/usr/bin/env python3
"""Interleaved decode-rate A/B: fresh prefill cache vs APCv2 junction restore.

Original mlx2 script (2026-11-22).  Question it answers: on the GSM8K A/B of
Inkling-2-Mini@440956c8 the ``apc_junction_checkpoints`` arm decoded at 27.9
tok/s median against 34.7 for the arm without it, but the arms ran one after
the other and two arms with identical tokens drifted 37.8 -> 34.7 tok/s the
same way.  Is decoding from a junction-restored cache slower per step, or was
it drift?

Scope: DIRECT-MODEL, one process, one model load.  Nothing here is HTTP
serving evidence, qualification, or a publishable speed claim (no thermal
hold): it is an attribution check whose two arms share every condition
except the cache they decode from.

Method, per route (``ordinary`` = ``BatchGenerator`` on the adapter's
model; ``dspark`` = ``adapter.create_external_batch`` under ``--policy``)
and per measured question:

1. Prompts are the GSM8K driver's own ``chat-fewshot`` request bodies
   (``scripts/spec_acceptance_gsm8k.py``: ``few_shot_turns`` + question +
   ``CHAT_INSTRUCTION``, ``reasoning_effort`` from ``--request``) rendered by
   the adapter's ``prompt_tokens``.  The junction is the longest common
   prefix of the two planting prompts (``--plant-indices``, default the run's
   first two questions, as the server found it).
2. Plant: a fresh generator prefills the second planting prompt with a
   ``StateBoundary(junction, JUNCTION)``; the generator's own boundary
   capture (``drain_state_checkpoints``) yields the checkpoint, which goes
   through ``serving.detach_checkpoint_graph`` and ``APCv2.store(...,
   retention_role="junction")`` -- the path serving's ``publish_interior``
   takes.
3. Arm ``fresh``: a new generator cold-prefills the measured prompt.  Arm
   ``restored``: ``APCv2.lookup`` (``require_sidecar`` on dspark, as
   serving) must hit at exactly the junction; ``materialize_restored_state``
   then ``insert(remaining, caches=[hit.cache], cache_states=[hit.sidecar])``
   -- serving's warm-hit insert.  Both lanes prefill to their first token
   before any timing.
4. One untimed warm-up block per arm, then ``--pairs`` timed pairs of
   ``--block-tokens`` each, alternating AB/BA (``interleave_schedule``).  Each
   block is bracketed by a synchronize of the generation stream (bound as
   serving's generation thread binds it) so a pipelined round left by one arm
   is charged to that arm.  Per block: tokens, seconds, verify rounds
   and accepted proposals (dspark scheduler counters), peak memory.
5. One untimed probe block per arm with ``mx.contiguous`` and
   ``mx.concatenate`` wrapped by counters (Python-visible copies).
6. Layout facts after prefill and after decode, per cache array: class,
   offset, shape, dtype, strides, C-contiguity, and whether its buffer
   overlaps a buffer of the stored APCv2 entry (a live COW alias).

EOS is ignored by default (``--stop-at-eos`` keeps it) so both arms decode the
same number of tokens.  The verdict (``decode_verdict``) compares the paired
restored/fresh rate ratios: token rate when both arms emitted identical
tokens, verify-round rate otherwise (different tokens change acceptance).

``--tiny`` runs the whole flow on CPU with the tiny random models of
``scripts/paired_direct_ab.py`` (plumbing check only; no ownership needed).
A real run needs ``--cpg-lock`` (the qualification lease) and refuses a
nonempty ``--output``.  Writes ``receipt.json`` atomically.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "mlx2.junction-restore-decode-ab.v1"
ARMS = ("fresh", "restored")
ROUTES = ("ordinary", "dspark")


# ------------------------------------------------------------ pure logic --

def interleave_schedule(pairs):
    """``[(pair, arm), ...]``: pair i runs AB when i is even, BA when odd."""
    if type(pairs) is not int or pairs < 1:
        raise ValueError("pairs must be a positive integer")
    order = []
    for pair in range(pairs):
        arms = ARMS if pair % 2 == 0 else ARMS[::-1]
        order.extend((pair, arm) for arm in arms)
    return order


def median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def block_rate(block, basis):
    """Tokens/s or verify rounds/s of one timed block (``None`` if unmeasurable)."""
    seconds = block.get("seconds")
    count = block.get("tokens") if basis == "tokens" else block.get("rounds")
    if not seconds or seconds <= 0 or not count:
        return None
    return count / seconds


def decode_verdict(blocks, *, tokens_identical, speculative, threshold=0.05,
                   min_pairs=3, consistency=0.75):
    """Paired restored/fresh comparison of timed blocks.

    ``basis`` is the token rate, except on a speculative route whose arms
    emitted different tokens: acceptance then differs, so the verify-round
    rate is compared instead (ordinary decode costs one step per token
    whatever the token).  Short blocks (the lane ended early) are left out.
    A pair's ratio is restored rate / fresh rate.  ``restored_slower`` needs
    the median ratio at most ``1 - threshold`` and at least ``consistency`` of
    the pairs below 1 (and symmetrically ``restored_faster``); fewer than
    ``min_pairs`` complete pairs is ``inconclusive``; anything else is
    ``no_material_difference``.
    """
    basis = "rounds" if speculative and not tokens_identical else "tokens"
    short = sum(bool(b.get("short")) for b in blocks)
    blocks = [b for b in blocks if not b.get("short")]
    pairs = {}
    for block in blocks:
        pairs.setdefault(block["pair"], {})[block["arm"]] = block_rate(block, basis)
    ratios = [p["restored"] / p["fresh"] for _, p in sorted(pairs.items())
              if p.get("fresh") and p.get("restored")]
    arms = {arm: {"median_rate": median(block_rate(b, basis) for b in blocks if b["arm"] == arm),
                  "median_tok_s": median(block_rate(b, "tokens") for b in blocks if b["arm"] == arm),
                  "blocks": sum(b["arm"] == arm for b in blocks)}
            for arm in ARMS}
    fresh, restored = arms["fresh"]["median_rate"], arms["restored"]["median_rate"]
    result = {"basis": basis, "rate_unit": "tok/s" if basis == "tokens" else "verify rounds/s",
              "short_blocks_excluded": short,
              "threshold": threshold, "consistency": consistency,
              "arms": arms, "pair_ratios": ratios, "pairs": len(ratios),
              "median_pair_ratio": median(ratios),
              "ratio_of_medians": restored / fresh if fresh and restored else None,
              "pairs_restored_slower": sum(r < 1 for r in ratios)}
    if len(ratios) < min_pairs:
        result["verdict"] = "inconclusive"
        return result
    m = result["median_pair_ratio"]
    slower = result["pairs_restored_slower"] / len(ratios)
    if m <= 1 - threshold and slower >= consistency:
        result["verdict"] = "restored_slower"
    elif m >= 1 + threshold and (1 - slower) >= consistency:
        result["verdict"] = "restored_faster"
    else:
        result["verdict"] = "no_material_difference"
    return result


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# -------------------------------------------------------- layout facts --

class _PyBuffer(ctypes.Structure):
    _fields_ = [("buf", ctypes.c_void_p), ("obj", ctypes.c_void_p),
                ("len", ctypes.c_ssize_t), ("itemsize", ctypes.c_ssize_t),
                ("readonly", ctypes.c_int), ("ndim", ctypes.c_int),
                ("format", ctypes.c_char_p), ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
                ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
                ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)), ("internal", ctypes.c_void_p)]


_PyBUF_RECORDS_RO = 0x0004 | 0x0008 | 0x0010  # FORMAT | ND | STRIDES


def buffer_facts(array):
    """Address range, strides and contiguity of one MLX array (evaluates it).

    Read through the buffer protocol, so a strided view reports its view
    strides and the address of its first element; ``span`` is the byte range
    the view can touch.  Off the timed path only.
    """
    view = memoryview(array)
    buf = _PyBuffer()
    get = ctypes.pythonapi.PyObject_GetBuffer
    get.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
    if get(array, ctypes.byref(buf), _PyBUF_RECORDS_RO) != 0:
        raise RuntimeError("buffer protocol refused the array")
    try:
        start = int(buf.buf or 0)
    finally:
        release = ctypes.pythonapi.PyBuffer_Release
        release.argtypes = [ctypes.POINTER(_PyBuffer)]
        release(ctypes.byref(buf))
    shape, strides = tuple(view.shape), tuple(view.strides)
    span = view.itemsize + sum((n - 1) * abs(s) for n, s in zip(shape, strides) if n > 0) if all(shape) else 0
    return {"shape": list(shape), "dtype": str(array.dtype), "strides": list(strides),
            "c_contiguous": bool(view.c_contiguous), "nbytes": int(view.nbytes),
            "start": start, "span": int(span)}


def overlaps(facts, ranges):
    lo, hi = facts["start"], facts["start"] + facts["span"]
    return any(lo < r_hi and r_lo < hi for r_lo, r_hi in ranges)


def _array_leaves(value, path, out, depth=0, seen=None):
    import mlx.core as mx

    seen = set() if seen is None else seen
    if depth > 6 or id(value) in seen:
        return
    if isinstance(value, mx.array):
        out.append((path, value))
        return
    seen.add(id(value))
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _array_leaves(item, f"{path}[{i}]", out, depth + 1, seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            _array_leaves(item, f"{path}.{key}", out, depth + 1, seen)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        for key, item in vars(value).items():
            _array_leaves(item, f"{path}.{key}", out, depth + 1, seen)


def array_leaves(tree, root="cache"):
    out = []
    _array_leaves(tree, root, out)
    return out


def layout_facts(tree, store_ranges=(), root="cache"):
    """Per-array layout of a cache tree, and which arrays alias the store."""
    rows = []
    for path, array in array_leaves(tree, root):
        try:
            facts = buffer_facts(array)
        except (RuntimeError, TypeError, ValueError) as error:
            rows.append({"path": path, "error": repr(error)})
            continue
        facts["path"] = path
        facts["aliases_store"] = overlaps(facts, store_ranges) if store_ranges else False
        rows.append(facts)
    return rows


def layout_summary(rows):
    good = [r for r in rows if "error" not in r]
    return {"arrays": len(rows), "unreadable": len(rows) - len(good),
            "non_contiguous": sum(not r["c_contiguous"] for r in good),
            "aliases_store": sum(r["aliases_store"] for r in good),
            "bytes": sum(r["nbytes"] for r in good),
            "aliased_bytes": sum(r["nbytes"] for r in good if r["aliases_store"])}


def cache_objects(tree):
    """Class / offset / array-field shapes of each layer cache (no evaluation)."""
    import mlx.core as mx

    rows = []
    for index, item in enumerate(tree if isinstance(tree, (list, tuple)) else [tree]):
        fields = {}
        for key, value in (vars(item).items() if hasattr(item, "__dict__") else ()):
            if isinstance(value, mx.array):
                fields[key] = list(value.shape)
            elif isinstance(value, (list, tuple)) and value and all(
                    isinstance(v, mx.array) or v is None for v in value):
                fields[key] = [None if v is None else list(v.shape) for v in value]
        rows.append({"index": index, "class": type(item).__name__,
                     "offset": getattr(item, "offset", None), "fields": fields})
    return rows


class OpCounter:
    """Count Python-level ``mx.contiguous`` / ``mx.concatenate`` calls."""

    NAMES = ("contiguous", "concatenate")

    def __init__(self):
        self.counts = dict.fromkeys(self.NAMES, 0)
        self._saved = {}

    def __enter__(self):
        import mlx.core as mx

        for name in self.NAMES:
            original = getattr(mx, name)
            self._saved[name] = original

            def wrapped(*args, _name=name, _original=original, **kwargs):
                self.counts[_name] += 1
                return _original(*args, **kwargs)

            setattr(mx, name, wrapped)
        return self

    def __exit__(self, *exc):
        import mlx.core as mx

        for name, original in self._saved.items():
            setattr(mx, name, original)
        return False


# --------------------------------------------------------------- engine --

class Context:
    """One model load and the generator factory for both routes."""

    def __init__(self, args):
        sys.path.insert(0, str(ROOT / "scripts"))
        sys.path.insert(0, str(ROOT / "src"))
        import mlx.core as mx

        self.mx, self.args = mx, args
        self.prefill_step = args.prefill_step
        if args.tiny:
            mx.set_default_device(mx.cpu)
        # As serving's generation thread does before it builds the adapter:
        # one stream for restores, merges and the generators' forwards, so no
        # cross-stream fence forms, and ``sync`` drains what the generators
        # dispatched (a bare ``mx.synchronize()`` drains only the default).
        from mlx2.runtime.generation_stream import (
            bind_generation_thread_stream,
            generation_stream,
        )

        self.streams = (bind_generation_thread_stream(), generation_stream)
        if args.tiny:
            from paired_direct_ab import tiny_external, tiny_ordinary

            self.ordinary_model = tiny_ordinary()
            self.external_target, self.external_draft = tiny_external()
            self.layout = "tiny"
            self.stops = ()
            self.identity = {"checkpoint": "tiny-random@none", "policy": None}
            return
        from mlx2.adapters.registry import resolve_adapter
        from mlx2.serving import generation_stop_token_ids

        cls = resolve_adapter(args.model, mtp=False, qualification_mode=True)
        policy = json.loads(Path(args.policy).read_text()) if args.policy else None
        self.adapter = cls(args.model, execution_policy=policy) if policy else cls(args.model)
        if "dspark" in args.routes and getattr(self.adapter, "draft_model", None) is None:
            raise SystemExit("refused: --policy bound no external drafter for the dspark route")
        self.ordinary_model = self.external_target = self.adapter.model
        self.layout = self.adapter.layout
        self.stops = generation_stop_token_ids(self.adapter) if args.stop_at_eos else ()
        if args.prefill_step is None:
            self.prefill_step = int(self.adapter.prefill_step_default())
        from mlx2.adapters.registry import inspect_model

        fingerprint = inspect_model(args.model).artifact_fingerprint
        self.identity = {"checkpoint": f"{Path(args.model).name}@sha256:{fingerprint}",
                         "adapter": f"{cls.__module__}.{cls.__qualname__}",
                         "fingerprint": self.adapter.identity.get("fingerprint"),
                         "policy": policy}

    def sync(self):
        for stream in self.streams:
            self.mx.synchronize(stream)
        self.mx.synchronize()

    def generator(self, route):
        stops = [[t] for t in self.stops]
        step = self.prefill_step or 512
        if route == "ordinary":
            from mlx2.runtime.generate import BatchGenerator

            return BatchGenerator(self.ordinary_model, completion_batch_size=1, prefill_batch_size=1,
                                  prefill_step_size=step, stop_tokens=stops)
        if self.args.tiny:
            from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

            return ExternalDraftBatchGenerator(self.external_target, draft_model=self.external_draft,
                                               binding="tiny", completion_batch_size=1,
                                               prefill_step_size=step, num_draft=2, stop_tokens=stops)
        return self.adapter.create_external_batch(completion_batch_size=1, prefill_step_size=step,
                                                  stop_tokens=stops)

    def prompts(self, indices):
        if self.args.tiny:
            # Shared 40-token "few-shot" head, then a per-question tail.
            head = [1 + (i * 7) % 100 for i in range(40)]
            return {i: head + [100 + (i * 3 + k) % 27 for k in range(9 + i % 4)] for i in indices}
        sys.path.insert(0, str(ROOT / "scripts"))
        from spec_acceptance_gsm8k import (
            CHAT_INSTRUCTION,
            dataset_files,
            few_shot_turns,
            load_split,
        )

        _revision, files = dataset_files()
        train, test = load_split(files["train"]), load_split(files["test"])
        shots = few_shot_turns(train, self.args.shots)
        out = {}
        for i in indices:
            request = {"messages": shots + [{"role": "user", "content": test[i]["question"] + CHAT_INSTRUCTION}],
                       **(self.args.request or {})}
            out[i] = [int(t) for t in self.adapter.prompt_tokens(request)]
        return out


def insert(gen, route, prompt, max_tokens, *, seed, cache=None, all_tokens=None, sidecar=None,
           boundaries=None, attestation=None):
    from mlx2.runtime.sample_utils import LaneRNG

    kwargs = {"max_tokens": [int(max_tokens)], "lane_rngs": [LaneRNG(seed)]}
    if cache is not None:
        kwargs.update(caches=[cache], all_tokens=[list(all_tokens)])
    if boundaries:
        kwargs["state_boundaries"] = [tuple(boundaries)]
    if route == "dspark":
        # Serving's greedy external-draft lane config (serving.py warm/cold
        # insert): no logprob rows unless a request asks for them.
        kwargs["sampling_configs"] = [{"sampling_temp": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
                                       "shared_prefix_attestation": attestation,
                                       "emit_logprobs": False}]
        if cache is not None:
            kwargs["cache_states"] = [sidecar]
    (uid,) = gen.insert([list(prompt)], **kwargs)
    return uid


def _responses(result):
    return result[1] if isinstance(result, tuple) else result


class Lane:
    """One generator holding one request; ``drive`` advances only it."""

    IDLE_LIMIT = 10000

    def __init__(self, gen, uid, route):
        self.gen, self.uid, self.route = gen, uid, route
        self.tokens, self.finished, self.checkpoints = [], False, []

    def stats(self):
        return dict(getattr(self.gen, "scheduler_stats", {}) or {})

    def drive(self, count):
        """Poll until ``count`` more tokens (or the end); returns tokens added."""
        start, idle = len(self.tokens), 0
        while not self.finished and len(self.tokens) - start < count:
            responses = _responses(self.gen.next())
            drain = getattr(self.gen, "drain_state_checkpoints", None)
            if drain is not None:
                self.checkpoints.extend(drain())
            take = getattr(self.gen, "take_lane_failures", None)
            failures = take() if take is not None else None
            if failures:
                raise RuntimeError(f"lane failed: {failures}")
            idle = 0 if responses else idle + 1
            if idle > self.IDLE_LIMIT:
                raise RuntimeError("generator stopped responding")
            for response in responses:
                if response.uid != self.uid:
                    continue
                self.tokens.append(int(response.token))
                if getattr(response, "finish_reason", None):
                    self.finished = True
        return len(self.tokens) - start


def plant_junction(ctx, route, prompt, junction, apc, key):
    """Capture a JUNCTION snapshot through the generator and store it in APCv2."""
    from mlx2.runtime.state_boundaries import (
        RETENTION_ROLE,
        BoundaryPurpose,
        StateBoundary,
    )
    from mlx2.serving import detach_checkpoint_graph

    gen = ctx.generator(route)
    try:
        uid = insert(gen, route, prompt, 1, seed=ctx.args.seed,
                     boundaries=[StateBoundary(junction, BoundaryPurpose.JUNCTION)])
        lane = Lane(gen, uid, route)
        lane.drive(1)
        found = [c for owner, c in lane.checkpoints if owner == uid and len(c["tokens"]) == junction]
        if not found:
            raise RuntimeError(f"{route}: no junction snapshot captured at {junction} "
                               f"(captured {[len(c['tokens']) for _, c in lane.checkpoints]})")
        checkpoint = found[0]
        sidecar = checkpoint.get("cache_sidecar")
        if sidecar is None and checkpoint.get("mtp_state"):
            from mlx2.runtime.apc_v2 import MTPAPCSidecar

            sidecar = MTPAPCSidecar(checkpoint["mtp_state"], checkpoint["covered_tokens"])
        detach_checkpoint_graph(checkpoint["target_cache"], sidecar)
        capabilities = apc.store(key, checkpoint["tokens"], checkpoint["target_cache"], sidecar=sidecar,
                                 retention_role=RETENTION_ROLE[BoundaryPurpose(checkpoint["purpose"])])
        if getattr(capabilities, "stored", None) is False:
            raise RuntimeError(f"{route}: APCv2 refused the junction snapshot: {capabilities}")
        return {"junction": junction, "purpose": BoundaryPurpose(checkpoint["purpose"]).name,
                "has_sidecar": sidecar is not None, "stats": lane.stats()}
    finally:
        gen.close()


def store_ranges(apc, key, tokens):
    """Byte ranges of the stored entry's arrays (target cache and sidecar)."""
    entry = apc._trie.get(key, list(tokens))
    ranges = []
    for _path, array in array_leaves([entry.prompt_cache, entry.sidecar], "entry"):
        try:
            facts = buffer_facts(array)
        except (RuntimeError, TypeError, ValueError):
            continue
        ranges.append((facts["start"], facts["start"] + facts["span"]))
    return ranges


def lane_tree(lane):
    """The lane's live target cache and draft state, wherever the route keeps them."""
    gen = lane.gen
    lanes = getattr(gen, "lanes", None)
    if isinstance(lanes, dict) and lane.uid in lanes:
        live = lanes[lane.uid]
        return {"target": live.cache, "draft": getattr(live, "draft_cache", None)}
    # BatchGenerator merges a lane's cache into its generation batch's
    # batched planes (a restored cache is copied in there, not used in place).
    batch = getattr(gen, "_generation_batch", None)
    return {"target": getattr(batch, "prompt_cache", None), "draft": None}


def restore_engagement(route, prefill_stats, layout_rows):
    """Why the restored arm did not decode from the restored cache, or ``None``.

    External insert silently re-prefills from scratch without a usable
    sidecar, so the restore must be visible in the generators' own counters:
    one ``paired_cache_resumes`` on the restored dspark lane (none on fresh),
    fewer prefill rounds than fresh on the ordinary route, and a readable
    live cache on both arms.
    """
    fresh, restored = prefill_stats["fresh"], prefill_stats["restored"]
    if route == "dspark" and (restored.get("paired_cache_resumes", 0) != 1
                              or fresh.get("paired_cache_resumes", 0) != 0):
        return "dspark restored lane did not resume from the APCv2 cache (paired_cache_resumes)"
    if route == "ordinary" and not (restored.get("prefill_rounds", 0) < fresh.get("prefill_rounds", 0)):
        return "ordinary restored lane prefilled as many rounds as the fresh lane"
    for arm in ARMS:
        if not layout_rows[arm]:
            return f"{arm} lane cache could not be located for layout facts"
    return None


def run_question(ctx, route, index, prompt, junction, apc, key):
    mx, args = ctx.mx, ctx.args
    budget = args.block_tokens * (args.pairs + 2) + args.block_tokens + 8
    gens, lanes = [], {}
    record = {"route": route, "test_index": index, "prompt_tokens": len(prompt), "junction": junction}
    try:
        gens.append(ctx.generator(route))
        lanes["fresh"] = Lane(gens[-1], insert(gens[-1], route, prompt, budget, seed=args.seed), route)
        hit = apc.lookup(key, prompt, require_sidecar=route == "dspark",
                         allow_target_only_plain=route != "dspark")
        if not hit.hit or hit.cached_tokens != junction:
            raise RuntimeError(f"{route}: junction lookup did not hit at {junction}: "
                               f"hit={hit.hit} cached={hit.cached_tokens} reason={hit.miss_reason}")
        from mlx2.runtime.models.cache import materialize_restored_state
        from mlx2.serving import shared_prefix_attestation

        materialize_restored_state(hit.cache, hit.sidecar)
        ranges = store_ranges(apc, key, prompt[:junction])
        # Positive control for the alias detector: the restored branch itself
        # shares the stored buffers before any lane writes to it.
        restore_rows = layout_facts({"target": hit.cache, "draft": hit.sidecar}, ranges, root="hit")
        record["layout_at_restore"] = {"store_arrays": len(ranges),
                                       "summary": layout_summary(restore_rows),
                                       "caches": cache_objects(hit.cache)}
        gens.append(ctx.generator(route))
        lanes["restored"] = Lane(gens[-1], insert(
            gens[-1], route, hit.remaining_tokens, budget, seed=args.seed, cache=hit.cache,
            all_tokens=prompt[:hit.cached_tokens], sidecar=hit.sidecar,
            attestation=shared_prefix_attestation(hit)), route)
        record["hit"] = {"cached_tokens": hit.cached_tokens, "remaining": len(hit.remaining_tokens)}
        for arm in ARMS:                       # prefill to the first token, untimed
            lanes[arm].drive(1)
        ctx.sync()
        prefill_stats = {arm: lanes[arm].stats() for arm in ARMS}
        record["prefill_stats"] = {arm: {k: prefill_stats[arm].get(k, 0) for k in
                                         ("prefill_rounds", "paired_cache_resumes")} for arm in ARMS}
        record["layout_after_prefill"] = {}
        rows_by_arm = {}
        for arm in ARMS:
            tree = lane_tree(lanes[arm])
            rows_by_arm[arm] = layout_facts(tree, ranges, root=arm)
            record["layout_after_prefill"][arm] = {"summary": layout_summary(rows_by_arm[arm]),
                                                   "caches": cache_objects(tree["target"]),
                                                   "arrays": rows_by_arm[arm]}
        refusal = restore_engagement(route, prefill_stats, rows_by_arm)
        if refusal:
            raise RuntimeError(f"{route}: restore not engaged: {refusal}")
        for arm in ARMS:                       # warm-up block, untimed
            lanes[arm].drive(args.block_tokens)
        blocks = []
        for pair, arm in interleave_schedule(args.pairs):
            lane = lanes[arm]
            ctx.sync()                         # the previous arm's dispatched work is its own
            before = lane.stats()
            mx.reset_peak_memory()
            started = time.perf_counter()
            added = lane.drive(args.block_tokens)
            ctx.sync()
            seconds = time.perf_counter() - started
            after = lane.stats()
            blocks.append({"pair": pair, "arm": arm, "tokens": added, "seconds": seconds,
                           "tok_s": added / seconds if seconds > 0 else None,
                           "short": added < args.block_tokens,
                           "rounds": after.get("external_rounds", 0) - before.get("external_rounds", 0),
                           "accepted": after.get("accepted_proposals", 0) - before.get("accepted_proposals", 0),
                           "peak_bytes": mx.get_peak_memory(), "finished": lane.finished})
        record["blocks"] = blocks
        probes = {}
        for arm in ARMS:                       # op-count probe, untimed
            with OpCounter() as counter:
                added = lanes[arm].drive(args.block_tokens)
            probes[arm] = {"tokens": added, **counter.counts}
        record["op_probe"] = probes
        record["layout_after_decode"] = {}
        for arm in ARMS:
            tree = lane_tree(lanes[arm])
            rows = layout_facts(tree, ranges, root=arm)
            record["layout_after_decode"][arm] = {"summary": layout_summary(rows),
                                                  "caches": cache_objects(tree["target"])}
        tokens = {arm: lanes[arm].tokens for arm in ARMS}
        shared = min(len(tokens["fresh"]), len(tokens["restored"]))
        record["tokens_identical"] = tokens["fresh"][:shared] == tokens["restored"][:shared]
        record["first_token_divergence"] = (None if record["tokens_identical"] else
                                            common_prefix(tokens["fresh"], tokens["restored"]))
        record["tokens"] = {arm: len(t) for arm, t in tokens.items()}
        record["verdict"] = decode_verdict(
            blocks, tokens_identical=record["tokens_identical"], speculative=route == "dspark",
            threshold=args.threshold)
        record["stats"] = {arm: {k: v for k, v in lanes[arm].stats().items()
                                 if isinstance(v, (int, float)) and v} for arm in ARMS}
    finally:
        for gen in gens:
            gen.close()
    return record


def overall_verdict(records, threshold):
    """Pool every question's blocks per route (pairs renumbered per question)."""
    out = {}
    for route in ROUTES:
        rows = [r for r in records if r.get("route") == route and r.get("blocks")]
        if not rows:
            continue
        blocks, offset = [], 0
        for r in rows:
            blocks.extend({**b, "pair": b["pair"] + offset} for b in r["blocks"])
            offset += 1 + max(b["pair"] for b in r["blocks"])
        out[route] = decode_verdict(blocks, tokens_identical=all(r["tokens_identical"] for r in rows),
                                    speculative=route == "dspark", threshold=threshold)
    return out


def write_atomic(path, document):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(document, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def revision(tree):
    return {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=tree).strip(),
            "dirty_tracked": bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], text=True, cwd=tree).strip())}


def parse_indices(text):
    return [int(i) for i in text.split(",") if i.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--policy", type=Path, help="execution policy JSON for the dspark route")
    parser.add_argument("--cpg-lock", type=Path, help="qualification lease (required unless --tiny)")
    parser.add_argument("--routes", default="ordinary,dspark")
    parser.add_argument("--plant-indices", default="0,1",
                        help="two GSM8K test indices; their common prefix is the junction")
    parser.add_argument("--measure-indices", default="2,3")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--request", type=json.loads, default={"reasoning_effort": "low"},
                        help="request fields the adapter renders (default: the A/B's reasoning_effort low)")
    parser.add_argument("--prefill-step", type=int, default=None, help="default: the adapter's")
    parser.add_argument("--block-tokens", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-at-eos", action="store_true",
                        help="keep the adapter's EOS stops (default: fixed-length decode)")
    parser.add_argument("--tiny", action="store_true", help="CPU plumbing check with tiny random models")
    args = parser.parse_args(argv)
    args.routes = [r for r in args.routes.split(",") if r]
    if any(r not in ROUTES for r in args.routes):
        parser.error(f"--routes must be drawn from {ROUTES}")
    plant, measure = parse_indices(args.plant_indices), parse_indices(args.measure_indices)
    if len(plant) != 2 or not measure:
        parser.error("--plant-indices needs two indices and --measure-indices at least one")
    if not args.tiny and not (args.model and args.cpg_lock):
        parser.error("--model and --cpg-lock are required unless --tiny")
    if "dspark" in args.routes and not args.tiny and not args.policy:
        parser.error("the dspark route needs --policy")
    if args.block_tokens < 1 or args.pairs < 1:
        parser.error("--block-tokens and --pairs must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("refusing nonempty evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": SCHEMA, "qualified": False,
               "kind": "direct-model interleaved decode A/B, no thermal hold (attribution, not publishable)",
               "source_revision": revision(ROOT),
               "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "questions": [], "errors": []}
    path = args.output / "receipt.json"
    if not args.tiny:
        sys.path.insert(0, str(ROOT / "scripts"))
        from qualification_gpu_ownership import require_qualification_lease

        owner = json.loads((args.cpg_lock / "owner.json").read_text())
        receipt["ownership"] = require_qualification_lease(
            task_id=owner["cpg_task"], generation=owner["cpg_generation"], cpg_owner_lock=args.cpg_lock)
        receipt["swap_before"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
    ctx = Context(args)
    receipt["identity"] = ctx.identity
    receipt["prefill_step"] = ctx.prefill_step
    from mlx2.runtime.apc_v2 import APCKey, APCv2

    prompts = ctx.prompts(sorted(set(plant + measure)))
    junction = common_prefix(prompts[plant[0]], prompts[plant[1]])
    receipt["junction"] = {"position": junction, "plant_indices": plant}
    for i in measure:
        if common_prefix(prompts[i], prompts[plant[1]]) < junction or len(prompts[i]) <= junction + 1:
            raise SystemExit(f"refused: question {i} does not extend the junction prefix")
    try:
        for route in args.routes:
            apc = APCv2(max_size=16, layout_name=f"{ctx.layout}-junction-ab-{route}")
            key = APCKey(f"junction-restore-ab-{route}")
            try:
                receipt.setdefault("plant", {})[route] = plant_junction(
                    ctx, route, prompts[plant[1]], junction, apc, key)
                for i in measure:
                    try:
                        record = run_question(ctx, route, i, prompts[i], junction, apc, key)
                    except Exception as error:  # noqa: BLE001 - record and go on
                        record = {"route": route, "test_index": i, "error": repr(error)}
                        receipt["errors"].append(record)
                    receipt["questions"].append(record)
                    write_atomic(path, receipt)
                    print(json.dumps({"route": route, "test_index": i,
                                      "verdict": (record.get("verdict") or {}).get("verdict"),
                                      "error": record.get("error")}), flush=True)
                receipt.setdefault("apc_stats", {})[route] = apc.lifetime_stats()
            finally:
                apc.clear(release_memory=True)
    finally:
        receipt["overall"] = overall_verdict(receipt["questions"], args.threshold)
        if not args.tiny:
            receipt["swap_after"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
        write_atomic(path, receipt)
    print(json.dumps({"overall": {r: v["verdict"] for r, v in receipt["overall"].items()}}), flush=True)
    return 1 if receipt["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
