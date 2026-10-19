"""Bounded B1 megakernel research, exact ONLY to sequential megakernel.

No serving/cache publication and no historical qualification inheritance.
Run under the CPG lease and paired-lock wrappers. Import/help are CPU-only.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
import signal
import statistics
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DONOR_REVISION = "f87a14e109c385a36ea83321a1db94e0c1efc9fc"
DONOR_DEFAULT = Path("~/Desktop/mlx-uag/mlx-lm-unified-worktrees/ane-shadow-mtp-20260906")
MODEL_DEFAULT = Path("~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP")
TARGET_CONTRACT = "flash-next-phase5exp-serial-e2-approximate-research-v1"
ENVIRONMENT = {
    "MLX_QWEN4_MEGAKERNEL": "1",
    "MLX_QWEN4_MEGAKERNEL_DUAL_WIDTH": "1",
    "MLX_QWEN4_MEGAKERNEL_MAX_WIDTH": "4",
    "MLX_QWEN4_MEGAKERNEL_REQUIRE_PRIMITIVES": "1",
    "MLX_QWEN4_MEGAKERNEL_TUNE": "off",
    "MLX_MK_EXP_FUSION": "phase5exp",
    "MLX_MK_EXP_WIDE_FUSION": "phase5exp",
    "MLX_MK_EXP_WIDE_MOE_E2_ORDER": "serial",
}
PROMPT = "Explain how a hash table handles collisions, with a short Python example. "


def donor_manifest(path, command):
    revision = command("git", "rev-parse", "HEAD", cwd=path)
    dirty = command("git", "status", "--porcelain", "--untracked-files=no", cwd=path)
    if revision != DONOR_REVISION or dirty:
        raise RuntimeError("donor must be the clean, pinned research revision")
    provenance = json.loads((ROOT / "provenance/flash-megakernel-ast-research.json").read_text())
    return {"revision": revision, "path": str(path), "files": {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in [*provenance["source_paths"], "LICENSE"]}}


def target_binding(revision, checkpoint, contract, donor):
    """Bind private target state to both mlx2 and donor source identities."""
    components = {
        "mlx2_revision": str(revision),
        "checkpoint": str(checkpoint),
        "target_contract": str(contract),
        "donor_revision": str(donor["revision"]),
        "donor_files": dict(sorted(donor["files"].items())),
    }
    encoded = json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
    return {"algorithm": "sha256", "digest": hashlib.sha256(encoded).hexdigest(),
            **components}


def configure(out):
    # Refuse inherited experiments that could silently change arithmetic or
    # residency. The explicitly selected set is recorded in every receipt.
    extras = {k: v for k, v in os.environ.items()
              if k.startswith(("MLX_MK_", "MLX_QWEN4_MEGAKERNEL"))
              and k not in ENVIRONMENT}
    if extras:
        raise RuntimeError(f"unscoped megakernel overrides: {sorted(extras)}")
    for key, value in ENVIRONMENT.items():
        if key in os.environ and os.environ[key] != value:
            raise RuntimeError(f"conflicting research setting: {key}")
        os.environ[key] = value
    os.environ["MLX_QWEN4_MEGAKERNEL_TUNE_CACHE"] = str(out / "device-primitives.json")


def import_kernel(path):
    alias = "_mlx2_research_megakernel"
    if alias in sys.modules:
        raise RuntimeError("kernel package already imported")
    package = types.ModuleType(alias)
    package.__path__ = [str(path / "mlx_lm/models")]
    sys.modules[alias] = package
    tune = importlib.import_module(alias + ".qwen4_megakernel_tune")
    entry, receipt = tune.ensure_tuned()
    if not entry or not entry.get("primitives", {}).get("ok"):
        raise RuntimeError(f"megakernel primitive gate refused: {receipt}")
    config = importlib.import_module(alias + ".qwen4_megakernel_config")
    config.invalidate()
    runtime = importlib.import_module(alias + ".qwen4_megakernel_runtime")
    return runtime, {"entry": entry, "receipt": receipt,
                     "config": config.config_receipt(autotune=False)}


def split_source_class(base, mx, audit=None):
    """Virtual fused rows; preserve progressive group packing and release.

    The donor shard source already concatenates gate then up on axis -2.
    This original live-module bridge supplies that layout lazily without
    changing the current adapter's loaded expert execution policy.
    """
    class FusedView:
        def __init__(self, gate, up):
            self.gate, self.up = gate, up
            self.rows = gate.weight.shape[-2]
            self.bits, self.group_size = gate.bits, gate.group_size
            if (up.bits, up.group_size) != (self.bits, self.group_size):
                raise ValueError("split expert quantization mismatch")
            self.bound = {}

        def get(self, part):
            if part in self.bound:
                return self.bound[part]
            a, b = getattr(self.gate, part), getattr(self.up, part)
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError("split expert shape/dtype mismatch")
            return mx.contiguous(mx.concatenate([a, b], axis=-2))

        def put(self, part, value):
            self.bound[part] = value
            setattr(self.gate, part, value[..., :self.rows, :])
            setattr(self.up, part, value[..., self.rows:, :])

        weight = property(lambda self: self.get("weight"), lambda self, value: self.put("weight", value))
        scales = property(lambda self: self.get("scales"), lambda self, value: self.put("scales", value))
        biases = property(lambda self: self.get("biases"), lambda self, value: self.put("biases", value))

    class SplitSource(base):
        def __init__(self, root):
            super().__init__(root)
            self.fused_views = {}
            self.invalidated_expert_views = 0

        def _resolve(self, key):
            try:
                return super()._resolve(key)
            except AttributeError:
                if not key.endswith(".gate_up_proj"):
                    raise
                if key not in self.fused_views:
                    stem = key.removesuffix("gate_up_proj")
                    self.fused_views[key] = FusedView(
                        super()._resolve(stem + "gate_proj"),
                        super()._resolve(stem + "up_proj"))
                return self.fused_views[key]

        def rebind(self, key, parts):
            inspect = audit is not None and ".switch_mlp." in key and len(audit) < 4
            module = self._resolve(key)
            owners = (("gate", module.gate), ("up", module.up)) if isinstance(module, FusedView) else (("direct", module),)
            old = {}
            if inspect:
                old = {name: owner.weight for name, owner in owners}
            super().rebind(key, parts)
            # The ordinary routed path retains (original arrays, one-expert
            # views) until its next forward. There is no next ordinary expert
            # forward after this transition, so invalidate those derived views
            # now, at the same weight-replacement boundary they depend on.
            for _, owner in owners:
                if getattr(owner, "__dict__", {}).pop("_mlx2_expert_views", None) is not None:
                    self.invalidated_expert_views += 1
            if inspect:
                for name, value in old.items():
                    refs = []
                    for ref in gc.get_referrers(value):
                        detail = {"type": type(ref).__name__}
                        if isinstance(ref, dict):
                            detail["keys"] = [str(k) for k, v in ref.items() if v is value][:8]
                        elif isinstance(ref, (tuple, list)):
                            detail["length"] = len(ref)
                            detail["parents"] = [{"type": type(parent).__name__,
                                "keys": [str(k) for k, v in parent.items() if v is ref][:8]}
                                for parent in gc.get_referrers(ref) if isinstance(parent, dict)][:4]
                        refs.append(detail)
                    audit.append({"key": key, "part": name, "refcount": sys.getrefcount(value), "referrers": refs[:12]})

    return SplitSource


def raw_array(value):
    import mlx.core as mx
    import numpy as np
    mx.eval(value)
    if not bool(mx.all(mx.isfinite(value)).item()):
        raise RuntimeError("non-finite research tensor")
    return {"shape": tuple(value.shape), "dtype": value.dtype,
            "raw": np.array(value.view(mx.uint8), copy=True)}


def restore_array(value):
    import mlx.core as mx
    return mx.array(value["raw"]).view(value["dtype"]).reshape(value["shape"])


def digest(value):
    row = raw_array(value)
    return {"shape": row["shape"], "dtype": str(row["dtype"]),
            "sha256": hashlib.sha256(row["raw"].tobytes()).hexdigest()}


def launch_kernel(decoder, embeddings, ple):
    if len(embeddings) == 1:
        return decoder.step(embeddings[0], ple_embedding=ple[0])
    return decoder.step_slab(embeddings, ple_embeddings=ple)


class ResearchTarget:
    """Private acknowledged transactions; deliberately no cache-export API."""

    def __init__(self, decoder, model):
        self.decoder, self.model = decoder, model
        self.layers = model.language_model.model.layers
        self.ple_index, = decoder.ple_layers
        self.history = None
        self.pending_history = None
        self.launches = []

    def seed(self, caches):
        self.decoder.seed_from_caches(
            caches, indexer_of=lambda i: self.layers[i].self_attn.indexer)
        self.history = copy.deepcopy(caches[self.ple_index])
        self.pending_history = None

    def launch(self, tokens):
        import mlx.core as mx
        if self.pending_history is not None:
            raise RuntimeError("research transaction already pending")
        if not 1 <= len(tokens) <= 4:
            raise ValueError("research supports B1 consecutive widths 1..4 only")
        self.pending_history = copy.deepcopy(self.history)
        start = time.perf_counter()
        ids = mx.array([tokens], mx.uint32)
        embeddings = self.model.language_model.model.embed_tokens(ids)[0].astype(mx.bfloat16)
        ple = self.layers[self.ple_index].ple.ple_embedding(ids, self.pending_history)[0].astype(mx.bfloat16)
        logits = launch_kernel(self.decoder, embeddings, ple)
        hidden = self.decoder.pending_mtp_hidden()
        selected = mx.argmax(logits, axis=-1).reshape(-1).tolist()
        elapsed = time.perf_counter() - start
        self.launches.append({"width": len(tokens), "position": self.decoder.position,
                              "seconds": elapsed, "acknowledged": True,
                              "variant": getattr(self.decoder.body, "last_variant", "single")})
        return logits.reshape(len(tokens), -1), hidden, selected, elapsed

    def commit(self):
        if self.pending_history is None:
            raise RuntimeError("no research transaction pending")
        self.decoder.commit()
        self.history, self.pending_history = self.pending_history, None

    def rollback(self):
        if self.pending_history is None:
            raise RuntimeError("no research transaction pending")
        self.decoder.rollback()
        self.pending_history = None

    def snapshot(self):
        if self.pending_history is not None:
            raise RuntimeError("cannot snapshot pending research state")
        d = self.decoder
        return {"position": d.position, "history": copy.deepcopy(self.history),
                "arrays": {name: raw_array(getattr(d, name))
                           for name in ("kv", "idxl", "cs", "rec", "pconv")}}

    def restore(self, snapshot):
        import mlx.core as mx
        if self.pending_history is not None:
            raise RuntimeError("rollback before restore")
        for name, value in snapshot["arrays"].items():
            setattr(self.decoder, name, restore_array(value))
        mx.eval(self.decoder.kv, self.decoder.idxl, self.decoder.cs,
                self.decoder.rec, self.decoder.pconv)
        self.decoder.position = snapshot["position"]
        self.history = copy.deepcopy(snapshot["history"])

    def state(self):
        d = self.decoder
        position = d.position
        result = {"position": position, "conv": digest(d.cs), "recurrent": digest(d.rec),
                  "ple_conv": digest(d.pconv), "ple_history": digest(self.history[3]),
                  "kv": digest(d.kv[:, :, :position])}
        for slot, _ in enumerate(d.attn_layers):
            result[f"raw_index_{slot}"] = digest(d.rawk[slot * d.total:slot * d.total + position])
            result[f"pooled_index_{slot}"] = digest(d.pooled[
                slot * d.pooled_stride:slot * d.pooled_stride + position // 4])
        return result


def differences(actual, expected):
    return [key for key in sorted(actual.keys() | expected.keys())
            if actual.get(key) != expected.get(key)]


def state_gate(target, tokens, cycles, write, cancelled):
    """Force every partial prefix, including cancel and a following overwrite."""
    import mlx.core as mx
    rows = []
    for width in (2, 3, 4):
        for cycle in range(cycles):
            if cancelled():
                raise InterruptedError("cancelled after acknowledged launch")
            inputs = tokens[cycle * width:cycle * width + width]
            base = target.snapshot()
            expected, outputs, features = [], [], []
            for token in inputs:
                logits, hidden, _, _ = target.launch([token])
                outputs.append(raw_array(logits))
                features.append(raw_array(hidden))
                target.commit()
                expected.append(target.state())
            sequential = target.snapshot()
            target.restore(base)
            logits, hidden, _, _ = target.launch(inputs)
            logit_equal = digest(logits) == digest(mx.concatenate([restore_array(x) for x in outputs], axis=0))
            feature_equal = digest(hidden) == digest(mx.concatenate([restore_array(x) for x in features], axis=1))
            target.commit()
            row = {"width": width, "cycle": cycle, "position": base["position"],
                   "logits_exact": logit_equal, "features_exact": feature_equal,
                   "state_differences": differences(target.state(), expected[-1]), "prefixes": []}
            rows.append(row)
            write(rows)
            if not logit_equal or not feature_equal or row["state_differences"]:
                raise RuntimeError(f"sequential-vs-slab counterexample: {row}")
            for prefix in range(width):
                target.restore(base)
                target.launch(inputs)
                target.rollback()
                if prefix:
                    target.launch(inputs[:prefix])
                    target.commit()
                    mismatch = differences(target.state(), expected[prefix - 1])
                else:
                    # Cancellation closes the pending transaction without
                    # committing PLE history or future ledger positions.
                    before = target.state()
                    target.restore(base)
                    mismatch = differences(before, target.state())
                target.launch([inputs[prefix]])
                target.commit()
                continuation = differences(target.state(), expected[prefix])
                row["prefixes"].append({"accepted_inputs": prefix,
                                        "state_differences": mismatch,
                                        "following_step_differences": continuation})
                write(rows)
                if mismatch or continuation:
                    raise RuntimeError(f"rollback/replay counterexample: {row}")
            target.restore(sequential)
    return rows


def ordinary(model, caches, anchor, count):
    import mlx.core as mx
    result, rows = [], []
    start = time.perf_counter()
    for _ in range(count):
        result.append(anchor)
        logits = model(mx.array([[anchor]], mx.uint32), cache=caches)[:, -1]
        anchor = int(mx.argmax(logits, axis=-1).item())
        rows.append(logits)
    elapsed = time.perf_counter() - start
    mx.eval(*rows)
    return result, mx.concatenate(rows, axis=0), elapsed


def capture_reference_layers(model, caches, anchor):
    import mlx.core as mx
    backbone = model.language_model.model
    counts = (1, 2, 4, 8, 16, 32, 48)
    taps = []
    # Existing observation hook, with no change to the forward's arithmetic.
    if "_external_capture" in backbone.__dict__:
        raise RuntimeError("unexpected instance capture override")
    object.__setattr__(backbone, "_external_capture", lambda hidden: hidden)
    try:
        backbone(mx.array([[anchor]], mx.uint32), cache=copy.deepcopy(caches),
                 capture_layers=tuple(n - 1 for n in counts), hidden_sink=taps)
        return dict(zip(counts, (raw_array(tap) for tap in taps), strict=True))
    finally:
        object.__delattr__(backbone, "_external_capture")


def localize_layers(target, runtime, caches, anchor, references, write):
    import mlx.core as mx
    d = target.decoder
    original = d.body, d.schedule, d.dual_width
    rows = []
    try:
        for count, expected in references.items():
            schedule = runtime.MS.build_token_schedule(
                d.pack, layer_types=d.layer_types, layers=list(range(count)),
                ple_layer_ids=d.ple_layer_ids, include_lm_head=True,
                geometry=d.geometry, fusion=d.exp_fusion)
            d.body = runtime.MegakernelBody(
                d.pack, schedule, gdn_layers=len(d.gdn_layers), vocab=d.geometry.vocab,
                threads=d.threads, groups=d.groups, max_query_width=1,
                spin_cap=d.spin_cap, geometry=d.geometry, fusion=d.exp_fusion)
            d.schedule, d.dual_width = schedule, False
            target.seed(caches)
            _, hidden, _, _ = target.launch([anchor])
            ref = restore_array(expected).astype(mx.float32)
            delta = hidden.astype(mx.float32) - ref
            rows.append({"completed_layers": count,
                         "max_abs": float(mx.max(mx.abs(delta)).item()),
                         "mean_abs": float(mx.mean(mx.abs(delta)).item()),
                         "relative_l2": float((mx.linalg.norm(delta) / mx.maximum(mx.linalg.norm(ref), 1e-12)).item()),
                         "reference_mean_abs": float(mx.mean(mx.abs(ref)).item())})
            target.rollback()
            write(rows)
    finally:
        if target.pending_history is not None:
            target.rollback()
        d.body, d.schedule, d.dual_width = original
    return rows


def ast_generate(target, drafter, head_cache, initial_features, anchor, count, profile, mode,
                 cancelled=lambda: False):
    """Actual AST planning and greedy tile law over a private linear target.

    This deliberately has no tree, cross-request batching or cache publication.
    Replay is measured explicitly; today's planner does not model its price.
    """
    import mlx.core as mx

    from mlx2.runtime.ast_planner import ASTExecutionPlanner, ReadyRow, RequestWork
    from mlx2.runtime.progressive_external_verify import verify_greedy_proposal_tile
    from mlx2.runtime.speculative_sampling import RequestRNG
    planner = ASTExecutionPlanner(profile, tile_cap=3, fixed_tile=False, source_limit=3)
    rng = RequestRNG(0)
    pending, pairing = initial_features, [anchor]
    outputs, rounds = [], []
    start = time.perf_counter()
    while len(outputs) < count:
        if cancelled():
            raise InterruptedError("cancelled between closed research transactions")
        remaining = count - len(outputs)
        if mode == "plain" or remaining == 1:
            _, _, selected, _ = target.launch([anchor])
            target.commit()
            if cancelled():
                raise InterruptedError("cancelled after closing research transaction")
            outputs.append(anchor)
            anchor = selected[-1]
            continue
        limit = min(3, remaining - 1)
        budget = planner.source_budgets([RequestWork(0, limit, target.decoder.position, 1,
                                                    planner.acceptance(0))])[0]
        depth = budget.depth if mode == "ast" else (1 if mode == "k1" else limit)
        before = time.perf_counter()
        proposed, _ = drafter.draft_distributions(
            [anchor], pending, head_cache, depth, [rng], [0.0], context_tokens=[pairing])
        if cancelled():
            raise InterruptedError("cancelled before research target launch")
        draft_seconds = time.perf_counter() - before
        drafts = proposed[0]
        if len(drafts) != depth:
            raise RuntimeError("incomplete native draft in bounded research")
        inputs, offset, committed_features = [anchor, *drafts], 0, []
        round_row = {"source_budget": budget.receipt(), "depth": depth,
                     "draft_seconds": draft_seconds, "tiles": [], "replay_seconds": 0.0}
        while offset < len(inputs):
            left = len(drafts) - offset
            plan = planner.plan_launch([ReadyRow(0, 0, left, target.decoder.position,
                                                 planner.acceptance(0))], terminal=True, max_lanes=1)
            tile = plan.tile if mode == "ast" else depth
            final = left <= tile
            width = min(tile, left) + int(final)
            _, hidden, selected, elapsed = target.launch(inputs[offset:offset + width])
            if mode == "ast":
                planner.observe_launch(plan, elapsed)
            verdict = verify_greedy_proposal_tile(drafts[offset:offset + min(tile, left)],
                                                  selected, rng, final=final)
            take = verdict.accepted + 1 if verdict.rejected else width
            next_anchor = selected[take - 1]
            if take < width:
                target.rollback()
                _, hidden, _, replay = target.launch(inputs[offset:offset + take])
                round_row["replay_seconds"] += replay
            target.commit()
            if cancelled():
                raise InterruptedError("cancelled after closing research transaction")
            committed_features.append(hidden[:, :take])
            outputs.extend(inputs[offset:offset + take])
            round_row["tiles"].append({"plan": plan.receipt() if mode == "ast" else None,
                                       "actual_width": width, "accepted_inputs": take,
                                       "seconds": elapsed})
            offset += take
            if verdict.rejected or final:
                anchor = next_anchor
                break
        accepted = offset - 1
        planner.observe_commit(0, accepted, min(depth, accepted + 1), source_depth=depth)
        round_row["accepted_drafts"] = accepted
        rounds.append(round_row)
        pending = mx.concatenate(committed_features, axis=1)
        pairing = inputs[1:offset] + [anchor]
    seconds = time.perf_counter() - start
    return {"mode": mode, "tokens": outputs, "seconds": seconds,
            "tok_s_diagnostic": len(outputs) / seconds, "rounds": rounds,
            "final_state": target.state()}


def ast_diagnostic(target, caches, features, ids, anchor, binding, thermal, write,
                   cancelled=lambda: False):
    import mlx.core as mx

    from mlx2.runtime.ast_planner import ExecutionProfile
    from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel
    from mlx2.runtime.speculative_sampling import RequestRNG
    target.model.select_external_feature_source("mtp_seed")
    draft = NativeMTPDraftModel(target.model, num_draft=3, revision=binding)
    head_base = draft.make_cache()
    draft.append_context(features[:, :-1], head_base, context_tokens=[ids[1:]])
    mx.eval([cache.state for cache in head_base])
    costs = []
    for width in range(1, 5):
        if cancelled():
            raise InterruptedError("cancelled before research calibration")
        target.seed(caches)
        samples = []
        for repeat in range(4):
            _, _, _, elapsed = target.launch([anchor] * width)
            target.rollback()
            if cancelled():
                raise InterruptedError("cancelled after closing calibration transaction")
            if repeat:
                samples.append(elapsed)
        costs.append(statistics.median(samples))
    draft_samples = []
    for repeat in range(4):
        if cancelled():
            raise InterruptedError("cancelled before draft calibration")
        before = time.perf_counter()
        draft.draft_distributions([anchor], features[:, -1:], copy.deepcopy(head_base),
                                  1, [RequestRNG(0)], [0.0], context_tokens=[[anchor]])
        if repeat:
            draft_samples.append(time.perf_counter() - before)
    profile = ExecutionProfile(binding=binding, device=str(mx.device_info()), build=mx.__version__,
        max_lanes=1, tile_widths=(1, 2, 3), terminal_widths=(2, 3, 4), packed_min_width=1,
        row_costs=tuple(costs), launch_cost=0, context_row_cost=0,
        draft_cost=statistics.median(draft_samples), cost_authority="same_process_diagnostic_calibration",
        backend="private_research_megakernel", max_position_rows=4, sequential_draft=True)
    result = {"row_seconds": costs, "draft_d1_seconds": draft_samples, "runs": [],
              "scope": "one synthetic 512-token continuation; 64 outputs; diagnostic only",
              "limitations": ["ordinary arithmetic mismatch remains unexplained",
                              "AST cost profile currently omits partial-prefix replay cost",
                              "not serving, tree, batch or DFlash2 evidence"]}
    oracle = None
    for repeat in range(3):
        order = ("plain", "k1", "k3", "ast") if repeat % 2 == 0 else ("ast", "k3", "k1", "plain")
        for mode in order:
            if cancelled():
                raise InterruptedError("cancelled before AST diagnostic arm")
            before = thermal()
            if before["die_max_c"] is None or before["die_max_c"] > 65:
                raise RuntimeError("AST diagnostic thermal gate unavailable or above 65C")
            target.seed(caches)
            row = ast_generate(target, draft, copy.deepcopy(head_base), features[:, -1:],
                               anchor, 64, profile, mode, cancelled)
            row.update(repeat=repeat, thermal_before=before, thermal_after=thermal())
            if oracle is None:
                oracle = row
            row["tokens_exact"] = row["tokens"] == oracle["tokens"]
            row["state_differences"] = differences(row["final_state"], oracle["final_state"])
            result["runs"].append(row)
            write(result)
            if not row["tokens_exact"] or row["state_differences"]:
                raise RuntimeError("AST composition changed the research target stream or state")
    return result


def main():
    from benchmark_ast_repair import atomic, command, host_load, load, ownership, source
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--donor", type=Path, default=DONOR_DEFAULT)
    ap.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cpg-owner", type=Path, required=True)
    ap.add_argument("--context", type=int, default=512, choices=(512, 4096))
    ap.add_argument("--cycles", type=int, default=4, choices=range(1, 9))
    ap.add_argument("--packing-diagnostic", action="store_true", help="stop after the first expert group, before any kernel launch")
    ap.add_argument("--layer-diagnostic", action="store_true", help="compare completed-layer HC against ordinary on the first decode input")
    ap.add_argument("--ast-diagnostic", action="store_true", help="bounded native-MTP and AST composition, not a production performance claim")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    revision = source(ROOT)
    owned = ownership(args.cpg_owner)
    donor = donor_manifest(args.donor, command)
    guards = load(ROOT / "scripts/run_with_gpu_locks.py", "megakernel_probe_locks")
    if guards.foreign_model_processes():
        raise RuntimeError("foreign model process present")
    report = {"schema": "mlx2.flash-megakernel-ast-research.v1", "source_revision": revision,
              "donor": donor, "ownership": owned, "contract": TARGET_CONTRACT,
              "implementation_state": "research_harness", "qualification": False,
              "approximate_operation_qualified": False, "selected": False,
              "ordinary_equivalent": False, "apcv2_publication": False,
              "observed_used": {"packing_and_state_gate": False,
                                "layer_localization": False,
                                "native_mtp_ast": False},
              "status": "running",
              "scope": "B1 linear direct-model diagnostic; not serving or sustained throughput"}
    output = args.out_dir / "result.json"
    def save():
        atomic(output, report)
    stopped = []
    def stop(signum, frame):
        stopped.append(signum)  # Never interrupt an in-flight persistent kernel.
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    save()
    try:
        from mlx2.apple_telemetry import TemperatureSampler
        temperatures = TemperatureSampler()
        def thermal():
            return {**temperatures.die_summary(), "pmset": command("pmset", "-g", "therm"),
                    "host_load": host_load(), "swap": command("sysctl", "vm.swapusage")}
        report["host_before"] = thermal()
        if report["host_before"]["die_max_c"] is None or report["host_before"]["die_max_c"] > 65:
            raise RuntimeError("thermal diagnostic start gate unavailable or above 65C")
        if any("VirtualMachine" in p["executable"] and p["cpu_percent"] > 100
               for p in report["host_before"]["host_load"]):
            raise RuntimeError("busy VM contaminates diagnostic")
        import mlx.core as mx

        from mlx2.adapters.flash_next import FlashNextAdapter
        adapter = FlashNextAdapter(str(args.model))
        model = adapter.model
        report["checkpoint"] = args.model.name + "@" + adapter.identity["fingerprint"]
        report["artifact"] = adapter.identity
        identity_module = load(ROOT / "scripts/paired_direct_ab.py", "megakernel_probe_identity")
        report["mlx"] = identity_module.mlx_identity(mx)
        report["target_binding"] = target_binding(
            revision, report["checkpoint"], TARGET_CONTRACT, donor)
        ids = adapter.tokenizer.encode(PROMPT * (args.context // 8 + 1))[:args.context]
        report["prompt_tokens"] = ids
        caches = model.make_cache()
        hidden, features = model.mtp_backbone(mx.array([ids], mx.uint32), cache=caches)
        logits = model.logits(hidden[:, -1:])
        mx.eval(logits, features)
        anchor = int(mx.argmax(logits[:, -1], axis=-1).item())
        layer_references = capture_reference_layers(model, caches, anchor) if args.layer_diagnostic else None
        ref_tokens, ref_logits, ref_seconds = ordinary(model, copy.deepcopy(caches), anchor, 32)
        report["ordinary_reference"] = {"tokens": ref_tokens, "seconds": ref_seconds,
                                        "tok_s_diagnostic": len(ref_tokens) / ref_seconds}
        save()
        print("ordinary reference complete; checking kernel primitives", flush=True)
        # FlashNextAdapter sanitizes inherited MLX_QWEN* settings at load.
        # Apply this private kernel profile only after the ordinary reference.
        if os.environ.pop("MLX_QWEN4_MEGAKERNEL", None) != "0":
            raise RuntimeError("ordinary adapter did not explicitly disable megakernel")
        configure(args.out_dir)
        report["environment"] = {k: v for k, v in os.environ.items()
                                 if k.startswith(("MLX_MK_", "MLX_QWEN4_MEGAKERNEL"))}
        runtime, primitives = import_kernel(args.donor)
        report["primitives"] = primitives
        save()
        original_source = runtime.MP.ModuleSource
        report["packing_reference_audit"] = []
        runtime.MP.ModuleSource = split_source_class(original_source, mx, report["packing_reference_audit"])
        # Packing has large single-use concatenations. Do not let allocator
        # caching add another unpriced generation of staging buffers. Restore
        # the ordinary allocator policy before decode; never relax a guard.
        cache_limit = mx.set_cache_limit(0)
        gc.collect()
        mx.clear_cache()
        report["packing_before"] = runtime.packing_memory_snapshot()
        report["packing_cache_limit"] = {"during": 0, "restore": cache_limit}
        save()
        class ObservedDecoder(runtime.MegakernelDecoder):
            def _guard_pack_memory(self, **kwargs):
                try:
                    super()._guard_pack_memory(**kwargs)
                finally:
                    report["packing_checks"] = self.pack_memory_checks
                    report["pack_estimate"] = self.pack_estimate
                    save()
                if args.packing_diagnostic and kwargs.get("stage") == "after_group" and kwargs.get("group_index") == 1:
                    raise RuntimeError("bounded packing-only diagnostic completed")
        try:
            decoder = ObservedDecoder(model, model.language_model.args,
                                      max_context=args.context + 256, validate=True)
        finally:
            runtime.MP.ModuleSource = original_source
            mx.set_cache_limit(cache_limit)
        report["split_packing_bridge"] = {"fused_views": len(decoder.source.fused_views),
                                           "invalidated_expert_views": decoder.source.invalidated_expert_views,
                                           "layout": "gate_then_up_axis_minus_two",
                                           "full_model_fusion": False}
        target = ResearchTarget(decoder, model)
        report["topology"] = decoder.status_fields()
        report["width_preparation"] = [decoder.prepare_width_wrapper(width) for width in (2, 3, 4)]
        save()
        if layer_references is not None:
            def layer_write(rows):
                report["layer_diagnostic"] = rows
                save()
            localize_layers(target, runtime, caches, anchor, layer_references, layer_write)
            report["observed_used"]["layer_localization"] = True
            save()
        target.seed(caches)
        # Teacher-forced comparison localizes arithmetic differences without
        # letting a changed argmax turn later rows into unrelated histories.
        rows = []
        for token in ref_tokens:
            value, _, _, _ = target.launch([token])
            rows.append(value)
            target.commit()
        candidate = mx.concatenate(rows, axis=0)
        delta = mx.abs(candidate.astype(mx.float32) - ref_logits.astype(mx.float32))
        report["ordinary_arithmetic_difference"] = {
            "max_abs": float(mx.max(delta).item()), "mean_abs": float(mx.mean(delta).item()),
            "argmax_mismatch_rows": int(mx.sum(mx.argmax(candidate, axis=-1) != mx.argmax(ref_logits, axis=-1)).item()),
            "per_row_max_abs": mx.max(delta, axis=-1).tolist(),
            "per_row_mean_abs": mx.mean(delta, axis=-1).tolist(),
            "ordinary_selected": mx.argmax(ref_logits, axis=-1).tolist(),
            "megakernel_selected": mx.argmax(candidate, axis=-1).tolist(),
            "interpretation": "separate approximate target, not ordinary equivalence"}
        report["ordinary_reference_exact"] = not bool(
            report["ordinary_arithmetic_difference"]["max_abs"])
        save()
        target.seed(caches)
        mega_tokens = []
        next_token = anchor
        for _ in range(40):
            mega_tokens.append(next_token)
            _, _, selected, _ = target.launch([next_token])
            next_token = selected[-1]
            target.commit()
        report["sequential_megakernel_tokens"] = mega_tokens
        target.seed(caches)
        def gate_write(rows):
            report["state_gate"] = rows
            save()
        print("checking sequential/slab state and partial-prefix rollback", flush=True)
        state_gate(target, mega_tokens, args.cycles, gate_write, lambda: bool(stopped))
        report["observed_used"]["packing_and_state_gate"] = True
        if args.ast_diagnostic:
            def ast_write(rows):
                report["ast_diagnostic"] = rows
                save()
            ast_diagnostic(target, caches, features, ids, anchor,
                           report["target_binding"]["digest"], thermal, ast_write,
                           lambda: bool(stopped))
            report["observed_used"]["native_mtp_ast"] = True
        report["status"] = "sequential_approximate_target_state_gate_passed"
        report["launches"] = target.launches
        report["topology_after"] = decoder.status_fields()
        report["host_after"] = thermal()
        if source(ROOT) != revision or donor_manifest(args.donor, command) != donor:
            raise RuntimeError("source changed during probe")
        save()
    except BaseException as exc:
        report["status"] = "counterexample_or_refusal"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["cause"] = repr(exc.__cause__)
        if hasattr(exc, "pack_memory_checks"):
            report["packing_checks"] = exc.pack_memory_checks
        save()
        raise
    finally:
        if "target" in locals():
            report["launches"] = target.launches
            report["topology_after"] = target.decoder.status_fields()
        report["signals"] = stopped
        report["cleanup"] = "process-local research state only; no APCv2 export or VM operation"
        save()


if __name__ == "__main__":
    main()
