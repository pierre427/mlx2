# SPDX-License-Identifier: MIT
"""Lockstep layer-major B1 chain verification with exact multi-row dense reuse.

``deepseek_verify_rows=lockstep_multirow_v1``. Each verified row executes the
ORDINARY width-one ``Block.__call__`` code path in its own greenlet. Rows
switch out only at pure dense projection sites: before attention (Qa/KV
pair, Qb, Engram KV) and after it (WoA, WoB, router, shared expert, head). Each
row reads its own decode offset through a floor view equal to the token-major
value. When every row is parked at the same site, a
multi-row kernel whose per-row arithmetic is the stock M=1 law reads the
coefficients once (runtime/gguf/source_multirow_dense.py); otherwise, and for
every unrecognized format or policy, each row's original projection runs
unchanged. Attention, compressor, indexer, MoE and all cache mutation stay
width-one and run atomically per row in row order (no site parks between the
first cache write and WoA), so row r sees exactly the state that token-major
decoding would show it.

Non-source compressed layers see their source pool through the same bounded
per-row owner view as ``verify_layerwise``. Local frontiers are captured right
after each row's attention, before the next row mutates that layer.

Implemented, unqualified, not selected by default. Provenance:
provenance/deepseek-v41-dspark-chain-verify.json.
"""
import hashlib
import re
from copy import copy
from functools import lru_cache
from pathlib import Path

POLICY = "lockstep_multirow_v1"
# Exact block-level pure sites. Compressor and indexer projections (including
# indexer.attn_q_b) run inside the attention critical region and stay inline.
_PRE_ATTENTION = re.compile(r"blk\.\d+\.(attn_q_a|attn_kv|attn_q_b|engram_kv)\.weight")
_POST_ATTENTION = re.compile(r"blk\.\d+\.(attn_output_b|ffn_gate_inp|ffn_down_shexp)\.weight|output\.weight")


def _post_attention(kind, name):
    return kind in ("grouped", "gate_up", "moe") or (
        kind == "linear" and type(name) is str and _POST_ATTENTION.fullmatch(name) is not None)


class _Park:
    __slots__ = ("args", "key", "kind", "kwargs", "name", "original")

    def __init__(self, kind, name, args, kwargs, original):
        self.kind, self.name, self.args, self.kwargs, self.original = kind, name, args, kwargs, original
        self.key = (kind, name)


def _coalescable(name):
    return type(name) is str and (_PRE_ATTENTION.fullmatch(name) is not None
                                  or _POST_ATTENTION.fullmatch(name) is not None)


class _RowState:
    """The real layer cache, except that offset reads at least the row floor.

    Attention reads ``state.offset`` on entry, before its pre-attention parks;
    lower rows may not have appended yet. Token-major decoding would show
    ``base + row`` there. All other reads, writes and methods are the real
    object's, and after this row's own append the real offset exceeds the floor.
    """
    __slots__ = ("_floor", "_real")

    def __init__(self, real, floor):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_floor", floor)

    def __getattr__(self, name):
        if name == "offset":
            real = self._real.offset
            return max(self._floor, real)
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        setattr(self._real, name, value)


class SourceRowsExecutor:
    """Run one parked site for R rows; exact multirow law or per-row originals."""

    def __init__(self, weights, stats):
        self.weights, self.stats = weights, stats

    def _bump(self, key, value=1):
        key = "lockstep_" + key
        self.stats[key] = self.stats.get(key, 0) + value

    def _admitted(self):
        if getattr(self.weights, "source_multirow_dense_contract", False):
            return True
        from .deepseek_v41_source_weights import SourceV41Weights
        return isinstance(self.weights, SourceV41Weights)

    def _gpu(self):
        import mlx.core as mx
        return mx.default_device() == mx.gpu and mx.metal.is_available()

    def run(self, parks):
        if len(parks) > 1 and self._gpu() and self._admitted():
            try:
                value = getattr(self, "_" + parks[0].kind)(parks)
            except _Decline:
                value = None
            if value is not None:
                self._bump("multirow_sites")
                self._bump("multirow_rows", len(parks))
                return value
        self._bump("row_sites", len(parks))
        return [p.original(*p.args, **p.kwargs) for p in parks]

    def _rows(self, parks, ndim):
        import mlx.core as mx
        xs = [p.args[0] for p in parks]
        if any(not isinstance(x, mx.array) or x.ndim != ndim or x.shape[:2] != (1, 1)
               or x.shape != xs[0].shape for x in xs):
            raise _Decline
        return xs

    def _moe(self, parks):
        # deepseek_verify_moe_rows=m1_rows_v1 only (LockstepRows installs the
        # moe site only then): one acquisition per distinct expert per layer.
        from . import deepseek_v41_dspark_chain_moe as rows_moe
        try:
            return rows_moe.rows_moe(self.weights, parks, self.stats)
        except rows_moe.KernelDecline:
            # Distinct from the precondition declines: routes were already observed.
            self._bump("moe_kernel_declines")
            raise _Decline from None
        except rows_moe.Decline:
            self._bump("moe_declines")
            raise _Decline from None

    def _qmv(self, prepared, name):
        import mlx.core as mx

        from ..runtime.gguf import source_multirow_dense as multirow
        q, scales = self.weights._quant[name]
        k = q.shape[1] * 4
        stacked = mx.concatenate([p.astype(mx.float32).reshape(1, k) for p in prepared])
        try:
            multirow.qmv_law(q.shape[0], k)
        except ValueError:
            raise _Decline from None
        self._bump("multirow_mxfp8_calls")
        return multirow.qmv(stacked, q, scales)

    def _linear(self, parks):
        import mlx.core as mx

        from ..runtime.gguf import source_multirow_dense as multirow
        w, name = self.weights, parks[0].name
        if any(p.kwargs for p in parks):
            raise _Decline
        xs = self._rows(parks, 3)
        quant = getattr(w, "_quant", None)
        if quant is not None and name in quant:
            # Same input law as the ordinary apply(); fp8_pp/q_b tiles only own
            # 99-row prefill shapes and decline at M=1.
            if not callable(getattr(w, "_dense_input", None)) or getattr(w, "source_fp8_pp_policy", "reference") not in (
                    "reference", "native_bf16_qb_wob_v1"):
                raise _Decline
            w._ready()
            out = self._qmv([w._dense_input(x) for x in xs], name)
            return [out[r].reshape(1, 1, -1) for r in range(len(xs))]
        weight = w.array(name)
        if not isinstance(weight, mx.array) or weight.ndim != 2 or weight.dtype not in (mx.float32, mx.bfloat16):
            raise _Decline
        try:
            multirow.gemv_bm(weight.shape[0], weight.shape[1])
        except ValueError:
            raise _Decline from None
        stacked = mx.concatenate([x.astype(mx.float32).reshape(1, 1, -1) for x in xs])
        self._bump("multirow_gemv_calls")
        out = multirow.gemv(stacked, weight)
        return [out[r].reshape(1, 1, -1) for r in range(len(xs))]

    def _grouped(self, parks):
        import mlx.core as mx

        from ..runtime.gguf import source_dense
        from ..runtime.gguf import source_multirow_dense as multirow
        w, name = self.weights, parks[0].name
        groups = parks[0].kwargs.get("groups", 8)
        if (any(p.kwargs.get("groups", 8) != groups or set(p.kwargs) - {"groups"} or len(p.args) != 2 for p in parks)
                or getattr(w, "source_woa_pp_policy", "reference") != "reference"
                # MXFP8 WoA storage keeps no resident BF16 coefficient; its rows
                # run the ordinary per-row MXFP8 consumer instead.
                or getattr(w, "source_woa_storage_policy", "reference") != "reference"
                or getattr(w, "source_bf16_loads_policy", "reference") not in ("woa_f32", "direct_f32")):
            raise _Decline
        session = getattr(w, "source_resident_bank_session", None)
        if session is not None and session.active:
            raise _Decline
        xs = self._rows(parks, 4)
        weight = w.array(name)
        if not source_dense.supports(xs[0].astype(mx.float32), weight, "wo_a"):
            raise _Decline
        stacked = mx.concatenate([x.astype(mx.float32).reshape(1, groups, -1) for x in xs])
        out = multirow.gemv(stacked, weight, groups=groups)
        key = "source_bf16_grouped_direct_f32"
        w.scheduled_grouped_calls[key] = w.scheduled_grouped_calls.get(key, 0) + len(xs)
        self._bump("multirow_grouped_calls")
        return [out[r].reshape(1, 1, groups, -1) for r in range(len(xs))]

    def _paired(self, parks):
        import mlx.core as mx
        w = self.weights
        if any(len(p.args) != 3 or set(p.kwargs) - {"rounded"} for p in parks):
            raise _Decline
        a, b = parks[0].args[1:]
        rounded = parks[0].kwargs.get("rounded", True)
        if (type(rounded) is not bool or any(p.args[1:] != (a, b) or p.kwargs.get("rounded", True) is not rounded
                                             for p in parks) or not callable(getattr(w, "can_fuse_dense", None))):
            raise _Decline
        xs = self._rows(parks, 3)
        if not all(w.can_fuse_dense(x, a, b) for x in xs):
            raise _Decline
        w._ready()
        prepared = [w._dense_input(x) for x in xs]
        outputs = []
        for name in (a, b):
            raw = self._qmv(prepared, name)
            outputs.append(raw.astype(mx.bfloat16).astype(mx.float32) if rounded else raw.astype(mx.float32))
        for key in ("source_qakv_qat_reuse_pairs", "source_qakv_input_qat_calls_avoided"):
            w.scheduled_dense_calls[key] = w.scheduled_dense_calls.get(key, 0) + len(xs)
        return [(outputs[0][r].reshape(1, 1, -1), outputs[1][r].reshape(1, 1, -1)) for r in range(len(xs))]

    def _gate_up(self, parks):
        w = self.weights
        if (any(len(p.args) != 3 or p.kwargs.get("ffn_input_token") is not None or set(p.kwargs) - {"ffn_input_token"}
                for p in parks)
                or not callable(getattr(w, "_dense_input", None))):
            raise _Decline
        gate, up = parks[0].args[1:]
        if any(p.args[1:] != (gate, up) for p in parks) or gate not in w._quant or up not in w._quant:
            raise _Decline
        xs = self._rows(parks, 3)
        w._ensure_dense()
        prepared = [w._dense_input(x) for x in xs]
        first, second = self._qmv(prepared, gate), self._qmv(prepared, up)
        # Reference QAT prepares twice, fused once; the prepared values agree.
        if getattr(w, "source_dense_qat_policy", "reference") == "fused_fp32":
            key = "source_dense_gate_up_qat_reused"
            w.scheduled_dense_calls[key] = w.scheduled_dense_calls.get(key, 0) + len(xs)
        return [(first[r].reshape(1, 1, -1), second[r].reshape(1, 1, -1)) for r in range(len(xs))]


class _Decline(Exception):
    """A structural refusal: the caller runs each row's original projection."""


class LockstepRows:
    """Greenlet scheduler; one ordinary row body per greenlet, row order kept."""

    def __init__(self, weights, executor, *, moe_rows=False):
        self.weights, self.executor = weights, executor
        self.moe_rows = bool(moe_rows)
        self._rows = {}
        self._hooks = {}

    def _in_row(self):
        from greenlet import getcurrent
        return self._rows.get(id(getcurrent()))

    def park(self, kind, name, original, args, kwargs):
        from greenlet import getcurrent
        row = self._in_row()
        if row is None:
            return original(*args, **kwargs)
        # Frontiers are taken at the first site after this row's attention.
        hook = self._hooks.pop(row, None) if _post_attention(kind, name) else None
        if hook is not None:
            hook()
        return getcurrent().parent.switch(_Park(kind, name, args, kwargs, original))

    def before_first_park(self, row, hook):
        self._hooks[row] = hook

    def install(self):
        w = self.weights
        names = ("linear", "grouped", "gate_up", "paired_project") + (("moe",) if self.moe_rows else ())
        originals = {name: getattr(w, name) for name in names if hasattr(w, name)}
        if any(name in vars(w) for name in originals):
            raise ValueError("lockstep verification found an existing weights override")
        scheduler = self

        def linear(name):
            apply = originals["linear"](name)
            if not _coalescable(name):
                return apply
            return lambda x, *a, **k: scheduler.park("linear", name, apply, (x, *a), k)
        w.linear = linear
        if "grouped" in originals:
            w.grouped = lambda x, name, **k: scheduler.park("grouped", name, originals["grouped"], (x, name), k)
        if "gate_up" in originals:
            w.gate_up = lambda x, g, u, **k: scheduler.park("gate_up", (g, u), originals["gate_up"], (x, g, u), k)
        if "moe" in originals:
            # Routed MoE follows the row's router and touches no cache state.
            w.moe = lambda x, ids, scores, prefix, **k: scheduler.park(
                "moe", prefix, originals["moe"], (x, ids, scores, prefix), k)
        if "paired_project" in originals:
            w.paired_project = lambda x, a, b, **k: scheduler.park(
                "paired", (a, b), originals["paired_project"], (x, a, b), k)
        return originals

    def uninstall(self, originals):
        for name in originals:
            vars(self.weights).pop(name, None)

    def run(self, bodies):
        from greenlet import GreenletExit, getcurrent, greenlet
        main = getcurrent()
        lets = [greenlet(body, parent=main) for body in bodies]
        self._rows = {id(g): row for row, g in enumerate(lets)}
        results, parked = [None] * len(lets), {}
        try:
            def advance(row, *value):
                out = lets[row].switch(*value)
                if lets[row].dead:
                    results[row] = out
                else:
                    if not isinstance(out, _Park):
                        raise RuntimeError("lockstep row yielded a foreign object")
                    parked[row] = out
            for row in range(len(lets)):
                advance(row)
            while parked:
                # Serve the lowest parked row's site, with every row parked at
                # the same site. Rows then resume strictly in increasing order.
                key = parked[min(parked)].key
                rows = [row for row in sorted(parked) if parked[row].key == key]
                values = self.executor.run([parked.pop(row) for row in rows])
                if len(values) != len(rows):
                    raise RuntimeError("lockstep executor returned a different row count")
                for row, value in zip(rows, values):
                    advance(row, value)
            return results
        finally:
            for g in lets:
                if not g.dead:
                    try:
                        g.throw(GreenletExit)
                    except GreenletExit:
                        pass
            self._rows, self._hooks = {}, {}


def refusal(model):
    """Return a reason string when this model/profile cannot use lockstep rows."""
    w = model.weights
    if getattr(model, "kv_storage_policy", "reference") == "native_packed":
        return "native packed KV requires serial snapshot/replay verification"
    if getattr(model, "ced_policy", "reference") != "reference":
        return "dependency CED is an ordinary prefill profile"
    if callable(getattr(w, "evaluate_block", None)):
        return "adapter-owned block transactions are not lockstep-compatible"
    if getattr(w, "source_row_packet_policy", "reference") != "reference":
        return "Source row packets own contiguous prefill frames"
    if getattr(w, "source_resident_bank_session", None) is not None:
        return "resident bank sessions own their own B1 transaction"
    if getattr(w, "source_resident_gather_policy", "reference") != "reference":
        return "resident gather requires its prepared B1 bank session"
    if getattr(w, "expert_io_policy", "blocking") == "overlap":
        return "overlapped expert tickets assume one in-flight row"
    if getattr(w, "source_ffn_input_policy", "reference") != "reference":
        return "FFN input tokens are frame-scoped to one row"
    if getattr(w, "dense_epilogue_policy", "reference") != "reference":
        return "fused dense epilogues bypass the lockstep projection sites"
    if getattr(model, "_mlx2_ds41_ast_active", None) is not None or getattr(model, "_mlx2_ds41_ast_poisoned", False):
        return "an AST transaction is active or poisoned"
    if getattr(w, "_source_current_frame", None) is not None:
        return "a Source frame is already active"
    try:
        import greenlet  # noqa: F401
    except ImportError:
        return "greenlet is unavailable"
    return None


def verify(model, inputs, cache, capture_layers=(), *, retain_frontiers=True, executor=None):
    """Logits[1,W,V], taps[1,W,3H] or None, local frontiers (layerwise format)."""
    reason = refusal(model)
    if reason is not None:
        raise ValueError("lockstep_multirow_v1 refused: " + reason)
    model._weight_executions += 1
    try:
        execution = getattr(model.weights, "execution_scope_for_rows", None)
        from contextlib import nullcontext
        scope = execution(inputs.shape[1]) if callable(execution) else nullcontext()
        with scope:
            return _verify(model, inputs, cache, tuple(capture_layers), retain_frontiers, executor)
    except BaseException as error:
        discard = getattr(model, "discard_engram_prefetch", None)
        if discard is not None:
            discard(error)
        raise
    finally:
        model._weight_executions -= 1


def _verify(model, inputs, cache, capture_layers, retain_frontiers, executor):
    import mlx.core as mx
    import numpy as np

    from ..runtime.models.deepseek_v4_cache import _copy

    if not 1 <= inputs.shape[1] <= 6:
        raise ValueError("lockstep verification requires one to six rows")
    if capture_layers and (tuple(sorted(set(capture_layers))) != capture_layers
                           or any(type(i) is not int or not 0 <= i < len(model.layers) for i in capture_layers)):
        raise ValueError("lockstep taps require ordered valid layer IDs")
    cache, ids = model.validate_forward(inputs, cache, 0)
    c, width = model.args, inputs.shape[1]
    stats = getattr(model, "verified_commit_stats", None)
    stats = {} if stats is None else stats
    from .deepseek_v41_dspark_chain_moe import LAW as MOE_ROWS
    lockstep = LockstepRows(model.weights, executor or SourceRowsExecutor(model.weights, stats),
                            moe_rows=getattr(model, "verify_moe_rows_policy", "reference") == MOE_ROWS)
    hashes, tails = None, []
    if model.hasher:
        tail = np.asarray(cache[0].hash_tail)
        hashes, _ = model.hasher(ids[0], tail)
        for token in ids[0]:
            _, tail = model.hasher(np.asarray([token]), tail)
            tails.append(mx.array(tail, mx.int32))
        for j, layer in enumerate(c.engram_layer_ids):
            model.engram_tables[layer].prefetch(hashes[:, j])
    stream = getattr(model.weights, "expert_stream", None)
    if stream is not None and stream.atlas is not None:
        from ..runtime.expert_stream_atlas import ExpertStreamAtlas
        stream.begin_context(ExpertStreamAtlas.keys(model.hasher.token_map[ids[0]],
                                                    np.asarray(cache[0].hash_tail)[::-1]))
    hidden, pre = [], []
    with model._weight_scope():
        for row in range(width):
            x = model.round(model.weights.embedding(inputs[:, row:row + 1].astype(mx.uint32)))
            hidden.append(mx.broadcast_to(x[:, :, None], (1, 1, c.hc_mult, c.hidden_size)))
            pre.append(mx.broadcast_to(mx.array([1.0] + [0.0] * (c.hc_mult - 1)), (1, 1, c.hc_mult)))
        model._weight_eval(hidden, pre)
    shared = [{} for _ in range(width)]
    taps = [[] for _ in range(width)]
    frontiers = [[] for _ in range(width - 1)] if retain_frontiers else []
    originals = lockstep.install()
    try:
        for state in cache:
            state._in_forward = True
        for layer, (block, state) in enumerate(zip(model.layers, cache)):
            base = state.offset
            with model._weight_scope():
                engram = None
                if layer in c.engram_layer_ids:
                    index = c.engram_layer_ids.index(layer)
                    engram = mx.array(model.engram_tables[layer].read(hashes[:, index]))

                def frontier(row, layer=layer, state=state):
                    if layer == 0 and tails:
                        state.hash_tail = tails[row]
                    if retain_frontiers and row < width - 1:
                        state.validate_state()
                        local = _copy(mx, (state.keys, state.buf_kv, state.buf_gate, state.hash_tail))
                        frontiers[row].append((state.meta_state, local, state._pool_count, state._pooled is not None))

                def body(row, layer=layer, block=block, state=state, engram=engram, base=base):
                    h = hidden[row]
                    if engram is not None:
                        h = model.engram(h, layer, None, rows=engram[row:row + 1])
                    if layer in capture_layers:
                        taps[row].append(model.round(mx.mean(h, axis=-2)))
                    visible = cache
                    if state.ratio and not state.source:
                        owner_index = c.kv_source_for(layer)
                        owner = copy(cache[owner_index])
                        owner.offset = base + row + 1  # token-major extent for this row
                        owner._pool_count = owner.offset // owner.ratio
                        visible = list(cache)
                        visible[owner_index] = owner
                    marked = []
                    def mark():
                        marked.append(True)
                        frontier(row)
                    lockstep.before_first_park(row, mark)
                    result = block(h, pre[row], _RowState(state, base + row), visible, shared[row])
                    if not marked:
                        lockstep._hooks.pop(row, None)
                        frontier(row)
                    return result

                for row, (h, p) in enumerate(lockstep.run([lambda r=row: body(r) for row in range(width)])):
                    hidden[row], pre[row] = h, p
                if model.eval_interval and mx.default_device() == mx.gpu:
                    mx.async_eval(hidden, pre)
                model._weight_eval(hidden, pre, [v.state for v in cache], shared, frontiers, taps)
        with model._weight_scope():
            def head(row):
                normed = model.norm(model.collapse(hidden[row], pre[row]), model.norm_weight)
                return model.linear(normed, "output.weight", rounded=False)
            logits = lockstep.run([lambda r=row: head(r) for row in range(width)])
            features = (mx.concatenate([mx.concatenate(row, -1) for row in taps], 1)
                        if capture_layers else None)
            model._weight_eval(logits, features)
    finally:
        lockstep.uninstall(originals)
    for state in cache:
        state.validate_state()
        state._in_forward = False
    stats["lockstep_calls"] = stats.get("lockstep_calls", 0) + 1
    stats["lockstep_rows"] = stats.get("lockstep_rows", 0) + width
    key = f"lockstep_width_{width}_calls"
    stats[key] = stats.get(key, 0) + 1
    if stream is not None:
        stream.end_context()
    return mx.concatenate(logits, 1), features, frontiers


@lru_cache(maxsize=1)
def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    paths = (Path(__file__), root / "runtime/gguf/source_multirow_dense.py",
             Path(__file__).with_name("deepseek_v41_dspark_chain_moe.py"))
    return hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest() for p in paths)).hexdigest()
