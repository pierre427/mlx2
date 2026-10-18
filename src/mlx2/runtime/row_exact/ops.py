# SPDX-License-Identifier: Apache-2.0
"""Exact multi-row operations for speculative verify windows.

Each op takes a ``[1, R, K]`` strip and returns what R separate one-row calls
would return, bit for bit, while doing the work in as few launches and weight
reads as the installed MLX build allows.  Every multi-row form is gated by a
parity canary (``canary.CANARIES``); a form that fails on this build falls
back to the per-row loop, which *is* the one-row call.  Routes are noted on
the open ``row_exact_verify`` window, so a receipt can say what ran.

Measured on M5 Max, mlx 0.32.4 (2026-10-17):

* quantized projections: ``row_exact_qmv`` (one weight read, qmv order per
  row); stock multi-row ``quantized_matmul`` is not row-invariant.
* dense bf16 projections: a batch of matrix-vector products with the weight
  broadcast (``weight[None] @ x[..., None]``) keeps decode's gemv order;
  a stock batched matmul over the rows does not (up to 38 mismatches/layer).
* row-independent ops (RMS norms, elementwise maps, reductions within a row,
  per-row top-k): one batched launch already gives each row its own bits.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
from mlx import nn

from .. import row_exact_verify as REV
from .canary import CANARIES, probe_input

_REQ = None
_VERDICTS = "_mlx2_row_exact_verdicts"


def _req():
    """``models.row_exact_qmv``, imported once (not per call)."""
    global _REQ
    if _REQ is None:
        from ..models import row_exact_qmv

        _REQ = row_exact_qmv
    return _REQ


def _module_arrays(module) -> tuple:
    """The arrays a verdict is earned on (held by the cache entry, so their
    identity cannot be reused by a later array while the entry lives)."""
    return tuple(
        module[name] if name in module else None
        for name in ("weight", "scales", "biases", "bias")
    )


def _same_arrays(held, current) -> bool:
    return len(held) == len(current) and all(a is b for a, b in zip(held, current))


def _module_verdict(module, slot, key, probe, rows, arrays=None) -> bool:
    """This module's own verdict for ``slot`` at ``rows``, cached on it.

    Weight-bound forms are probed per module (``CANARIES.exact_private``):
    a verdict earned on other arrays (another module of the same shape, or
    this module before ``update`` / adapter fusion replaced its weights) is
    never reused.  The verify path calls these ops hundreds of times per
    window, so a hit is one dict read plus an identity check of the held
    arrays.  ``key`` and ``probe`` are factories so a hit builds neither.
    ``slot`` must name what else can change between calls on one module
    (input dtype and width); ``arrays`` defaults to the module's own.
    """
    device = mx.default_device()
    current = _module_arrays(module) if arrays is None else arrays
    slot = (slot, int(rows))
    # Hit or miss under the registry lock (an uncontended re-entrant acquire
    # costs a fraction of a microsecond): a first use is single-flight, and a
    # reset, which takes the same lock, is a barrier for every verdict.
    with CANARIES._lock:
        cache = module.__dict__.get(_VERDICTS)
        if cache is None:
            cache = {}
            object.__setattr__(module, _VERDICTS, cache)
        generation = CANARIES.generation
        hit = cache.get(slot)
        if (
            hit is not None
            and hit[0] == generation
            and hit[1] == device
            and _same_arrays(hit[2], current)
        ):
            return hit[3]
        verdict = CANARIES.exact_private(key(), probe(), rows=rows)
        if CANARIES.generation != generation:
            return False
        cache[slot] = (generation, device, current, verdict)
        return verdict


_LOGITS_VERDICTS: dict = {}


def _note(stage: str, route: str, count: int = 1) -> None:
    window = REV.current()
    if window is not None:
        window.note(stage, route, count)


def strip_rows(x: mx.array) -> int:
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"row-exact ops take one [1, R, K] strip, got {tuple(x.shape)}")
    return int(x.shape[1])


def per_row(fn: Callable[[mx.array], mx.array], x: mx.array) -> mx.array:
    """The reference form: one call per row, concatenated."""
    return mx.concatenate([fn(x[:, row : row + 1]) for row in range(x.shape[1])], axis=1)


def matvec_rows(weight: mx.array, x: mx.array) -> mx.array:
    """``x @ weight.T`` for a strip as R broadcast matrix-vector products."""
    return (weight[None] @ x[0][:, :, None])[..., 0][None]


def _single_output(weight: mx.array) -> bool:
    """A one-output dense layer (``weight`` ``[1, K]``, e.g. a bf16 MoE shared
    gate).  Its one-row call ``x @ weight.T`` is an M=N=1 product that no
    broadcast or multi-row form reproduces: on MLX 0.32.4 (M5 Max) every form
    tried missed it on about 1 in 8192 bf16 rows at K=2560 (wider layers
    matched on all rows), which a canary of a few probe rows cannot see; a
    Flash-Next verify row hit it on real data (2026-10-17).  Such a layer
    always runs one one-row call per row."""
    return int(weight.shape[0]) == 1


def _dense_probe(weight, dtype):
    def probe(rows):
        x = probe_input(rows, int(weight.shape[-1]), dtype)
        return matvec_rows(weight, x), per_row(lambda row: row @ weight.T, x)

    return probe


def _plain_class(module) -> type:
    """The module's class before an installed row-exact class swap.

    A route that installs row-exact projections swaps each module's class for
    a dynamic subclass carrying ``_mlx2_row_exact_base``; its one-row call is
    the base class's call, so the op treats it as that class.
    """
    cls = type(module)
    return cls.__dict__.get("_mlx2_row_exact_base", cls)


def dense_linear(linear: nn.Linear, x: mx.array, *, stage: str = "dense") -> mx.array:
    """An unquantized ``nn.Linear`` over a strip with one-row arithmetic per row."""
    rows = strip_rows(x)
    if rows <= 1:
        return linear(x)
    if _plain_class(linear) is not nn.Linear:
        _note(stage, "per_row_unsupported")
        return per_row(linear, x)
    weight = linear.weight
    if _single_output(weight):
        _note(stage, "per_row_single_output")
        return per_row(linear, x)
    if "bias" in linear:
        # nn.Linear fuses its bias (addmm); the broadcast form adds it after.
        # A zero bias would make the two look identical in a canary, so a
        # biased layer always takes its own one-row call.
        _note(stage, "per_row_biased")
        return per_row(linear, x)
    biased = False

    def form(strip):
        return matvec_rows(weight, strip)

    def probe():
        def run(probe_rows):
            # The reference is the module's own one-row call: a biased Linear
            # fuses its bias (addmm), which the broadcast form does not.
            sample = probe_input(probe_rows, int(weight.shape[-1]), x.dtype)
            return form(sample), per_row(linear, sample)

        return run

    def key():
        return ("dense_linear", tuple(weight.shape), str(weight.dtype), str(x.dtype), biased)

    if not _module_verdict(linear, ("dense_linear", x.dtype, biased), key, probe, rows):
        _note(stage, "per_row_canary")
        return per_row(linear, x)
    _note(stage, "matvec_rows")
    return form(x)


def dense_logits(
    weight: mx.array, x: mx.array, *, stage: str = "dense", owner: mx.array | None = None
) -> mx.array:
    """``x @ weight.T`` (no module) with one-row arithmetic per row.

    ``owner`` is the stored array ``weight`` was derived from on this call
    (``router_select`` casts the router weight per call): the verdict is
    held against it, so a per-call cast neither re-probes nor accumulates.
    """
    rows = strip_rows(x)
    if rows <= 1:
        return x @ weight.T
    if _single_output(weight):
        _note(stage, "per_row_single_output")
        return per_row(lambda row: row @ weight.T, x)
    # A bare weight (a router) is weight-bound too: its verdict is its own,
    # held next to the array so the identity cannot be reused.
    identity = weight if owner is None else owner
    with CANARIES._lock:  # hit or miss: single-flight, and a reset is a barrier
        # str(device): Device objects are fresh per call and do not hash stably.
        slot = (id(identity), rows, x.dtype, CANARIES.generation, str(mx.default_device()))
        generation = slot[3]
        hit = _LOGITS_VERDICTS.get(slot)
        if hit is not None and hit[0] is identity:
            verdict = hit[1]
        else:
            # Drop entries of earlier generations so they hold no old weights.
            for stale in [k for k in _LOGITS_VERDICTS if k[3] != CANARIES.generation]:
                del _LOGITS_VERDICTS[stale]
            key = ("dense_matvec", tuple(weight.shape), str(weight.dtype), str(x.dtype))
            verdict = CANARIES.exact_private(
                key, _dense_probe(weight, x.dtype), rows=rows
            )
            if CANARIES.generation != generation:
                verdict = False
            else:
                _LOGITS_VERDICTS[slot] = (identity, verdict)
    if not verdict:
        _note(stage, "per_row_canary")
        return per_row(lambda row: row @ weight.T, x)
    _note(stage, "matvec_rows")
    return matvec_rows(weight, x)


def quantized_linear_route(
    linear: nn.QuantizedLinear,
    x: mx.array,
    *,
    stage: str = "quantized",
    call=None,
) -> tuple[mx.array, str]:
    """``(y, route)`` for a quantized projection over a strip, qmv (or the
    module's lane law) arithmetic per row.

    ``call`` is the module's one-row call when it is not ``linear`` itself
    (a tied embedding's ``as_linear``).  A lane-prepared module runs its lane
    law through ``row_exact_qmv``; the canary then compares it with the
    module's own one-row (lane) call, so the lane's row invariance is
    checked per module, shape, dtype, row count and build like every other
    form.  ``route`` is what ran: the kernel's route, or ``per_row_canary``.
    """
    REQ = _req()
    stock = linear if call is None else call
    rows = strip_rows(x)
    if rows <= 1:
        return stock(x), "one_row"
    # A tied quantized embedding projects through its ``as_linear`` call
    # (passed as ``call``); row_exact_qmv runs it like a QuantizedLinear.
    supported = isinstance(linear, nn.QuantizedLinear) or (
        call is not None and isinstance(linear, nn.QuantizedEmbedding)
    )
    if not supported:
        _note(stage, "per_row_unsupported")
        return per_row(stock, x), "per_row_unsupported"

    def key():
        return (
            "quantized_rows",
            int(linear.bits),
            int(linear.group_size),
            getattr(linear, "mode", "affine"),
            tuple(linear.weight.shape),
            str(linear.scales.dtype),
            str(linear.biases.dtype) if "biases" in linear else None,
            str(x.dtype),
            "bias" in linear,
            REQ.lane_preparation(linear) is not None,
        )

    def probe():
        def run(probe_rows):
            sample = probe_input(probe_rows, int(x.shape[-1]), x.dtype)
            return REQ.quantized_linear(linear, sample, call)[0], per_row(stock, sample)

        return run

    slot = ("quantized_rows", x.dtype, x.shape[-1], call is not None,
            REQ.lane_preparation(linear) is not None)
    if not _module_verdict(linear, slot, key, probe, rows):
        _note(stage, "per_row_canary")
        return per_row(stock, x), "per_row_canary"
    y, route = REQ.quantized_linear(linear, x, call)
    _note(stage, route)
    return y, route


def quantized_linear(
    linear: nn.QuantizedLinear, x: mx.array, *, stage: str = "quantized"
) -> mx.array:
    """A quantized projection over a strip with qmv arithmetic per row."""
    return quantized_linear_route(linear, x, stage=stage)[0]


def _quantized_group_key(linears, x: mx.array) -> tuple:
    return (
        "quantized_group",
        str(x.dtype),
        *(
            (
                int(module.bits),
                int(module.group_size),
                getattr(module, "mode", "affine"),
                tuple(module.weight.shape),
                str(module.scales.dtype),
                str(module.biases.dtype) if "biases" in module else None,
                "bias" in module,
            )
            for module in linears
        ),
    )


def quantized_linears_route(linears, x: mx.array, *, stage: str = "quantized"):
    """``(outputs, routes)`` of ``quantized_linears``: ``"group_kernel"`` for
    one grouped launch, else the list of each member's own route."""
    REQ = _req()
    linears = tuple(linears)
    if strip_rows(x) <= 1 or any(REQ.lane_preparation(m) is not None for m in linears):
        # A lane-prepared member runs its own law: each member separately,
        # each through its own canary.
        results = [quantized_linear_route(m, x, stage=stage) for m in linears]
        return tuple(y for y, _ in results), [route for _, route in results]
    return _quantized_linears(linears, x, stage=stage)


def quantized_linears(linears, x: mx.array, *, stage: str = "quantized") -> tuple:
    """Several quantized projections of one strip (outputs only)."""
    return quantized_linears_route(linears, x, stage=stage)[0]


def _quantized_linears(linears, x: mx.array, *, stage: str = "quantized"):
    """Several quantized projections of one strip, qmv arithmetic per row.

    Projections sharing their quantization run as one grouped launch
    (``row_exact_qmv.quantized_linears``, route ``group_kernel``); otherwise
    each runs its own exact form.  The canary compares the whole group with
    one one-row call per row and projection.
    """
    REQ = _req()
    linears = tuple(linears)
    rows = strip_rows(x)
    if not all(isinstance(module, nn.QuantizedLinear) for module in linears):
        _note(stage, "per_row_unsupported", len(linears))
        return (
            tuple(per_row(module, x) for module in linears),
            ["per_row_unsupported"] * len(linears),
        )

    def probe():
        def run(probe_rows):
            sample = probe_input(probe_rows, int(x.shape[-1]), x.dtype)
            grouped, _ = REQ.quantized_linears(linears, sample)
            return (
                mx.concatenate(grouped, axis=-1),
                mx.concatenate([per_row(module, sample) for module in linears], axis=-1),
            )

        return run

    # The verdict lives on the group's first module; the slot names the
    # members by identity, as row_exact_qmv's group launch plans do.
    slot = (
        "quantized_group",
        tuple(map(id, linears)),
        x.dtype,
        x.shape[-1],
    )
    arrays = tuple(array for module in linears for array in _module_arrays(module))
    if not _module_verdict(
        linears[0], slot, lambda: _quantized_group_key(linears, x), probe, rows,
        arrays=arrays,
    ):
        _note(stage, "per_row_canary", len(linears))
        return (
            tuple(per_row(module, x) for module in linears),
            ["per_row_canary"] * len(linears),
        )
    outputs, routes = REQ.quantized_linears(linears, x, per_member=True)
    if routes == "group_kernel":
        _note(stage, routes)
    else:
        for route in routes:  # each member's own route when not grouped
            _note(stage, route)
    return outputs, routes


def linear(module, x: mx.array, *, stage: str) -> mx.array:
    """Dispatch a quantized or dense projection to its exact multi-row form."""
    if isinstance(module, nn.QuantizedLinear):
        return quantized_linear(module, x, stage=stage)
    return dense_linear(module, x, stage=stage)


def row_independent(
    fn: Callable[[mx.array], mx.array],
    x: mx.array,
    *,
    stage: str,
    key: tuple,
    width: int | None = None,
) -> mx.array:
    """A batched call of an op that only reduces within each row.

    ``key`` names the op and its configuration for the canary (for example
    ``("rms_norm", eps, weight.shape)``); the probe feeds random strips of the
    same width and dtype.
    """
    rows = strip_rows(x)
    if rows <= 1:
        return fn(x)
    probe_width = int(x.shape[-1] if width is None else width)

    def probe(probe_rows):
        sample = probe_input(probe_rows, probe_width, x.dtype)
        return fn(sample), per_row(fn, sample)

    if not CANARIES.exact(
        ("row_independent", *key, probe_width, str(x.dtype)), probe, rows=rows
    ):
        _note(stage, "per_row_canary")
        return per_row(fn, x)
    _note(stage, "batched_row_independent")
    return fn(x)


def router_select(
    weight: mx.array,
    x: mx.array,
    select: Callable[[mx.array], tuple[mx.array, mx.array]],
    *,
    compute_dtype=None,
    key: tuple,
    stage: str = "router",
) -> tuple[mx.array, mx.array]:
    """Router logits (exact matvec rows) then a per-row expert selection.

    ``select`` maps logits ``[1, R, E]`` to ``(indices, scores)`` and must only
    reduce within each row (top-k, sigmoid, per-row normalization).
    """
    rows = strip_rows(x)
    if compute_dtype is not None:
        source = weight
        weight = weight.astype(compute_dtype)
        x = x.astype(compute_dtype)
    else:
        source = weight
    if rows <= 1:
        return select(x @ weight.T)
    logits = dense_logits(weight, x, stage=f"{stage}_logits", owner=source)
    experts = int(weight.shape[0])

    def probe(probe_rows):
        sample = probe_input(probe_rows, experts, logits.dtype, seed=911)
        batched = select(sample)
        rowwise = [select(sample[:, row : row + 1]) for row in range(probe_rows)]
        return (
            mx.concatenate([batched[0].astype(mx.float32), batched[1].astype(mx.float32)], axis=-1),
            mx.concatenate(
                [
                    mx.concatenate([item[0] for item in rowwise], axis=1).astype(mx.float32),
                    mx.concatenate([item[1] for item in rowwise], axis=1).astype(mx.float32),
                ],
                axis=-1,
            ),
        )

    if not CANARIES.exact(
        ("router_select", *key, experts, str(logits.dtype)), probe, rows=rows
    ):
        _note(stage, "per_row_canary")
        chosen = [select(logits[:, row : row + 1]) for row in range(rows)]
        return (
            mx.concatenate([item[0] for item in chosen], axis=1),
            mx.concatenate([item[1] for item in chosen], axis=1),
        )
    _note(stage, "batched_selection")
    return select(logits)


def routed_experts(
    experts: Callable[[mx.array, mx.array], mx.array],
    x: mx.array,
    indices: mx.array,
    *,
    num_experts: int,
    key: tuple,
    stage: str = "routed_experts",
) -> mx.array:
    """A routed-expert gather over a strip: ``experts(x, indices)`` for all rows.

    ``x`` is ``[1, R, K]`` and ``indices`` ``[1, R, top_k]``; the canary feeds
    random rows with random distinct expert choices and compares the batched
    gather with one call per row.
    """
    rows = strip_rows(x)
    if rows <= 1:
        return experts(x, indices)
    top_k = int(indices.shape[-1])

    def probe(probe_rows):
        sample = probe_input(probe_rows, int(x.shape[-1]), x.dtype, seed=707)
        order = mx.argsort(
            mx.random.uniform(shape=(1, probe_rows, num_experts), key=mx.random.key(708 + probe_rows)),
            axis=-1,
        )
        chosen = order[..., :top_k].astype(indices.dtype)
        batched = experts(sample, chosen)
        rowwise = mx.concatenate(
            [experts(sample[:, row : row + 1], chosen[:, row : row + 1]) for row in range(probe_rows)],
            axis=1,
        )
        return batched, rowwise

    full_key = ("routed_experts", *key, num_experts, top_k, int(x.shape[-1]), str(x.dtype))
    if isinstance(experts, nn.Module):
        # The experts' own weights feed the probe: a per-module verdict held
        # next to every parameter array it was earned on.
        from mlx.utils import tree_flatten

        arrays = tuple(value for _, value in tree_flatten(experts.parameters()))
        verdict = _module_verdict(
            experts,
            ("routed_experts", x.dtype, int(x.shape[-1]), top_k),
            lambda: full_key,
            lambda: probe,
            rows,
            arrays=arrays,
        )
    else:
        verdict = CANARIES.exact(full_key, probe, rows=rows)
    if not verdict:
        _note(stage, "per_row_canary")
        return mx.concatenate(
            [experts(x[:, row : row + 1], indices[:, row : row + 1]) for row in range(rows)],
            axis=1,
        )
    _note(stage, "batched_gather")
    return experts(x, indices)


def combine_experts(
    routed: mx.array, scores: mx.array, *, stage: str = "expert_combine"
) -> mx.array:
    """``sum_k routed[..., k, :] * scores[..., k]`` per row, in the output dtype.

    ``routed`` is ``[1, R, top_k, D]`` and ``scores`` ``[1, R, top_k]``.  The sum
    runs over each row's own expert axis.
    """
    rows = int(routed.shape[1])

    def combine(values, weights):
        return (values * weights[..., None]).sum(axis=-2).astype(values.dtype)

    if rows <= 1:
        return combine(routed, scores)
    top_k, width = int(routed.shape[2]), int(routed.shape[3])

    def probe(probe_rows):
        values = (
            mx.random.normal((1, probe_rows, top_k, width), key=mx.random.key(313 + probe_rows))
        ).astype(routed.dtype)
        weights = mx.random.uniform(
            shape=(1, probe_rows, top_k), key=mx.random.key(919 + probe_rows)
        ).astype(scores.dtype)
        rowwise = mx.concatenate(
            [
                combine(values[:, row : row + 1], weights[:, row : row + 1])
                for row in range(probe_rows)
            ],
            axis=1,
        )
        return combine(values, weights), rowwise

    key = ("expert_combine", top_k, width, str(routed.dtype), str(scores.dtype))
    if not CANARIES.exact(key, probe, rows=rows):
        _note(stage, "per_row_canary")
        return mx.concatenate(
            [combine(routed[:, row : row + 1], scores[:, row : row + 1]) for row in range(rows)],
            axis=1,
        )
    _note(stage, "batched_combine")
    return combine(routed, scores)
