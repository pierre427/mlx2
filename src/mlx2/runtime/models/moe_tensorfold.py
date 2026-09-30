"""Opt-in TensorFold-style coalescing of routed MoE gate/up projections.

``SwitchGLU`` normally launches two gathered projections that read the same
input with the same expert indices.  ``install`` stacks each block's gate and
up expert banks along the expert axis into one ``2E``-expert table and routes
every assignment twice (``e`` for gate, ``E + e`` for up), so one
``gather_mm``/``gather_qmm`` launch computes both.  Each row keeps the stock
output width and reduction depth, so kernel selection matches the two stock
launches; the halves come back as ``out[0]`` and ``out[1]``.

Storage: the original projections are rebound to ``table[:E]`` and
``table[E:]``, contiguous zero-copy views of the stacked table.  Nothing is
resident twice, the stock path (and a direct ``gate_proj(...)`` call) reads
row-contiguous expert matrices exactly as before, and the stacked table stays
outside the model's parameter tree.  An output-axis ``[gate | up]`` stack
would leave per-expert strided views, and Metal's sorted ``gather_qmm_rhs``
copies a non-contiguous weight whole on every call.

Admission is structural and fails closed: exact ``SwitchGLU`` with exact
``SwitchLinear``/``QuantizedSwitchLinear`` pairs of identical type, shapes,
dtypes and quantization, and no parameters outside that contract.  Per call
the coalesced path runs only when the process switch is on, the block is not
training, and the pair still holds the installed views; a later ``update``,
``load_weights``, quantize, shard or stream rewrite is detected, counted,
and served by the stock path.  Detection and counters are host-side, so under
``mx.compile`` they reflect the trace, not each replay.

Installing selects the mechanism for that model; nothing installs it by
default and no adapter branches on it.  The receipt reports ``installed``,
``selected``, and ``qualified`` separately; observed use is
``stats()["calls"]``.  The law is exact on CPU; GPU exactness is pending, so
a route that installs it binds ``LAW_ID`` into its cache identity
(``apc_fingerprint``) until a GPU gate shows it bitwise equal to stock.
"""

from __future__ import annotations

from collections import Counter

import mlx.core as mx
from mlx import nn

from . import switch_layers as _switch_layers
from .switch_layers import QuantizedSwitchLinear, SwitchGLU, SwitchLinear

LAW_ID = "moe-gate-up-tensorfold-v1"
_ATTR = "_tensorfold_gate_up"  # read by SwitchGLU.__call__; outside the parameter tree

# Process-wide switch for paired A/B measurement (stock path when off).  An
# installed model keeps the stacked views either way; ``uninstall`` restores
# independent banks.
ENABLED = [True]

# Host-side counters with a fixed key set: calls, assignments, sorted_calls,
# unsorted_calls, tail_padded_calls, fallback_training, stale_dropped.
STATS: Counter = Counter()

_ALLOWED = {
    SwitchLinear: frozenset({"weight", "bias"}),
    QuantizedSwitchLinear: frozenset({"weight", "scales", "biases", "bias"}),
}


class TensorFoldUnsupported(ValueError):
    """The pair cannot be coalesced without changing its structural contract."""


def _arrays(module) -> dict:
    return {k: v for k, v in module.items() if isinstance(v, mx.array)}


def _validate_pair(gate, up) -> None:
    if not isinstance(gate, nn.Module) or not isinstance(up, nn.Module):
        raise TensorFoldUnsupported("gate/up are not modules")
    if type(gate) is not type(up):
        raise TensorFoldUnsupported("gate/up projection types differ")
    allowed = _ALLOWED.get(type(gate))
    if allowed is None:
        raise TensorFoldUnsupported("gate/up are not generic SwitchLinear projections")
    if set(gate.keys()) != set(up.keys()):
        raise TensorFoldUnsupported("gate/up parameter sets differ")
    if not set(gate.keys()) <= allowed or len(_arrays(gate)) != len(gate):
        raise TensorFoldUnsupported("gate/up carry state outside the SwitchLinear contract")
    if type(gate) is QuantizedSwitchLinear:
        if "scales" not in gate:
            raise TensorFoldUnsupported("quantized gate/up have no resident scales")
        for name in ("group_size", "bits", "mode"):
            if getattr(gate, name) != getattr(up, name):
                raise TensorFoldUnsupported(f"gate/up quantization {name} differs")
    if "weight" not in gate or gate["weight"].ndim != 3:
        raise TensorFoldUnsupported("gate/up weight is not an expert-major bank")
    experts = int(gate["weight"].shape[0])
    for name in sorted(gate.keys()):
        left, right = gate[name], up[name]
        if left.shape != right.shape:
            raise TensorFoldUnsupported(f"gate/up {name} shapes differ")
        if left.dtype != right.dtype:
            raise TensorFoldUnsupported(f"gate/up {name} dtypes differ")
        if left.ndim < 2 or int(left.shape[0]) != experts:
            raise TensorFoldUnsupported(f"gate/up {name} is not expert-major")


def _format(module) -> str:
    if type(module) is QuantizedSwitchLinear:
        return f"{module.mode}-q{module.bits}-g{module.group_size}"
    return f"dense-{str(module['weight'].dtype).removeprefix('mlx.core.')}"


class _GateUpGroup:
    """One ``2E``-expert projection whose halves back the block's gate/up views."""

    __slots__ = ("format", "gate", "gate_views", "num_experts", "projection", "up", "up_views")

    def __init__(self, gate, up):
        _validate_pair(gate, up)
        kind = type(gate)
        experts = int(gate["weight"].shape[0])
        projection = kind.__new__(kind)
        nn.Module.__init__(projection)
        if kind is QuantizedSwitchLinear:
            projection.group_size = gate.group_size
            projection.bits = gate.bits
            projection.mode = gate.mode
            if "biases" not in gate:
                projection.biases = None
        self.gate_views, self.up_views = {}, {}
        for name in sorted(gate.keys()):
            table = mx.concatenate([gate[name], up[name]], axis=0)
            mx.eval(table)
            projection[name] = table
            self.gate_views[name] = table[:experts]
            self.up_views[name] = table[experts:]
            setattr(gate, name, self.gate_views[name])  # frees the original bank
            setattr(up, name, self.up_views[name])
            mx.eval(self.gate_views[name], self.up_views[name])
        projection.freeze()
        self.projection = projection
        self.num_experts = experts
        self.gate, self.up = gate, up
        self.format = _format(gate)

    def current(self, module) -> bool:
        """True while ``module`` still computes with the installed views."""
        if module.get("gate_proj") is not self.gate or module.get("up_proj") is not self.up:
            return False
        for half, views in ((self.gate, self.gate_views), (self.up, self.up_views)):
            arrays = _arrays(half)
            if arrays.keys() != views.keys():
                return False
            if any(arrays[name] is not view for name, view in views.items()):
                return False
        return True

    def admit(self, module) -> bool:
        if not ENABLED[0]:
            return False
        if module.training:
            # The stacked table is not in the parameter tree: gradients must
            # flow through the ordinary projections.
            STATS["fallback_training"] += 1
            return False
        if not self.current(module):
            # Weights were rebound after install.  Drop the stale table; the
            # stock path reads the live parameters.  A half that still holds
            # a view keeps the table alive until install()/uninstall().
            module.__dict__.pop(_ATTR, None)
            STATS["stale_dropped"] += 1
            return False
        return True

    def __call__(self, x, indices, *, sorted_indices: bool):
        rows = int(indices.size)
        STATS["calls"] += 1
        STATS["assignments"] += rows
        STATS["sorted_calls" if sorted_indices else "unsorted_calls"] += 1
        trim = None
        limit = _switch_layers._SORTED_GATHER_TAIL_ROWS
        if (sorted_indices and _switch_layers._SORTED_GATHER_TAIL_BUG
                and 2 * rows > limit and (2 * rows) % 64):
            # Doubling the rows can enter the sorted-gather ragged-tail
            # corruption window that _gather_sort guards for n rows.  Pad the
            # sorted rows (last row repeated, still sorted) so 2n is a
            # multiple of 64; the padded outputs are trimmed.
            pad = 32 - rows % 32
            x = mx.concatenate([x, mx.broadcast_to(x[-1:], (pad,) + x.shape[1:])], axis=0)
            indices = mx.concatenate([indices, mx.broadcast_to(indices[-1:], (pad,))], axis=0)
            trim = rows
            STATS["tail_padded_calls"] += 1
        # x broadcasts over the leading axis: one launch, every assignment
        # routed to its gate expert and to its up expert.  Still sorted.
        routed = mx.stack([indices, indices + self.num_experts])
        # MLX's sorted gather_qmm path assumes one sorted rhs-index stream.
        # The gate/up axis broadcasts ``x`` across two individually sorted
        # streams; on Metal, large rows with native quantized K geometry can
        # be read as one stream and return incorrect rows. Dense gather_mm is
        # exact, but quantized groups must decline that specialization. This
        # remains one coalesced gather_qmm launch.
        safe_sorted = sorted_indices and type(self.projection) is SwitchLinear
        out = self.projection(x, routed, sorted_indices=safe_sorted)
        if trim is not None:
            return out[0, :trim], out[1, :trim]
        return out[0], out[1]


def _release(module, group: _GateUpGroup) -> None:
    """Give every half still on the stacked table its own bank, then drop the table."""
    for half, views in ((group.gate, group.gate_views), (group.up, group.up_views)):
        owned = [name for name, view in views.items() if half.get(name) is view]
        copies = {name: mx.array(views[name]) for name in owned}
        mx.eval(list(copies.values()))
        for name, copy in copies.items():
            setattr(half, name, copy)
    module.__dict__.pop(_ATTR, None)


def install(model) -> dict:
    """Coalesce every admissible ``SwitchGLU`` in ``model``; returns a receipt."""
    covered: Counter = Counter()
    refused: Counter = Counter()
    for _name, module in model.named_modules():
        if not isinstance(module, SwitchGLU):
            continue
        if type(module) is not SwitchGLU:
            refused["SwitchGLU subclass owns its forward"] += 1
            continue
        group = module.__dict__.get(_ATTR)
        if group is not None:
            if group.current(module):
                covered["already"] += 1
                continue
            _release(module, group)
        try:
            group = _GateUpGroup(module.get("gate_proj"), module.get("up_proj"))
        except TensorFoldUnsupported as exc:
            refused[str(exc)] += 1
            continue
        object.__setattr__(module, _ATTR, group)
        covered[group.format] += 1
    installed = sum(1 for _name, module in model.named_modules()
                    if type(module) is SwitchGLU and module.__dict__.get(_ATTR) is not None)
    return {
        "law_id": LAW_ID,
        "installed": installed,
        "selected": installed > 0 and ENABLED[0],
        "qualified": False,
        "covered": dict(covered),
        "refused": dict(refused),
    }


def uninstall(model) -> int:
    """Restore independent gate/up banks and drop the stacked tables."""
    restored = 0
    for _name, module in model.named_modules():
        group = module.__dict__.get(_ATTR) if isinstance(module, SwitchGLU) else None
        if group is not None:
            _release(module, group)
            restored += 1
    return restored


def apc_fingerprint(base, receipt):
    """APCv2 namespace for state computed while the coalesced law is installed."""
    if not receipt or not receipt.get("installed"):
        return base
    return (base, "moe-gate-up-tensorfold", receipt["law_id"])


def set_enabled(enabled: bool) -> None:
    """Process-wide on/off for paired A/B measurement."""
    ENABLED[0] = bool(enabled)


def stats() -> dict:
    return dict(STATS)
