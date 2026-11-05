"""Process-local repeated MoE residual branch. Provenance: looped-moe-plan.json."""

from collections import Counter

# Keep opt-in: L23/top3 repaired one selected prompt but did not improve the
# frozen32K trajectory; L23+L26 caused bounded runaway repetition.
LOOP_COUNTS = Counter()


def validate_policy(selections):
    seen = set()
    for p in selections:
        if set(p) - {"layer", "top_x", "extra_passes", "gain", "include_shared"}:
            raise ValueError("unknown loop selection field")
        layer = p["layer"]
        width = p["top_x"]
        passes = p.get("extra_passes", 1)
        if type(layer) is not int or not 1 <= layer <= 26 or layer in seen:
            raise ValueError("unique MoE layers L01..L26 required")
        if type(width) is not int or not 1 <= width <= 256:
            raise ValueError("top_x must be1..256")
        if type(passes) is not int or not 1 <= passes <= 3:
            raise ValueError("extra_passes must be1..3")
        if p.get("gain", 1.0) not in (0.0, 0.25, 0.5, 1.0):
            raise ValueError("unsupported residual gain")
        if type(p.get("include_shared", False)) is not bool:
            raise ValueError("include_shared must be boolean")
        seen.add(layer)
    return selections


def install(budget, base_install=None):
    """Call once per process in place of budget.install_probe()."""
    import mlx.core as mx

    from mlx2.adapters.kimi_linear import KimiLinearAdapter
    from mlx2.runtime.models import kimi_linear as runtime

    original_gate = runtime.Gate.__call__
    original_layer = runtime.DecoderLayer.__call__
    validate_policy(budget.CONTROL.get("loops", []))
    (base_install or budget.install_probe)()
    original_diagnostics = KimiLinearAdapter.diagnostics

    def layer(self, x, mask=None, cache=None, **kwargs):
        hidden = original_layer(self, x, mask=mask, cache=cache, **kwargs)
        if not isinstance(self.mlp, runtime.SparseMoeBlock):
            return hidden
        index = self.mlp.gate._budget_layer
        selection = next(
            (p for p in budget.CONTROL.get("loops", []) if p["layer"] == index), None
        )
        if selection is None or selection.get("gain", 1.0) == 0:
            return hidden
        for _ in range(selection.get("extra_passes", 1)):
            normalized = self.post_attention_layernorm(hidden)
            gate = self.mlp.gate
            previous = gate.top_k
            try:
                gate.top_k = selection["top_x"]
                ids, weights = original_gate(gate, normalized)
            finally:
                gate.top_k = previous
            expert = self.mlp.switch_mlp(normalized, ids)
            delta = (
                (expert.astype(mx.float32) * weights[..., None])
                .sum(-2)
                .astype(normalized.dtype)
            )
            if selection.get("include_shared", False):
                delta = delta + self.mlp.shared_experts(normalized)
            hidden = hidden + selection.get("gain", 1.0) * delta
            rows = normalized.size // normalized.shape[-1]
            LOOP_COUNTS[f"layer_{index}_extra_pass_calls"] += 1
            LOOP_COUNTS[f"layer_{index}_extra_router_rows"] += rows
            LOOP_COUNTS[f"layer_{index}_extra_expert_assignments"] += (
                rows * selection["top_x"]
            )
            LOOP_COUNTS["extra_router_rows"] += rows
            LOOP_COUNTS["extra_expert_assignments"] += rows * selection["top_x"]
            if selection.get("include_shared", False):
                LOOP_COUNTS["extra_shared_rows"] += rows
        return hidden

    def diagnostics(self):
        return {
            **original_diagnostics(self),
            "looped_moe_probe": {
                "selections": budget.CONTROL.get("loops", []),
                "counts": dict(LOOP_COUNTS),
                "counter_scope": "Frontend evaluated extra passes; not backend completion counters",
                "attention_repeated": False,
                "qualified": False,
            },
        }

    runtime.DecoderLayer.__call__ = layer
    KimiLinearAdapter.diagnostics = diagnostics
