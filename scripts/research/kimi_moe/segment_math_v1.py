"""Research-only per-layer expert bounds; explicit stock Gate8 outside selections."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[3] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src


def validate(segments):
    seen = set()
    for s in segments:
        if set(s) - {'start_layer', 'end_layer', 'mode', 'k', 'minimum', 'maximum', 'ratio'}:
            raise ValueError('Unknown segment field')
        first, last = s['start_layer'], s['end_layer']
        if type(first) is not int or type(last) is not int or not 1 <= first <= last <= 26:
            raise ValueError('MoE layers must be in1..26')
        layers = set(range(first, last + 1))
        if seen & layers:
            raise ValueError('Overlapping segments')
        seen.update(layers)
        if s['mode'] == 'fixed':
            if type(s['k']) is not int or not 3 <= s['k'] <= 12:
                raise ValueError('Fixed count must be3..12')
            if set(s) != {'start_layer', 'end_layer', 'mode', 'k'}:
                raise ValueError('Fixed fields differ')
        elif s['mode'] == 'adaptive':
            lo, hi = s['minimum'], s['maximum']
            if type(lo) is not int or type(hi) is not int or not 3 <= lo <= hi <= 12:
                raise ValueError('Bounds must satisfy3<=min<=max<=12')
            if s['ratio'] not in [.1, .2, .3, .5, .7]:
                raise ValueError('Unknown relative raw score threshold')
            if set(s) != {'start_layer', 'end_layer', 'mode', 'minimum', 'maximum', 'ratio'}:
                raise ValueError('Adaptive fields differ')
        else:
            raise ValueError('Unknown segment mode')
    return segments


def selection(policy, layer):
    return next((s for s in policy.get('segments', []) if s['start_layer'] <= layer <= s['end_layer']), None)


def allowed(policy, layer):
    if policy['mode'] == 'fixed':
        return [8]
    s = selection(policy, layer)
    if s is None:
        return [8]
    return [s['k']] if s['mode'] == 'fixed' else list(range(s['minimum'], s['maximum'] + 1))


def install(budget, base_install):
    import mlx.core as mx

    from mlx2.runtime.models import kimi_linear as runtime

    true_gate = runtime.Gate.__call__
    true_block = runtime.SparseMoeBlock.__call__
    base_install()  # Adapter load assigns layer identities; layout binds full CONTROL.

    def record(layer, count, rows):
        budget.COUNTS[f'rows_k{count}'] += rows
        budget.COUNTS[f'layer_{layer}_k{count}'] += rows
        budget.COUNTS['expert_assignments'] += rows * count

    def gate(self, x):
        segment = selection(budget.CONTROL, self._budget_layer)
        count = segment['k'] if segment is not None and segment['mode'] == 'fixed' else 8
        previous = self.top_k
        try:
            self.top_k = count
            ids, weights = true_gate(self, x)  # Original argpartition membership/order/combine.
        finally:
            self.top_k = previous
        rows = x.size // x.shape[-1]
        budget.COUNTS['router_calls'] += 1
        budget.COUNTS['router_rows'] += rows
        record(self._budget_layer, count, rows)
        return ids, weights

    def block(self, x):
        s = selection(budget.CONTROL, self.gate._budget_layer)
        if s is None or s['mode'] == 'fixed':
            return true_block(self, x)  # Explicit original sparse branch + original Gate.
        maximum = s['maximum']
        logits = x.astype(mx.float32) @ runtime.dense_weight(self.gate.gate_proj, mx.float32).T
        scores = mx.sigmoid(logits)
        choice = scores + self.gate.expert_bias
        ids = mx.argsort(-choice, axis=-1)[..., :maximum]
        raw = mx.take_along_axis(scores, ids, -1)
        keep = raw >= s['ratio'] * raw.max(-1, keepdims=True)
        order = mx.argsort(mx.arange(maximum) + mx.where(keep, 0, maximum), axis=-1)
        ids = mx.take_along_axis(ids, order, -1).reshape(-1, maximum)
        raw = mx.take_along_axis(raw, order, -1).reshape(-1, maximum)
        widths = mx.maximum(s['minimum'], keep.sum(-1)).reshape(-1).tolist()
        flat = x.reshape(-1, x.shape[-1])
        out = mx.zeros_like(flat)
        for count in sorted(set(widths)):
            positions = mx.array([i for i, width in enumerate(widths) if width == count], mx.int32)
            selected = raw[positions, :count]
            weights = selected / (selected.sum(-1, keepdims=True) + 1e-20) * self.gate.routed_scaling_factor
            expert = self.switch_mlp(flat[positions], ids[positions, :count])
            values = (expert.astype(mx.float32) * weights[..., None]).sum(-2).astype(x.dtype)
            out = out.at[positions].add(values)
            record(self.gate._budget_layer, count, len(positions))
        budget.COUNTS['router_calls'] += 1
        budget.COUNTS['router_rows'] += len(widths)
        self.gate.calls['router_calls'] += 1
        self.gate.calls['router_tokens'] += len(widths)
        out = out.reshape(x.shape)
        return out + self.shared_experts(x) if self.shared_experts is not None else out

    runtime.Gate.__call__ = gate
    runtime.SparseMoeBlock.__call__ = block
