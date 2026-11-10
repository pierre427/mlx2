"""Untimed, caller-owned AST M16 versus B1 routed-MoE localization.

Diagnostic calls are explicit extra component oracles, never candidate tile
work. All production results are returned unchanged. No GPU runs on import.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
from contextlib import contextmanager, ExitStack
import importlib.util
from pathlib import Path
import numpy as np


@contextmanager
def _instrument_one(model, check_live, *, layer=14, node=12, compact=False, host_budget_bytes=20<<20):
    if not callable(check_live):
        raise ValueError("AST MoE diagnostic needs an owned live callback")
    if type(layer) is not int or not 0 <= layer <= 14 or node != 12:
        raise ValueError("AST diagnostic is bounded to layers0..14/node12")
    import mlx.core as mx
    from mlx2.runtime.gguf import source_gate_up_prefill as pp, source_middle_fusion as mf
    from mlx2.runtime.gguf import source_gate_up_fusion as gu
    from mlx2.runtime.gguf.source_q1 import matmul as q1_matmul
    from mlx2.runtime.models.deepseek_v41_quant import fused_quantize
    spec = importlib.util.spec_from_file_location("ds41_ast_comparison", Path(__file__).with_name("ds41_ast_target_components.py"))
    diagnostics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostics)
    compare_arrays = diagnostics.compare_arrays
    weights, stream = model.weights, model.weights.expert_stream
    if getattr(weights, "source_prefill_batch_policy", None) != "gate_up":
        raise ValueError("AST localization requires the actual gate_up PP profile")
    original, pp_original, middle_original = weights.moe, pp.pp_middle, mf.middle
    report = dict(layer=layer, node=node, extra_component_oracles=True,
                  candidate_outputs_unchanged=True, qualified=False,
                  performance_claim=False, extra_moe_oracle_calls=1,
                  compact_node_only=compact, experts=[], raw_arrays={})
    active, captured, ordinary_calls = [False], [], [0]
    def compare(a, b):
        return compare_arrays(a, b, atol=0., rtol=0.)
    def host(name, value):
        if compact and name in ("candidate_x", "candidate_ids", "candidate_scores", "candidate_output", "prepared_x"):
            value = value[node:node+1]
        array = np.asarray(value)
        total = sum(v.nbytes for v in report["raw_arrays"].values()) + array.nbytes
        if total > host_budget_bytes:
            raise MemoryError("AST per-layer diagnostic host byte ceiling")
        report["raw_arrays"][name] = array.copy()
    def candidate_node(name):
        value = report["raw_arrays"][name]
        return value if compact else value[node:node+1]
    def pp_hook(x, experts, rows, scores, counts, **kwargs):
        result = pp_original(x, experts, rows, scores, counts, **kwargs)
        if active[0]:
            packed = result[0] if isinstance(result, tuple) else result
            offsets = np.cumsum((0,) + tuple(counts))
            maps = np.asarray(rows)
            for i, expert in enumerate(experts):
                captured.append((expert, maps[offsets[i]:offsets[i+1]].copy(),
                                 packed[offsets[i]:offsets[i+1]], "production_batched_middle"))
        return result
    def middle_hook(gate, up, scores, slots, **kwargs):
        result = middle_original(gate, up, scores, slots, **kwargs)
        if active[0]:
            captured.append((None, np.asarray(slots).copy(), result,
                             (gate, up)))
        return result
    def moe(x, ids, scores, prefix, **kwargs):
        target = prefix == f"blk.{layer}"
        if not target:
            return original(x, ids, scores, prefix, **kwargs)
        if x.shape[0] == 1 and "candidate_seen" in report:
            ordinal = ordinary_calls[0]
            ordinary_calls[0] += 1
            result = original(x, ids, scores, prefix, **kwargs)
            # The surrounding full-tree probe visits each branch's root+five
            # ordinary rows; third branch depth2 is ordinal14.
            if ordinal == 14:
                mx.eval(x, ids, scores, result)
                for name, value in (("ordinary_node12_x", x), ("ordinary_node12_ids", ids),
                                    ("ordinary_node12_scores", scores), ("ordinary_node12_output", result)):
                    host(name, value)
                report["ordinary_node12"] = dict(
                    x=compare(x, candidate_node("candidate_x")),
                    ids=compare(ids, candidate_node("candidate_ids")),
                    scores=compare(scores, candidate_node("candidate_scores")),
                    output_vs_m1_same_input=compare(result, report["raw_arrays"]["same_input_m1_output"]))
            return result
        if x.shape != (16, 5120) or "candidate_seen" in report:
            raise ValueError("AST diagnostic expected exactly one M16 candidate per selected layer")
        check_live()
        active[0] = True
        try:
            output = original(x, ids, scores, prefix, **kwargs)
        finally:
            active[0] = False
        mx.eval(x, ids, scores, output, [item[2] for item in captured])
        values = np.asarray(ids)
        order = np.argsort(values.reshape(-1), kind="stable")
        sorted_ids = values.reshape(-1)[order]
        for name, value in (("candidate_x", x), ("candidate_ids", ids),
                            ("candidate_scores", scores), ("candidate_output", output)):
            host(name, value)
        report["candidate_seen"] = True
        # Source ownership remains in the current dense/completion window.
        # Reacquisition is deliberately forbidden for this diagnostic.
        if not stream._active:
            raise ValueError("AST component oracle requires the original live expert window")
        prepared = fused_quantize(x.astype(mx.float32), "fp8").astype(x.dtype).astype(mx.float32)
        host("prepared_x", prepared)
        for expert_id in sorted(map(int, values[node])):
            data = stream._leased.get((layer, expert_id))
            if data is None:
                raise ValueError("AST diagnostic lost the producing expert lease")
            positions = order[sorted_ids == expert_id]
            rows = positions // values.shape[1]
            local = int(np.flatnonzero(rows == node)[0])
            record = next((item for item in captured if item[0] is data), None)
            if record is None:
                record = next((item for item in captured if item[0] is None and np.array_equal(item[1], positions)), None)
            if record is None:
                raise ValueError("AST diagnostic omitted the actual production middle")
            middle = record[2]
            slot = int(np.flatnonzero(values[node] == expert_id)[0])
            one_score = scores[node:node+1, slot].reshape(1)
            m1, gate1, up1 = gu.middle(prepared, mx.array([node], mx.uint32), one_score,
                                      data, exp_spelling=mf.exp_spelling(), return_projections=True)
            if isinstance(record[3], tuple):
                gate_full, up_full = record[3]
                component_scope = "production_stock_fallback_projections"
            else:
                middle_replay, gate_full, up_full = pp_original(
                    prepared, [data], mx.array(rows, mx.uint32),
                    scores.reshape(-1)[mx.array(positions, mx.uint32)], (len(rows),),
                    exp_spelling=mf.exp_spelling(), return_projections=True)
                mx.eval(middle_replay)
                component_scope = "replayed_identical_local_M_batched_projections"
            def down(v):
                if "down.biases" in data:
                    raw = q1_matmul(v, data["down.qweight"], data["down.scales"], data["down.biases"])
                else:
                    raw = mx.quantized_matmul(v, data["down.qweight"], scales=data["down.scales"], mode="mxfp4")
                return raw.astype(mx.bfloat16).astype(mx.float32)
            full_down, one_down = down(middle), down(m1)
            mx.eval(m1, gate1, up1, gate_full, up_full, full_down, one_down)
            entry = dict(expert=expert_id, kind=data.layout.kind, local_M=len(rows), local_row=local,
                         projection_capture=component_scope,
                         gate=compare(gate_full[local:local+1], gate1),
                         up=compare(up_full[local:local+1], up1),
                         middle=compare(middle[local:local+1], m1),
                         replayed_down=compare(full_down[local:local+1], one_down))
            report["experts"].append(entry)
            for name, value in (("production_middle", middle), ("gate_full", gate_full),
                    ("up_full", up_full), ("m1_middle", m1), ("m1_gate", gate1),
                    ("m1_up", up1), ("replayed_full_down", full_down), ("m1_down", one_down)):
                if compact and name in ("production_middle", "gate_full", "up_full", "replayed_full_down"):
                    value = value[local:local+1]
                host(f"expert{expert_id}_{name}", value)
        m1_output = original(x[node:node+1], ids[node:node+1], scores[node:node+1], prefix)
        mx.eval(m1_output)
        host("same_input_m1_output", m1_output)
        report["same_input_moe"] = compare(output[node:node+1], m1_output)
        check_live()
        return output
    weights.moe, pp.pp_middle, mf.middle = moe, pp_hook, middle_hook
    try:
        yield report
    finally:
        weights.moe, pp.pp_middle, mf.middle = original, pp_original, middle_original


@contextmanager
def instrument(model, check_live, *, layer=14, node=12, layers=None):
    """Compose disjoint layer hooks; keep row12 raw arrays below20MiB total."""
    selected = (layer,) if layers is None else tuple(layers)
    if (not selected or len(set(selected)) != len(selected)
            or any(type(i) is not int or not 0 <= i <= 14 for i in selected)):
        raise ValueError("AST diagnostic layers must be unique within0..14")
    if layers is None:
        with _instrument_one(model, check_live, layer=layer, node=node) as report:
            yield report
        return
    reports = []
    aggregate = dict(layers=[], raw_arrays={}, extra_component_oracles=True,
                     extra_moe_oracle_calls=len(selected), candidate_outputs_unchanged=True,
                     selected_layers=list(selected), compact_node_only=True,
                     host_byte_limit=20<<20, qualified=False, performance_claim=False)
    with ExitStack() as stack:
        for selected_layer in selected:
            reports.append(stack.enter_context(_instrument_one(model, check_live,
                layer=selected_layer, node=node, compact=True,
                host_budget_bytes=(20<<20)//len(selected))))
        yield aggregate
    for row in reports:
        aggregate["raw_arrays"].update({f"layer{row['layer']}_" + name: value
                                        for name, value in row.pop("raw_arrays").items()})
        aggregate["layers"].append(row)
    aggregate["host_raw_bytes"] = sum(v.nbytes for v in aggregate["raw_arrays"].values())
    aggregate["first_same_input_moe_difference"] = next((row["layer"] for row in sorted(reports, key=lambda r: r["layer"])
        if not row.get("same_input_moe", {}).get("raw_bits_equal", False)), None)
    aggregate["first_ordinary_node12_input_difference"] = next((row["layer"] for row in sorted(reports, key=lambda r: r["layer"])
        if not row.get("ordinary_node12", {}).get("x", {}).get("raw_bits_equal", False)), None)
