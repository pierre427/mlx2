#!/usr/bin/env python3
"""Diagnostic exclusive-time profile of ordinary Nemotron 3.5 B1 decode.

The plain arm measures an uninstrumented one-token forward.  The component and
layer arms insert ``mx.eval`` at existing module boundaries, so their absolute
times include a reported synchronization floor.  They are attribution data,
not performance qualification.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--plain-reps", type=int, default=12)
    parser.add_argument("--profile-reps", type=int, default=3)
    parser.add_argument("--settle-seconds", type=int, default=60)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        raise SystemExit("refusing Metal execution without --i-own-the-gpu")
    if not 32 <= args.context <= 2048:
        raise ValueError("context must be between 32 and 2048 tokens")

    import mlx.core as mx
    import nemotron35_probe_common as common
    from mlx import nn

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models import nemotron_h, switch_layers

    owner_before = common.require_owner(args.cpg_owner_lock)
    host_before = common.host_snapshot()
    admission_errors = common.admission_errors(
        host_before, minimum_available_bytes=4 * common.GIB, performance=True
    )
    if admission_errors:
        raise RuntimeError(f"B1 profiler admission failed: {admission_errors}")

    adapter = resolve_adapter(args.model)(str(args.model))
    model = adapter.model
    tokenizer = adapter.tokenizer
    if not 0 <= args.settle_seconds <= 300:
        raise ValueError("settle seconds must be between 0 and 300")
    time.sleep(args.settle_seconds)
    names = {id(module): name for name, module in model.named_modules()}
    prompt = ("Profile ordinary single-token decode mechanics. " * 96).strip()
    ids = tokenizer.encode(prompt)[: args.context + 1]
    if len(ids) <= args.context:
        ids = (ids * (args.context // max(1, len(ids)) + 2))[: args.context + 1]
    cache = model.make_cache()
    for start in range(0, args.context, 128):
        mx.eval(model(mx.array([ids[start : start + 128]]), cache=cache))
    base_cache = copy.deepcopy(cache)
    token = mx.array([[ids[args.context]]], dtype=mx.int32)

    mode = [None]
    stack = []
    totals = defaultdict(float)
    calls = defaultdict(int)

    original_moe_call = nemotron_h.NemotronHMoE.__call__

    def expert_phase(name, function):
        """Evaluate one expert suboperation and charge only its device wall time."""
        started = time.perf_counter()
        output = function()
        if isinstance(output, tuple):
            mx.eval(*output)
        else:
            mx.eval(output)
        totals["expert_phase:" + name] += time.perf_counter() - started
        calls["expert_phase:" + name] += 1
        return output

    def profiled_moe_call(self, x):
        if mode[0] != "expert_ops":
            return original_moe_call(self, x)

        # Drain the previous block before starting the expert timer. Otherwise
        # lazy upstream work is charged to the router merely because it is the
        # first consumer of x.
        mx.eval(x)
        total_started = time.perf_counter()
        residuals = x
        gate = self.gate
        logits = expert_phase(
            "router_projection",
            lambda: (
                x.astype(mx.float32) @ gate.weight.astype(mx.float32).T
                if gate.config.router_compute_dtype == "float32"
                else x @ gate.weight.T
            ),
        )
        inds, scores = expert_phase(
            "router_select_top22",
            lambda: nemotron_h.group_expert_select(
                logits,
                gate.e_score_correction_bias,
                gate.top_k,
                gate.n_group,
                gate.topk_group,
                gate.routed_scaling_factor,
                gate.norm_topk_prob,
            ),
        )

        if self.moe_latent_size is not None:
            x = expert_phase("latent_in", lambda: self.fc1_latent_proj(x))

        switch = self.switch_mlp
        routed = mx.expand_dims(x, (-2, -3))
        token_length = int(inds.shape[-2]) if inds.ndim >= 2 else 1
        do_sort, counter = switch_layers._switch_sort_decision(
            switch._sort_policy,
            assignments=int(inds.size),
            token_length=token_length,
            invariant=switch_layers._invariant.active(),
        )
        switch._sort_counts[counter] += 1
        idx, inv_order = inds, None
        if do_sort:
            routed, idx, inv_order = expert_phase(
                "route_sort_and_gather",
                lambda: switch_layers._gather_sort(routed, inds),
            )
        else:
            expert_phase("route_expand", lambda: routed)
        if self.training:
            idx = mx.stop_gradient(idx)

        routed = expert_phase(
            "routed_fc1", lambda: switch.fc1(routed, idx, sorted_indices=do_sort)
        )
        routed = expert_phase("relu2", lambda: switch.activation(routed))
        routed = expert_phase(
            "routed_fc2", lambda: switch.fc2(routed, idx, sorted_indices=do_sort)
        )
        if do_sort:
            routed = expert_phase(
                "route_unsort",
                lambda: switch_layers._scatter_unsort(routed, inv_order, inds.shape),
            )
        routed = routed.squeeze(-2)
        y = expert_phase(
            "weighted_top22_combine",
            lambda: (routed * scores[..., None]).sum(axis=-2).astype(routed.dtype),
        )

        if self.moe_latent_size is not None:
            y = expert_phase("latent_out", lambda: self.fc2_latent_proj(y))

        if self.config.n_shared_experts is not None:
            shared = expert_phase(
                "shared_up", lambda: self.shared_experts.up_proj(residuals)
            )
            shared = expert_phase("shared_relu2", lambda: nn.relu2(shared))
            shared = expert_phase(
                "shared_down", lambda: self.shared_experts.down_proj(shared)
            )
            y = expert_phase("routed_shared_add", lambda: y + shared)

        mx.eval(y)
        totals["expert_phase:total_wall"] += time.perf_counter() - total_started
        calls["expert_phase:total_wall"] += 1
        return y

    nemotron_h.NemotronHMoE.__call__ = profiled_moe_call

    def timed(active_mode, category, function):
        def wrapped(*positional, **keywords):
            if mode[0] != active_mode:
                return function(*positional, **keywords)
            stack.append(0.0)
            started = time.perf_counter()
            output = function(*positional, **keywords)
            mx.eval(output)
            elapsed = time.perf_counter() - started
            child = stack.pop()
            name = category(positional) if callable(category) else category
            totals[name] += elapsed - child
            calls[name] += 1
            if stack:
                stack[-1] += elapsed
            return output

        return wrapped

    dense_category = lambda pos: "dense_qmm:" + names.get(id(pos[0]), "?")
    switch_category = lambda pos: "expert_qmm:" + names.get(id(pos[0]), "?")
    layer_category = lambda pos: (
        "layer:" + names.get(id(pos[0]), "?") + ":" + pos[0].block_type
    )
    nn.QuantizedLinear.__call__ = timed(
        "components", dense_category, nn.QuantizedLinear.__call__
    )
    switch_layers.QuantizedSwitchLinear.__call__ = timed(
        "components", switch_category, switch_layers.QuantizedSwitchLinear.__call__
    )
    switch_layers.SwitchMLP.__call__ = timed(
        "components", "expert_dispatch_activation", switch_layers.SwitchMLP.__call__
    )
    nemotron_h.MoEGate.__call__ = timed(
        "components", "moe_router", nemotron_h.MoEGate.__call__
    )
    nemotron_h.NemotronHMoE.__call__ = timed(
        "components", "moe_other", nemotron_h.NemotronHMoE.__call__
    )
    nemotron_h.NemotronHMLP.__call__ = timed(
        "components", "dense_mlp_other", nemotron_h.NemotronHMLP.__call__
    )
    nemotron_h.NemotronHAttention.__call__ = timed(
        "components", "attention_other", nemotron_h.NemotronHAttention.__call__
    )
    nemotron_h.NemotronHMamba2Mixer._conv = timed(
        "components", "mamba_conv", nemotron_h.NemotronHMamba2Mixer._conv
    )
    nemotron_h.NemotronHMamba2Mixer._ssm_update_arrays = timed(
        "components", "mamba_ssm", nemotron_h.NemotronHMamba2Mixer._ssm_update_arrays
    )
    nemotron_h.MambaRMSNormGated.__call__ = timed(
        "components", "mamba_gated_norm", nemotron_h.MambaRMSNormGated.__call__
    )
    nemotron_h.NemotronHMamba2Mixer.__call__ = timed(
        "components", "mamba_other", nemotron_h.NemotronHMamba2Mixer.__call__
    )
    nn.RMSNorm.__call__ = timed("components", "rms_norm", nn.RMSNorm.__call__)
    nemotron_h.NemotronHBlock.__call__ = timed(
        "layers", layer_category, nemotron_h.NemotronHBlock.__call__
    )

    def one_forward(*, return_output=False):
        local_cache = copy.deepcopy(base_cache)
        started = time.perf_counter()
        output = model(token, cache=local_cache)
        mx.eval(output)
        elapsed = time.perf_counter() - started
        return (elapsed, output) if return_output else elapsed

    # Compile/warm every instrumented graph outside retained samples.
    for selected in (None, "components", "layers", "expert_ops"):
        mode[0] = selected
        one_forward()
    mode[0] = None
    _reference_elapsed, reference_output = one_forward(return_output=True)
    mode[0] = "expert_ops"
    _profiled_elapsed, profiled_output = one_forward(return_output=True)
    exact_mismatches = int(mx.sum(reference_output != profiled_output).item())
    if exact_mismatches:
        raise RuntimeError(
            f"instrumented expert decomposition changed {exact_mismatches} logits"
        )
    mode[0] = None
    plain = [one_forward() for _ in range(args.plain_reps)]

    z = mx.zeros((1,))
    mx.eval(z)
    sync = []
    for _ in range(200):
        started = time.perf_counter()
        mx.eval(z + 1)
        sync.append(time.perf_counter() - started)

    profiles = {}
    sync_floor = statistics.median(sync)
    for selected in ("components", "layers", "expert_ops"):
        samples = []
        for _ in range(args.profile_reps):
            totals.clear()
            calls.clear()
            stack.clear()
            mode[0] = selected
            elapsed = one_forward()
            exclusive = dict(totals)
            # total_wall is an overlapping summary across all expert phases;
            # do not subtract it a second time from the forward remainder.
            exclusive["unwrapped"] = elapsed - sum(
                value
                for key, value in exclusive.items()
                if key != "expert_phase:total_wall"
            )
            samples.append((elapsed, exclusive, dict(calls)))
        categories = sorted({key for _, row, _ in samples for key in row})
        medians = {
            key: statistics.median(row.get(key, 0.0) for _, row, _ in samples)
            for key in categories
        }
        total = sum(
            max(0.0, value)
            for key, value in medians.items()
            if key != "expert_phase:total_wall"
        )
        profiles[selected] = {
            "synchronized_forward_ms": 1000
            * statistics.median(sample[0] for sample in samples),
            "exclusive": [
                {
                    "name": key,
                    "ms": 1000 * value,
                    "sync_floor_adjusted_ms": 1000
                    * max(
                        0.0,
                        value - samples[0][2].get(key, 0) * sync_floor,
                    ),
                    "percent_of_exclusive_sum": (
                        None
                        if key == "expert_phase:total_wall"
                        else (100 * max(0.0, value) / total if total else 0.0)
                    ),
                    "calls": samples[0][2].get(key, 0),
                }
                for key, value in sorted(
                    medians.items(), key=lambda item: item[1], reverse=True
                )
            ],
        }
    mode[0] = None
    report = {
        "schema": "mlx2.nemotron35-b1-exclusive-profile.v1",
        "model": str(args.model.resolve()),
        "context_tokens": args.context,
        "shape": {"batch": 1, "sequence": 1, "speculative_proposals": 0},
        "plain_forward_ms": 1000 * statistics.median(plain),
        "plain_forward_tok_s": 1.0 / statistics.median(plain),
        "sync_floor_us": 1e6 * statistics.median(sync),
        "expert_decomposition_exact_logit_mismatches": exact_mismatches,
        "profiles": profiles,
        "owner_before": owner_before,
        "owner_after": common.require_owner(args.cpg_owner_lock),
        "host_before": host_before,
        "host_after": common.host_snapshot(),
        "limitations": [
            "per-boundary mx.eval synchronization inflates absolute component times",
            "exclusive component percentages diagnose attribution and are not performance evidence",
            "plain forward excludes HTTP, tokenization, sampling, streaming and scheduler overhead",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
