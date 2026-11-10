# SPDX-License-Identifier: MIT
"""Width-one Source MoE rows that mix packed original/Q1 experts with Q2 tier experts.

Opt-in ``deepseek_source_q2_tier_math=mixed_v1``. Under the reduced tiers a B1
row may hold Q2 objects, which the fused batch gate/up and down kernels do not
consume, so the reference path drops the WHOLE layer call to the per-expert
loop (even the packed originals lose the batch path). mixed_v1 instead:

* runs the packed original/Q1 occurrences through the same batch gate/up/middle
  kernel and the same fused down dot the non-tier B1 path uses
  (``source_gate_up_fusion.batch_middle`` + ``source_down_fusion.down_combine``);
* runs each Q2 expert with the reference Q2 law (same QAT input, block-Hadamard
  rotation for g32h, MLX affine 2-bit matmuls, fused middle, BF16 rounding),
  except that the rotated gate/up input H x is formed once per row and bank
  rotation rather than once per projection and expert (the same values);
* sums every route occurrence in the reference order (ascending expert ID,
  then ascending route slot) with the ordered F32 combine of BF16-rounded
  contributions.

A row with no Q2 occurrence is the ordinary fused row (one down_combine). A
row the batch kernels cannot take declines BEFORE any graph is built and the
caller runs the reference loop on the experts already acquired. Provenance:
provenance/deepseek-v41-source-q1-on-miss.json ("q2_tier_math mixed_v1").
"""
import hashlib

import numpy as np

POLICY = 'mixed_v1'
POLICIES = ('reference', POLICY)
REVISION = 'source-q2-tier-mixed-row-v1'


def implementation_identity():
    """Digest of this module's source; binds the numerics profile to the code."""
    from pathlib import Path
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def validate_policy(value, *, expert_math):
    if type(value) is not str or value not in POLICIES:
        raise ValueError('invalid Source Q2 tier math policy')
    if value != 'reference' and expert_math != 'routed_fused_fp32':
        raise ValueError('Source Q2 tier mixed_v1 requires routed_fused_fp32 expert math')


def _count(counts, key, amount=1):
    counts[key] = counts.get(key, 0) + amount


def q2_projection(activation, data, kind):
    """The reference Q2 matmul: MLX affine 2-bit at the bank's group size."""
    import mlx.core as mx
    return mx.quantized_matmul(activation, data[kind + '.q2weight'], data[kind + '.q2scales'],
                               data[kind + '.q2biases'], transpose=True, group_size=data.group_size,
                               bits=2).astype(mx.float32)


def q2_part(rotated, data, scores, slots):
    """One Q2 expert's BF16-rounded contribution rows (one per route occurrence).

    ``rotated`` is the QAT input already multiplied by the bank rotation (H x
    for g32h, x for g64), one row per occurrence. Same law as the reference
    branch of ``SourceV41Weights._moe_routes`` under routed_fused_fp32.
    """
    import mlx.core as mx
    from ..runtime.gguf.source_middle_fusion import middle as fused_middle
    from .deepseek_v41_source_q1_on_miss import rotate_activation
    gate, up = q2_projection(rotated, data, 'gate'), q2_projection(rotated, data, 'up')
    down_input = fused_middle(gate, up, scores, slots)
    down = q2_projection(rotate_activation(down_input, data.rotation), data, 'down')
    return down.astype(mx.bfloat16).astype(mx.float32)


def plan_row(flat, order, experts):
    """CPU-only partition of one B1 row; returns None when mixed_v1 must decline.

    ``flat`` is the row's uint32 route ids, ``order`` its stable argsort and
    ``experts`` the acquired objects by expert id. Occurrences are listed in
    the reference sum order: ascending expert id, then ascending route slot.
    """
    from .deepseek_v41_source_buffers import PackedExpert
    from .deepseek_v41_source_q1_on_miss import is_q2
    occurrences, reduced, locations = [], {}, {}
    for index in range(order.size):
        slot = int(order[index]); expert = int(flat[slot])
        data = experts[expert]
        if is_q2(data):
            reduced.setdefault(expert, []).append(slot)
        elif isinstance(data, PackedExpert):
            locations.setdefault(expert, []).append(slot)
        else:
            return None   # e.g. an unpacked Q1 dict: the reference loop owns it
        occurrences.append((expert, slot))
    if any(occurrences[i] > occurrences[i + 1] for i in range(len(occurrences) - 1)):
        raise ValueError('Source Q2 mixed row occurrences out of reference order')
    # A packed expert routed once goes to the batch kernels (the non-tier B1
    # law). One routed to several slots (route-drop zero-weight copies) keeps
    # the reference per-expert law: the reference computes it with M>1 kernels.
    packed = [(e, slots[0]) for e, slots in locations.items() if len(slots) == 1]
    repeated = {e: slots for e, slots in locations.items() if len(slots) > 1}
    return dict(occurrences=occurrences, packed=packed, repeated=repeated, reduced=reduced)


def reference_packed_part(all_selected, data, scores, locations):
    """The reference loop's packed-expert law for one expert at several route slots."""
    import mlx.core as mx
    from ..runtime.gguf import source_gate_up_fusion as gu
    from ..runtime.gguf.source_middle_fusion import exp_spelling
    rows = mx.zeros((len(locations),), mx.uint32)
    score_rows = scores[mx.array(locations, mx.uint32)]
    down_input = gu.middle(all_selected, rows, score_rows, data, exp_spelling=exp_spelling())
    if 'down.biases' in data:
        from ..runtime.gguf.source_q1 import matmul
        down = matmul(down_input, data['down.qweight'], data['down.scales'], data['down.biases'])
    else:
        down = mx.quantized_matmul(down_input, data['down.qweight'], scales=data['down.scales'], mode='mxfp4')
    return down.astype(mx.bfloat16).astype(mx.float32)


def mixed_row(all_selected, values, scores, order, experts, counts):
    """Output F32[1,5120] for one B1 row, or None (decline, no graph built).

    The caller has acquired every expert in ascending id order (the reference
    acquisition order) and owns the completion window; leases are unchanged.
    """
    import mlx.core as mx
    from ..runtime.gguf import source_gate_up_fusion as gu
    from ..runtime.gguf.source_down_fusion import down_combine
    from ..runtime.gguf.source_middle_fusion import exp_spelling
    from ..runtime.gguf.source_mxfp4_pair import _device_supported
    from .deepseek_v41_source_math import ordered_combine
    from .deepseek_v41_source_q1_on_miss import rotate_activation
    flat = values.reshape(-1)
    if values.shape[0] != 1 or all_selected is None or all_selected.shape[0] != 1:
        return None
    plan = plan_row(flat, order, experts)
    if plan is None:
        _count(counts, 'official_source_q2_mixed_declined_unpacked')
        return None
    flat_scores = scores.reshape(-1)
    packed = plan['packed']
    if packed:
        objects = [experts[e] for e, _ in packed]
        slots = mx.array([s for _, s in packed], mx.uint32)
        if not (gu.batch_supports(all_selected, objects, flat_scores, slots) and _device_supported()):
            _count(counts, 'official_source_q2_mixed_declined_batch')
            return None
    for expert, locations in plan['repeated'].items():
        rows = mx.zeros((len(locations),), mx.uint32)
        if not (gu.supports(all_selected, rows, flat_scores[mx.array(locations, mx.uint32)], experts[expert])
                and _device_supported()):
            _count(counts, 'official_source_q2_mixed_declined_repeated')
            return None
    _count(counts, 'official_source_q2_mixed_rows')
    if packed and not plan['reduced'] and not plan['repeated']:
        # No Q2 in this call (e.g. a Q4-recovery layer): the ordinary fused row.
        mid = gu.batch_middle(all_selected, objects, flat_scores, slots, exp_spelling=exp_spelling())
        output = down_combine([mid[i:i + 1] for i in range(len(objects))], objects, [e for e, _ in packed])
        _count(counts, 'official_source_q2_mixed_all_packed_rows')
        _count(counts, 'official_source_gate_up_middle_batch_fused')
        _count(counts, 'official_source_down_combine_fused')
        _count(counts, 'official_source_q2_mixed_packed_experts', len(objects))
        return output
    parts = {}
    if packed:
        mid = gu.batch_middle(all_selected, objects, flat_scores, slots, exp_spelling=exp_spelling())
        _, raw = down_combine([mid[i:i + 1] for i in range(len(objects))], objects,
                              [e for e, _ in packed], return_contributions=True)
        rounded = raw.astype(mx.bfloat16).astype(mx.float32)
        for index, (expert, slot) in enumerate(packed):
            parts[(expert, slot)] = rounded[index:index + 1]
        _count(counts, 'official_source_gate_up_middle_batch_fused')
        _count(counts, 'official_source_down_contributions_fused')
        _count(counts, 'official_source_q2_mixed_packed_experts', len(objects))
    for expert, locations in plan['repeated'].items():
        part = reference_packed_part(all_selected, experts[expert], flat_scores, locations)
        for row, slot in enumerate(locations):
            parts[(expert, slot)] = part[row:row + 1]
        _count(counts, 'official_source_gate_up_middle_fused')
        _count(counts, 'official_source_q2_mixed_repeated_packed_experts')
    rotated = {}
    for expert, locations in plan['reduced'].items():
        data = experts[expert]
        if data.rotation not in rotated:
            rotated[data.rotation] = rotate_activation(all_selected, data.rotation)
        elif data.rotation is not None:
            _count(counts, 'official_source_q2_mixed_rotations_shared')
        x = rotated[data.rotation]
        if len(locations) > 1:
            x = x[mx.zeros((len(locations),), mx.uint32)]
        part = q2_part(x, data, flat_scores, mx.array(locations, mx.uint32))
        for row, slot in enumerate(locations):
            parts[(expert, slot)] = part[row:row + 1] if len(locations) > 1 else part
        _count(counts, 'official_source_q2_tier_experts')
        _count(counts, 'official_source_q2_mixed_q2_experts')
        if data.rotation is not None:
            _count(counts, 'official_source_q2_rotated_experts')
    ordered = [parts[occurrence] for occurrence in plan['occurrences']]
    # One part per occurrence in (expert id, slot) order is the direct combine:
    # the same F32 sum, in the same order, the reference offsets map selects.
    output = ordered_combine(ordered, values, order)
    _count(counts, 'official_source_q2_mixed_ordered_combine')
    return output
