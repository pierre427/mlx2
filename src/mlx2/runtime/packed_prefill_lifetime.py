"""Explicit stagewise cold-prefill lifetime and conservative byte admission.

No model/runtime import occurs here. Tensor math stays in the model methods;
this owner records evaluation roots and counts every declared live component.
"""
from __future__ import annotations
import os
import sys
import time



def gdn_evaluation_wave_plan(counts, activation_bound, max_segments=1):
    """Conservative live-array plan for pinned394 BF16 depthwise GDN.

    Shared reserve retains residual/norm, full qkv/z/b/a and all outputs/join.
    Per-token wave reserve includes possible contiguous copies and FP32 norm
    temporaries, not only the custom kernel's two outputs. No workspace credit.
    """
    if (type(max_segments) is not int or max_segments not in (1,2,4) or
            type(activation_bound) is not int or activation_bound <= 0 or
            type(counts) is not tuple or not 1 <= len(counts) <= 20 or
            any(type(n) is not int or not 256 <= n <= 8192 for n in counts)):
        raise ValueError('explicit bounded GDN wave/count/charge required')
    shared = 78016 * sum(counts)
    groups=[];start=0
    while start < len(counts):
        chosen=1
        if max_segments > 1:
            for size in range(2,min(max_segments,len(counts)-start)+1):
                peak=shared+222720*sum(counts[start:start+size])+6496256*size
                if peak <= activation_bound: chosen=size
        groups.append(tuple(range(start,start+chosen)));start+=chosen
    if max_segments > 1 and not any(len(g)>1 for g in groups):
        raise MemoryError('no grouped GDN evaluation fits existing activation charge')
    peaks=tuple(shared+222720*sum(counts[i] for i in g)+6496256*len(g) for g in groups)
    return {'max_segments':max_segments,'groups':tuple(groups),'segment_lengths':counts,
            'shared_reserve_bytes':shared,'working_bytes_per_token':222720,
            'working_bytes_per_segment':6496256,'grouped_peak_bytes':max((p for p,g in zip(peaks,groups) if len(g)>1),default=0),
            'activation_bound_bytes':activation_bound,'qualified':False}


def _require_grouped_gdn_policy(candidate, max_segments):
    if (os.environ.get('MLX_GDN_PACKED')!='1' or os.environ.get('MLX_GDN_CORE')!='0' or
            os.environ.get('MLX2_PAGED_GDN_EVAL_WAVE_MAX_SEGMENTS')!=str(max_segments)):
        raise ValueError('grouped GDN requires explicit packed1/core0 environment')
    import_guard=sys.modules.get('mlx2.runtime.models.import_env')
    check=getattr(import_guard,'assert_profile_applied',None)
    if not callable(check):raise ValueError('grouped GDN import environment proof absent')
    check('N20 materialized GDN waves',environ=os.environ)
    for index in candidate.layer_map.recurrent:
        module=candidate.trunk.layers[index].linear_attn;conv=module.conv1d
        if (getattr(conv,'groups',None)!=10240 or getattr(conv,'stride',None)!=1 or
                getattr(conv,'dilation',None)!=1 or getattr(conv,'padding',None)!=0 or
                getattr(conv.weight,'shape',None)!=(10240,4,1) or hasattr(conv,'bias') or
                str(conv.weight.dtype).split('.')[-1]!='bfloat16' or
                str(module.norm.weight.dtype).split('.')[-1]!='bfloat16'):
            raise ValueError('grouped GDN requires pinned BF16 depthwise conv/gated norm')

def n20_stagewise_charge_components(candidate, counts, caps):
    if (getattr(candidate, '_prefill_packed_n20', False) is not True or
            getattr(candidate, '_prefill_layer_lifetime', False) is not True or
            getattr(candidate, '_prefill_eval_block_size', 1) != 1 or
            type(counts) is not tuple or type(caps) is not tuple or
            not 1 <= len(counts) <= 20 or len(counts) != len(caps) or
            any(type(n) is not int or not 256 <= n <= 8192 or
                type(cap) is not int or not 1 <= cap <= 192 or n + cap > 8192
                for n, cap in zip(counts, caps))):
        raise ValueError('explicit bounded N20 layer lifetime/count/cap required')
    a = candidate.args
    if (a.hidden_size != 5120 or a.intermediate_size != 17408 or
            a.num_attention_heads != 24 or a.num_key_value_heads != 4 or
            getattr(a, 'head_dim', None) != 256 or
            a.linear_num_key_heads != 16 or a.linear_num_value_heads != 48 or
            a.linear_key_head_dim != 128 or a.linear_value_head_dim != 128 or
            a.linear_conv_kernel_dim != 4 or candidate.native_layer_count != 16 or
            len(candidate.layer_map.recurrent) != 48 or
            candidate.native_dtype_preflight() != 'bfloat16'):
        raise ValueError('stagewise contract requires proven dense BF16 D256 geometry')
    for index in candidate.layer_map.recurrent:
        module = candidate.trunk.layers[index].linear_attn
        if (getattr(module, 'training', True) is not False or
                getattr(module, '_prefill_scan_chunk', 0) != 0 or
                getattr(module, '_gdn_state_dtype', None) is not None or
                getattr(module, 'sharding_group', None) is not None or
                hasattr(module, '_prefill_counts') or
                not callable(getattr(module, 'mixed_materialized', None))):
            raise ValueError('stagewise recurrent kernel/state policy differs')
    mlp_mode=getattr(candidate,'_prefill_mlp_materialization_mode','staged_qmm')
    if mlp_mode not in ('staged_qmm','single_eval_qmm','staged_bf16','single_eval_bf16',
                        'tiled_q4_swiglu','packed_gate_up_qmm'):
        raise ValueError('stagewise MLP experiment policy differs')
    if any(not callable(getattr(layer.mlp, 'materialized', None)) or
            hasattr(layer.mlp, '_prefill_counts') or
            getattr(layer.mlp,'_prefill_materialization_mode',mlp_mode)!=mlp_mode
            for layer in candidate.trunk.layers):
        raise ValueError('stagewise MLP capability absent before allocation')
    if mlp_mode in ('tiled_q4_swiglu','packed_gate_up_qmm'):
        geometry=getattr(candidate,'_prefill_mlp_geometry',None)
        if (type(geometry) is not dict or geometry.get('layer_count')!=len(candidate.trunk.layers) or
                any(getattr(layer.mlp,'_prefill_mlp_geometry',None)!=geometry.get('geometry')
                    for layer in candidate.trunk.layers)):
            raise ValueError('inferred tiled SwiGLU geometry proof absent')
    rows, longest = sum(counts), max(counts)
    h, intermediate = a.hidden_size, a.intermediate_size
    q = 24 * 256
    kv = 4 * 256
    conv, value, key = 2 * 16 * 128 + 48 * 128, 48 * 128, 16 * 128
    # Inputs/residuals and simultaneous output arrays are explicit. The MLP
    # peak includes both original projections plus the evaluated SwiGLU output.
    mlp = rows * (2 * h + 3 * intermediate) * 2
    fa = rows * (3 * h + 4 * q + 4 * kv) * 2
    # Full projected inputs and concatenated readouts, plus the single ordinary
    # segment's conv/QK/FP32 gated-norm working arrays. Segments never overlap.
    gdn = rows * (2 * h + conv + 3 * value + 2 * 48) * 2
    gdn += longest * ((4 * conv + 4 * key) * 2 + 3 * value * 4 + 8 * 48 * 4)
    state_per_lane = 48 * (48 * 128 * 128 * 4 + 3 * conv * 2)
    pages = 16 * sum((n + cap + 63) // 64 + 2 for n, cap in zip(counts, caps))
    components = {
        'arena_bytes': 2 * pages * kv * 64 * 2,
        'persistent_recurrent_bytes': len(counts) * state_per_lane,
        'recurrent_successor_overlap_bytes': len(counts) * state_per_lane,
        'layer_activation_bytes': max(mlp, fa, gdn),
        'workspace_bytes': 512 << 20,
        'final_logits_bytes': len(counts) * a.vocab_size * 2,
        # All 16 FA Q1 scratch blocks may coexist until final decode evaluation.
        'decode_scratch_bytes': 16 * len(counts) * 24 * 128 * (256 + 2) * 4,
    }
    if mlp_mode.endswith('_bf16'):
        # Gate, up and down have the same element count. A single-eval graph may
        # retain all three dequantized BF16 tables until its output completes;
        # charge that worst case rather than relying on allocator scheduling.
        components['mlp_dequantized_projection_bytes'] = 3 * h * intermediate * 2
    wave=getattr(candidate,'_prefill_gdn_eval_wave_max_segments',1)
    expanded=getattr(candidate,'_prefill_gdn_eval_wave_expanded_charge',False)
    if type(expanded) is not bool or expanded and wave==1:
        raise ValueError('expanded GDN charge requires a grouped wave')
    if expanded:
        unconstrained=gdn_evaluation_wave_plan(counts,sys.maxsize,wave)
        required=max(
            unconstrained['shared_reserve_bytes']+
            unconstrained['working_bytes_per_token']*sum(counts[i] for i in group)+
            unconstrained['working_bytes_per_segment']*len(group)
            for group in unconstrained['groups'])
        extra=max(0,required-components['layer_activation_bytes'])
        if extra:components['gdn_grouped_wave_extra_bytes']=extra
    activation_bound=(components['layer_activation_bytes']+
                      components.get('gdn_grouped_wave_extra_bytes',0))
    plan=gdn_evaluation_wave_plan(counts,activation_bound,wave)
    if plan['max_segments']>1:_require_grouped_gdn_policy(candidate,wave)
    return components


class StageMaterialization:
    """Retain the exact stage roots until an explicit evaluation succeeds.

    This is a submitted-root guard, not a whole-process allocation meter.
    The source ledger separately covers residual/caller-held arrays and
    temporary coexistence; admission also counts model parameters/headroom.
    """
    def __init__(self, mx, activation_bound, *, gdn_wave_plan=None):
        if gdn_wave_plan is not None:
            expected=gdn_evaluation_wave_plan(gdn_wave_plan['segment_lengths'],activation_bound,gdn_wave_plan['max_segments'])
            if gdn_wave_plan != expected:raise ValueError('materializer wave plan/source charge differs')
        self.gdn_wave_plan = gdn_wave_plan
        self.mx = mx
        self.activation_bound = activation_bound
        self.failure_roots = ()
        self.evaluations = 0
        self.stages = []
        self.maximum_evaluated_bytes = 0
        self.stage_seconds = {}
        self.stage_maximum_evaluated_bytes = {}
        self.stage_memory_maxima = {}

    def retain_pending_roots(self, *values):
        # Capture before constructing the next lazy segment: a graph-build
        # error must retain earlier outputs/cache leaves as well as projections.
        self.failure_roots = tuple(values)

    def __call__(self, stage, *values):
        if not values:
            raise ValueError('materialization requires explicit roots')
        self.failure_roots = tuple(values)
        size = sum(getattr(v, 'nbytes', 0) for v in {id(v): v for v in values}.values())
        if size > self.activation_bound:
            raise MemoryError('materialized stage exceeds source-bound activation charge')
        self.maximum_evaluated_bytes = max(self.maximum_evaluated_bytes, size)
        began = time.perf_counter()
        self.mx.eval(*values)
        elapsed = time.perf_counter() - began
        self.evaluations += 1
        self.stages.append(stage)
        self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + elapsed
        self.stage_maximum_evaluated_bytes[stage] = max(
            self.stage_maximum_evaluated_bytes.get(stage, 0), size)
        getters = {
            'active_bytes': getattr(self.mx, 'get_active_memory', None),
            'cache_bytes': getattr(self.mx, 'get_cache_memory', None),
            'peak_bytes': getattr(self.mx, 'get_peak_memory', None),
        }
        memory = self.stage_memory_maxima.setdefault(stage, {})
        for label, getter in getters.items():
            if callable(getter):
                memory[label] = max(memory.get(label, 0), int(getter()))
        self.failure_roots = ()
