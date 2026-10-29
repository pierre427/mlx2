# SPDX-License-Identifier: MIT
"""Source-only native HALF profile; loader capabilities retain no arrays.

Provenance: deepseek-v41-source-native-mx-policy.json. Ordinary source leases,
admission and eviction own all physical buffers. No work or GPU import here.
"""
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re

POLICY = 'original_pp_half_v1'
MAX_ATTESTATIONS = 40*384
COUNTER = 'official_source_native_mx_'


def configuration_supported(args):
    return all(getattr(args,k,None)==v for k,v in (('hidden_size',5120),('moe_intermediate_size',2304),
        ('n_routed_experts',384),('num_hidden_layers',40),('num_experts_per_tok',6),
        ('activation_dtype','bfloat16'),('simulate_quantization',True)))


def validate_policy(policy, *, expert_math, prefill_batch, wide_math='reference', wide_dtype='reference',
                    prefill_down='reference', ffn_input='reference'):
    if type(policy) is not str or policy not in ('reference',POLICY):
        raise ValueError('Invalid Source native MX policy')
    if policy!='reference' and (expert_math!='routed_fused_fp32' or prefill_batch!='gate_up'
            or any(v!='reference' for v in (wide_math,wide_dtype,prefill_down,ffn_input))):
        raise ValueError('Source native MX requires routed_fused_fp32/gate_up/reference wide,dtype,DOWN,FFN-input')


@lru_cache(maxsize=1)
def implementation_identity():
    import mlx.core as mx
    own=Path(__file__);root=own.parent.parent
    files=[own,own.parent/'deepseek_v41_source_weights.py',own.parent/'deepseek_v41_source_lfu.py',
        own.parent/'deepseek_v41_source_buffers.py',root/'runtime/gguf/source_native_mx_runtime.py',
        root/'runtime/gguf/source_native_mx.py',root/'runtime/gguf/source_middle_fusion.py']
    records={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    library=Path(mx.__file__).resolve().parent/'lib/libmlx.dylib'
    records.update(mlx_version=importlib.metadata.version('mlx'),
        mlx_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        os_version=platform.mac_ver()[0],os_kernel=platform.version(),
        device_name=mx.device_info().get('device_name'),policy=POLICY)
    return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _key(prefix,expert):
    if (type(prefix) is not str or re.fullmatch(r'blk\.(0|[1-9][0-9]?)',prefix) is None
            or type(expert) is not int or not 0<=expert<384 or int(prefix.split('.')[1])>=40):
        raise ValueError('Native MX Source expert key')
    return int(prefix.split('.')[1]),expert


class Attestations:
    """Bounded scalar metadata; no PackedExpert/backing/view/HostExpert refs."""
    def __init__(self,artifact,stream):
        self.artifact,self.stream=artifact,stream
        self.entries={}
        self.stats={'loads':0,'eligible_loads':0,'ineligible_loads':0,'missing_bindings':0}

    def loaded(self,key,host,packed):
        from .deepseek_v41_source_weights import expert_names
        from ..runtime.gguf.source_native_mx_runtime import attest_host,bind_expert
        if (type(key) is not tuple or len(key)!=2 or any(type(v) is not int for v in key)
                or not 0<=key[0]<40 or not 0<=key[1]<384):
            raise ValueError('Native MX loaded expert key')
        if len(self.entries)>=MAX_ATTESTATIONS and key not in self.entries:
            raise ValueError('Native MX metadata bound exceeded')
        files=sorted({self.artifact['tensors'][n]['file'] for n in expert_names(*key)})
        seals={n:self.artifact['seals'][n] for n in files}
        seal=hashlib.sha256(json.dumps(seals,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        a=attest_host(host,source_fingerprint=self.artifact['identity']['fingerprint'],
            source_seals_sha256=seal,layer=key[0],expert=key[1])
        self.entries[key]=bind_expert(packed,a)
        self.stats['loads']+=1
        self.stats['eligible_loads' if a.eligible else 'ineligible_loads']+=1

    def get(self,prefix,expert,data):
        key=_key(prefix,expert)
        if not self.stream._active or self.stream._leased.get(key) is not data:
            raise ValueError('Native MX binding outside exact existing expert lease')
        bound=self.entries.get(key)
        if bound is None or bound.expert_object_id!=id(data) or bound.backing_object_id!=id(data.backing):
            self.stats['missing_bindings']+=1
            return None
        if (bound.attestation.layer,bound.attestation.expert)!=key:
            raise ValueError('Native MX attested expert identity changed')
        return bound

    def receipt(self):
        return dict(self.stats,metadata_entries=len(self.entries),maximum_metadata_entries=MAX_ATTESTATIONS,
            retained_gpu_arrays=0,coefficient_scan='CPU once per newly loaded original packed expert',
            reference_policy_enable_changes_math=False)


def _count(counters,key,amount=1):
    counters[COUNTER+key]=counters.get(COUNTER+key,0)+amount


def _admit(owner,x,data,prefix,expert,*,prefill_rows,role):
    import mlx.core as mx
    from .deepseek_v41_source_buffers import PackedExpert
    if getattr(owner,'source_native_mx_policy','reference')=='reference':return None
    validate_policy(owner.source_native_mx_policy,expert_math=owner.expert_math_policy,
        prefill_batch=owner.source_prefill_batch_policy,wide_math=owner.source_prefill_wide_math,
        wide_dtype=owner.source_prefill_wide_dtype,prefill_down=owner.source_prefill_down_policy,
        ffn_input=owner.source_ffn_input_policy)
    from ..runtime.gguf.source_native_mx_runtime import canonical_original_layout
    reason=None
    if not isinstance(data,PackedExpert) or data.layout.kind!='original':reason='q1_or_unpacked'
    elif data.layout!=canonical_original_layout():reason='layout'
    elif type(prefill_rows) is not int or not 2<=prefill_rows<=99:reason='prefill_scope'
    elif (not isinstance(x,mx.array) or x.dtype!=mx.float32 or x.ndim!=2
            or x.shape[1]!=(5120 if role=='gu' else 2304)):reason='input_geometry'
    elif not 16<=x.shape[0]<=99:reason='expert_rows'
    else:
        from ..runtime.gguf.source_mxfp4_pair import _device_supported
        if mx.default_device()!=mx.gpu or not _device_supported():reason='device'
    if reason is not None:
        _count(owner.scheduled_expert_calls,'fallback_'+role+'_'+reason)
        return None
    bound=owner._source_native_mx_bound(prefix,expert,data)
    if bound is None:
        _count(owner.scheduled_expert_calls,'fallback_'+role+'_missing_attestation')
        return None
    if bound.attestation.eligible is not True:
        _count(owner.scheduled_expert_calls,'fallback_'+role+'_coefficient_domain')
        return None
    _count(owner.scheduled_expert_calls,'eligible_'+role+'_jobs')
    return bound


def maybe_gate_up(owner,x,data,prefix,expert,*,prefill_rows):
    bound=_admit(owner,x,data,prefix,expert,prefill_rows=prefill_rows,role='gu')
    if bound is None:return None
    from ..runtime.gguf.source_native_mx_runtime import prepare_half,project_half
    half=prepare_half(x)
    pair=(project_half(half,data,bound,projection='gate'),project_half(half,data,bound,projection='up'))
    for key in ('gate','up','gate_up_jobs','half_casts_gu'):_count(owner.scheduled_expert_calls,key)
    return pair


def maybe_down(owner,x,data,prefix,expert,*,prefill_rows):
    bound=_admit(owner,x,data,prefix,expert,prefill_rows=prefill_rows,role='down')
    if bound is None:return None
    from ..runtime.gguf.source_native_mx_runtime import prepare_half,project_half
    raw=project_half(prepare_half(x),data,bound,projection='down')
    for key in ('down','half_casts_down'):_count(owner.scheduled_expert_calls,key)
    return raw
