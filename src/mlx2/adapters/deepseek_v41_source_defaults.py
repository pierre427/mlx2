# SPDX-License-Identifier: MIT
"""CPU metadata scope for defaults measured on one Source precision map.

This guard selects no defaults itself. The adapter constructor still validates
all artifact, expert-file, numeric-profile and capacity contracts. Cache budgets
are deliberately independent of the immutable original/Q1 precision assignment.
"""
import hashlib
import json
from pathlib import Path
import stat

SOURCE_FINGERPRINT='c30fd25e3cc79ba7044686fa3f9a7a1c0f514f77ee6293d2f5f62bc409e02ddb'
HOT_MAP_SHA256='371051887455ccaf61f60c8e3fb74f41c7f53a2f389c88c581991007ff4024e8'
Q1_FINGERPRINT='09508a4061700fbfc06f37f024e4dbee027e7bd90bd1fe6ecc38ec8b3fc27289'
HOT_BYTES=60_000_000_000


def hot_map_provenance(raw,mapping):
    """Distinguish the measured map from recalibrated or foreign maps."""
    from .deepseek_v41_source_hotmap import provenance
    return provenance(raw,mapping,measured_sha256=HOT_MAP_SHA256,
                      source=SOURCE_FINGERPRINT,hot_budget_bytes=HOT_BYTES)


def _metadata(path,limit):
    """Bound reads to regular metadata files and reject concurrent changes."""
    before=path.lstat()
    if not stat.S_ISREG(before.st_mode)or before.st_size>limit:return None
    with path.open('rb')as stream:
        raw=stream.read(limit+1)
    after=path.lstat()
    fields=lambda value:(value.st_dev,value.st_ino,value.st_mode,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    if len(raw)>limit or fields(before)!=fields(after):return None
    value=json.loads(raw)
    return (raw,value)if type(value)is dict else None


# Defaults whose arithmetic requires other selected kernels. When an operator
# explicitly overrides a prerequisite, only the dependent default is declined;
# the rest of the measured bundle still applies. Values are never changed.
_DEPENDENT_DEFAULTS={
    'deepseek_source_hc_mega':(('deepseek_source_hc_coefficients','fused_fp32'),
                               ('deepseek_hc','reference'),('deepseek_source_hc_boundary','reference')),
    'deepseek_source_woa_storage':(('deepseek_source_bf16_loads',('woa_f32','direct_f32')),),
}


def default_route_policy_filter(route,policy,operator_policy):
    """Drop dependent defaults whose prerequisite the operator overrode."""
    if type(operator_policy)is not dict:return dict(policy)
    result=dict(policy)
    for key,requirements in _DEPENDENT_DEFAULTS.items():
        if key not in result or key in operator_policy:continue
        for name,allowed in requirements:
            if name in operator_policy and operator_policy[name]not in((allowed,)if type(allowed)is str else allowed):
                del result[key];break
    return result


def default_route_policy_applies(route,operator_policy):
    """Recognize exactly the measured ordinary Source/Hot60/Q1g64 profile."""
    if route!='ordinary'or type(operator_policy)is not dict:return False
    # These are independent serving/state/resource profiles, outside the
    # ordinary adaptive B1 evidence for the selected kernel defaults.
    for name,reference in (
        ('deepseek_physical_batch','b1'),('deepseek_vision','reference'),
        ('deepseek_ced','reference'),('deepseek_source_dspark_ast','off'),
        ('deepseek_source_dspark_prepare','off'),
        ('deepseek_source_resident_gather','reference'),
        ('deepseek_source_resident_prefill','reference'),
    ):
        if operator_policy.get(name,reference)!=reference:return False
    # Packed KV/candidate scoring now have consumers for the selected fused
    # QAT and all-retained broadcast laws; source_policy below still fails
    # closed for any combination without them.
    if operator_policy.get('deepseek_kv_storage','reference')not in ('reference','native_packed','context_gated_v1'):return False
    if operator_policy.get('deepseek_candidate_score','reference')not in ('reference','candidate_only'):return False
    # The position grid only restricts which adaptive partitions occur (whole
    # grid cells); the measured kernel bundle still applies. Canonical M1 is
    # a different arithmetic row law and stays outside this scope.
    if operator_policy.get('deepseek_source_prefill_partition','adaptive_v1')not in ('adaptive_v1','position_grid_v1'):return False
    if operator_policy.get('draft_model')is not None or operator_policy.get('ast_prompt_lookup',False)is not False:return False
    expert=operator_policy.get('deepseek_source_experts')
    if type(expert)is not dict or expert.get('mode')!='hot_original_q1':return False
    # Exact split pools and a live maximum keep the measured precision law;
    # dynamic precision is a different, approximate numerical profile.
    if expert.get('residency','shared_v1')=='dynamic_precision_v1':return False
    if type(expert.get('hot_bytes'))is not int or expert['hot_bytes']!=HOT_BYTES:return False
    names=('hotset_path','q1_root')
    if any(type(expert.get(name))is not str or not Path(expert[name]).is_absolute()for name in names):return False
    try:
        # Import only when the completed class calls its guard. Validate the
        # same merged adapter policy the server would select, with explicit
        # operator values winning; never recurse through registry resolution.
        from .deepseek_v41_source import DeepSeekV41SourceAdapter, source_policy
        namespace=vars(DeepSeekV41SourceAdapter)
        candidate=default_route_policy_filter(route,namespace.get('default_route_execution_policy',{}).get(route,{}),operator_policy)
        owned=namespace.get('adapter_owned_default_policy_keys',frozenset())
        merged={**candidate,**operator_policy}
        source_policy({key:value for key,value in merged.items()if key in owned})
        hot=_metadata(Path(expert['hotset_path']),1<<20)
        if hot is None:return False
        raw,mapping=hot
        if (mapping.get('source')!=SOURCE_FINGERPRINT
                or mapping.get('hot_budget_bytes')!=HOT_BYTES):return False
        # The pinned measured map, or a schema-valid recalibration bound to
        # the same Source and hot budget. Either digest binds the identity.
        if hot_map_provenance(raw,mapping)['kind'] not in ('measured','recalibrated'):return False
        q1=_metadata(Path(expert['q1_root'])/'manifest.json',32<<20)
        if q1 is None:return False
        _,manifest=q1
        fingerprint=manifest.pop('fingerprint',None)
        if (fingerprint!=Q1_FINGERPRINT or manifest.get('source')!=SOURCE_FINGERPRINT
                or manifest.get('quantization')!='source-q1-affine-g64-lloyd4-f16-v1'
                or manifest.get('dense_modified')is not False):return False
        actual=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return actual==Q1_FINGERPRINT
    except (OSError,ValueError,TypeError,OverflowError,RecursionError):
        return False
