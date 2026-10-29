# SPDX-License-Identifier: MIT
"""Embedded official V4.1 DSpark binding; never reuse a foreign sidecar.

CPU inspection creates virtual tensor names over the SAME sealed extents.
No bytes are converted and the target hot-original/Q1 map never selects MTP
experts. Device construction is explicit and requires a live owner callback.
See provenance/deepseek-v41-dspark-ast.json.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, replace

from .deepseek_v41_dspark_artifact import DSparkConfig as EmbeddedDSparkConfig
from .deepseek_v41_dspark_artifact import dspark_admission_plan

SOURCE_FINGERPRINT='c30fd25e3cc79ba7044686fa3f9a7a1c0f514f77ee6293d2f5f62bc409e02ddb'
POLICY='dspark_ast_3x5_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class DSparkConfig:
    stages:int=3
    block_size:int=5
    target_layers:tuple=(37,38,39)
    noise_token_id:int=128799
    markov_rank:int=256
    expert_count:int=128
    top_k:int=3
    hidden_size:int=5120
    vocab_size:int=129280
    window:int=128

    @classmethod
    def from_artifact(cls,artifact):
        if artifact['identity']['fingerprint']!=SOURCE_FINGERPRINT:
            raise ValueError('DSpark requires exact official V4.1 Source artifact')
        c=artifact['config']['text_config']
        actual=(c.get('num_nextn_predict_layers'),c.get('dspark_block_size'),tuple(c.get('dspark_target_layer_ids',())),
            c.get('dspark_noise_token_id'),c.get('dspark_markov_rank'),c.get('dspark_n_routed_experts'),c.get('dspark_num_experts_per_tok'))
        expected=(3,5,(37,38,39),128799,256,128,3)
        if actual!=expected:raise ValueError('Embedded V4.1 DSpark geometry differs from pinned3x5 head')
        target=artifact['args']
        if (target.hidden_size,target.num_hidden_layers,target.n_routed_experts,target.num_experts_per_tok,
                target.vocab_size,target.hc_mult)!=(5120,40,384,6,129280,4):
            raise ValueError('DSpark target geometry is incompatible')
        return cls()


def inspect_dspark(artifact):
    """Bound existing stat-sealed CPU index to this embedded head namespace."""
    from .deepseek_v41_source_artifact import aliases
    config=DSparkConfig.from_artifact(artifact)
    c=replace(artifact['args'],num_hidden_layers=3,n_routed_experts=128,num_experts_per_tok=3,
        compress_ratios=(0,0,0),kv_source_layer_ids=(),index_source_layer_ids=(),candidate_source_layer_id=-1,
        engram_layer_ids=(),engram_num_embeddings=())
    tensors={}
    original_names={}
    for name,row in artifact['tensors'].items():
        if name.startswith('mtp.'):
            virtual='layers.'+name[4:]
            tensors[virtual]=row
            original_names[virtual]=name
    mapping={alias:name for alias,name in aliases(c).items()if alias.startswith('blk.')and not alias.endswith('exp_probs_b_vl.bias')}
    mapping.update({'main_proj.weight':'layers.0.main_proj.weight','main_norm.weight':'layers.0.main_norm.weight',
        'draft_norm.weight':'layers.2.norm.weight','markov_embed.weight':'layers.2.markov_head.embed.weight',
        'markov_head.weight':'layers.2.markov_head.head.weight','confidence.weight':'layers.2.confidence_head.proj.weight'})
    required=set(mapping.values())
    for stage in range(3):
        for expert in range(128):
            required.update(f'layers.{stage}.ffn.experts.{expert}.{part}.{field}'for part in ('w1','w3','w2')for field in ('weight','scale'))
    if required-tensors.keys():raise ValueError('Pinned Source lacks complete embedded DSpark tensors')
    for stage in range(3):
        for expert in range(128):
            for part in ('w1','w3','w2'):
                shape=(5120,2304)if part=='w2'else(2304,5120)
                p=f'layers.{stage}.ffn.experts.{expert}.{part}.'
                if tensors[p+'weight']['dtype']!='I8'or tuple(tensors[p+'weight']['shape'])!=(shape[0],shape[1]//2):
                    raise ValueError('DSpark original MXFP4 expert geometry changed')
                if tensors[p+'scale']['dtype']!='F8_E8M0'or tuple(tensors[p+'scale']['shape'])!=(shape[0],shape[1]//32):
                    raise ValueError('DSpark original E8M0 expert scales changed')
    dense=sum(tensors[n]['size']+tensors.get(n.removesuffix('weight')+'scale',{}).get('size',0)for n in mapping.values())
    expert=sum(row['size']for name,row in tensors.items()if '.experts.'in name)
    binding={'schema': 'mlx2.v41-embedded-dspark.v1','source_fingerprint': SOURCE_FINGERPRINT,'config': asdict(config),
        'tensor_extents': {n:{k:v for k,v in tensors[n].items()if k in ('file','offset','size','dtype','shape')}for n in sorted(tensors)}}
    fingerprint=digest(binding)
    head_artifact={**artifact,'args':c,'aliases':mapping,'tensors':tensors,
        'identity':{**artifact['identity'],'fingerprint':fingerprint,'weight_sha256':fingerprint},
        'inventory':{'storage_bytes':{'dense':dense,'expert':expert,'engram':0}}}
    return {'config': config,'artifact': head_artifact,'binding': binding,'fingerprint': fingerprint,
        'checkpoint': 'DeepSeek-V4.1-Flash-1017-Source@'+SOURCE_FINGERPRINT,
        'tensor_count': len(tensors),'dense_payload_bytes': dense,'expert_table_bytes': expert,
        'head_namespace': 'mtp.*','expert_precision': 'original_MXFP4_E2M1_E8M0_g32','qualified': False}


def source_dspark_admission_plan(artifact, *, expert_cache_bytes=512 << 20, record=None):
    """Eager AST-head tensors plus one bounded streamed head execution.

    The configured expert cache is an eviction ceiling, not an eager payload.
    Five positions can select at most15 original experts in a single stage;
    its provider window completes before the next stage starts.
    """
    from .deepseek_v41_dspark_artifact import dspark_workspace_bound
    from .deepseek_v41_source_weights import weight_plan
    record = inspect_dspark(artifact) if record is None else record
    bounds = weight_plan(record['artifact'], expert_cache_bytes)
    payload = bounds['dense_payload_bytes']
    prepared = bounds['dense_prepared_bound_bytes']
    staging = bounds['dense_staging_bound_bytes']
    progress = (min(record['config'].expert_count, record['config'].top_k * record['config'].block_size)
                * bounds['minimum_expert_cache_bytes'] + bounds['expert_staging_bound_bytes'])
    workspace = dspark_workspace_bound(artifact['args']) + progress
    return {'schema': 'mlx2.v41-source-dspark-ast-admission.v1', 'binding': record['fingerprint'],
        'payload_bytes': payload, 'prepared_bound_bytes': prepared, 'staging_bound_bytes': staging,
        'workspace_bound_bytes': workspace, 'imminent_head_expert_progress_bytes': progress,
        'reservation_bytes': payload + prepared + staging + workspace,
        'expert_cache_maximum_bytes': expert_cache_bytes, 'expert_cache_maximum_reserved': False,
        'qualification': 'unqualified'}


def _check_ast_head_memory(target_model, admission, *, loaded=()):
    import mlx.core as mx

    from ..memory import host_available_bytes
    from .deepseek_v41_source_auxiliary import (
        auxiliary_memory_requirement,
        check_auxiliary_memory,
    )
    required = auxiliary_memory_requirement(admission, loaded=loaded)
    available = host_available_bytes(host_signals=True)
    if type(available) is not int or available < 0:
        raise MemoryError('Source DSpark AST requires measured available memory')
    stream = getattr(target_model.weights, 'expert_stream', None)
    reclaim = getattr(stream, 'rebalance_headroom', None)
    if callable(reclaim) and getattr(stream, 'adaptive_residency_enabled', False):
        limit = admission.get('admission_limit_bytes')
        protected = required if limit is None else max(required, available + int(mx.get_active_memory()) + required - limit)
        reclaim(available_bytes=available, protected_bytes=protected)
    return check_auxiliary_memory(admission, loaded=loaded,
        active_bytes=int(mx.get_active_memory()), available_bytes=host_available_bytes(host_signals=True))


def load_source_dspark(target_model,*,expert_cache_bytes=512<<20,check_live):
    """Construct the actual head under parent-owned GPU/CPU execution only."""
    if not callable(check_live)or check_live()is not True:raise ValueError('DSpark requires live execution owner')
    from ..memory import host_available_bytes
    from ..runtime.drafters.deepseek_v41_dspark import SourceDSparkBackbone
    from .deepseek_v41_source_auxiliary import stage_auxiliary_plans
    from .deepseek_v41_source_weights import SourceV41Weights
    record=inspect_dspark(target_model.weights.artifact)
    plan=source_dspark_admission_plan(target_model.weights.artifact,
        expert_cache_bytes=expert_cache_bytes,record=record)
    admission=dict(getattr(target_model.weights,'bounds',{}))
    stage_auxiliary_plans(admission,{'dspark_ast':plan})
    target_stream=getattr(target_model.weights,'expert_stream',None)
    old_headroom=getattr(target_stream,'_adaptive_headroom',None)
    old_protected=getattr(target_stream,'_adaptive_protected_bytes',None)
    reserved_headroom=None
    weights=None
    try:
        if type(old_headroom)is int:
            reserved_headroom=max(old_headroom,plan['workspace_bound_bytes'])
            target_stream._adaptive_headroom=reserved_headroom
        admission['before_load']=_check_ast_head_memory(target_model,admission)
        weights=SourceV41Weights(record['artifact'],expert_cache_bytes=expert_cache_bytes,
            expert_read_workers=2,expert_math_policy='reference')
        weights.expert_stream.configure_adaptive_residency(enabled=True,
            available_bytes=lambda:host_available_bytes(host_signals=True),
            headroom_bytes=plan['workspace_bound_bytes'],
            working_set_experts=record['config'].top_k,staging_within_cache=True)
        backbone=SourceDSparkBackbone(target_model,weights,record,check_live=check_live)
        admission['after_load']=_check_ast_head_memory(target_model,admission,loaded=('dspark_ast',))
        backbone.admission=admission
        backbone._target_headroom_restore=(target_stream,old_headroom,reserved_headroom,
            old_protected,getattr(target_stream,'_adaptive_protected_bytes',None))
        return backbone
    except BaseException as error:
        if weights is not None:
            try:
                weights.close()
            except BaseException as cleanup:  # noqa: BLE001 - retain the original failure if retirement also fails
                error.add_note('DSpark AST head cleanup also failed: '+repr(cleanup))
        if reserved_headroom is not None and target_stream._adaptive_headroom==reserved_headroom:
            target_stream._adaptive_headroom=old_headroom
        if type(old_protected)is int:
            target_stream._adaptive_protected_bytes=old_protected
        raise

# The serial embedded B1 route has a separate configuration/owner contract.
def create_source_dspark(
    target, artifact, *, admitted_bytes, num_draft=5, confidence_threshold=0.0, owner_callback=None,
    verify_schedule="fixed", verify_costs=None, draft_cost=None, admission="off",
):
    if type(num_draft) is not int or not 1 <= num_draft <= 5:
        raise ValueError("embedded DSpark num_draft must be1..5")
    from ..runtime.drafters.deepseek_v41_dspark import DeepSeekV41DSparkBackbone
    from .deepseek_v41_dspark_external import DeepSeekV41DSparkDraftModel
    from .deepseek_v41_dspark_weights import SourceDSparkWeights

    weights = SourceDSparkWeights(
        artifact, target.weights, admitted_bytes=admitted_bytes, load=False
    )
    try:
        if owner_callback is not None:
            owner_callback(weights)
        weights.load()
        plan = weights.plan
        backbone = DeepSeekV41DSparkBackbone(
            EmbeddedDSparkConfig.from_artifact(artifact), target, weights, plan["binding"]
        )
        return DeepSeekV41DSparkDraftModel(
            backbone, confidence_threshold=confidence_threshold, verify_schedule=verify_schedule,
            verify_costs=verify_costs, draft_cost=draft_cost, admission=admission,
        ).bind(target)
    except BaseException as error:
        try:
            weights.close()
        except BaseException as cleanup:  # noqa: BLE001 - retain owner and re-raise original construction failure
            error.add_note("DSpark construction owner retained after cleanup fence failure: " + repr(cleanup))
        raise


def configure_source_dspark(
    adapter,
    *,
    artifact=None,
    num_draft=None,
    admitted_bytes=None,
    confidence_threshold=0.0,
):
    from ..contracts import Capability
    from ..sampling_defaults import SamplingDefaults, VendorSampling

    if getattr(adapter, "draft_weights", None) is not None or getattr(adapter, "draft_model", None) is not None:
        raise ValueError("DSpark cannot replace an existing draft owner")
    artifact = adapter.source_artifact if artifact is None else artifact
    count = (
        adapter.kernel_policy.get("num_draft", 5) if num_draft is None else num_draft
    )
    plan = dspark_admission_plan(artifact)
    if admitted_bytes is None:
        admitted_bytes = (adapter.native_admission or {}).get(
            "dspark_reservation_bytes", 0
        )
    if admitted_bytes < plan["reservation_bytes"]:
        raise MemoryError(
            "Source DSpark reservation was not included before target admission"
        )
    draft = create_source_dspark(
        adapter.model,
        artifact,
        admitted_bytes=admitted_bytes,
        num_draft=count,
        confidence_threshold=confidence_threshold,
        owner_callback=lambda owner: setattr(adapter, "draft_weights", owner),
        verify_schedule=adapter.kernel_policy.get("deepseek_dspark_verify_schedule", "fixed"),
        verify_costs=adapter.kernel_policy.get("deepseek_dspark_verify_costs"),
        draft_cost=adapter.kernel_policy.get("deepseek_dspark_draft_cost"),
        admission=adapter.kernel_policy.get("deepseek_dspark_admission", "off"),
    )
    adapter.draft_model = draft
    adapter.draft_weights = draft.backbone.w
    adapter.external_policy = {"draft_model": "embedded_dspark", "num_draft": count}
    adapter.external_record = {
        "kind": draft.kind,
        "identity": plan["binding"],
        "tensor_count": plan["tensor_count"],
        "qualification": "unqualified",
        "admission": plan,
        "target_artifact": adapter.identity["fingerprint"],
    }
    adapter.external_source = {
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "revision": "1bae0e1088993b9b82de08f59de5be5a34e42170",
        "embedded": True,
    }
    adapter.layout += ":embedded-dspark-v1:" + draft.execution_binding
    adapter.profile += (
        ":embedded-dspark-"
        + hashlib.sha256(draft.execution_binding.encode()).hexdigest()[:16]
    )
    adapter.default_route = "external_draft"
    adapter.descriptor = replace(
        adapter.descriptor,
        capabilities=adapter.descriptor.capabilities | {Capability.EXTERNAL_DRAFT},
        cache_layout=adapter.layout,
        metadata={
            **adapter.descriptor.metadata,
            "qualification": "unqualified",
            "draft": "embedded DSpark greedy B1",
        },
    )
    adapter.sampling_defaults = VendorSampling.single(
        SamplingDefaults(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            source="unqualified embedded V4.1 DSpark greedy-only route",
        ),
        model="DeepSeek-V4.1-Flash-1017+DSpark",
    )
    return plan
