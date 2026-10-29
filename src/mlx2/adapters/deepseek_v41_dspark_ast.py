# SPDX-License-Identifier: MIT
"""Bind embedded V4.1 DSpark to the existing fixed3x5 AST scheduler.

The actual target-tree provider requires an offline source/artifact/device
numerical oracle. No serial target fallback is permitted inside AST. Ordinary
terminal work keeps the existing exact snapshot/replay owner.
"""
import hashlib
import json
from types import MethodType, SimpleNamespace

from ..runtime.cache_operations import CacheOperationsContract, CachePlaneDeclaration
from ..runtime.models.cache import register_cache_class


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class SourceDSparkDraftCache:
    cache_operations_contract=CacheOperationsContract(family='deepseek_v41_dspark',topology='checkpointed_hybrid',
        state_revision='v41-source-dspark-support-v1',cache_layout='committed-support-b1',
        numerics='attention_input_hc_mean_bf16',
        planes=(CachePlaneDeclaration('recurrent_state',('state',),('meta_state',)),),
        exact_prefix=True,restores_at_checkpoints=True,rollback=None)

    def __init__(self,identity,target_identity):
        from ..runtime.drafters.deepseek_v4_dspark import DSparkHistory
        self.history=DSparkHistory(identity,3,128);self.target_identity=target_identity

    @property
    def offset(self):return self.history.next_position

    @property
    def state(self):return(self.history.main_history,*self.history.stage_kv)

    @property
    def meta_state(self):
        h=self.history
        return('v41-source-dspark-support-v1',h.identity,self.target_identity,str(h.next_position),str(h.start_position))

    @classmethod
    def from_state(cls,state,meta_state):
        from ..runtime.drafters.deepseek_v4_dspark import DSparkSnapshot
        if len(meta_state)!=5 or meta_state[0]!='v41-source-dspark-support-v1'or len(state)!=4:
            raise ValueError('V4.1 DSpark cache revision/inventory differs')
        if any(type(x)is not str or not x.isascii()or not x.isdecimal()for x in meta_state[3:]):
            raise ValueError('V4.1 DSpark cache positions must be canonical integers')
        result=cls(meta_state[1],meta_state[2])
        result.history.restore(DSparkSnapshot(meta_state[1],int(meta_state[3]),int(meta_state[4]),state[0],tuple(state[1:])))
        return result

    @property
    def nbytes(self):return sum(v.nbytes for v in self.state if v is not None)

    def is_trimmable(self):return False


register_cache_class(SourceDSparkDraftCache)


class SourceDSparkDraftModel:
    """Existing complete-continuation protocol over one embedded block."""
    receipt_kind='external_deepseek_v41_source_dspark_ast'
    proposal_distribution='deterministic_point_mass'
    continuation_protocol='dspark'
    supports_sampling=False
    supports_logits_processors=True
    supports_ordered_alternatives=True
    maximum_proposal_length=5
    proposal_rows=5
    qualified=False

    def __init__(self,backbone):
        self.backbone=backbone
        c=backbone.config
        self.config=SimpleNamespace(target_layer_ids=c.target_layers,feature_width=3*c.hidden_size,
            hidden_size=c.hidden_size,vocab_size=c.vocab_size,block_size=5,num_hidden_layers=3,
            intermediate_size=backbone.model.args.moe_intermediate_size,num_key_value_heads=1,head_dim=512)
        self.execution_binding=_hash([backbone.identity,'AST3x5-one16rowtile-v1'])
        self.adaptive_confidence_features=[[]]
        self._last_paths=None

    @property
    def receipt_settings(self):
        return dict(self.backbone.receipt(),binding=self.execution_binding,packed_pattern=[5,5,5],
            target_tile_rows=16,sampling='greedy_only',confidence_use='telemetry_not_target_acceptance')

    def make_cache(self):return[SourceDSparkDraftCache(self.backbone.identity,self.backbone.target.identity)]

    def batch_caches(self,rows):
        if len(rows)!=1:raise ValueError('DSpark AST3x5 owns exactly one request')
        return rows[0]

    def append_context(self,hidden,cache):
        if len(cache)!=1 or cache[0].history.identity!=self.backbone.identity or cache[0].target_identity!=self.backbone.target.identity:
            raise ValueError('DSpark target/head cache binding changed')
        self.backbone.record_features(hidden,cache[0].offset,cache[0].history)

    def _paths(self,anchors,hidden,cache,*,processors=None,histories=None):
        from ..runtime.speculative_topology import processors_branch_safe
        if len(anchors)!=1:raise ValueError('DSpark AST requires greedy B1')
        active=[]if processors is None else processors[0]
        if not processors_branch_safe(active):raise ValueError('DSpark AST accepts only stateless branch-safe forbidden-token processors')
        length=len(histories[0])if histories else cache[0].offset
        def forbidden(depth):
            return tuple(sorted({int(t)for processor in active for t in processor.forbidden_token_ids_at_length(length+1+depth)}))
        self.append_context(hidden,cache)
        paths,confidence=self.backbone.propose_paths(int(anchors[0]),cache[0].history,forbidden_at_depth=forbidden if active else None)
        self._last_paths=paths
        self.adaptive_confidence_features=[list(confidence[0])]
        return paths,confidence

    def draft_distributions(self,anchors,hidden,cache,proposal_length,rngs,temperatures,
                            *,logits_processors=None,processor_histories=None,**kwargs):
        from ..runtime.proposal_providers import PointMassLaw
        if len(temperatures)!=1 or float(temperatures[0])!=0 or kwargs or not 0<=proposal_length<=5:
            raise ValueError('DSpark AST draft geometry/sampling is unsupported')
        if proposal_length==0:
            self.append_context(hidden,cache);return[[]],[[]]
        paths,_=self._paths(anchors,hidden,cache,processors=logits_processors,histories=processor_histories)
        row=list(paths[0][:proposal_length])
        return[row],[[PointMassLaw(token)for token in row]]

    def draft_packed_paths(self,anchors,hidden,cache,pattern,*,logits_processors=None,
                           processor_histories=None,context_tokens=None,strict=True):
        from .proposal_path_sources import SourceContinuation
        if tuple(pattern)!=(5,5,5):raise ValueError('V4.1 DSpark requires exactly3x5 proposals')
        paths,confidence=self._paths(anchors,hidden,cache,processors=logits_processors,histories=processor_histories)
        return[[SourceContinuation(path,tuple(scores),float(-rank),'dspark_distinct_root_rank_proxy')
            for rank,(path,scores)in enumerate(zip(paths,confidence))]]

    def propose_complete_paths(self,context,limit):
        from .proposal_path_sources import SourceContinuation
        if self._last_paths is None or context.depth!=5:raise ValueError('DSpark complete-path context is unavailable')
        return[SourceContinuation(path,(None,)*5,float(-rank),'dspark_distinct_root_rank_proxy')
            for rank,path in enumerate(self._last_paths[:limit])]

    def ordered_alternative_proposals(self,primary_tokens,*,branch_count,strategy,shared_prefix_tokens):
        if branch_count!=3 or strategy not in ('confidence_tree','distinct_root')or self._last_paths is None:
            raise ValueError('DSpark AST requires its fixed3x5 distinct-root slate')
        if len(primary_tokens)!=1 or tuple(primary_tokens[0])!=self._last_paths[0]:
            raise ValueError('DSpark AST primary differs from its private head slate')
        return[[list(path)for path in self._last_paths]]

    def extra_memory_reservation(self,*args,**kwargs):return 64<<20


def _tree_only(self,*args,**kwargs):
    raise ValueError('V4.1 DSpark AST requires admitted16row tree provider; no serial AST fallback')


def _copy_source_cache(cache,*,descriptor_aliases=False):
    """Fork the exact six-plane Source frontier through its owned state law.

    The local band, partial groups, shared pools/index and Engram tail are
    independently copied by from_state. Source has no generic KV values
    plane and must never enter the paired keys/values truncation heuristic.
    """
    from ..runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
    if not cache or any(type(item)is not DeepSeekV41LayerCache for item in cache):
        raise ValueError('Source AST private fork requires its six-plane cache')
    if any(item.pending for item in cache):
        raise ValueError('Source AST private fork requires a committed six-plane frontier')
    if descriptor_aliases:
        owner=getattr(cache,'cow_owner',None)
        if owner is not None:
            if getattr(cache,'closed',False):raise ValueError('Source AST APCv2 branch is closed')
            return owner.branch_live(tuple(cache),None,lineage_generation=cache.cow_generation,
                metadata=getattr(cache,'cow_metadata',None))
        from ..runtime.cow_cache import snapshot_prompt_cache_descriptors
        return snapshot_prompt_cache_descriptors(cache)[0]
    return [DeepSeekV41LayerCache.from_state(item.state,item.meta_state,xp=item.xp)for item in cache]


_AST_TARGET_ATTRIBUTES = (
    'progressive_external_verify_protocol', 'progressive_external_target_rows',
    'progressive_external_exact_tile_widths', 'progressive_external_exact_terminal_input_widths',
    'progressive_external_exact_max_lanes', 'progressive_external_private_request_dispatch',
    'progressive_external_compact_exact_rows', 'progressive_external_alternative_row_capacity',
)
_ABSENT = object()


def _attribute_snapshot(owner, names):
    # Preserve the instance dictionary, so deleting an installed override also
    # reveals an original class method/property rather than shadowing it.
    return {name: vars(owner).get(name, _ABSENT) for name in names}


def _restore_attributes(owner, snapshot):
    for name, value in snapshot.items():
        if value is _ABSENT:
            if name in vars(owner):
                delattr(owner, name)
        else:
            setattr(owner, name, value)


def install_ast_target_contract(model):
    from ..runtime.progressive_external_verify import (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    )
    model.progressive_external_verify_protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION
    model.progressive_external_target_rows=MethodType(_tree_only,model)
    model.progressive_external_exact_tile_widths=(1,2)
    model.progressive_external_exact_terminal_input_widths=(2,3)
    model.progressive_external_exact_max_lanes=1
    model.progressive_external_private_request_dispatch=True
    model.progressive_external_compact_exact_rows=False
    model.progressive_external_alternative_row_capacity=16


class SourceDSparkASTBinding:
    """Cold adapter-owned serving glue, with explicit offline oracle gate."""
    def __init__(self,adapter,*,check_live,expert_cache_bytes=512<<20,oracle_path=None,expert_law='ordinary_m16_v1',forward_law='reference'):
        from ..runtime.progressive_external_verify import (
            progressive_target_protocol_binding,
        )
        from .deepseek_v41_ast_target import SourceASTTargetTreeExecutor
        from .deepseek_v41_dspark import load_source_dspark
        from .proposal_path_sources import build_continuation_drafter
        self.adapter=adapter;self.model=adapter.model;self.oracle_path=oracle_path
        self.bank_manager=None
        self.backbone=None
        self._closed=False
        if getattr(adapter,'_dspark_ast_binding',None)is not None:raise ValueError('DSpark AST is already bound')
        if getattr(adapter,'draft_model',None)is not None:raise ValueError('DSpark AST cannot replace another live draft route')
        if (forward_law!='reference' and
                getattr(self.model,'_mlx2_source_ast_target_laws',None)!=(expert_law,forward_law)):
            raise ValueError('Source batched AST target laws must be bound at load; reload with oracle preparation policy')
        if (expert_law=='source_ast_hybrid_rows_v1'
                and getattr(self.model.weights,'source_prefill_partition_policy','adaptive_v1')!='canonical_m1_v1'):
            raise ValueError('Hybrid Source AST binding requires canonical_m1_v1 resource hooks')
        if (getattr(self.model.weights,'source_resident_bank_session',None)is not None
                and forward_law=='reference'):
            raise ValueError('Close the diagnostic B1 bank session before binding AST16row target')
        self._model_contract = _attribute_snapshot(
            self.model, (*_AST_TARGET_ATTRIBUTES, '_source_ast_bank_manager')
        )
        self._weight_contract = _attribute_snapshot(self.model.weights, ('_source_ast_bank_manager',))
        try:
            self.backbone=load_source_dspark(self.model,expert_cache_bytes=expert_cache_bytes,check_live=check_live)
            self.backend=SourceDSparkDraftModel(self.backbone)
            if expert_law=='source_ast_hybrid_rows_v1':
                from .deepseek_v41_ast_bank_manager import SourceASTBankManager
                if (getattr(self.model,'_source_ast_bank_manager',None)is not None
                        or getattr(self.model.weights,'_source_ast_bank_manager',None)is not None):
                    raise ValueError('Hybrid AST bank manager already has another binding owner')
                self.bank_manager=SourceASTBankManager(self.model,ownercheck=check_live)
                self.model._source_ast_bank_manager=self.bank_manager
                self.model.weights._source_ast_bank_manager=self.bank_manager
            install_ast_target_contract(self.model)
            self.target_artifact=adapter.identity['fingerprint']
            self.target_binding=progressive_target_protocol_binding(self.model,'exact')
            self.provider=SourceASTTargetTreeExecutor(self.model,expert_law=expert_law,forward_law=forward_law)
            target_revision=_hash([self.target_artifact,self.model.identity])
            draft_revision=_hash(self.backbone.identity)
            tokenizer_revision=_hash(getattr(adapter,'tokenizer_receipt',{'target':self.target_artifact}))
            self.policy={'sources': ['external'],'limit': 3,'ast_planner': True,'packed_pattern': [5,5,5]}
            session_revision=_hash([target_revision,draft_revision,tokenizer_revision,self.policy])
            self.draft=build_continuation_drafter(self.model,self.backend,self.policy,target_revision=target_revision,
                draft_revision=draft_revision,tokenizer_revision=tokenizer_revision,session_revision=session_revision)
            self.binding=_hash([self.backend.execution_binding,session_revision])
            if oracle_path is not None:self.provider.with_ast_oracle(oracle_path,artifact=self.target_artifact,
                capture_layers=(37,38,39),target_binding=self.target_binding)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 - quarantine retains failed-construction resources
                from .deepseek_v41_mlx_weights import _quarantine_upload
                # A constructor cannot return an uncertain owner. Keep every
                # borrowed target/head/manager reachable for an explicit retry.
                _quarantine_upload(self)
                error.add_note('DSpark AST binding cleanup also failed: '+repr(cleanup))
            finally:
                # A failed public configure never publishes an AST contract,
                # including when its cleanup fence itself needs a retry.
                self._restore_target_contract()
            raise

    def oracle_template(self):
        """Binding template only. The caller must supply actual numerical proof."""
        from .ast_execution_profile import build_execution_profile
        profile=build_execution_profile(self.model,self.draft,binding=self.binding,tile_cap=2,
            max_lanes=1,num_draft=5,alternative_provider=None)
        return {'schema': 'mlx2.deepseek-v41-source-ast-oracle.v1',
            'binding': self.provider.oracle_binding(artifact=self.target_artifact,capture_layers=(37,38,39),
                target_binding=self.target_binding,profile=profile),'input_widths': [16],
            'checks': {key:False for key in self.provider.oracle_checks},
            'receipt_sha256': None,'numerical_classification': 'unclassified'}

    def create_external_batch(self,**kwargs):
        from ..runtime.external_speculative import ExternalDraftBatchGenerator
        from .deepseek_v4_external import (
            DeepSeekV4VerifyRows,
            DeepSeekV4VerifyTransaction,
        )
        if self._closed:raise ValueError('DSpark AST binding is closed')
        if self.oracle_path is None:raise ValueError('DSpark AST serving needs an actual16row exact numerical oracle')
        fixed={'completion_batch_size': 1,'num_draft': 5,'minimum_draft_proposals': 5,'continuation_pool': self.policy,
            'continuation_pool_ast_bridge': True,'progressive_verification_tile': 2,
            'progressive_verification_tile_policy': None,'progressive_alternative_branches': 3,
            'progressive_alternative_strategy': 'confidence_tree','progressive_alternative_depth_policy': 'full',
            'ast_topology': '3x5','cost_per_commit_handoff': False,'ast_target_tree_executor_provider': self.provider}
        for key,value in fixed.items():
            if key in kwargs and kwargs[key]!=value:raise ValueError('DSpark AST fixed3x5 option differs: '+key)
            kwargs[key]=value
        descriptor_aliases=self.provider.forward_law!='reference'
        class OwnedSourceAST(ExternalDraftBatchGenerator):
            @staticmethod
            def _copy_reference_cache(cache,*,preserve_capacity=False):
                return _copy_source_cache(cache)
            def _copy_ast_private_cache(self,cache):
                result=_copy_source_cache(cache,descriptor_aliases=descriptor_aliases)
                key='external_source_ast_six_plane_private_forks'
                self.scheduler_stats[key]=self.scheduler_stats.get(key,0)+1
                return result
            def _target_owner(self,rows):return DeepSeekV4VerifyRows(self.model,rows,self.layers)
            def _cohort_target_forward(self,inputs,transaction,cohort,*,site,taps=True):
                if not isinstance(transaction,DeepSeekV4VerifyTransaction):raise TypeError('Foreign Source ordinary owner')
                return transaction.forward(inputs,taps=taps)
        return OwnedSourceAST(self.model,draft_model=self.draft,binding=self.binding,
            target_artifact_fingerprint=self.target_artifact,**kwargs)

    def receipt(self):
        return {'schema': 'mlx2.v41-source-dspark-ast-binding.v1','binding': self.binding,
            'head': self.backbone.receipt(),'target': self.provider.receipt(),'packed_pattern': [5,5,5],
            'target_tile_rows': 16,'defaulted': False,'qualified': False,
            'target_laws_bound_at_load': getattr(self.model,'_mlx2_source_ast_target_laws',None),
            'kv_preparation_policy': getattr(self.model,'source_ast_kv_preparation_policy','reference'),
            'serving_oracle_supplied': self.oracle_path is not None,
            'hybrid_bank_manager': None if self.bank_manager is None else self.bank_manager.receipt()}

    def close(self):
        if not self._closed:
            if self.bank_manager is not None:
                self.bank_manager.close()
            if self.backbone is not None:
                self.backbone.close()
            self._restore_target_contract()
            self._closed=True
            from .deepseek_v41_mlx_weights import _release_upload
            _release_upload(self)

    def _restore_target_contract(self):
        _restore_attributes(self.model, self._model_contract)
        _restore_attributes(self.model.weights, self._weight_contract)
