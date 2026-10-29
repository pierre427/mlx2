# SPDX-License-Identifier: MIT
"""Private Source layer-checkpoint addresses bound to APCv2 parent segments.

Metadata only: these checkpoints are NOT added to the parent APC manifest and
carry no exact publication admission. The owning transaction retains its lease.
"""
from dataclasses import dataclass

from ..runtime.cache_planes import CacheLayerSegment, CachePlaneFingerprint, CachePlaneKind, CacheSegmentKey

PROFILE='source_ast_native_batched_segments_v1'
M1_PROFILE='source_ast_m1_batch_axis_segments_candidate_v1'
KNOWN_PROFILES=(PROFILE,M1_PROFILE)


def _validate_profile(profile):
    if type(profile)is not str or profile not in KNOWN_PROFILES:
        raise ValueError('Source AST segment manifest has unknown numerical profile')
    return profile


@dataclass(frozen=True)
class SourceLayerCheckpointManifest:
    layer_index: int
    lineage_id: str
    generation: int
    source_revision_token: str
    parent_segments: tuple
    prefix: CacheLayerSegment
    checkpoints: tuple
    numerical_profile: str = PROFILE

    def __post_init__(self):
        _validate_profile(self.numerical_profile)
        if any(dict(item.fingerprint.identity).get('numerical_profile')!=self.numerical_profile
               for item in(self.prefix,*self.checkpoints)):
            raise ValueError('Source AST layer fingerprints have a foreign numerical profile')

    def assert_parent_current(self,owner):
        if (owner.lineage_id!=self.lineage_id or
                not owner.branchable(expected_generation=self.generation)):
            raise ValueError('Source AST parent lineage/generation/segment invalidated')
        current={item.key:item.fingerprint for item in owner.segment_manifest.segments}
        if any(current.get(item.key)!=item.fingerprint for item in self.parent_segments):
            raise ValueError('Source AST parent layer checkpoint fingerprint changed')

    def receipt(self):
        def entry(item):
            key=item.key
            return dict(kind=key.kind.value,layer_index=key.layer_index,segment_index=key.segment_index,
                token_start=key.token_start,token_stop=key.token_stop,role=key.role,
                logical_bytes=item.logical_bytes,fingerprint=item.fingerprint.digest)
        return dict(schema='mlx2.source-ast-private-layer-segments.v1',layer_index=self.layer_index,
            lineage_id=self.lineage_id,generation=self.generation,source_revision_token=self.source_revision_token,
            numerical_profile=self.numerical_profile,parent_segments=[entry(item)for item in self.parent_segments],
            prefix=entry(self.prefix),private_checkpoints=[entry(item)for item in self.checkpoints],
            published_to_APCv2=False,exact_publication_admitted=False,
            required_gate='Owning transaction source-bound exact numerical admission',
            logical_bytes_scope='Visible state payload; capacity/scratch accounting remains separate')


def build_layer_manifest(descriptor,layer_index,*,lineage_id,generation,source_revision_token,parent_segments=(),numerical_profile=PROFILE):
    """Describe atomic six-plane checkpoints without materializing any node.

    Source layers depend on the complete compressed/index prefix, even though
    the local attention plane retains only a sliding window. All16 checkpoints
    are private alternatives within this lineage; they are not16 APC entries.
    """
    if (type(layer_index)is not int or layer_index<0 or type(generation)is not int or generation<0
            or type(lineage_id)is not str or not lineage_id or type(source_revision_token)is not str
            or not source_revision_token):
        raise ValueError('Source AST segment manifest requires explicit layer/lineage/generation/revision')
    numerical_profile=_validate_profile(numerical_profile)
    descriptor.assert_current();base=descriptor.base
    if getattr(base,'ast_private_profile',None)not in(None,numerical_profile):
        raise ValueError('Source AST segment manifest has foreign private numerical profile')
    parents=tuple(item for item in parent_segments if item.key.layer_index==layer_index
                  and item.key.kind==CachePlaneKind.ATTENTION_RING)
    def segment(index,start,stop,role,nbytes):
        key=CacheSegmentKey(CachePlaneKind.ATTENTION_RING,layer_index,index,start,stop,role)
        fingerprint=CachePlaneFingerprint.from_fields(key.kind,schema_version=3,
            lineage=lineage_id,generation=generation,source_revision=source_revision_token,
            numerical_profile=numerical_profile,cache_identity=base.identity,ratio=base.ratio,
            source=base.source,head_dim=base.head_dim,index_head_dim=base.index_head_dim,
            window_size=base.window_size,history=base.history,compressed_vocab=base.compressed_vocab,
            parent_fingerprints=tuple(item.fingerprint.digest for item in parents),
            layer=layer_index,segment=index,role=role,token_start=start,token_stop=stop)
        return CacheLayerSegment(key,fingerprint,int(nbytes))
    start=0 if base.source else max(0,base.offset-base.window_size)
    prefix=segment(0,start,base.offset,'source_prefix_checkpoint'if base.source else'local_prefix_checkpoint',
                   sum(0 if value is None else value.nbytes for value in base.state))
    checkpoints=[]
    for node in range(16):
        stop=descriptor.offset(node);start=0 if base.source else max(0,stop-base.window_size)
        nbytes=len(descriptor.local_rows[node])*base.head_dim*4
        if base.source:
            nbytes+=len(descriptor.logical_rows[node])*(base.head_dim+base.index_head_dim)*4
            nbytes+=descriptor.buf_kv[node].nbytes+descriptor.buf_gate[node].nbytes
        if base.history:nbytes+=12
        checkpoints.append(segment(node+1,start,stop,f'private_ast_node_checkpoint:{node}',nbytes))
    return SourceLayerCheckpointManifest(layer_index,lineage_id,generation,source_revision_token,
                                         parents,prefix,tuple(checkpoints),numerical_profile)


def checkpoint_manifest(transaction,path):
    """Return JSON-safe selected-path addresses, never publish or materialize.

    Require actual APCv2 layer keys; plain private snapshots cannot be labelled
    APC segment lineage. The transaction's revision/layout/path and parent
    generation authority are checked again at this metadata boundary.
    """
    numerical_profile=_validate_profile(transaction.profile)
    path=transaction._validate(path)
    owner=transaction.cow_owner
    if owner is None:
        raise ValueError('Source AST checkpoint manifest needs an actual APCv2 parent owner')
    parents=owner.segment_manifest.segments
    layers=[]
    for index,descriptor in enumerate(transaction.segments):
        if descriptor is None:
            raise ValueError('Source AST layer checkpoint is incomplete')
        manifest=build_layer_manifest(descriptor,index,lineage_id=owner.lineage_id,
            generation=transaction.cow_generation,source_revision_token=transaction.provider_revision_token,
            parent_segments=parents,numerical_profile=numerical_profile)
        if not manifest.parent_segments:
            raise ValueError('Source AST parent has no APCv2 layer-segment address for this layer')
        manifest.assert_parent_current(owner)
        record=manifest.receipt()
        record['exact_publication_admitted']=transaction._exact_publication_admitted
        selected=record.pop('private_checkpoints')
        record['selected_checkpoint']=record['prefix']if not path else selected[path[-1]]
        layers.append(record)
    return dict(schema='mlx2.source-ast-private-checkpoint-manifest.v1',path=list(path),
        selected_node=None if not path else path[-1],lineage_id=owner.lineage_id,
        generation=transaction.cow_generation,source_revision_token=transaction.provider_revision_token,
        numerical_profile=numerical_profile,apcv2_lineage_bound=True,layers=layers,
        published_to_APCv2=False,exact_publication_admitted=transaction._exact_publication_admitted,
        node_materializations=0,
        scope='Private metadata bound to existing APCv2 parent keys; no insertion, donation or numerical admission')
