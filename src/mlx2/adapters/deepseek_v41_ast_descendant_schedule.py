# SPDX-License-Identifier: MIT
"""Small CPU metadata plan for descendant-oriented KV distribution.

Consumer masks describe eligibility, not dtype conversion, softmax masking or
GPU memory transactions. No array data is read and no attention path selects it.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RowRun:
    plane: str
    source: str
    first: int
    count: int
    consumers: int


@dataclass(frozen=True)
class DescendantSchedule:
    runs: tuple
    physical_rows: tuple
    logical_row_uses: tuple
    node_count: int

    def receipt(self, *, head_dim=512, index_head_dim=128):
        if any(type(v)is not int or v<1 for v in(head_dim,index_head_dim)):
            raise ValueError('Invalid descendant schedule dimensions')
        dims={'local':head_dim,'pool':head_dim,'index':index_head_dim}
        physical=dict(self.physical_rows);logical=dict(self.logical_row_uses)
        return dict(node_count=self.node_count,unique_physical_rows=physical,
            logical_row_uses=logical,fp32_unique_bytes={p:n*dims[p]*4 for p,n in physical.items()},
            fp32_logical_bytes={p:n*dims[p]*4 for p,n in logical.items()},
            selected_mask_intersection='Required on GPU for pool/index; ancestry eligibility only',
            dtype_conversion_saving_claim=False,hardware_read_saving_claim=False,
            qualified=False,selected=False)


def _paths(parents):
    if not 1<=len(parents)<=16:raise ValueError('Descendant schedule needs1..16 nodes')
    paths=[]
    for node,parent in enumerate(parents):
        if type(parent)is not int or (parent!=-1 if node==0 else not 0<=parent<node):
            raise ValueError('Descendant schedule requires one topologically ordered root')
        paths.append((node,)if parent<0 else paths[parent]+(node,))
    return tuple(paths)


def _runs(plane,rows,prefix_count,*,common_prefix=False):
    users={}
    for node,visible in enumerate(rows):
        for row in(visible[prefix_count:]if common_prefix else visible):users[row]=users.get(row,0)|(1<<node)
    runs=([RowRun(plane,'prefix',0,prefix_count,(1<<len(rows))-1)]if common_prefix and prefix_count else[])
    for row,mask in sorted(users.items()):
        source='prefix'if row<prefix_count else'delta'
        if runs and (runs[-1].source,runs[-1].consumers,runs[-1].first+runs[-1].count)==(source,mask,row):
            previous=runs.pop();runs.append(RowRun(plane,source,previous.first,previous.count+1,mask))
        else:runs.append(RowRun(plane,source,row,1,mask))
    return runs,len(users)+(prefix_count if common_prefix else 0),sum(map(len,rows))


def build_schedule(parents,local_rows,prefix_local_count,*,logical_rows=None,prefix_pool_count=0,completed_nodes=()):
    """Plan actual local-window eligibility and optional compressed ancestry.

    Pool/index rows remain candidates; this never reads selected GPU IDs.
    Input row tuples use the existing prefix+delta physical table numbering.
    """
    parents=tuple(parents);paths=_paths(parents);local_rows=tuple(tuple(v)for v in local_rows)
    if (len(local_rows)!=len(paths)or type(prefix_local_count)is not int or prefix_local_count<0
            or type(prefix_pool_count)is not int or prefix_pool_count<0):
        raise ValueError('Descendant row inventory mismatch')
    for node,rows in enumerate(local_rows):
        if any(type(v)is not int for v in rows)or tuple(sorted(set(rows)))!=rows:
            raise ValueError('Local rows must be unique chronological physical indices')
        prefix=tuple(v for v in rows if v<prefix_local_count)
        delta=tuple(v-prefix_local_count for v in rows if v>=prefix_local_count)
        chronological=tuple(range(prefix_local_count))+tuple(prefix_local_count+v for v in paths[node])
        if (any(v<0 for v in prefix)or not delta or delta!=paths[node][-len(delta):]
                or rows!=chronological[-len(rows):]):
            raise ValueError('Local rows contain a sibling/future row or omit the query frontier')
    runs,physical,logical=_runs('local',local_rows,prefix_local_count)
    counts=[('local',physical)];uses=[('local',logical)]
    if logical_rows is not None:
        completed_nodes=tuple(completed_nodes);logical_rows=tuple(tuple(v)for v in logical_rows)
        if (tuple(sorted(set(completed_nodes)))!=completed_nodes or any(type(v)is not int or not 0<=v<len(paths)for v in completed_nodes)
                or len(logical_rows)!=len(paths)):
            raise ValueError('Compressed completion inventory mismatch')
        delta={node:prefix_pool_count+i for i,node in enumerate(completed_nodes)}
        common_prefix=tuple(range(prefix_pool_count))
        for node,rows in enumerate(logical_rows):
            expected=common_prefix+tuple(delta[v]for v in paths[node]if v in delta)
            if rows!=expected:raise ValueError('Compressed rows contain missing, sibling or future ancestry')
        for plane in('pool','index'):
            more,physical,logical=_runs(plane,logical_rows,prefix_pool_count,common_prefix=True)
            runs.extend(more);counts.append((plane,physical));uses.append((plane,logical))
    elif prefix_pool_count or completed_nodes:
        raise ValueError('Compressed metadata supplied without logical row inventory')
    return DescendantSchedule(tuple(runs),tuple(counts),tuple(uses),len(paths))


def descriptor_stamp(segment):
    return (id(segment),id(segment.base),
            tuple(id(getattr(segment,name,None))for name in('kv','pooled_delta','index_delta','local_table','pool_table','index_table')),
            tuple(tuple(v)for v in segment.local_rows),
            tuple(tuple(v)for v in segment.logical_rows),tuple(segment.completed_nodes),
            segment.prefix_local_count,segment.prefix_count)


@dataclass(frozen=True)
class BoundSchedule:
    schedule: DescendantSchedule
    segment: object
    stamp: tuple
    epoch_key: tuple
    check_live: object

    def validate(self,epoch_key):
        if epoch_key!=self.epoch_key:raise ValueError('Descendant schedule belongs to another epoch')
        if self.check_live()is False:raise ValueError('Descendant schedule owner is no longer live')
        self.segment.assert_current()
        if descriptor_stamp(self.segment)!=self.stamp:raise ValueError('Descendant descriptor row map changed')
        return self.schedule


def bind_schedule(segment,parents,*,epoch_key,check_live):
    """Bind metadata to the actual caller's COW/transaction capability.

    Future consumers call validate at frame boundaries; no fence is introduced.
    The caller retains descriptor roots through its existing completion fence.
    """
    if not isinstance(epoch_key,tuple)or not epoch_key or not callable(check_live):
        raise ValueError('Descendant schedule requires an epoch key and live capability')
    try:hash(epoch_key)
    except TypeError as error:raise ValueError('Descendant epoch key must be immutable')from error
    if check_live()is False:raise ValueError('Descendant schedule owner is no longer live')
    segment.assert_current()
    source=segment.base.source
    schedule=build_schedule(parents,segment.local_rows,segment.prefix_local_count,
        logical_rows=segment.logical_rows if source else None,
        prefix_pool_count=segment.prefix_count if source else 0,
        completed_nodes=segment.completed_nodes if source else())
    return BoundSchedule(schedule,segment,descriptor_stamp(segment),epoch_key,check_live)
