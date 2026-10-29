# SPDX-License-Identifier: MIT
"""Pure CPU adaptive Source memory advice; cache maxima are not reservations.

Low/unknown availability selects smaller streaming work rather than refusing
because the maximum cache or an entire expert layer will not fit. This module
never allocates, evicts, changes precision, or grants ownership. Resizers must
drain/reclaim and remeasure before relying on prospective reclaimed capacity.
"""
from dataclasses import dataclass

ORIGINAL_EXPERT_BYTES=18_800_640
Q1_EXPERT_BYTES=6_635_520
GIB=1<<30


def _bytes(value,name,*,optional=False,positive=False):
    if optional and value is None:return value
    if type(value)is not int or value<(1 if positive else 0):
        raise ValueError('Source memory '+name+' must be '+('positive'if positive else'nonnegative')+' integer bytes')
    return value


@dataclass(frozen=True)
class SourceMemorySnapshot:
    physical_bytes: int | None
    available_bytes: int | None = None
    resident_expert_bytes: int = 0
    pinned_expert_bytes: int = 0
    max_single_allocation_bytes: int | None = None
    allocation_failed_bytes: int = 0

    def __post_init__(self):
        _bytes(self.physical_bytes,'physical',optional=True,positive=True)
        _bytes(self.available_bytes,'available',optional=True)
        for name in ('resident_expert_bytes','pinned_expert_bytes','allocation_failed_bytes'):
            _bytes(getattr(self,name),name)
        _bytes(self.max_single_allocation_bytes,'single allocation limit',optional=True,positive=True)
        if self.pinned_expert_bytes>self.resident_expert_bytes:
            raise ValueError('Source memory pinned experts exceed measured resident experts')


@dataclass(frozen=True)
class SourceMemoryNeed:
    cache_ceiling_bytes: int = 75_000_000_000
    dense_estimated_bytes: int = 0
    dense_resident_bytes: int = 0
    dense_next_allocation_bytes: int = 0
    expert_payload_bytes: tuple = (ORIGINAL_EXPERT_BYTES,)*6
    expert_staging_bytes: int = ORIGINAL_EXPERT_BYTES+(16<<20)
    decode_workspace_bytes: int = 32<<20
    kv_growth_bytes: int = 0
    phase: str = 'decode'
    requested_prefill_rows: int = 1
    prefill_fixed_bytes: int = 32<<20
    prefill_per_row_bytes: int = 2<<20
    largest_workspace_allocation_bytes: int = 0
    route_working_set_known: bool | None = None

    def __post_init__(self):
        for name in ('cache_ceiling_bytes','dense_estimated_bytes','dense_resident_bytes',
                     'dense_next_allocation_bytes','expert_staging_bytes','decode_workspace_bytes',
                     'kv_growth_bytes','prefill_fixed_bytes','prefill_per_row_bytes',
                     'largest_workspace_allocation_bytes'):
            _bytes(getattr(self,name),name)
        if (type(self.expert_payload_bytes)is not tuple or not self.expert_payload_bytes
                or len(self.expert_payload_bytes)>384):
            raise ValueError('Source memory requires actual-format imminent expert payloads')
        for size in self.expert_payload_bytes:_bytes(size,'expert payload',positive=True)
        if self.phase not in ('decode','prefill') or (self.route_working_set_known is not None
                and type(self.route_working_set_known)is not bool):
            raise ValueError('Source memory phase/route knowledge is invalid')
        if type(self.requested_prefill_rows)is not int or not 1<=self.requested_prefill_rows<=1024:
            raise ValueError('Source memory prefill rows must be1..1024')


def plan_source_memory(snapshot,need):
    """Return bounded soft targets and an execution strategy, not admission.

    target_cache_bytes includes payload plus staging, matching Source's store
    budget convention. Pinned experts can temporarily exceed a soft target;
    callers retain them through completion rather than freeing live storage.
    dense_resident_bytes is already reflected in host availability and is not
    subtracted twice. The whole dense estimate is never a startup requirement.
    """
    if not isinstance(snapshot,SourceMemorySnapshot)or not isinstance(need,SourceMemoryNeed):
        raise ValueError('Source memory planner requires validated snapshot and need')
    largest=max(need.expert_payload_bytes)
    fused_payload=sum(need.expert_payload_bytes)
    sequential_total=largest+need.expert_staging_bytes
    fused_total=fused_payload+need.expert_staging_bytes
    imminent=need.dense_next_allocation_bytes+need.kv_growth_bytes
    available=snapshot.available_bytes
    if available is not None and snapshot.physical_bytes is not None:
        available=min(available,snapshot.physical_bytes)
    # Only actual unleased expert allocations are prospective reclaim credit;
    # no advisory service quota or hypothetical cache maximum is credited.
    reclaimable=snapshot.resident_expert_bytes-snapshot.pinned_expert_bytes
    prospective=None if available is None else available+reclaimable
    workspace=need.decode_workspace_bytes if need.phase=='decode' else need.prefill_fixed_bytes
    one_row=0 if need.phase=='decode' else need.prefill_per_row_bytes
    route_known=(need.phase=='decode')if need.route_working_set_known is None else need.route_working_set_known
    fused_fits=(not snapshot.allocation_failed_bytes and prospective is not None and route_known
                and prospective>=imminent+workspace+one_row+fused_total)
    strategy='fused_top6'if fused_fits and need.phase=='decode'else'batched_experts'if fused_fits else'sequential_expert'
    active_payload=fused_payload if fused_fits else largest
    minimum_total=active_payload+need.expert_staging_bytes
    free_for_work=None if prospective is None else prospective-imminent-workspace-minimum_total
    rows=need.requested_prefill_rows
    if need.phase=='prefill':
        if free_for_work is None or snapshot.allocation_failed_bytes:rows=1
        elif need.prefill_per_row_bytes:
            rows=max(1,min(rows,max(0,free_for_work)//need.prefill_per_row_bytes))
    else:rows=1
    chosen_workspace=workspace+(need.prefill_per_row_bytes*rows if need.phase=='prefill'else 0)
    # Availability is incremental. Existing expert payload is already charged
    # by the OS and can remain in a cache target without being re-reserved.
    target_room=(0 if available is None else max(0,available+snapshot.resident_expert_bytes-imminent-chosen_workspace))
    if snapshot.physical_bytes is not None:
        target_room=min(target_room,max(0,snapshot.physical_bytes-need.dense_resident_bytes-chosen_workspace))
    target=min(need.cache_ceiling_bytes,max(minimum_total,target_room))
    target_payload=max(0,target-need.expert_staging_bytes)
    pressure=(prospective is None or prospective<imminent+chosen_workspace+minimum_total)
    # A summed workspace/growth charge is not one indivisible allocation.
    largest_allocation=max(need.dense_next_allocation_bytes,largest,need.largest_workspace_allocation_bytes)
    known_impossible=(snapshot.max_single_allocation_bytes is not None
                      and largest_allocation>snapshot.max_single_allocation_bytes)
    configured_too_small=need.cache_ceiling_bytes<sequential_total
    status=('actual_allocation_failed'if snapshot.allocation_failed_bytes else
            'known_single_allocation_limit'if known_impossible else
            'configured_expert_ceiling_too_small'if configured_too_small else
            'attempt_minimum_stream'if pressure else'adaptive_ready')
    return dict(schema='mlx2.source-adaptive-memory.v1',status=status,
        cache_ceiling_bytes=need.cache_ceiling_bytes,target_cache_bytes=target,
        target_payload_bytes=target_payload,expert_staging_bytes=need.expert_staging_bytes,
        minimum_sequential_payload_bytes=largest,minimum_sequential_total_bytes=sequential_total,
        fused_payload_bytes=fused_payload,fused_total_bytes=fused_total,
        active_payload_bytes=active_payload,expert_strategy=strategy,
        working_set_experts=len(need.expert_payload_bytes)if fused_fits else 1,
        prefill_chunk_rows=rows,prefill_strategy='batched_experts'if fused_fits else'ordinary_sequential_expert',
        workspace_bytes=chosen_workspace,kv_growth_bytes=need.kv_growth_bytes,
        dense_next_allocation_bytes=need.dense_next_allocation_bytes,
        dense_estimated_bytes=need.dense_estimated_bytes,dense_resident_bytes=need.dense_resident_bytes,
        route_working_set_known=route_known,
        dense_estimate_is_admission_requirement=False,
        dense_loading_strategy='incremental_tensors_with_possible_OS_paging',
        dense_streaming_or_paging_likely=bool(snapshot.physical_bytes is not None and need.dense_estimated_bytes>snapshot.physical_bytes),
        live_available_bytes=available,prospective_reclaim_bytes=reclaimable,
        reclaim_before_allocate=bool(reclaimable and (pressure or target_payload<snapshot.resident_expert_bytes)),
        remeasure_after_reclaim=True,pinned_payload_bytes=snapshot.pinned_expert_bytes,
        drain_pinned_epochs_before_new_acquire=snapshot.pinned_expert_bytes>target_payload,
        pressure_attempt=pressure,permit_reservation_based_rejection=False,
        known_impossible_single_allocation=known_impossible,largest_imminent_allocation_bytes=largest_allocation,
        actual_allocation_failed_bytes=snapshot.allocation_failed_bytes,
        precision_change=False,complete_layer_minimum_bytes=0,fixed_host_reserve_bytes=0,
        prefill_partition_changes=need.phase=='prefill'and rows<need.requested_prefill_rows,
        numerical_partition_notice='Changed prefill partitions require explicit route/APCv2 numerical identity; preserve ordinary reference',
        scope='CPU soft targets; no allocation success, eviction, GPU use, qualification or performance claim')
