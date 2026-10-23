// SPDX-License-Identifier: Apache-2.0
// Adapted from recursive-oss/rec 8af99abaf50b33903147d89eed91d1bb3ac642dd; see provenance/rec-flint-fused.json.
    uint tid=thread_index_in_threadgroup,lane=thread_index_in_simdgroup;
    uint group=threadgroup_position_in_grid.x;
    threadgroup uint descriptor[64];
    threadgroup bfloat16_t n[2048];
    if(tid<64)descriptor[tid]=metadata[parameters[0]*64+tid];
    for(uint i=tid;i<2048;i+=256)n[i]=normalized[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    expert_team_gateup(weights,descriptor,n,indices,silu_table,activation,lane,simdgroup_index_in_threadgroup,group);
