// SPDX-License-Identifier: Apache-2.0
// Quantized dot arithmetic adapted from recursive-oss/rec 8af99abaf50b33903147d89eed91d1bb3ac642dd.
// Original mlx2 ownership: each SIMD writes complete output rows, including
// all routed experts, the shared expert and residual. No cross-group waits.
    uint lane=thread_index_in_simdgroup;
    uint sg=threadgroup_position_in_grid.x*8+simdgroup_index_in_threadgroup;
    uint first=sg*4;
    threadgroup uint descriptor[64];
    if(thread_index_in_threadgroup<64)descriptor[thread_index_in_threadgroup]=metadata[parameters[0]*64+thread_index_in_threadgroup];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup uint* d=descriptor;
    float routed[4]={0.0f,0.0f,0.0f,0.0f};
    for(uint role=0;role<7;++role) {
        bool shared=role==6;
        Matrix m=weight_matrix(weights,d+(shared?55:40));
        uint expert=shared?0:indices[role];
        const device bfloat16_t* input=activation+role*512;
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint b=0;b<m.k;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16(input+scalar,v);
            for(uint j=0;j<4;++j) {
                uint row=expert*2048+first+j,gi=row*(m.k/64)+scalar/64;
                a[j]+=dot16(m.w+size_t(row)*(m.k/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint j=0;j<4;++j) {
            bfloat16_t value=bfloat16_t(simd_sum(a[j]));
            if(!shared)routed[j]+=float(value)*routing_weights[role];
            else if(lane==0) {
                bfloat16_t mlp=bfloat16_t(float(bfloat16_t(routed[j]))+float(value));
                result[first+j]=bfloat16_t(float(residual[first+j])+float(mlp));
            }
        }
    }
