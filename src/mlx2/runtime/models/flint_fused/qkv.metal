// SPDX-License-Identifier: Apache-2.0
// Adapted from recursive-oss/rec 8af99abaf50b33903147d89eed91d1bb3ac642dd; see provenance/rec-flint-fused.json.

    uint tid=thread_index_in_threadgroup,lane=thread_index_in_simdgroup;
    uint group=threadgroup_position_in_grid.x;
    uint layer=parameters[0],position=logical_position[0];
    threadgroup float reduction[33];
    threadgroup uint descriptor[64];
    alignas(8) threadgroup bfloat16_t n[2048];
    bfloat16_t h[8];
    for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;h[j]=hidden[i];}
    if(tid<64)descriptor[tid]=metadata[layer*64+tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup uint* d=descriptor;
    const device uchar* bank=weights;
    bool global=d[1]!=0;
    rms_256_w4(h,(const device bfloat16_t*)(bank+d[3]),n,tid,lane,simdgroup_index_in_threadgroup,reduction,1e-06f);
    for(uint head=group;head<40;head+=36) {
        uint local=head<32?head:head<36?head-32:head-36;
        Matrix m=weight_matrix(bank,d+(head<32?10:head<36?15:20));
        m.w+=size_t(local)*128*1024;m.s+=local*128*32;m.b+=local*128*32;m.n=128;
        device bfloat16_t* out=(head<32?queries:head<36?new_keys:new_values)+local*128;
        q4_local<2048>(m,n,out,lane,simdgroup_index_in_threadgroup,8);
        threadgroup_barrier(mem_flags::mem_device);
        if(simdgroup_index_in_threadgroup==0 && head<36)
            head_norm(out,(const device bfloat16_t*)(bank+d[head<32?5:6]),1,!global,position,13.287712379549449f,1e-06f,lane,0);
        threadgroup_barrier(mem_flags::mem_device);
    }
