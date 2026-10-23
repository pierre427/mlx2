// SPDX-License-Identifier: Apache-2.0
// Adapted from recursive-oss/rec 8af99abaf50b33903147d89eed91d1bb3ac642dd; see provenance/rec-flint-fused.json.
struct Matrix {
    const device uchar* w;
    const device bfloat16_t* s;
    const device bfloat16_t* b;
    uint n, k;
};




inline Matrix weight_matrix(const device uchar* bank, const device uint* d) {
    return {bank+d[0], (const device bfloat16_t*)(bank+d[1]),
            (const device bfloat16_t*)(bank+d[2]),d[3],d[4]};
}

inline float load16(const device bfloat16_t* x, thread float* v) {
    float sum=0.0f;
    for(uint i=0;i<16;i+=4) {
        sum+=float(x[i])+float(x[i+1])+float(x[i+2])+float(x[i+3]);
        v[i]=x[i];v[i+1]=x[i+1]/16.0f;
        v[i+2]=x[i+2]/256.0f;v[i+3]=x[i+3]/4096.0f;
    }
    return sum;
}

inline float load16_local(const threadgroup bfloat16_t* x, thread float* v) {
    float sum=0.0f;
    for(uint i=0;i<16;i+=4) {
        sum+=float(x[i])+float(x[i+1])+float(x[i+2])+float(x[i+3]);
        v[i]=x[i];v[i+1]=x[i+1]/16.0f;
        v[i+2]=x[i+2]/256.0f;v[i+3]=x[i+3]/4096.0f;
    }
    return sum;
}

inline float dot16(const device uchar* w,thread float* x,float s,float b,float sum) {
    const device ushort* p=(const device ushort*)w;
    float a=0.0f;
    for(uint j=0;j<4;++j)
        a+=x[4*j]*(p[j]&15u)+x[4*j+1]*(p[j]&240u)
           +x[4*j+2]*(p[j]&3840u)+x[4*j+3]*(p[j]&61440u);
    return s*a+sum*b;
}

template<uint K>
inline void q4_local(Matrix m,const threadgroup bfloat16_t* x,device bfloat16_t* out,
               uint lane,uint sg,uint groups,uint expert=0) {
    for(uint first=sg*4;first<m.n;first+=groups*4) {
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint block=0;block<K;block+=512) {
            uint scalar=block+lane*16;
            float v[16];float sum=load16_local(x+scalar,v);
            for(uint r=0;r<4;++r) {
                uint row=expert*m.n+first+r;uint gi=row*(K/64)+scalar/64;
                a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint r=0;r<4;++r) {
            float v=simd_sum(a[r]);if(lane==0)out[first+r]=bfloat16_t(v);
        }
    }
}

inline void head_norm(device bfloat16_t* x,const device bfloat16_t* w,uint heads,
                      bool rotary,uint position,float logbase,float eps,uint lane,uint sg) {
    for(uint h=sg;h<heads;h+=288) {
        device bfloat16_t* p=x+h*128;
        float squared=0.0f;for(uint j=0;j<4;++j){float v=p[lane*4+j];squared+=v*v;}
        float inverse=metal::precise::rsqrt(simd_sum(squared)/128.0f+eps);
        if(rotary) {
            for(uint j=0;j<2;++j) {
                uint pair=lane*2+j;
                bfloat16_t a=bfloat16_t(float(w[pair])*float(p[pair])*inverse);
                bfloat16_t b=bfloat16_t(float(w[pair+64])*float(p[pair+64])*inverse);
                float theta=float(position)*metal::exp2(-float(pair)/64.0f*logbase);
                float c=metal::fast::cos(theta),s=metal::fast::sin(theta);
                p[pair]=bfloat16_t(float(a)*c-float(b)*s);
                p[pair+64]=bfloat16_t(float(a)*s+float(b)*c);
            }
        } else for(uint j=0;j<4;++j){uint d=lane*4+j;p[d]=bfloat16_t(float(w[d])*float(p[d])*inverse);}
    }
}

inline Matrix weight_matrix(const device uchar* bank,const threadgroup uint* d) {
    return {bank+d[0],(const device bfloat16_t*)(bank+d[1]),
            (const device bfloat16_t*)(bank+d[2]),d[3],d[4]};
}

inline void rms_256_w4(const thread bfloat16_t* x,const device bfloat16_t* w,
                threadgroup bfloat16_t* out,uint tid,uint lane,uint sg,
                threadgroup float* temp,float eps) {

    float sums[2];
    for(uint leaf_set=0;leaf_set<2;++leaf_set) {
        float squared=0.0f;
        for(uint j=0;j<4;++j){float v=float(x[leaf_set*4+j]);squared+=v*v;}
        sums[leaf_set]=simd_sum(squared);
    }
    if(lane==0){temp[sg]=sums[0];temp[sg+8]=sums[1];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float leaf=lane<16?temp[lane]:0.0f;
    float total=simd_sum(leaf);
    float inverse=metal::precise::rsqrt(total/2048.0f+eps);
    for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;out[i]=bfloat16_t(float(w[i])*float(x[j])*inverse);}
    threadgroup_barrier(mem_flags::mem_threadgroup);

}

inline void expert_team_gateup(const device uchar* bank,const threadgroup uint* d,
                      const threadgroup bfloat16_t* x,const constant uint* ids,
                      const device bfloat16_t* silu,device bfloat16_t* out,
                      uint lane,uint local_sg,uint group) {
    bool shared=group>=24;
    uint role=shared?6:group/4, team=shared?group-24:group%4;
    uint teams=shared?8:4, width=shared?1024:512;
    Matrix g=weight_matrix(bank,d+(shared?45:30));
    Matrix u=weight_matrix(bank,d+(shared?50:35));
    uint expert=shared?0:ids[role];
    uint role_sg=team*8+local_sg;
    for(uint local=role_sg*2;local<width;local+=teams*8*2) {
        float ga[2]={0.0f,0.0f},ua[2]={0.0f,0.0f};
        for(uint b=0;b<2048;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16_local(x+scalar,v);
            for(uint j=0;j<2;++j) {
                uint row=expert*g.n+local+j,gi=row*32+scalar/64;
                ga[j]+=dot16(g.w+size_t(row)*1024+scalar/2,v,float(g.s[gi]),float(g.b[gi]),sum);
                ua[j]+=dot16(u.w+size_t(row)*1024+scalar/2,v,float(u.s[gi]),float(u.b[gi]),sum);
            }
        }
        for(uint j=0;j<2;++j) {
            float2 val=simd_sum(float2(ga[j],ua[j]));
            if(lane==0) {
                bfloat16_t g=bfloat16_t(val.x),u=bfloat16_t(val.y);
                if(shared)out[role*512+local+j]=bfloat16_t(silu[as_type<uint>(float(g))>>16]*u);
                else {auto tail=1/(1+metal::exp(metal::abs(g)));bfloat16_t sig=g<0?tail:1-tail;out[role*512+local+j]=bfloat16_t(bfloat16_t(g*sig)*u);}
            }
        }
    }
}
