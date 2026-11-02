"""Scalar packed K-format semantics shared by all packed consumers.

No kernels compile on import. See provenance/gguf-k-quants.json.
"""

HEADER = r"""
inline float gguf_k_half(device const uchar *p) {
    const ushort bits=ushort(p[0]) | (ushort(p[1])<<8);
    return float(as_type<half>(bits));
}
inline float gguf_k_value(device const uchar *row, uint k, uint format) {
    if(format==2) {
        device const uchar *p=row+(k/32)*18;
        uint i=k%32;
        return gguf_k_half(p)*float(int((p[2+i%16]>>(4*(i/16)))&15)-8);
    }
    uint i=k%256;
    if(format==11) {
        device const uchar *p=row+(k/256)*110;
        uint g=i/16;
        uint lo=(p[96+g%8]>>(4*(g/8)))&15;
        uint hi=(p[104+g%4]>>(2*(g/4)))&3;
        int scale=int(lo|(hi<<4))-32;
        int q=int((p[32+(i/128)*32+i%32]>>(2*((i/32)%4)))&3);
        q-=4*(1-int((p[i%32]>>(i/32))&1));
        return (gguf_k_half(p+108)*float(scale))*float(q);
    }
    if(format==12 || format==13) {
        uint size=format==12 ? 144 : 176;
        device const uchar *p=row+(k/256)*size;
        uint g=i/32, sc=p[4+g%4], mn=p[8+g%4], aux=p[12+g%4];
        uint scale=g<4 ? sc&63 : (aux&15)|((sc>>2)&48);
        uint minimum=g<4 ? mn&63 : (aux>>4)|((mn>>2)&48);
        uint off=format==12 ? 16 : 48;
        uint q=(p[off+(i/64)*32+i%32]>>(4*((i/32)%2)))&15;
        if(format==13) q|=((p[16+i%32]>>(i/32))&1)<<4;
        return (gguf_k_half(p)*float(scale))*float(q)-gguf_k_half(p+2)*float(minimum);
    }
    device const uchar *p=row+(k/256)*210;
    uint lo=(p[(i/128)*64+i%64]>>(4*((i/64)%2)))&15;
    uint hi=(p[128+(i/128)*32+i%32]>>(2*((i/32)%4)))&3;
    int q=int(lo|(hi<<4))-32;
    int scale=int(as_type<char>(p[192+i/16]));
    return (gguf_k_half(p+208)*float(scale))*float(q);
}
"""
