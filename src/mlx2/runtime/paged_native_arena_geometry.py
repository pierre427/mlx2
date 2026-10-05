"""Host preflight for exact-byte native arena allocation; no tensor imports."""
MAX_HOST_PLANE_BYTES=12<<30
SHAPE_DIM_MAX=(1<<31)-1


def require_arena_storage(native,plane_bytes):
    if type(plane_bytes) is not int or not 0<plane_bytes<=MAX_HOST_PLANE_BYTES:
        raise ValueError('native arena preallocation requires exact positive bounded plane bytes')
    getter=getattr(native,'arena_storage_capability',None)
    if not callable(getter):raise ValueError('native arena allocation capability ABI missing before admission')
    cap=getter()
    fixed={'version':1,'layout':'contiguous_uint8_2d_large','large_plane_threshold_bytes':SHAPE_DIM_MAX,'large_plane_alignment_bytes':4096,'exact_byte_allocation':True}
    if (type(cap) is not dict or set(cap)!={*fixed,'max_plane_bytes'} or
            any(type(cap.get(k)) is not type(v) or cap.get(k)!=v for k,v in fixed.items()) or
            type(cap.get('max_plane_bytes')) is not int or not 4096<=cap['max_plane_bytes']<=MAX_HOST_PLANE_BYTES or cap['max_plane_bytes']%4096):
        raise ValueError('native arena allocation capability contract differs')
    if plane_bytes>cap['max_plane_bytes']:raise MemoryError('native arena plane exceeds actual Metal buffer capability before charge')
    if plane_bytes>SHAPE_DIM_MAX:
        if plane_bytes%4096:raise ValueError('large native arena plane requires exact4096-byte alignment without padding')
        shape=(plane_bytes//4096,4096)
    else:shape=(plane_bytes,)
    if any(dim>SHAPE_DIM_MAX for dim in shape):raise ValueError('native arena shape exceeds signed dimension bound')
    return {'capability':dict(cap),'plane_bytes':plane_bytes,'plane_shape':shape,'allocation_bytes':2*plane_bytes,'padding_bytes':0}
