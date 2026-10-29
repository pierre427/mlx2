# SPDX-License-Identifier: MIT
"""Uncompiled explicit-resource resident lookup probe; no expert math.

All packed backings are ordinary MLX kernel inputs. The component checks device
lookup, format and generation guards without a GPU address table. It cannot be
selected as a serving route. Provenance: deepseek-v41-source-resident.json.
"""
import numpy as np

from ...adapters.deepseek_v41_source_resident import PILOT_MAX_EXPERTS, ResidentEpoch


def descriptors(epoch):
    if not isinstance(epoch, ResidentEpoch):
        raise ValueError('Resident probe requires a live ownership epoch')
    bindings = epoch.bindings
    if not 1 <= len(bindings) <= PILOT_MAX_EXPERTS:
        raise ValueError('Resident probe binding bound exceeded')
    table = np.zeros((384, 4), np.uint32)
    sizes = np.zeros((len(bindings),), np.uint32)
    serials = set()
    layer = bindings[0].key[0]
    for slot, binding in enumerate(bindings):
        if binding.key[0] != layer or binding.serial in serials:
            raise ValueError('Resident probe layer/serial mismatch')
        serials.add(binding.serial)
        if not 0 < binding.serial < (1 << 64) or not 0 < binding.nbytes < (1 << 32):
            raise ValueError('Resident probe descriptor integer overflow')
        table[binding.key[1]] = (slot + 1, 1 if binding.expert.layout.kind == 'original' else 2,
                                 binding.serial & 0xffffffff, binding.serial >> 32)
        sizes[slot] = binding.nbytes
    table.flags.writeable = sizes.flags.writeable = False
    return table, sizes


def kernel_spec(expert_count):
    if type(expert_count) is not int or not 1 <= expert_count <= PILOT_MAX_EXPERTS:
        raise ValueError('Resident probe requires1..15 explicitly declared resources')
    names = ['ids', 'table', 'sizes', 'generation', 'expected_generation']
    names.extend(f'expert{i}' for i in range(expert_count))
    selects = '\n'.join(f'case {i}: backing=expert{i}; break;' for i in range(expert_count))
    source = r'''
const uint route=thread_position_in_grid.x;
if(route>=6)return;
const uint id=ids[route];
for(uint c=0;c<4;++c)probe[route*4+c]=0;
status[route]=0;
if(generation[0]!=expected_generation[0] || generation[1]!=expected_generation[1]) {
    status[route]=3; return;
}
if(id>=384) {status[route]=2;return;}
const uint encoded=table[id*4];
if(encoded==0) {status[route]=1;return;}
const uint slot=encoded-1;
'''+f'if(slot>={expert_count}){{status[route]=4;return;}}\n'+r'''
const uint kind=table[id*4+1];
if((kind!=1 && kind!=2) || sizes[slot]==0) {status[route]=4;return;}
const device uchar* backing=nullptr;
switch(slot) {
'''+selects+r'''
}
// Explicit inputs declare and retain every resource reachable through switch.
// Bounds come from host-validated actual backing sizes, not a GPU address.
probe[route*4]=id;
probe[route*4+1]=kind;
probe[route*4+2]=backing[0];
probe[route*4+3]=backing[sizes[slot]-1];
'''
    return dict(input_names=names, output_names=['probe', 'status'], header='', source=source)


def build(epoch, ids, *, expected_generation):
    """Build a lazy six-route read probe; caller supplies ownership and GPU lease.

    CPU tests exercise only its pre-compilation device refusal. It does not evaluate,
    read router IDs, or release epoch resources. Call epoch.retire only after a
    wait covering BOTH outputs succeeds. This is a GPU component entry point,
    not authority to execute outside the user's approved ownership window.
    """
    import mlx.core as mx
    if mx.default_device()!=mx.gpu or not mx.metal.is_available():
        raise ValueError('Source resident probe requires an approved owned Metal device')
    if (type(expected_generation) is not int or not 0 <= expected_generation < (1 << 64)
            or ids.shape != (6,) or ids.dtype != mx.uint32):
        raise ValueError('Resident probe route/generation geometry mismatch')
    table, sizes = descriptors(epoch)
    backings = [binding.expert.backing for binding in epoch.bindings]
    if any(not binding.expert.has_compact_backing or b.dtype != mx.uint8 or b.shape != (binding.nbytes,)
           for b, binding in zip(backings, epoch.bindings)):
        raise ValueError('Resident probe backing geometry mismatch')
    split = lambda n: mx.array([n & 0xffffffff, n >> 32], mx.uint32)
    kernel = mx.fast.metal_kernel(name=f'mlx2_source_resident_probe_e{len(backings)}_v1',
                                  **kernel_spec(len(backings)), ensure_row_contiguous=False)
    probe, status = kernel(inputs=[ids, mx.array(table), mx.array(sizes), split(epoch.generation),
                                  split(expected_generation), *backings],
                           output_shapes=[(6, 4), (6,)], output_dtypes=[mx.uint32, mx.uint32],
                           grid=(6, 1, 1), threadgroup=(32, 1, 1))
    epoch.note_roots(probe, status)
    return probe, status
