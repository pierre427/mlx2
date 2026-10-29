# SPDX-License-Identifier: MIT
"""Default-off native table component; no serving route or pointer integers.

Direct table lookup only admits hazard-tracked actual buffers. The optional
tracked-staging mode is limited to tiny component data, never expert banks.
Provenance: provenance/deepseek-v41-source-native-resource-pilot.json.
"""
from dataclasses import dataclass

PROFILE='source_native_indirect_resource_pilot_v1'


def _owned_gpu(mx,check_live):
    if not callable(check_live)or check_live()is not True:
        raise ValueError('Native resource pilot requires callable live ownership')
    if mx.default_device()!=mx.gpu:
        raise ValueError('Native resource pilot requires owned GPU backend')


@dataclass
class NativeResourceResult:
    output: object
    token: object
    native: object
    epoch: object=None

    def receipt(self):
        return dict(self.native.receipt(self.token),profile=PROFILE,serving_backend=False,qualified=False)


def build(epoch,ids,wanted_serials,*,expected_generation,check_live,mode='direct_array_v1'):
    """Build a lazy native read using one already-published ownership epoch.

    Native graph dependencies preserve upload order; the CPU epoch must still
    have published each backing only after upload completion. All output/status
    columns share one root. Epoch retirement must wait that root and every
    independent subsequent consumer; failed waits retain the epoch normally.
    """
    import mlx.core as mx
    _owned_gpu(mx,check_live)
    from .deepseek_v41_source_resident import ResidentEpoch
    if not isinstance(epoch,ResidentEpoch):raise ValueError('Live ResidentEpoch required')
    if mode!='direct_array_v1':
        raise ValueError('Actual expert epochs support direct capability probe only; tiny staging is separate')
    bindings=epoch.bindings
    if not 1<=len(bindings)<=15:raise ValueError('Native component admits1..15resources')
    if any(not b.expert.has_compact_backing for b in bindings):raise ValueError('Compact actual backing required')
    import _ds41_resident_pilot as native
    output,token=native.lookup([b.expert.backing for b in bindings],ids,wanted_serials,
        [b.key[1]for b in bindings],[1 if b.expert.layout.kind=='original'else 2 for b in bindings],
        [b.serial for b in bindings],epoch.generation,expected_generation,mode,mx.default_stream(mx.gpu),True)
    epoch.note_roots(output)
    return NativeResourceResult(output,token,native,epoch)


def component_lookup(backings,ids,wanted_serials,*,keys,kinds,serials,generation,
                     expected_generation,check_live,mode='tracked_staging_v1'):
    """Tiny explicit component entry. Native code enforces payload/staging caps."""
    import mlx.core as mx
    _owned_gpu(mx,check_live)
    import _ds41_resident_pilot as native
    output,token=native.lookup(list(backings),ids,wanted_serials,list(keys),list(kinds),list(serials),
        generation,expected_generation,mode,mx.default_stream(mx.gpu),True)
    return NativeResourceResult(output,token,native)
