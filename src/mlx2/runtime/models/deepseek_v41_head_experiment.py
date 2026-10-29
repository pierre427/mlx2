# SPDX-License-Identifier: MIT
"""Prepared whole-head controls for a future actual-weight paired experiment.

The direct BF16/F32 kernel already exists in source_dense. This module ensures
that its reference is prepared once, whole-vocabulary geometry is preserved,
and load/preparation costs are not silently put into steady-state timing.
Provenance: provenance/deepseek-v41-head-experiment.json.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

REFERENCE = 'source_head_prepared_f32_v1'
DIRECT = 'source_head_direct_bf16_f32_v1'
WEIGHT_SHAPE = (129280, 5120)


def plan(profile, rows=1):
    if profile not in (REFERENCE, DIRECT):
        raise ValueError('Unknown Source head experiment profile')
    if type(rows) is not int or not 1 <= rows <= (1 if profile == DIRECT else 1024):
        raise ValueError('Source head experiment row extent is unsupported')
    words = WEIGHT_SHAPE[0]*WEIGHT_SHAPE[1]
    return dict(profile=profile, weight_shape=WEIGHT_SHAPE,
                coefficient_dtype='bfloat16' if profile == DIRECT else 'float32',
                coefficient_bytes=words*(2 if profile == DIRECT else 4),
                persistent_companion_bytes=0, rows=rows,
                output_shape=(1, rows, WEIGHT_SHAPE[0]), output_dtype='float32',
                preparation_scope='before steady timing; retained separately in admission/TTFT evidence',
                qualification='unqualified')


@dataclass(frozen=True)
class PreparedHead:
    coefficients: object
    profile: str
    checkpoint: str
    source_revision: str
    implementation: str

    def receipt(self):
        return dict(**plan(self.profile), checkpoint=self.checkpoint,
                    source_revision=self.source_revision, implementation_sha256=self.implementation,
                    coefficient_owner='one retained prepared object; no permanent companion')


def prepare(weight, *, profile, checkpoint, source_revision):
    """Prepare exactly one control's coefficient representation; no fence added."""
    import mlx.core as mx
    plan(profile)
    if (not isinstance(checkpoint, str) or '@' not in checkpoint
            or not isinstance(source_revision, str) or not source_revision):
        raise ValueError('Source head preparation requires checkpoint and source identities')
    if weight.shape != WEIGHT_SHAPE or weight.dtype != mx.bfloat16:
        raise ValueError('Source head preparation requires the full original BF16 head')
    value = weight if profile == DIRECT else weight.astype(mx.float32)
    return PreparedHead(value, profile, checkpoint, source_revision, implementation_identity())


def project(x, prepared):
    """Build a fresh consumer graph over a previously prepared full head."""
    import mlx.core as mx
    if not isinstance(prepared, PreparedHead) or prepared.implementation != implementation_identity():
        raise ValueError('Source head prepared implementation identity mismatch')
    p = plan(prepared.profile, x.shape[1] if x.ndim == 3 else 0)
    dtype = mx.bfloat16 if prepared.profile == DIRECT else mx.float32
    if (x.shape != (1, p['rows'], WEIGHT_SHAPE[1]) or x.dtype != mx.float32
            or prepared.coefficients.shape != WEIGHT_SHAPE or prepared.coefficients.dtype != dtype):
        raise ValueError('Source head prepared consumer geometry/dtype mismatch')
    if prepared.profile == REFERENCE:
        return x @ prepared.coefficients.T
    from ..gguf.source_dense import project as direct
    # Explicit experiment refuses unavailable native execution; it never
    # introduces a hidden per-token full coefficient cast as a fallback.
    return direct(x, prepared.coefficients, 'head', algorithm='gemv')


@lru_cache(maxsize=1)
def implementation_identity():
    from ..gguf import source_dense
    paths = (Path(__file__), Path(source_dense.__file__))
    return hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in paths)).hexdigest()
