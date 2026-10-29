# SPDX-License-Identifier: MIT
"""Evidence-bound Source tile dispatch and complete-stage probe inventory.

No performance thresholds are invented here. Callers supply a preregistered
minimum ratio and exact-shape, exact-source measurements. Missing evidence
keeps the reference. No coefficient, model, or device allocation occurs.
"""
from dataclasses import dataclass
import math
import statistics

from .deepseek_v41_source_prefill import Job

STAGE_COSTS = ('route_plan', 'metadata_upload', 'acquire_and_lease', 'input_qat',
               'row_prepare', 'gate_up', 'scored_middle', 'middle_qat', 'down',
               'ordered_combine', 'completion')
PROBE_ROWS = (1, 3, 5, 9, 12, 13, 15, 16, 17, 31, 32, 33, 34,
              63, 64, 65, 95, 96, 97, 98, 99)
TILES = ((8, 32), (16, 32), (16, 64), (32, 32), (32, 64))


@dataclass(frozen=True)
class ShapeKey:
    hardware: str
    checkpoint: str
    source_revision: str
    source_snapshot: str
    kind: str
    projection: str
    rows: int
    layout: str
    numerical_profile: str

    def __post_init__(self):
        if (any(type(s) is not str or not s for s in (self.hardware, self.checkpoint,
                self.source_revision, self.source_snapshot, self.layout, self.numerical_profile))
                or '@' not in self.checkpoint or self.kind not in ('original', 'q1')
                or self.projection not in ('gate', 'up', 'down', 'ffn')
                or type(self.rows) is not int or not 1 <= self.rows <= 6144):
            raise ValueError('Source tile dispatch requires a complete exact-shape binding')


@dataclass(frozen=True)
class TileMeasurement:
    key: ShapeKey
    candidate: str
    tile: tuple
    reference_seconds: tuple
    candidate_seconds: tuple
    evidence: str
    cost_scope: tuple
    numerical_gate_passed: bool

    def __post_init__(self):
        if (not isinstance(self.key, ShapeKey) or type(self.candidate) is not str
                or not self.candidate or self.candidate == 'reference'
                or self.tile not in TILES or type(self.evidence) is not str or not self.evidence
                or type(self.numerical_gate_passed) is not bool):
            raise ValueError('Source tile measurement binding is incomplete')
        if (type(self.reference_seconds) is not tuple or type(self.candidate_seconds) is not tuple
                or len(self.reference_seconds) != len(self.candidate_seconds)
                or not self.reference_seconds
                or any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0
                       for t in self.reference_seconds + self.candidate_seconds)):
            raise ValueError('Source tile measurement needs finite paired positive samples')
        if type(self.cost_scope) is not tuple or len(set(self.cost_scope)) != len(self.cost_scope):
            raise ValueError('Source tile measurement cost scope is invalid')

    @property
    def paired_ratio(self):
        return statistics.median(a / b for a, b in zip(self.reference_seconds, self.candidate_seconds))


@dataclass(frozen=True)
class TileSelection:
    candidate: str
    tile: tuple | None
    reason: str
    evidence: str | None
    qualification: str = 'unqualified'


def select_tile(key, measurements, *, minimum_ratio):
    """Select only source-matched whole-stage evidence with a passed oracle.

    This is not a runtime default or qualification authority. No caller may
    transfer a wide-row gain to M1 or an original-expert result to Q1.
    """
    if not isinstance(key, ShapeKey):
        raise ValueError('Source tile selection key is invalid')
    if type(minimum_ratio) not in (int, float) or not math.isfinite(minimum_ratio) or minimum_ratio < 1:
        raise ValueError('Source tile minimum ratio must be explicitly preregistered and >=1')
    measurements = tuple(measurements)
    if any(not isinstance(m, TileMeasurement) for m in measurements):
        raise ValueError('Source tile selection requires validated measurements')
    eligible = [m for m in measurements if m.key == key and m.numerical_gate_passed
                and set(m.cost_scope) == set(STAGE_COSTS) and m.paired_ratio > minimum_ratio]
    if not eligible:
        return TileSelection('reference', None, 'no_matching_whole_stage_win', None)
    best = max(eligible, key=lambda m: (m.paired_ratio, m.candidate, m.tile))
    return TileSelection(best.candidate, best.tile, 'matching_whole_stage_evidence', best.evidence)


def row_slot_cost(jobs, tile_rows):
    """CPU row-capacity arithmetic, explicitly not hardware occupancy."""
    if type(tile_rows) is not int or tile_rows not in (1, 2, 4, 8, 16, 32):
        raise ValueError('Source tile row capacity is unsupported')
    jobs = tuple(jobs)
    if any(not isinstance(j, Job) or type(j.count) is not int or j.count <= 0 for j in jobs):
        raise ValueError('Source tile row capacity requires complete expert jobs')
    active = sum(j.count for j in jobs)
    capacity = sum(((j.count + tile_rows - 1) // tile_rows) * tile_rows for j in jobs)
    return dict(active_rows=active, scheduled_rows=capacity, padded_rows=capacity - active,
                useful_row_fraction=active / capacity if capacity else None,
                scope='independent-expert CPU row-capacity arithmetic; not GPU occupancy or traffic')


def probe_inventory():
    """Bounded experiment axes; kernels are never compiled or run here."""
    return tuple(dict(kind=kind, rows=rows, tile=tile,
                      projection=projection, input_layout=layout,
                      preserve_whole_expert_M=True, cost_scope=STAGE_COSTS,
                      numerical_profile='candidate_requires_explicit_oracle',
                      route='decode_control' if rows == 1 else 'prefill',
                      coefficient_repack_bytes=0)
                 for kind in ('original', 'q1') for rows in PROBE_ROWS
                 for tile in TILES for projection in ('gate', 'up', 'down')
                 for layout in ('mapped_rows', 'fresh_compact_rows'))
