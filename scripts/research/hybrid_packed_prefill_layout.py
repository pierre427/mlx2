"""Host-only row contract for a future cold packed hybrid prefill.

No runtime installs this plan. It performs no tensor math, native submission,
state publication or qualification. Projection padding is deliberately absent.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillSegment:
    lane_id: int
    rows: int
    row_start: int

    @property
    def last_row(self):
        return self.row_start + self.rows - 1


@dataclass(frozen=True)
class ColdHybridPrefillLayout:
    segments: tuple[PrefillSegment, ...]
    total_rows: int

    @property
    def gdn_parts(self):
        """(batch rows, sequence length, packed start); bind private cache later."""
        return tuple((1, s.rows, s.row_start) for s in self.segments)

    @property
    def logit_rows(self):
        return tuple(s.last_row for s in self.segments)

    def lane_position(self, packed_row):
        if type(packed_row) is not int or not 0 <= packed_row < self.total_rows:
            raise ValueError('packed query row outside real rows')
        for segment in self.segments:
            if packed_row < segment.row_start + segment.rows:
                return segment.lane_id, packed_row - segment.row_start
        raise AssertionError('uncovered real row')

    def causal_key_visible(self, packed_query, key_lane, local_key):
        """Cold absolute positions start at zero; never admit another lane."""
        lane, position = self.lane_position(packed_query)
        if type(key_lane) is not int or type(local_key) is not int or local_key < 0:
            raise ValueError('invalid logical key coordinate')
        segment = next((s for s in self.segments if s.lane_id == key_lane), None)
        if segment is None or local_key >= segment.rows:
            raise ValueError('logical key outside admitted lane')
        return lane == key_lane and local_key <= position


def cold_layout(lanes, *, max_lane_rows=8192, max_total_rows=16384):
    """lanes=((unique nonnegative lane ID, actual prompt row count), ...).

    Supports only one/two cold full-acceptance lanes. This cannot represent a
    checkpoint successor, chunk continuation, speculation or an empty lane.
    Bounds are host planning limits, not measured device memory admission.
    """
    if (type(max_lane_rows) is not int or max_lane_rows < 1 or
            type(max_total_rows) is not int or max_total_rows < 1 or
            type(lanes) is not tuple or not 1 <= len(lanes) <= 2):
        raise ValueError('bounded one/two cold lanes required')
    result, start, seen = [], 0, set()
    for lane in lanes:
        if (type(lane) is not tuple or len(lane) != 2 or
                type(lane[0]) is not int or lane[0] < 0 or lane[0] in seen or
                type(lane[1]) is not int or not 1 <= lane[1] <= max_lane_rows):
            raise ValueError('unique lane ID and positive bounded real rows required')
        lane_id, rows = lane
        seen.add(lane_id)
        result.append(PrefillSegment(lane_id, rows, start))
        start += rows
    if start > max_total_rows:
        raise ValueError('real packed rows exceed host bound')
    return ColdHybridPrefillLayout(tuple(result), start)
