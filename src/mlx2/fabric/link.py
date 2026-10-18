from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from mlx2.fabric import FabricError


@dataclass(frozen=True)
class LinkReceipt:
    peer_nid: str
    transport: str
    rtt_ms_median: float
    rtt_ms_p90: float
    rtt_samples: int
    upload_mbps: float | None
    download_mbps: float | None
    payload_bytes: int
    measured_at: float

    def as_dict(self) -> dict:
        return asdict(self)


def _timed_roundtrip(
    roundtrip: Callable[[bytes, int], bytes], upload: bytes, download_len: int
) -> float:
    started = time.perf_counter()
    returned = roundtrip(upload, download_len)
    elapsed = time.perf_counter() - started
    if not isinstance(returned, bytes) or len(returned) != download_len:
        actual = len(returned) if isinstance(returned, bytes) else "non-bytes"
        raise FabricError(f"link probe length mismatch: expected {download_len}, got {actual}")
    return elapsed


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]


def probe_link(
    roundtrip: Callable[[bytes, int], bytes],
    *,
    peer_nid: str,
    transport: str = "tls1.3/tcp",
    rtt_samples: int = 20,
    payload_bytes: int = 8 << 20,
) -> LinkReceipt:
    if rtt_samples < 1:
        raise FabricError("link probe requires at least one RTT sample")
    if payload_bytes < 0:
        raise FabricError("link probe payload must be non-negative")
    small = bytes(64)
    for _ in range(2):
        _timed_roundtrip(roundtrip, small, 64)
    rtts = [_timed_roundtrip(roundtrip, small, 64) * 1000 for _ in range(rtt_samples)]
    upload_s = None
    download_s = None
    if payload_bytes:
        upload_s = _timed_roundtrip(roundtrip, bytes(payload_bytes), 64)
        download_s = _timed_roundtrip(roundtrip, small, payload_bytes)
    return LinkReceipt(
        peer_nid=peer_nid,
        transport=transport,
        rtt_ms_median=statistics.median(rtts),
        rtt_ms_p90=_p90(rtts),
        rtt_samples=rtt_samples,
        upload_mbps=(
            payload_bytes * 8 / upload_s / 1_000_000
            if upload_s is not None and upload_s > 0
            else None
        ),
        download_mbps=(
            payload_bytes * 8 / download_s / 1_000_000
            if download_s is not None and download_s > 0
            else None
        ),
        payload_bytes=payload_bytes,
        measured_at=time.time(),
    )


def echo_response(body: bytes, download_len: int, max_len: int = 64 << 20) -> bytes:
    if download_len < 0 or max_len < 0:
        raise FabricError("echo response lengths must be non-negative")
    length = min(download_len, max_len)
    if not body:
        return bytes(length)
    repeats = (length + len(body) - 1) // len(body)
    return (body * repeats)[:length]


class LinkBudget:
    PER_ROUND_KINDS = ("prefill_offload", "remote_draft", "pipeline")

    def __init__(self, per_round_budget_ms: float = 5.0) -> None:
        self.per_round_budget_ms = per_round_budget_ms

    def admit(
        self,
        kind: str,
        receipt: LinkReceipt | None,
        *,
        history: Sequence[LinkReceipt] = (),
    ) -> tuple[bool, str]:
        if kind == "route":
            if receipt is None:
                return True, "whole-request route; link_unmeasured"
            return True, f"whole-request route; latest RTT p90 {receipt.rtt_ms_p90:g} ms"
        if kind not in self.PER_ROUND_KINDS:
            return False, f"unknown link operation: {kind}"
        receipts = list(history[-3:]) if history else ([receipt] if receipt else [])
        if not receipts:
            return False, "link_unmeasured"
        worst_p90 = max(item.rtt_ms_p90 for item in receipts)
        label = f"worst RTT p90(last {len(receipts)}) {worst_p90:g} ms"
        if worst_p90 > self.per_round_budget_ms:
            return (
                False,
                f"{label} > budget {self.per_round_budget_ms:g} ms",
            )
        return True, f"{label} <= budget {self.per_round_budget_ms:g} ms"
