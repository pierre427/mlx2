"""Bit-exact (batch-invariant) verify mode for quantized matmuls.

The mlx fork's batched verify matmuls pick a kernel by row count M: ``qmv_fast``
at M = 1, ``qmv_wide`` for small M, NAX ``affine_qmv_nax`` around M = 8..16 on
M5, and split-K ``qmm`` above that. Each one reduces in a different order, so a
greedy request can produce different tokens depending on how many lanes share
its verify call. (At 4 lanes, M = 12 goes to NAX; see
``qualification/runs/qmm-nax-small-m-20260919``.)

The fork's bit-exact mode (``mx.metal.set_qmv_bitexact``) computes every row
of a matmul with at most ``qmv_bitexact_max_m`` rows using exactly the
arithmetic of the single-row ``qmv_fast`` call. Each row is therefore
bit-identical to decoding it alone, whatever M is. The mode's route counter,
``mx.metal.qmv_bitexact_dispatches``, is a host atomic that is bumped at
encode time, so reading it never synchronizes with the GPU.

This module owns the serving side of the mode:

- **Policy.** Default off.
- **Capability probe.** Fails closed at startup when the installed mlx does
  not have the mode.
- **Activation.** The mode is process-global. It is set before the first
  request, so there is never a batch that mixes modes.
- **Counters.** Sync-free counters that are always on.
- **Receipt.** A request's receipt says ``verify_bitexact: true`` only when:
  - the mode was on for the request's whole lifetime;
  - the fork reports the mode as on;
  - the route counter advanced while the request ran (the mechanism ran).
  The coverage is quantized matmuls. The non-matmul sources of width
  dependence that are *not* covered are listed in every receipt, so no
  receipt claims more than the kernels guarantee.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

SCHEMA = "mlx2.verify-bitexact.v1"
SCOPE = "quantized_matmul"
REQUIRED_API = (
    "set_qmv_bitexact",
    "qmv_bitexact",
    "qmv_bitexact_max_m",
    "qmv_bitexact_dispatches",
)

# Width-dependent computation the bit-exact matmul mode does not cover.
# Inventory, 2026-09-19. See docs/SERVING.md, "Bit-exact verify".
RESIDUAL_WIDTH_DEPENDENCE = (
    # Unquantized (for example BF16 mtp.fc) matmuls switch between gemv and
    # steel gemm by M. On the 27B this is the draft head only, so it changes
    # MTP acceptance but never the committed (target-verified) tokens.
    "unquantized_matmul",
    # MoE gathers whose batch exceeds 8 * max_m rows (long prefills) keep the
    # tiled gather_qmm / rhs kernels.
    "moe_gather_above_bound",
    # SDPA over a joined, padded KV slab. Segmented batch caches attend each
    # row over its own KV and are not affected. A joined slab picks the
    # 2-pass block count from the padded length.
    "joined_slab_sdpa",
    # A QSA shared-suffix merge order depends on the sharing topology.
    "qsa_shared_suffix_merge",
    # Non-split qmm above max_m (prefill) is treated as row-invariant. The
    # fork GPU test measures this rather than assuming it.
    "qmm_above_max_m_unverified",
)


class VerifyBitexactUnavailable(ValueError):
    """The request or server asked for bit-exact verify and it cannot be given."""


def _metal(mx_module=None):
    if mx_module is None:
        import mlx.core as mx_module  # noqa: PLC0415
    return getattr(mx_module, "metal", None)


def capability(mx_module=None) -> dict:
    """Probe the installed mlx for the fork's bit-exact mode (host only)."""
    metal = _metal(mx_module)
    missing = [name for name in REQUIRED_API if not hasattr(metal, name)]
    if metal is None or missing:
        return {
            "available": False,
            "reason": "mlx_without_bitexact_qmv",
            "missing_api": missing if metal is not None else list(REQUIRED_API),
        }
    try:
        metal_ok = bool(metal.is_available())
    except Exception:  # noqa: BLE001 - a probe never raises
        metal_ok = False
    if not metal_ok:
        return {"available": False, "reason": "metal_unavailable", "missing_api": []}
    return {
        "available": True,
        "reason": None,
        "missing_api": [],
        "max_m": int(metal.qmv_bitexact_max_m()),
    }


@dataclass(frozen=True)
class VerifyBitexactPolicy:
    enabled: bool = False

    @classmethod
    def from_value(cls, value) -> "VerifyBitexactPolicy":
        if value is None or value is False:
            return cls(False)
        if value is True:
            return cls(True)
        if isinstance(value, str):
            text = value.strip().casefold()
            if text in {"", "off", "0", "false", "no"}:
                return cls(False)
            if text in {"on", "1", "true", "yes"}:
                return cls(True)
        if isinstance(value, dict):
            return cls.from_value(value.get("enabled", False))
        raise ValueError(f"verify_bitexact must be on/off, got {value!r}")

    def as_dict(self) -> dict:
        return {"enabled": self.enabled}


class VerifyBitexactHandle:
    """A process-global bit-exact mode that the serving engine has turned on."""

    def __init__(self, metal, *, max_m: int):
        self._metal = metal
        self.max_m = int(max_m)
        self._lock = threading.Lock()
        self._previous = None
        self._active = False
        # Increments whenever the mode is switched, so a receipt can tell
        # whether the mode stayed on for a request's whole lifetime.
        self._epoch = 0
        self.counts = {
            "activations": 0,
            "requests": 0,
            "requests_explicit": 0,
            "receipts_true": 0,
            "receipts_false": 0,
        }

    # -- mode ---------------------------------------------------------------
    def activate(self) -> None:
        with self._lock:
            previous = bool(self._metal.set_qmv_bitexact(True))
            if not bool(self._metal.qmv_bitexact()):
                raise VerifyBitexactUnavailable(
                    "mlx refused to enable the bit-exact qmv mode"
                )
            if not self._active:
                self._previous = previous
            self._active = True
            self._epoch += 1
            self.counts["activations"] += 1

    def remove(self) -> None:
        with self._lock:
            if self._active:
                self._metal.set_qmv_bitexact(bool(self._previous))
                self._active = False
                self._epoch += 1

    @property
    def active(self) -> bool:
        return self._active

    def dispatches(self) -> int:
        return int(self._metal.qmv_bitexact_dispatches())

    # -- per request ----------------------------------------------------------
    def begin_request(self, *, explicit: bool) -> dict:
        """Snapshot taken when a job is prepared (host reads only)."""
        self.counts["requests"] += 1
        if explicit:
            self.counts["requests_explicit"] += 1
        return {
            "epoch": self._epoch,
            "active": self._active,
            "dispatches": self.dispatches(),
            "explicit": bool(explicit),
        }

    def request_receipt(self, start: dict | None) -> dict:
        """Fail closed: ``verify_bitexact`` is true only with evidence."""
        reason = None
        delta = 0
        if not start:
            reason = "no_request_snapshot"
        else:
            delta = self.dispatches() - int(start.get("dispatches", 0))
            if not start.get("active") or not self._active:
                reason = "mode_not_active"
            elif start.get("epoch") != self._epoch:
                reason = "mode_changed_during_request"
            elif not bool(self._metal.qmv_bitexact()):
                reason = "mlx_mode_off"
            elif delta <= 0:
                reason = "no_bitexact_dispatch_observed"
        ok = reason is None
        self.counts["receipts_true" if ok else "receipts_false"] += 1
        return {
            "schema": SCHEMA,
            "verify_bitexact": ok,
            "reason": reason,
            "scope": SCOPE,
            "max_m": self.max_m,
            "requested": bool(start and start.get("explicit")),
            # Process-global counter: concurrent requests share it.  The claim
            # rests on the mode staying on for this request's lifetime; the
            # delta shows the route engaged at all.
            "dispatches_during_request": max(0, int(delta)),
            "residual_width_dependence": list(RESIDUAL_WIDTH_DEPENDENCE),
        }

    def status(self) -> dict:
        return {
            "schema": SCHEMA,
            "enabled": True,
            "active": self._active,
            "scope": SCOPE,
            "max_m": self.max_m,
            "dispatches": self.dispatches(),
            "counts": dict(self.counts),
            "residual_width_dependence": list(RESIDUAL_WIDTH_DEPENDENCE),
        }


def bind_for_serving(policy: VerifyBitexactPolicy, *, mx_module=None):
    """Turn the mode on for this process, or fail closed.

    Returns ``(handle, settings)``. ``settings`` goes into the qualification
    identity, so a bit-exact qualification never binds a server that is not
    bit-exact.
    """
    if not policy.enabled:
        return None, None
    probe = capability(mx_module)
    if not probe["available"]:
        raise VerifyBitexactUnavailable(
            "--verify-bitexact requires an mlx build with the bit-exact qmv mode "
            f"({probe['reason']}; missing {probe['missing_api']})"
        )
    handle = VerifyBitexactHandle(_metal(mx_module), max_m=probe["max_m"])
    handle.activate()
    settings = {
        "enabled": True,
        "schema": SCHEMA,
        "scope": SCOPE,
        "max_m": handle.max_m,
    }
    return handle, settings


def engine_status(engine: Any) -> dict:
    handle = getattr(engine, "verify_bitexact_handle", None)
    policy = getattr(engine, "verify_bitexact_policy", None)
    if handle is None:
        return {
            "schema": SCHEMA,
            "enabled": bool(policy is not None and policy.enabled),
            "active": False,
        }
    return handle.status()


def request_wants_bitexact(request) -> bool:
    return bool((request or {}).get("verify_bitexact", False))


def check_request(request, handle) -> None:
    """Reject, rather than silently degrade, a request asking for bit-exact."""
    if request_wants_bitexact(request) and (handle is None or not handle.active):
        raise VerifyBitexactUnavailable(
            "verify_bitexact requires a server started with --verify-bitexact "
            "on an mlx build with the bit-exact qmv mode"
        )
