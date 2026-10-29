# SPDX-License-Identifier: Apache-2.0
"""V4.1 payload parking behind the generic quiescent serving transaction.

The model object, tokenizer, source manifest and APCv2 identity survive. Native
weight, expert and PLE payloads do not. No target math or cache layout changes.
"""

import copy
import gc
import threading


def resume_headroom(headroom_bytes):
    """Current execution capacity after the existing service/driver reserves."""
    from ..memory import execution_headroom, host_memory_gib, metal_advisory_gib
    from ..runtime.memory_policy import SelfMTPLaneAdmissionController

    advisory = metal_advisory_gib()
    if advisory is None:
        raise MemoryError("cannot measure model activation advisory")
    service, driver = SelfMTPLaneAdmissionController.host_scaled_reserves(host_memory_gib(), advisory)
    measured = execution_headroom(host_signals=True)
    reserve = int((service+driver)*(1 << 30))+headroom_bytes
    return dict(execution_headroom_bytes=measured, service_driver_workspace_reserve_bytes=reserve,
                available_payload_bytes=max(0, measured-reserve))


class DeepSeekV41Residency:
    def __init__(self, adapter):
        self.adapter = adapter
        self.state = "active"
        self._thread = None
        self._expected = self._runtime_binding()
        self.last_admission = None
        self._synchronize = None
        # Teardown can finish before a quarantined syscall. Keep its exact
        # host owners reachable after adapter.close drops tensor providers.
        stream = adapter.weight_stream
        pages = getattr(adapter.weights, "dense_pages", None)
        pools = [("dense", getattr(pages, "pool", None)),
                 ("experts", getattr(stream, "pool", None)),
                 ("expert_hints", getattr(stream, "hint_pool", None))]
        pools.extend((f"ple_{layer}", getattr(table, "pool", None))
                     for layer, table in adapter.engram_tables.items())
        self._retirement_pools = tuple((name, pool) for name, pool in pools if pool is not None)
        self._retirement_bound_bytes = self.reservation_bytes()

    def retirement_ready(self):
        """A shared reservation survives late reads and descriptor finalizers."""
        return self.retirement_receipt()["ownership_drained"]

    def retirement_receipt(self):
        owners = {name:pool.retirement_receipt() for name,pool in self._retirement_pools}
        drained = all(item.get("ownership_drained") is True for item in owners.values())
        return dict(ownership_drained=drained,
                    retained_bytes=0 if drained else self._retirement_bound_bytes,
                    accounting="conservative admitted reservation; not measured allocation",
                    owners=owners)

    def bind_execution_synchronizer(self, synchronize):
        if not callable(synchronize):
            raise ValueError("model residency requires an execution synchronizer")
        self._owner()
        self._synchronize = synchronize
        self.adapter.weights.bind_dense_synchronizer(synchronize)
        stream = self.adapter.weight_stream
        if stream is not None and stream.elastic is not None:
            stream.bind_execution_synchronizer(synchronize)

    def _owner(self):
        current = threading.get_ident()
        if self._thread is None:
            self._thread = current
        elif current != self._thread:
            raise ValueError("V4.1 residency must run on its evaluation owner")

    def _runtime_binding(self):
        a = self.adapter
        if getattr(a, "model", None) is None or getattr(a, "weights", None) is None:
            raise ValueError("V4.1 residency provider is closed")
        w = a.weights
        weight_keys = ("compute_policy", "expert_prefill_policy", "dense_prefill_policy",
                       "expert_decode_policy", "expert_buffer_policy", "shared_policy")
        weight_keys += tuple(f"expert_{key}_policy" for key in ("small", "reuse", "iq1", "down", "binding", "io", "groups", "simd"))
        weight_keys += ("dense_epilogue_policy", "dense_pair_policy")
        model_keys = ("hc_policy", "qat_policy", "pointwise_policy", "attention_epilogue_policy",
                      "hc_epilogue_policy", "verification_commit_policy", "verify_rows_policy")
        return dict(schema="mlx2.v41-weight-residency.v1", artifact=a.identity["fingerprint"],
                    source_sha256=a.identity["weight_sha256"], path=a.identity["path"],
                    profile=a.profile, cache_layout=a.layout, model_identity=a.model.identity,
                    policy=copy.deepcopy(a.kernel_policy),
                    weights={key:getattr(w,key,None) for key in weight_keys},
                    model={key:getattr(a.model,key,None) for key in model_keys})

    def binding(self):
        a = self.adapter
        current = self._runtime_binding()
        if current != self._expected:
            raise ValueError("V4.1 residency source/profile binding changed")
        a.weights.verify_source()
        artifact = a.artifact_inspector(a.identity["path"])
        if (artifact["identity"]["fingerprint"] != current["artifact"]
                or artifact["weight_stat"] != a.weights._expected_stat):
            raise ValueError("V4.1 prepared artifact changed while parked")
        return copy.deepcopy(current)

    def resident_bytes(self):
        a = self.adapter
        weights = getattr(a, "weights", None)
        if weights is None:
            return 0
        stream = getattr(a, "weight_stream", None)
        total = stream.live_bytes() if stream is not None else 0
        if stream is None or stream.dense_provider is not weights:
            total += weights.resident_bytes()
        covered = stream.ple_tables.values() if stream is not None else ()
        total += sum(len(t._cache)*t.row_bytes for t in a.engram_tables.values() if t not in covered)
        return total

    def reservation_bytes(self):
        a = self.adapter
        stream = a.weight_stream
        total = stream.resume_reservation_bytes() if stream is not None else 0
        if stream is None or stream.dense_provider is not a.weights:
            total += a.weights.resume_reservation_bytes()
        covered = stream.ple_tables.values() if stream is not None else ()
        total += sum(t.resume_reservation_bytes() for t in a.engram_tables.values() if t not in covered)
        # Small model controls, tokenizer/atlas metadata and runtime objects are
        # not tensor-page payload. Keep the original load remainder allowance.
        return total+(256 << 20)

    def admit(self):
        self._owner()
        self.binding()
        request = self.adapter.weight_streaming_request
        workspace = max(5 << 30, request.headroom_bytes if request is not None else 0)
        reading = resume_headroom(workspace)
        required = self.reservation_bytes()
        self.last_admission = {**reading, "required_resume_bytes": required,
                               "accepted": required <= reading["available_payload_bytes"]}
        if not self.last_admission["accepted"]:
            raise MemoryError("V4.1 activation exceeds current host/advisory headroom")
        return dict(self.last_admission)

    def park(self):
        self._owner()
        if getattr(self.adapter, "model", None) is None:
            raise ValueError("V4.1 residency provider is closed")
        if self.state == "parked":
            return self.receipt()
        a = self.adapter
        try:
            if self._synchronize is not None:
                self._synchronize()
            else:
                import mlx.core as mx
                mx.synchronize()
            # Demand windows must be closed before dropping any model binding.
            # Each provider independently owns/drains its in-flight syscalls.
            for table in a.engram_tables.values():
                table.park()
            if a.weight_stream is not None:
                a.weight_stream.park()
            a.model.release_weight_references()
            a.weights.park()
            gc.collect()
            import mlx.core as mx
            mx.clear_cache()
            self.state = "parked"
            if self.resident_bytes():
                raise RuntimeError("V4.1 parked provider retained tensor payload")
        except BaseException:
            self.state = "failed"
            raise
        return self.receipt()

    def activate(self):
        self._owner()
        self.binding()
        if self.state == "active":
            return self.receipt()
        if self.state != "parked":
            raise ValueError("failed V4.1 parking requires a complete reload")
        self.state = "activating"
        a = self.adapter
        try:
            a.weights.activate()
            for table in a.engram_tables.values():
                table.activate()
            if a.weight_stream is not None:
                a.weight_stream.activate()
            a.model.restore_weight_references()
            self.binding()
            self.state = "active"
        except BaseException:
            self.state = "failed"
            raise
        return self.receipt()

    def receipt(self):
        a = self.adapter
        weights = getattr(a, "weights", None)
        return dict(schema="mlx2.v41-weight-residency.v1", state=self.state if weights else "closed",
                    logical_payload_bytes=self.resident_bytes(),
                    dense=weights.dense_receipt() if weights is not None else None,
                    expert=a.weight_stream.receipt() if getattr(a,"weight_stream",None) is not None else None,
                    ple={str(k):t.diagnostics() for k,t in a.engram_tables.items()},
                    admission=self.last_admission, qualification="unqualified",
                    retirement_ready=self.retirement_ready(),
                    retired_io={name:pool.counters() for name,pool in self._retirement_pools},
                    retained="tokenizer, architecture, source manifest and optional prediction metadata; APCv2 owns prefix state")
