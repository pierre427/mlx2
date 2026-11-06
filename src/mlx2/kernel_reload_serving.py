"""Worker-owned developer kernel changes over a resident model.

The scheduler owns draining; this boundary owns cache invalidation and receipts.
It deliberately contains no model-family decisions or tensor mathematics.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from contextlib import contextmanager
from functools import wraps

from .host_counters import count_add


def _rejection(error, reason):
    error.kernel_reload_rejection = reason
    return error


def registered_engine_constructor(initialize):
    """Register before adapter resolution can import shared tensor modules."""
    @wraps(initialize)
    def construct(self, *args, **kwargs):
        if kwargs.get("_validate_only", False):
            return initialize(self, *args, **kwargs)
        from .kernel_reload import register_owner, unregister_owner

        register_owner(self)
        try:
            return initialize(self, *args, **kwargs)
        except BaseException:
            unregister_owner(self)
            raise
    return construct


class KernelWeightSpaceConflict(RuntimeError):
    """Unsupported execution-epoch composition, before any model/cache change."""

    code = "kernel_weight_space_incompatible"
    status = 409

    def __init__(self, operation, conflicts):
        super().__init__("kernel reload and weight-space execution cannot compose; "
                         "restart with one execution mode")
        self.receipt = {
            "schema": "mlx2.runtime-compatibility.v1", "operation": operation,
            "status": "refused", "reason": self.code,
            "conflicts": list(conflicts), "qualified": False,
            "model_changed": False, "cache_changed": False,
        }


class KernelReloadMixin:
    def _weight_space_lifecycle_lock(self):
        lifecycle = getattr(self, "_lora_lifecycle_lock", None)
        if lifecycle is None:
            lifecycle = self._lora_lifecycle_lock = threading.RLock()
        return lifecycle

    def _weight_space_conflict(self, operation, conflicts):
        from .runtime.weight_space_metrics import record_weight_space

        error = KernelWeightSpaceConflict(operation, conflicts)
        receipts = getattr(self, "operation_receipts", None)
        if receipts is not None:
            receipts.append(error.receipt)
        record_weight_space("runtime_compatibility", "rejected")
        return error

    def _kernel_weight_space_conflicts(self):
        conflicts = []
        if getattr(self, "weight_space_live", None) is not None:
            conflicts.append("live_weight_space")
        if getattr(getattr(self, "lora_session", None), "weight_program", None) is not None:
            conflicts.append("global_weight_program")
        manager = getattr(self, "multi_lora", None)
        if manager is not None:
            # Registration is serialized by the same lifecycle lock as reload.
            registry = getattr(manager, "registry", {})
            if any(getattr(adapter, "weight_program", None) is not None
                   for adapter in tuple(registry.values())):
                conflicts.append("registered_weight_program")
        if getattr(self, "_weight_program_deployment", {}).get("default_adapter") is not None:
            conflicts.append("weight_program_default")
        conflicts.extend(name for name in ("weight_overlays", "layer_excision", "expert_masks",
                                           "ablation_overlays")
                         if getattr(self, name, None) is not None)
        return conflicts

    def _check_kernel_weight_space_compatibility(self):
        conflicts = self._kernel_weight_space_conflicts()
        if conflicts:
            raise _rejection(self._weight_space_conflict("kernel_reload", conflicts),
                             "weight_space_conflict")

    def _check_weight_space_kernel_compatibility(self, operation="weight_space_bind"):
        phase = getattr(self, "_kernel_reload_phase", "idle")
        if (getattr(self, "_kernel_reload_identity", None) is not None
                or getattr(self, "_kernel_reload_epoch", 0) != 0
                or getattr(self, "_kernel_reload_request", None) is not None
                or phase != "idle"
                or getattr(self, "_service_state", None) == "kernel_reload_failed"):
            raise self._weight_space_conflict(operation, ["kernel_reload_execution"])

    @contextmanager
    def _weight_space_control_section(self):
        with self._weight_space_lifecycle_lock():
            self._check_weight_space_kernel_compatibility()
            yield

    def _init_kernel_reload(self, root):
        from .kernel_reload import KernelReloader

        self._kernel_reloader = KernelReloader(root) if root is not None else None
        self._kernel_reload_mutex = threading.Lock()
        self._kernel_reload_request = None
        self._kernel_reload_epoch = 0
        self._kernel_reload_identity = None
        self._kernel_reload_last = None
        self._kernel_reload_rebind = None
        self._kernel_reload_phase = "idle"

    def kernel_reload_status(self):
        reloader = getattr(self, "_kernel_reloader", None)
        return {
            "enabled": reloader is not None,
            "epoch": getattr(self, "_kernel_reload_epoch", 0),
            "execution_identity": getattr(self, "_kernel_reload_identity", None),
            "pending": getattr(self, "_kernel_reload_request", None) is not None,
            "phase": getattr(self, "_kernel_reload_phase", "idle"),
            "last_reload": getattr(self, "_kernel_reload_last", None),
            "weight_space_conflicts": self._kernel_weight_space_conflicts(),
            **(reloader.status() if reloader is not None else {}),
        }

    def _kernel_cache_revision(self, base):
        identity = getattr(self, "_kernel_reload_identity", None)
        return base if identity is None else f"{base}:kernel-dev:{identity}"

    def _kernel_route_receipt(self, receipt):
        identity = getattr(self, "_kernel_reload_identity", None)
        if identity is None and getattr(self, "_service_state", None) != "kernel_reload_failed":
            return receipt
        result = {
            **(receipt if isinstance(receipt, dict) else {"route": receipt}), "qualified": False,
            "qualification": "unqualified", "kernel_reload": {
                "epoch": self._kernel_reload_epoch,
                "execution_identity": identity,
            },
        }
        if "serving_qualified" in result:
            result["serving_qualified"] = False
        if isinstance(result.get("representation"), dict):
            result["representation"] = self._kernel_route_receipt(result["representation"])
        return result

    def _check_kernel_reload_resume(self):
        if getattr(self, "_kernel_reload_request", None) is not None:
            raise RuntimeError("kernel reload owns the drain; wait for its receipt")
        if getattr(self, "_service_state", None) == "kernel_reload_failed":
            raise RuntimeError("kernel reload failed; restart this development process")

    def _kernel_reload_timing(self, stage, seconds):
        count_add(self.counts, f"kernel_reload_{stage}_seconds_sum", seconds)
        count_add(self.counts, f"kernel_reload_{stage}_seconds_count", 1)

    def reload_kernels(self, *, modules=(), settings=(), calls=(), drain_timeout_seconds=120):
        started = time.monotonic()
        count_add(self.counts, "kernel_reload_requests", 1)
        try:
            return self._reload_kernels(
                modules=modules, settings=settings, calls=calls,
                drain_timeout_seconds=drain_timeout_seconds,
            )
        except BaseException as error:
            if not getattr(error, "kernel_reload_recorded", False):
                count_add(self.counts, "kernel_reload_rejected", 1)
                count_add(self.counts, "kernel_reload_rejected_" + getattr(error, "kernel_reload_rejection", "other"), 1)
            raise
        finally:
            self._kernel_reload_timing("total", time.monotonic() - started)

    def _reload_kernels(self, *, modules=(), settings=(), calls=(), drain_timeout_seconds=120):
        """Compile a plan, drain, and wait for the evaluation worker's receipt."""
        from .kernel_reload import assert_exclusive_owner

        reloader = getattr(self, "_kernel_reloader", None)
        if reloader is None:
            raise _rejection(RuntimeError("kernel reload is disabled; configure --kernel-dev-root"), "disabled")
        if (isinstance(drain_timeout_seconds, bool)
                or not isinstance(drain_timeout_seconds, (int, float))
                or not math.isfinite(drain_timeout_seconds)
                or not 0.1 <= drain_timeout_seconds <= 3600):
            raise _rejection(ValueError("drain_timeout_seconds must be between 0.1 and 3600"), "invalid_request")
        if not self._kernel_reload_mutex.acquire(blocking=False):
            raise _rejection(RuntimeError("a kernel reload is already pending"), "busy")
        try:
            if (not self.ready.is_set() or self.error or not self.thread.is_alive()
                    or self._worker_stopped or self.stop_event.is_set()):
                unavailable = RuntimeError(self.error or "model is not ready")
                unavailable.status = 503
                raise _rejection(unavailable, "not_ready")
            try:
                assert_exclusive_owner(self)
            except RuntimeError as error:
                raise _rejection(error, "owner_conflict")
            # Serialize against registration and live binding through admission
            # closure. Release before waiting: the worker must own its drain.
            with self._weight_space_lifecycle_lock():
                # Syntax/path validation never executes edited source and never
                # closes admission or clears caches on an invalid request.
                self._check_kernel_weight_space_compatibility()
                if self._service_state == "kernel_reload_failed":
                    raise _rejection(RuntimeError("kernel reload requires a serving, healthy model"), "not_ready")
                self._kernel_reload_phase = "preparing"
                prepare_started = time.monotonic()
                try:
                    plan = reloader.prepare(modules=modules, settings=settings, calls=calls, adapter=self.adapter)
                except Exception as error:  # noqa: BLE001 - Annotate and re-raise the original preparation failure.
                    raise _rejection(error, "prepare")
                finally:
                    self._kernel_reload_timing("prepare", time.monotonic() - prepare_started)
                request = {"plan": plan, "event": threading.Event(), "result": None,
                           "error": None, "started": time.monotonic()}
                with self._admission_section(), self.lock:
                    self._check_kernel_weight_space_compatibility()
                    if self._worker_stopped or self.stop_event.is_set():
                        unavailable = RuntimeError(self.error or "generation worker stopped")
                        unavailable.status = 503
                        raise _rejection(unavailable, "not_ready")
                    if self._service_state != "serving":
                        raise _rejection(RuntimeError("kernel reload requires a serving, healthy model"), "not_ready")
                    self._kernel_reload_request = request
                    self._kernel_reload_phase = "draining"
                    self._drain_started_monotonic = time.monotonic()
                    self._drain_deadline = self._drain_started_monotonic + drain_timeout_seconds
                    self._drain_suspend = self._drain_park_model = False
                    self._quiesce_worker_owned = False
                    self._quiesce_complete.clear()
                    self._transition_service_locked("draining", {
                        "operation": "kernel_reload", "status": "accepted",
                        "drain_timeout_seconds": drain_timeout_seconds,
                    })
            while not request["event"].wait(0.1):
                if not self.thread.is_alive():
                    self._cancel_pending_kernel_reload(request, self.error or "worker stopped during kernel reload")
                elif self.stop_event.is_set():
                    # A claimed operation must settle on its evaluation owner;
                    # it cannot be both rejected here and completed there.
                    self._cancel_pending_kernel_reload(request, "worker stopped before kernel reload", only_unclaimed=True)
            if request["error"] is not None:
                raise request["error"]
            return request["result"]
        finally:
            if self._kernel_reload_request is None:
                # A refused request must not leave "preparing" behind; a
                # quarantined engine keeps reporting its failed phase.
                self._kernel_reload_phase = "failed" if self._service_state == "kernel_reload_failed" else "idle"
            self._kernel_reload_mutex.release()

    def _cancel_pending_kernel_reload(self, request=None, reason="worker stopped during kernel reload", *, only_unclaimed=False):
        """Settle an unexecuted change without touching model or cache state."""
        with self.lock:
            pending = getattr(self, "_kernel_reload_request", None)
            if pending is None or (request is not None and pending is not request):
                return
            if only_unclaimed and self._quiesce_worker_owned:
                return
            # Only a worker-claimed drain can have started purging or applying.
            claimed = self._quiesce_worker_owned
            error = RuntimeError(reason)
            error.kernel_reload_recorded = True
            if not claimed:
                # Nothing ran; the engine is unavailable, as for generation.
                error.status = 503
            pending["error"] = error
            receipt = {
                "schema": "mlx2.kernel-reload.v1", "operation": "kernel_reload",
                "status": "failed" if claimed else "rejected", "error": reason,
                "epoch": self._kernel_reload_epoch, "apcv2": None,
                "weights_reloaded": None, "qualified": False,
                "seconds": time.monotonic() - pending["started"],
            }
            if claimed:
                receipt["failed_stage"] = "other"
            else:
                receipt["rejection"] = "worker_stopped"
                receipt["code_executed"] = False
                receipt["apcv2_purged"] = False
            self._kernel_reload_last = receipt
            self.operation_receipts.append(receipt)
            if claimed:
                count_add(self.counts, "kernel_reload_failed", 1)
                count_add(self.counts, "kernel_reload_failed_other", 1)
            else:
                count_add(self.counts, "kernel_reload_rejected", 1)
                count_add(self.counts, "kernel_reload_rejected_worker_stopped", 1)
            self._kernel_reload_request = None
            self._kernel_reload_phase = "failed" if claimed else "idle"
            # Unclaimed, kernels and caches are unchanged, but the worker that
            # would resume serving has stopped: report what any stopped
            # engine reports, not quarantine and not "serving".
            self._transition_service_locked("kernel_reload_failed" if claimed else "stopped", receipt)
            self._drain_deadline = self._drain_started_monotonic = None
            self._drain_suspend = self._drain_park_model = False
            self._quiesce_worker_owned = False
            self._quiesce_complete.set()
            pending["event"].set()

    def _worker_reload_kernels(self, action, mx, batch=None):
        """Consume an already claimed drain on the model evaluation thread."""
        request = getattr(self, "_kernel_reload_request", None)
        if request is None:
            return False
        from .kernel_reload import exclusive_owner

        receipt = {
            "schema": "mlx2.kernel-reload.v1", "operation": "kernel_reload",
            "status": "failed", "artifact": dict(getattr(self.adapter, "identity", {})),
            "source_runtime": self.snapshot.get("runtime"),
            "weights_reloaded": None, "qualified": False,
            "apcv2": None, "epoch": self._kernel_reload_epoch,
            "timings": {},
        }
        receipt["timings"]["drain"] = action.get("drain_duration_seconds", time.monotonic() - request["started"])
        self._kernel_reload_timing("drain", receipt["timings"]["drain"])
        target = "kernel_reload_failed"
        try:
            if action["timed_out"]:
                target = "serving"
                receipt["status"] = "drain_timeout"
                raise TimeoutError("kernel reload could not drain; kernels and caches are unchanged")
            verify = getattr(self._kernel_reloader, "verify_sources", None)
            if callable(verify):
                # An edit saved during the drain invalidates the plan before
                # any code executes or cache is purged: refuse and keep serving.
                target, receipt["status"] = "serving", "rejected"
                verify(request["plan"])
                target, receipt["status"] = "kernel_reload_failed", "failed"
            # This also blocks a concurrent engine construction from loading
            # modules while their implementations are being replaced.
            with exclusive_owner(self), self.submission_lock, self.prompt_lock:
                mx.synchronize()
                for capsule in tuple(getattr(self, "fanout_capsules", {}).values()):
                    capsule.close()
                getattr(self, "fanout_capsules", {}).clear()
                release = getattr(batch, "release_cache_capsules", None)
                if callable(release):
                    release()
                self._kernel_reload_phase = "purging"
                purge_started = time.monotonic()
                try:
                    purge = self.clear_prefix_caches(reason="kernel_reload")
                finally:
                    receipt["timings"]["purge"] = time.monotonic() - purge_started
                    self._kernel_reload_timing("purge", receipt["timings"]["purge"])
                receipt["apcv2"] = purge
                for key, metric in (("entries", "entries"), ("bytes", "bytes"), ("disk_bytes", "disk_bytes"),
                                    ("sidecars", "sidecars"), ("pinned_parks_lost", "parked")):
                    count_add(self.counts, f"kernel_reload_apc_{metric}_purged", (purge.get("apc") or {}).get(key, 0))
                count_add(self.counts, "kernel_reload_host_prompt_entries_purged", purge.get("host_prompt_cache_entries", 0))
                count_add(self.counts, "kernel_reload_mlx_cache_bytes_released", purge.get("mlx_cache_bytes_released", 0))
                tokenizer_cache = getattr(self, "incremental_tokenizer_cache", None)
                if tokenizer_cache is not None:
                    tokenizer_cache.clear()
                invalidate = getattr(self.adapter, "invalidate_feature_cache", None)
                if callable(invalidate):
                    invalidate()
                self._kernel_reload_phase = "applying"
                apply_started = time.monotonic()
                count_add(self.counts, "kernel_reload_weights_checked", 1)
                try:
                    changed = self._kernel_reloader.apply(request["plan"], adapter=self.adapter)
                finally:
                    receipt["timings"]["apply"] = time.monotonic() - apply_started
                    self._kernel_reload_timing("apply", receipt["timings"]["apply"])
                for key, metric in (("modules", "modules"), ("functions_patched", "functions"),
                                    ("classes_preserved", "classes"), ("caches_invalidated", "caches"),
                                    ("settings", "settings"), ("calls", "setter_calls")):
                    value = changed.get(key, [])
                    count_add(self.counts, f"kernel_reload_{metric}", value if isinstance(value, int) else len(value))
                count_add(self.counts, "kernel_reload_layer_targets", sum(item.get("matched_count", 0) for item in changed.get("calls", []) if item.get("target", "").startswith("layers:")))
                if changed.get("weights_preserved"):
                    count_add(self.counts, "kernel_reload_weights_preserved", 1)
                # Some MLX builds expose this; re-created compiled wrappers in
                # the reloader are still required on builds which do not.
                clear_compile = getattr(mx, "clear_compile_cache", None)
                if callable(clear_compile):
                    clear_compile()
                mx.synchronize()
                mx.clear_cache()
                self._kernel_reload_epoch += 1
                self._kernel_reload_identity = hashlib.sha256(json.dumps({
                    "previous": self._kernel_reload_identity, "change": changed,
                    "nonce": uuid.uuid4().hex,
                }, sort_keys=True, default=str).encode()).hexdigest()
                self.model_revision += 1
                rebind = getattr(self, "_kernel_reload_rebind", None)
                if rebind is not None:
                    rebind()
                previous_handoff = getattr(self, "_prefill_handoff_execution_digest", None)
                if previous_handoff is not None:
                    self._prefill_handoff_execution_digest = hashlib.sha256(
                        (previous_handoff + self._kernel_reload_identity).encode()
                    ).hexdigest()
                self.qualification = None
                self.qualification_state = "unqualified"
                self.adaptive_profile_binding = None
                receipt.update(changed)
                receipt.update(status="completed", epoch=self._kernel_reload_epoch,
                               execution_identity=self._kernel_reload_identity,
                               model_revision=self.model_revision, weights_reloaded=False)
                with self.lock:
                    self.snapshot.update(
                        qualification="unqualified", qualified_capabilities=[],
                        route_receipt=self._kernel_route_receipt(self.snapshot.get("route_receipt")),
                        kernel_reload=receipt,
                    )
                    self.snapshot["runtime"] = {
                        **(self.snapshot.get("runtime") or {}),
                        "kernel_reload_identity": self._kernel_reload_identity,
                    }
                    self.snapshot.pop("adaptive_profile", None)
                target = "serving"
                request["result"] = receipt
        except BaseException as error:  # noqa: BLE001 - Developer code must not bypass quarantine via SystemExit.
            receipt["error"] = f"{type(error).__name__}: {error}"
            if target == "kernel_reload_failed" and getattr(error, "code_executed", None) is False:
                # apply() refused its plan before executing any source (an
                # edit saved during synchronization or the purge). Kernels,
                # weights and revision are unchanged; only the purged caches
                # are gone, which the receipt's apcv2 entry records.
                target, receipt["status"] = "serving", "rejected"
            if receipt["status"] == "rejected":
                receipt["code_executed"] = False
                receipt["apcv2_purged"] = receipt["apcv2"] is not None
            request["error"] = error if isinstance(error, Exception) else RuntimeError(receipt["error"])
            request["error"].kernel_reload_recorded = True
            if getattr(error, "weights_preserved", None) is False:
                count_add(self.counts, "kernel_reload_weights_failed", 1)
            if receipt["status"] == "rejected":
                receipt["rejection"] = "prepare"
                count_add(self.counts, "kernel_reload_rejected_prepare", 1)
            if target == "kernel_reload_failed":
                stage = {"purging": "purge", "applying": "apply"}.get(self._kernel_reload_phase, "other")
                count_add(self.counts, "kernel_reload_failed_" + stage, 1)
                receipt["failed_stage"] = stage
                self.qualification = None
                self.qualification_state = "unqualified"
                self.snapshot.update(qualification="unqualified", qualified_capabilities=[])
        finally:
            receipt["seconds"] = time.monotonic() - request["started"]
            self._kernel_reload_last = receipt
            self.operation_receipts.append(receipt)
            count_add(self.counts, "kernel_reload_" + receipt["status"], 1)
            with self.lock:
                self._kernel_reload_request = None
                self._kernel_reload_phase = "failed" if target == "kernel_reload_failed" else "idle"
                self._transition_service_locked(target, receipt)
                if target == "kernel_reload_failed":
                    self.snapshot["route_receipt"] = self._kernel_route_receipt(self.snapshot.get("route_receipt"))
                self._drain_deadline = self._drain_started_monotonic = None
                self._drain_suspend = self._drain_park_model = False
                self._quiesce_worker_owned = False
                self._quiesce_complete.set()
            request["event"].set()
        return True
