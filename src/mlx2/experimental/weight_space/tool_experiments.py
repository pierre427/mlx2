"""Host-bound experimental tools; identifiers never confer source or execution authority."""

from __future__ import annotations

import hashlib
import re
import time
from functools import wraps
from types import MappingProxyType

import numpy as np

from .excision import ExcisionCapture, LayerExcisionPlan, fit_boundary_healing
from .experts import (
    ExpertKey,
    ExpertMaskPlan,
    ExpertRecord,
    ExpertResidencyController,
    plan_residency_budget,
    select_expert_mask,
)
from .memory import (
    CalibrationSplit,
    CapturedActivations,
    MemorySource,
    calibrate_memory_program,
)
from .overlays import (
    QuantizationSpec,
    TensorTarget,
    compare_quality,
    teacher_forced_quality,
)


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value):
        raise ValueError("expected a bounded registered identifier")
    return value


def _register(registry, name, value):
    _id(name)
    if name in registry or len(registry) >= 128:
        raise ValueError("registered identifier exists or registry is full")
    registry[name] = value


def _lookup(registry, name):
    _id(name)
    if name not in registry:
        raise ValueError("unknown host-registered identifier")
    return registry[name]


def _target_id(target):
    expert = "dense" if target.expert is None else f"E{target.expert}"
    name = f"L{target.layer}.{expert}.{target.projection}"
    return name if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", name) else "T" + hashlib.sha256(name.encode()).hexdigest()


def _serialized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


def _fit_workspace(rows, input_width, output_width, total_rows, rank):
    # Thin SVD + QR, scoring/residual copies and the small rank-space SVD.
    # Conservative host admission only; vendor LAPACK scratch is not measured.
    narrow = min(rows, input_width)
    return int(8 * (4 * rows * input_width + 4 * narrow * (input_width + output_width)
                    + 16 * narrow * narrow + 4 * total_rows * output_width
                    + 4 * rank * (input_width + output_width)) + 128 * 1024)


class ExperimentTools:
    """Mixin initialized by WeightSpaceToolService, using its session and lock.

    Register/attach methods are host-only and deliberately absent from MCP. Each
    exposed operation refuses if its host source/provider capability is absent.
    """

    def _init_experiments(self):
        self._overlay_manager = None
        self._overlay_targets = {}
        self._expert_sources, self._expert_plans = {}, {}
        self._expert_mask_manager, self._expert_runtime_names = None, {}
        self._excision_inventories, self._excision_captures, self._excision_plans = {}, {}, {}
        self._excision_manager = None
        self._memory_calibrations, self._memory_publications = {}, {}
        self._quality_captures = {}

    def experiment_capabilities(self):
        with self._lock:
            return {
                "quantization_overlays_registered": self._overlay_manager is not None,
                "quantization_target_count": len(self._overlay_targets),
                "expert_source_ids": sorted(self._expert_sources),
                "expert_plan_ids": sorted(self._expert_plans),
                "expert_mask_runtime_registered": self._expert_mask_manager is not None,
                "excision_inventory_ids": sorted(self._excision_inventories),
                "excision_capture_ids": sorted(self._excision_captures),
                "excision_plan_names": sorted(self._excision_plans),
                "excision_runtime_registered": self._excision_manager is not None,
                "memory_calibration_ids": sorted(self._memory_calibrations),
                "quality_capture_ids": sorted(self._quality_captures),
                "candidate_creation_changes_default": False,
                "experiment_programs_qualified": False,
            }

    @_serialized
    def attach_experiment_hosts(self, *, overlay_manager=None, excision_manager=None, expert_mask_manager=None):
        """Host attachment convenience; callers cannot invoke this as a tool."""
        if overlay_manager is not None:
            self.attach_overlay_manager(overlay_manager)
        if excision_manager is not None:
            self.attach_excision_manager(excision_manager)
        if expert_mask_manager is not None:
            self.attach_expert_mask_manager(expert_mask_manager)

    @_serialized
    def attach_overlay_manager(self, manager):
        """Host-only adapter capability, with an already configured bounded bank."""
        if (getattr(manager, "bank", None) is None or not callable(getattr(manager, "source", None))
                or not callable(getattr(manager, "source_shape", None))
                or not callable(getattr(manager.bank, "check_capacity", None))
                or getattr(manager, "checkpoint", None) != manager.bank.checkpoint):
            raise ValueError("overlay manager requires bounded source geometry/provider and matching configured bank")
        targets = {_target_id(target): target for target in manager.bindings}
        if (not targets or len(targets) > 100000
                or any(not isinstance(target, TensorTarget) for target in targets.values())):
            raise ValueError("overlay targets must be a bounded host binding inventory")
        if len(targets) != len(manager.bindings):
            raise ValueError("overlay target identifiers collide")
        with self._lock:
            if self._overlay_manager is not None:
                raise ValueError("overlay manager cannot be replaced")
            self._overlay_manager, self._overlay_targets = manager, MappingProxyType(targets)

    def _overlay(self):
        if self._overlay_manager is None:
            raise RuntimeError("host has not registered a quantization overlay provider")
        return self._overlay_manager

    def inspect_quantization_targets(self, *, offset=0, limit=32):
        self._overlay()
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 64:
            raise ValueError("target page requires nonnegative offset and limit 1..64")
        names = sorted(self._overlay_targets)
        page = names[offset:offset + limit]
        return {"targets": [{"target_id": name, **self._overlay_targets[name].to_dict()} for name in page],
                "next_offset": offset + limit if offset + limit < len(names) else None,
                "total": len(names), "materialized_weights": False}

    @_serialized
    def create_quantization_overlay(self, name, target_id, *, bits=4, format="affine_uniform", group_size=64):
        manager = self._overlay()
        _id(name)
        if target_id not in self._overlay_targets:
            raise ValueError("unknown host-registered tensor target")
        spec = QuantizationSpec(bits, group_size, format)
        target = self._overlay_targets[target_id]
        manager.bank.check_capacity(name, manager.source_shape(target), spec)
        source = manager.source(target)
        receipt = manager.bank.create(name, target, source["weights"], source_format=source["source_format"],
                                      spec=spec, source_tensor_fingerprint=source["source_tensor_fingerprint"])
        return {**receipt, "request_selection": {"weight_overlay": name},
                "default_changed": False, "task_quality_evaluated": False}

    def inspect_quantization_overlay(self, name=None):
        manager = self._overlay()
        return manager.bank.snapshot() if name is None else manager.bank.inspect(_id(name))

    @_serialized
    def compose_quantization_overlays(self, name, names):
        if not isinstance(names, list) or not 1 <= len(names) <= 32:
            raise ValueError("compose requires 1..32 registered overlay names")
        for item in names:
            _id(item)
        receipt = self._overlay().bank.compose(_id(name), names)
        return {**receipt, "request_selection": {"weight_overlay": name}, "default_changed": False}

    @_serialized
    def delete_quantization_overlay(self, name):
        self._overlay().bank.delete(_id(name))
        return {"name": name, "deleted": True, "checkpoint_weights_changed": False}

    @_serialized
    def register_expert_source(self, source_id, *, plan, importance, records=(), controller=None):
        if not isinstance(plan, ExpertMaskPlan):
            raise TypeError("host expert source requires ExpertMaskPlan")
        plan = ExpertMaskPlan.from_dict(plan.to_dict())
        if set(importance) != {layer.layer for layer in plan.layers}:
            raise ValueError("importance must cover every registered layer")
        scores = {}
        for layer in plan.layers:
            value = np.asarray(importance[layer.layer])
            if value.shape != (layer.num_experts,) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError("host importance geometry or finite values invalid")
            with np.errstate(over="ignore"):
                converted = np.asarray(value, dtype="<f8")
            if not np.isfinite(converted).all():
                raise ValueError("importance scores overflow float64")
            scores[layer.layer] = np.frombuffer(converted.tobytes(), dtype="<f8")
        records = tuple(records)
        if len(records) > 65536 or any(not isinstance(record, ExpertRecord) for record in records):
            raise ValueError("host records must be bounded ExpertRecord values")
        if controller is not None and (not isinstance(controller, ExpertResidencyController)
                                       or controller.plan.plan_id != plan.plan_id):
            raise ValueError("host controller must bind the exact registered plan")
        with self._lock:
            self._check_registry_bytes(sum(array.nbytes for array in scores.values()))
            _register(self._expert_sources, source_id,
                      {"plan": plan, "importance": MappingProxyType(scores), "records": records, "controller": controller})
            self._registered_bytes += sum(array.nbytes for array in scores.values())

    @_serialized
    def attach_expert_mask_manager(self, manager):
        if not callable(getattr(manager, "register", None)) or not callable(getattr(manager, "describe", None)):
            raise TypeError("host expert mask manager registration and description required")
        if self._expert_mask_manager is not None:
            raise ValueError("expert mask manager cannot be replaced")
        self._expert_mask_manager = manager

    def inspect_expert_mask_runtime(self):
        if self._expert_mask_manager is None:
            raise RuntimeError("host has not registered an inference expert-mask manager")
        return self._expert_mask_manager.describe()

    def inspect_expert_source(self, source_id):
        source = _lookup(self._expert_sources, source_id)
        controller = source["controller"]
        return {"source_id": source_id, "plan": source["plan"].to_dict(),
                "indexed_records": len(source["records"]), "stream_controller_registered": controller is not None,
                "residency": controller.receipt() if controller else None,
                "importance_metric": "host_supplied_calibration_scores_not_a_quality_verdict"}

    @_serialized
    def compile_expert_mask(self, source_id, keep_counts):
        source = _lookup(self._expert_sources, source_id)
        if not isinstance(keep_counts, dict) or len(keep_counts) > 256:
            raise ValueError("keep_counts must map original layer IDs to integer counts")
        counts = {}
        for key, value in keep_counts.items():
            if not isinstance(key, str) or not key.isascii() or not key.isdecimal() or str(int(key)) != key:
                raise ValueError("layer keys must be canonical decimal original IDs")
            counts[int(key)] = value
        base = source["plan"]
        geometries = {layer.layer: {"num_experts": layer.num_experts, "top_k": layer.top_k,
                                    "normalize": layer.normalize, "score_scale": layer.score_scale}
                      for layer in base.layers}
        candidate = select_expert_mask(importance=dict(source["importance"]), keep_counts=counts,
                                       geometries=geometries, checkpoint=base.checkpoint,
                                       source_revision=base.source_revision, source_fingerprint=base.source_fingerprint,
                                       calibration_digest=base.calibration_digest,
                                       quantization_profile=base.quantization_profile,
                                       execution_profile=base.execution_profile,
                                       overlay_fingerprint=base.overlay_fingerprint, tie_policy=base.tie_policy)
        with self._lock:
            if candidate.plan_id not in self._expert_plans and len(self._expert_plans) >= 128:
                raise ValueError("expert plan registry is full")
            if self._expert_mask_manager is not None and candidate.plan_id not in self._expert_runtime_names:
                name = "mask_" + candidate.plan_id.removeprefix("sha256:")
                self._expert_mask_manager.register(name, candidate)
                self._expert_runtime_names[candidate.plan_id] = name
            if candidate.plan_id not in self._expert_plans:
                _register(self._expert_plans, candidate.plan_id, (source_id, candidate))
        return {"source_id": source_id, "plan": candidate.to_dict(), "default_changed": False,
                "runtime_mask_selected": False, "task_quality_evaluated": False,
                "runtime_registered": candidate.plan_id in self._expert_runtime_names,
                "request_selection": {"expert_mask": self._expert_runtime_names[candidate.plan_id]}
                if candidate.plan_id in self._expert_runtime_names else None}

    def plan_expert_residency(self, source_id, plan_id, *, memory_budget_bytes,
                             nonexpert_reserved_bytes, chunk_bytes, overlay_bytes=0,
                             maximum_resident_bytes=None):
        source = _lookup(self._expert_sources, source_id)
        owner, candidate = _lookup(self._expert_plans, plan_id)
        if owner != source_id:
            raise ValueError("expert plan belongs to a different registered source")
        return plan_residency_budget(plan=candidate, records=source["records"],
                                     memory_budget_bytes=memory_budget_bytes,
                                     nonexpert_reserved_bytes=nonexpert_reserved_bytes,
                                     chunk_bytes=chunk_bytes, overlay_bytes=overlay_bytes,
                                     maximum_resident_bytes=maximum_resident_bytes)

    def _expert_controller(self, source_id):
        controller = _lookup(self._expert_sources, source_id)["controller"]
        if controller is None:
            raise RuntimeError("host has not registered an expert byte reader/controller")
        return controller

    @_serialized
    def prefetch_experts(self, source_id, experts):
        if not isinstance(experts, list) or not 1 <= len(experts) <= 32:
            raise ValueError("prefetch requires 1..32 indexed expert keys")
        keys = []
        for value in experts:
            if not isinstance(value, dict) or set(value) - {"layer", "expert", "projection"} or not {"layer", "expert"} <= set(value):
                raise ValueError("expert key needs layer/expert and optional projection")
            keys.append(ExpertKey(**value))
        controller = self._expert_controller(source_id)
        # Validate the whole batch before queue mutation.
        for key in keys:
            controller._record(key)
        return {"states": [controller.prefetch(key) for key in keys],
                "source_id": source_id, "reads_started": False}

    @_serialized
    def advance_expert_stream(self, source_id, *, max_chunks=1):
        if type(max_chunks) is not int or not 1 <= max_chunks <= 16:
            raise ValueError("tool IO advancement is limited to 1..16 chunks")
        controller = self._expert_controller(source_id)
        return {"progress": controller.step(max_chunks=max_chunks), "residency": controller.receipt()}

    @_serialized
    def register_excision_inventory(self, inventory_id, *, checkpoint, source_revision, layer_count, hidden_size):
        # Constructing one legal interior mask validates source and geometry.
        validated = LayerExcisionPlan(checkpoint, source_revision, layer_count, (1,), hidden_size)
        with self._lock:
            _register(self._excision_inventories, inventory_id,
                      {"checkpoint": validated.checkpoint, "source_revision": validated.source_revision,
                       "layer_count": validated.layer_count, "hidden_size": validated.hidden_size})

    @_serialized
    def register_excision_capture(self, capture_id, capture, split):
        if not isinstance(capture, ExcisionCapture) or not isinstance(split, CalibrationSplit):
            raise TypeError("host excision capture and calibration split required")
        with self._lock:
            size = capture.excised_outputs.nbytes + capture.reference_outputs.nbytes
            self._check_registry_bytes(size)
            _register(self._excision_captures, capture_id, (capture, split))
            self._registered_bytes += size

    @_serialized
    def attach_excision_manager(self, manager):
        if not callable(getattr(manager, "register", None)):
            raise TypeError("host excision manager registration capability required")
        with self._lock:
            if self._excision_manager is not None:
                raise ValueError("excision manager cannot be replaced")
            self._excision_manager = manager

    def _register_excision(self, name, candidate):
        with self._lock:
            _id(name)
            if name in self._excision_plans or len(self._excision_plans) >= 128:
                raise ValueError("excision name exists or registry is full")
            size = candidate.healing.nbytes if candidate.healing is not None else 0
            self._check_registry_bytes(size)
            if self._excision_manager is not None:
                self._excision_manager.register(name, candidate)
            self._excision_plans[name] = candidate
            self._registered_bytes += size
        return {**candidate.receipt(), "name": name, "default_changed": False,
                "runtime_registered": self._excision_manager is not None,
                "request_selection": {"layer_excision": name} if self._excision_manager is not None else None}

    @_serialized
    def compile_layer_excision(self, inventory_id, name, removed_layers):
        inventory = _lookup(self._excision_inventories, inventory_id)
        if not isinstance(removed_layers, list) or len(removed_layers) > 256:
            raise ValueError("removed_layers must be a bounded original-index list")
        return self._register_excision(name, LayerExcisionPlan(**inventory, removed_layers=tuple(removed_layers)))

    def inspect_layer_excision(self, name):
        return _lookup(self._excision_plans, name).receipt()

    @_serialized
    def fit_excision_healing(self, name, capture_id, output_name, *, rank=8, ridge=1e-4):
        candidate = _lookup(self._excision_plans, name)
        capture, split = _lookup(self._excision_captures, capture_id)
        if type(rank) is not int or not 1 <= rank <= 128:
            raise ValueError("healing rank must be 1..128")
        _id(output_name)
        if output_name in self._excision_plans or len(self._excision_plans) >= 128:
            raise ValueError("excision name exists or registry is full")
        self._check_registry_bytes(2 * candidate.hidden_size * min(rank, candidate.hidden_size, len(split.calibration_ids)) * 4)
        self._check_analysis_workspace(_fit_workspace(len(split.calibration_ids), candidate.hidden_size,
                                                       candidate.hidden_size, len(capture.sample_ids), rank))
        healed, report = fit_boundary_healing(candidate, capture, split=split, rank=rank, ridge=ridge)
        return {"plan": self._register_excision(output_name, healed), "report": report}

    @_serialized
    def register_memory_calibration(self, capture_id, *, captures, split, source, source_revision,
                                    store=None, context=None, maximum_ttl_seconds=3600):
        if not isinstance(source, MemorySource) or not isinstance(split, CalibrationSplit):
            raise TypeError("host memory source and calibration split required")
        if source.session != self.session_id:
            raise ValueError("memory source session differs from the bound tool session")
        if (store is None) != (context is None):
            raise ValueError("authoritative memory store and context must be paired")
        if context is not None and (context.session, context.tenant, context.model) != (source.session, source.tenant, source.context_model):
            raise ValueError("memory authority context differs from captured source")
        if type(maximum_ttl_seconds) is not int or not 1 <= maximum_ttl_seconds <= 86400:
            raise ValueError("host memory TTL ceiling must be 1..86400 seconds")
        captures = tuple(captures)
        if not 1 <= len(captures) <= 32 or any(not isinstance(c, CapturedActivations) for c in captures):
            raise ValueError("bounded captured activations required")
        with self._lock:
            size = sum(capture.inputs.nbytes + capture.target_deltas.nbytes for capture in captures)
            self._check_registry_bytes(size)
            _register(self._memory_calibrations, capture_id,
                      {"captures": captures, "split": split, "source": source,
                       "source_revision": source_revision, "store": store, "context": context,
                       "maximum_ttl_seconds": maximum_ttl_seconds})
            self._registered_bytes += size

    def inspect_memory_calibration(self, capture_id):
        binding = _lookup(self._memory_calibrations, capture_id)
        return {"capture_id": capture_id, "source": binding["source"].to_dict(),
                "targets": [{"target": c.target, "samples": len(c.sample_ids),
                             "input_shape": list(c.inputs.shape), "delta_shape": list(c.target_deltas.shape)}
                            for c in binding["captures"]],
                "authority_registered": binding["store"] is not None,
                "claim_only_until_authoritative_publish": True,
                "maximum_ttl_seconds": binding["maximum_ttl_seconds"]}

    @_serialized
    def calibrate_memory(self, capture_id, *, rank=8, ridge=1e-4, strength=1.0):
        binding = _lookup(self._memory_calibrations, capture_id)
        if type(rank) is not int or not 1 <= rank <= 128:
            raise ValueError("memory rank must be 1..128")
        self._check_registry_bytes(max((c.inputs.shape[1] + c.target_deltas.shape[1]) *
                                       min(rank, len(binding["split"].calibration_ids), c.inputs.shape[1], c.target_deltas.shape[1]) * 4
                                       for c in binding["captures"]))
        fit_sizes = [_fit_workspace(len(binding["split"].calibration_ids), c.inputs.shape[1],
                                   c.target_deltas.shape[1], len(c.sample_ids), rank)
                     for c in binding["captures"]]
        retained_fits = sum((c.inputs.shape[1] + c.target_deltas.shape[1]) *
                            min(rank, len(binding["split"].calibration_ids), c.inputs.shape[1], c.target_deltas.shape[1]) * 4
                            for c in binding["captures"])
        self._check_analysis_workspace(max(fit_sizes) + retained_fits)
        result = calibrate_memory_program(binding["captures"], split=binding["split"], source=binding["source"],
                                           source_revision=binding["source_revision"], rank=rank, ridge=ridge, strength=strength)
        self.register_program(result.program.program_id, result.program)
        return {"program_id": result.program.program_id, "report": result.report,
                "promotion_scope": "session", "semantic_source_changed": False}

    def _memory_authority(self, capture_id):
        binding = _lookup(self._memory_calibrations, capture_id)
        if binding["store"] is None:
            raise RuntimeError("host has not registered authoritative semantic memory")
        return binding

    @_serialized
    def publish_memory_program(self, capture_id, program_id, *, ttl_seconds=300):
        binding = self._memory_authority(capture_id)
        if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= binding["maximum_ttl_seconds"]:
            raise ValueError("memory TTL exceeds the host ceiling")
        program = self._program(program_id)
        if program.provenance.get("memory_source") != binding["source"].to_dict():
            raise ValueError("memory program belongs to a different source capture")
        receipt = binding["store"].publish(binding["context"], program,
                                            expected_revision=binding["source"].session_revision,
                                            expires_at_ns=time.time_ns() + ttl_seconds * 10**9)
        with self._lock:
            self._memory_publications[capture_id] = receipt
        return receipt

    @_serialized
    def mount_memory_program(self, capture_id):
        binding = self._memory_authority(capture_id)
        receipt = _lookup(self._memory_publications, capture_id)
        mounted = binding["store"].mount(binding["context"], checkpoint=binding["source"].checkpoint,
                                          expected_directory_revision=receipt["directory_revision"],
                                          expected_capsule_digest=receipt["capsule_digest"])
        return mounted.receipt

    @_serialized
    def revoke_memory_program(self, capture_id):
        binding = self._memory_authority(capture_id)
        receipt = _lookup(self._memory_publications, capture_id)
        result = binding["store"].revoke(binding["context"], expected_revision=receipt["directory_revision"],
                                         expected_capsule_digest=receipt["capsule_digest"])
        with self._lock:
            del self._memory_publications[capture_id]
        return result

    @_serialized
    def register_quality_capture(self, capture_id, *, reference_logits, candidate_logits,
                                  next_token_ids, tokenizer_fingerprint, input_digest, mask=None):
        arrays = []
        for value in (reference_logits, candidate_logits):
            array = np.asarray(value)
            if (array.ndim != 2 or not 0 < array.size <= 8 * 1024 * 1024
                    or array.dtype.kind not in "fiu" or not np.isfinite(array).all()):
                raise ValueError("quality logits must be bounded finite matrices")
            with np.errstate(over="ignore"):
                copied = np.asarray(array, dtype="<f4")
            if not np.isfinite(copied).all():
                raise ValueError("quality logits overflow float32")
            arrays.append(np.frombuffer(copied.tobytes(), dtype="<f4").reshape(copied.shape))
        if arrays[0].shape != arrays[1].shape:
            raise ValueError("quality arms must have identical token/vocabulary geometry")
        labels = np.asarray(next_token_ids)
        if labels.dtype.kind not in "iu":
            raise ValueError("quality labels must be integer token IDs")
        labels = np.frombuffer(np.asarray(labels, dtype="<i8").tobytes(), dtype="<i8").reshape(labels.shape)
        selected = np.ones(labels.shape, dtype=bool) if mask is None else np.asarray(mask)
        if selected.dtype != np.bool_:
            raise ValueError("quality mask must be boolean")
        selected = np.frombuffer(selected.tobytes(), dtype=np.bool_).reshape(selected.shape)
        # Validate exact alignment before accepting the host-owned capture.
        teacher_forced_quality(arrays[0], labels, tokenizer_fingerprint=tokenizer_fingerprint,
                               input_digest=input_digest, mask=selected)
        digest = hashlib.sha256()
        for array in (*arrays, labels, selected):
            digest.update(array.tobytes())
        with self._lock:
            size = sum(array.nbytes for array in (*arrays, labels, selected))
            self._check_registry_bytes(size)
            _register(self._quality_captures, capture_id,
                      {"reference": arrays[0], "candidate": arrays[1], "labels": labels,
                       "mask": selected, "tokenizer_fingerprint": tokenizer_fingerprint,
                       "input_digest": input_digest, "capture_digest": digest.hexdigest()})
            self._registered_bytes += size

    @_serialized
    def evaluate_quantization_quality(self, capture_id):
        capture = _lookup(self._quality_captures, capture_id)
        self._check_analysis_workspace(capture["reference"].size * 80 + 128 * 1024)
        options = {"tokenizer_fingerprint": capture["tokenizer_fingerprint"],
                   "input_digest": capture["input_digest"], "mask": capture["mask"]}
        reference = teacher_forced_quality(capture["reference"], capture["labels"], **options)
        candidate = teacher_forced_quality(capture["candidate"], capture["labels"], **options)
        return {**compare_quality(reference, candidate), "capture_digest": capture["capture_digest"],
                "evidence_scope": "host_supplied_aligned_logits_only", "model_execution_verified": False,
                "deployment_verdict": False}
