"""CPU-only checks for non-authoritative AST physical-target discovery."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from mlx2.runtime.ast_physical_target import (
    PROVIDER_PROTOCOL,
    ASTPhysicalTargetCandidate,
    ASTPhysicalTargetDiscovery,
    ast_serving_admission_receipt,
    discover_ast_physical_target,
    discover_ast_physical_target_from_layout,
)
from mlx2.runtime.flat_segmented_ast import FlatSegmentDescriptor

_SOURCE_REVISION = "88dc275480dd92561788cfa29876a19866b4a065"
_ARTIFACT = "Qwen3.8-27B-oQ4e-mtp@sha256:" + "a" * 64
_SOURCE_PATH = "src/mlx2/runtime/ast_physical_target.py"
_SOURCE_SHA256 = hashlib.sha256(b"reviewed fixture source").hexdigest()
_OWNER_SHA256 = hashlib.sha256(b"owner capability").hexdigest()
_TARGET_SHA256 = hashlib.sha256(b"target capability").hexdigest()


def _request(lengths=(1, 2, 3, 4)):
    descriptor = FlatSegmentDescriptor.from_query_lengths(lengths)
    physical_rows = len(lengths) * max(lengths)
    return ASTPhysicalTargetDiscovery(
        descriptor=descriptor,
        capture_layers=(3, 7),
        fidelity="exact",
        fallback_target_protocol="row-exact-v1",
        artifact_fingerprint="a" * 64,
        fallback_plan=tuple(
            sorted(
                {
                    "backend_mode": "padded",
                    "logical_rows": descriptor.logical_rows,
                    "physical_rows": physical_rows,
                    "padding_rows": physical_rows - descriptor.logical_rows,
                    "max_query_len": descriptor.max_segment_rows,
                }.items()
            )
        ),
    )


def _progressive_batch(model, draft, **kwargs):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    configure_row_exact = kwargs.pop("configure_row_exact", True)
    if configure_row_exact:
        model.configure_target_verify_row_exact(True)
    tile = kwargs.pop("progressive_verification_tile", 2)
    return ExternalDraftBatchGenerator(
        model,
        draft_model=draft,
        binding="a" * 64,
        num_draft=3,
        completion_batch_size=1,
        prefill_step_size=8,
        progressive_verification_tile=tile,
        **kwargs,
    )


class _Provider:
    protocol = PROVIDER_PROTOCOL
    source_revision_token = _SOURCE_REVISION
    source_identity = ((_SOURCE_PATH, _SOURCE_SHA256),)

    def __init__(self, *, mutation=None):
        self.mutation = mutation
        self.calls = 0
        self.admission_calls = 0
        self.revision_checks = 0
        self.identity = {
            "source_revision": _SOURCE_REVISION,
            "build": "mlx-fixture",
            "device": "cpu-fixture",
            "artifact": _ARTIFACT,
        }

    def validate_revision(self):
        self.revision_checks += 1
        if self.mutation == "stale":
            raise RuntimeError("source revision changed")
        if self.mutation == "drift_during_discovery" and self.revision_checks > 1:
            raise RuntimeError("source revision changed")

    def discover(self, request):
        self.calls += 1
        geometry = request.descriptor.geometry_sha256
        if self.mutation == "geometry":
            geometry = "b" * 64
        candidate_identity = dict(self.identity)
        candidate_sources = dict(self.source_identity)
        if self.mutation == "candidate_revision":
            candidate_identity["source_revision"] = "b" * 40
        if self.mutation == "candidate_hash":
            candidate_sources[_SOURCE_PATH] = "b" * 64
        candidate = ASTPhysicalTargetCandidate.create(
            geometry_sha256=geometry,
            capture_layers_sha256=request.capture_layers_sha256,
            fidelity=("near-tie" if self.mutation == "fidelity" else request.fidelity),
            fallback_target_protocol=request.fallback_target_protocol,
            identity=(
                {**candidate_identity, "artifact": "fixture@sha256:" + "b" * 64}
                if self.mutation == "artifact"
                else candidate_identity
            ),
            source_hashes=candidate_sources,
            hard_limits={
                "max_lanes": 4,
                "max_segment_rows": 4,
                "max_total_rows": 12,
            },
            owner_protocol="native-owner-v2",
            native_target_protocol="native-target-v2",
            owner_capability_manifest_sha256=_OWNER_SHA256,
            target_capability_manifest_sha256=_TARGET_SHA256,
            decline_reasons=("serving_state_lineage_bridge_unavailable",),
        )
        if self.mutation == "protocol_drift":
            self.protocol = "ast-physical-target-provider-v2"
        if self.mutation == "source_drift":
            self.source_identity = ((_SOURCE_PATH, "b" * 64),)
        return candidate

    def admit_serving_physical_target(
        self, *, session_scope_sha256, discovery, **_kwargs
    ):
        self.admission_calls += 1
        return ast_serving_admission_receipt(
            discovery=discovery,
            session_scope_sha256=session_scope_sha256,
            source_manifest_sha256=_SOURCE_SHA256,
            build_manifest_sha256=hashlib.sha256(b"fixture build").hexdigest(),
            device_manifest_sha256=hashlib.sha256(b"fixture device").hexdigest(),
            owner_capability_manifest_sha256=_OWNER_SHA256,
            target_capability_manifest_sha256=_TARGET_SHA256,
        )


def test_candidate_receipt_is_exactly_bound_but_cannot_authorize_execution():
    request = _request()
    provider = _Provider()
    receipt = discover_ast_physical_target(
        provider, request, provider_getter=lambda: provider
    )

    assert provider.calls == 1
    assert provider.revision_checks == 2
    assert receipt["geometry_sha256"] == request.descriptor.geometry_sha256
    assert receipt["physical_rows"] == receipt["logical_rows"] == 10
    assert receipt["identity"]["source_revision"] == _SOURCE_REVISION
    assert receipt["source_hashes"] == {_SOURCE_PATH: _SOURCE_SHA256}
    assert receipt["candidate_implemented"] is True
    assert receipt["serving_implemented"] is False
    assert receipt["implemented"] is False
    assert receipt["qualified"] is False
    assert receipt["eligible"] is False
    assert receipt["selected"] is False
    assert receipt["default"] is False
    assert receipt["observed_used"] is False
    assert receipt["admission_manifest_sha256"] is None
    assert receipt["authority"] == "live_admission_required_not_discovery"
    assert receipt["apcv2_authoritative"] is False
    body = {key: value for key, value in receipt.items() if key != "manifest_sha256"}
    assert receipt["manifest_sha256"] == hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    assert "serving_state_lineage_bridge_unavailable" in receipt["decline_reasons"]
    # The complete discovery record remains deterministic, JSON-only metadata.
    assert json.loads(json.dumps(receipt, sort_keys=True)) == receipt


@pytest.mark.parametrize(
    "mutation",
    [
        "stale",
        "drift_during_discovery",
        "geometry",
        "artifact",
        "candidate_revision",
        "candidate_hash",
        "protocol_drift",
        "source_drift",
        "fidelity",
    ],
)
def test_stale_or_geometry_drift_declines_without_candidate_authority(mutation):
    provider = _Provider(mutation=mutation)
    receipt = discover_ast_physical_target(
        provider, _request(), provider_getter=lambda: provider
    )
    assert receipt["implemented"] is False
    assert receipt["eligible"] is False
    assert receipt["selected"] is False
    assert receipt["observed_used"] is False
    assert receipt["decline_reasons"] == ["provider_validation_failed"]


def test_arbitrary_provider_exception_cannot_break_padded_fallback():
    class Broken(_Provider):
        def discover(self, request):
            raise KeyError("missing capability")

    provider = Broken()
    receipt = discover_ast_physical_target(
        provider, _request(), provider_getter=lambda: provider
    )
    assert receipt["selected"] is False
    assert receipt["decline_reasons"] == ["provider_validation_failed"]


def test_provider_replacement_during_discovery_declines():
    holder = {}

    class Replacing(_Provider):
        def discover(self, request):
            candidate = super().discover(request)
            holder["provider"] = _Provider()
            return candidate

    provider = Replacing()
    holder["provider"] = provider
    receipt = discover_ast_physical_target(
        provider, _request(), provider_getter=lambda: holder["provider"]
    )
    assert receipt["decline_reasons"] == ["provider_validation_failed"]
    assert receipt["candidate_implemented"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"capture_layers": ()},
        {"fidelity": "x" * 129},
        {"artifact_fingerprint": "not-a-digest"},
        {"fallback_plan": (("backend_mode", "padded"),) * 5},
    ],
)
def test_invalid_discovery_request_is_bounded_fail_soft(overrides):
    provider = _Provider()
    values = {
        "query_lengths": (1, 2),
        "capture_layers": (3, 7),
        "fidelity": "exact",
        "fallback_target_protocol": "row-exact-v1",
        "artifact_fingerprint": "a" * 64,
        "fallback_plan": tuple(
            sorted(
                {
                    "backend_mode": "padded",
                    "logical_rows": 3,
                    "physical_rows": 4,
                    "padding_rows": 1,
                    "max_query_len": 2,
                }.items()
            )
        ),
    }
    values.update(overrides)
    receipt = discover_ast_physical_target_from_layout(
        provider, provider_getter=lambda: provider, **values
    )
    assert receipt["decline_reasons"] == ["discovery_request_invalid"]
    assert receipt["selected"] is receipt["default"] is False
    assert isinstance(receipt["manifest_sha256"], str)


def test_over_capacity_candidate_declines_and_never_opens_a_transaction():
    request = _request((4, 4, 4, 1))
    provider = _Provider()
    receipt = discover_ast_physical_target(
        provider, request, provider_getter=lambda: provider
    )
    assert provider.calls == 1
    assert receipt["implemented"] is False
    assert receipt["decline_reasons"] == ["provider_validation_failed"]
    assert not hasattr(provider, "begin")


def test_candidate_rejects_self_asserted_or_incomplete_identity():
    request = _request()
    with pytest.raises(ValueError, match="identity fields"):
        ASTPhysicalTargetCandidate.create(
            geometry_sha256=request.descriptor.geometry_sha256,
            capture_layers_sha256=request.capture_layers_sha256,
            fidelity=request.fidelity,
            fallback_target_protocol=request.fallback_target_protocol,
            identity={"artifact": _ARTIFACT},
            source_hashes={_SOURCE_PATH: _SOURCE_SHA256},
            hard_limits={
                "max_lanes": 4,
                "max_segment_rows": 4,
                "max_total_rows": 12,
            },
            owner_protocol="native-owner-v2",
            native_target_protocol="native-target-v2",
            owner_capability_manifest_sha256=_OWNER_SHA256,
            target_capability_manifest_sha256=_TARGET_SHA256,
            decline_reasons=("unavailable",),
        )


def test_default_off_generator_has_no_provider_or_discovery_receipt():
    from test_external_dflash2_cpu import drain, tiny

    model, draft = tiny()
    batch = _progressive_batch(model, draft)
    assert batch.ast_physical_target_provider is None
    uid = batch.insert([[1, 2, 3]], max_tokens=[4])[0]
    _output, final = drain(batch)
    layout = final[uid].speculative_receipt["progressive_verification"][
        "last_ragged_layout"
    ]
    assert layout is not None
    assert "physical_target_discovery" not in layout


def test_installed_provider_is_inert_when_physical_target_is_off():
    from test_external_dflash2_cpu import drain, tiny

    model, draft = tiny()
    provider = _Provider()
    model.ast_physical_target_provider = provider
    batch = _progressive_batch(model, draft)
    assert batch.ast_physical_target_provider is provider
    uid = batch.insert([[1, 2, 3]], max_tokens=[4])[0]
    _output, final = drain(batch)
    layout = final[uid].speculative_receipt["progressive_verification"][
        "last_ragged_layout"
    ]
    assert provider.calls == 0
    assert "physical_target_discovery" not in layout
    assert "physical_target_serving" not in layout
    assert layout["backend_mode"] == "padded"


def test_installed_provider_failure_is_not_consulted_while_feature_is_off():
    from test_external_dflash2_cpu import drain, tiny

    class Broken(_Provider):
        def discover(self, request):
            raise AttributeError("injected incomplete provider")

    model, draft = tiny()
    model.ast_physical_target_provider = Broken()
    batch = _progressive_batch(model, draft)
    uid = batch.insert([[1, 2, 3]], max_tokens=[4])[0]
    output, final = drain(batch)
    assert len(output[uid]) == 4
    layout = final[uid].speculative_receipt["progressive_verification"][
        "last_ragged_layout"
    ]
    assert model.ast_physical_target_provider.calls == 0
    assert "physical_target_discovery" not in layout
    assert layout["backend_mode"] == "padded"


def test_requested_physical_target_requires_selected_progressive_exact_route():
    from test_external_dflash2_cpu import tiny

    model, draft = tiny()
    model.ast_physical_target_provider = _Provider()
    with pytest.raises(
        ValueError, match="requires selected exact progressive verification"
    ):
        _progressive_batch(
            model,
            draft,
            ast_physical_target=True,
            progressive_verification_tile=None,
        )


def test_requested_physical_target_requires_live_adapter_row_exact_executor():
    from test_external_dflash2_cpu import tiny

    model, draft = tiny()
    model.configure_target_verify_row_exact(True)
    model.progressive_external_target_rows = None
    model.ast_physical_target_provider = _Provider()
    with pytest.raises(
        ValueError, match="requires selected exact progressive verification"
    ):
        _progressive_batch(
            model,
            draft,
            ast_physical_target=True,
            configure_row_exact=False,
        )


def _cache_digest(caches):
    rows = []
    for cache in caches:
        values = []
        for value in cache.state:
            if value is None:
                values.append(None)
                continue
            host = np.asarray(value)
            values.append(
                (
                    tuple(host.shape),
                    str(host.dtype),
                    hashlib.sha256(host.tobytes(order="C")).hexdigest(),
                )
            )
        rows.append((type(cache).__name__, int(cache.offset), tuple(values)))
    return tuple(rows)


def _padded_run(provider):
    from test_external_dflash2_cpu import drain, tiny

    model, draft = tiny()
    if provider is not None:
        model.ast_physical_target_provider = provider
    batch = _progressive_batch(model, draft)
    uid = batch.insert([[1, 2, 3]], max_tokens=[4])[0]
    output, final = drain(batch)
    response = final[uid]
    progressive = response.speculative_receipt["progressive_verification"]
    counters = tuple(
        (name, progressive[name])
        for name in (
            "rounds",
            "target_launches",
            "target_rows",
            "padding_rows",
            "ragged_launches",
            "rejections",
            "full_nonfinal_tiles",
            "tile_checkpoints",
        )
    )
    return output[uid], tuple(response.all_tokens), _cache_digest(response.prompt_cache), counters


def test_provider_failure_is_exactly_equivalent_to_provider_absent_execution():
    class Broken(_Provider):
        def discover(self, request):
            raise KeyError("injected discovery failure")

    assert _padded_run(None) == _padded_run(Broken())


def test_feature_off_provider_is_zero_call_and_exactly_padded_equivalent():
    provider = _Provider()
    assert _padded_run(None) == _padded_run(provider)
    assert provider.calls == provider.revision_checks == 0


@pytest.mark.parametrize(
    "failure", ["discovery", "missing_admission", "admission", "missing_bind", "bind"]
)
def test_explicit_physical_target_request_fails_closed_before_padded_execution(
    failure,
):
    from test_external_dflash2_cpu import drain, tiny

    class Requested(_Provider):
        def validate_revision(self):
            if failure == "discovery":
                raise RuntimeError("injected unavailable provenance")
            return super().validate_revision()

        def bind_serving_physical_target(self, **_kwargs):
            if failure == "bind":
                raise RuntimeError("injected binding failure")
            raise AssertionError("unexpected binding")

        def admit_serving_physical_target(self, **kwargs):
            if failure == "admission":
                raise RuntimeError("injected admission failure")
            return super().admit_serving_physical_target(**kwargs)

    provider = Requested()
    if failure == "missing_bind":
        provider.bind_serving_physical_target = None
    if failure == "missing_admission":
        provider.admit_serving_physical_target = None
    model, draft = tiny()
    model.ast_physical_target_provider = provider
    batch = _progressive_batch(model, draft, ast_physical_target=True)
    batch.insert([[1, 2, 3]], max_tokens=[4])
    message = {
        "discovery": "requested AST physical target is unavailable",
        "missing_admission": "requested AST physical target lacks live admission",
        "admission": "requested AST physical target admission or binding failed",
        "missing_bind": "requested AST physical target lacks a serving binding",
        "bind": "requested AST physical target admission or binding failed",
    }[failure]
    with pytest.raises(RuntimeError, match=message):
        drain(batch)


def test_native_execution_failure_aborts_binding_and_advances_ast_memo():
    from test_external_dflash2_cpu import drain, tiny

    class Binding:
        def __init__(self):
            self.aborted = 0

        def execute(self):
            raise RuntimeError("injected native execution failure")

        def abort(self):
            self.aborted += 1

    class Provider(_Provider):
        def __init__(self):
            super().__init__()
            self.binding = Binding()

        def bind_serving_physical_target(self, **_kwargs):
            return self.binding

    class Memo:
        def __init__(self):
            self.advances = []

        def advance(self, scopes, cause):
            self.advances.append((tuple(scopes), cause))

        def register_scope(self, _scope):
            return False

    model, draft = tiny()
    provider = Provider()
    memo = Memo()
    model.ast_physical_target_provider = provider
    batch = _progressive_batch(
        model,
        draft,
        ast_physical_target=True,
        ast_memo=memo,
    )
    batch.insert([[1, 2, 3]], max_tokens=[4])
    with pytest.raises(RuntimeError, match="injected native execution failure"):
        drain(batch)
    assert provider.binding.aborted == 1
    assert len(memo.advances) == 1
    assert memo.advances[0][1] == "transaction_abort"


def test_native_publish_failure_aborts_binding_and_advances_ast_memo():
    import copy

    import mlx.core as mx
    from test_external_dflash2_cpu import drain, tiny

    from mlx2.runtime.ast_live_state import ASTPhysicalResult
    from mlx2.runtime.progressive_external_verify import (
        execute_progressive_target_rows,
    )

    class Binding:
        def __init__(self, model, **kwargs):
            self.model = model
            self.descriptor = kwargs["descriptor"]
            self.layers = tuple(kwargs["capture_layers"])
            self.flat_inputs = tuple(kwargs["flat_inputs"])
            self.work_rows = tuple(copy.deepcopy(row) for row in kwargs["rows"])
            self.aborted = 0

        def execute(self):
            outputs = []
            for lane, row in enumerate(self.work_rows):
                tokens = self.flat_inputs[self.descriptor.lane_slice(lane)]
                logits, features, _detail = execute_progressive_target_rows(
                    self.model,
                    mx.array([tokens]),
                    row,
                    self.layers,
                    protocol=self.model.progressive_external_verify_protocol,
                    fidelity="exact",
                    input_lengths=[len(tokens)],
                )
                mx.eval(logits, features)
                outputs.extend(
                    (logits[0, index], features[0, index])
                    for index in range(len(tokens))
                )
            return ASTPhysicalResult(
                self.descriptor,
                tuple(outputs),
                None,
                storage="flat_row_pairs",
                invocation_count=1,
            )

        def commit(self, _accepted):
            raise RuntimeError("injected native publish failure")

        def abort(self):
            self.aborted += 1

    class Provider(_Provider):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.binding = None

        def bind_serving_physical_target(self, **kwargs):
            self.binding = Binding(self.model, **kwargs)
            return self.binding

    class Memo:
        def __init__(self):
            self.advances = []

        def advance(self, scopes, cause):
            self.advances.append((tuple(scopes), cause))

        def register_scope(self, _scope):
            return False

    model, draft = tiny()
    provider = Provider(model)
    memo = Memo()
    model.ast_physical_target_provider = provider
    batch = _progressive_batch(
        model, draft, ast_physical_target=True, ast_memo=memo
    )
    batch.insert([[1, 2, 3]], max_tokens=[4])
    with pytest.raises(RuntimeError, match="injected native publish failure"):
        drain(batch)
    assert provider.binding.aborted == 1
    assert len(memo.advances) == 1
    assert memo.advances[0][1] == "transaction_abort"


@pytest.mark.parametrize("post_execute_drift", [False, True])
def test_enabled_physical_target_executes_progressive_seam_and_emits_receipt(
    post_execute_drift,
):
    import copy

    import mlx.core as mx
    from test_external_dflash2_cpu import drain, tiny

    from mlx2.runtime.ast_live_state import ASTPhysicalResult
    from mlx2.runtime.progressive_external_verify import (
        execute_progressive_target_rows,
    )

    class Binding:
        def __init__(self, provider, model, rows, descriptor, layers, flat_inputs):
            self.provider = provider
            self.model = model
            self.descriptor = descriptor
            self.layers = tuple(layers)
            self.flat_inputs = tuple(flat_inputs)
            self.base_rows = tuple(tuple(row) for row in rows)
            self.work_rows = tuple(copy.deepcopy(row) for row in rows)
            self.result = None
            self.committed = False
            self.aborted = False

        def execute(self):
            self.provider.native_round_active = True
            outputs = []
            for lane, row in enumerate(self.work_rows):
                lane_tokens = self.flat_inputs[self.descriptor.lane_slice(lane)]
                logits, features, _detail = execute_progressive_target_rows(
                    self.model,
                    mx.array([lane_tokens]),
                    row,
                    self.layers,
                    protocol=self.model.progressive_external_verify_protocol,
                    fidelity="exact",
                    input_lengths=[len(lane_tokens)],
                )
                mx.eval(logits, features)
                outputs.extend(
                    (logits[0, row], features[0, row])
                    for row in range(len(lane_tokens))
                )
            self.result = ASTPhysicalResult(
                self.descriptor,
                tuple(outputs),
                None,
                storage="flat_row_pairs",
                invocation_count=1,
            )
            return self.result

        def commit(self, accepted):
            self.provider.native_round_active = False
            self.committed = True
            committed = tuple(copy.deepcopy(row) for row in self.base_rows)
            for lane, count in enumerate(accepted):
                lane_tokens = self.flat_inputs[self.descriptor.lane_slice(lane)]
                mx.eval(
                    self.model(
                        mx.array([lane_tokens[:count]]), cache=committed[lane]
                    )
                )
            return committed

        def abort(self):
            self.provider.native_round_active = False
            self.aborted = True

        def receipt(self):
            return {
                "implemented": True,
                "qualified": False,
                "selected": True,
                "default": False,
                "observed_used": self.committed,
                "result": self.result.receipt(),
            }

    class Provider(_Provider):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.bind_calls = 0
            self.binding = None
            self.native_round_active = False

        def bind_serving_physical_target(self, **kwargs):
            self.bind_calls += 1
            self.binding = Binding(
                self,
                self.model,
                kwargs["rows"],
                kwargs["descriptor"],
                kwargs["capture_layers"],
                kwargs["flat_inputs"],
            )
            return self.binding

    model, draft = tiny()
    provider = Provider(model)
    model.ast_physical_target_provider = provider
    batch = _progressive_batch(model, draft, ast_physical_target=True)
    if post_execute_drift:
        target_law = batch._target_law

        def drift_from_target_law(*args, **kwargs):
            result = target_law(*args, **kwargs)
            model.ast_physical_target_provider = _Provider()
            return result

        batch._target_law = drift_from_target_law
    else:
        target_law = batch._target_law

        def reject_per_row_target_law(*_args, **_kwargs):
            if provider.native_round_active:
                raise AssertionError(
                    "native greedy rows must use one batched reduction"
                )
            return target_law(*_args, **_kwargs)

        batch._target_law = reject_per_row_target_law
    uid = batch.insert(
        [[1, 2, 3]],
        max_tokens=[4],
        **(
            {"sampling_configs": [{"sampling_temp": 0.8}]}
            if post_execute_drift
            else {}
        ),
    )[0]
    if post_execute_drift:
        with pytest.raises(RuntimeError, match="provider binding drifted"):
            drain(batch)
        assert provider.binding.committed is False
        assert provider.binding.aborted is True
        return
    output, final = drain(batch)
    assert len(output[uid]) == 4
    assert provider.bind_calls == 1
    assert provider.binding.committed is True
    assert provider.binding.aborted is False
    progressive = final[uid].speculative_receipt["progressive_verification"]
    layout = progressive["last_ragged_layout"]
    serving = layout["physical_target_serving"]
    assert serving["selected"] is serving["observed_used"] is True
    assert serving["default"] is serving["qualified"] is False
    assert serving["result"]["padding_rows"] == 0
    assert layout["fallback_backend_mode"] == layout["backend_mode"] == "padded"
    assert layout["executed_backend_mode"] == "native_flat_logical_rows"
    executor = progressive["target_row_executor"]
    assert executor["token_width_hist"] == {"2": 1}
    assert executor["last"]["adapter_detail"] == {
        "route": "native_flat_logical_rows",
        "geometry_sha256": layout["geometry_sha256"],
        "logical_rows": 2,
        "physical_rows": 2,
        "padding_rows": 0,
    }
    assert batch.scheduler_stats[
        "external_progressive_verify_b1_aligned_tile_launches"
    ] == 0


@pytest.mark.parametrize("replacement", [None, "replacement"])
def test_provider_replacement_or_removal_after_generator_construction_declines(
    replacement,
):
    from test_external_dflash2_cpu import drain, tiny

    model, draft = tiny()
    original = _Provider()
    model.ast_physical_target_provider = original
    batch = _progressive_batch(model, draft, ast_physical_target=True)
    model.ast_physical_target_provider = (
        None if replacement is None else _Provider()
    )
    batch.insert([[1, 2, 3]], max_tokens=[4])
    with pytest.raises(
        RuntimeError, match="requested AST physical target is unavailable"
    ):
        drain(batch)
    assert original.calls == 0


@pytest.mark.parametrize("phase", ["admission", "bind"])
def test_provider_replacement_during_live_callbacks_fails_before_execution(phase):
    from test_external_dflash2_cpu import drain, tiny

    class Replacing(_Provider):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.executed = False

        def admit_serving_physical_target(self, **kwargs):
            admission = super().admit_serving_physical_target(**kwargs)
            if phase == "admission":
                self.model.ast_physical_target_provider = _Provider()
            return admission

        def bind_serving_physical_target(self, **_kwargs):
            if phase == "bind":
                self.model.ast_physical_target_provider = _Provider()

            class NeverExecute:
                def execute(inner_self):
                    self.executed = True
                    raise AssertionError("provider replacement reached execution")

                def abort(inner_self):
                    return None

            return NeverExecute()

    model, draft = tiny()
    provider = Replacing(model)
    model.ast_physical_target_provider = provider
    batch = _progressive_batch(model, draft, ast_physical_target=True)
    batch.insert([[1, 2, 3]], max_tokens=[4])
    with pytest.raises(
        RuntimeError, match="requested AST physical target admission or binding failed"
    ):
        drain(batch)
    assert provider.executed is False


def test_target_tree_and_native_physical_target_conflict_before_binding():
    from test_ast_target_tree_provider_cpu import (
        _provider_profile,
        _ReferenceChainProvider,
    )
    from test_external_dflash2_cpu import drain, tiny

    from mlx2.runtime.ast_planner import ASTExecutionPlanner

    model, draft = tiny()
    native_provider = _Provider()
    model.ast_physical_target_provider = native_provider
    batch = _progressive_batch(model, draft, ast_physical_target=True)
    batch.ast_execution_planner = ASTExecutionPlanner(
        _provider_profile(), tile_cap=2, source_limit=3, minimum_depth=3
    )
    batch.ast_target_tree_executor_provider = _ReferenceChainProvider(batch)
    draft_distributions = draft.draft_distributions

    def planner_compatible_draft(*args, **kwargs):
        kwargs.pop("ast_source_budgets", None)
        return draft_distributions(*args, **kwargs)

    draft.draft_distributions = planner_compatible_draft
    batch.insert([[1, 2, 3]], max_tokens=[4])
    with pytest.raises(
        RuntimeError, match="conflicts with target-tree backend"
    ):
        drain(batch)
    assert native_provider.calls == 0
    assert native_provider.admission_calls == 0
