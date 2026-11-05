"""CPU-only compiled-program admission, cache identity and deployment trials.

Synthetic tiny-hybrid@tiny-hybrid is a fixture, not a model qualification.
"""

import functools
import json

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from test_tenant_auth import served as served_fixture

from mlx2.experimental.weight_space.programs import create_program, export_program
from mlx2.runtime.apc_suspension import lora_file_receipt, verify_lora_receipt
from mlx2.runtime.lora import install_lora
from mlx2.runtime.multi_lora import MultiLoRAManager
from mlx2.serving import request_apc_scope

CHECKPOINT = "tiny@tiny-hybrid"
SOURCE = "e448f070b84642bfc5d11dc04719e40336326a68"
served = served_fixture


def artifact(path, *, key="linear", dims=(4, 3), checkpoint=CHECKPOINT,
             provenance=None):
    program = create_program(
        {key: (np.full((dims[0], 1), 0.01, np.float32),
               np.full((1, dims[1]), 0.02, np.float32))},
        checkpoint=checkpoint, source_revision=SOURCE,
        construction="analytical_directional_repair",
        provenance=provenance or {"source": "synthetic CPU fixture"},
    )
    export_program(program, path)
    return program


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)

    def __call__(self, x):
        return self.linear(x)


def test_single_program_binding_numerics_and_reversible_restore(tmp_path):
    path = tmp_path / "program"
    program = artifact(path)
    model = Tiny()
    original = model.linear
    x = mx.array([[1.0, 2.0, 3.0, 4.0]])
    base = model(x)
    with pytest.raises(ValueError, match="exact loaded base"):
        install_lora(model, name="program", path=path)
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        install_lora(model, name="program", path=path, base_checkpoint="other@other")
    assert model.linear is original
    session = install_lora(model, name="program", path=path, base_checkpoint=CHECKPOINT)
    a, b = program.factor_map["linear"]
    expected = base + mx.array(np.array(x) @ a @ b)
    assert mx.array_equal(model(x), expected)
    assert session.weight_program["program_id"] == program.program_id
    session.restore(model)
    assert model.linear is original
    assert mx.array_equal(model(x), base)


def test_program_identity_includes_provenance_and_survives_suspension(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    artifact(one, provenance={"source": "first"})
    artifact(two, provenance={"source": "second"})
    manager = MultiLoRAManager(Tiny(), max_loras=2, max_lora_rank=8,
                               delta_precision="fp32", base_checkpoint=CHECKPOINT)
    first = manager.register("one", one)
    second = manager.register("two", two)
    assert first["fingerprint"] != second["fingerprint"]
    receipt = lora_file_receipt(one)
    assert receipt["fingerprint"] == first["fingerprint"]
    assert "weight_program.json" in receipt["files"]
    assert verify_lora_receipt(receipt, str(tmp_path)) == receipt
    assert request_apc_scope({"_mlx2_lora_fingerprint": first["fingerprint"]}) != request_apc_scope(
        {"_mlx2_lora_fingerprint": second["fingerprint"]}
    )


def test_program_rejects_wrong_arithmetic_and_metadata_downgrade(tmp_path):
    path = tmp_path / "program"
    artifact(path)
    manager = MultiLoRAManager(Tiny(), max_loras=1, max_lora_rank=1,
                               base_checkpoint=CHECKPOINT)
    with pytest.raises(ValueError, match="delta_precision fp32"):
        manager.prepare("program", path)
    config_path = path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config.pop("weight_program")
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="config does not match"):
        install_lora(Tiny(), name="program", path=path, base_checkpoint=CHECKPOINT)


def test_registered_program_cannot_change_before_materialization(tmp_path):
    path = tmp_path / "program"
    artifact(path)
    manager = MultiLoRAManager(Tiny(), max_loras=1, max_lora_rank=1,
                               delta_precision="fp32", base_checkpoint=CHECKPOINT)
    manager.register("program", path)
    manifest_path = path / "weight_program.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["source"] = "changed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        manager.acquire("program")
    assert not manager.pins


def test_program_engagement_requires_a_called_covered_target(tmp_path):
    path = tmp_path / "program"
    artifact(path)
    model = Tiny()
    manager = MultiLoRAManager(model, max_loras=1, max_lora_rank=1,
                               delta_precision="fp32", base_checkpoint=CHECKPOINT)
    manager.register("program", path)
    slot, _ = manager.acquire("program")
    manager.bind_uid(17, slot)
    manager.bind_rows([17])
    assert manager.row_forward_count(17) == 1
    assert manager.program_application_count(17) == 0
    mx.eval(model(mx.ones((1, 4))))
    assert manager.program_application_count(17) == 1
    assert manager.status()["counts"]["program_delta_applications"] == 1
    manager.clear_rows()
    mx.eval(model(mx.ones((1, 4))))
    assert manager.program_application_count(17) == 1
    manager.unbind_uid(17)
    assert manager.status()["counts"]["program_delta_applications"] == 1
    manager.release(slot)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    from test_multi_lora import make_engine

    from mlx2 import memory, serving
    from mlx2.runtime import os_memory

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": SOURCE})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    value = make_engine(tmp_path, max_loras=2, max_lora_rank=4, max_lanes=1)
    yield value
    value.close()


def register_engine_program(engine, tmp_path, *, name="repair", provenance=None):
    key = "model.layers.3.self_attn.q_proj"
    module = dict(engine.adapter.model.named_modules())[key]
    dims = (module.weight.shape[1], module.weight.shape[0])
    program = artifact(tmp_path / name, key=key, dims=dims, provenance=provenance)
    loaded = engine.load_lora_adapter(name, name)
    return program, loaded


def validation(program, fingerprint):
    return {
        "passed": True, "repair_passed": True, "regressions_passed": True,
        "replay_passed": True, "program_id": program.program_id,
        "checkpoint": program.checkpoint, "source_revision": program.source_revision,
        "fingerprint": fingerprint, "evidence_digest": "fixture-evidence",
        "tool_suite_digest": "fixture-tools",
    }


def test_replay_is_cold_and_does_not_change_deployment_default(engine, tmp_path):
    from test_multi_lora import LONG, collect, request

    program, loaded = register_engine_program(engine, tmp_path)
    before = engine.weight_program_deployment_state()
    # Populate this candidate namespace; the trial must still replay fully.
    collect(engine.submit(request(LONG, "repair")))
    job = engine.submit_weight_program_replay(
        request(LONG), name="repair", expected_fingerprint=loaded["fingerprint"],
    )
    _, receipt = collect(job)
    assert receipt["cached_tokens"] == 0
    assert receipt["weight_program"]["program_id"] == program.program_id
    assert receipt["weight_program"]["observed_used"] is True
    assert receipt["qualification"] == "unqualified"
    assert engine.weight_program_deployment_state() == before
    assert job.request["skip_writing_prefix_cache"] is True
    with pytest.raises(ValueError, match="old cache"):
        engine.submit_weight_program_replay(
            {**request(LONG), "cache_handle": "old"}, name="repair",
            expected_fingerprint=loaded["fingerprint"],
        )


def test_actual_replay_publication_does_not_replace_live_session(engine, tmp_path):
    from test_multi_lora import LONG, collect, request

    from mlx2.experimental.weight_space.live import LiveWeightSpaceService
    from mlx2.experimental.weight_space.self_repair import RepairSession

    _, loaded = register_engine_program(engine, tmp_path)
    session = RepairSession("chat", {"messages": [{"role": "user", "content": "original"}]})
    live = LiveWeightSpaceService.__new__(LiveWeightSpaceService)
    live.config = {"tenant_id": "default", "session": session, "validators": {}, "raw": {}}
    live.engine = engine
    live._bound = lambda: None
    live.live_session_observed = False
    live._error = None
    engine.weight_space_live = live
    body = {**request(LONG), "session_id": "chat", "messages": [{"role": "user", "content": "current"}]}
    collect(engine.submit(body))
    before = session.snapshot()
    assert before["revision"] == 1
    collect(engine.submit_weight_program_replay(
        body, name="repair", expected_fingerprint=loaded["fingerprint"],
    ))
    assert session.snapshot() == before


def test_deployment_promotion_cas_default_alias_and_rollback(engine, tmp_path):
    from test_multi_lora import LONG, collect, request

    from mlx2.server import _generation_model

    program, loaded = register_engine_program(engine, tmp_path)
    previous = engine.weight_program_deployment_state()
    proof = validation(program, loaded["fingerprint"])
    with pytest.raises(ValueError, match="passing bound validation"):
        engine.promote_weight_program("repair", expected_fingerprint=loaded["fingerprint"],
                                      expected_generation=0, validation_receipt={**proof, "repair_passed": False})
    promoted = engine.promote_weight_program(
        "repair", expected_fingerprint=loaded["fingerprint"], expected_generation=0,
        validation_receipt=proof,
    )
    assert promoted["generation"] == 1
    assert engine.status()["qualification"] == "unqualified"
    assert _generation_model(engine.status(), {"model": "tiny:base"}) == "tiny:base"
    for selected in (None, "tiny"):
        job = engine.submit(request(LONG, selected))
        _, receipt = collect(job)
        assert job.lora_name == "repair"
        assert receipt["lora"]["deployment_generation"] == 1
    base_job = engine.submit(request(LONG, "tiny:base"))
    _, base_receipt = collect(base_job)
    assert base_job.lora_name is None
    assert "weight_program" not in base_receipt
    with pytest.raises(ValueError, match="generation changed"):
        engine.promote_weight_program("repair", expected_fingerprint=loaded["fingerprint"],
                                      expected_generation=0, validation_receipt=proof)
    with pytest.raises(ValueError, match="rollback"):
        engine.unload_lora_adapter("repair")
    restored = engine.rollback_weight_program_default(expected_generation=1, previous_state=previous)
    assert restored["generation"] == 2 and restored["default_adapter"] is None
    assert engine._prepare_job(request(LONG, "tiny"), tenant_id="default").lora_name is None


def test_private_program_cannot_be_promoted(engine, tmp_path):
    program, loaded = register_engine_program(
        engine, tmp_path, provenance={"parent": {"contains_private_memory": True}},
    )
    with pytest.raises(ValueError, match="private memory"):
        engine.promote_weight_program("repair", expected_fingerprint=loaded["fingerprint"],
                                      expected_generation=0,
                                      validation_receipt=validation(program, loaded["fingerprint"]))


def test_program_install_uses_the_validated_config(tmp_path, monkeypatch):
    from mlx2.runtime import lora

    path = tmp_path / "program"
    artifact(path)
    real = lora.read_lora_config

    def drifted(candidate):
        # A config that changes between the first read and validation must
        # not size or scale the installed modules.
        resolved, config = real(candidate)
        return resolved, {**config, "scale": 20.0}

    monkeypatch.setattr(lora, "read_lora_config", drifted)
    model = Tiny()
    original = model.linear
    with pytest.raises(ValueError, match="validated weight program"):
        install_lora(model, name="program", path=path, base_checkpoint=CHECKPOINT)
    assert model.linear is original


def test_single_session_program_reports_observed_use(tmp_path, monkeypatch):
    from test_multi_lora import LONG, collect, make_engine, request

    from mlx2 import memory, serving
    from mlx2.runtime import os_memory

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": SOURCE})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    single = make_engine(tmp_path, max_lanes=1)  # no max_loras: install_lora path
    try:
        assert single.multi_lora is None
        _, loaded = register_engine_program(single, tmp_path)
        assert loaded["weight_program"]["observed_used"] is False
        _, receipt = collect(single.submit(request(LONG)))
        assert receipt["weight_program"]["selected"] is True
        assert receipt["weight_program"]["observed_used"] is True
        assert single.status()["lora"]["weight_program"]["observed_used"] is True
    finally:
        single.close()


def test_promoted_program_metrics_report_unqualified(engine, tmp_path):
    from mlx2.prometheus import render_engine_metrics

    program, loaded = register_engine_program(engine, tmp_path)
    engine.promote_weight_program(
        "repair", expected_fingerprint=loaded["fingerprint"], expected_generation=0,
        validation_receipt=validation(program, loaded["fingerprint"]),
    )
    assert engine.status()["qualification"] == "unqualified"
    info = [line for line in render_engine_metrics(engine).splitlines()
            if line.startswith("mlx2_model_info{")]
    assert len(info) == 1 and 'qualification="unqualified"' in info[0], info


def test_unconfigured_weight_space_state_matches_status(engine):
    from mlx2.prometheus import render_engine_metrics

    status = engine.status()
    text = render_engine_metrics(engine)
    for feature, key in (("quantization_overlay", "weight_overlays"),
                         ("layer_excision", "layer_excision"),
                         ("expert_mask", "expert_masks")):
        for flag in ("implemented", "qualified", "selected", "observed_used"):
            sample = (f'mlx2_weight_space_runtime_state{{feature="{feature}",'
                      f'state="{flag}"}} {int(status[key][flag] is True)}')
            assert sample in text, sample
    assert status["weight_overlays"]["implemented"] is True


def _candidate(engine, tmp_path, name="repair-abc"):
    from mlx2.experimental.weight_space.self_repair import ServingRepairBackend

    key = "model.layers.3.self_attn.q_proj"
    module = dict(engine.adapter.model.named_modules())[key]
    base = getattr(module, "base", module)
    program = artifact(tmp_path / name, key=key, dims=(base.weight.shape[1], base.weight.shape[0]))
    backend = ServingRepairBackend(engine, observe_job=lambda job: job)
    return program, backend, backend.register(name, tmp_path / name, program)


def test_repair_candidate_is_not_client_selectable(engine, tmp_path):
    """A candidate loaded through the repair backend runs only as host replay,
    the validated session override or the deployed default; a client naming
    it (or batch submission) sees an unknown model, and client status hides it."""
    from test_multi_lora import LONG, collect, request

    from mlx2.api_resources import ResourceNotFound
    from mlx2.server import _generation_model

    program, backend, candidate = _candidate(engine, tmp_path)
    assert "repair-abc" in engine.multi_lora.registry
    status = engine.status()
    assert "repair-abc" not in status["multi_lora"]["registered"]
    assert "repair-abc" not in status["multi_lora"]["registered_fingerprints"]
    assert status["multi_lora"]["private_candidates"] == 1
    assert all(r.get("name") != "repair-abc" for r in status["recent_operation_receipts"])
    assert engine.private_weight_program_names() == {"repair-abc"}
    with pytest.raises(ResourceNotFound, match="unknown model"):
        _generation_model(status, {"model": "repair-abc"})
    with pytest.raises(ResourceNotFound, match="unknown model"):
        engine.submit(request(LONG, "repair-abc"))
    with pytest.raises(ResourceNotFound, match="unknown model"):
        engine._prepare_job(request(LONG, "repair-abc"), tenant_id="other")
    # Host replay still selects it.
    _, receipt = collect(engine.submit_weight_program_replay(
        request(LONG), name="repair-abc", expected_fingerprint=candidate["fingerprint"]))
    assert receipt["weight_program"]["program_id"] == program.program_id
    # The validated session override (host-written fingerprint) selects it.
    session_request = {**request(LONG, "repair-abc"),
                       "_mlx2_weight_session_expected_fingerprint": candidate["fingerprint"]}
    assert engine._prepare_job(session_request, tenant_id="default").lora_name == "repair-abc"
    # Promoted, it is the default and may also be named.
    previous = engine.weight_program_deployment_state()
    engine.promote_weight_program("repair-abc", expected_fingerprint=candidate["fingerprint"],
                                  expected_generation=0,
                                  validation_receipt=validation(program, candidate["fingerprint"]))
    assert engine._prepare_job(request(LONG, "repair-abc"), tenant_id="default").lora_name == "repair-abc"
    assert engine._prepare_job(request(LONG), tenant_id="default").lora_name == "repair-abc"
    engine.rollback_weight_program_default(expected_generation=1, previous_state=previous)
    with pytest.raises(ResourceNotFound, match="unknown model"):
        engine._prepare_job(request(LONG, "repair-abc"), tenant_id="default")
    # Released with the candidate; a same-name operator load is ordinary.
    backend.unregister(candidate)
    assert engine.private_weight_program_names() == frozenset()
    engine.load_lora_adapter("repair-abc", "repair-abc")
    assert engine._prepare_job(request(LONG, "repair-abc"), tenant_id="default").lora_name == "repair-abc"
    assert "repair-abc" in engine.status()["multi_lora"]["registered"]


def test_duplicate_registration_keeps_the_first_candidate_hidden(engine, tmp_path):
    from test_multi_lora import LONG, request

    from mlx2.api_resources import ResourceNotFound

    program, backend, candidate = _candidate(engine, tmp_path)
    with pytest.raises(ValueError, match="already loaded"):
        backend.register("repair-abc", tmp_path / "repair-abc", program)
    with pytest.raises(ValueError, match="already reserved"):
        engine.reserve_private_weight_program("repair-abc")
    with pytest.raises(ResourceNotFound):
        engine._prepare_job(request(LONG, "repair-abc"), tenant_id="default")
    # A stale fingerprint-bound cleanup never clears another pending reservation.
    backend.unregister(candidate)
    token = engine.reserve_private_weight_program("repair-abc")
    engine.release_private_weight_program("repair-abc", fingerprint=candidate["fingerprint"])
    assert "repair-abc" in engine._private_weight_programs
    engine.release_private_weight_program("repair-abc", token=token)
    assert engine._private_weight_programs == {}


def test_reserved_name_with_other_content_is_ordinary(engine, tmp_path):
    from test_multi_lora import LONG, request

    from mlx2.api_resources import ResourceNotFound

    _, loaded = register_engine_program(engine, tmp_path, name="repair-x")
    token = engine.reserve_private_weight_program("repair-x")
    with pytest.raises(ResourceNotFound):  # pending: any content is hidden
        engine._prepare_job(request(LONG, "repair-x"), tenant_id="default")
    engine.bind_private_weight_program("repair-x", token, "other-fingerprint")
    assert engine._prepare_job(request(LONG, "repair-x"), tenant_id="default").lora_name == "repair-x"
    assert "repair-x" in engine.status()["multi_lora"]["registered"]
    engine.release_private_weight_program("repair-x", token=token)
    token = engine.reserve_private_weight_program("repair-x")
    engine.bind_private_weight_program("repair-x", token, loaded["fingerprint"])
    with pytest.raises(ResourceNotFound):
        engine._prepare_job(request(LONG, "repair-x"), tenant_id="default")


def test_http_discovery_and_prompt_endpoints_hide_repair_candidates(engine, tmp_path, served):
    from test_tenant_auth import _call

    _candidate(engine, tmp_path)
    app = served(None, engine)
    status, _, raw = _call(app.base, "GET", "/v1/models")
    assert status == 200
    assert "repair-abc" not in [model["id"] for model in json.loads(raw)["data"]]
    status, _, raw = _call(app.base, "GET", "/v1/status")
    assert status == 200 and b"repair-abc" not in raw
    body = {"model": "repair-abc", "messages": [{"role": "user", "content": "hi"}]}
    for path in ("/tokenize", "/apply-template"):
        status, _, raw = _call(app.base, "POST", path, body)
        assert status == 404 and b"unknown model" in raw, path
        # The base model resolves (the fixture tokenizer then cannot render).
        status, _, raw = _call(app.base, "POST", path, {**body, "model": "tiny"})
        assert status != 404 and b"unknown model" not in raw, path


def test_released_candidate_receipts_stay_out_of_client_status(engine, tmp_path):
    from test_multi_lora import LONG, collect, request

    _, backend, candidate = _candidate(engine, tmp_path)
    collect(engine.submit_weight_program_replay(
        request(LONG), name="repair-abc", expected_fingerprint=candidate["fingerprint"]))
    collect(engine.submit(request(LONG)))
    for _ in range(2):  # while registered, then after release
        status = engine.status()
        assert "repair-abc" not in json.dumps(status["recent_receipts"])
        assert "repair-abc" not in json.dumps(engine.recent_receipts(tenant_id="default"))
        assert "repair-abc" not in json.dumps(status["recent_operation_receipts"])
        assert "repair-abc" not in status["multi_lora"]["rows_by_adapter"]
        assert status["recent_receipts"]  # the base request's receipt remains
        backend.unregister(candidate)
    assert "repair-abc" not in engine.multi_lora.registry


def test_registration_refuses_a_name_an_operator_adapter_holds(engine, tmp_path):
    from test_multi_lora import LONG, request

    from mlx2.experimental.weight_space.self_repair import ServingRepairBackend

    program, _ = register_engine_program(engine, tmp_path, name="repair-op")
    backend = ServingRepairBackend(engine, observe_job=lambda job: job)
    with pytest.raises(ValueError, match="already loaded"):
        backend.register("repair-op", tmp_path / "repair-op", program)
    assert engine._private_weight_programs == {}
    assert engine._prepare_job(request(LONG, "repair-op"), tenant_id="default").lora_name == "repair-op"


def test_operator_reload_after_candidate_cleanup_keeps_its_receipts(engine, tmp_path):
    from test_multi_lora import LONG, collect, request

    _, backend, candidate = _candidate(engine, tmp_path)
    backend.unregister(candidate)
    engine.load_lora_adapter("repair-abc", "repair-abc")  # same artifact, operator-owned
    collect(engine.submit(request(LONG, "repair-abc")))
    status = engine.status()
    assert any((r.get("lora") or {}).get("name") == "repair-abc" for r in status["recent_receipts"])
    assert any(r.get("name") == "repair-abc" and not r.get("private_candidate")
               for r in status["recent_operation_receipts"])
    assert status["multi_lora"]["rows_by_adapter"].get("repair-abc")


def test_ordinary_unload_ends_the_candidate_reservation(engine, tmp_path):
    from test_multi_lora import LONG, request

    _, backend, candidate = _candidate(engine, tmp_path)
    engine.unload_lora_adapter("repair-abc")  # operator endpoint, before controller cleanup
    assert engine._private_weight_programs == {}
    engine.load_lora_adapter("repair-abc", "repair-abc")  # same artifact
    assert engine._prepare_job(request(LONG, "repair-abc"), tenant_id="default").lora_name == "repair-abc"
    assert "repair-abc" in engine.status()["multi_lora"]["registered"]
    backend.unregister(candidate)  # late controller cleanup: the name is no longer its own
    assert "repair-abc" in engine.multi_lora.registry


# Compiled PLE chain (qwen4_exp ``_compiled_chain``): a program covering
# ``ple.key_proj``/``ple.value_proj`` runs inside the traced head, whose
# replays skip the Python wrappers.  The gate is patched open on CPU, as in
# test_qwen4_ple_compile_cache.py (mx.compile traces on CPU too).
PLE_DIMS = {"ple.key_proj": (16, 64), "ple.value_proj": (16, 16)}


def ple_program(path):
    program = create_program(
        {key: (np.full((n_in, 1), 0.01, np.float32), np.full((1, n_out), 0.02, np.float32))
         for key, (n_in, n_out) in PLE_DIMS.items()},
        checkpoint=CHECKPOINT, source_revision=SOURCE,
        construction="analytical_directional_repair",
        provenance={"source": "synthetic CPU fixture"},
    )
    export_program(program, path)
    return program


class PLEModel(nn.Module):
    def __init__(self):
        super().__init__()
        from qsa_oracle import tiny_args

        from mlx2.runtime.models import qwen4_exp as QE

        mx.random.seed(7)
        self.ple = QE.PLELayer(tiny_args(ple_layer_ids=[1]), 0, 0)
        self.ple.set_dtype(mx.bfloat16)


@pytest.fixture
def ple_gate(monkeypatch):
    from mlx2.runtime.models import qwen4_exp as QE

    def open_gate(compiled):
        monkeypatch.setattr(QE, "_PLE_COMPILE", compiled)
        monkeypatch.setattr(QE.mx, "default_device", lambda: mx.gpu)
        QE.qwen4_ple_compile_status(reset=True)

    yield open_gate
    QE.qwen4_ple_compile_status(reset=True)


def ple_inputs(layer):
    key = mx.random.key(3)
    k1, k2, k3 = mx.random.split(key, 3)
    hidden = mx.random.normal((1, 2, 64), key=k1).astype(mx.bfloat16)
    embeddings = mx.random.normal((1, 2, 16), key=k2).astype(mx.bfloat16)
    state = mx.random.normal((1, layer.short_conv_state_len, 64), key=k3).astype(mx.bfloat16)
    return hidden, embeddings, state


def ple_requests(layer, count_fn, *, requests=2):
    """Warm the signature, then run ``requests`` one-call requests.

    Returns each request's covered-call delta since its admission and the
    outputs, so compiled and eager runs compare call for call.
    """
    from mlx2.runtime.models import qwen4_exp as QE

    hidden, embeddings, state = ple_inputs(layer)
    for _ in range(2):  # cold_eager, then trace (MIN_SEEN 2): warmed
        mx.eval(layer._run_device_chain(hidden, embeddings, None, state, True))
    executed = spy_compiled_chain(layer) if QE._PLE_COMPILE else None
    deltas, outputs = [], []
    for _ in range(requests):
        baseline = count_fn()
        out = layer._run_device_chain(hidden, embeddings, None, state, True)
        mx.eval(out)
        deltas.append(count_fn() - baseline)
        outputs.append(out)
    if executed is not None:
        # ``hits`` also counts a cached None (eager) entry: require that the
        # compiled head and tail themselves ran, and nothing fell back.
        assert executed == {"head": requests, "tail": requests}
        assert QE.qwen4_ple_compile_status()["counts"].get("fallbacks", 0) == 0
    return deltas, outputs


def spy_compiled_chain(layer):
    """Wrap the one cached compiled (head, tail) and count their calls."""
    (signature, (params, compiled)), = layer._ple_compile_cache.items()
    assert compiled is not None and all(map(callable, compiled))
    executed = {"head": 0, "tail": 0}

    def spy(name, fn):
        @functools.wraps(fn)  # keeps the counted-chain marker
        def run(*args):
            executed[name] += 1
            return fn(*args)
        return run

    layer._ple_compile_cache[signature] = (
        params, (spy("head", compiled[0]), spy("tail", compiled[1])))
    return executed


def test_compiled_ple_single_session_program_counts_like_eager(tmp_path, ple_gate):
    from types import SimpleNamespace

    from mlx2.runtime.lora import weight_program_receipt
    from mlx2.runtime.models import qwen4_exp as QE
    from mlx2.serving import ServingEngine

    ple_program(tmp_path / "program")
    runs = {}
    for compiled in (False, True):
        ple_gate(compiled)
        model = PLEModel()
        session = install_lora(model, name="program", path=tmp_path / "program",
                               base_checkpoint=CHECKPOINT)
        receipts = []
        deltas, outputs = ple_requests(
            model.ple, lambda s=session: s.program_application_count)
        if compiled:
            assert QE.qwen4_ple_compile_status()["counts"]["hits"] == 2
        for delta in deltas:
            # The serving receipt reads the session delta since admission.
            job = SimpleNamespace(
                weight_program=weight_program_receipt(session.weight_program, selected=True),
                weight_program_session_baseline=(session, session.program_application_count - delta),
            )
            receipts.append(ServingEngine._job_weight_program_receipt(
                SimpleNamespace(multi_lora=None), job))
        runs[compiled] = deltas, outputs, receipts, session.program_application_count
    eager, compiled = runs[False], runs[True]
    assert compiled[0] == eager[0] == [2, 2]  # key_proj + value_proj per call
    assert compiled[3] == eager[3]
    assert all(r["observed_used"] is True for r in compiled[2])
    for got, want in zip(compiled[1], eager[1]):
        assert all(mx.array_equal(a, b).item() for a, b in zip(got, want, strict=True))


def test_compiled_ple_multi_lora_program_counts_like_eager(tmp_path, ple_gate):
    from mlx2.runtime.models import qwen4_exp as QE

    ple_program(tmp_path / "program")
    runs = {}
    for compiled in (False, True):
        ple_gate(compiled)
        model = PLEModel()
        manager = MultiLoRAManager(model, max_loras=1, max_lora_rank=1,
                                   delta_precision="fp32", base_checkpoint=CHECKPOINT)
        try:
            manager.register("program", tmp_path / "program")
            slot, _ = manager.acquire("program")
            manager.bind_uid(17, slot)
            manager.bind_rows([17])
            deltas, outputs = ple_requests(
                model.ple, lambda m=manager: m.program_application_count(17))
            if compiled:
                assert QE.qwen4_ple_compile_status()["counts"]["hits"] == 2
            counts = manager.status()["counts"]
            runs[compiled] = (deltas, outputs, manager.program_application_count(17),
                              counts["program_delta_applications"], counts["delta_applications"])
            manager.clear_rows()
            manager.unbind_uid(17)
            manager.release(slot)
        finally:
            manager.detach()
    eager, compiled = runs[False], runs[True]
    assert compiled[0] == eager[0] == [2, 2]
    assert compiled[2:] == eager[2:]
    for got, want in zip(compiled[1], eager[1]):
        assert all(mx.array_equal(a, b).item() for a, b in zip(got, want, strict=True))


def test_compiled_ple_without_program_is_unchanged(ple_gate, monkeypatch):
    from mlx2.runtime.models import qwen4_exp as QE

    ple_gate(True)
    traced = []
    real = QE.mx.compile

    def spy(fn, *args, **kwargs):
        traced.append(fn)
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(QE.mx, "compile", spy)
    layer = PLEModel().ple
    hidden, embeddings, state = ple_inputs(layer)
    _deltas, outputs = ple_requests(layer, lambda: 0)
    reference = layer._device_chain(hidden, embeddings, None, state, True)
    assert QE.qwen4_ple_compile_status()["counts"]["hits"] == 2
    # No covered module: the plain head/tail functions are what mx.compile sees.
    assert [fn.__name__ for fn in traced] == ["head", "tail"]
    for out in outputs:
        assert all(mx.array_equal(a, b).item() for a, b in zip(out, reference, strict=True))


def test_compiled_ple_tail_failure_counts_one_eager_run(tmp_path, ple_gate):
    """A tail that raises after the compiled head ran falls back to eager:
    the request counts that one eager chain, not the head's calls too."""
    from mlx2.runtime.models import qwen4_exp as QE

    ple_program(tmp_path / "program")
    ple_gate(True)
    model = PLEModel()
    session = install_lora(model, name="program", path=tmp_path / "program",
                           base_checkpoint=CHECKPOINT)
    layer = model.ple
    hidden, embeddings, state = ple_inputs(layer)
    for _ in range(2):  # warm: the counted head/tail are cached
        mx.eval(layer._run_device_chain(hidden, embeddings, None, state, True))
    (signature, (params, (head, _tail))), = layer._ple_compile_cache.items()
    head_calls = []

    @functools.wraps(head)  # keeps the counted-chain marker
    def head_spy(*args):
        head_calls.append(1)
        return head(*args)

    def failing_tail(*args):
        raise RuntimeError("tail trace failed")

    layer._ple_compile_cache[signature] = (params, (head_spy, failing_tail))
    baseline = session.program_application_count
    out = layer._run_device_chain(hidden, embeddings, None, state, True)
    mx.eval(out)
    assert head_calls == [1]  # the compiled head ran before the tail failed
    assert QE.qwen4_ple_compile_status()["counts"]["fallbacks"] == 1
    assert session.program_application_count - baseline == 2  # one eager run
    reference = layer._device_chain(hidden, embeddings, None, state, True)
    assert all(mx.array_equal(a, b).item() for a, b in zip(out, reference, strict=True))
