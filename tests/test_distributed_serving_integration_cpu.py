"""CPU-only dense-adapter serving integration for two-rank transactions."""

import hashlib
from types import SimpleNamespace as NS

import pytest

from mlx2.api_resources import CapabilityUnavailable
from mlx2.contracts import Capability, ModelDescriptor
from mlx2.runtime.distributed_serving import ClusterIdentity, DenseReplicaBinding
from mlx2.serving import ServingEngine

ARTIFACT = hashlib.sha256(b"dense-artifact").hexdigest()
TOKENIZER = hashlib.sha256(b"tokenizer").hexdigest()
TENANT = hashlib.sha256(b"tenant").hexdigest()


def make_binding(*, rank0=None, rank1=None, reference=None):
    transform = lambda value: value.upper()
    cluster = ClusterIdentity(
        "cpu-pair",
        "source-revision",
        ARTIFACT,
        TOKENIZER,
        "dense_fixture:v1.dense_replica",
    )
    return DenseReplicaBinding(
        cluster,
        {0: rank0 or transform, 1: rank1 or transform},
        reference or transform,
        available_bytes={0: 4096, 1: 4096},
        reserved_bytes={0: 2048, 1: 2048},
        initial_state_revision=7,
    )


def adapter(binding, *, variant="v1", fingerprint=ARTIFACT):
    descriptor = ModelDescriptor(
        model_type="dense_fixture",
        family="dense_fixture",
        variant=variant,
        state_planes=frozenset(),
        capabilities=frozenset({Capability.TEXT}),
    )
    return NS(
        descriptor=descriptor,
        identity={"fingerprint": fingerprint},
        distributed_dense=binding,
    )


def engine_for(value):
    engine = ServingEngine.__new__(ServingEngine)
    engine.adapter = value
    calls = []

    def exclusive(operation, callback, **kwargs):
        calls.append((operation, kwargs))
        return callback(engine.adapter)

    engine._exclusive_adapter_operation = exclusive
    return engine, calls


def execute(binding, *, revision=7, request_id="request-1"):
    return binding.execute(
        b"dense input",
        request_id=request_id,
        tenant_sha256=TENANT,
        state_revision=revision,
    )


def test_dense_replica_publishes_only_after_rank_and_reference_agreement():
    binding = make_binding()

    output, receipt = execute(binding)

    assert output == b"DENSE INPUT"
    assert receipt["status"] == "completed"
    assert receipt["published_state_revision"] == 8
    assert receipt["execution"] == "dense_replica"
    assert receipt["ordinary_reference"] is receipt["observed_used"] is True
    assert receipt["qualified"] is receipt["cross_host_observed"] is False
    assert receipt["sharded_model"] is False
    status = binding.status()
    assert status["state_revision"] == 8 and status["completed"] == 1
    assert status["selected"] is False and status["observed_used"] is True
    receipt["rank_capabilities"][0]["available_bytes"] = 0
    assert binding.status()["last_receipt"]["rank_capabilities"][0][
        "available_bytes"
    ] == 4096
    with pytest.raises(RuntimeError, match="stale"):
        execute(binding, revision=7, request_id="stale")


def test_rank_divergence_aborts_without_state_publication():
    repaired = {"value": False}
    binding = make_binding(
        rank1=lambda value: value.upper() if repaired["value"] else b"different"
    )

    with pytest.raises(RuntimeError, match="rank outputs disagree"):
        execute(binding)

    status = binding.status()
    assert status["state_revision"] == 7
    assert status["completed"] == 0 and status["aborted"] == 1
    assert status["last_receipt"]["published_state_revision"] is None
    assert status["last_receipt"]["transaction_state_revision"] == 7
    assert status["last_receipt"]["transaction_terminal"] is True
    assert status["last_receipt"]["observed_used"] is False
    # The poisoned transaction cannot publish, while a fresh transaction may
    # retry the unchanged revision after the adapter fault is repaired.
    repaired["value"] = True
    output, receipt = execute(binding, request_id="rank-retry")
    assert output == b"DENSE INPUT" and receipt["published_state_revision"] == 8


def test_ordinary_reference_mismatch_aborts_and_same_revision_can_retry():
    repaired = {"value": False}
    binding = make_binding(
        reference=lambda value: value.upper() if repaired["value"] else b"wrong"
    )

    with pytest.raises(RuntimeError, match="ordinary reference"):
        execute(binding)

    assert binding.status()["state_revision"] == 7
    assert binding.status()["last_receipt"]["transaction_state_revision"] == 8
    repaired["value"] = True
    output, receipt = execute(binding, request_id="retry")
    assert output == b"DENSE INPUT" and receipt["published_state_revision"] == 8
    assert binding.status()["aborted"] == 1


def test_rank_failure_aborts_and_never_calls_later_rank_or_reference():
    called = []

    def fail(_value):
        called.append("rank0")
        raise ValueError("rank failure")

    binding = make_binding(
        rank0=fail,
        rank1=lambda value: called.append("rank1") or value,
        reference=lambda value: called.append("reference") or value,
    )
    with pytest.raises(ValueError, match="rank failure"):
        execute(binding)
    assert called == ["rank0"]
    assert binding.status()["last_receipt"]["published_state_revision"] is None
    assert binding.status()["state_revision"] == 7


def test_reference_exception_after_rank_commit_aborts_at_step_revision():
    def failed_reference(_value):
        raise LookupError("reference failure")

    binding = make_binding(reference=failed_reference)
    with pytest.raises(LookupError, match="reference failure"):
        execute(binding)
    status = binding.status()
    assert status["state_revision"] == 7
    assert status["last_receipt"]["transaction_state_revision"] == 8
    assert status["last_receipt"]["published_state_revision"] is None


def test_second_rank_exception_aborts_without_advancing_revision():
    binding = make_binding(rank1=lambda _value: (_ for _ in ()).throw(OSError("rank1")))
    with pytest.raises(OSError, match="rank1"):
        execute(binding)
    status = binding.status()
    assert status["state_revision"] == 7
    assert status["last_receipt"]["transaction_state_revision"] == 7
    assert status["last_receipt"]["published_state_revision"] is None


def test_serving_engine_uses_exclusive_distributed_admission():
    binding = make_binding()
    engine, calls = engine_for(adapter(binding))

    output, receipt = engine.execute_distributed_dense(
        b"dense input",
        request_id="request-1",
        tenant_sha256=TENANT,
        state_revision=7,
    )

    assert output == b"DENSE INPUT" and receipt["published_state_revision"] == 8
    assert calls == [
        ("distributed_dense", {"admission_class": "distributed", "admitted": False})
    ]
    assert engine._distributed_dense_state(engine.adapter) == {
        "distributed_dense": binding.status()
    }


@pytest.mark.parametrize("fault", ["missing", "adapter", "artifact"])
def test_serving_engine_fails_closed_before_rank_work(fault):
    calls = []
    binding = make_binding(
        rank0=lambda value: calls.append(0) or value.upper(),
        rank1=lambda value: calls.append(1) or value.upper(),
    )
    value = adapter(binding)
    if fault == "missing":
        value.distributed_dense = None
    elif fault == "adapter":
        value = adapter(binding, variant="foreign")
    else:
        value = adapter(binding, fingerprint=hashlib.sha256(b"foreign").hexdigest())
    engine, _exclusive = engine_for(value)

    error = CapabilityUnavailable if fault == "missing" else RuntimeError
    with pytest.raises(error):
        engine.execute_distributed_dense(
            b"dense input",
            request_id="request-1",
            tenant_sha256=TENANT,
            state_revision=7,
        )
    assert calls == []


@pytest.mark.parametrize("payload", [b"", b"x" * ((1 << 20) + 1), "not-bytes"])
def test_payload_contract_is_bounded_before_rank_work(payload):
    calls = []
    binding = make_binding(
        rank0=lambda value: calls.append(0) or value,
        rank1=lambda value: calls.append(1) or value,
    )
    with pytest.raises((TypeError, ValueError)):
        binding.execute(
            payload,
            request_id="request-1",
            tenant_sha256=TENANT,
            state_revision=7,
        )
    assert calls == []
