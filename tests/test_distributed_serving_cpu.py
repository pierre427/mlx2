import hashlib

import pytest

from mlx2.runtime.distributed_serving import (
    ClusterIdentity,
    Phase,
    RankMessage,
    RequestIdentity,
    RequestTransaction,
    admission_receipt,
)

H = hashlib.sha256(b"artifact").hexdigest()
T = hashlib.sha256(b"tokenizer").hexdigest()
TENANT = hashlib.sha256(b"tenant").hexdigest()
PAYLOAD = hashlib.sha256(b"token-12").hexdigest()


def identities():
    cluster = ClusterIdentity("pair-a", "abc123", H, T, "fixture.adapter.pipeline")
    request = RequestIdentity("r1", TENANT, cluster.digest, 9, 4)
    return cluster, request


def message(request, phase, sequence, rank, payload=PAYLOAD, revision=9):
    return RankMessage(phase, request.digest, sequence, rank, revision, payload)


def test_admission_binds_two_rank_route_identity_and_memory():
    cluster, request = identities()
    cap = {
        rank: {
            "cluster_sha256": cluster.digest,
            "route": cluster.route,
            "available_bytes": 4096,
            "reserved_bytes": 2048,
        }
        for rank in (0, 1)
    }
    receipt = admission_receipt(cluster, request, rank_capabilities=cap)
    assert receipt["admitted"] is True
    assert receipt["qualified"] is False
    assert receipt["request_sha256"] == request.digest


@pytest.mark.parametrize("mutation", ["missing", "identity", "route", "capacity"])
def test_admission_fails_closed_on_rank_mismatch(mutation):
    cluster, request = identities()
    cap = {
        rank: {"cluster_sha256": cluster.digest, "route": cluster.route,
               "available_bytes": 4096, "reserved_bytes": 2048}
        for rank in (0, 1)
    }
    if mutation == "missing":
        cap.pop(1)
    elif mutation == "identity":
        cap[1]["cluster_sha256"] = H
    elif mutation == "route":
        cap[1]["route"] = "other"
    else:
        cap[1]["reserved_bytes"] = 4097
    with pytest.raises((RuntimeError, ValueError)):
        admission_receipt(cluster, request, rank_capabilities=cap)


def admitted_transaction():
    _, request = identities()
    txn = RequestTransaction(request)
    txn.apply(message(request, Phase.ADMIT, 0, 0))
    txn.apply(message(request, Phase.ACK, 1, 0))
    txn.apply(message(request, Phase.ACK, 2, 1))
    return request, txn


def test_two_rank_step_publishes_only_after_equal_outputs_then_commits():
    request, txn = admitted_transaction()
    assert txn.apply(message(request, Phase.STEP, 3, 0)) is None
    step = txn.apply(message(request, Phase.STEP, 4, 1))
    assert step["state_revision"] == 10
    commit = txn.apply(message(request, Phase.COMMIT, 5, 0, revision=10))
    assert commit["published_state_revision"] == 10
    assert commit["steps"] == 1 and txn.terminal


def test_rank_divergence_aborts_without_publishing_state():
    request, txn = admitted_transaction()
    txn.apply(message(request, Phase.STEP, 3, 0))
    with pytest.raises(RuntimeError, match="outputs disagree"):
        txn.apply(message(request, Phase.STEP, 4, 1, hashlib.sha256(b"other").hexdigest()))
    assert txn.terminal


def test_admitted_max_tokens_fails_closed_before_an_extra_step():
    cluster, request = identities()
    request = RequestIdentity("r1", TENANT, cluster.digest, 9, 1)
    txn = RequestTransaction(request)
    txn.apply(message(request, Phase.ADMIT, 0, 0))
    txn.apply(message(request, Phase.ACK, 1, 0))
    txn.apply(message(request, Phase.ACK, 2, 1))
    txn.apply(message(request, Phase.STEP, 3, 0))
    txn.apply(message(request, Phase.STEP, 4, 1))
    with pytest.raises(RuntimeError, match="token limit"):
        txn.apply(message(request, Phase.STEP, 5, 0, revision=10))
    assert txn.terminal


@pytest.mark.parametrize("fault", ["gap", "duplicate", "revision", "foreign"])
def test_faults_fail_closed_before_state_publication(fault):
    request, txn = admitted_transaction()
    candidate = message(request, Phase.STEP, 3, 0)
    if fault == "gap":
        candidate = message(request, Phase.STEP, 4, 0)
    elif fault == "duplicate":
        txn.apply(candidate)
    elif fault == "revision":
        candidate = message(request, Phase.STEP, 3, 0, revision=10)
    elif fault == "foreign":
        candidate = RankMessage(Phase.STEP, H, 3, 0, 9, PAYLOAD)
    with pytest.raises(RuntimeError):
        txn.apply(candidate)
    assert txn.terminal
    with pytest.raises(RuntimeError, match="terminal"):
        txn.apply(message(request, Phase.COMMIT, 5, 0))


def test_stale_cancel_poisoning_cannot_be_followed_by_step_or_commit():
    request, txn = admitted_transaction()
    with pytest.raises(RuntimeError, match="revision"):
        txn.apply(message(request, Phase.CANCEL, 3, 1, revision=10))
    assert txn.terminal
    with pytest.raises(RuntimeError, match="terminal"):
        txn.apply(message(request, Phase.STEP, 3, 0))
    with pytest.raises(RuntimeError, match="terminal"):
        txn.apply(message(request, Phase.COMMIT, 3, 0))


def test_cancel_from_worker_discards_partial_step_and_is_terminal():
    request, txn = admitted_transaction()
    txn.apply(message(request, Phase.STEP, 3, 0))
    receipt = txn.apply(message(request, Phase.CANCEL, 4, 1))
    assert receipt["published_state_revision"] is None and txn.terminal
    with pytest.raises(RuntimeError, match="terminal"):
        txn.apply(message(request, Phase.COMMIT, 5, 0))


def test_types_reject_bool_and_nonfinite_style_values():
    cluster, request = identities()
    with pytest.raises(ValueError):
        RequestIdentity("x", TENANT, cluster.digest, True, 1)
    with pytest.raises(ValueError):
        ClusterIdentity("x", "r", H, T, "route", world_size=True)
    with pytest.raises(ValueError):
        admission_receipt(cluster, request, rank_capabilities={
            0: {"cluster_sha256": cluster.digest, "route": cluster.route,
                "available_bytes": 1, "reserved_bytes": 0},
            1: {"cluster_sha256": cluster.digest, "route": cluster.route,
                "available_bytes": 1, "reserved_bytes": False},
        })
