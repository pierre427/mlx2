"""Bug sweep 2026-11-06, ds41 lane regressions (CPU only)."""
import pytest


def test_omlx_close_releases_resident_expert_banks():
    """DS41-1: close() must drop resident banks exactly as park() does."""
    from mlx2.adapters.deepseek_v41_omlx_weights import OmlxV41Weights
    mx = pytest.importorskip("mlx.core")

    class Closable:
        failure = None

        def close(self, *_):
            pass

        def counters(self):
            return {"weight_read_inflight": 0}

    with mx.stream(mx.cpu):
        weights = object.__new__(OmlxV41Weights)
        weights.expert_stream, weights._read_pool = Closable(), Closable()
        weights._dense, weights._quant, weights._pending_dense = {}, {}, {}
        weights._pending_dense_host = None
        bank = {kind: (mx.zeros((4, 8, 2), mx.uint32), mx.zeros((4, 8, 1), mx.bfloat16),
                       mx.zeros((4, 8, 1), mx.bfloat16)) for kind in ("gate", "up", "down")}
        weights._banks, weights._bank_views = {0: bank}, {0: [{"gate.qweight": bank["gate"][0][0]}]}
        assert weights.resident_bytes() > 0
        weights.close()
        assert weights._closed
        assert weights._banks == {} and weights._bank_views == {}
        assert weights.resident_bytes() == 0


def test_prepare_artifact_survives_a_stale_seal_temporary(tmp_path, monkeypatch):
    """DS41-3: a crashed run's seal temporary must not refuse later prepares."""
    import hashlib

    from mlx2.adapters import deepseek_v41_omlx_artifact as artifact

    payload = b"weights"
    (tmp_path / "a.bin").write_bytes(payload)
    pins = {"files": {"a.bin": {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}}
    monkeypatch.setattr(artifact, "PINS", pins)
    monkeypatch.setattr(artifact, "inspect_artifact", lambda root: "inspected")
    target = tmp_path / artifact.SEAL
    target.with_suffix(".tmp").write_text("partial")  # legacy fixed-name leftover
    (tmp_path / (target.name + ".crashed.tmp")).write_text("partial")
    assert artifact.prepare_artifact(tmp_path) == "inspected"
    assert target.exists()
    assert target.stat().st_mode & 0o777 == 0o644  # mkstemp's 0600 widened as before
    leftovers = sorted(p.name for p in tmp_path.glob(target.name + ".*.tmp"))
    assert leftovers == [target.name + ".crashed.tmp"]  # this run cleaned up after itself


def _tiny_omlx_artifact():
    from types import SimpleNamespace

    def t(size):
        return {"size": size}

    tensors = {"embed.weight": t(1000), "norm.weight": t(10)}
    quant = {}
    for name in ("attn.wq",):
        quant[name] = {"module": name}
        tensors.update({name + ".weight": t(200), name + ".scales": t(20), name + ".biases": t(20)})
    experts = {}
    for kind in ("gate", "up", "down"):
        module = "layers.0.ffn.experts." + kind
        experts[kind] = {"module": module}
        tensors.update({module + ".weight": t(400), module + ".scales": t(50), module + ".biases": t(50)})
    return {"tensors": tensors, "args": SimpleNamespace(num_hidden_layers=2, n_routed_experts=4),
            "inventory": {"expert_item_bytes": 300, "storage_bytes": {"expert": 2400}},
            "dense": {"embed": "embed.weight", "norm": "norm.weight"}, "quant": quant,
            "experts": [experts, experts], "identity": {"fingerprint": "f"}}


def test_omlx_plan_reports_full_staging_but_reserves_the_route_terms():
    """DS41-2: the plan reports the largest dense tensor's staging and a bank
    kind's full staging (weight+scales+biases), but the reservation keeps the
    route's original terms; reserving the staging cost decode speed for no
    demonstrated safety (qualification/runs/ds41-omlx-plan-ab-20261106)."""
    from mlx2.adapters.deepseek_v41_omlx_weights import CHUNK_BYTES, omlx_weight_plan

    artifact = _tiny_omlx_artifact()
    plan = omlx_weight_plan(artifact, 300, bank_layers=(0,))
    dense = 1000 + 10 + 240
    assert plan["dense_staging_bound_bytes"] == 1000 + CHUNK_BYTES
    assert plan["expert_bank_staging_bound_bytes"] == 500
    assert plan["total_reservation_bytes"] == (
        dense + 2 * CHUNK_BYTES + 300 + plan["expert_active_bound_bytes"] + 300
        + plan["expert_bank_bytes"] + 400)
    no_bank = omlx_weight_plan(artifact, 300)
    assert no_bank["expert_bank_staging_bound_bytes"] == 0
    assert no_bank["total_reservation_bytes"] == dense + 2 * CHUNK_BYTES + 300 + 1200 + 300
