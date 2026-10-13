"""L2-05 (sweep 2026-10-12): segmented self-MTP true-batches through the generic
segmented view when the model declares no state-plane provider (Xing4.0), and
reports a provider that declines a view in the segmented-MTP receipt."""

import json
from pathlib import Path

import mlx.core as mx

XING_FIXTURE = Path(__file__).parent / "fixtures" / "xing4_0_tiny"


def _xing_model():
    from mlx2.runtime.models.xing4_0 import Model, ModelArgs

    model = Model(ModelArgs.from_dict(
        json.loads((XING_FIXTURE / "config.json").read_text())))
    model.load_weights(list(model.sanitize(
        mx.load(str(XING_FIXTURE / "weights.safetensors"))).items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def _xing_run(model):
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG
    from mlx2.runtime.segmented_self_mtp import segmented_self_mtp_stats

    segmented_self_mtp_stats(reset=True)
    config = {"persistent": True, "num_draft": 2, "segment_aware_live_tip": True,
              "segment_aware_cohort_size": 2, "rate_gate": False}
    gen = BatchGenerator(model, completion_batch_size=2, prefill_batch_size=2,
                         prefill_step_size=4, self_mtp=config)
    gen.insert([[3, 17, 5, 9, 22, 41, 7], [8, 8, 30, 2, 11]], max_tokens=[12, 12],
               lane_rngs=[LaneRNG(0), LaneRNG(1)],
               self_mtp_configs=[{"sampling_temp": 0.0}] * 2)
    tokens = {}
    for _ in range(60):
        for response in gen.next()[1]:
            tokens.setdefault(response.uid, []).append(response.token)
    return tokens, segmented_self_mtp_stats()


def test_l2_05_segmented_mtp_true_batches_without_state_plane_provider(monkeypatch):
    new_stream = mx.new_stream
    monkeypatch.setattr(mx, "new_stream", lambda device: new_stream(mx.cpu))
    model = _xing_model()
    assert getattr(model, "state_plane_operations", None) is None
    tokens, stats = _xing_run(model)
    assert stats["true_batched_requests"] > 0
    assert stats["true_batched_declined"] == 0
    assert stats["true_batched_engaged"] == stats["true_batched_requests"]
    # The receipt names the generic view rather than a provider-built one.
    assert stats["true_batched_generic_view"] > 0
    # Exactness: the serial B1 consumer emits the same token streams.
    monkeypatch.setattr(model, "mtp_segmented_true_batch", False, raising=False)
    serial_tokens, serial = _xing_run(model)
    assert serial["true_batched_engaged"] == 0
    assert tokens == serial_tokens


def test_l2_05_declared_provider_without_view_operations_reports_decline(monkeypatch):
    from mlx2.runtime.state_plane_operations import StatePlaneContract

    class _ContractOnly:
        provider_revision = "contract-only-test"

        def contract(self, cache):
            return StatePlaneContract(
                "contract-only-test", type(cache).__name__, "exact",
                frozenset({"checkpoint", "private_successor", "rollback", "promotion"}))

    new_stream = mx.new_stream
    monkeypatch.setattr(mx, "new_stream", lambda device: new_stream(mx.cpu))
    model = _xing_model()
    model.state_plane_operations = _ContractOnly()
    _tokens, stats = _xing_run(model)
    assert stats["true_batched_requests"] > 0
    assert stats["true_batched_engaged"] == 0
    assert stats["true_batched_generic_view"] == 0
    # Every declined proposal cycle (and any K=0 fast-path attempt) names
    # the reason in the segmented-MTP receipt.
    assert stats["true_batched_declined"] > 0
    assert stats["true_batched_declined_no_view_operations"] >= stats["true_batched_declined"]
    assert stats["true_batched_declined_view_refused"] == 0
