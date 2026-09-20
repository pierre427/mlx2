"""rm10: per-position draft confidence for cost-aware self-MTP depth."""

import json
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.mtp_confidence import (
    DraftConfidenceProbe,
    DraftConfidenceRow,
    LogisticConfidence,
    MTPAcceptanceLogger,
    Top1ProbConfidence,
    confidence_model_from_dict,
    evaluate_confidence,
    expected_calibration_error,
    fit_logistic_confidence,
    fit_sequential_temperatures,
    load_acceptance_log,
    load_confidence_model,
    roc_auc,
)

from test_batched_mtp import _tiny_qwen4_model


def _row(top1, depth, accepted, uid=0, tokens=None):
    tokens = tokens if tokens is not None else tuple(range(len(top1)))
    return DraftConfidenceRow(
        uid=uid,
        features=tuple((float(p), 0.1, 1.0) for p in top1),
        tokens=tuple(tokens),
        prev_token=5,
        verify_depth=depth,
        accepted=accepted,
    )


def _synthetic_rows(n, seed=0, positions=4, miscal=1.8):
    """Rows whose true conditional acceptance q maps to an overconfident top1."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = rng.uniform(-1.5, 2.5)
        logit_q = base - 0.3 * np.arange(positions)
        q = 1 / (1 + np.exp(-logit_q))
        top1 = 1 / (1 + np.exp(-(miscal * logit_q + 0.7)))
        accepted = 0
        while accepted < positions and rng.random() < q[accepted]:
            accepted += 1
        rows.append(_row(top1, positions, accepted, uid=i))
    return rows


# --- labels, metrics ----------------------------------------------------------


def test_row_labels_are_conditional_and_censored():
    row = _row([0.9, 0.8, 0.7, 0.6, 0.5], depth=3, accepted=1)
    assert row.labels() == [1, 0, None, None, None]
    full = _row([0.9, 0.8, 0.7], depth=3, accepted=3)
    assert full.labels() == [1, 1, 1]
    assert row.prev_tokens() == [5, 0, 1, 2, 3]


def test_ece_and_auc_known_values():
    assert expected_calibration_error([0.9] * 10, [1] * 9 + [0]) == pytest.approx(0.0)
    assert expected_calibration_error([1.0] * 4, [0] * 4) == pytest.approx(1.0)
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert roc_auc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)


def test_sts_reduces_survival_calibration_error():
    rows = _synthetic_rows(1500, seed=1)
    raw = Top1ProbConfidence()
    temps = fit_sequential_temperatures(raw, rows, 4)
    calibrated = Top1ProbConfidence(temperatures=temps)
    before = evaluate_confidence(raw, rows, 4)
    after = evaluate_confidence(calibrated, rows, 4)
    assert all(t > 1.0 for t in temps[:2])  # overconfident drafter -> soften
    assert after["expected_length_mae"] <= before["expected_length_mae"] + 1e-9
    assert abs(after["expected_length_bias"]) < abs(before["expected_length_bias"])
    # Ranking is invariant under a per-position temperature.
    for b, a in zip(before["positions"], after["positions"]):
        assert a["auc"] == pytest.approx(b["auc"])


def test_logistic_fit_learns_signal_and_token_bias():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(1200):
        good = rng.random() < 0.5
        top1 = rng.uniform(0.3, 0.9, 3)
        # Token 7 is always rejected; top1 carries the rest of the signal.
        tokens = tuple(7 if (not good and j == 0) else 1 for j in range(3))
        accepted = 0
        while accepted < 3:
            if tokens[accepted] == 7 or rng.random() > top1[accepted]:
                break
            accepted += 1
        rows.append(_row(top1, 3, accepted, uid=i, tokens=tokens))
    model = fit_logistic_confidence(rows, positions=3, buckets=64)
    report = evaluate_confidence(model, rows, 3)
    baseline = evaluate_confidence(Top1ProbConfidence(), rows, 3)
    assert report["positions"][0]["auc"] > baseline["positions"][0]["auc"] + 0.1
    assert report["positions"][0]["ece"] < 0.05
    again = confidence_model_from_dict(json.loads(json.dumps(model.to_dict())))
    assert again.sha256() == model.sha256()
    np.testing.assert_allclose(again.predict(rows[0]), model.predict(rows[0]))


def test_confidence_model_loading_fails_closed(tmp_path):
    assert isinstance(load_confidence_model("top1"), Top1ProbConfidence)
    with pytest.raises(ValueError):
        confidence_model_from_dict({"schema": "other", "kind": "top1"})
    with pytest.raises(ValueError):
        confidence_model_from_dict(
            {"schema": "mlx2.mtp_confidence.v1", "kind": "top1", "temperatures": [0]}
        )
    with pytest.raises(ValueError):
        confidence_model_from_dict(
            {"schema": "mlx2.mtp_confidence.v1", "kind": "logistic", "weights": [1.0] * 4, "position_bias": [float("nan")]}
        )
    path = tmp_path / "m.json"
    path.write_text(json.dumps(Top1ProbConfidence(temperatures=(1.5,)).to_dict()))
    assert load_confidence_model(str(path)).temperatures == (1.5,)


# --- cost and selection ---------------------------------------------------------


def test_acceptance_logger_roundtrip_and_bound(tmp_path):
    path = tmp_path / "log.jsonl"
    logger = MTPAcceptanceLogger(path, max_records=2, lookahead=1)
    assert logger.probe.lookahead == 1
    rows = [_row([0.9, 0.5, 0.4], 2, 1, uid=u) for u in range(3)]
    assert logger.record(rows) == 2
    assert logger.dropped == 1
    logger.close()
    loaded = load_acceptance_log(path)
    assert [r.uid for r in loaded] == [0, 1]
    assert loaded[0].labels() == [1, 0, None]
    record = json.loads(path.read_text().splitlines()[0])
    assert record["labels"] == [1, 0, None]


# --- serving policy -----------------------------------------------------------


def test_cli_acceptance_log_flags_default_off_and_fail_closed():
    from mlx2.server import build_parser, serving_engine_kwargs

    parser = build_parser()
    args = parser.parse_args(["--model", "m"])
    assert args.mtp_acceptance_log is None
    assert args.mtp_acceptance_log_lookahead == 0
    # The adaptive-depth flag is untouched by the logger.
    assert args.adaptive_mtp_depth is False
    assert not hasattr(args, "adaptive_mtp_controller")

    args = parser.parse_args(
        [
            "--model", "m", "--qualification-mode",
            "--mtp-acceptance-log", "/tmp/a.jsonl",
            "--mtp-acceptance-log-lookahead", "2",
        ]
    )
    kwargs = serving_engine_kwargs(
        args,
        None,
        native_mtp=True,
        approximate_kv=None,
        max_request_bytes=1 << 21,
    )
    assert kwargs["mtp_acceptance_log"] == {
        "path": "/tmp/a.jsonl",
        "lookahead": 2,
    }
    assert kwargs["adaptive_mtp_depth"] is False


# --- tiny model on CPU ----------------------------------------------------------


def _greedy_lane(model, uid, prompt, num_draft):
    from mlx2.runtime.hybrid_speculative import prepare_self_mtp_lane
    from mlx2.runtime.sample_utils import LaneRNG

    return prepare_self_mtp_lane(
        mx.array(prompt, mx.uint32),
        model,
        uid=uid,
        max_tokens=24,
        prompt_cache=None,
        mtp_state=None,
        lane_rng=LaneRNG(900 + uid),
        num_draft=num_draft,
        sampling_temp=0.0,
        sampling_top_p=1.0,
        sampling_top_k=0,
        sampling_min_p=0.0,
        accept_rule="residual",
        logits_processors=[],
        prefill_step_size=4,
        share_qsa_indices=False,
    )[0]


def _run_cycles(model, probe, cycles=4, sites=None):
    from mlx2.runtime import hybrid_speculative as hs

    lanes = [
        _greedy_lane(model, 0, [1, 7, 3, 9, 2, 8], 2),
        _greedy_lane(model, 1, [4, 4, 11, 2, 6, 5, 1], 2),
    ]
    batch = hs.attach_self_mtp_lanes(model, None, lanes)
    tokens = [[], []]
    proposals = []
    original = hs.record_verify_sync

    def record(site):
        if sites is not None:
            sites.append(site)
        return original(site)

    with patch.object(hs, "record_verify_sync", side_effect=record):
        for _ in range(cycles):
            for lane in batch.lanes:
                lane.confidence_probe = probe
            proposal = hs.propose_batched_self_mtp(model, batch)
            hs.commit_batched_self_mtp(
                batch,
                proposal,
                emitted_counts=[len(row) for row in proposal.outputs],
                terminal=[False, False],
            )
            for row, outputs in enumerate(proposal.outputs):
                tokens[row].extend(out.token for out in outputs)
            proposals.append(proposal)
    return tokens, proposals


def test_probe_lookahead_is_greedy_bit_exact_and_sync_free():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(1010)
        model = _tiny_qwen4_model()
        base_sites, probe_sites = [], []
        base_tokens, base_props = _run_cycles(model, None, sites=base_sites)
        probe = DraftConfidenceProbe(
            lookahead=2, projection=mx.array(np.ones((64, 2), np.float32))  # MTP post hidden = hc_count x hidden
        )
        probe_tokens, probe_props = _run_cycles(model, probe, sites=probe_sites)
        assert probe_tokens == base_tokens
        assert [p.accepted_lengths for p in probe_props] == [
            p.accepted_lengths for p in base_props
        ]
        # No extra host synchronisation: features ride the accept boundary.
        assert probe_sites == base_sites
        assert all(not p.draft_features for p in base_props)
        first = probe_props[0]
        assert len(first.draft_features) == 2
        for row, depth in enumerate(first.draft_depths):
            feats = first.draft_features[row]
            assert len(feats) == depth + 2  # verified + lookahead
            assert all(len(f) == 3 + 2 for f in feats)
            assert all(0.0 < f[0] <= 1.0 and f[1] >= -1e-5 and f[2] >= 0 for f in feats)
            assert len(first.draft_feature_tokens[row]) == depth + 2
    finally:
        mx.set_default_device(previous)


def test_batch_generator_acceptance_log_matches_ordinary_output(tmp_path):
    """The logger (with lookahead) observes a real self-MTP run and changes
    nothing about it: the tokens still match the ordinary route exactly."""
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(922)
        model = _tiny_qwen4_model()
        prompt = [1, 7, 3, 9, 2, 8, 4]
        ordinary = BatchGenerator(
            model, completion_batch_size=1, prefill_batch_size=1, prefill_step_size=32
        )
        log_path = tmp_path / "accept.jsonl"
        mtp = BatchGenerator(
            model,
            completion_batch_size=1,
            prefill_batch_size=1,
            prefill_step_size=32,
            self_mtp={
                "num_draft": 3,
                "persistent": True,
                "segment_aware_live_tip": True,
                "segment_aware_cohort_size": 1,
            },
            mtp_acceptance_log={"path": str(log_path), "lookahead": 1},
        )
        ordinary.insert([prompt], max_tokens=[14])
        mtp.insert(
            [prompt],
            max_tokens=[14],
            lane_rngs=[LaneRNG(17)],
            self_mtp_configs=[{"sampling_temp": 0.0}],
        )
        outputs = [[], []]
        receipts = []
        for _ in range(200):
            for index, batch in enumerate((ordinary, mtp)):
                _, responses = batch.next()
                outputs[index].extend(r.token for r in responses)
                if index == 1:
                    receipts.extend(
                        r.mtp_receipt for r in responses if getattr(r, "mtp_receipt", None)
                    )
            if min(map(len, outputs)) >= 14:
                break
        assert outputs[1] == outputs[0]
        stats = mtp.scheduler_stats
        # Mechanism assertions: features observed and log written.
        assert stats["mtp_confidence_feature_cycles"] > 0
        assert stats["mtp_acceptance_log_records"] > 0
        # The logger does not turn on adaptive depth.
        assert "adaptive_mtp_boundaries" not in stats
        assert all(r.get("adaptive_depth") is None for r in receipts)
        ordinary.close()
        mtp.close()
        rows = load_acceptance_log(log_path)
        assert len(rows) == stats["mtp_acceptance_log_records"]
        assert any(len(r.features) > r.verify_depth for r in rows)  # lookahead seen
    finally:
        mx.set_default_device(previous)


def test_offline_trainer_end_to_end(tmp_path):
    import importlib.util
    from pathlib import Path

    log = tmp_path / "accept.jsonl"
    logger = MTPAcceptanceLogger(log)
    rows = _synthetic_rows(800, seed=5)
    # Many requests (uids) so the per-request holdout split is populated.
    logger.record(rows)
    logger.close()
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_mtp_confidence.py"
    spec = importlib.util.spec_from_file_location("train_mtp_confidence", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.main([
        "--log", str(log), "--out", str(tmp_path / "m.json"),
        "--out-top1", str(tmp_path / "t.json"), "--report", str(tmp_path / "r.json"),
        "--buckets", "16",
    ])
    assert report["rows"]["train"] > 0 and report["rows"]["holdout"] > 0
    model = load_confidence_model(str(tmp_path / "m.json"))
    top1 = load_confidence_model(str(tmp_path / "t.json"))
    assert isinstance(model, LogisticConfidence) and isinstance(top1, Top1ProbConfidence)
    holdout = report["holdout"]
    assert abs(holdout["top1_sts"]["expected_length_bias"]) < abs(
        holdout["top1_raw"]["expected_length_bias"]
    )
