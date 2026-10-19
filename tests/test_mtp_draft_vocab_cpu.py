"""CPU-only contract checks for the reduced native-MTP proposal head."""

from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace

import pytest


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(tmp_path, count=4096):
    for name, content in (
        ("config.json", json.dumps({"model_type": "qwen4_exp", "vocab_size": count + 64})),
        ("model.safetensors.index.json", '{"weight_map":{}}'),
        ("tokenizer.json", "{}"),
    ):
        (tmp_path / name).write_text(content)
    ids = tmp_path / "mtp_draft_vocab.ids"
    ids.write_text("".join(f"{value}\n" for value in range(count)))
    license_file = tmp_path / "mtp_draft_vocab.ids.LICENSE"
    license_file.write_text("SPDX-License-Identifier: Apache-2.0\n")
    manifest = {
        "schema": "mlx2.mtp-draft-vocab.v1",
        "vocab_size": count + 64,
        "token_count": count,
        "ids_sha256": _sha(ids),
        "license_sha256": _sha(license_file),
        "config_sha256": _sha(tmp_path / "config.json"),
        "index_sha256": _sha(tmp_path / "model.safetensors.index.json"),
        "tokenizer_sha256": _sha(tmp_path / "tokenizer.json"),
        "source_repository": "https://example.invalid/source",
        "source_revision": "a" * 40,
        "corpus_profile": "english-code-test",
    }
    (tmp_path / "mtp_draft_vocab.json").write_text(json.dumps(manifest))
    return manifest


def test_manifest_is_exactly_bound_to_ids_and_model_files(tmp_path):
    from mlx2.runtime.mtp_draft_vocab import DraftVocabError, load_manifest

    expected = _artifact(tmp_path)
    manifest, ids = load_manifest(tmp_path)
    assert manifest.receipt() == expected
    assert ids[:3] == (0, 1, 2) and len(ids) == 4096
    (tmp_path / "tokenizer.json").write_text('{"changed":true}')
    with pytest.raises(DraftVocabError, match="binding mismatch for tokenizer.json"):
        load_manifest(tmp_path)


def test_adapter_identity_changes_with_draft_vocabulary_sidecars(tmp_path):
    from mlx2.adapters.flash_next import artifact_identity

    _artifact(tmp_path)
    (tmp_path / "ple_rows.bin").write_bytes(b"ple")
    before = artifact_identity(tmp_path)["fingerprint"]
    ids = tmp_path / "mtp_draft_vocab.ids"
    ids.write_text(ids.read_text() + "# reviewed\n")
    after = artifact_identity(tmp_path)["fingerprint"]
    assert before != after


def test_reduced_head_full_slice_identity_and_minus_inf_scatter_cpu():
    import mlx.core as mx
    from mlx import nn

    from mlx2.runtime.mtp_draft_vocab import ReducedMTPHead

    mx.set_default_device(mx.cpu)
    source = nn.Linear(3, 8, bias=True)
    source.weight = mx.arange(24, dtype=mx.float32).reshape(8, 3) / 10
    source.bias = mx.arange(8, dtype=mx.float32)
    hidden = mx.array([[1.0, -2.0, 0.5]], dtype=mx.float32)
    exact = source(hidden)
    full = ReducedMTPHead(source, range(8), 8)(hidden)
    reduced = ReducedMTPHead(source, (1, 4, 7), 8)(hidden)
    mx.eval(exact, full, reduced)
    assert mx.array_equal(exact, full).item()
    assert reduced[0, [1, 4, 7]].tolist() == exact[0, [1, 4, 7]].tolist()
    assert all(math.isinf(reduced[0, value].item()) for value in (0, 2, 3, 5, 6))


def test_quantized_head_rows_are_sliced_without_dequantizing_cpu():
    import mlx.core as mx
    from mlx import nn

    from mlx2.runtime.mtp_draft_vocab import ReducedMTPHead

    mx.set_default_device(mx.cpu)
    source = nn.QuantizedLinear.from_linear(
        nn.Linear(64, 64, bias=False), group_size=32, bits=4
    )
    hidden = mx.arange(64, dtype=mx.float32)[None] / 64
    exact = source(hidden)
    head = ReducedMTPHead(source, (1, 7, 31, 63), 64)
    sliced = head(hidden)
    mx.eval(exact, sliced)
    assert sliced[0, [1, 7, 31, 63]].tolist() == exact[0, [1, 7, 31, 63]].tolist()
    assert math.isinf(sliced[0, 0].item())
    assert head.quantized and head.bits == 4 and head.group_size == 32


def test_installation_is_separate_from_target_head_and_reports_observed_use():
    import mlx.core as mx
    from mlx import nn

    from mlx2.runtime.mtp_draft_vocab import (
        DraftVocabManifest,
        install_reduced_mtp_head,
    )

    mx.set_default_device(mx.cpu)
    target_head = nn.Linear(4, 8, bias=False)
    model = SimpleNamespace(
        mtp=object(),
        language_model=SimpleNamespace(
            args=SimpleNamespace(tie_word_embeddings=False), lm_head=target_head
        ),
    )
    manifest = DraftVocabManifest(
        schema="mlx2.mtp-draft-vocab.v1",
        vocab_size=8,
        token_count=3,
        ids_sha256="a",
        license_sha256="license",
        config_sha256="b",
        index_sha256="c",
        tokenizer_sha256="d",
        source_repository="repo",
        source_revision="revision",
        corpus_profile="test",
    )
    receipt = install_reduced_mtp_head(model, manifest, (1, 3, 7))
    assert model.language_model.lm_head is target_head
    assert model.mtp_draft_head is not target_head
    assert receipt["enabled"] and receipt["token_count"] == 3
    assert model.mtp_draft_vocab_calls == 0
    assert model.mtp_draft_vocab_full_bypasses == 0
    assert model.mtp_draft_vocab_enabled is True


def test_restricted_proposal_rejection_law_recovers_target_distribution():
    # Enumerate one rejection-sampling step. q is zero outside its slice; the
    # accepted mass plus the normalized positive residual must equal p exactly.
    p = [0.1, 0.2, 0.3, 0.4]
    q = [0.0, 0.6, 0.0, 0.4]
    accepted = [min(pi, qi) for pi, qi in zip(p, q)]
    rejected = 1.0 - sum(accepted)
    residual = [max(pi - qi, 0.0) for pi, qi in zip(p, q)]
    total = sum(residual)
    emitted = [a + rejected * r / total for a, r in zip(accepted, residual)]
    assert emitted == pytest.approx(p)


def test_request_compact_greedy_proposal_is_isolated_from_target():
    import mlx.core as mx
    from mlx import nn

    from mlx2.runtime.mtp_draft_vocab import (
        DraftVocabError, DraftVocabManifest, RequestCompactGreedyHead,
        ReducedMTPHead,
    )

    mx.set_default_device(mx.cpu)
    target = nn.Linear(4, 8, bias=False)
    manifest = DraftVocabManifest(
        schema="mlx2.mtp-draft-vocab.v1", vocab_size=8, token_count=3,
        ids_sha256="a", license_sha256="b", config_sha256="c",
        index_sha256="d", tokenizer_sha256="e", source_repository="repo",
        source_revision="revision", corpus_profile="test",
    )
    request = RequestCompactGreedyHead(ReducedMTPHead(target, (1, 3, 7), 8), manifest)
    hidden = mx.random.normal((1, 2, 4))
    full_before = target(hidden)
    ids, compact = request.propose(hidden, greedy=True)
    mx.eval(full_before, ids, compact)
    assert compact.shape == (1, 2, 3)
    assert mx.allclose(
        compact, mx.take(full_before, mx.array([1, 3, 7]), axis=-1),
        atol=1e-5, rtol=1e-5,
    ).item()
    assert mx.array_equal(ids, mx.take(request.head.token_ids, mx.argmax(compact, axis=-1))).item()
    assert mx.array_equal(full_before, target(hidden)).item()
    with pytest.raises(DraftVocabError, match="greedy-only"):
        request.propose(hidden, greedy=False)


def test_request_compact_head_requires_bound_artifact(tmp_path):
    import mlx.core as mx
    from mlx import nn
    from mlx2.runtime.mtp_draft_vocab import DraftVocabError, RequestCompactGreedyHead

    mx.set_default_device(mx.cpu)
    _artifact(tmp_path)
    source = nn.Linear(4, 4160, bias=False)
    request = RequestCompactGreedyHead.from_bound_artifact(tmp_path, source)
    assert request.head.vocab_size == 4160
    assert request.head.token_ids.size == 4096
    (tmp_path / "tokenizer.json").write_text('{"modified":true}')
    with pytest.raises(DraftVocabError, match="binding mismatch"):
        RequestCompactGreedyHead.from_bound_artifact(tmp_path, source)


def test_constrained_lane_selects_full_vocab_proposal_path():
    from mlx2.runtime.hybrid_speculative import _mtp_proposal_step

    calls = []

    class Model:
        def mtp_step(self, *_args):
            calls.append("reduced")
            return "reduced"

        def mtp_step_full_vocab(self, *_args):
            calls.append("full")
            return "full"

    model = Model()
    plain = [SimpleNamespace(logits_processors=[])]
    constrained = [SimpleNamespace(logits_processors=[object()])]
    assert _mtp_proposal_step(model, None, None, None, plain) == "reduced"
    assert _mtp_proposal_step(model, None, None, None, constrained) == "full"
    assert calls == ["reduced", "full"]


def test_policy_is_default_off_and_opt_in_is_receipted():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    assert "mtp_draft_vocab" not in FlashNextPolicy().as_dict()
    selected = FlashNextPolicy(mtp_draft_vocab=True)
    assert selected.as_dict()["mtp_draft_vocab"] is True
    assert "mtp_draft_vocab" not in selected.batch_config(max_lanes=1, prefill_step=8)


def test_coverage_run_counts_trailing_admitted_tokens_capped_at_window():
    from mlx2.runtime.mtp_draft_vocab import (
        advance_coverage_run,
        coverage_admits,
        membership_mask,
    )

    member = membership_mask((1, 2, 3), 8)
    assert member == bytes([0, 1, 1, 1, 0, 0, 0, 0])
    assert advance_coverage_run(0, member, 4, (1, 2, 3)) == 3
    assert advance_coverage_run(0, member, 4, (1, 2, 3, 1, 2, 3)) == 4
    assert advance_coverage_run(4, member, 4, (5,)) == 0
    assert advance_coverage_run(0, member, 4, (1, 5, 2)) == 1
    # An id outside the mask is never admitted.
    assert advance_coverage_run(3, member, 4, (99,)) == 0
    # The pending ``cur`` completes the window; window 0 never gates.
    assert coverage_admits(3, member, 4, 2)
    assert not coverage_admits(3, member, 4, 6)
    assert not coverage_admits(2, member, 4, 2)
    assert coverage_admits(0, member, 0, 6)


def _gate_model(calls, enabled=True):
    class Model:
        mtp_draft_vocab_enabled = enabled
        mtp_draft_vocab_coverage_bypasses = 0

        def mtp_step(self, *_args):
            calls.append("reduced")
            return "reduced"

        def mtp_step_full_vocab(self, *_args):
            calls.append("full")
            return "full"

    return Model()


def _lane(run, cur, window=4, processors=()):
    from mlx2.runtime.mtp_draft_vocab import membership_mask

    return SimpleNamespace(
        logits_processors=list(processors),
        coverage_member=membership_mask(range(4), 8),
        coverage_window=window,
        coverage_run=run,
        cur=cur,
    )


def test_coverage_gate_is_batch_wide_and_counted():
    from mlx2.runtime.hybrid_speculative import _mtp_proposal_step

    calls = []
    model = _gate_model(calls)
    covered, uncovered = _lane(3, 1), _lane(3, 6)
    assert _mtp_proposal_step(model, None, None, None, [covered]) == "reduced"
    # One uncovered lane sends the whole batch through the full head.
    assert _mtp_proposal_step(model, None, None, None, [covered, uncovered]) == "full"
    assert model.mtp_draft_vocab_coverage_bypasses == 1
    # Constrained lanes keep their own bypass and do not count as coverage.
    constrained = _lane(3, 1, processors=[object()])
    assert _mtp_proposal_step(model, None, None, None, [constrained]) == "full"
    assert model.mtp_draft_vocab_coverage_bypasses == 1
    # Gate off (window 0) or reduced head disabled: no coverage bypass.
    assert _mtp_proposal_step(model, None, None, None, [_lane(0, 6, window=0)]) == "reduced"
    disabled = _gate_model(calls, enabled=False)
    assert _mtp_proposal_step(disabled, None, None, None, [uncovered]) == "reduced"
    assert disabled.mtp_draft_vocab_coverage_bypasses == 0
    assert calls == ["reduced", "full", "full", "reduced", "reduced"]


def test_lane_coverage_survives_recovery_snapshot_without_copying_the_mask():
    from mlx2.runtime.hybrid_speculative import (
        _advance_lane_coverage,
        _snapshot_segmented_recovery_row,
    )

    lane = _lane(1, 1)
    _advance_lane_coverage(lane, [1, 2])
    assert lane.coverage_run == 3
    pair = SimpleNamespace(target=[], draft=[])
    fields, *_ = _snapshot_segmented_recovery_row((lane, pair))
    assert fields["coverage_run"] == 3
    assert fields["coverage_member"] is lane.coverage_member
    _advance_lane_coverage(lane, [7])
    assert lane.coverage_run == 0 and fields["coverage_run"] == 3


def test_install_receipts_the_coverage_window_and_rejects_bad_values():
    import mlx.core as mx
    from mlx import nn

    from mlx2.runtime.mtp_draft_vocab import (
        DraftVocabError,
        DraftVocabManifest,
        install_reduced_mtp_head,
    )

    mx.set_default_device(mx.cpu)

    def model():
        return SimpleNamespace(
            mtp=object(),
            language_model=SimpleNamespace(
                args=SimpleNamespace(tie_word_embeddings=False),
                lm_head=nn.Linear(4, 8, bias=False),
            ),
        )

    manifest = DraftVocabManifest(
        schema="mlx2.mtp-draft-vocab.v1", vocab_size=8, token_count=3,
        ids_sha256="a", license_sha256="l", config_sha256="b", index_sha256="c",
        tokenizer_sha256="d", source_repository="r", source_revision="v",
        corpus_profile="t",
    )
    m = model()
    receipt = install_reduced_mtp_head(m, manifest, (1, 3, 7))
    assert receipt["coverage_window"] == 16
    assert m.mtp_draft_vocab_member == bytes([0, 1, 0, 1, 0, 0, 0, 1])
    assert m.mtp_draft_vocab_coverage_bypasses == 0
    m = model()
    assert install_reduced_mtp_head(m, manifest, (1, 3, 7), coverage_window=0)[
        "coverage_window"
    ] == 0
    for bad in (-1, 1025, 8.0, True):
        with pytest.raises(DraftVocabError, match="coverage window"):
            install_reduced_mtp_head(model(), manifest, (1, 3, 7), coverage_window=bad)


def test_policy_coverage_window_is_receipted_only_with_the_reduced_head():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    assert "mtp_draft_vocab_coverage_window" not in FlashNextPolicy().as_dict()
    assert FlashNextPolicy(mtp_draft_vocab=True).as_dict()[
        "mtp_draft_vocab_coverage_window"
    ] == 16
    assert FlashNextPolicy(
        mtp_draft_vocab=True, mtp_draft_vocab_coverage_window=0
    ).as_dict()["mtp_draft_vocab_coverage_window"] == 0
    with pytest.raises(ValueError, match="requires mtp_draft_vocab"):
        FlashNextPolicy(mtp_draft_vocab_coverage_window=8)
    for bad in (-1, 1025, "16", True):
        with pytest.raises(ValueError, match="coverage_window"):
            FlashNextPolicy(mtp_draft_vocab=True, mtp_draft_vocab_coverage_window=bad)
