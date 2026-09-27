"""Adversarial CPU check of the 36-row dense-FC/coarse capture gate."""

import importlib.util
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (Path(__file__).resolve().parents[1] / "scripts/research" /
          "analyze_coarse_head_densefc_full.py")
SPEC = importlib.util.spec_from_file_location("analyze_coarse_head_densefc_full", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_full_gate_accepts_matched_rows_and_rejects_history_tamper(monkeypatch, tmp_path):
    corpus = np.arange(65536, dtype=np.uint32)
    monkeypatch.setattr(MODULE, "validate_provenance", lambda *_: {
        "corpus_token_count": len(corpus),
        "corpus_tokens_sha256": __import__("hashlib").sha256(
            corpus.astype("<u4").tobytes()).hexdigest(),
    })
    monkeypatch.setattr(MODULE.capture_runner, "corpus_tokens", lambda *_: corpus)
    strata = list(product(MODULE.capture_runner.CONTEXTS,
                          MODULE.capture_runner.SEEDS, range(3)))
    full = np.full((36, 248320), -40, dtype=np.float32)
    full[:, :20] = np.linspace(20, 1, 20, dtype=np.float32)
    ids = np.tile(np.arange(64, dtype=np.int32), (36, 1))
    scores = full[:, :64].copy()
    history_rows = []
    metadata = {key: [] for key in MODULE.META}
    digests = []
    for context, seed, position in strata:
        history = np.concatenate((corpus[:context],
                                  np.asarray([seed, *range(position)], dtype=np.uint32)))
        meta = dict(context_length=context, lane_id=0, seed=seed,
                    draft_position=position, target_offset=context,
                    draft_offset=context - 1)
        for key, value in meta.items():
            metadata[key].append(value)
        history_rows.append(history)
        digests.append(MODULE.teacher_forced_row_sha256(history, **meta))
    capture = tmp_path / "rows.npz"
    manifest = tmp_path / "rows.json"
    manifest.write_text(json.dumps({"synthetic_fixture": True}))
    arrays = {
        "dense_fc_full_logits": full,
        "target_logits": full.copy(),
        "dense_fc_coarse_candidate_ids": ids,
        "dense_fc_candidate_exact_logits": scores,
        "row_token_ids": np.concatenate(history_rows),
        "row_token_offsets": np.cumsum([0, *map(len, history_rows)], dtype=np.int64),
        **{key: np.asarray(values, np.int64) for key, values in metadata.items()},
        **{key: np.asarray(digests, dtype="S64") for key in MODULE.ARM_DIGESTS},
    }
    np.savez_compressed(capture, **arrays)
    report = MODULE.analyze(capture, manifest, tmp_path)
    assert report["gpu_ab_admissible"]
    assert report["summary"]["rows"] == 36
    assert report["summary"]["max_full_q_support_misses"] == 0
    arrays["row_token_ids"][130] += 1
    np.savez_compressed(capture, **arrays)
    with pytest.raises(ValueError, match="corpus prefix mismatch"):
        MODULE.analyze(capture, manifest, tmp_path)
