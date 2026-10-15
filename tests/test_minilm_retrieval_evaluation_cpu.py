"""The diagnostic metric oracle and data pins, without public text fixtures."""

import importlib.util
import math
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/evaluate_minilm_retrieval_cpu.py"
spec = importlib.util.spec_from_file_location("minilm_retrieval_evaluation", PATH)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_metrics_account_for_missing_relevance_and_rank_discount():
    actual = evaluation.metrics(["negative", "relevant", "other"], {"relevant", "missing"})
    assert actual["recall_at_10"] == 0.5
    assert actual["mrr_at_10"] == 0.5
    assert actual["ndcg_at_10"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert evaluation.metrics(["a", "b"], {"c"}) == {
        "recall_at_10": 0, "mrr_at_10": 0, "ndcg_at_10": 0,
    }
    with pytest.raises(ValueError):
        evaluation.metrics(["a", "a"], {"a"})


def test_corpus_order_and_scores_are_independent_of_input_order():
    assert evaluation.stable_order(["a", "b", "c"]) == evaluation.stable_order(["c", "a", "b"])
    scores = evaluation.lexical_scores({"one": "alpha beta", "two": "gamma delta"}, {"q": "alpha"})
    assert scores["q"]["one"] > scores["q"].get("two", 0)


def test_mutated_public_dataset_is_refused_before_decoding(tmp_path):
    (tmp_path / "corpus.jsonl.gz").write_bytes(b"not the reviewed public dataset")
    with pytest.raises(ValueError, match="content mismatch"):
        evaluation.load_dataset(tmp_path)
