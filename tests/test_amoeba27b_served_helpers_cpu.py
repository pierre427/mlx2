"""scripts/amoeba27b_served.py helpers: near-tie classification of width-W vs B1 probes, and timing."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _resp(tokens, windows, text):
    return {"choices": [{"message": {"content": text}, "logprobs": {"content": [
        {"token": t, "logprob": w[t], "top_logprobs": [{"token": k, "logprob": v} for k, v in w.items()]}
        for t, w in zip(tokens, windows)]}}]}


def test_near_tie_probe_classifies(monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    sv = importlib.import_module("amoeba27b_served")
    probes = [("p0", "Paris"), ("p1", None), ("p2", "Paris"), ("p3", "Paris")]
    monkeypatch.setattr(sv, "PROBES", probes)
    w_close = {"A": -0.1, "B": -0.3}
    w_far = {"A": -0.01, "B": -5.0}
    ref = [_resp(["A", "x"], [w_close, {"x": 0, "y": -1}], "Paris x"),
           _resp(["A", "x"], [w_close, {"x": 0, "y": -1}], "two words here"),
           _resp(["A", "x"], [w_far, {"x": 0, "y": -1}], "Paris x"),
           _resp(["A", "x"], [w_close, {"x": 0, "y": -1}], "Paris")]
    arm = [_resp(["A", "x"], [w_close, {"x": 0, "y": -1}], "Paris x"),          # identical
           _resp(["B", "x"], [{"A": -0.2, "B": -0.15}, {"x": 0, "y": -1}], "two other words"),  # near tie
           _resp(["B", "x"], [{"A": -3.0, "B": -0.01}, {"x": 0, "y": -1}], "Paris y"),          # high margin
           _resp(["B", "x"], [{"A": -0.2, "B": -0.15}, {"x": 0, "y": -1}], "Rome")]             # oracle fails
    it = iter(arm)
    monkeypatch.setattr(sv, "post", lambda base, body: next(it))
    _, nt = sv.near_tie_probe("http://x", 2, ref)
    assert nt["counts"] == {"identical": 1, "near_tie_equivalent": 1, "unclassified": 1, "failed": 1,
                            "no_evidence": 0}


def test_perf_of():
    sys.path.insert(0, str(ROOT / "scripts"))
    sv = importlib.import_module("amoeba27b_served")
    recs = [{"mlx2": {"ttft_seconds": 0.5, "elapsed_seconds": 2.5}, "usage": {"completion_tokens": 41}},
            {"mlx2": {"ttft_seconds": 1.5, "elapsed_seconds": 3.5}, "usage": {"completion_tokens": 21}}]
    p = sv.perf_of(recs)
    assert p["ttft_p50_s"] == 1.0 and p["decode_tok_s_per_request_p50"] == 15.0 and p["completion_tokens"] == 62
