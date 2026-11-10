"""Ablation-ladder harness: comparison classes, oracles, sweep expansion (CPU, no server)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ablation_ladder", ROOT / "scripts/research/ds41_atlas/ablation_ladder.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)


def tok(t, lp, top):
    return {"token": t, "logprob": lp, "top": top}


def test_identical_and_failed():
    a = [tok("x", 0, [("x", 0), ("y", -9)])]
    assert L.compare(a, a, True, True)["class"] == "identical"
    assert L.compare(a, a, True, False)["class"] == "failed"


def test_near_tie_vs_real_divergence():
    base = [tok("a", 0, [("a", 0), ("b", -9)]), tok("b", -0.3, [("b", -0.3), ("c", -0.6)])]
    near = [tok("a", 0, [("a", 0), ("b", -9)]), tok("c", -0.4, [("c", -0.4), ("b", -0.5)])]
    far = [tok("a", 0, [("a", 0), ("b", -9)]), tok("z", -0.1, [("z", -0.1), ("b", -3.0)])]
    r = L.compare(base, near, True, True)
    assert r["class"] == "near_tie_equivalent" and r["first_divergence"] == 1
    assert L.compare(base, far, True, True)["class"] == "diverged_pass"
    assert L.compare(base, far, True, False)["class"] == "diverged_fail"
    assert L.compare(base, near, True, False)["class"] == "diverged_fail"


def test_prefix_length_difference():
    base = [tok("a", 0, [("a", 0), ("b", -9)])]
    longer = base + [tok("b", 0, [("b", 0), ("c", -9)])]
    r = L.compare(base, longer, True, True)
    assert r["class"] == "diverged_pass" and r["first_divergence"] == 1


def test_oracles():
    o = L.oracles({"gsm8k-6": {"reference": "260"}})
    assert o["translate-6"][1]("The train to Munich is fifty minutes late due to a signal failure.")
    assert not o["translate-6"][1]("The train is late.")
    assert o["translate-7"][1]("A biblioteca estendeu seu horário de funcionamento durante o período de exames.")
    assert o["extract-7"][1]("```yaml\nname: Maple Server\nport: 8443\ntls: enabled\nadmins:\n  - alice\n  - bob\nretries: 5\n```")
    assert not o["extract-7"][1]("```yaml\nname: Maple Server\nport: 8080\n```")
    assert o["gsm8k-6"][1]("... so the total is 260 sheep.") and o["gsm8k-6"][1]("#### 260")
    assert not o["gsm8k-6"][1]("The answer is 250.")


def test_sweep_expansion_and_merge():
    spec = {"arms": [{"name": "exact", "policy": {}}],
            "sweeps": [{"name": "l13_26_q4_$N", "values": [256, 64],
                        "template": {"tiers": {"overrides": [{"layers": [13, 26], "q4_experts": "$N", "tag": "q4=$N"}]}}}]}
    arms = L.expand(spec)
    assert [a["name"] for a in arms] == ["exact", "l13_26_q4_256", "l13_26_q4_64"]
    ov = arms[2]["policy"]["tiers"]["overrides"][0]
    assert ov["q4_experts"] == 64 and ov["tag"] == "q4=64"
    base = {"deepseek_source_experts": {"mode": "original_lfu", "budget_bytes": 1}}
    m = L.merged(base, {"deepseek_source_experts": {"budget_bytes": 2}, "x": 1})
    assert m["deepseek_source_experts"] == {"mode": "original_lfu", "budget_bytes": 2} and m["x"] == 1
    assert base["deepseek_source_experts"]["budget_bytes"] == 1


def test_gsm8k_references_found():
    prompts, gsm = L.load_prompts(["gsm8k-6", "gsm8k-7", "translate-6"])
    assert set(gsm) == {"gsm8k-6", "gsm8k-7"} and all(v["reference"] for v in gsm.values())


def test_resident_pass_needs_zero_reads_from_every_bank():
    zero = {"v41_expert_cache_misses": 0, "v41_expert_cache_hits": 900, "v41_expert_bytes_read": 0,
            "v41_expert_demand_wait_ns": 0, "v41_expert_exl3_bytes_read": 0}
    assert L.resident(zero)
    assert not L.resident({**zero, "v41_expert_cache_misses": 1})
    assert not L.resident({**zero, "v41_expert_bytes_read": 18_800_000})
    assert not L.resident({**zero, "v41_expert_exl3_bytes_read": 13_315_584})


def test_warm_summary_reports_resident_passes(tmp_path):
    t = [tok("x", 0, [("x", 0), ("y", -9)])]
    ex = {"v41_expert_cache_misses": 0, "v41_expert_demand_wait_ns": 0}
    for arm, rp, dec in (("base", [3, 3], 2.0), ("exl3", [2, 3], 3.0)):
        (tmp_path / arm).mkdir()
        L._write(tmp_path / arm / "replies.json", {"p": {"reply": "x", "finish": "stop", "tokens": t, "decode_tok_s": dec,
                                                       "ttft_s": 1.5, "completion_tokens": 1, "experts": ex,
                                                       "resident_passes": rp, "tokens_stable": True}})
        L._write(tmp_path / arm / "done.json", {})
    rows = L.summarize(tmp_path, ["base", "exl3"], {"p": ("any", lambda _r: True)})
    assert rows[1]["resident_passes"] == [2, 3] and rows[1]["median_ttft_s"] == 1.5
    assert "resident passes" in (tmp_path / "summary.md").read_text()
