"""CPU checks for the GPU-gated interior-checkpoint qualification harness."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qualify_interior_checkpoints.py"


def _module():
    spec = importlib.util.spec_from_file_location("qualify_interior", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(index, *, hits, ttft, match=True, prompt=1000, cached=0):
    return {
        "index": index, "ttft_s": ttft, "prompt_tokens": prompt,
        "cached_tokens": cached, "matches_cold": match,
        "mechanism": {
            "interior_hits": hits, "turn_boundary_hits": hits,
            "interior_resident_bytes": 10, "memory_admission_deferred": 0,
        },
    }


def _run(arm, hits, ttft, *, match=True):
    return {
        "arm": arm,
        "workloads": {
            name: [_row(0, hits=0, ttft=1.0), _row(1, hits=hits, ttft=ttft, match=match)]
            for name in ("shared_system", "rag", "linear")
        },
    }


def test_harness_refuses_a_null_interior_arm_and_requires_exactness():
    module = _module()
    names = ["shared_system", "rag", "linear"]
    null = module.summarize([_run("off", 0, 1.0), _run("auto", 0, 0.2)], names)
    assert null["gates"]["refused_arms"] and not null["gates"]["go"]

    good = module.summarize([_run("off", 0, 1.0), _run("auto", 1, 0.2)], names)
    assert good["gates"]["refused_arms"] == []
    assert good["gates"]["ttft_cut"]["auto/shared_system"] > 0.3
    # The linear control is in this fixture too, and it also got faster.
    assert good["gates"]["go"]

    wrong = module.summarize([_run("off", 0, 1.0), _run("auto", 1, 0.2, match=False)], names)
    assert wrong["gates"]["correctness_diffs"] > 0 and not wrong["gates"]["go"]


def test_harness_requires_gpu_ownership_and_supports_dry_run(tmp_path):
    base = [sys.executable, str(SCRIPT), "--model", "unused", "--out", str(tmp_path / "o.json")]
    refused = subprocess.run(base, capture_output=True, text=True)
    assert refused.returncode == 2 and "--i-own-the-gpu" in refused.stderr
    dry = subprocess.run(base + ["--dry-run", "--rounds", "2"], capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    plan = json.loads(dry.stdout)
    # Arms interleave A B C / C B A across rounds.
    assert [arm for _, arm in plan["schedule"]] == ["off", "pow2", "auto", "auto", "pow2", "off"]
    assert plan["policies"]["auto"]["apc_interior_checkpoints"] == "auto"
    assert not (tmp_path / "o.json").exists()


def test_harness_workloads_are_deterministic_and_share_prefixes():
    module = _module()
    first, second = module.build_workloads(0.05), module.build_workloads(0.05)
    assert first == second
    shared = first["shared_system"]
    assert all(messages[0] == shared[0][0] for messages in shared)
    assert len({messages[1]["content"] for messages in shared}) == len(shared)


def test_harness_refuses_prompts_larger_than_the_server_context(tmp_path):
    # Regression: the original default --scale 1.0 built a 167K-token RAG
    # prompt and a 262K-token linear turn against --max-context 65536.
    base = [sys.executable, str(SCRIPT), "--model", "unused", "--out", str(tmp_path / "o.json"),
            "--dry-run"]
    too_big = subprocess.run(base + ["--scale", "1.0"], capture_output=True, text=True)
    assert too_big.returncode == 2 and "--max-context" in too_big.stderr
    default = subprocess.run(base, capture_output=True, text=True)
    assert default.returncode == 0, default.stderr
    plan = json.loads(default.stdout)
    assert plan["scale"] == 0.2
    assert max(plan["max_prompt_chars"].values()) / 1.2 < 65536


def test_harness_requests_name_the_served_model():
    # Regression: the first GPU smoke sent model="default" and the server
    # answered 404 "unknown model" to the first chat request.
    module = _module()
    body = module.chat_body([{"role": "user", "content": "hi"}], model="served", max_tokens=4)
    assert body["model"] == "served"
    assert body["temperature"] == 0 and body["stream"] is True
    source = SCRIPT.read_text()
    assert '"default"' not in source
    # warm, cold reference, and the cold determinism replay
    assert source.count("model=served_model") == 3


def test_harness_reports_cold_replay_diffs_separately():
    module = _module()
    names = ["shared_system"]
    rows = [_row(0, hits=0, ttft=1.0), _row(1, hits=1, ttft=0.2)]
    rows[0]["cold_matches_cold"] = False
    summary = module.summarize(
        [{"arm": "off", "workloads": {"shared_system": [_row(0, hits=0, ttft=1.0)]}},
         {"arm": "auto", "workloads": {"shared_system": rows}}],
        names,
    )
    # A model that will not reproduce its own greedy output is reported, not
    # silently folded into the exactness verdict.
    assert summary["gates"]["cold_replay_diffs"] == 1
    assert summary["gates"]["correctness_diffs"] == 0


def test_cold_reference_is_taken_once_per_arm_and_round():
    # Regression: references keyed only by (arm, workload, index) compared a
    # round-1 warm answer with a round-0 cold answer from a different server
    # process, which this model does not reproduce (6 false diffs, including
    # requests with no cache hit).
    source = SCRIPT.read_text()
    assert "key = (arm, round_index, name, index)" in source


def test_server_subprocess_is_pinned_to_this_checkout(tmp_path, monkeypatch):
    # Regression: the server was spawned with an inherited environment, so a
    # queued run from a worktree could import mlx2 from another checkout.
    module = _module()
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else/src")
    first = module.server_env()["PYTHONPATH"].split(os.pathsep)
    assert first[0] == str(SCRIPT.resolve().parents[1] / "src")
    assert "/somewhere/else/src" in first
