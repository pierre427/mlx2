"""Exercise the host-sync arm declarations and report without starting a server."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_launcher = SCRIPTS / "spec_round_hostsync_unit_20261126.sh"
if not _launcher.is_file():
    pytest.skip("the host-sync shell launcher is private", allow_module_level=True)
LAUNCHER = _launcher.read_text()


def test_raon_arms_pin_host_policy_and_both_switches(tmp_path):
    selection = LAUNCHER.split("case $MODEL_KEY in\n", 1)[1].split("\nesac", 1)[0]
    policy = LAUNCHER.split('  if [[ $MODEL_KEY == raon ]]; then\n', 1)[1].split("\n  fi", 1)[0]
    script = f'case $MODEL_KEY in\n{selection}\nesac\n{policy}\nprint -rl -- $ARM_TABLE'
    table = subprocess.check_output(
        ["zsh", "-c", script], text=True,
        env={**os.environ, "MODEL_KEY": "raon", "M": "/models", "RUNS": str(tmp_path)},
    )
    actual = {}
    for line in table.splitlines():
        name, switches, args, *_ = line.split("|")
        assert args == "--external-draft --execution-policy @raon-host"
        env = dict(item.split("=") for item in switches.split())
        actual[name] = env["MLX2_SPEC_BATCHED_GREEDY_LAWS"], env["MLX2_DFLASH2_BATCHED_SELECTOR_READS"]
    assert actual == {"off": ("0", "0"), "greedy-laws": ("1", "0"),
                      "selector-reads": ("0", "1"), "both": ("1", "1")}
    assert json.loads((tmp_path / "raon-host.json").read_text()) == {
        "draft_model": "/models/Raon-1-mini-dflash2", "num_draft": 5, "device_round": False,
        "mla_flash": "off", "mla_decode": "off", "proposal_composition": {
            "prompt_lookup": True, "ngram_min": 3, "ngram_max": 6, "lookback": 256,
            "min_context_match": 4, "max_sources": 8},
    }


@pytest.mark.parametrize(("off", "treatment", "passed"), [
    ([0, 0, 0], [1, 2, 3], True),
    ([0, 0, 0], [1, 1, 3], False),
    ([0, 0, 0], [1, 2, 2], False),
    ([0, 1, 1], [1, 2, 3], False),
    ([0, 0, 0], [None, 2, 3], False),
    ([0, 0, 0], [1, "stale", 3], False),
])
def test_raon_report_requires_measured_greedy_laws(tmp_path, off, treatment, passed):
    row = {"index": 0, "test_index": 0, "text": "2", "token_ids": [2], "completion_tokens": 1,
           "accept_hist": {"0": 1}, "depth_hist": {"1": 1}, "predicted": 2, "finish_reason": "stop"}
    receipt = {"checkpoint": "fixture@sha256:test", "dataset": {"revision": "fixture", "questions": 1,
               "offset": 0}, "driver_revision": {"head": "fixture"},
               "arms": [{"summary": {"questions": 1}, "rows": [row]}]}
    for name, series in (("off", off), ("greedy-laws", treatment)):
        arm = tmp_path / name
        (arm / "gsm8k").mkdir(parents=True)
        (arm / "gsm8k" / "receipt.json").write_text(json.dumps(receipt))
        for stage, rounds in zip(("before-measure", "final", "gsm8k"), series, strict=True):
            if rounds is not None:
                stats = {"external_rounds": 10}
                fresh = rounds != "stale"
                if not fresh:
                    rounds = 2
                if rounds:
                    stats.update(external_batched_greedy_law_rounds=rounds,
                                 external_batched_greedy_law_rows=3 * rounds)
                (arm / f"status-{stage}.json").write_text(json.dumps({"scheduler": stats, "snapshot_read": {"fresh": fresh}}))
    report = LAUNCHER.split("<<'COMPARE'", 1)[1].split("\n", 1)[1].split("\nCOMPARE\n", 1)[0]
    result = subprocess.run(
        [sys.executable, "-", str(tmp_path), "off,greedy-laws", "raon"],
        input=report, text=True, check=False, env={**os.environ, "PYTHONPATH": str(SCRIPTS)},
    )
    assert result.returncode == (0 if passed else 1)
    evidence = json.loads((tmp_path / "engagement.json").read_text())
    assert all(arm["greedy_laws"]["passed"] for arm in evidence.values()) is passed
