"""CPU tests for scripts/prepare_pld_workload.py with a pure fake tokenizer.

No model, weights, MLX, network or real tokenizer is used here.
"""

import json
import subprocess
import sys

import pytest

from scripts import prepare_pld_workload as P
from scripts import qualify_ragged_pld as Q


class FakeTokenizer:
    """Deterministic: one id per character, wrapped like a chat template."""

    def __init__(self, scale=1, as_batch_encoding=True):
        self.scale, self.as_batch_encoding = scale, as_batch_encoding
        self.calls = []

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
        self.calls.append(kwargs)
        text = "<user>" + messages[0]["content"] * self.scale + "</user><assistant>"
        if not tokenize:
            return text
        ids = [ord(c) % 50000 for c in text]
        return {"input_ids": ids} if self.as_batch_encoding else ids


def test_import_is_light():
    probe = subprocess.run(
        [sys.executable, "-c", "import sys, scripts.prepare_pld_workload; "
         "print('mlx.core' in sys.modules, 'transformers' in sys.modules)"],
        capture_output=True, text=True, env={"PYTHONPATH": "src:."},
    )
    assert probe.returncode == 0 and probe.stdout.split() == ["False", "False"], probe.stderr


@pytest.mark.parametrize("lanes,tasks", [
    (2, ["ledger_copy", "capital_sentences"]),
    (4, ["ledger_copy", "code", "count_lines", "capital_sentences"]),
])
def test_build_workload_pins_unequal_lanes_with_hashes(lanes, tasks):
    fake = FakeTokenizer()
    workload = P.build_workload(fake, lanes, 3, {"enable_thinking": False})
    assert [l["task"] for l in workload["lanes"]] == tasks
    assert len(workload["prompts"]) == len(workload["max_tokens"]) == lanes
    assert len({len(p) for p in workload["prompts"]}) == lanes
    assert len(set(workload["max_tokens"])) == lanes and min(workload["max_tokens"]) > 3
    assert all(c == {"enable_thinking": False} for c in fake.calls)
    again = P.build_workload(FakeTokenizer(as_batch_encoding=False), lanes, 3, {"enable_thinking": False})
    assert again["prompts"] == workload["prompts"]  # deterministic, both return shapes
    for receipt, ids in zip(workload["lanes"], workload["prompts"]):
        assert receipt["token_ids_sha256"] == P._sha(json.dumps(ids).encode())
        assert receipt["prompt_tokens"] == len(ids)


def test_texts_follow_the_retained_sanity_constructor():
    source = (P.ROOT / P.SOURCE_CONSTRUCTOR).read_text()
    texts = P.lane_texts(0)
    assert "Count from 1 to 20, one number per line." in source
    assert texts["count_lines"][0] == "Count from 1 to 20, one number per line."
    assert "Ledger line 0: account 0 moved 0 credits." in texts["ledger_copy"][0]
    assert 'f"Ledger line {i}: account {(i * 13 + rnd) % 500}' in source


def test_oversized_or_invalid_output_is_refused():
    with pytest.raises(SystemExit, match="exceeds 16384"):
        P.build_workload(FakeTokenizer(scale=40), 4, 0, {})
    with pytest.raises(SystemExit, match="must be 2 or 4"):
        P.build_workload(FakeTokenizer(), 3, 0, {})

    class Broken(FakeTokenizer):
        def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
            return "text" if not tokenize else {"input_ids": [1, -2]}

    with pytest.raises(SystemExit, match="did not produce token ids"):
        P.build_workload(Broken(), 2, 0, {})


def test_cli_refuses_a_directory_without_tokenizer_files(tmp_path):
    with pytest.raises(SystemExit, match="no local tokenizer files"):
        P.main(["--tokenizer", str(tmp_path), "--out", str(tmp_path / "w.json")])
    with pytest.raises(SystemExit, match="round 0..19"):
        P.main(["--tokenizer", str(tmp_path), "--round", "20", "--out", str(tmp_path / "w.json")])


def test_driver_loads_a_prepared_file_and_refuses_unknown_keys(tmp_path):
    workload = P.build_workload(FakeTokenizer(), 2, 0, {})
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"prompts": workload["prompts"], "max_tokens": workload["max_tokens"],
                                "receipt": {"schema": P.SCHEMA}}))
    prompts, caps, receipt, source = Q.load_workload(good)
    assert prompts == workload["prompts"] and caps == workload["max_tokens"]
    assert receipt == {"schema": P.SCHEMA} and source.startswith("explicit token file sha256 ")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"prompts": [[1, 2]], "max_tokens": [4], "lane_policies": []}))
    with pytest.raises(SystemExit, match="optional receipt"):
        Q.load_workload(bad)
