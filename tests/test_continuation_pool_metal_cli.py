"""CPU-only validation harness admission, precision pins and probe rollback."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2.runtime.proposal_pool import (
    PoolRound,
    ProposalPath,
    ProposalPool,
    ProposalSession,
    ProposalSource,
)

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/validate_continuation_pool_metal.py"
)
spec = importlib.util.spec_from_file_location("continuation_pool_metal_cli", SCRIPT)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64


def args(*extra):
    return harness.parser().parse_args(
        [
            "--model",
            "/missing-target",
            "--draft",
            "/missing-draft",
            "--out",
            "/missing-output",
            *extra,
        ]
    )


def draft():
    session = ProposalSession(A, B, C, 100)
    source = ProposalSource("external", "xpress", D, A, B, C, 100)
    return SimpleNamespace(
        session=session,
        source_records={"external": source},
        proposal_pool=ProposalPool(session, [source]),
        last_continuation_selections=(),
    )


def test_execution_requires_explicit_gpu_ownership():
    with pytest.raises(ValueError, match="i-own-the-gpu"):
        harness.preflight(args())


def test_row_exact_candidate_is_explicit_and_reported_before_tensor_load():
    assert harness.preflight(args("--dry-run"))["target_verify_row_exact"] is False
    assert (
        harness.preflight(args("--dry-run", "--target-verify-row-exact"))[
            "target_verify_row_exact"
        ]
        is True
    )


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--context-tokens", "4097"),
        ("--context-tokens", "15"),
        ("--prefill-step", "15"),
        ("--prefill-step", "513"),
        ("--max-tokens", "16"),
        ("--max-tokens", "49"),
        ("--repetitions", "0"),
        ("--repetitions", "4"),
        ("--deadline-seconds", "901"),
        ("--deadline-seconds", "0"),
    ],
)
def test_cells_are_bounded(flag, value):
    with pytest.raises(ValueError):
        harness.preflight(args("--dry-run", flag, value))


@pytest.mark.parametrize("precision", ["artifact", "float32-diagnostic"])
def test_dry_run_under_hard_tensor_import_guard(precision):
    code = """
import sys,runpy
class Guard:
 def find_spec(self,fullname,*args,**kwargs):
  if fullname.startswith(('mlx','numpy','transformers')):raise RuntimeError(fullname)
sys.meta_path.insert(0,Guard())
sys.argv=[sys.argv[1],'--model','/missing-target','--draft','/missing-draft','--out','/missing-output','--dry-run','--compute-precision',sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT), precision],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["will_execute"] is False
    assert plan["max_sequences"] == plan["max_depth"] == 15
    assert plan["cost_path_widths"] == [1, 5, 15]
    assert plan["cost_depths"] == list(range(16))
    assert plan["performance_claim"] is False
    assert plan["compute_precision"] == precision


def test_precision_namespace_changes_all_source_records_and_isolates_critic():
    one, two = draft(), draft()
    old = one.proposal_pool.ranking_registry
    artifact = harness.bind_compute_precision(one, "artifact")
    fp32 = harness.bind_compute_precision(two, "float32-diagnostic")
    assert artifact != fp32
    assert (
        one.session.session_revision
        == one.source_records["external"].session_revision
        == artifact
    )
    assert (
        two.session.session_revision
        == two.source_records["external"].session_revision
        == fp32
    )
    assert one.proposal_pool.ranking_registry is not old
    assert two.proposal_pool.ranking_registry is not old
    assert one.proposal_pool.sources["external"] == one.source_records["external"]


def test_precision_binding_refuses_pending_selections():
    d = draft()
    p = d.proposal_pool
    p.select([ProposalPath("p", "external", D, A, B, (1,))], PoolRound(A, B, "r", 0))
    with pytest.raises(ValueError, match="precede"):
        harness.bind_compute_precision(d, "float32-diagnostic")


class Probe:
    def __init__(self, fail=False):
        self.draft = draft()
        self.fail = fail
        self.restores = 0
        self._continuation_open_selections = []

    def _snapshot_round(self, lanes):
        return copy.deepcopy(vars(lanes[0]))

    def _restore_round(self, lanes, snapshot):
        lanes[0].__dict__.clear()
        lanes[0].__dict__.update(copy.deepcopy(snapshot))
        self.restores += 1

    def _propose(self, lanes):
        lanes[0].state += 1
        paths = [ProposalPath(str(i), "external", D, A, B, (i,)) for i in range(15)]
        selection = self.draft.proposal_pool.select(paths, PoolRound(A, B, "probe", 0))
        self._continuation_open_selections.append(selection)
        self.draft.last_continuation_selections = (selection,)
        if self.fail:
            raise RuntimeError("later provider failed")
        return [SimpleNamespace(continuation_selection=selection)]


def test_actual_proposal_cost_discards_tickets_restores_state_no_feedback():
    probe = Probe()
    lane = SimpleNamespace(state=7)
    clock = iter([0.0, 1.0, 2.0, 4.0, 5.0, 9.0])
    result = harness.measure_proposal_cost(
        probe, lane, lambda: None, 2, lambda: next(clock)
    )
    assert result["seconds"] == 3.0 and result["samples_seconds"] == [2.0, 4.0]
    assert result["selected_sequence_counts"] == [15, 15, 15]
    assert lane.state == 7 and probe.restores == 3
    assert not probe.draft.proposal_pool._pending
    assert probe.draft.proposal_pool.feedback_revision == 0
    assert probe.draft.proposal_pool.ranking_registry.revision == 0
    assert result["critic_observations_published"] is False


def test_partial_provider_failure_cleanup_restores_state_and_pending_ticket():
    probe = Probe(fail=True)
    lane = SimpleNamespace(state=7)
    with pytest.raises(RuntimeError, match="provider"):
        harness.measure_proposal_cost(probe, lane, lambda: None, 1, lambda: 0.0)
    assert lane.state == 7 and probe.restores == 1
    assert not probe.draft.proposal_pool._pending
    assert probe.draft.proposal_pool.ranking_registry.revision == 0


def test_real_cpu_provider_probe_restores_committed_boundary_and_pins_precision():
    import mlx.core as mx
    from test_external_continuation_pool_cpu import batch, prefill

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    generator = None
    try:
        _model, _base, d, generator = batch()
        pin = harness.bind_compute_precision(d, "float32-diagnostic")
        lane = prefill(generator, [[1, 2, 3]])[0]
        offsets = [cache.offset for cache in lane.draft_cache]
        history = list(lane.history)
        rng = lane.rng.snapshot()
        anchor = lane.anchor
        result = harness.measure_proposal_cost(generator, lane, lambda: None, 1)
        assert result["selected_sequence_counts"] == [15, 15]
        assert result["seconds"] > 0
        assert [cache.offset for cache in lane.draft_cache] == offsets
        assert lane.history == history and lane.anchor == anchor
        assert lane.rng.snapshot() == rng
        assert not d.proposal_pool._pending
        assert (
            d.proposal_pool.feedback_revision
            == d.proposal_pool.ranking_registry.revision
            == 0
        )
        assert d.session.session_revision == pin
    finally:
        if generator is not None:
            generator.close()
        mx.set_default_device(previous)


def test_ordinary_reached_reference_reserves_prompt_anchor_and_steps_emitted_s1():
    import numpy as np

    class Model:
        def __init__(self):
            self.calls = []

        def make_cache(self):
            return [SimpleNamespace(offset=0)]

        def __call__(self, inputs, cache):
            self.calls.append(inputs.tolist())
            cache[0].offset += inputs.shape[1]
            return np.zeros((1, inputs.shape[1], 3), dtype=np.float32)

        def forward_with_taps(self, *_args, **_kwargs):
            raise AssertionError("reference must use original ordinary entry")

    model = Model()
    fake_mx = SimpleNamespace(array=np.array, eval=lambda *_: None, float32=np.float32)
    rows = list(
        harness.ordinary_reached_rows(model, fake_mx, [1, 2, 3, 4, 5], [6, 7, 8], 2)
    )
    assert model.calls == [[[1, 2]], [[3, 4]], [[5]], [[6]], [[7]]]
    assert [history for history, _, _ in rows] == [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6, 7],
    ]
    assert [offsets for _, _, offsets in rows] == [[5], [6], [7]]


def test_every_reached_audit_excludes_unmatched_future_rows_and_detects_late_drift():
    import numpy as np

    prompt, emitted = [1, 2], [3, 4, 5]
    law = np.array([0.2, 0.3, 0.5])
    raw = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    observations = [
        {
            "history_tokens": prompt + emitted[:i],
            "reachable": True,
            "raw_logits": raw.copy(),
            "law": law.copy(),
        }
        for i in range(3)
    ]
    # Callback rows for a discarded draft branch are not reached by this walk.
    observations.append(
        {"history_tokens": [1, 2, 9], "reachable": True, "raw_logits": raw, "law": law}
    )
    observations.append(
        {"history_tokens": prompt, "reachable": False, "raw_logits": raw, "law": law}
    )
    observations[-3]["law"] = np.array([0.3, 0.3, 0.4])
    observations[-3]["raw_logits"][0] += 0.5
    references = [(prompt + emitted[:i], raw, [2 + i]) for i in range(3)]
    result = harness.audit_reached_laws(
        observations, prompt, emitted, references, lambda _: law, temperature=0.8
    )
    assert result["first_processed_law_l1_vs_ordinary"] == 0
    assert (
        result["reached_prefix_count"] == result["compared_reached_prefix_count"] == 3
    )
    assert result["unreached_callback_count"] == 2
    assert result["candidate_callback_count"] == 5
    assert not result["every_reached_law_matches_ordinary"]
    assert result["prefix_diagnostics"][2]["raw_logits_max_abs_vs_ordinary"] == 0.5
    assert result["maximum_processed_law_l1_vs_ordinary"] == pytest.approx(0.2)


def test_reached_audit_missing_callback_and_history_mismatch_fail_closed():
    import numpy as np

    raw = np.array([0.0, 1.0])
    law = np.array([0.25, 0.75])
    result = harness.audit_reached_laws(
        [], [1], [2], [([1], raw, [1])], lambda _: law, temperature=0.8
    )
    assert result["missing_reached_prefixes"] == [0]
    assert result["first_processed_law_l1_vs_ordinary"] is None
    assert not result["every_reached_law_matches_ordinary"]
    with pytest.raises(ValueError, match="history"):
        harness.audit_reached_laws(
            [], [1], [2], [([9], raw, [1])], lambda _: law, temperature=0.8
        )


def test_max_absolute_law_gate_and_raw_value_equality_are_independent():
    import numpy as np

    raw = np.array([0.0, 1.0], dtype=np.float32)
    law = np.array([0.25, 0.75])
    observation = {
        "history_tokens": [1],
        "reachable": True,
        "raw_logits": raw.copy(),
        "law": law + [1.1e-4, -1.1e-4],
    }
    result = harness.audit_reached_laws(
        [observation], [1], [2], [([1], raw, [1])], lambda _: law, temperature=0.8
    )
    assert result["maximum_processed_law_l1_vs_ordinary"] < 0.01
    assert result["maximum_processed_law_max_abs_vs_ordinary"] > 1e-4
    assert not result["every_reached_law_matches_ordinary"]
    assert result["every_reached_raw_logit_values_equal"]
    assert result["processed_law_tolerances"] == {"l1": 0.01, "max_absolute": 1e-4}
    observation["law"] = law.copy()
    observation["raw_logits"][0] = 0.5
    result = harness.audit_reached_laws(
        [observation], [1], [2], [([1], raw, [1])], lambda _: law, temperature=0.8
    )
    assert result["every_reached_law_matches_ordinary"]
    assert not result["every_reached_raw_logit_values_equal"]
    assert result["prefix_diagnostics"][0]["raw_logits_max_abs_vs_ordinary"] == 0.5


def test_actual_cpu_pool_all_sampled_draws_match_original_prefix_s1_reference():
    import mlx.core as mx
    import numpy as np
    from test_external_continuation_pool_cpu import batch, prefill

    from mlx2.runtime.sample_utils import LaneRNG, make_transformed_logprobs
    from mlx2.runtime.speculative_sampling import probability

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    generator = None
    try:
        model, _base, _draft, generator = batch()
        prompt = [1, 2, 3]
        lane = prefill(
            generator,
            [prompt],
            sampling_configs=[{"sampling_temp": 0.8}],
            lane_rngs=[LaneRNG(329)],
        )[0]
        observations = []
        original = generator._target_law

        def tracked(current, logits, history, *args, **kwargs):
            law = original(current, logits, history, *args, **kwargs)
            observations.append(
                {
                    "history_tokens": [*history, *kwargs.get("history_suffix", ())],
                    "reachable": kwargs.get("reachable", args[0] if args else True),
                    "raw_logits": np.asarray(logits.astype(mx.float32)).copy(),
                    "law": law.copy(),
                }
            )
            return law

        generator._target_law = tracked
        while lane.generated < lane.maximum:
            generator._round([lane])
        emitted = [response.token for response in lane.ready]
        transform = make_transformed_logprobs(0.8)
        result = harness.audit_reached_laws(
            observations,
            prompt,
            emitted,
            harness.ordinary_reached_rows(model, mx, prompt, emitted, 64),
            lambda logits: probability(np.asarray(mx.exp(transform(logits[None])[0]))),
            temperature=0.8,
        )
        assert result["every_reached_law_matches_ordinary"]
        assert result["compared_reached_prefix_count"] == len(emitted) == lane.rng.draws
        assert (
            result["unreached_callback_count"]
            == result["duplicate_reached_callback_count"]
            == 0
        )
        assert len(emitted) > 1
        assert (
            max(
                row["raw_logits_max_abs_vs_ordinary"]
                for row in result["prefix_diagnostics"]
            )
            < 1e-5
        )
        assert any(row["emitted_position"] > 0 for row in result["prefix_diagnostics"])
    finally:
        if generator is not None:
            generator.close()
        mx.set_default_device(previous)


@pytest.mark.parametrize("context", [128, 1024, 4096])
@pytest.mark.parametrize("width", [1, 2, 4])
def test_context_ladder_exact_geometry_preflight(context, width):
    selected = args(
        "--dry-run", "--context-tokens", str(context), "--batch-size", str(width)
    )
    plan = harness.preflight(selected)
    assert plan["context_tokens"] == context
    assert plan["batch_size"] == width
    assert plan["prefill_step"] == 128
    assert plan["max_sequences"] == 15
    assert plan["memory_admission"]["model_constructed"] is False


def test_default_geometry_preserves_128_chunk_and_two_request_batch():
    selected = args("--dry-run")
    assert (selected.context_tokens, selected.prefill_step, selected.batch_size) == (
        128,
        128,
        2,
    )
    with pytest.raises(SystemExit):
        args("--dry-run", "--batch-size", "3")


def test_long_context_dry_run_reads_no_artifact_and_imports_no_tensor_module():
    code = """
import sys,runpy
from pathlib import Path
class Guard:
 def find_spec(self,fullname,*a,**k):
  if fullname.startswith(('mlx','numpy','transformers','psutil')):raise RuntimeError(fullname)
sys.meta_path.insert(0,Guard())
original=Path.open
def guarded(path,*a,**k):
 if str(path).startswith('/missing-'):raise RuntimeError('artifact read')
 return original(path,*a,**k)
Path.open=guarded
sys.argv=[sys.argv[1],'--model','/missing-target','--draft','/missing-draft','--out','/missing-output','--dry-run','--context-tokens','4096','--batch-size','1']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["context_tokens"] == 4096


@pytest.mark.parametrize("length", [128, 1024, 4096])
def test_prompt_tokens_extend_to_exact_length_preserving_legacy_prefix(length):
    class Tokenizer:
        def __init__(self):
            self.calls = []

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            self.calls.append(len(text))
            return [ord(character) for character in text]

    tokenizer = Tokenizer()
    text = "xy "
    actual = harness.exact_prompt_tokens(tokenizer, text, length)
    assert len(actual) == length
    legacy = tokenizer.encode(text * 32, add_special_tokens=False)
    assert actual[: min(len(legacy), length)] == legacy[:length]
    assert tokenizer.calls[0] == len(text) * 32
    if length == 128:
        assert actual == [ord(character) for character in (text * 64)[:128]]


def test_legacy_128_prompt_is_unchanged_when_initial_corpus_is_long_enough():
    tokenizer = SimpleNamespace(encode=lambda text, **_: [ord(value) for value in text])
    text = "abcdef "
    assert (
        harness.exact_prompt_tokens(tokenizer, text, 128)
        == tokenizer.encode(text * 32)[:128]
    )


@pytest.mark.parametrize("tokens", [[], [True], [-1], [1.0]])
def test_prompt_tokenizer_invalid_output_fails_closed(tokens):
    tokenizer = SimpleNamespace(encode=lambda *_a, **_k: tokens)
    with pytest.raises(ValueError, match="token"):
        harness.exact_prompt_tokens(tokenizer, "hello", 4096)


def memory_configs():
    target = {
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 2560,
        "vocab_size": 151936,
    }
    draft_config = {
        **target,
        "num_hidden_layers": 5,
        "dflash_config": {"target_layer_ids": [1, 9, 17, 25, 33]},
    }
    return target, draft_config


@pytest.mark.parametrize("width", [1, 2, 4])
def test_memory_forecast_reserves_full_15_paths_per_request_and_rollback(width):
    selected = args(
        "--dry-run",
        "--context-tokens",
        "4096",
        "--max-tokens",
        "48",
        "--batch-size",
        str(width),
    )
    target, companion = memory_configs()
    result = harness.context_memory_forecast(target, companion, 9_000_000_000, selected)
    assert result["kv_capacity_tokens"] == 4352
    assert result["target_row_bytes"] == 4352 * 144 * 1024
    assert result["draft_row_bytes"] == 4352 * 20 * 1024
    assert result["maximal_cohort_branch_reservation"] == 15 * width
    assert result["phases"]["generation"]["target_rows"] == 4 + 16 * width
    assert result["phases"]["cost_probe"]["target_rows"] == 16
    assert result["phases"]["ordinary_law_audit"]["target_rows"] == 5
    assert result["required_bytes"] == max(
        item["required_bytes"] for item in result["phases"].values()
    )
    assert "not observed" in result["authority"]


def test_memory_forecast_uses_each_plane_actual_element_width():
    target, companion = memory_configs()
    selected = args("--dry-run", "--context-tokens", "1024")
    f16 = harness.context_memory_forecast(
        target, companion, 10, selected, target_element_bytes=2, draft_element_bytes=2
    )
    mixed = harness.context_memory_forecast(
        target, companion, 10, selected, target_element_bytes=4, draft_element_bytes=2
    )
    assert mixed["target_row_bytes"] == 2 * f16["target_row_bytes"]
    assert mixed["draft_row_bytes"] == f16["draft_row_bytes"]
    with pytest.raises(ValueError, match="element"):
        harness.context_memory_forecast(
            target, companion, 10, selected, target_element_bytes=1
        )


@pytest.mark.parametrize(
    "available,recommended", [(0, 10**12), (10**12, 0), (1, 10**12), (10**12, 1)]
)
def test_unsafe_geometry_refused_before_constructor_or_policy_export(
    monkeypatch, available, recommended
):
    target, companion = memory_configs()
    monkeypatch.setattr(
        harness,
        "artifact_memory_metadata",
        lambda _: (target, companion, 9_000_000_000, 2, 2),
    )
    selected = args("--dry-run", "--context-tokens", "4096", "--batch-size", "1")
    report = harness.preflight(selected)
    called = []
    with pytest.raises(ValueError, match="15-path"):
        harness.construct_admitted_adapter(
            selected,
            report,
            {"max_recommended_working_set_size": recommended},
            lambda *_a, **_k: called.append(True),
            available,
        )
    assert not called
    assert report["memory_admission"]["status"] == "refused"
    assert report["skipped_geometries"] == [
        {
            "reason": "memory_admission",
            "context_tokens": 4096,
            "batch_size": 1,
            "paths_per_request": 15,
            "cost_policy_exported": False,
            "executed": False,
        }
    ]
    assert "continuation_costs" not in report


def test_memory_exact_boundary_admits_and_keeps_full_policy(monkeypatch):
    target, companion = memory_configs()
    selected = args("--dry-run", "--context-tokens", "4096", "--batch-size", "1")
    monkeypatch.setattr(
        harness,
        "artifact_memory_metadata",
        lambda _: (target, companion, 9_000_000_000, 2, 2),
    )
    required = harness.context_memory_forecast(
        target, companion, 9_000_000_000, selected
    )["required_bytes"]
    captured = []
    sentinel = object()

    def factory(*a, **k):
        captured.append((a, k))
        return sentinel

    report = harness.preflight(selected)
    assert (
        harness.construct_admitted_adapter(
            selected,
            report,
            {"max_recommended_working_set_size": required},
            factory,
            required,
        )
        is sentinel
    )
    assert report["memory_admission"]["status"] == "admitted"
    assert captured[0][1]["execution_policy"]["continuation_pool"]["limit"] == 15


def test_managed_batch_prefill_and_hook_cleanup_on_failure():
    selected = args("--dry-run", "--context-tokens", "4096", "--batch-size", "1")
    original = object()
    batch = SimpleNamespace(_target_law=original, close=lambda: closed.append(True))
    captured, closed = [], []

    def factory(**kwargs):
        captured.append(kwargs)
        return batch

    adapter = SimpleNamespace(create_external_batch=factory)
    with (
        pytest.raises(RuntimeError, match="late"),
        harness.managed_batch(adapter, selected) as value,
    ):
        value._target_law = object()
        raise RuntimeError("late audit failure")
    assert batch._target_law is original
    assert closed == [True]
    assert captured[0]["prefill_step_size"] == 128
    assert captured[0]["completion_batch_size"] == 1


@pytest.mark.parametrize("length", [1024, 4096])
def test_long_context_original_oracle_chunking_is_independent_of_context(length):
    import numpy as np

    calls = []

    class Model:
        def make_cache(self):
            return [SimpleNamespace(offset=0)]

        def __call__(self, inputs, cache):
            calls.append(inputs.tolist())
            cache[0].offset += inputs.shape[1]
            return np.zeros((1, inputs.shape[1], 3), dtype=np.float32)

    model = Model()
    mx = SimpleNamespace(array=np.array, eval=lambda *_: None, float32=np.float32)
    prompt = list(range(length))
    rows = list(harness.ordinary_reached_rows(model, mx, prompt, [7, 8], 128))
    assert sum(len(value[0]) for value in calls[:-2]) == length - 1
    assert max(len(value[0]) for value in calls[:-2]) == 128
    assert calls[-2:] == [[[length - 1]], [[7]]]
    assert [offsets for _, _, offsets in rows] == [[length], [length + 1]]


@pytest.mark.parametrize("dtype,bytes_per", [("BF16", 2), ("F16", 2), ("F32", 4)])
@pytest.mark.parametrize("precision", ["artifact", "float32-diagnostic"])
def test_artifact_memory_headers_preserve_valid_floating_precision(
    tmp_path, monkeypatch, dtype, bytes_per, precision
):
    import struct

    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    target, companion = memory_configs()
    paths = []
    for name, config in (("target", target), ("draft", companion)):
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config))
        header = json.dumps(
            {
                "weight": {
                    "dtype": dtype,
                    "shape": [2, 3],
                    "data_offsets": [0, 6 * bytes_per],
                }
            }
        ).encode()
        (path / "model.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + b"\0" * (6 * bytes_per)
        )
        paths.append(path)
    selected = args(
        "--dry-run",
        "--model",
        str(paths[0]),
        "--draft",
        str(paths[1]),
        "--compute-precision",
        precision,
    )
    actual = harness.artifact_memory_metadata(selected)
    expected = 4 if precision == "float32-diagnostic" else bytes_per
    assert actual[:2] == (target, companion)
    assert actual[2:] == (12 * expected, expected, expected)


def test_real_tiny_cost_probe_failure_closes_batch_and_discards_snapshots(monkeypatch):
    import mlx.core as mx
    import numpy as np
    from test_external_continuation_pool_cpu import batch

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    generator = None
    try:
        _model, _base, _draft, generator = batch()
        original = generator._target_law
        closed = []
        native_close = generator.close

        def close():
            closed.append(True)
            native_close()

        generator.close = close
        adapter = SimpleNamespace(
            model=generator.model, create_external_batch=lambda **_: generator
        )

        def fail(*_a, **_k):
            generator._target_law = object()
            raise RuntimeError("later cost probe failure")

        monkeypatch.setattr(harness, "measure_proposal_cost", fail)
        with pytest.raises(RuntimeError, match="later cost"):
            harness.measure_context_costs(
                adapter, args("--dry-run"), [1, 2, 3], mx, np, {}
            )
        assert closed == [True]
        assert generator._target_law == original
        assert not generator.lanes and not generator.boundaries
        assert not generator.draft.proposal_pool._pending
    finally:
        if generator is not None:
            generator.close()
        mx.set_default_device(previous)


def test_adapter_and_target_hook_restored_on_later_generation_failure(monkeypatch):
    import types

    import mlx2.adapters.standard_decoder as standard_adapter

    original = lambda *_a, **_k: None
    model = SimpleNamespace(forward_with_taps=original)
    closed = []
    adapter = SimpleNamespace(
        model=model,
        draft_model=SimpleNamespace(receipt_settings={}),
        identity={"fingerprint": A},
        tokenizer=SimpleNamespace(encode=lambda *_a, **_k: [1] * 4096),
        close=lambda: closed.append(True),
    )
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.gpu = "fake_gpu_spy_no_real_device_calls"
    fake_mx.set_default_device = lambda _: None
    fake_mx.device_info = lambda: {
        "name": "Apple M3",
        "max_recommended_working_set_size": 10**12,
    }
    fake_mx.metal = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setattr(sys.modules["mlx"], "core", fake_mx)
    monkeypatch.setattr(
        standard_adapter, "StandardDecoderAdapter", lambda *_a, **_k: adapter
    )
    monkeypatch.setattr(harness, "construct_admitted_adapter", lambda *_a: adapter)
    monkeypatch.setattr(harness, "bind_compute_precision", lambda *_: A)
    monkeypatch.setattr(harness, "sources", dict)
    monkeypatch.setattr(
        harness,
        "measure_context_costs",
        lambda *_a: (
            {"1": [1.0] * 16, "5": [1.0] * 16, "15": [1.0] * 16},
            {},
            {"seconds": 1.0},
            frozenset(),
            [127],
        ),
    )

    def fail(*_a, **_k):
        assert model.forward_with_taps is not original
        raise RuntimeError("later law audit failed")

    monkeypatch.setattr(harness, "check_continuation_generation", fail)
    selected = args("--dry-run")
    with pytest.raises(RuntimeError, match="later law"):
        harness.run(selected, harness.preflight(selected))
    assert closed == [True]
    assert model.forward_with_taps is original


def test_diagnostic_capture_memory_limit_fails_before_copy_and_never_omits_rows():
    import numpy as np

    observations = {1: [], 2: []}
    fake_mx = SimpleNamespace(float32=np.float32)
    logits = np.array([0.0, 1.0], dtype=np.float32)
    law = np.array([0.2, 0.8], dtype=np.float64)
    harness.capture_law_observation(
        observations, 1, logits, law, [1], True, fake_mx, np, 2
    )
    harness.capture_law_observation(
        observations, 2, logits, law, [2], False, fake_mx, np, 2
    )

    class NoCopy:
        def astype(self, *_):
            raise AssertionError("must refuse before tensor copy")

    with pytest.raises(ValueError, match="reserved memory"):
        harness.capture_law_observation(
            observations, 1, NoCopy(), law, [3], True, fake_mx, np, 2
        )
    assert len(observations[1]) == len(observations[2]) == 1
    assert observations[2][0]["reachable"] is False
