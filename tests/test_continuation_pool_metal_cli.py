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
    assert harness.preflight(args("--dry-run", "--target-verify-row-exact"))[
        "target_verify_row_exact"
    ] is True


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--context-tokens", "129"),
        ("--context-tokens", "15"),
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
