"""Host-only regression for one-chunk overflow admission."""

from __future__ import annotations

import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from mlx2.runtime.prefill_plan import (
    checkpoint_bounded_prefill_rows,
    next_prefill_checkpoint,
    prefill_fits_one_chunk,
    shared_prefill_budget_width,
    single_round_prefill_rows,
)

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mlx2"
    / "runtime"
    / "generate.py"
)


def _batch_generator_harness():
    tree = ast.parse(SOURCE.read_text())
    production = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenerator"
    )
    methods = [
        node
        for node in production.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_bounded_prefill_chunks",
            "_decode_first_mode",
            "_peek_interior_checkpoint",
            "_shared_prefill_width",
            "_admit_one_chunk_overflow",
        }
    ]
    harness = ast.ClassDef(
        name="BatchGenerator",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    namespace = {
        "checkpoint_bounded_prefill_rows": checkpoint_bounded_prefill_rows,
        "next_prefill_checkpoint": next_prefill_checkpoint,
        "prefill_fits_one_chunk": prefill_fits_one_chunk,
        "shared_prefill_budget_width": shared_prefill_budget_width,
        "single_round_prefill_rows": single_round_prefill_rows,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)  # noqa: S102
    return namespace["BatchGenerator"]


def _host(*, autoscale):
    batch_type = _batch_generator_harness()
    host = batch_type()
    host.prefill_batch_size = 2
    host.completion_batch_size = 8
    host.prefill_step_size = 8192
    host.prefill_step_autoscale = autoscale
    host.adaptive_prefill = False
    host.prefill_order = None
    host._generation_batch = []
    host._currently_processing = [
        [[list(range(81919)), [81919]], 0, 81920, True, 0, 0.0, None],
        [[list(range(73727)), [73727]], 0, 73728, True, 0, 0.0, None],
    ]
    prompt = (3, [list(range(999)), [999]], 8, [], [], None, [], None, 0.0, None)
    host._unprocessed_sequences = deque([prompt])
    host.state_budget = None

    class PromptBatch(list):
        uids = [1, 2]

    host._prompt_batch = PromptBatch([0, 0])
    host.scheduler_stats = {}
    host._fairness = lambda: SimpleNamespace(enabled=False)
    host._make_batch = lambda n, indices=None: [(n, indices)]
    host._decode_first_round_mode = "off"
    host.prefill_depth_budget = None
    host.PREFILL_DEPTH_FLOOR = 128
    host._interior_checkpoint_positions = {}
    host.decode_first_bumps = []
    host.decode_first = SimpleNamespace(
        prefill_token_budget=None,
        bump=host.decode_first_bumps.append,
    )
    host._sync_decode_first_stats = lambda: None
    return host


def test_overflow_admission_uses_execution_autoscale_cap():
    autoscaled = _host(autoscale=True)
    assert not autoscaled._admit_one_chunk_overflow(8192)
    assert autoscaled._prompt_batch == [0, 0]

    fixed = _host(autoscale=False)
    assert fixed._admit_one_chunk_overflow(8192)
    assert fixed._prompt_batch == [0, 0, (1, [0])]


def test_reserved_final_token_is_not_counted_as_a_prefill_row():
    boundary = _host(autoscale=True)
    boundary._unprocessed_sequences = deque(
        [(3, [list(range(512)), [512]], 8, [], [], None, [], None, 0.0, None)]
    )
    assert boundary._admit_one_chunk_overflow(8192)
    assert boundary._prompt_batch == [0, 0, (1, [0])]


def test_active_reserved_final_token_blocks_unneeded_overflow():
    boundary = _host(autoscale=True)
    boundary._currently_processing = [
        [[list(range(512)), [512]], 0, 513, True, 0, 0.0, None],
        [[list(range(73727)), [73727]], 0, 73728, True, 0, 0.0, None],
    ]
    assert not boundary._admit_one_chunk_overflow(8192)
    assert boundary._prompt_batch == [0, 0]


def test_shared_budget_rejects_candidate_that_would_need_another_slice():
    shared = _host(autoscale=False)
    shared._decode_first_round_mode = "all"
    shared.decode_first.prefill_token_budget = 512
    shared._unprocessed_sequences = deque(
        [(3, [list(range(512)), [512]], 8, [], [], None, [], None, 0.0, None)]
    )

    assert shared._shared_prefill_width(8192, 3, record=False) == 170
    assert not shared._admit_one_chunk_overflow(8192)
    assert shared._prompt_batch == [0, 0]
    assert shared.decode_first_bumps == []

    shared._unprocessed_sequences.clear()
    assert not shared._admit_one_chunk_overflow(8192)
    assert shared.decode_first_bumps == []


def test_depth_bound_and_checkpoint_reject_false_one_chunk_candidates():
    warm = _host(autoscale=False)
    warm.prefill_depth_budget = 1
    warm._unprocessed_sequences = deque(
        [
            (
                3,
                [list(range(512)), [512]],
                8,
                [],
                list(range(1024)),
                None,
                [],
                None,
                0.0,
                None,
            )
        ]
    )
    assert not warm._admit_one_chunk_overflow(8192)
    assert warm._prompt_batch == [0, 0]
    assert warm.scheduler_stats == {}

    checkpoint = _host(autoscale=False)
    checkpoint._interior_checkpoint_positions = {3: deque([128])}
    checkpoint._unprocessed_sequences = deque(
        [(3, [list(range(512)), [512]], 8, [], [], None, [], None, 0.0, None)]
    )
    assert not checkpoint._admit_one_chunk_overflow(8192)
    assert list(checkpoint._interior_checkpoint_positions[3]) == [128]
    assert checkpoint.scheduler_stats == {}


def test_active_depth_and_checkpoint_clamps_do_not_block_safe_overflow():
    for use_depth_bound in (True, False):
        active = _host(autoscale=False)
        active._currently_processing[0] = [
            [list(range(512)), [512]],
            0,
            513,
            True,
            1024 if use_depth_bound else 0,
            0.0,
            None,
        ]
        active._unprocessed_sequences = deque(
            [(3, [list(range(100)), [100]], 8, [], [], None, [], None, 0.0, None)]
        )
        if use_depth_bound:
            active.prefill_depth_budget = 1
        else:
            active._interior_checkpoint_positions = {1: deque([128])}

        assert active._admit_one_chunk_overflow(8192)
        assert active._prompt_batch == [0, 0, (1, [0])]
        if not use_depth_bound:
            assert list(active._interior_checkpoint_positions[1]) == [128]


def test_execution_checkpoint_clamp_follows_prefill_input_override():
    generator = next(
        node
        for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenerator"
    )
    round_method = next(
        node
        for node in generator.body
        if isinstance(node, ast.FunctionDef) and node.name == "_round"
    )
    source = ast.get_source_segment(SOURCE.read_text(), round_method)
    media_override = 'if len(seq) > 6 and seq[6] is not None:'
    row_limit = 'n = min(len(segments[0]), step_size)'
    checkpoint_clamp = 'n = checkpoint_bounded_prefill_rows(n, covered, next_checkpoint)'
    assert source.index(media_override) < source.index(row_limit)
    assert source.index(row_limit) < source.index(checkpoint_clamp)
