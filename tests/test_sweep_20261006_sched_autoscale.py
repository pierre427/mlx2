"""Prompt-length autoscale uses one length basis (sweep 2026-10-06, SS-4).

Selection and the self-MTP path sized a warm request from history + new
tokens; ordinary execution and overflow admission sized it from the new
tokens only, so a 40K-history + 3K-new request was planned as a 2048-row
chunk and executed in 512-row chunks.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_decode_first_publish import tiny_model

from mlx2.runtime import generate as G


@pytest.fixture(scope="module")
def model():
    return tiny_model()


def _warm(model):
    gen = G.BatchGenerator(
        model, completion_batch_size=4, prefill_batch_size=4,
        prefill_step_size=8192, prefill_step_autoscale=True,
    )
    history = [((i * 7) % 50) + 1 for i in range(40000)]
    new = [((i * 3) % 50) + 1 for i in range(3000)]
    (uid,) = gen.insert([new], max_tokens=[1], all_tokens=[history])
    return gen, uid


def test_execution_step_matches_selection_step_for_warm_prompt(model):
    gen, _uid = _warm(model)
    chunks = []
    real = gen._record_prefill_chunk
    gen._record_prefill_chunk = lambda uid, width: (
        chunks.append(width), real(uid, width)
    )
    try:
        queued = gen._unprocessed_sequences[0]
        selected_step = gen._prefill_chunk_length(
            queued[1], len(queued[4]) + sum(map(len, queued[1]))
        )
        assert selected_step == 2048
        gen.next()
        assert chunks[0] == selected_step, (selected_step, chunks)
    finally:
        gen.close()


def test_overflow_judges_a_warm_queued_row_by_its_full_prompt(model):
    """A 40K-history row with 1500 new tokens beside a full prefill slot.

    Its complete 41.5K prompt selects the 2048-row rung, so the new tokens
    fit one chunk and take the overflow slot; judged by the 1.5K new tokens
    alone (the 512-row rung) it was refused.
    """
    gen = G.BatchGenerator(
        model, completion_batch_size=4, prefill_batch_size=1,
        prefill_step_size=8192, prefill_step_autoscale=True,
    )
    try:
        gen.insert([[((i * 5) % 50) + 1 for i in range(20000)]], max_tokens=[1])
        gen.next()  # the long cold prompt now holds the only prefill slot
        history = [((i * 7) % 50) + 1 for i in range(40000)]
        new = [((i * 3) % 50) + 1 for i in range(1500)]
        gen.insert([new], max_tokens=[1], all_tokens=[history])
        gen.next()
        assert gen.scheduler_stats.get("short_prefill_overflow_admissions", 0) == 1
    finally:
        gen.close()
