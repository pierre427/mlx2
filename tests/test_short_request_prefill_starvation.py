"""omlx#3726: a short request arriving during a long chunked prefill must
get prefill service before the long prefill finishes.

Drives the real BatchGenerator on CPU with a tiny hybrid GDN + MTP model,
configured like serving.py builds it (prefill_batch_size=min(2, lanes),
prefill_batch_window=1, adaptive_prefill=True).  Measures, in scheduler
rounds (calls to next()), when the short request emits its first token versus
when the long request does.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.sample_utils import LaneRNG

STEP = 32
LONG = [(7 * i + 3) % 120 + 2 for i in range(40 * STEP)]  # 40 chunks
SHORT = [(5 * i + 1) % 120 + 2 for i in range(12)]  # < one chunk


def tiny_model():
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=4,
        mtp_num_hidden_layers=1, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=1 << 14,
    )
    mx.random.seed(7)
    m = TextModel(args)
    m.eval()
    mx.eval(m.parameters())
    return m


def make(model, mtp, lanes=4):
    kw = {}
    if mtp:
        kw["self_mtp"] = {"num_draft": 2, "persistent": True, "rate_gate": False,
                          "prefill_step_size": STEP}
    return BatchGenerator(model, completion_batch_size=lanes,
                          prefill_batch_size=min(2, lanes), prefill_step_size=STEP,
                          prefill_batch_window=1, adaptive_prefill=True, **kw)


def insert(gen, prompt, mtp, seed):
    extra = ({"lane_rngs": [LaneRNG(seed)],
              "self_mtp_configs": [{"sampling_temp": 0.0}]} if mtp else {})
    return gen.insert([prompt], max_tokens=[4], **extra)[0]


def first_token_rounds(mtp, longs=1, lanes=4):
    model = tiny_model()
    gen = make(model, mtp, lanes)
    first = {}
    try:
        long_uids = [insert(gen, LONG, mtp, 10 + i) for i in range(longs)]
        for _ in range(3):  # long prefill(s) now in progress
            gen.next()
        short_uid = insert(gen, SHORT, mtp, 99)
        for rnd in range(1, 400):
            _p, rs = gen.next()
            for r in rs:
                first.setdefault(r.uid, rnd)
            if short_uid in first and all(u in first for u in long_uids):
                break
        stats = dict(gen.scheduler_stats)
    finally:
        gen.close()
    return first[short_uid], min(first[u] for u in long_uids), stats


@pytest.mark.parametrize("mtp", [False, True], ids=["ordinary", "self_mtp"])
def test_short_request_is_not_starved_by_one_long_prefill(mtp):
    short, long_, _ = first_token_rounds(mtp)
    print(f"\nmtp={mtp}: short first token @round {short}, long @round {long_}")
    assert short < long_, "short request waited for the whole long prefill"
    assert short <= 4


def test_ordinary_short_request_with_two_long_prefills_in_flight():
    """prefill_batch_size=2 slots both held by long prefills."""
    short, long_, _ = first_token_rounds(False, longs=2)
    print(f"\nordinary 2 longs: short @round {short}, first long @round {long_}")
    assert short < long_, "short request waited for a long prefill slot"


def _overflow_host(queue, *, budget_ok=True):
    from collections import deque
    from types import SimpleNamespace

    host = SimpleNamespace(
        prefill_batch_size=2,
        completion_batch_size=8,
        prefill_step_size=STEP,
        adaptive_prefill=False,
        _generation_batch=[],
        _currently_processing=[[[list(range(10 * STEP))]], [[list(range(9 * STEP))]]],
        _unprocessed_sequences=deque(queue),
        state_budget=None if budget_ok is None else object(),
        made=[],
    )

    class Prompt(list):
        def extend(self, rows):
            host.made.append(rows)

    host._prompt_batch = Prompt([0, 0])
    host._fairness = lambda: SimpleNamespace(enabled=False)
    host._sync_budget_mutation = lambda: None
    host._candidate_admission_state = lambda seq: seq[0]
    host._admit_states = lambda states: 1 if budget_ok else 0
    host._make_batch = lambda n, indices=None: ("rows", n, indices)
    return host


def test_overflow_admits_the_exact_short_row_not_a_media_or_head_row():
    long_seq = (1, [list(range(5 * STEP))], 8, [], [], None, [], None, 0.0, None)
    media = (2, [list(range(4))], 8, [], [], None, [], None, 0.0, {"pixels": 1})
    short = (3, [list(range(4))], 8, [], [], None, [], None, 0.0, None)
    host = _overflow_host([long_seq, media, short])
    assert BatchGenerator._admit_one_chunk_overflow(host, STEP)
    assert host.made == [("rows", 1, [2])]


def test_overflow_budgets_the_selected_row():
    long_seq = (1, [list(range(5 * STEP))], 8, [], [], None, [], None, 0.0, None)
    short = (3, [list(range(4))], 8, [], [], None, [], None, 0.0, None)
    host = _overflow_host([long_seq, short], budget_ok=False)
    assert not BatchGenerator._admit_one_chunk_overflow(host, STEP)
    assert host.made == []
