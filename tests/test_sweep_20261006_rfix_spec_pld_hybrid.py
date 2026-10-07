"""Review fixes for prompt lookup on hybrid (GDN ``ArraysCache``) targets.

Codex review of SPEC-10: an aborted hybrid transaction rewound KV but not
recurrent state, and PLD swallowed the abort; a mixed cohort ran its
proposal-free lanes through the verify forward while their receipts said
``ordinary_target``.  And GPU evidence after SPEC-10: B4 PLD aggregated at
about the B1 rate because the cohort fragmented.
"""
import mlx.core as mx
import pytest
from test_segmented_mtp import _tiny_hybrid_mtp_model
from test_sweep_20261006_spec_pld_hybrid import PROMPTS, _greedy

from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.pld import PromptLookupBatchGenerator


class _Fault:
    """Raises from one layer's MLP while armed: the layers before it (and
    that layer's mixer) have already written their caches."""

    def __init__(self, mlp):
        self.mlp = mlp
        self.armed = False
        self.fired = 0

    def __call__(self, x):
        if self.armed:
            self.armed = False
            self.fired += 1
            raise RuntimeError("injected verify-forward fault")
        return self.mlp(x)


class _Forwards:
    """Logs decode forwards; arms the fault on the first wide mixed round."""

    def __init__(self, model, fault=None, arm_when=None):
        self.model = model
        self.fault = fault
        self.arm_when = arm_when
        self.log = []

    def __call__(self, tokens, cache=None, **kwargs):
        recurrent = [c for c in cache if isinstance(c, ArraysCache)]
        entry = (tokens.shape[0], tokens.shape[1], any(c.speculating for c in recurrent))
        self.log.append(entry)
        if self.arm_when is not None and self.arm_when(entry):
            self.fault.armed = True
            self.arm_when = None
        return self.model(tokens, cache=cache, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


def _state(cache):
    """Host copy of every plane: recurrent arrays and KV fill levels."""
    out = []
    for entry in cache:
        if isinstance(entry, ArraysCache):
            out.append(tuple(None if a is None else mx.array(a).tolist() for a in entry.cache))
        else:
            out.append(int(entry.offset))
    return out


def _drive(generator, uids, tokens, *, on_error=None, polls=800):
    for _ in range(polls):
        try:
            _prompts, responses = generator.next()
        except RuntimeError as error:
            if on_error is None:
                raise
            on_error(error)
            continue
        for response in responses:
            tokens[response.uid].append(response.token)
        if not generator.lanes:
            return


# ------------------------------------------------ abort rewinds recurrent rows
@pytest.mark.parametrize("plain", [False, True], ids=["verify", "plain"])
def test_hybrid_abort_rewinds_recurrent_rows_after_a_partial_forward(plain):
    model = _tiny_hybrid_mtp_model()
    rows = []
    for prompt in PROMPTS[:2]:
        cache = model.make_cache()
        model(mx.array([prompt]), cache=cache)
        mx.eval([c.state for c in cache])
        rows.append(cache)
    before = [_state(row) for row in rows]
    layer = model.language_model.model.layers[3]
    fault = _Fault(layer.mlp)
    layer.mlp = fault
    lengths = [1, 1] if plain else [3, 1]
    transaction = HybridVerifyRows(rows).begin(lengths, plain_single_token=plain)
    fault.armed = True
    with pytest.raises(RuntimeError, match="injected"):
        width = max(lengths)
        model(mx.array([[7] * width, [9] * width]), cache=transaction.caches)
    assert fault.fired == 1
    # Layers 0-2 and layer 3's attention wrote; the abort must undo all of it.
    transaction.abort()
    assert [_state(row) for row in rows] == before
    assert not any(c.speculating for row in rows for c in row if isinstance(c, ArraysCache))


def test_hybrid_abort_raises_when_a_row_cannot_be_rewound(monkeypatch):
    model = _tiny_hybrid_mtp_model()
    rows = [model.make_cache() for _ in range(2)]
    for prompt, cache in zip(PROMPTS[:2], rows):
        model(mx.array([prompt]), cache=cache)
    transaction = HybridVerifyRows(rows).begin([2, 2])
    model(mx.array([[7, 8], [9, 10]]), cache=transaction.caches)
    kv = next(c for c in rows[0] if type(c) is KVCache)
    monkeypatch.setattr(kv, "trim", lambda n: 0)
    with pytest.raises(RuntimeError, match="could not rewind KV"):
        transaction.abort()
    assert transaction.closed


# ------------------------------------------ PLD recovers from a failed round
def _fault_run(arm_when, *, break_abort=False, monkeypatch=None):
    model = _tiny_hybrid_mtp_model()
    layer = model.language_model.model.layers[3]
    fault = _Fault(layer.mlp)
    layer.mlp = fault
    forwards = _Forwards(model, fault, arm_when)
    generator = PromptLookupBatchGenerator(
        forwards, completion_batch_size=4, prefill_step_size=64,
        prompt_lookup={"num_draft": 3, "ngram_min": 2, "ngram_max": 3},
    )
    if break_abort:
        from mlx2.runtime import hybrid_verify_rows

        real_abort = hybrid_verify_rows.HybridVerifyTransaction.abort

        def failing_abort(self):
            # A rewind that throws part-way: the KV rows are rewound, the
            # recurrent rows are not.  PLD must not trust these lanes.
            for row, bases in zip(self.owner.rows, self._kv_base):
                for cache, base in zip(row, bases):
                    if base is not None and int(cache.offset) > base:
                        cache.trim(int(cache.offset) - base)
            self._recurrent_base = []
            real_abort(self)
            raise RuntimeError("abort failed")

        monkeypatch.setattr(hybrid_verify_rows.HybridVerifyTransaction, "abort", failing_abort)
    uids = generator.insert(PROMPTS, max_tokens=[20] * 4,
                            caches=[model.make_cache() for _ in PROMPTS])
    tokens = {uid: [] for uid in uids}
    errors = []
    try:
        _drive(generator, uids, tokens, on_error=errors.append)
    finally:
        generator.close()
    return model, [tokens[uid] for uid in uids], errors, generator.scheduler_stats, fault


def _mixed_verify(entry):
    rows, width, speculating = entry
    return rows == 4 and 1 < width <= 4 and speculating


def test_pld_failed_hybrid_round_leaves_lanes_exact():
    model, outputs, errors, _stats, fault = _fault_run(_mixed_verify)
    assert fault.fired == 1 and len(errors) == 1
    for prompt, output in zip(PROMPTS, outputs):
        assert output == _greedy(model, prompt, 20)


def test_pld_rebuilds_every_lane_when_the_abort_itself_fails(monkeypatch):
    model, outputs, errors, stats, fault = _fault_run(
        _mixed_verify, break_abort=True, monkeypatch=monkeypatch
    )
    assert fault.fired == 1 and len(errors) == 1
    assert "injected" in str(errors[0])
    rebuilt = stats["pld_recovery_checkpoint_restores"] + stats["pld_recovery_full_rebuilds"]
    assert rebuilt == 4
    for prompt, output in zip(PROMPTS, outputs):
        assert output == _greedy(model, prompt, 20)


# --------------------------------------- truthful labels in mixed cohorts
def test_proposal_free_lane_in_a_verify_forward_is_not_labelled_ordinary():
    """A mixed cohort runs every lane through the verify forward (one forward
    serves the round); only a round that is one plain decode step for the
    whole cohort may say ``ordinary_target``."""
    model = _tiny_hybrid_mtp_model()
    generator = PromptLookupBatchGenerator(
        model, completion_batch_size=4, prefill_step_size=64,
        prompt_lookup={"num_draft": 3, "ngram_min": 2, "ngram_max": 3, "adaptive": False},
    )
    paths = {}
    begin = generator._begin_verify

    def recording_begin(lanes, lengths):
        # In begin order per lane: a pipelined plain round begins before
        # the host has booked the round it follows, so ``cycles`` lags.
        transaction = begin(lanes, lengths)
        plain = max(lengths) == 1 and getattr(transaction, "plain", True)
        for lane in lanes:
            rounds = [key for key in paths if key[0] == lane.uid]
            paths[(lane.uid, len(rounds) + 1)] = plain
        return transaction

    generator._begin_verify = recording_begin
    uids = generator.insert(PROMPTS, max_tokens=[24] * 4,
                            caches=[model.make_cache() for _ in PROMPTS])
    receipts = []
    try:
        for _ in range(500):
            _prompts, responses = generator.next()
            receipts.extend((r.uid, r.speculative_receipt) for r in responses)
            if not generator.lanes:
                break
    finally:
        generator.close()
    assert len(receipts) == 24 * len(uids)
    mixed = 0
    for uid, receipt in receipts:
        plain = paths[(uid, receipt["cycles"])]
        assert receipt["current_target_path"] == ("plain_decode" if plain else "verify_rows")
        if receipt["current_execution"] == "ordinary_target":
            assert plain and receipt["round_proposed"] == 0
        if receipt["round_proposed"] == 0 and not plain:
            mixed += 1
            assert receipt["current_execution"] == "prompt_lookup_verify"
            assert receipt["execution"] == "prompt_lookup_verify"
    assert mixed > 0  # the cohort really mixed proposing and plain lanes


# ----------------------------------- B4 cohort: one forward, O(lanes) host
_CYCLES = ([5, 6, 7, 8, 9, 10], [20, 21, 22, 23], [40, 41, 42, 43, 44, 45, 46], [60, 61, 62, 63])


class _Copying:
    """The tiny hybrid target, steered to continue each lane's cycle.

    A strong per-token bias makes greedy decode follow a successor map, so
    prompt lookup accepts real drafts (the third and fourth cycles differ
    from their prompts, so those lanes reject part of their drafts); the
    caches still run the real hybrid layers.  Counts decode forwards.
    """

    def __init__(self):
        import numpy as np

        self.model = _tiny_hybrid_mtp_model()
        bias = np.zeros((128, 128), np.float32)
        for cycle in _CYCLES:
            for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                bias[a, b] = 100.0
        self.bias = mx.array(bias)
        self.forwards = []

    def __call__(self, tokens, cache=None, **kwargs):
        self.forwards.append(tuple(tokens.shape))
        return self.model(tokens, cache=cache, **kwargs) + self.bias[tokens]

    def __getattr__(self, name):
        return getattr(self.model, name)


def _steady_state(repeat, polls=6):
    import sys

    model = _Copying()
    prompts = [prompt * repeat for prompt in PROMPTS]
    generator = PromptLookupBatchGenerator(
        model, completion_batch_size=4, prefill_step_size=4096,
        prompt_lookup={"num_draft": 6, "ngram_min": 2, "ngram_max": 3, "adaptive": False},
    )
    generator.insert(prompts, max_tokens=[400] * 4,
                     caches=[model.make_cache() for _ in prompts])
    while any(lane.anchor is None for lane in generator.lanes.values()):
        generator.next()
    generator.next()  # every lane has decoded once
    evals, real_eval = [0], mx.eval

    def counting_eval(*args, **kwargs):
        evals[0] += 1
        return real_eval(*args, **kwargs)

    calls = [0]

    def profile(_frame, event, _arg):
        if event in ("call", "c_call"):
            calls[0] += 1

    stats0 = dict(generator.scheduler_stats)
    forwards0 = len(model.forwards)
    delivered = []
    mx.eval = counting_eval
    sys.setprofile(profile)
    try:
        for _ in range(polls):
            _prompts, responses = generator.next()
            delivered.append(len(responses))
    finally:
        sys.setprofile(None)
        mx.eval = real_eval
        generator.close()
    stats = {k: v - stats0.get(k, 0) for k, v in generator.scheduler_stats.items()
             if isinstance(v, int)}
    return model.forwards[forwards0:], delivered, evals[0], calls[0], stats


def test_b4_cohort_runs_one_forward_per_round_with_bounded_host_work():
    forwards, delivered, evals, calls, stats = _steady_state(repeat=2)
    polls = len(delivered)
    # One target forward per poll, carrying all four lanes.
    assert len(forwards) == polls and all(rows == 4 for rows, _width in forwards)
    assert stats["pld_batched_rounds"] == polls
    assert stats["pld_batched_lanes"] == 4 * polls
    # Every verified token leaves in its own poll: drafts were accepted, and
    # no lane sat a round out draining (the cohort stays whole).
    assert stats["pld_accepted"] > 0
    assert sum(delivered) == stats["pld_accepted"] + stats["pld_bonus"]
    assert max(delivered) > 4
    # One host sync per round (forward, rows and greedy tokens together),
    # and no per-round recovery capture.
    assert evals == polls
    assert stats["pld_recovery_checkpoint_captures"] == 0
    # Host work does not grow with the context: the same rounds over a
    # prompt eight times longer make about the same number of Python calls.
    _f, long_delivered, long_evals, long_calls, _s = _steady_state(repeat=16)
    assert long_evals == len(long_delivered)
    assert long_calls < 1.3 * calls, (calls, long_calls)
