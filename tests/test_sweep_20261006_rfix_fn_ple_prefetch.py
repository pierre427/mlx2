"""Known-tail PLE prefetch stages one slice ahead, not the whole tail.

GPU 2026-10-07 (Flash-Next MTP, 16K prompt beside one decoding lane): every
bounded prefill slice resubmitted the entire remaining prompt to the PLE
prefetch pool, 44 submissions and 3.0 M staged rows for one 16K prompt while
the forwards looked up 0.24 M (the staging cache holds 8192 rows, so most
were dropped before use).  That host work grows with the remaining prompt
on every slice and runs outside the measured forward, beside live decode.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_decode_first_publish import tiny_model
from test_mtp_cohort_formation import SELF_MTP, _admission

from mlx2.runtime import generate as G


def test_sliced_mtp_prefill_prefetches_one_slice_ahead():
    model = tiny_model()
    staged = []

    def ple_prefetch_verify(previous_tokens, tokens):
        staged.append((len(previous_tokens), len(tokens)))
        return 1

    model.ple_prefetch_verify = ple_prefetch_verify
    gen = G.BatchGenerator(
        model,
        completion_batch_size=8,
        prefill_step_size=128,
        self_mtp=dict(SELF_MTP, prefill_step_size=128, prefetch_known_tail_ple=True),
        mtp_admission=_admission(),
    )
    try:
        prompt = [(7 * i) % 50 + 1 for i in range(2048)]
        gen.insert([prompt], max_tokens=[4])
        for _ in range(40):
            gen.next()
            if not gen._unprocessed_sequences:
                break
        assert not gen._unprocessed_sequences
        assert len(staged) >= 16
        # Each slice stages its own rows and the next slice's: at most two
        # steps per submission, about twice the prompt in total (it was the
        # whole remaining tail each time, ~8.5x the prompt here).
        assert max(tokens for _, tokens in staged) <= 2 * 128
        assert sum(tokens for _, tokens in staged) <= 2 * len(prompt) + 1
        # The committed prefix each submission hashes against advances.
        assert [prev for prev, _ in staged] == sorted(prev for prev, _ in staged)
    finally:
        gen.close()
