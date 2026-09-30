"""MLX2_GREEDY_BATCH_SAMPLER=1 samples greedy lanes in one batched argmax.

Output must equal the per-lane closures, and the shared sampler must be
called on a multi-row batch (the point of the switch).  The sampler lives at
module level in ``serving``, which imports MLX lazily: this test is what
caught it referencing an undefined ``mx``.
"""
import threading

import route_harness as rh
from mlx2 import serving


def _two_greedy_lanes(monkeypatch, enabled):
    monkeypatch.setenv("MLX2_GREEDY_BATCH_SAMPLER", "1" if enabled else "0")
    rh.patch_host(monkeypatch)
    rows_seen = []
    original = serving._greedy_batch_sampler  # unpatched; monkeypatch persists across calls

    def counting(logprobs):
        rows_seen.append(int(logprobs.shape[0]))
        return original(logprobs)

    counting.batch_groupable = True
    monkeypatch.setattr(serving, "_GREEDY_BATCH_SAMPLER", counting)
    model, vocab = rh.tiny_qwen38_mtp()
    engine = rh.make_engine(model, vocab, mtp=False, max_lanes=2)
    results = [None, None]
    try:
        def worker(i):
            results[i] = rh.run(
                engine, {"tokens": [3, 5, 7, 11 + i], "max_tokens": 12, "temperature": 0}
            )

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        engine.close()
    for result in results:
        assert "error" not in result, result
    return [result["tokens"] for result in results], rows_seen


def test_greedy_batch_sampler_matches_per_lane_and_batches_rows(monkeypatch):
    per_lane, rows_off = _two_greedy_lanes(monkeypatch, enabled=False)
    batched, rows_on = _two_greedy_lanes(monkeypatch, enabled=True)
    assert rows_off == [], "default keeps the per-lane closures"
    assert rows_on, "the shared sampler must be used when enabled"
    assert max(rows_on) == 2, rows_on
    assert batched == per_lane
    assert all(len(tokens) == 12 for tokens in per_lane), per_lane
