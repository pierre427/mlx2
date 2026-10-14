"""APCv2 entries must hold only the bytes APCv2 charges for them.

lilicorr-headonly-m3-20261014: ``mlx2.server`` on Qwen3-4B bf16 with
``--max-lanes 16`` on a 36 GiB M3 Pro died with Metal out-of-memory after
5.5 minutes of 16 concurrent chat requests; ``--max-lanes 8`` ran the same
workload for 16 minutes.  At the fault the server logged 27.73 GiB active
against 7.49 GiB of weights and 1.06 GiB of APCv2 resident: about 19 GiB
that no admission term priced.  Lane admission was not the fault -- it
subtracts measured active memory, weights included, and had already held
the batch at 8 lanes.

The end-of-request checkpoint is ``BatchKVCache.extract``: ``mx.contiguous``
of a row slice, which is lazy.  Stored unevaluated, each APCv2 entry held
the whole pre-filter batch K/V buffer (every row, at capacity) as its graph
input while APCv2 charged it for one row, so pinned memory grew with lane
width and request count until decode ran out.

These drive a real ServingEngine (tiny Qwen3 standard decoder, CPU, ordinary
route) at 8 lanes and compare what ``apc.clear()`` releases with what APCv2
accounted.  Before the fix the release was ~8x the accounting; after it the
two agree.
"""

import random
import threading

import mlx.core as mx
from route_harness import collect, make_adapter, patch_host

from mlx2.serving import ServingEngine

LANES = 8
REQUESTS = 32


def tiny_qwen3(seed=0, vocab=128):
    from mlx2.runtime.models.standard_decoder import Model, ModelArgs

    args = ModelArgs(
        model_type="qwen3", hidden_size=64, num_hidden_layers=4,
        intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=vocab, num_key_value_heads=2, max_position_embeddings=1 << 14,
        rope_theta=10000.0, tie_word_embeddings=True, head_dim=16,
    )
    mx.random.seed(seed)
    model = Model(args)
    model.eval()
    mx.eval(model.parameters())
    return model, vocab


def _drive(engine, lanes, requests):
    """``lanes`` closed-loop clients, ``requests`` greedy completions in all."""
    lock = threading.Lock()
    issued = [0]
    errors = []

    def client(seed):
        rng = random.Random(seed)
        while True:
            with lock:
                if issued[0] >= requests:
                    return
                issued[0] += 1
            prompt = [rng.randrange(2, 100) for _ in range(rng.randrange(24, 96))]
            out = collect(engine.submit({
                "tokens": prompt,
                "max_tokens": rng.randrange(24, 80),
                "temperature": 0,
            }))
            if "error" in out:
                errors.append(out["error"])

    threads = [threading.Thread(target=client, args=(k,)) for k in range(lanes)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_end_of_request_checkpoints_do_not_pin_the_batch(monkeypatch):
    patch_host(monkeypatch)
    model, vocab = tiny_qwen3()
    mx.synchronize()
    engine = ServingEngine(
        "tiny", adapter_factory=make_adapter(model, vocab),
        qualification_mode=True, mtp=False, max_lanes=LANES,
        max_inflight=32, prefill_step=16,
    )
    try:
        assert engine.ready.wait(120), engine.error
        assert engine.error is None
        assert _drive(engine, LANES, REQUESTS) == []
        apc = engine.apc
        accounted = int(apc.nbytes)
        assert accounted > 0, "the ordinary route stored no checkpoint"
        mx.synchronize()
        before = mx.get_active_memory()
        cleared = apc.clear()
        mx.synchronize()
        released = before - mx.get_active_memory()
        assert cleared["bytes"] == accounted
        # The entries own their rows: clearing frees what was charged, not
        # one batch buffer per entry (~LANES x before the fix).
        assert released <= 1.5 * accounted, (released, accounted)
    finally:
        engine.close()


def test_detach_checkpoint_graph_releases_the_batch_buffer():
    """The helper itself: a lazy row copy stops holding its batch once run."""
    from mlx2.runtime.models.cache import BatchKVCache
    from mlx2.serving import detach_checkpoint_graph

    mx.synchronize()
    base = mx.get_active_memory()
    batch = BatchKVCache([0] * 16)
    keys = mx.zeros((16, 2, 64, 16))
    batch.update_and_fetch(keys, keys)
    mx.eval(batch.keys, batch.values)
    full = mx.get_active_memory() - base
    row = batch.extract(3)
    assert detach_checkpoint_graph([row], None) == 2
    mx.synchronize()
    del batch, keys
    held = mx.get_active_memory() - base
    row_bytes = row.keys.nbytes + row.values.nbytes
    assert held <= 2 * row_bytes < full, (held, row_bytes, full)
