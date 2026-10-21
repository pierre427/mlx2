"""The generation thread is single-stream and lanes enter it evaluated.

apc-warm-hit-hang-20261021: an APCv2 warm hit's restore aliases (built on the
generation thread's own default stream, outside ``with mx.stream(...)``) were
consumed by the first prefill under ``generation_stream``, a different GPU
stream, so MLX bridged them with a Fence inside one evaluation; under
``MLX_METAL_FAST_SYNCH=1`` that fence spun forever on Laguna-S-2.1 and on
Qwen3.8-27B at B4.  Two mechanisms now close the class, and these tests pin
each one independently of the other:

* ``bind_generation_thread_stream`` makes the serving loop's default stream
  the generation stream, so nothing built on that thread can cross streams;
* a lane's cache is materialized before ``batch.insert`` even when lookup
  did not materialize it (a staged or requantized cache).

CPU only: the hang itself needs Metal.
"""

import threading

import mlx.core as mx
from test_apc_hits_hybrid_gdn_self_mtp import host  # noqa: F401
from test_apcv2_restore_materialized_cpu import _engine, _lazy, _run, tiny_laguna

from mlx2.runtime import apc_v2
from mlx2.runtime import generate as generate_module
from mlx2.runtime.generate import bind_generation_thread_stream, generation_stream
from mlx2.runtime.models import cache as cache_module
from mlx2.runtime.models.cache import restored_state_arrays

PREFILL_STEP = 16


def _resolved_generation_stream():
    with mx.stream(generation_stream):
        return mx.default_stream(generation_stream.device)


def test_bind_generation_thread_stream_makes_the_thread_single_stream():
    device = generation_stream.device
    main_before = mx.default_stream(device)
    seen = {}

    def body():
        seen["before"] = mx.default_stream(device)
        seen["bound"] = bind_generation_thread_stream()
        seen["after"] = mx.default_stream(device)
        seen["generation"] = _resolved_generation_stream()

    worker = threading.Thread(target=body)
    worker.start()
    worker.join(30)
    assert not worker.is_alive()
    assert seen["before"] != seen["generation"], "a fresh thread starts on its own default stream"
    assert seen["bound"] == seen["generation"]
    assert seen["after"] == seen["generation"]
    # The binding is per thread: the caller's default stream is untouched.
    assert mx.default_stream(device) == main_before


def _prompts(vocab):
    shared = [(7 * i + 3) % (vocab - 2) + 1 for i in range(2 * PREFILL_STEP)]
    first = shared + [(11 * i + 5) % (vocab - 2) + 1 for i in range(PREFILL_STEP + 1)]
    second = shared + [(5 * i + 1) % (vocab - 2) + 1 for i in range(PREFILL_STEP + 9)]
    return shared, first, second


def test_serving_loop_inserts_lanes_on_the_generation_stream(host, monkeypatch):  # noqa: F811
    """Every ``insert`` on the generation thread sees the generation stream as its default."""
    model, vocab = tiny_laguna()
    device = generation_stream.device
    observed = []
    original = generate_module.BatchGenerator.insert_segments

    def insert_segments(self, *args, **kwargs):
        observed.append((mx.default_stream(device), _resolved_generation_stream()))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(generate_module.BatchGenerator, "insert_segments", insert_segments)
    shared, first, second = _prompts(vocab)
    engine = _engine(model, vocab)
    try:
        _run(engine, first)
        _, cached = _run(engine, second)
    finally:
        engine.close()
    assert cached == len(shared)
    assert len(observed) == 2
    for default, generation in observed:
        assert default == generation, (default, generation)


def test_without_the_binding_inserts_run_on_another_stream(host, monkeypatch):  # noqa: F811
    """Negative control: the check detects an unbound generation thread."""
    import mlx2.runtime.generation_stream as stream_module

    monkeypatch.setattr(stream_module, "bind_generation_thread_stream", lambda: None)
    model, vocab = tiny_laguna()
    device = generation_stream.device
    observed = []
    original = generate_module.BatchGenerator.insert_segments

    def insert_segments(self, *args, **kwargs):
        observed.append((mx.default_stream(device), _resolved_generation_stream()))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(generate_module.BatchGenerator, "insert_segments", insert_segments)
    shared, first, second = _prompts(vocab)
    engine = _engine(model, vocab)
    try:
        _run(engine, first)
        _, cached = _run(engine, second)
    finally:
        engine.close()
    assert cached == len(shared)
    assert observed and all(default != generation for default, generation in observed)


def _lazy_at_insert(monkeypatch, vocab, model):
    """(arrays, lazy) per non-empty lane cache handed to ``insert_segments``."""
    observed = []
    original = generate_module.BatchGenerator.insert_segments

    def insert_segments(self, segments, max_tokens=None, caches=None, *args, **kwargs):
        for cache in caches or ():
            if cache is not None:
                arrays = restored_state_arrays(cache)
                observed.append((len(arrays), sum(_lazy(a) for a in arrays)))
        return original(self, segments, max_tokens, caches, *args, **kwargs)

    monkeypatch.setattr(generate_module.BatchGenerator, "insert_segments", insert_segments)
    shared, first, second = _prompts(vocab)
    engine = _engine(model, vocab)
    try:
        _run(engine, first)
        _, cached = _run(engine, second)
    finally:
        engine.close()
    assert cached == len(shared)
    warm = [row for row in observed if row[0] > 0]
    assert warm, observed
    return warm


def test_lane_cache_is_materialized_before_insert_even_when_lookup_leaves_it_lazy(
    host, monkeypatch  # noqa: F811
):
    """The insert-site guard stands on its own: lookup's materialization is disabled here."""
    monkeypatch.setattr(apc_v2, "materialize_restored_state", lambda *trees: 0)
    model, vocab = tiny_laguna()
    for arrays, lazy in _lazy_at_insert(monkeypatch, vocab, model):
        assert lazy == 0, f"{lazy} of {arrays} lane-cache arrays reached insert lazy"


def test_without_both_guards_lazy_restored_state_reaches_insert(host, monkeypatch):  # noqa: F811
    """Negative control: with lookup's and insert's materialization off, the restore is lazy."""
    monkeypatch.setattr(apc_v2, "materialize_restored_state", lambda *trees: 0)
    monkeypatch.setattr(cache_module, "materialize_restored_state", lambda *trees: 0)
    model, vocab = tiny_laguna()
    warm = _lazy_at_insert(monkeypatch, vocab, model)
    assert all(lazy > 0 for _arrays, lazy in warm), warm


def test_binding_module_does_not_import_the_model_modules():
    # The serving thread binds the stream before the adapter pins its
    # import-time environment; importing runtime.models.base first made every
    # import-order-checking adapter (Qwen3.8-27B) refuse to load.
    import subprocess
    import sys

    code = (
        "import sys, mlx2.runtime.generation_stream as g; "
        "assert 'mlx2.runtime.models.base' not in sys.modules, 'models imported'; "
        "import mlx2.runtime.generate as gen; "
        "assert gen.generation_stream is g.generation_stream; "
        "assert gen.bind_generation_thread_stream is g.bind_generation_thread_stream"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
