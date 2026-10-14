import numpy as np
import pytest
import mlx.core as mx

from mlx2.runtime.offline_capture import capture_windows
from mlx2.adapters.qwen35_capture import Qwen35CaptureDriver
from test_disk_weight_streaming import (
    _dense_model,
    _dense_checkpoint,
    _shards,
    _quantizer,
)
from mlx2.runtime.streamed_load import (
    WeightStreamRequest,
    load_streamed,
    trunk_mlp_targets,
)


def _capture(model, windows, manager=None, order="layer_major", budget=4 << 20):
    outputs, states = [], []

    def sink(index, tokens, hidden, cache):
        logits = model.logits(hidden)
        mx.eval(logits)
        outputs.append(np.asarray(logits).copy())
        states.append(
            [
                [
                    (np.asarray(a).copy() if isinstance(a, mx.array) else a)
                    for a in c.state
                ]
                for c in cache
            ]
        )

    receipt = capture_windows(
        Qwen35CaptureDriver(model, manager),
        windows,
        revision="tiny-frozen",
        sink=sink,
        order=order,
        activation_budget_bytes=budget,
    )
    return outputs, states, receipt


def test_independent_windows_match_ordinary_logits_and_state():
    mx.random.seed(72)
    model = _dense_model()
    windows = [[3, 4, 8], [7, 2, 1, 10, 6], [3]]
    outputs, states, receipt = _capture(model, windows)
    for index, tokens in enumerate(windows):
        cache = model.make_cache()
        logits = model(mx.array([tokens], mx.uint32), cache=cache)
        mx.eval(logits, [c.state for c in cache])
        np.testing.assert_array_equal(outputs[index], np.asarray(logits))
        for actual, expected in zip(states[index], cache):
            for a, b in zip(actual, expected.state):
                if isinstance(b, mx.array):
                    np.testing.assert_array_equal(a, np.asarray(b))
    assert receipt["state_published"] is False


def test_streamed_capture_reuses_readers_and_keeps_exact_windows(tmp_path):
    root, weight_map, records = _dense_checkpoint(tmp_path)
    model = _dense_model()
    loaded = load_streamed(
        model,
        root,
        _shards(weight_map),
        request=WeightStreamRequest(
            mode="dense_mlp", budget_bytes=1 << 20, read_workers=2
        ),
        sanitize=model.sanitize,
        quantize=_quantizer(model),
        records=records,
        weight_map=weight_map,
        dense_targets=lambda m: trunk_mlp_targets(
            m, layers_prefix="language_model.model.layers."
        ),
    )
    loaded.manager.begin_serving()
    windows = [[3, 4, 8], [7, 2, 1, 10, 6], [3]]
    before = loaded.manager.counters()["dense_stream_page_ins_total"]
    try:
        layer, _, _ = _capture(model, windows, loaded.manager)
        after = loaded.manager.counters()
        reads = after["dense_stream_page_ins_total"] - before
        assert reads == 12  # 4 layers * 3 MLP projections, once each
        assert after["dense_stream_capture_reuses_total"] == 24
        assert loaded.manager.live_bytes() == 0
        plain, _, _ = _capture(model, windows, loaded.manager, "window_major")
        assert (
            loaded.manager.counters()["dense_stream_page_ins_total"]
            - after["dense_stream_page_ins_total"]
            == 36
        )
        for a, b in zip(layer, plain):
            np.testing.assert_array_equal(a, b)
    finally:
        loaded.manager.close()


def test_budget_refuses_before_sink():
    model = _dense_model()
    with pytest.raises(MemoryError, match="activation/cache"):
        _capture(model, [[3, 4, 8]], budget=1)
    for token in (2**32, -1, True, 64):
        with pytest.raises(ValueError):
            _capture(model, [[token]])


def test_weight_budget_and_scope_cleanup_on_exception(tmp_path):
    root, weight_map, records = _dense_checkpoint(tmp_path)
    model = _dense_model()
    loaded = load_streamed(
        model,
        root,
        _shards(weight_map),
        request=WeightStreamRequest(mode="dense_mlp", budget_bytes=1 << 20),
        sanitize=model.sanitize,
        quantize=_quantizer(model),
        records=records,
        weight_map=weight_map,
        dense_targets=lambda m: trunk_mlp_targets(
            m, layers_prefix="language_model.model.layers."
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="sink failure"):

            def bad(index, tokens, hidden, cache):
                raise RuntimeError("sink failure")

            capture_windows(
                Qwen35CaptureDriver(model, loaded.manager),
                [[3]],
                revision="tiny",
                sink=bad,
            )
        assert loaded.manager.live_bytes() == 0
        from mlx2.runtime.weight_stream import WorkingSetTooSmall

        loaded.manager.plan.staging_ceiling_bytes = 1
        with pytest.raises(WorkingSetTooSmall):
            _capture(model, [[3]], loaded.manager)
    finally:
        loaded.manager.close()


def test_moe_windows_match_independent_ordinary(monkeypatch):
    from test_disk_weight_streaming import _moe_model

    model = _moe_model(monkeypatch, fused=False)
    windows = [[3, 4, 8], [7, 2], [3]]
    got, _, _ = _capture(model, windows)
    for out, tokens in zip(got, windows):
        expected = model(mx.array([tokens], mx.uint32), cache=model.make_cache())
        mx.eval(expected)
        np.testing.assert_array_equal(out, np.asarray(expected))


def test_reused_reader_still_checks_revision_and_releases_scope(tmp_path):
    import os
    from mlx2.runtime.weight_stream import WeightSourceChanged, StreamingUnavailable

    root, weight_map, records = _dense_checkpoint(tmp_path)
    model = _dense_model()
    loaded = load_streamed(
        model,
        root,
        _shards(weight_map),
        request=WeightStreamRequest(mode="dense_mlp", budget_bytes=1 << 20),
        sanitize=model.sanitize,
        quantize=_quantizer(model),
        records=records,
        weight_map=weight_map,
        dense_targets=lambda m: trunk_mlp_targets(
            m, layers_prefix="language_model.model.layers."
        ),
    )
    manager = loaded.manager
    try:
        reader = next(iter(manager._readers.values()))
        with manager.capture_layer("language_model.model.layers.0"):
            with pytest.raises(StreamingUnavailable, match="nested"):
                with manager.capture_layer("language_model.model.layers.0"):
                    pass
            reader.load()
            path = reader.source_paths[0]
            st = path.stat()
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
            with pytest.raises(WeightSourceChanged):
                reader.load()
        assert manager.live_bytes() == 0
    finally:
        manager.close()
