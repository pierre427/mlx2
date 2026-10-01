"""Disk weight streaming: early-load MoE experts and dense MLP paging (CPU only).

Every test runs on the CPU device (``tests/conftest.py``).  The evidence here
is synthetic: tiny real Qwen3.6 / Qwen3.8 model classes, real quantized
safetensors checkpoints, the adapters' own loaders, and APCv2 itself.  It does
not -- and cannot -- establish Metal memory peaks or real oversized-model
behaviour; those need the dedicated hardware run.

"Never evaluated" is proven by instrumentation: ``mx.eval``/``mx.async_eval``
are wrapped and every raw array ``mx.load`` returned for a streamed tensor is
checked against what was explicitly evaluated, with the ordinary loader as the
negative control that must trip the same instrument.
"""

import json
import os
import resource
import sys
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from mlx2.runtime import weight_stream  # noqa: E402
from mlx2.runtime.streamed_load import (  # noqa: E402
    WeightStreamRequest,
    declared_stream_modes,
    force_stock_expert_arithmetic,
    load_streamed,
    require_declared,
    trunk_mlp_targets,
)
from mlx2.runtime.weight_stream import (  # noqa: E402
    BoundShards,
    StreamingUnavailable,
    TensorOrigins,
    WeightSourceChanged,
    WorkingSetTooSmall,
    apc_weight_stream_fingerprint,
    concat_ledger,
    install_expert_streaming,
    record_concat,
)

GROUP = 64
BITS = 4
QUANT = {"group_size": GROUP, "bits": BITS, "mode": "affine"}

TINY_MOE = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 64,
    "intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 64,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 3,
    "full_attention_interval": 4,
    "mtp_num_hidden_layers": 0,
    "partial_rotary_factor": 0.5,
    "rope_parameters": None,
    "max_position_embeddings": 256,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
}

TINY_DENSE = {
    "model_type": "qwen3_5",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 64,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 4,
    "mtp_num_hidden_layers": 0,
    "partial_rotary_factor": 0.5,
    "rope_parameters": None,
    "max_position_embeddings": 256,
}

PROMPT = [3, 7, 1, 12, 9, 30, 4, 18]


# ---------------------------------------------------------------------------
# synthetic checkpoints and models
# ---------------------------------------------------------------------------


def _moe_model(monkeypatch, *, fused, interval=4, layers=None, shared_fold=False):
    from mlx2.runtime.models import qwen3_next
    from mlx2.runtime.models.qwen36_35b import Model, ModelArgs

    monkeypatch.setattr(qwen3_next, "_MOE_FUSED_GATE_UP", fused)
    monkeypatch.setattr(qwen3_next, "_MOE_SHARED_IN_GATHER", shared_fold)
    text = dict(TINY_MOE, full_attention_interval=interval)
    if layers is not None:
        text["num_hidden_layers"] = layers
    return Model(ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": text}))


def _dense_model(*, interval=4):
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs

    text = dict(TINY_DENSE, full_attention_interval=interval)
    return Model(ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": text}))


def _official(key):
    """The released checkpoint layout: sanitize renames it (identity kept)."""
    if key.startswith("language_model.model."):
        return key.replace("language_model.model.", "model.language_model.", 1)
    return key


def _write_checkpoint(root, weights, streamed_marker):
    """Two shards: streamed tensors alone in the second, the rest in the first."""
    root.mkdir(parents=True, exist_ok=True)
    names = sorted(weights)
    streamed = [name for name in names if streamed_marker(name)]
    other = [name for name in names if name not in streamed]
    files = {
        "model-00001-of-00002.safetensors": other,
        "model-00002-of-00002.safetensors": streamed,
    }
    weight_map = {}
    for (name, keys) in files.items():
        mx.save_safetensors(str(root / name), {key: weights[key] for key in keys})
        weight_map.update({key: name for key in keys})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    records = [
        (name, (root / name).stat().st_size, (root / name).stat().st_mtime_ns)
        for name in sorted(files)
    ]
    return weight_map, records


def _is_expert(name):
    return ".switch_mlp." in name


def _is_trunk_mlp(name):
    return ".mlp." in name and ".layers." in name and "mtp" not in name.split(".")


def _quantize_all(model):
    nn.quantize(
        model, group_size=GROUP, bits=BITS,
        class_predicate=lambda _p, m: hasattr(m, "to_quantized"),
    )
    mx.eval(model.parameters())


def _moe_checkpoint(tmp_path, monkeypatch, *, interval=4, layers=None, seed=3):
    mx.random.seed(seed)
    # The checkpoint ships split gate/up tables (fused=False source); a fused
    # live tree must therefore be built through the concat ledger.
    source = _moe_model(monkeypatch, fused=False, interval=interval, layers=layers)
    _quantize_all(source)
    weights = {_official(k): v for (k, v) in tree_flatten(source.parameters())}
    root = tmp_path / f"moe-{interval}-{layers}"
    (weight_map, records) = _write_checkpoint(root, weights, _is_expert)
    return root, weight_map, records


def _dense_checkpoint(tmp_path, *, interval=4, seed=5):
    mx.random.seed(seed)
    source = _dense_model(interval=interval)
    _quantize_all(source)
    weights = {_official(k): v for (k, v) in tree_flatten(source.parameters())}
    root = tmp_path / f"dense-{interval}"
    (weight_map, records) = _write_checkpoint(root, weights, _is_trunk_mlp)
    return root, weight_map, records


def _quantizer(model):
    def quantize(weights):
        def predicate(name, module):
            if name in QUANT:
                return QUANT[name]
            return hasattr(module, "to_quantized") and f"{name}.scales" in weights

        nn.quantize(model, group_size=GROUP, bits=BITS, mode="affine",
                    class_predicate=predicate)

    return quantize


def _shards(weight_map):
    return sorted(set(weight_map.values()))


def _resident(model, root, weight_map, *, prune=None):
    """The ordinary eager reference: exactly the adapters' disabled path."""
    from mlx2.runtime.ubc_evict import load_shards_evicting

    files = [root / name for name in _shards(weight_map)]
    weights = model.sanitize(load_shards_evicting(files, sanitize=prune))
    _quantizer(model)(weights)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    force_stock_expert_arithmetic(model)
    return model


def _moe_request(budget, **overrides):
    return WeightStreamRequest(
        mode="moe_experts", budget_bytes=int(budget), read_workers=2, **overrides
    )


def _streamed_moe(model, root, weight_map, records, *, budget, **overrides):
    loaded = load_streamed(
        model,
        root,
        _shards(weight_map),
        request=_moe_request(budget, **overrides),
        sanitize=model.sanitize,
        quantize=_quantizer(model),
        records=records,
        weight_map=weight_map,
        shard_prune=model.shard_prune,
        top_k=TINY_MOE["num_experts_per_tok"],
        on_installed=lambda manager: setattr(
            manager, "forced_stock", force_stock_expert_arithmetic(model)
        ),
    )
    loaded.manager.begin_serving()
    return loaded


def _streamed_dense(model, root, weight_map, records, *, staging=1 << 30, **overrides):
    loaded = load_streamed(
        model,
        root,
        _shards(weight_map),
        request=WeightStreamRequest(
            mode="dense_mlp", budget_bytes=int(staging), read_workers=2, **overrides
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
    return loaded


# Small enough that two experts per projection is the steady capacity: wide
# prefill calls overflow and evict.
SMALL_BUDGET = 2 * 8 * 5120


def _logits(model, tokens, cache=None):
    out = model(mx.array([tokens]), cache=cache)
    mx.eval(out)
    return np.array(out)


def _descriptors_open_on(paths):
    wanted = {(os.stat(p).st_dev, os.stat(p).st_ino) for p in paths}
    limit = min(resource.getrlimit(resource.RLIMIT_NOFILE)[0], 65536)
    found = 0
    for fd in range(limit):
        try:
            info = os.fstat(fd)
        except OSError:
            continue
        found += (info.st_dev, info.st_ino) in wanted
    return found


class _EvalSpy:
    """Records every raw array ``mx.load`` returned for a watched tensor name,
    and, at the moment of each explicit ``mx.eval``/``mx.async_eval``, which
    of them were passed.  Watched arrays are held alive, so an identity match
    at call time cannot be a reused ``id``; evaluated arrays are retained too,
    so later identity checks against them are sound."""

    def __init__(self, monkeypatch, watch):
        self.watch = watch
        self.watched = []
        self._watched_ids = set()
        self.hits = []
        self.evaluated = []
        real_load, real_eval, real_async = mx.load, mx.eval, mx.async_eval

        def load(path, *args, **kwargs):
            arrays = real_load(path, *args, **kwargs)
            if isinstance(arrays, dict):
                for (name, value) in arrays.items():
                    if watch(name):
                        self.watched.append(value)
                        self._watched_ids.add(id(value))
            return arrays

        def record(args):
            for (_, leaf) in tree_flatten(list(args)):
                if isinstance(leaf, mx.array):
                    self.evaluated.append(leaf)
                    if id(leaf) in self._watched_ids:
                        self.hits.append(leaf)

        def evaluate(*args):
            record(args)
            return real_eval(*args)

        def evaluate_async(*args):
            record(args)
            return real_async(*args)

        monkeypatch.setattr(mx, "load", load)
        monkeypatch.setattr(mx, "eval", evaluate)
        monkeypatch.setattr(mx, "async_eval", evaluate_async)

    def watched_evaluated(self):
        return list(self.hits)

    def was_evaluated(self, array):
        return any(leaf is array for leaf in self.evaluated)


@pytest.fixture
def bound_shards(monkeypatch):
    """Every ``BoundShards`` the streamed loader creates, with its own fds.

    Counting every descriptor on a shard would also count MLX's lazy-load
    readers, which live as long as any raw array does; the claim under test
    is that *our* bound descriptors are released."""
    from mlx2.runtime import streamed_load

    created = []

    class Capturing(BoundShards):
        def __init__(self, *args, **kwargs):
            created.append(self)
            self.bound_fds = {}
            super().__init__(*args, **kwargs)
            for fd in self._fds.values():
                info = os.fstat(fd)
                self.bound_fds[fd] = (info.st_dev, info.st_ino)

    monkeypatch.setattr(streamed_load, "BoundShards", Capturing)
    return created


def _assert_released(created):
    assert created, "the streamed loader never bound its shards"
    for shards in created:
        assert shards._closed and not shards._fds
        for (fd, identity) in shards.bound_fds.items():
            try:
                info = os.fstat(fd)
            except OSError:
                continue
            assert (info.st_dev, info.st_ino) != identity, "a bound descriptor is still open"


# ---------------------------------------------------------------------------
# bound descriptors and source identity
# ---------------------------------------------------------------------------


def test_bound_shards_refuse_identity_drift_unknown_shards_and_escapes(tmp_path, monkeypatch):
    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
    names = _shards(weight_map)
    stale = [(n, size, mtime + 1) for (n, size, mtime) in records]
    with pytest.raises(WeightSourceChanged):
        BoundShards(root, names, records=stale, weight_map=weight_map)
    with pytest.raises(StreamingUnavailable, match="identity"):
        BoundShards(root, names, records=records[:1], weight_map=weight_map)
    with pytest.raises(StreamingUnavailable, match="escapes"):
        BoundShards(root, ["../outside.safetensors"], records=records)
    paths = [root / n for n in names]
    assert _descriptors_open_on(paths) == 0, "a refused bind must close what it opened"
    bound = BoundShards(root, names, records=records, weight_map=weight_map)
    assert _descriptors_open_on(paths) == len(paths)
    assert bound.sources_receipt() == sorted([list(r) for r in records])
    bound.close()
    bound.close()  # idempotent
    assert _descriptors_open_on(paths) == 0


def test_a_stray_unindexed_shard_is_never_bound(tmp_path, monkeypatch):
    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
    mx.save_safetensors(str(root / "stray.safetensors"), {"x": mx.zeros((2,))})
    bound = BoundShards(root, _shards(weight_map), records=records, weight_map=weight_map)
    try:
        assert all(p.name != "stray.safetensors" for p in bound.paths)
        assert "x" not in bound.index
    finally:
        bound.close()


def test_mutation_truncation_and_rename_replace_are_all_refused(tmp_path, monkeypatch):
    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
    names = _shards(weight_map)
    expert = root / names[1]
    bound = BoundShards(root, names, records=records, weight_map=weight_map)
    try:
        bound.check_all()
        before = bound.pread(expert, 0, 64)
        # Atomic rename-replace: the bound inode itself is unchanged and still
        # readable (so bytes of two revisions can never mix), but unlinking it
        # moves its ctime, so the strict detector refuses it as well.
        replacement = root / "replacement.tmp"
        replacement.write_bytes(expert.read_bytes()[::-1])
        os.replace(replacement, expert)
        assert bound.pread(expert, 0, 64) == before
        with pytest.raises(WeightSourceChanged):
            bound.check(expert)
        with pytest.raises(WeightSourceChanged):
            bound.verify_paths()
    finally:
        bound.close()

    (root, weight_map, records) = _moe_checkpoint(tmp_path / "again", monkeypatch)
    expert = root / _shards(weight_map)[1]
    bound = BoundShards(root, _shards(weight_map), records=records, weight_map=weight_map)
    try:
        info = os.stat(expert)
        os.utime(expert, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
        with pytest.raises(WeightSourceChanged):
            bound.check(expert)
    finally:
        bound.close()

    (root, weight_map, records) = _moe_checkpoint(tmp_path / "third", monkeypatch)
    expert = root / _shards(weight_map)[1]
    bound = BoundShards(root, _shards(weight_map), records=records, weight_map=weight_map)
    try:
        os.truncate(expert, os.stat(expert).st_size // 2)
        with pytest.raises(WeightSourceChanged):
            bound.check(expert)
    finally:
        bound.close()


def test_origins_accept_renames_and_ledgered_output_axis_concat_only(tmp_path):
    a = mx.zeros((4, 6, 8), dtype=mx.uint32)
    b = mx.ones((4, 6, 8), dtype=mx.uint32)
    c = mx.zeros((2, 6, 8), dtype=mx.uint32)  # an extra "expert" row block
    path = tmp_path / "parts.safetensors"
    mx.save_safetensors(str(path), {"p.a": a, "p.b": b, "p.c": c})
    bound = BoundShards(tmp_path, ["parts.safetensors"])
    try:
        origins = TensorOrigins(bound.index, name_of=bound.name_of)
        raw = mx.load(str(path))
        origins.record_shard(bound.paths[0], raw)
        renamed = {"live.a": raw["p.a"]}
        spec = origins.resolve_expert("live", "weight", renamed["live.a"])
        assert spec.parts[0].shape == (4, 6, 8)
        with concat_ledger(origins.concats):
            fused = mx.concatenate([raw["p.a"], raw["p.b"]], axis=-2)
            record_concat(fused, [raw["p.a"], raw["p.b"]], -2)
            stacked = mx.concatenate([raw["p.a"], raw["p.c"]], axis=0)
            record_concat(stacked, [raw["p.a"], raw["p.c"]], 0)
        spec = origins.resolve_expert("live", "weight", fused)
        assert len(spec.parts) == 2 and spec.expert_shape_concat() == (12, 8)
        with pytest.raises(StreamingUnavailable, match="output axis"):
            origins.resolve_expert("live", "weight", stacked)
        with pytest.raises(StreamingUnavailable, match="transformed"):
            origins.resolve_expert("live", "weight", raw["p.a"].astype(mx.int32))
        with pytest.raises(StreamingUnavailable, match="renames only"):
            origins.resolve_tensor("live", "weight", fused)
        # Outside a ledger, recording is a no-op.
        record_concat(fused, [raw["p.a"]], -2)
    finally:
        bound.close()


def test_declarations_are_never_inherited():
    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.adapters.nemotron3_super import Nemotron3SuperAdapter
    from mlx2.adapters.qwen35_122b import Qwen35122BA10BAdapter
    from mlx2.adapters.qwen35_9b import Qwen359BAdapter
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    assert declared_stream_modes(Qwen3827BAdapter) == {"dense_mlp"}
    assert declared_stream_modes(Qwen3635BA3BAdapter) == {"moe_experts"}
    assert declared_stream_modes(Qwen35122BA10BAdapter) == {"moe_experts"}
    for undeclared in (Qwen359BAdapter, FlashNextAdapter, Nemotron3SuperAdapter):
        assert declared_stream_modes(undeclared) == frozenset()
    dense = WeightStreamRequest(mode="dense_mlp", budget_bytes=1 << 20)
    with pytest.raises(ValueError, match="does not declare"):
        require_declared(Qwen359BAdapter, dense)
    with pytest.raises(ValueError, match="does not declare"):
        # Refused before the (missing) artifact is touched.
        Qwen359BAdapter("/missing/artifact", weight_streaming=dense)
    with pytest.raises(ValueError, match="does not declare"):
        Qwen3635BA3BAdapter("/missing/artifact", weight_streaming=dense)


@pytest.mark.parametrize(
    "bad",
    [
        dict(mode="nope", budget_bytes=1),
        dict(mode="moe_experts", budget_bytes=0),
        dict(mode="moe_experts", budget_bytes=True),
        dict(mode="moe_experts", budget_bytes=1.5),
        dict(mode="moe_experts", budget_bytes=1, read_workers=0),
        dict(mode="moe_experts", budget_bytes=1, mtp_resident=1),
        dict(mode="moe_experts", budget_bytes=1, admission_limit_bytes=-1),
    ],
)
def test_stream_request_validation(bad):
    with pytest.raises(ValueError):
        WeightStreamRequest(**bad)


# ---------------------------------------------------------------------------
# early-load MoE experts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fused", [False, True], ids=["split", "ledgered-fusion"])
def test_early_load_never_evaluates_streamed_tables(tmp_path, monkeypatch, fused):
    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
    spy = _EvalSpy(monkeypatch, _is_expert)
    model = _moe_model(monkeypatch, fused=fused)
    loaded = _streamed_moe(model, root, weight_map, records, budget=1 << 30)
    try:
        assert spy.watched, "the instrument saw no expert arrays at all"
        assert spy.watched_evaluated() == []
        # Positive control: the retained remainder WAS explicitly evaluated.
        params = dict(tree_flatten(model.parameters()))
        assert spy.was_evaluated(params["language_model.model.embed_tokens.weight"])
        assert not any(".switch_mlp." in key for key in params)
        plan = loaded.manager.plan.as_dict()
        assert plan["load_phase"] == "pre_materialization"
        assert plan["load_peak_bounded"] is True
        assert plan["top_k"] == TINY_MOE["num_experts_per_tok"]
        assert plan["excluded_bytes"] > 0
        assert plan["resident_remainder_bytes"] == loaded.remainder_bytes > 0
        # No out-of-tree reference to a raw expert array survives the load:
        # each is held exactly as often as a control held only by a list.
        control = [mx.zeros((1,))]
        (baseline,) = [sys.getrefcount(array) for array in control]
        counts = [sys.getrefcount(array) for array in spy.watched]
        assert counts == [baseline] * len(spy.watched)
    finally:
        loaded.manager.close()


def test_ordinary_loader_trips_the_same_instrument(tmp_path, monkeypatch):
    """Negative control: the shard-eager reference path evaluates experts."""
    (root, weight_map, _records) = _moe_checkpoint(tmp_path, monkeypatch)
    spy = _EvalSpy(monkeypatch, _is_expert)
    model = _moe_model(monkeypatch, fused=False)
    _resident(model, root, weight_map, prune=model.shard_prune)
    assert spy.watched and spy.watched_evaluated()


@pytest.mark.parametrize("fused", [False, True], ids=["split", "ledgered-fusion"])
@pytest.mark.parametrize("budget", [SMALL_BUDGET, 1 << 30], ids=["evicting", "roomy"])
def test_streamed_moe_matches_the_stock_resident_reference(tmp_path, monkeypatch, fused, budget):
    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
    reference = _resident(
        _moe_model(monkeypatch, fused=fused), root, weight_map, prune=None
    )
    model = _moe_model(monkeypatch, fused=fused)
    loaded = _streamed_moe(model, root, weight_map, records, budget=budget)
    manager = loaded.manager
    try:
        assert sum(manager.forced_stock.values()) > 0
        # Prefill (wide: routes to every expert) then decode steps.
        cache_s, cache_r = model.make_cache(), reference.make_cache()
        assert np.array_equal(
            _logits(model, PROMPT, cache_s), _logits(reference, PROMPT, cache_r)
        )
        for token in (5, 40, 2, 2, 61):
            assert np.array_equal(
                _logits(model, [token], cache_s), _logits(reference, [token], cache_r)
            )
        stats = manager.stats
        assert stats.page_ins > 0 and stats.load_page_ins == 0
        assert stats.peak_transient_bytes <= manager.plan.transient_bound_bytes
        assert stats.peak_tracked_bytes <= manager.reserved_bytes()
        assert stats.peak_staging_bytes <= manager.plan.staging_window * manager.plan.expert_bytes
        assert stats.resident_bytes == manager.live_bytes() <= manager.plan.ceiling_bytes
        if budget == SMALL_BUDGET:
            assert manager.plan.capacity_experts < TINY_MOE["num_experts"]
            assert stats.evictions > 0 and stats.overflows > 0
            assert manager.unfilled_reserve_bytes() == (
                manager.reserved_bytes() - manager.live_bytes()
            )
    finally:
        manager.close()
    assert manager.live_bytes() == 0


def test_host_staging_window_bounds_live_payload_objects():
    """Lifetime, not a counter mirror: count real payload objects alive while
    a wide miss set streams through the window."""
    import gc
    import threading as _threading
    from concurrent.futures import ThreadPoolExecutor

    lock = _threading.Lock()
    live = {"now": 0, "peak": 0}

    class Payload:
        def __init__(self):
            with lock:
                live["now"] += 1
                live["peak"] = max(live["peak"], live["now"])

        def __del__(self):
            with lock:
                live["now"] -= 1

    window = 4
    reader = object.__new__(weight_stream.ExpertSliceReader)
    reader._pool = ThreadPoolExecutor(max_workers=2)
    reader.window = window
    reader._outstanding = 0
    reader.expert_bytes = 1
    reader.read_bytes = lambda _expert: Payload()
    consumer_peak = 0
    try:
        for (_expert, payload) in reader.iter_many(range(40)):
            consumer_peak = max(consumer_peak, reader.outstanding_bytes())
            del payload
            gc.collect()
    finally:
        reader._pool.shutdown(wait=True)
    gc.collect()
    assert live["now"] == 0
    assert live["peak"] <= window, f"{live['peak']} payloads alive for a window of {window}"
    assert consumer_peak <= window


def test_transient_window_restarts_at_each_forward(tmp_path, monkeypatch):
    """A one-layer model repeats its group every forward; the window must not
    accumulate across forwards whose host index read evaluated the last one."""
    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch, interval=1, layers=1)
    model = _moe_model(monkeypatch, fused=True, interval=1, layers=1)
    loaded = _streamed_moe(model, root, weight_map, records, budget=1 << 30)
    try:
        _logits(model, PROMPT)
        first = loaded.manager.stats.peak_transient_bytes
        for _ in range(3):
            _logits(model, PROMPT)
        assert first > 0
        assert loaded.manager.stats.peak_transient_bytes == first
    finally:
        loaded.manager.close()


def test_a_mutation_during_reads_fails_before_rows_are_consumed(tmp_path, monkeypatch):
    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
    model = _moe_model(monkeypatch, fused=False)
    loaded = _streamed_moe(model, root, weight_map, records, budget=1 << 30)
    manager = loaded.manager
    try:
        shards = manager._handles
        expert_path = root / _shards(weight_map)[1]
        real_pread = shards.pread

        def mutating_pread(path, offset, length):
            data = real_pread(path, offset, length)
            info = os.stat(expert_path)
            os.utime(expert_path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
            return data

        shards.pread = mutating_pread
        cache = next(iter(manager.caches.values()))
        resident = cache.resident
        with pytest.raises(WeightSourceChanged):
            cache.acquire([0, 1])
        assert cache.resident == resident, "rows read across a mutation were kept"
    finally:
        manager.close()


def test_streaming_installs_once(tmp_path, monkeypatch):
    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
    model = _moe_model(monkeypatch, fused=False)
    loaded = _streamed_moe(model, root, weight_map, records, budget=1 << 30)
    try:
        with pytest.raises(StreamingUnavailable, match="installs once"):
            install_expert_streaming(model, root, ceiling_bytes=1 << 30, top_k=2)
    finally:
        loaded.manager.close()


def test_load_and_serving_counters_are_separate(tmp_path, monkeypatch):
    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
    model = _moe_model(monkeypatch, fused=False)
    loaded = load_streamed(
        model, root, _shards(weight_map), request=_moe_request(1 << 30),
        sanitize=model.sanitize, quantize=_quantizer(model), records=records,
        weight_map=weight_map, top_k=2,
    )
    manager = loaded.manager
    try:
        _logits(model, PROMPT)  # e.g. the adapter's load-time dtype probe
        counters = manager.counters()
        assert counters["stream_load_page_ins_total"] > 0
        assert counters["stream_page_ins_total"] == 0
        manager.begin_serving()
        for cache in manager.caches.values():
            cache.clear()
        _logits(model, PROMPT)
        assert manager.counters()["stream_page_ins_total"] > 0
    finally:
        manager.close()


def test_preload_budget_refuses_before_any_evaluation(tmp_path, monkeypatch, bound_shards):
    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
    spy = _EvalSpy(monkeypatch, lambda name: True)
    model = _moe_model(monkeypatch, fused=False)
    with pytest.raises(WorkingSetTooSmall, match="admission allows"):
        load_streamed(
            model, root, _shards(weight_map),
            request=_moe_request(1 << 30, admission_limit_bytes=1 << 20),
            sanitize=model.sanitize, quantize=_quantizer(model), records=records,
            weight_map=weight_map, top_k=2,
        )
    assert spy.watched and spy.watched_evaluated() == []
    _assert_released(bound_shards)


def test_ceiling_below_one_routed_step_refuses_with_the_real_top_k(tmp_path, monkeypatch, bound_shards):
    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
    model = _moe_model(monkeypatch, fused=False)
    # 12 projections x top-2 x 2560 B is the floor; one expert per layer is not.
    with pytest.raises(WorkingSetTooSmall, match="floor"):
        load_streamed(
            model, root, _shards(weight_map), request=_moe_request(12 * 1 * 2560),
            sanitize=model.sanitize, quantize=_quantizer(model), records=records,
            weight_map=weight_map, top_k=2,
        )
    _assert_released(bound_shards)


def test_untracked_sanitize_transforms_are_refused(tmp_path, monkeypatch, bound_shards):
    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
    model = _moe_model(monkeypatch, fused=False)

    def casting_sanitize(weights):
        out = model.sanitize(weights)
        key = "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales"
        out[key] = out[key].astype(mx.float16).astype(mx.float32)
        return out

    with pytest.raises(StreamingUnavailable, match="transformed during sanitize"):
        load_streamed(
            model, root, _shards(weight_map), request=_moe_request(1 << 30),
            sanitize=casting_sanitize, quantize=_quantizer(model), records=records,
            weight_map=weight_map, top_k=2,
        )
    _assert_released(bound_shards)


def test_shared_expert_folding_is_refused_through_the_ledger(tmp_path, monkeypatch):
    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
    model = _moe_model(monkeypatch, fused=False, shared_fold=True)
    with pytest.raises(StreamingUnavailable):
        load_streamed(
            model, root, _shards(weight_map), request=_moe_request(1 << 30),
            sanitize=model.sanitize, quantize=_quantizer(model), records=records,
            weight_map=weight_map, top_k=2,
        )


# ---------------------------------------------------------------------------
# the adapters' own load seams (real constructors, stub tokenizer only)
# ---------------------------------------------------------------------------


def _stub_tokenizer(monkeypatch, *modules, fail=False):
    import transformers

    from mlx2.runtime import tokenizer_integrity, tokenizer_utils

    class Tokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            if fail:
                raise RuntimeError("tokenizer exploded after the stream was installed")
            return SimpleNamespace(name="stub")

    monkeypatch.setattr(transformers, "AutoTokenizer", Tokenizer)
    monkeypatch.setattr(tokenizer_integrity, "repair_loaded_tokenizer", lambda *_a, **_k: {})
    monkeypatch.setattr(
        tokenizer_utils, "TokenizerWrapper",
        lambda tokenizer, **kwargs: SimpleNamespace(tokenizer=tokenizer, **kwargs),
    )
    for module in modules:
        monkeypatch.setattr(module, "resolve_eos_token_ids", lambda *_a, **_k: [0])


def _artifact(root, weight_map, records, text):
    return {
        "config": {"model_type": "qwen3_5_moe", "text_config": dict(text),
                   "quantization": dict(QUANT)},
        "weight_map": weight_map,
        "has_mtp": False,
        "mtp_tensor_count": 0,
        "identity": {"path": str(root), "fingerprint": "0" * 64, "files": records},
    }


def test_qwen36_adapter_streams_before_materializing(tmp_path, monkeypatch, bound_shards):
    from mlx2.adapters import qwen36_35b as adapter_module

    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
    _moe_model(monkeypatch, fused=True)  # pin the fused flag for the adapter
    artifact = _artifact(root, weight_map, records, TINY_MOE)
    monkeypatch.setattr(adapter_module, "inspect_artifact", lambda _p: artifact)
    monkeypatch.setattr(
        adapter_module, "configure_environment",
        lambda *_a, **_k: {"MLX_LM_COMPILED_DECODE": "0"},
    )
    _stub_tokenizer(monkeypatch, adapter_module)
    spy = _EvalSpy(monkeypatch, _is_expert)
    adapter = adapter_module.Qwen3635BA3BAdapter(
        str(root), weight_streaming=_moe_request(1 << 30)
    )
    manager = adapter.weight_stream
    try:
        assert manager is not None and manager in adapter._tables
        assert spy.watched and spy.watched_evaluated() == []
        receipt = manager.receipt()
        assert receipt["load_phase"] == "pre_materialization"
        assert receipt["forced_stock"]["fused_expert_kernel"] == TINY_MOE["num_hidden_layers"]
        assert receipt["qualification"] == "unqualified"
        # The dtype probe ran on the streamed model: load evidence only.
        assert manager.stats.load_page_ins > 0 and manager.stats.page_ins == 0
        assert manager.stats.phase == "serve"
        _logits(adapter.model, PROMPT)
        assert manager.stats.page_ins > 0
    finally:
        adapter.close()
    adapter.close()  # idempotent
    _assert_released(bound_shards)


def test_qwen36_constructor_failure_after_install_closes_descriptors(tmp_path, monkeypatch, bound_shards):
    from mlx2.adapters import qwen36_35b as adapter_module

    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
    artifact = _artifact(root, weight_map, records, TINY_MOE)
    monkeypatch.setattr(adapter_module, "inspect_artifact", lambda _p: artifact)
    monkeypatch.setattr(adapter_module, "configure_environment", lambda *_a, **_k: {})
    _stub_tokenizer(monkeypatch, adapter_module, fail=True)
    with pytest.raises(RuntimeError, match="exploded"):
        adapter_module.Qwen3635BA3BAdapter(
            str(root), weight_streaming=_moe_request(1 << 30)
        )
    _assert_released(bound_shards)
    assert _descriptors_open_on([root / n for n in _shards(weight_map)]) == 0


def test_qwen36_refuses_the_routed_candidate_with_streaming():
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter

    with pytest.raises(ValueError, match="moe_routed_candidate"):
        Qwen3635BA3BAdapter(
            "/missing", execution_policy={"moe_routed_candidate": True},
            weight_streaming=_moe_request(1 << 30),
        )


def test_qwen35_122b_adapter_streams_before_materializing(tmp_path, monkeypatch):
    from mlx2.adapters import qwen35_122b as adapter_module

    (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
    _moe_model(monkeypatch, fused=False)
    artifact = _artifact(root, weight_map, records, TINY_MOE)
    monkeypatch.setattr(adapter_module, "inspect_artifact", lambda _p: artifact)
    monkeypatch.setattr(adapter_module, "configure_environment", lambda *_a, **_k: {})
    _stub_tokenizer(monkeypatch, adapter_module)
    spy = _EvalSpy(monkeypatch, _is_expert)
    adapter = adapter_module.Qwen35122BA10BAdapter(
        str(root), weight_streaming=_moe_request(SMALL_BUDGET)
    )
    try:
        manager = adapter.weight_stream
        assert spy.watched and spy.watched_evaluated() == []
        assert manager.plan.load_phase == "pre_materialization"
        reference = _resident(
            _moe_model(monkeypatch, fused=False), root, weight_map
        )
        assert np.array_equal(_logits(adapter.model, PROMPT), _logits(reference, PROMPT))
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# dense trunk-MLP paging
# ---------------------------------------------------------------------------


def test_dense_streaming_matches_resident_and_pages_every_forward(tmp_path, monkeypatch):
    (root, weight_map, records) = _dense_checkpoint(tmp_path)
    # The eager reference evaluates its MLP tables by design; build it before
    # the instrument watches the streamed load.
    reference = _resident(_dense_model(), root, weight_map)
    spy = _EvalSpy(monkeypatch, _is_trunk_mlp)
    model = _dense_model()
    loaded = _streamed_dense(model, root, weight_map, records)
    manager = loaded.manager
    try:
        assert spy.watched and spy.watched_evaluated() == []
        modules = dict(model.named_modules())
        mlp = [p for p in modules if p.endswith((".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj"))]
        assert len(mlp) == 3 * TINY_DENSE["num_hidden_layers"]
        for path in mlp:
            assert not isinstance(modules[path], nn.QuantizedLinear)
        # Attention, GDN projections, embeddings and lm_head stay resident.
        assert isinstance(modules["language_model.lm_head"], nn.QuantizedLinear)
        assert isinstance(modules["language_model.model.layers.0.linear_attn.in_proj_qkv"],
                          nn.QuantizedLinear)
        cache_s, cache_r = model.make_cache(), reference.make_cache()
        before = manager.stats.page_ins
        assert np.array_equal(_logits(model, PROMPT, cache_s), _logits(reference, PROMPT, cache_r))
        assert manager.stats.page_ins - before == len(mlp)
        for token in (9, 33):
            assert np.array_equal(
                _logits(model, [token], cache_s), _logits(reference, [token], cache_r)
            )
        assert manager.stats.page_ins - before == 3 * len(mlp)
        # Each projection is evaluated before the next is read: the tracked
        # peak is one projection (host + device), never their sum.
        assert manager.stats.peak_tracked_bytes == manager.plan.transient_bound_bytes
        assert manager.plan.transient_bound_bytes == 2 * manager.plan.max_projection_bytes
        assert manager.plan.excluded_bytes > 2 * manager.plan.max_projection_bytes
        assert manager.counters()["dense_stream_page_ins_total"] == 3 * len(mlp)
    finally:
        manager.close()


def test_dense_staging_below_the_largest_projection_refuses_before_evaluation(
    tmp_path, monkeypatch, bound_shards
):
    (root, weight_map, records) = _dense_checkpoint(tmp_path)
    spy = _EvalSpy(monkeypatch, lambda name: True)
    with pytest.raises(WorkingSetTooSmall, match="twice"):
        _streamed_dense(_dense_model(), root, weight_map, records, staging=1024)
    assert spy.watched and spy.watched_evaluated() == []
    _assert_released(bound_shards)


def _dense_artifact(root, weight_map, records):
    artifact = _artifact(root, weight_map, records, TINY_DENSE)
    artifact["config"]["model_type"] = "qwen3_5"
    return artifact


def test_qwen38_adapter_dense_seam_and_refusals(tmp_path, monkeypatch, bound_shards):
    from mlx2.adapters import qwen38_27b as adapter_module
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    (root, weight_map, records) = _dense_checkpoint(tmp_path)
    artifact = _dense_artifact(root, weight_map, records)
    monkeypatch.setattr(Qwen3827BAdapter, "artifact_inspector", staticmethod(lambda _p: artifact))
    monkeypatch.setattr(
        Qwen3827BAdapter, "environment_configurator",
        staticmethod(lambda: {"MLX_LM_COMPILED_DECODE": "0"}),
    )
    _stub_tokenizer(monkeypatch, adapter_module)
    request = WeightStreamRequest(mode="dense_mlp", budget_bytes=1 << 30, read_workers=2)
    with pytest.raises(ValueError, match="external draft"):
        Qwen3827BAdapter(str(root), execution_policy={"draft_model": "x"},
                         weight_streaming=request)
    with pytest.raises(ValueError, match="TensorFold"):
        Qwen3827BAdapter(str(root), execution_policy={"tensorfold_prefill": True},
                         weight_streaming=request)
    with pytest.raises(ValueError, match="does not declare"):
        Qwen3827BAdapter(str(root), weight_streaming=_moe_request(1 << 30))
    spy = _EvalSpy(monkeypatch, _is_trunk_mlp)
    adapter = Qwen3827BAdapter(str(root), require_mtp=False, weight_streaming=request,
                               execution_policy={"fp32_head_logits": True})
    try:
        manager = adapter.weight_stream
        assert spy.watched and spy.watched_evaluated() == []
        assert manager.mode == "dense_mlp" and manager in adapter._tables
        assert manager.stats.load_page_ins > 0 and manager.stats.page_ins == 0
        _logits(adapter.model, PROMPT)
        assert manager.stats.page_ins == 3 * TINY_DENSE["num_hidden_layers"]
        assert adapter.fp32_head  # lm_head stays resident, so fp32 logits compose
    finally:
        adapter.close()
    _assert_released(bound_shards)


def test_qwen38_constructor_failure_after_dense_install_closes(tmp_path, monkeypatch):
    from mlx2.adapters import qwen38_27b as adapter_module
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    (root, weight_map, records) = _dense_checkpoint(tmp_path)
    artifact = _dense_artifact(root, weight_map, records)
    monkeypatch.setattr(Qwen3827BAdapter, "artifact_inspector", staticmethod(lambda _p: artifact))
    monkeypatch.setattr(Qwen3827BAdapter, "environment_configurator", staticmethod(lambda: {}))
    _stub_tokenizer(monkeypatch, adapter_module, fail=True)
    with pytest.raises(RuntimeError, match="exploded"):
        Qwen3827BAdapter(
            str(root),
            weight_streaming=WeightStreamRequest(mode="dense_mlp", budget_bytes=1 << 30),
        )
    assert _descriptors_open_on([root / n for n in _shards(weight_map)]) == 0


# ---------------------------------------------------------------------------
# APCv2: warm exact prefix continues exactly and avoids prefix I/O
# ---------------------------------------------------------------------------


def _clear_expert_caches(manager):
    for cache in getattr(manager, "caches", {}).values():
        cache.clear()


@pytest.mark.parametrize("mode", ["moe_experts", "dense_mlp"])
def test_apcv2_warm_prefix_continues_exactly_and_avoids_prefix_page_ins(tmp_path, monkeypatch, mode):
    from mlx2.runtime.apc_v2 import APCKey, APCv2

    # The real hybrid topology: gated-delta recurrent layers (ArraysCache,
    # restored only at exact checkpoints) plus full attention (KVCache).
    prefix, suffix = PROMPT, [11, 2, 47]
    if mode == "moe_experts":
        (root, weight_map, records) = _moe_checkpoint(tmp_path, monkeypatch)
        reference = _resident(_moe_model(monkeypatch, fused=True), root, weight_map)
        model = _moe_model(monkeypatch, fused=True)
        manager = _streamed_moe(model, root, weight_map, records, budget=1 << 30).manager
    else:
        (root, weight_map, records) = _dense_checkpoint(tmp_path)
        reference = _resident(_dense_model(), root, weight_map)
        model = _dense_model()
        manager = _streamed_dense(model, root, weight_map, records).manager
    try:
        # Reference: the same chunking, resident, no APC.
        cold_cache = reference.make_cache()
        _logits(reference, prefix, cold_cache)
        expected = _logits(reference, suffix, cold_cache)

        # Cold streamed request: prefix and suffix both paged in.
        _clear_expert_caches(manager)
        start = manager.stats.page_ins
        cache = model.make_cache()
        _logits(model, prefix, cache)
        cold = _logits(model, suffix, cache)
        cold_page_ins = manager.stats.page_ins - start
        assert np.array_equal(cold, expected)

        # A first request publishes its prefix state into APCv2 ...
        apc = APCv2(max_size=4, layout_name="disk-stream-test-v1")
        key = APCKey(apc_weight_stream_fingerprint("text-token-v1", manager))
        publish = model.make_cache()
        _logits(model, prefix, publish)
        mx.eval([c.state for c in publish])
        capabilities = apc.store(key, prefix, publish)
        assert capabilities.topology == "checkpointed_hybrid"
        # ... the weight cache is then cold again (evicted), and a warm
        # request restores only request state: no reader, no weight entries.
        _clear_expert_caches(manager)
        hit = apc.lookup(key, prefix + suffix)
        assert hit.hit and hit.cached_tokens == len(prefix)
        assert hit.remaining_tokens == suffix
        restored = list(hit.cache)
        assert all(not hasattr(c, "_stream_reader") for c in restored)
        start = manager.stats.page_ins
        warm = _logits(model, suffix, restored)
        warm_page_ins = manager.stats.page_ins - start
        assert np.array_equal(warm, expected), "warm continuation diverged from cold"
        assert 0 < warm_page_ins < cold_page_ins
        close = getattr(hit.cache, "close", None)
        if callable(close):
            close()
    finally:
        manager.close()


def test_apc_namespace_wrapper_is_identity_when_off_and_peelable_when_on():
    from mlx2.runtime.apc_v2 import _numerics_layers

    assert apc_weight_stream_fingerprint("text-token-v1", None) == "text-token-v1"
    manager = SimpleNamespace(apc_revision=lambda: "weight-stream-v2:moe_experts:pre_materialization")
    wrapped = apc_weight_stream_fingerprint(("text-token-v1", "lane-matmul", "x"), manager)
    (base, layers) = _numerics_layers(wrapped)
    assert base == "text-token-v1"
    assert layers[0] == ("weight-stream", "weight-stream-v2:moe_experts:pre_materialization")
    assert layers[1] == ("lane-matmul", "x")


# ---------------------------------------------------------------------------
# engine contract, admission, qualification and metrics
# ---------------------------------------------------------------------------


def test_dense_policy_validation():
    from mlx2.serving import dense_weight_streaming_policy

    assert dense_weight_streaming_policy(None) == {
        "enabled": False, "staging_gib": 0.0, "read_workers": 16,
    }
    for bad in (
        [], {"nope": 1}, {"enabled": 1}, {"enabled": True},
        {"staging_gib": float("nan")}, {"staging_gib": True},
        {"staging_gib": -1}, {"read_workers": 0}, {"read_workers": True},
    ):
        with pytest.raises(ValueError):
            dense_weight_streaming_policy(bad)
    assert dense_weight_streaming_policy({"enabled": True, "staging_gib": 1})[
        "staging_gib"
    ] == 1.0


class _DeclaredDense:
    weight_streaming_modes = frozenset({"dense_mlp"})
    calls = []

    def __init__(self, model_path, **kwargs):
        type(self).calls.append(kwargs)
        raise RuntimeError("stop after delivery")


class _DeclaredMoE(_DeclaredDense):
    weight_streaming_modes = frozenset({"moe_experts"})
    calls = []


class _Undeclared(_DeclaredMoE):  # inherits a declaration it must not get
    calls = []


DENSE_POLICY = {"dense_weight_streaming": {"enabled": True, "staging_gib": 0.5}}
MOE_POLICY = {"moe_expert_streaming": {"enabled": True, "cache_gib": 0.5}}


def _engine(factory, policy, **kwargs):
    from mlx2.serving import ServingEngine

    engine = ServingEngine(
        "unused", adapter_factory=factory, max_lanes=1, max_inflight=1,
        execution_policy=policy, **kwargs,
    )
    engine.thread.join(timeout=30)
    return engine


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lane_matmul": "exact"}, "lane_matmul=exact"),
        ({"lane_matmul": "crossover"}, "lane_matmul=crossover"),
        ({"execution_policy_extra": {"draft_model": "/x"}}, "external draft"),
        ({"execution_policy_extra": {"tensorfold_prefill": True}}, "tensorfold_prefill"),
        ({"execution_policy_extra": {"sp_qmm": True}}, "sp_qmm"),
    ],
)
def test_engine_refuses_unevidenced_combinations_before_loading(kwargs, match):
    from mlx2.serving import ServingEngine

    extra = kwargs.pop("execution_policy_extra", {})
    for policy in (DENSE_POLICY, MOE_POLICY):
        with pytest.raises(ValueError, match=match):
            ServingEngine.validate_arguments(
                "unused", max_lanes=1, max_inflight=1,
                execution_policy={**policy, **extra}, **kwargs,
            )


def test_engine_refuses_dense_on_undeclared_and_mixed_or_multilane():
    from mlx2.serving import ServingEngine

    with pytest.raises(ValueError, match="not declared"):
        ServingEngine("unused", adapter_factory=_Undeclared, max_lanes=1,
                      max_inflight=1, execution_policy=DENSE_POLICY)
    with pytest.raises(ValueError, match="mutually exclusive"):
        ServingEngine.validate_arguments(
            "unused", max_lanes=1, max_inflight=1,
            execution_policy={**DENSE_POLICY, **MOE_POLICY},
        )
    with pytest.raises(ValueError, match="max_lanes=1"):
        ServingEngine.validate_arguments(
            "unused", max_lanes=2, max_inflight=2, execution_policy=DENSE_POLICY
        )


def test_engine_delivers_a_typed_request_only_to_declaring_classes():
    _DeclaredDense.calls.clear()
    engine = _engine(_DeclaredDense, DENSE_POLICY, lane_matmul="auto")
    try:
        assert engine.weight_streaming_route == "early"
        assert engine.lane_matmul == "off"
        assert engine.weight_streaming_lane_matmul == "auto->off (weight streaming)"
        (call,) = _DeclaredDense.calls
        request = call["weight_streaming"]
        assert isinstance(request, WeightStreamRequest)
        assert request.mode == "dense_mlp" and request.budget_bytes == 1 << 29
        assert "execution_policy" not in call  # the stream policy is engine-owned
    finally:
        engine.close()

    _DeclaredMoE.calls.clear()
    engine = _engine(_DeclaredMoE, MOE_POLICY)
    try:
        assert engine.weight_streaming_route == "early"
        assert _DeclaredMoE.calls[0]["weight_streaming"].mode == "moe_experts"
    finally:
        engine.close()

    _Undeclared.calls.clear()
    engine = _engine(_Undeclared, MOE_POLICY)
    try:
        # Inherited declarations do not count: legacy post-materialization.
        assert engine.weight_streaming_route == "legacy"
        assert "weight_streaming" not in _Undeclared.calls[0]
    finally:
        engine.close()


class _Manager:
    def __init__(self, reserved, live):
        self._reserved, self._live = reserved, live

    def reserved_bytes(self):
        return self._reserved

    def unfilled_reserve_bytes(self):
        return max(0, self._reserved - self._live)


def test_reservation_is_unfilled_for_admission_and_full_for_oom_recovery():
    from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController
    from mlx2.serving import ServingEngine, moe_expert_streaming_policy

    engine = ServingEngine.__new__(ServingEngine)
    engine.moe_expert_streaming_policy = moe_expert_streaming_policy(
        {"enabled": True, "cache_gib": 4}
    )
    engine.expert_stream = _Manager(reserved=6 << 30, live=4 << 30)
    assert engine.expert_stream_reserve_gib() == pytest.approx(6.0)
    assert engine.stream_unfilled_reserve_gib() == pytest.approx(2.0)
    engine._hard_reserve_gib = 10.0
    assert engine.hard_reserve_gib == pytest.approx(12.0)

    plain = SelfMTPLaneAdmissionController(host_memory_gib=36.0)
    live = SelfMTPLaneAdmissionController(
        host_memory_gib=36.0, stream_reserve_gib=engine.stream_unfilled_reserve_gib
    )
    assert live.hard_reserve_gib == pytest.approx(plain.hard_reserve_gib + 2.0)
    engine.expert_stream._live = 6 << 30  # the cache filled: nothing left to reserve
    assert live.hard_reserve_gib == pytest.approx(plain.hard_reserve_gib)


def test_qualification_needs_serving_phase_page_ins():
    from mlx2.qualification import required_feature_checks
    from scripts.qualify_serving import feature_observations, unobservable_features

    base = {"speculation": "self_mtp", "execution_policy": {}, "environment": {}}
    dense = {**base, "dense_weight_streaming": {"enabled": True, "staging_gib": 1.0}}
    assert "feature_dense_weight_streaming" not in required_feature_checks(base)
    assert "feature_dense_weight_streaming" in required_feature_checks(dense)
    assert unobservable_features(["dense_weight_streaming", "moe_expert_streaming"]) == []
    load_only = {"counts": {"dense_stream_load_page_ins_total": 12,
                            "stream_load_page_ins_total": 9}}
    assert feature_observations(load_only)["dense_weight_streaming"] == 0
    assert feature_observations(load_only)["moe_expert_streaming"] == 0
    serving = {"counts": {"dense_stream_page_ins_total": 3}}
    assert feature_observations(serving)["dense_weight_streaming"] == 3


def test_optional_stream_metrics_render_only_when_present():
    import threading as _threading
    from collections import Counter

    from mlx2.batch_metrics import BatchRuntimeMetrics
    from mlx2.serving import ServingEngine

    def scrape(counters):
        engine = ServingEngine.__new__(ServingEngine)
        engine.lock = _threading.Lock()
        engine.counts = Counter()
        engine.queued_jobs = 0
        engine.snapshot = {"state": "ready", "model": "m", "qualification": "unqualified"}
        engine.batch_metrics = BatchRuntimeMetrics()
        engine.expert_stream = SimpleNamespace(counters=lambda: counters)
        return engine.prometheus_metrics()

    text = scrape({"dense_stream_page_ins_total": 5,
                   "dense_stream_peak_tracked_bytes": 1024})
    assert ('mlx2_runtime_events_total{component="dense_stream",event="page_in"} 5'
            in text)
    assert "mlx2_dense_stream_peak_tracked_bytes 1024" in text
    assert 'component="dense_stream"' not in scrape({})
