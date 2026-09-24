"""CPU tests for the Qwen prefill component profiler (tiny random-weight models)."""

import importlib.util
import math
from pathlib import Path

import mlx.core as mx
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile_prefill_qwen.py"
spec = importlib.util.spec_from_file_location("profile_prefill_qwen", SCRIPT)
prof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prof)


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _run(arch, tmp_path):
    out = tmp_path / f"{arch}.json"
    report, cache = prof.run(
        ["--cpu-tiny", "--tiny-arch", arch, "--repeats", "3", "--out", str(out)],
        return_cache=True,
    )
    assert out.exists()
    return report, cache


def _reference_cache(arch, report):
    """Plain production chunked prefill of the same stream, no profiler."""
    from mlx2.runtime.models.cache import make_prompt_cache

    model = prof.tiny_model(arch)
    plan = report["plan"]
    tokens = mx.array(prof.tiny_tokens(plan["context"]), dtype=mx.int32)[None]
    cache = make_prompt_cache(model)
    for start in range(0, plan["context"], plan["chunk"]):
        model(tokens[:, start:start + plan["chunk"]], cache=cache)
        mx.eval([c.state for c in cache])
    return cache


def _valid_arrays(c):
    from mlx2.runtime.models.cache import KVCache

    if type(c) is KVCache:
        return [c.keys[..., : c.offset, :], c.values[..., : c.offset, :]], c.offset
    return list(c.cache), None


@pytest.mark.parametrize("arch", ["dense", "moe"])
def test_cpu_tiny_end_to_end_structure(arch, tmp_path):
    report, _ = _run(arch, tmp_path)
    assert report["evidence"] is False and report["qualification"] is False
    assert report["architecture"]["moe"] is (arch == "moe")

    a = report["part_a"]
    plan = report["plan"]
    assert len(a["chunks"]) == plan["context"] // plan["chunk"]
    assert [c["start"] for c in a["chunks"]] == list(range(0, plan["context"], plan["chunk"]))
    assert all(c["ms"] > 0 for c in a["chunks"])
    assert a["total_prefill_s"] > 0 and a["tokens_per_s"] > 0
    assert math.isfinite(a["fit_all_chunks"]["slope_ms_per_1k_offset"])

    mlp = prof.MOE_SEAMS if arch == "moe" else prof.DENSE_MLP_SEAMS
    b = report["part_b"]
    assert [p["offset"] for p in b["probes"]] == plan["offsets"] == [0, 64, 128, 192]
    for p in b["probes"]:
        assert p["embed_ms"] > 0 and p["cache_unmutated"] is True
        assert p["capture_max_abs_diff"] == 0.0
        for kind, seams in (("gdn", prof.GDN_SEAMS), ("fa", prof.FA_SEAMS)):
            r = p[kind]
            assert set(r["seams_ms"]) == set(seams) | set(mlp)
            assert all(v > 0 for v in r["seams_ms"].values())
            assert r["whole_layer_ms"] > 0 and r["seam_sum_ms"] > 0
            assert sum(r["seam_share"].values()) == pytest.approx(1.0, abs=1e-9)
            assert math.isfinite(r["seam_sum_over_whole"]) and r["seam_sum_over_whole"] > 0
            assert r["mirror_max_abs_diff"] <= 1e-5
        assert report["architecture"]["layers"] == 8

    s = report["scaling"]
    assert s["layer_counts"] == {"gdn": 4, "fa": 4}
    assert set(s["category_share"]) == set(prof.CATEGORIES)
    assert sum(s["category_share"].values()) == pytest.approx(1.0, abs=1e-9)
    assert all(v >= 0 for v in s["category_ms"].values())
    used = {prof.SEAM_CATEGORY[n] for n in prof.GDN_SEAMS + prof.FA_SEAMS + mlp}
    assert all(s["category_ms"][c] > 0 for c in used)
    assert all(s["category_ms"][c] == 0 for c in set(prof.CATEGORIES) - used - {"other"})
    for key in ("ratio_whole_layer_estimate_over_measured", "ratio_seam_estimate_over_measured"):
        assert math.isfinite(s[key]) and s[key] > 0

    c = report["checks"]
    assert c["cache_unmutated"] is True
    assert c["capture_max_abs_diff"] == 0.0


@pytest.mark.parametrize("arch", ["dense", "moe"])
def test_part_b_leaves_the_real_prefill_state_bit_identical(arch, tmp_path):
    """Probing at every chunk must not change what the prefill computes."""
    report, profiled = _run(arch, tmp_path)
    reference = _reference_cache(arch, report)
    assert len(profiled) == len(reference)
    for live, ref in zip(profiled, reference):
        (la, lo), (ra, ro) = _valid_arrays(live), _valid_arrays(ref)
        assert lo == ro
        assert len(la) == len(ra)
        for x, y in zip(la, ra):
            assert (x is None) == (y is None)
            if x is not None:
                assert x.shape == y.shape
                assert mx.array_equal(x, y).item()


def test_copy_cache_isolates_live_state():
    from mlx2.runtime.models.cache import ArraysCache, KVCache

    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 2, 64, 8)), mx.ones((1, 2, 64, 8)))
    mx.eval(kv.keys, kv.values)
    before = prof.cache_fingerprint(mx, kv)
    c = prof.copy_cache(kv)
    c.update_and_fetch(mx.full((1, 2, 64, 8), 3.0), mx.full((1, 2, 64, 8), 3.0))  # in-buffer write
    mx.eval(c.keys)
    c2 = prof.copy_cache(kv)
    c2.update_and_fetch(mx.full((1, 2, 512, 8), 5.0), mx.full((1, 2, 512, 8), 5.0))  # growth
    mx.eval(c2.keys)
    assert kv.offset == 64 and prof.cache_fingerprint(mx, kv) == before

    ac = ArraysCache(2)
    ac[0], ac[1] = mx.zeros((1, 2, 4)), mx.ones((1, 2, 3, 3))
    before = prof.cache_fingerprint(mx, ac)
    cc = prof.copy_cache(ac)
    cc[0], cc[1] = mx.ones((1, 2, 4)), mx.zeros((1, 2, 3, 3))
    assert prof.cache_fingerprint(mx, ac) == before


def test_offsets_interp_and_gpu_refusal():
    assert prof.default_offsets(131072, 2048) == [0, 32768, 65536, 129024]
    assert prof.default_offsets(256, 64) == [0, 64, 128, 192]
    with pytest.raises(SystemExit):
        prof.parse_offsets("100", 256, 64)
    assert prof.interp(-5, [0, 10], [1.0, 3.0]) == 1.0
    assert prof.interp(5, [0, 10], [1.0, 3.0]) == 2.0
    assert prof.interp(50, [0, 10], [1.0, 3.0]) == 3.0
    fit = prof.linear_fit([0, 1, 2], [1.0, 3.0, 5.0])
    assert fit["slope_ms_per_1k_offset"] == pytest.approx(2.0)
    with pytest.raises(SystemExit, match="i-own-the-gpu"):
        prof.run(["--model", "/nonexistent"])
    plan = prof.run(["--model", "/nonexistent", "--dry-run"])
    assert plan["offsets"] == [0, 32768, 65536, 129024]
