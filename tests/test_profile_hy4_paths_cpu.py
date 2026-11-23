"""scripts/profile_hy4_paths.py on the tiny Hy4-flash fixtures (CPU, tiny sizes).

Checks the guards (no Metal without --i-own-the-gpu, no overwrite of a
nonempty output directory) and the JSON schema of every section: decode
(pipelined + split + syncs + graph ops + components), DFly2 AST rounds with
the verify-shaped components, and prefill chunks (sampled components,
effective TFLOP/s).  Timings on the CPU are not meaningful; only the shape
of the evidence is asserted.
"""

import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_cpu import _tiny_artifact, checkpoint, tiny_config
from test_hy_v4_flash_mlx_pack_cpu import pack_config, write_pack

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("profile_hy4_paths", ROOT / "scripts/profile_hy4_paths.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)

TINY = ["--cpu", "--decode-contexts", "6", "24", "--decode-steps", "3", "--decode-warmup", "1",
        "--decode-split-steps", "2", "--component-reps", "1", "--component-warmup", "0",
        "--prefill-chunks", "12", "80", "--prefill-reps", "1", "--prefill-warmup", "0",
        "--prefill-component-reps", "1", "--prefill-step", "8", "--ast-max-tokens", "10"]


def _tokenizer(path, vocab):
    from tokenizers import Tokenizer, models

    Tokenizer(models.WordLevel({f"t{i}": i for i in range(vocab)}, unk_token="t0")).save(
        str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "t2"}))


def _check_components(c, *, sample):
    assert c["mode"] == ("sample" if sample else "all")
    assert c["leaf_sum_ms"] > 0 and c["step_ms"] > 0
    assert c["residual_ms"] == pytest.approx(c["step_ms"] - c["leaf_sum_ms"], abs=1e-3)
    if c["rows_per_forward"] <= P.MIRROR_CHECK_MAX_ROWS:
        assert c["mirror_exact"] is True
    names = {r["component"] for r in c["rows"]}
    for name in ("attention", "attn.project", "attn.finish", "attn.core", "attn.indexer", "moe",
                 "moe.router", "moe.experts", "moe.shared", "mlp", "engram", "hc_attn_pre",
                 "hc_merge_mlp", "head.lm_head", "head.sampler_argmax"):
        assert name in names, name
    for r in c["rows"]:
        for key in ("component", "parent", "group", "layers_measured", "layers_in_model", "per_layer_ms",
                    "ms", "pct_of_step", "bytes", "gb_s", "pct_of_peak_bw", "flops", "tflops", "derived"):
            assert key in r
        if not r["derived"]:
            assert r["ms"] >= 0
    by = {(r["component"], r["group"]): r for r in c["rows"]}
    total = sum(r["layers_in_model"] for (n, _), r in by.items() if n == "attention")
    assert total == 6  # every decoder layer is counted once, sampled or not
    assert all(v >= 1 for v in c["unique_experts"].values())


def _load(out):
    doc = json.loads((out / "profile.json").read_text())
    assert doc["schema"] == P.SCHEMA
    assert (out / "profile.md").read_text().startswith("# Hy4-flash path profile")
    assert doc["checkpoint"] and "@" in doc["checkpoint"]
    assert doc["mlx2"]["commit"] and doc["finished"] and doc["peak_gbs"] == 546.0
    assert doc["device"].startswith("Device(cpu")
    for section in doc["sections"]:
        assert {"before", "after"} <= set(doc["swap"][section])
    return doc


def test_guards(tmp_path):
    with pytest.raises(SystemExit):
        P.parse(["--model", "x", "--out", str(tmp_path / "o")])  # neither --cpu nor --i-own-the-gpu
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "keep").write_text("evidence")
    with pytest.raises(SystemExit):
        P.parse(["--model", "x", "--out", str(tmp_path / "full"), "--cpu"])
    with pytest.raises(SystemExit):
        P.parse(["--model", "x", "--out", str(tmp_path / "o"), "--cpu", "--sections", "decode,bogus"])


def test_all_sections_on_the_tiny_dfly_artifact(tmp_path):
    config = tiny_config(compressed_indexer_group_topk=8)  # dense below 16 positions
    (tmp_path / "art").mkdir()
    path = _tiny_artifact(tmp_path / "art", config, checkpoint(config))
    out = tmp_path / "out"
    _load_doc = P.main(["--model", str(path), "--out", str(out), *TINY])
    doc = _load(out)
    assert doc["sections"].keys() == {"decode", "ast", "prefill"}

    decode = doc["sections"]["decode"]
    assert decode["generator_b1"]["tokens"] == 3
    for ctx in ("6", "24"):
        cell = decode[ctx]
        assert cell["pipelined"]["tok_s"] > 0 and cell["pipelined"]["steps"] == 3
        s = cell["split"]
        assert s["steps"] == 2 and s["build_ms"] > 0 and s["gpu_wait_ms"] > 0
        assert s["engram_hash_ms"] >= 0 and s["engram_read_ms"] >= 0
        assert s["steps_detail"][0]["engram"]["rows_read"] > 0
        assert cell["host_syncs_per_step"]["waits"] >= 1
        ops = cell["graph_ops_per_step"]
        assert ops["estimated_kernels"] > 0 and ops["primitives"] >= ops["estimated_kernels"]
        _check_components(cell["components"], sample=False)
    groups = {r["group"] for r in decode["24"]["components"]["rows"]}
    assert "indexed_sparse_gather" in groups and "sliding" in groups
    assert "indexed_dense" in {r["group"] for r in decode["6"]["components"]["rows"]}

    ast = doc["sections"]["ast"]
    assert ast["plain"]["rounds"] >= 1 and ast["plain"]["committed_per_round_mean"] > 0
    inst = ast["instrumented"]["instrumented"]
    assert inst["verify_rounds"] >= 1 and inst["verify_ms_median"] > 0 and inst["draft_ms_median"] > 0
    assert inst["verify_rows_hist"] and inst["committed_per_verify_round_mean"] is not None
    assert ast["verify_rows"] >= 1 and ast["verify_forward"]["graph_ops"]["estimated_kernels"] > 0
    _check_components(ast["verify_components"], sample=False)
    assert ast["executor_receipt"]["observed_used"]

    prefill = doc["sections"]["prefill"]
    for chunk in ("12", "80"):
        cell = prefill[chunk]
        assert cell["rows"] == int(chunk) and cell["normal_ms"] > 0
        assert cell["engram_host_normal"]["rows_read"] > 0
        assert cell["effective_tflops"]["moe"]["flops"] > 0
        assert cell["effective_tflops"]["attention"]["flops"] > 0
        _check_components(cell["components"], sample=True)
    assert "indexed_sparse_mask" in {r["group"] for r in prefill["80"]["components"]["rows"]}
    assert any(r["scaled"] for r in prefill["80"]["components"]["rows"])


def test_quantized_pack_decode_and_prefill(tmp_path):
    """The pack layout (4-bit + q2_0 expert banks, 8-bit trunk): expert sub-stages and labels."""
    config = pack_config()
    root, _, _ = write_pack(tmp_path, config, checkpoint(config, seed=7, scale=0.2))
    _tokenizer(root, config["text_config"]["vocab_size"])
    out = tmp_path / "out"
    P.main(["--model", str(root), "--out", str(out), *TINY, "--sections", "decode,prefill",
            "--decode-contexts", "24", "--prefill-chunks", "12", "--no-generator-b1",
            "--policy-json", json.dumps({"vision": False})])
    doc = _load(out)
    assert set(doc["model"]["mlp"]) >= {"dense_mlp", "moe_q2_0", "moe_affine4"}
    rows = doc["sections"]["decode"]["24"]["components"]["rows"]
    names = {(r["component"], r["group"]) for r in rows}
    assert ("moe.experts.up", "moe_affine4") in names and ("moe.experts.down", "moe_q2_0") in names
    experts = [r for r in rows if r["component"] == "moe.experts"]
    assert all(r["bytes"] > 0 and r["gb_s"] is not None for r in experts)
    _check_components(doc["sections"]["prefill"]["12"]["components"], sample=True)
