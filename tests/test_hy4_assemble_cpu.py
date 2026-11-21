"""scripts/assemble_hy4_mlx_pack.py: an MLX pack + the original's DFly2/vision tensors, CPU.

Tiny fixtures: the MLX pack layout of test_hy_v4_flash_mlx_pack_cpu (no DFly2,
no vision) and a tiny "original" checkpoint that carries both.  The
assembled artifact must inspect with the drafter and the tower, hardlink the
pack's files, carry the copied tensors byte for byte, and serve DFly2 greedy
equal to ordinary greedy on the quantized pack trunk.
"""

import importlib.util
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from test_hy_v4_flash_cpu import TOKENS, _write_artifact, checkpoint  # noqa: E402
from test_hy_v4_flash_mlx_pack_cpu import pack_config, write_pack  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VISION = {"num_hidden_layers": 1, "hidden_size": 16}


def _assembler():
    spec = importlib.util.spec_from_file_location("assemble_hy4_mlx_pack",
                                                  ROOT / "scripts" / "assemble_hy4_mlx_pack.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vision_tensors():
    from mlx2.adapters.hy_v4_flash import vision_tensor_names

    rng = np.random.default_rng(5)
    return {name: rng.standard_normal((3, 4)).astype(np.float32) for name in vision_tensor_names(1)}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    config = pack_config()
    config.update(vision_config=VISION, image_token_id=90)
    W = checkpoint(config, seed=7, scale=0.2)
    original = _write_artifact(tmp_path_factory.mktemp("original"), config,
                               {**W, **_vision_tensors()})
    meta = original / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / "config.json.metadata").write_text("0123456789abcdef0123456789abcdef01234567\nx\n1.0\n")
    pack_dir, pack_cfg, ref = write_pack(tmp_path_factory.mktemp("pack"), config, W)
    (pack_dir / "hy_v4_flash.py").write_text("raise SystemExit('third-party code must not be carried')\n")
    out = tmp_path_factory.mktemp("assembled") / "out"
    assert _assembler().main([str(pack_dir), str(original), str(out)]) == 0
    return config, W, original, pack_dir, out


def test_assembled_artifact_inspects_with_dfly_and_vision(built):
    from mlx2.adapters import hy_v4_flash as A

    _, _, _, pack_dir, out = built
    plain, assembled = A.inspect_artifact(pack_dir), A.inspect_artifact(out)
    assert not plain["has_dfly"] and not plain["has_vision"]
    assert assembled["has_dfly"] and assembled["has_mtp"] and assembled["has_vision"]
    assert assembled["engram_table_formats"] == ["iq4_nl"]
    assert assembled["identity"]["fingerprint"] != plain["identity"]["fingerprint"]
    assert not (out / "hy_v4_flash.py").exists()  # pack code is never carried
    gguf = out / "engram-iq4nl.gguf"
    assert os.stat(gguf).st_ino == os.stat(pack_dir / "engram-iq4nl.gguf").st_ino  # hardlink
    record = json.loads((out / "config.json").read_text())["mlx2_assembly"]
    assert set(record["added_shards"]) == {"model-dfly2.safetensors", "model-vision.safetensors"}
    assert record["original"]["revision"] == "0123456789abcdef0123456789abcdef01234567"


def test_copied_tensors_are_byte_identical(built):
    from safetensors.numpy import load_file

    _, W, original, _, out = built
    dfly = load_file(str(out / "model-dfly2.safetensors"))
    vision = load_file(str(out / "model-vision.safetensors"))
    source = {**load_file(str(original / "model-00001.safetensors"))}
    assert set(dfly) == {k for k in W if k.startswith("model.dfly_layers.")}
    for name, value in {**dfly, **vision}.items():
        np.testing.assert_array_equal(value, source[name])
        assert value.dtype == source[name].dtype


def test_assembled_dfly_greedy_equals_ordinary_greedy(built):
    from mlx2.adapters import hy_v4_flash as A

    _, _, _, _, out = built
    model, drafter, readers = A.load_model(out, A.inspect_artifact(out), keep_dfly=True,
                                           engram_table_format="iq4_nl", vision=False)
    try:
        assert drafter is not None
        prompt = TOKENS[:14]
        cache = model.make_cache()
        want = [int(mx.argmax(model(mx.array([prompt]), cache)[0, -1]).item())]
        for _ in range(15):
            want.append(int(mx.argmax(model(mx.array([[want[-1]]]), cache)[0, -1]).item()))
        tokens, receipt = H.dfly_greedy_generate(model, drafter, prompt, max_tokens=16, num_draft=3)
        assert tokens == want
        assert receipt["verify_rounds"] >= 1 and receipt["qualified"] is False
    finally:
        for r in readers.values():
            r.close()


def test_assembler_refuses_nonempty_output_and_a_pack_that_already_has_dfly(built, tmp_path):
    _, _, original, pack_dir, out = built
    with pytest.raises(SystemExit, match="not empty"):
        _assembler().main([str(pack_dir), str(original), str(out)])
    with pytest.raises(SystemExit, match="already carries"):
        _assembler().main([str(out), str(original), str(tmp_path / "again")])


def test_a_failed_assembly_leaves_no_partial_artifact(built, tmp_path, monkeypatch):
    _, _, original, pack_dir, _ = built
    module = _assembler()

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(module, "copy_tensors", boom)
    out = tmp_path / "partial"
    with pytest.raises(OSError, match="disk full"):
        module.main([str(pack_dir), str(original), str(out)])
    assert not out.exists()
