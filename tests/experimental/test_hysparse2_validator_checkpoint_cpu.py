"""Exercise the validator checkpoint gate with all MLX execution forced to CPU."""

import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import optimizers

from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.model import Model
from mlx2.experimental.hysparse2.train import file_hash, save_checkpoint
from mlx2.experimental.hysparse2.validate import run


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = mx.default_device()
    setter = mx.set_default_device
    setter(mx.cpu)
    # The public validator is GPU-only. This regression tests its source gate,
    # then real tiny-model math on CPU, without entering gpu_guard or Metal.
    monkeypatch.setattr(mx, "set_default_device", lambda requested: setter(mx.cpu))
    for name in (
        "set_memory_limit",
        "set_cache_limit",
        "reset_peak_memory",
        "clear_cache",
    ):
        monkeypatch.setattr(mx, name, lambda *args: None)
    for name in ("get_active_memory", "get_peak_memory"):
        monkeypatch.setattr(mx, name, lambda: 0)
    yield
    setter(previous)


def inputs(tmp_path, mode="model"):
    model = Model(Config.smoke())
    checkpoint = save_checkpoint(
        tmp_path / "checkpoints", model, optimizers.Adam(1e-3), 0, {}, mode=mode
    )
    tokens = tmp_path / "tokens.npy"
    np.save(tokens, np.arange(40, dtype=np.uint32) % model.config.vocab_size)
    return SimpleNamespace(
        checkpoint=checkpoint,
        tokens=tokens,
        lengths=[4],
        memory_limit_gib=1,
        dtype="float32",
        output=tmp_path / "receipt.json",
    )


@pytest.mark.parametrize(
    "change", ["corrupt", "missing", "wrong_digest", "model_ple", "binding"]
)
def test_invalid_ple_checkpoint_refused_before_context_math(
    tmp_path, monkeypatch, change
):
    args = inputs(tmp_path)
    sidecar = args.checkpoint / "semantic-ple.safetensors"
    state_path = args.checkpoint / "state.json"
    state = json.loads(state_path.read_text())
    if change == "corrupt":
        sidecar.write_bytes(b"corrupt")
    elif change == "missing":
        sidecar.unlink()
    elif change == "wrong_digest":
        state["permanent_sidecar"]["sha256"] = "0" * 64
        state_path.write_text(json.dumps(state))
    elif change == "binding":
        state["capsule_binding"] = {"unbound": True}
        state_path.write_text(json.dumps(state))
    else:
        path = args.checkpoint / "model.safetensors"
        tensors = mx.load(str(path))
        tensors["semantic_ple.value.weight"] += 0.1
        mx.eval(tensors)
        mx.save_safetensors(str(path), tensors)

    def poison(*args, **kwargs):
        pytest.fail("invalid checkpoint reached context evaluation")

    monkeypatch.setattr(Model, "prefill", poison)
    with pytest.raises(ValueError, match="checkpoint"):
        run(args)
    assert not args.output.exists()
    assert mx.default_device() == mx.cpu


@pytest.mark.parametrize("mode", ["model", "full"])
def test_valid_checkpoint_records_verified_bytes_without_rewriting(tmp_path, mode):
    args = inputs(tmp_path, mode)
    names = ["model.safetensors", "state.json", "semantic-ple.safetensors"]
    before = {name: file_hash(args.checkpoint / name) for name in names}
    receipt = run(args)
    identity = receipt["checkpoint_identity"]
    assert identity["model_sha256"] == before["model.safetensors"]
    assert identity["state_sha256"] == before["state.json"]
    assert identity["semantic_ple_sha256"] == before["semantic-ple.safetensors"]
    assert identity["optimizer_state_restored"] is False
    assert identity["exact_training_resume"] is False
    assert receipt["contexts"][0]["finite_logits"] is True
    assert before == {name: file_hash(args.checkpoint / name) for name in names}
    assert mx.default_device() == mx.cpu
