import pytest

from test_apc_hits_hybrid_gdn_self_mtp import host  # noqa: F401 - fixture

from mlx2.adapters import mlx_vlm_pin
from mlx2.adapters.mlx_vlm_pin import (
    MLX_VLM_REVISION,
    mlx_vlm_runtime,
    require_pinned_mlx_vlm,
)


def _runtime(revision, version="0.7.3"):
    return {"version": version, "source": "index", "editable": False, "revision": revision}


def test_pinned_revision_is_accepted():
    runtime = _runtime(MLX_VLM_REVISION)
    assert require_pinned_mlx_vlm(runtime) is runtime


@pytest.mark.parametrize("revision", [None, "653f1f13e238abb313fd45071bbd04b3de414635"])
def test_other_or_unknown_revision_fails_closed(revision):
    with pytest.raises(RuntimeError, match="not the pinned revision"):
        require_pinned_mlx_vlm(_runtime(revision, version="0.6.17"))


def test_missing_mlx_vlm_fails_closed(monkeypatch):
    monkeypatch.setattr(mlx_vlm_pin, "mlx_vlm_runtime", lambda: None)
    with pytest.raises(RuntimeError, match="require the optional mlx-vlm runtime"):
        require_pinned_mlx_vlm()


def test_runtime_identity_does_not_bind_mlx_vlm():
    # Text routes never import mlx-vlm; installing the multimodal extra must
    # not invalidate their qualification receipts.
    from mlx2.serving import runtime_identity

    assert "mlx_vlm" not in runtime_identity()


def _settings(host, *, runtime):
    from mlx2 import serving
    from test_apc_hits_hybrid_gdn_self_mtp import make_adapter, tiny_qwen38_mtp

    model, vocab = tiny_qwen38_mtp()
    base = make_adapter(model, vocab)

    class Adapter(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if runtime is not None:
                self.mlx_vlm_runtime = runtime

    engine = serving.ServingEngine(
        "tiny", adapter_factory=Adapter, qualification_mode=True,
        max_lanes=1, prefill_step=16,
    )
    try:
        assert engine.ready.wait(60), engine.error
        return engine.status()["settings"]
    finally:
        engine.close()


def test_only_mlx_vlm_routes_bind_the_revision_in_settings(host):
    runtime = _runtime(MLX_VLM_REVISION)
    assert _settings(host, runtime=runtime)["mlx_vlm"] == runtime
    assert "mlx_vlm" not in _settings(host, runtime=None)


def test_pyproject_pin_matches_adapter_pin():
    from pathlib import Path

    text = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert f"mlx-vlm.git@{MLX_VLM_REVISION}" in text
