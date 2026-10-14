"""An undecidable MTP-head RMSNorm convention refuses the whole load, loudly.

Pierre's decision (2026-10-14): when a head norm's (+1) convention cannot be
decided, the model must not load at all -- no "serve the trunk with MTP
off" fallback -- and the refusal must be loud: an ERROR/CRITICAL log line
naming the checkpoint, the tensors, their statistics and the remediation;
the engine never becomes ready, requests fail with that text, and server
startup exits non-zero with it.

The tensor modules are stubbed (no production weights), but each stub runs
the real ``TextModel.sanitize`` of its family on a synthetic undecidable
head, so the refusal travels the real adapter, engine and server paths.
"""

import json
import logging
import socket
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_norm_convention import _checkpoint, _qwen4_checkpoint, _vec  # noqa: E402
from test_qwen36_35b_port import _chat_tokenizer, make_artifact  # noqa: E402

from mlx2.adapters.norm_repair import NormConventionError  # noqa: E402
from mlx2.runtime.models import qwen4_exp, qwen38_27b  # noqa: E402

UNDECIDABLE = "language_model.mtp.layers.0.self_attn.q_norm.weight"


RAW_LAYOUT = False  # set per test: a raw (HF) trunk must refuse the same way


def _undecidable_qwen38_head():
    weights = _checkpoint(mx.bfloat16, raw_head=(), raw_layout=RAW_LAYOUT)
    # gap +0.40 to the 1.46 trunk family: neither shifted nor raw.
    weights[UNDECIDABLE] = _vec(1.06, mx.bfloat16)
    return weights


class _Qwen36Model:
    """Tensor-module stand-in whose sanitize is the real Qwen3.5-family one."""

    apc_v2_layout = "stub-layout"

    def __init__(self, *_args, **_kwargs):
        self.language_model = SimpleNamespace(
            mtp=object(), args=SimpleNamespace(tie_word_embeddings=False)
        )
        self.model = SimpleNamespace(set_eager_dispatch=lambda *_a: None)

    def shard_prune(self, weights):
        return weights

    def sanitize(self, _weights):
        return qwen38_27b.TextModel.sanitize(self.language_model, _undecidable_qwen38_head())

    def load_weights(self, *_a, **_k):
        raise AssertionError("an undecidable head must refuse before the strict load")


def _patch_qwen36(monkeypatch, tmp_path):
    from mlx2.adapters import qwen36_35b
    from mlx2.runtime import ubc_evict
    from mlx2.runtime.models import qwen36_35b as tensors

    _chat_tokenizer(tmp_path)
    make_artifact(tmp_path, mtp=True)
    monkeypatch.setattr(qwen36_35b, "configure_environment", lambda *_a: {})
    monkeypatch.setattr(tensors, "Model", _Qwen36Model)
    monkeypatch.setattr(ubc_evict, "load_shards_evicting", lambda *_a, **_k: {})
    return qwen36_35b.Qwen3635BA3BAdapter


def _assert_loud_text(text, path=None):
    assert "NormConventionError" in text or "cannot decide the RMSNorm" in text
    assert UNDECIDABLE in text  # the tensor
    assert "gap +0.4" in text  # its statistics
    assert "KNOWN_UNSHIFTED_NORMS" in text  # the remediation
    if path is not None:
        assert str(path) in text


@pytest.fixture(params=[False, True], ids=["converted_trunk", "raw_trunk"])
def trunk_layout(request, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "RAW_LAYOUT", request.param)
    return request.param


def test_qwen36_adapter_load_refuses(monkeypatch, tmp_path, trunk_layout):
    adapter_class = _patch_qwen36(monkeypatch, tmp_path)
    with pytest.raises(NormConventionError) as error:
        adapter_class(str(tmp_path))
    _assert_loud_text(str(error.value))


def test_flash_next_adapter_load_refuses(monkeypatch, tmp_path):
    import mlx.nn as nn

    from mlx2.adapters import flash_next
    from mlx2.runtime import ubc_evict
    from mlx2.runtime.models import import_env, qwen4_ple_nvme

    monkeypatch.delenv("MLX_QWEN4_NORM_CONVENTION", raising=False)
    monkeypatch.delenv("MLX_QWEN4_NORM_CONVENTION_UNCHECKED", raising=False)

    class _FlashModel(_Qwen36Model):
        def __init__(self, *_a, **_k):
            super().__init__()
            self.language_model = SimpleNamespace(
                args=SimpleNamespace(tie_word_embeddings=False),
                NORM_CONVENTION_HINT=qwen4_exp.TextModel.NORM_CONVENTION_HINT,
            )

        def sanitize(self, _weights):
            return qwen4_exp.TextModel.sanitize(
                self.language_model,
                _qwen4_checkpoint(mx.bfloat16, raw_head=(), head_qk=3.68),
            )

    _chat_tokenizer(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen4_exp", "quantization": {"group_size": 64, "bits": 4},
        "text_config": {"max_position_embeddings": 4096},
    }))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model.safetensors"}})
    )
    (tmp_path / "model.safetensors").write_bytes(b"metadata-only")
    (tmp_path / "ple_rows.bin").write_bytes(b"test-sidecar")
    monkeypatch.setattr(flash_next, "configure_environment", lambda *_a, **_k: {})
    monkeypatch.setattr(import_env, "assert_profile_applied", lambda *_a, **_k: None)
    monkeypatch.setattr(flash_next, "artifact_identity", lambda path: {"path": str(path)})
    monkeypatch.setattr(qwen4_exp, "Model", _FlashModel)
    monkeypatch.setattr(qwen4_exp.ModelArgs, "from_dict", classmethod(lambda cls, config: config))
    monkeypatch.setattr(ubc_evict, "load_shards_evicting", lambda *_a, **_k: {})
    monkeypatch.setattr(ubc_evict, "ubc_evict_paths", lambda *_a, **_k: None)
    monkeypatch.setattr(qwen4_ple_nvme, "verify_sidecar_against_artifact", lambda *_a: {})
    monkeypatch.setattr(qwen4_ple_nvme, "verify_sidecar_content", lambda *_a: "converted")
    monkeypatch.setattr(nn, "quantize", lambda *_a, **_k: None)
    with pytest.raises(NormConventionError, match="too far above") as error:
        flash_next.FlashNextAdapter(str(tmp_path))
    assert "mtp.layers.0.self_attn.q_norm.weight" in str(error.value)
    assert "MLX_QWEN4_NORM_CONVENTION" in str(error.value)


def _engine(adapter_class, model_path):
    from mlx2 import serving

    return serving.ServingEngine(
        str(model_path), adapter_factory=adapter_class, qualification_mode=True, mtp=False
    )


def test_engine_never_becomes_ready_and_reports_loudly(monkeypatch, tmp_path, caplog, trunk_layout):
    from mlx2 import serving

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    adapter_class = _patch_qwen36(monkeypatch, tmp_path)
    caplog.set_level(logging.ERROR)
    engine = _engine(adapter_class, tmp_path)
    try:
        engine.thread.join(30)
        assert not engine.thread.is_alive()
        assert not engine.ready.is_set()
        _assert_loud_text(engine.error, tmp_path)
        assert engine.status()["error"] == engine.error
        with pytest.raises(RuntimeError) as refused:
            engine.submit({"tokens": [1, 2, 3], "max_tokens": 1})
        _assert_loud_text(str(refused.value), tmp_path)
    finally:
        engine.close()
    loud = [r for r in caplog.records if r.levelno >= logging.ERROR]
    text = "\n".join(r.getMessage() for r in loud)
    _assert_loud_text(text, tmp_path)
    assert not any(
        UNDECIDABLE in r.getMessage() and r.levelno < logging.ERROR for r in caplog.records
    ), "the refusal must never be downgraded below ERROR"


def test_server_startup_exits_non_zero_with_the_refusal(monkeypatch, tmp_path, capsys, trunk_layout):
    from mlx2 import exit_trace, server, serving
    from mlx2.adapters import registry

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    adapter_class = _patch_qwen36(monkeypatch, tmp_path)
    monkeypatch.setattr(exit_trace.ExitTrace, "install", lambda self, fault_log=None: self)
    monkeypatch.setattr(
        registry, "inspect_model",
        lambda path: types.SimpleNamespace(artifact_fingerprint="sha256:fake"),
    )
    monkeypatch.setattr(
        server, "resolve_route_selection",
        lambda args, policy, resolution: types.SimpleNamespace(
            native_mtp=False, route="ordinary", source="engine_argument"
        ),
    )
    monkeypatch.setattr(server, "resolve_execution_policy_defaults", lambda policy, *a, **k: policy)
    monkeypatch.setattr(registry, "resolve_adapter", lambda *_a, **_k: adapter_class)
    monkeypatch.setattr(
        sys, "argv",
        ["mlx2.server", "--model", str(tmp_path), "--port", "0", "--qualification-mode"],
    )
    with pytest.raises(SystemExit) as exited:
        server.main()
    code = exited.value.code
    assert code not in (0, None)
    _assert_loud_text(str(code), tmp_path)
