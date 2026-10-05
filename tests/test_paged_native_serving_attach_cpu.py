"""Host-only admission checks for explicit native serving attachment."""

import sys
from pathlib import Path
from threading import RLock
from types import ModuleType, SimpleNamespace

import pytest

from mlx2 import serving


def _job(**request):
    return SimpleNamespace(
        uid=3, preempted=False, cached_tokens=0,
        request={"paged_native_qwen3": True,
                 "skip_writing_prefix_cache": True, **request},
        cancelled=SimpleNamespace(is_set=lambda: False),
    )


def test_native_attach_refuses_cache_writing_or_unmatched_warm_state_before_factory(monkeypatch):
    native = ModuleType("_paged_kv_native")
    native.__file__ = "/tmp/fake-paged-native.so"
    monkeypatch.setitem(sys.modules, "_paged_kv_native", native)
    adapter = SimpleNamespace(create_native_paged_qwen3_request=lambda **_:
                              pytest.fail("factory ran before APC gate"))
    for job in (_job(skip_writing_prefix_cache=False), _job()):
        if job.request["skip_writing_prefix_cache"]:
            job.cached_tokens = 2
        with pytest.raises(ValueError, match="explicit no-write|nonempty token prompt"):
            serving.install_explicit_native_qwen3_request(
                object(), adapter, job, prompt_tokens=[1, 2], maximum=2,
                lifecycle_lock=RLock(), price_path="p", manifest_path="m",
                mlx_wheel_path="w")


def test_native_attach_uses_live_identity_price_then_installs_or_retires(monkeypatch, tmp_path):
    from mlx2.runtime import paged_native_batch_lifecycle, paged_pack_price

    native = ModuleType("_paged_kv_native")
    native.__file__ = "/tmp/fake-paged-native.so"
    monkeypatch.setitem(sys.modules, "_paged_kv_native", native)
    identity_mod = ModuleType("mlx2.runtime.paged_price_identity")
    calls = []
    identity_mod.cached_live_price_identity = lambda *args, **kwargs: (
        calls.append((args, kwargs)) or {"host": "test"})
    monkeypatch.setitem(sys.modules, identity_mod.__name__, identity_mod)
    monkeypatch.setattr(paged_pack_price, "load_price", lambda *_, **kwargs:
                        SimpleNamespace(profile_id="p1", complete_request_evidence_sha256="a" * 64))
    price_path = tmp_path / "price.json"
    price_path.write_text('{"measurement_scope":"complete_request"}')
    probe = SimpleNamespace(native_probe=SimpleNamespace(published=True), reason="accepted")
    probes = []
    monkeypatch.setattr(paged_native_batch_lifecycle, "probe_queued_native_qwen3",
                        lambda *args, **kwargs: (probes.append((args, kwargs)) or probe))
    monkeypatch.setattr(paged_native_batch_lifecycle,
                        "prepare_queued_native_first_response",
                        lambda *_, **__: "prepared")
    retired = []
    owner = SimpleNamespace(
        fully_retired=True,
        close=lambda: retired.append("close"),
        reap_retired=lambda: retired.append("retired"),
        reap_quarantine=lambda: retired.append("quarantine"))
    writer = SimpleNamespace(pool=SimpleNamespace(free_count=8),
                             poisoned=False, pending_epochs=(),
                             ledger=SimpleNamespace(pending_count=0))
    candidate = SimpleNamespace(backend=SimpleNamespace(writer=writer))
    created = []
    adapter = SimpleNamespace(
        identity={"path": "/tmp/model", "fingerprint": "r1"},
        create_native_paged_qwen3_request=lambda **kwargs:
        (created.append(kwargs) or (owner, candidate)))
    batch = SimpleNamespace(install_native_queued=lambda *_, **__:
                            {"selected": True, "reason": "native_installed"})
    result = serving.install_explicit_native_qwen3_request(
        batch, adapter, _job(), prompt_tokens=[1, 2], maximum=2,
        lifecycle_lock=RLock(), price_path=price_path, manifest_path="manifest",
        mlx_wheel_path="wheel")
    assert result["selected"]
    assert not retired
    assert calls[0][1]["adapter_artifact_root"] == Path("/tmp/model").resolve()
    warm = _job()
    warm.cached_tokens = 2
    apc_cache = object()
    result = serving.install_explicit_native_qwen3_request(
        batch, adapter, warm, prompt_tokens=[1, 2, 3], cached_tokens=2,
        apc_cache=apc_cache, maximum=2, lifecycle_lock=RLock(),
        price_path=price_path, manifest_path="manifest", mlx_wheel_path="wheel")
    assert result["selected"]
    assert created[-1]["cached_tokens"] == 2
    assert created[-1]["apc_cache"] is apc_cache
    assert probes[-1][0][2].proposed_rows == 1
    assert probes[-1][1]["context_tokens"] == (3,)
    batch.install_native_queued = lambda *_, **__: {
        "selected": False, "reason": "cancelled"}
    with pytest.raises(ValueError, match="handoff refused: cancelled"):
        serving.install_explicit_native_qwen3_request(
            batch, adapter, _job(), prompt_tokens=[1, 2], maximum=2,
            lifecycle_lock=RLock(), price_path=price_path, manifest_path="manifest",
            mlx_wheel_path="wheel")
    assert retired == ["close", "retired", "quarantine"]


def test_explicit_native_identity_is_preverified_only_for_configured_capability(
    monkeypatch, tmp_path,
):
    from mlx2.runtime import paged_price_identity

    calls = []
    monkeypatch.setattr(paged_price_identity, "cached_live_price_identity",
                        lambda *args, **kwargs: (calls.append((args, kwargs)) or
                                                {"source_commit": "test"}))
    native = ModuleType("_paged_kv_native")
    native.__file__ = str(tmp_path / "native.so")
    monkeypatch.setitem(sys.modules, "_paged_kv_native", native)
    adapter = SimpleNamespace(
        identity={"path": str(tmp_path)},
        create_native_paged_qwen3_request=lambda **_: None)
    for key in ("MLX2_NATIVE_PAGED_PRICE", "MLX2_NATIVE_PAGED_MANIFEST",
                "MLX2_NATIVE_PAGED_MLX_WHEEL"):
        monkeypatch.delenv(key, raising=False)
    assert serving.preverify_explicit_native_qwen3_price_identity(adapter) is None
    assert not calls
    monkeypatch.setenv("MLX2_NATIVE_PAGED_PRICE", str(tmp_path / "price.json"))
    monkeypatch.setenv("MLX2_NATIVE_PAGED_MANIFEST", str(tmp_path / "artifact.json"))
    monkeypatch.setenv("MLX2_NATIVE_PAGED_MLX_WHEEL", str(tmp_path / "mlx.whl"))
    assert serving.preverify_explicit_native_qwen3_price_identity(adapter) == {
        "source_commit": "test"}
    assert len(calls) == 1
    assert calls[0][1]["adapter_artifact_root"] == tmp_path.resolve()


def test_native_attach_retains_owner_if_retirement_raises(monkeypatch, tmp_path):
    from mlx2.runtime import paged_native_batch_lifecycle, paged_pack_price

    native = ModuleType("_paged_kv_native")
    native.__file__ = "/tmp/fake-paged-native.so"
    monkeypatch.setitem(sys.modules, "_paged_kv_native", native)
    identity_mod = ModuleType("mlx2.runtime.paged_price_identity")
    identity_mod.cached_live_price_identity = lambda *_, **__: {"host": "test"}
    monkeypatch.setitem(sys.modules, identity_mod.__name__, identity_mod)
    monkeypatch.setattr(paged_pack_price, "load_price", lambda *_, **__:
                        SimpleNamespace(profile_id="p1", complete_request_evidence_sha256="a" * 64))
    price_path = tmp_path / "price.json"
    price_path.write_text('{"measurement_scope":"complete_request"}')
    monkeypatch.setattr(paged_native_batch_lifecycle, "probe_queued_native_qwen3",
                        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("probe failed")))

    class PendingOwner:
        fully_retired = False

        def close(self):
            raise RuntimeError("terminal pending")

        def reap_retired(self):
            pass

        def reap_quarantine(self):
            pass

    owner = PendingOwner()
    adapter = SimpleNamespace(
        identity={"path": "/tmp/model", "fingerprint": "r1"},
        create_native_paged_qwen3_request=lambda **_: (
            owner, SimpleNamespace(backend=SimpleNamespace(
                writer=SimpleNamespace(pool=SimpleNamespace(free_count=8))))))
    with pytest.raises(RuntimeError, match="terminal pending"):
        serving.install_explicit_native_qwen3_request(
            object(), adapter, _job(), prompt_tokens=[1, 2], maximum=2,
            lifecycle_lock=RLock(), price_path=price_path, manifest_path="manifest",
            mlx_wheel_path="wheel")
    assert any(item[0] is owner for item in serving._NATIVE_ADMISSION_ORPHANS)
    serving._NATIVE_ADMISSION_ORPHANS[:] = [
        item for item in serving._NATIVE_ADMISSION_ORPHANS if item[0] is not owner]
