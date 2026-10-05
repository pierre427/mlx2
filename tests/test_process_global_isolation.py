"""Adapters must not change process-global route selections under a live one.

Codex review (flip-isolation 2026-10-02, P1): Flash-Next set
``switch_layers._RHS_PAD_POLICY`` and ``moe_nax_gather.MODE`` at load, and
Qwen3.6 set ``moe_nax_gather.MODE``.  A second load with another value
switched the first adapter's arithmetic behind its receipt and APCv2
namespace, and a failed load left its value behind.  Both are now claimed
through ``adapters.process_globals``: a conflicting load is refused, equal
values are allowed, and a failed load rolls the globals (and the profile's
``os.environ`` edits) back.  Stubs only; no model loads.
"""

import gc
import os

import pytest

from mlx2.adapters import process_globals
from mlx2.adapters.process_globals import ProcessGlobalConflict
from mlx2.runtime.models import moe_nax_gather, switch_layers
from tests.test_qwen36_35b_port import (
    _LoadedModel,
    _chat_tokenizer,
    _flash_next_stub_load,
    make_artifact,
)

PROFILE_PROBE = "MLX2_ISOLATION_TEST_PROBE"


class _Boom(RuntimeError):
    """The stubbed tensor model failed after the globals were claimed."""


@pytest.fixture(autouse=True)
def _fresh_globals(monkeypatch):
    monkeypatch.setattr(process_globals, "_HOLDERS", {})
    monkeypatch.setattr(moe_nax_gather, "MODE", "off")
    monkeypatch.setattr(switch_layers, "_RHS_PAD_POLICY", "floor")
    monkeypatch.delenv(PROFILE_PROBE, raising=False)
    yield


def _flash_next(tmp_path, monkeypatch):
    flash_next = _flash_next_stub_load(tmp_path, monkeypatch)

    def pin(*_a, **_k):
        # Stands in for the profile's os.environ edits.
        os.environ[PROFILE_PROBE] = "pinned"
        return {}

    monkeypatch.setattr(flash_next, "configure_environment", pin)
    return flash_next


def _qwen36(tmp_path, monkeypatch):
    from mlx2.adapters import qwen36_35b
    from mlx2.runtime import ubc_evict
    from mlx2.runtime.models import import_env
    from mlx2.runtime.models import qwen36_35b as tensors

    _chat_tokenizer(tmp_path)
    make_artifact(tmp_path)
    monkeypatch.setattr(tensors, "Model", _LoadedModel)
    monkeypatch.setattr(ubc_evict, "load_shards_evicting", lambda *_a, **_k: {})
    monkeypatch.setattr(import_env, "assert_profile_applied", lambda *_a, **_k: None)
    return qwen36_35b


def _fail_tensor_model(monkeypatch, module):
    def boom(*_a, **_k):
        raise _Boom

    monkeypatch.setattr(module, "Model", boom)


# --- Flash-Next: pad policy and NAX mode -------------------------------------


def test_flash_next_refuses_a_pad_policy_a_live_adapter_does_not_run(tmp_path, monkeypatch):
    flash_next = _flash_next(tmp_path, monkeypatch)
    a = flash_next.FlashNextAdapter(str(tmp_path))
    assert switch_layers._RHS_PAD_POLICY == "adaptive"
    assert moe_nax_gather.MODE == "fused"
    with pytest.raises(ProcessGlobalConflict, match="moe_rhs_pad_policy"):
        flash_next.FlashNextAdapter(
            str(tmp_path), execution_policy={"moe_rhs_pad_policy": "floor"}
        )
    # A keeps running the arithmetic its receipt names.
    assert switch_layers._RHS_PAD_POLICY == "adaptive"
    with pytest.raises(ProcessGlobalConflict, match="moe_nax_gather"):
        flash_next.FlashNextAdapter(
            str(tmp_path), execution_policy={"moe_nax_gather": "off"}
        )
    assert moe_nax_gather.MODE == "fused"
    # Once A is closed the other value loads.
    a.close()
    flash_next.FlashNextAdapter(
        str(tmp_path), execution_policy={"moe_rhs_pad_policy": "floor"}
    )
    assert switch_layers._RHS_PAD_POLICY == "floor"


def test_flash_next_identical_values_load_side_by_side(tmp_path, monkeypatch):
    flash_next = _flash_next(tmp_path, monkeypatch)
    a = flash_next.FlashNextAdapter(str(tmp_path))
    b = flash_next.FlashNextAdapter(
        str(tmp_path),
        execution_policy={"moe_rhs_pad_policy": "adaptive", "moe_nax_gather": "fused"},
    )
    assert len(process_globals.live_selections()) == 2
    assert switch_layers._RHS_PAD_POLICY == "adaptive"
    assert moe_nax_gather.MODE == "fused"
    del a, b


def test_flash_next_failed_load_restores_globals_and_environment(tmp_path, monkeypatch):
    from mlx2.runtime.models import qwen4_exp

    flash_next = _flash_next(tmp_path, monkeypatch)
    _fail_tensor_model(monkeypatch, qwen4_exp)
    with pytest.raises(_Boom):
        flash_next.FlashNextAdapter(str(tmp_path))
    assert switch_layers._RHS_PAD_POLICY == "floor"
    assert moe_nax_gather.MODE == "off"
    assert PROFILE_PROBE not in os.environ
    assert process_globals.live_selections() == []


def test_failed_load_beside_a_live_adapter_keeps_its_state(tmp_path, monkeypatch):
    from mlx2.runtime.models import qwen4_exp

    flash_next = _flash_next(tmp_path, monkeypatch)
    a = flash_next.FlashNextAdapter(str(tmp_path))
    os.environ.pop(PROFILE_PROBE)  # A's pinned profile as the baseline
    _fail_tensor_model(monkeypatch, qwen4_exp)
    with pytest.raises(_Boom):
        flash_next.FlashNextAdapter(str(tmp_path))
    assert switch_layers._RHS_PAD_POLICY == "adaptive"
    assert moe_nax_gather.MODE == "fused"
    assert PROFILE_PROBE not in os.environ
    # A is still registered: a conflicting load is still refused.
    with pytest.raises(ProcessGlobalConflict):
        flash_next.FlashNextAdapter(
            str(tmp_path), execution_policy={"moe_rhs_pad_policy": "always"}
        )
    assert [owner for owner, _ in process_globals.live_selections()] == [
        "the Flash-Next adapter"
    ]
    del a


def test_a_refused_load_changes_nothing(tmp_path, monkeypatch):
    flash_next = _flash_next(tmp_path, monkeypatch)
    a = flash_next.FlashNextAdapter(str(tmp_path))
    os.environ.pop(PROFILE_PROBE)
    with pytest.raises(ProcessGlobalConflict):
        flash_next.FlashNextAdapter(
            str(tmp_path), execution_policy={"moe_rhs_pad_policy": "floor"}
        )
    assert PROFILE_PROBE not in os.environ
    del a


def test_an_unreachable_adapter_does_not_block_a_load(tmp_path, monkeypatch):
    flash_next = _flash_next(tmp_path, monkeypatch)
    a = flash_next.FlashNextAdapter(str(tmp_path))
    a._cycle = a  # reachable only through a cycle: needs the collector
    del a
    flash_next.FlashNextAdapter(
        str(tmp_path), execution_policy={"moe_rhs_pad_policy": "floor"}
    )
    assert switch_layers._RHS_PAD_POLICY == "floor"


# --- Qwen3.6: NAX mode, and across adapters -----------------------------------


def test_qwen36_refuses_a_nax_mode_a_live_adapter_does_not_run(tmp_path, monkeypatch):
    qwen36_35b = _qwen36(tmp_path, monkeypatch)
    a = qwen36_35b.Qwen3635BA3BAdapter(str(tmp_path))
    assert moe_nax_gather.MODE == "fused"
    with pytest.raises(ProcessGlobalConflict, match="moe_nax_gather"):
        qwen36_35b.Qwen3635BA3BAdapter(
            str(tmp_path), execution_policy={"moe_nax_gather": "off"}
        )
    assert moe_nax_gather.MODE == "fused"
    # Same value: allowed.
    b = qwen36_35b.Qwen3635BA3BAdapter(
        str(tmp_path), execution_policy={"moe_nax_gather": "fused"}
    )
    del a, b


def test_qwen36_failed_load_restores_the_nax_mode(tmp_path, monkeypatch):
    from mlx2.runtime.models import qwen36_35b as tensors

    qwen36_35b = _qwen36(tmp_path, monkeypatch)
    _fail_tensor_model(monkeypatch, tensors)
    with pytest.raises(_Boom):
        qwen36_35b.Qwen3635BA3BAdapter(str(tmp_path))
    assert moe_nax_gather.MODE == "off"
    assert process_globals.live_selections() == []


def test_flash_next_and_qwen36_share_the_nax_claim(tmp_path, monkeypatch):
    fn_root = tmp_path / "fn"
    q_root = tmp_path / "q36"
    fn_root.mkdir()
    q_root.mkdir()
    flash_next = _flash_next(fn_root, monkeypatch)
    qwen36_35b = _qwen36(q_root, monkeypatch)
    a = flash_next.FlashNextAdapter(str(fn_root))
    with pytest.raises(ProcessGlobalConflict, match="Flash-Next"):
        qwen36_35b.Qwen3635BA3BAdapter(
            str(q_root), execution_policy={"moe_nax_gather": "gather"}
        )
    assert moe_nax_gather.MODE == "fused"
    del a


def test_qwen122_refuses_to_load_under_a_conflicting_live_claim(tmp_path, monkeypatch):
    # The 122B adapter sets neither global but runs both: it claims the
    # values in force.  Under a live holder of another value it is refused.
    from tests.test_qwen_adapter_import_guard import _qwen122

    from mlx2.runtime.models import import_env

    monkeypatch.setattr(import_env, "assert_profile_applied", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mlx2.adapters.qwen35_122b.configure_environment", lambda *_a, **_k: {}
    )

    class Holder:
        pass

    holder = Holder()
    process_globals.claim(
        holder, "a live test holder", {process_globals.MOE_NAX_GATHER: ("fused", None)}
    ).commit()
    with pytest.raises(ProcessGlobalConflict, match="a live test holder"):
        _qwen122(monkeypatch, tmp_path)
    del holder


# --- Registry --------------------------------------------------------------


def test_registry_releases_on_close_and_collection():
    class Holder:
        pass

    seen = []

    def setter(value):
        seen.append(value)
        return "old"

    a = Holder()
    claim = process_globals.claim(a, "a", {"x": ("one", setter)})
    claim.commit()
    with pytest.raises(ProcessGlobalConflict):
        process_globals.claim(Holder(), "b", {"x": ("two", setter)})
    assert seen == ["one"]  # the refused claim set nothing
    process_globals.release(a)
    process_globals.claim(Holder(), "c", {"x": ("two", setter)}).rollback()
    assert seen == ["one", "two", "old"]
    b = Holder()
    process_globals.claim(b, "b", {"x": ("two", None)}).commit()
    del b
    gc.collect()
    assert process_globals.live_selections() == []


def test_pending_claim_blocks_conflicting_and_equal_claims_until_finished():
    class Holder:
        pass

    state = {"x": "old"}

    def setter(value):
        old = state["x"]
        state["x"] = value
        return old

    first, second = Holder(), Holder()
    pending = process_globals.claim(first, "first", {"x": ("one", setter)})
    assert state["x"] == "one"
    assert process_globals.live_selections() == []
    for value in ("one", "two"):
        with pytest.raises(ProcessGlobalConflict, match="pending"):
            process_globals.claim(second, "second", {"x": (value, setter)})
        assert state["x"] == "one"
    pending.rollback()
    assert state["x"] == "old"
    process_globals.claim(second, "second", {"x": ("two", setter)}).commit()
    assert state["x"] == "two"
    process_globals.release(second)
