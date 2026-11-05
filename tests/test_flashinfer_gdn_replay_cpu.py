"""Synthetic CPU GDN fixtures; no model loaded; base 96dfb0325c2a81098ed5fb02085ce3800f0963e0."""

import builtins
import hashlib
import shutil
import subprocess
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from mlx2.adapters.qwen38_replay_fold import output_only_replay_candidate
from mlx2.experimental.flashinfer_gdn_replay import (
    CPU_LAW,
    NATIVE_LAW,
    OutputOnlyReplay,
    ReplayBinding,
    build_native_candidate,
    metal_source,
    state_element_offset,
)

BINDING = ReplayBinding(
    "synthetic-gdn@" + hashlib.sha256(b"gdn-fixture-v1").hexdigest(),
    "96dfb0325c2a81098ed5fb02085ce3800f0963e0",
    "fixture-v1",
)


def fixture():
    rng = np.random.default_rng(671)
    state = rng.normal(size=(2, 6, 4, 3, 4)).astype(np.float32) * 0.1
    q = rng.normal(size=(2, 4, 5, 2, 4)).astype(np.float32) * 0.1
    k = rng.normal(size=q.shape).astype(np.float32) * 0.1
    v = rng.normal(size=(2, 4, 5, 4, 3)).astype(np.float32)
    g = rng.uniform(0.6, 1, size=(2, 4, 5, 4)).astype(np.float32)
    beta = rng.uniform(0, 1, size=g.shape).astype(np.float32)
    slots = np.array([4, -1, 1, 3], dtype=np.int64)
    lengths = np.array([5, 0, 3, 5], dtype=np.int64)
    return state, (q, k, v, g, beta, slots, lengths)


def independent_recurrence(state, args):
    """Full per-token states are test-only and independent of the replay tape."""
    q, k, v, g, beta, slots, lengths = args
    state = state.copy()
    history = [state.copy()]
    outputs = np.zeros_like(v)
    for token in range(q.shape[2]):
        for layer in range(len(state)):
            for row, slot in enumerate(slots):
                if slot < 0 or token >= lengths[row]:
                    continue
                for head in range(state.shape[2]):
                    kh = head // (state.shape[2] // k.shape[3])
                    for vr in range(state.shape[3]):
                        prediction = np.float32(0)
                        for channel in range(state.shape[4]):
                            state[layer, slot, head, vr, channel] *= g[
                                layer, row, token, head
                            ]
                            prediction += np.float32(
                                state[layer, slot, head, vr, channel]
                                * k[layer, row, token, kh, channel]
                            )
                        correction = np.float32(
                            np.float32(v[layer, row, token, head, vr] - prediction)
                            * beta[layer, row, token, head]
                        )
                        readout = np.float32(0)
                        for channel in range(state.shape[4]):
                            state[layer, slot, head, vr, channel] += np.float32(
                                k[layer, row, token, kh, channel] * correction
                            )
                            readout += np.float32(
                                state[layer, slot, head, vr, channel]
                                * q[layer, row, token, kh, channel]
                            )
                        outputs[layer, row, token, head, vr] = readout
        history.append(state.copy())
    return outputs, history


@pytest.mark.parametrize("accepted_count", [0, 1, 3, 5])
def test_output_only_verify_and_prefix_fold_match_independent_oracle(accepted_count):
    state, args = fixture()
    expected, history = independent_recurrence(state, args)
    pool = OutputOnlyReplay(state, BINDING, enabled=True)
    tx = pool.verify(*args, binding=BINDING, epoch=0)
    np.testing.assert_array_equal(pool.state(), state)
    np.testing.assert_array_equal(tx.outputs, expected)
    assert not tx.outputs.flags.writeable
    assert tx.receipt()["final_state_output"] is False
    assert tx.receipt()["law"] == CPU_LAW
    assert tx.receipt()["observed_used"] and not tx.receipt()["native_observed_used"]
    accepted = np.minimum(args[-1], accepted_count)
    result = tx.commit(accepted, binding=BINDING, epoch=0)
    reference = state.copy()
    for row, slot in enumerate(args[-2]):
        if slot >= 0:
            reference[:, slot] = history[int(accepted[row])][:, slot]
    np.testing.assert_array_equal(result, reference)
    np.testing.assert_array_equal(result[:, [0, 2, 5]], state[:, [0, 2, 5]])
    assert pool.epoch == 1 and tx.closed


def test_row_layout_changes_addressing_without_changing_arithmetic():
    state, args = fixture()
    first = OutputOnlyReplay(state, BINDING, enabled=True, layout="value_key")
    second = OutputOnlyReplay(state, BINDING, enabled=True, layout="key_value")
    tx1 = first.verify(*args, binding=BINDING, epoch=0)
    tx2 = second.verify(*args, binding=BINDING, epoch=0)
    np.testing.assert_array_equal(tx1.outputs, tx2.outputs)
    accepted = np.array([4, 0, 2, 5], dtype=np.int64)
    np.testing.assert_array_equal(
        tx1.commit(accepted, binding=BINDING, epoch=0),
        tx2.commit(accepted, binding=BINDING, epoch=0),
    )


def test_cancel_stale_epoch_binding_and_overlapping_ownership():
    state, args = fixture()
    pool = OutputOnlyReplay(state, BINDING, enabled=True)
    tx = pool.verify(*args, binding=BINDING, epoch=0)
    with pytest.raises(ValueError, match="still owned"):
        pool.verify(*args, binding=BINDING, epoch=0)
    with pytest.raises(ValueError, match="stale"):
        tx.commit(
            args[-1], binding=replace(BINDING, contract_revision="changed"), epoch=0
        )
    pool.invalidate()
    with pytest.raises(ValueError, match="stale"):
        tx.commit(args[-1], binding=BINDING, epoch=1)
    tx.cancel()
    np.testing.assert_array_equal(pool.state(), state)
    newer = pool.verify(*args, binding=BINDING, epoch=1)
    tx.cancel()  # an old handle cannot cancel the replacement transaction
    assert pool._pending is newer
    newer.cancel()


def test_tape_owns_inputs_and_state_owns_checkpoint():
    state, args = fixture()
    expected, history = independent_recurrence(state, args)
    pool = OutputOnlyReplay(state, BINDING, enabled=True)
    frozen = state.copy()
    state[:] = 111
    tx = pool.verify(*args, binding=BINDING, epoch=0)
    q, k, v, g, beta, slots, lengths = args
    for array in (q, k, v, g, beta):
        array[:] = 0
    slots[:] = -1
    lengths[:] = 0
    np.testing.assert_array_equal(tx.outputs, expected)
    result = tx.commit(np.array([5, 0, 3, 5], dtype=np.int64), binding=BINDING, epoch=0)
    np.testing.assert_array_equal(result, history[5])
    np.testing.assert_array_equal(result[:, [0, 2, 5]], frozen[:, [0, 2, 5]])


def test_validate_all_accept_lengths_before_any_slot_publication():
    state, args = fixture()
    pool = OutputOnlyReplay(state, BINDING, enabled=True)
    tx = pool.verify(*args, binding=BINDING, epoch=0)
    with pytest.raises(ValueError, match="accepted"):
        tx.commit(np.array([5, 0, 4, 1], dtype=np.int64), binding=BINDING, epoch=0)
    np.testing.assert_array_equal(pool.state(), state)
    assert not tx.closed
    tx.cancel()


def test_slots_lengths_bounds_alignment_and_tape_budget_refuse():
    state, args = fixture()
    for indices in ([1, -1, 1, 3], [6, -1, 1, 3], [-2, -1, 1, 3]):
        bad = (*args[:-2], np.array(indices, dtype=np.int64), args[-1])
        pool = OutputOnlyReplay(state, BINDING, enabled=True)
        with pytest.raises(ValueError):
            pool.verify(*bad, binding=BINDING, epoch=0)
    pool = OutputOnlyReplay(state, BINDING, enabled=True, max_tape_bytes=16)
    with pytest.raises(ValueError, match="tape"):
        pool.verify(*args, binding=BINDING, epoch=0)
    assert pool._pending is None
    with pytest.raises(ValueError, match="epoch"):
        pool.verify(*args, binding=BINDING, epoch=True)
    with pytest.raises(TypeError, match="ReplayBinding"):
        OutputOnlyReplay(state, "wrong", enabled=True)


def test_owner_serializes_concurrent_verifies_and_refuses_second(monkeypatch):
    import mlx2.experimental.flashinfer_gdn_replay as module

    state, args = fixture()
    pool = OutputOnlyReplay(state, BINDING, enabled=True)
    entered, release = threading.Event(), threading.Event()
    original = module._step

    def blocked(*a):
        entered.set()
        assert release.wait(3)
        return original(*a)

    monkeypatch.setattr(module, "_step", blocked)
    with ThreadPoolExecutor(max_workers=2) as worker:
        first = worker.submit(pool.verify, *args, binding=BINDING, epoch=0)
        assert entered.wait(3)
        second = worker.submit(pool.verify, *args, binding=BINDING, epoch=0)
        assert not second.done()
        release.set()
        tx = first.result(timeout=3)
        with pytest.raises(ValueError, match="still owned"):
            second.result(timeout=3)
        tx.cancel()


def test_64bit_state_address_and_layout_boundary():
    shape = (48, 4096, 48, 128, 128)
    value = state_element_offset(shape, 47, 4095, 47, 127, 127)
    assert value > 2**31 and value == int(np.prod(shape, dtype=object)) - 1
    assert state_element_offset(
        shape, 0, 1, 0, 7, 3, layout="key_value"
    ) != state_element_offset(shape, 0, 1, 0, 7, 3)
    with pytest.raises(ValueError, match="64-bit"):
        state_element_offset((2**60, 2, 2, 2, 2), 2**60 - 1, 1, 1, 1, 1)


def test_adapter_optin_import_and_native_refusal_do_not_import_device(monkeypatch):
    imported = []
    original = builtins.__import__

    def guarded(name, *a, **kw):
        if name.split(".")[0] in {"mlx", "torch", "cutlass"}:
            imported.append(name)
            raise AssertionError("device module import")
        return original(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", guarded)
    state, args = fixture()
    pool = output_only_replay_candidate(
        state,
        checkpoint=BINDING.checkpoint,
        source_revision=BINDING.source_revision,
        enabled=True,
    )
    tx = pool.verify(*args, binding=pool.binding, epoch=0)
    tx.cancel()
    with pytest.raises(ValueError, match="enabled=True"):
        build_native_candidate(None, *args, binding=BINDING, epoch=0)
    assert not imported


def test_native_host_builds_compact_successor_and_enforces_epoch(monkeypatch):
    calls = []

    class FakeArray:
        def __init__(self, data, dtype=None):
            a = np.asarray(data, dtype=dtype)
            self.shape, self.dtype = a.shape, a.dtype

    fake = types.ModuleType("mlx.core")
    fake.array = FakeArray
    fake.float32 = np.dtype("float32")
    fake.int64 = np.dtype("int64")
    fake.uint64 = np.dtype("uint64")
    fake.contiguous = lambda x: x

    def make_kernel(**spec):
        def launch(**args):
            calls.append((spec, args))
            return [
                FakeArray(np.zeros(shape, dtype=dtype))
                for shape, dtype in zip(args["output_shapes"], args["output_dtypes"])
            ]

        return launch

    fake.fast = types.SimpleNamespace(metal_kernel=make_kernel)
    pkg = types.ModuleType("mlx")
    pkg.__path__ = []
    pkg.core = fake
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", fake)
    state, args = fixture()
    # Native candidate needs four-channel state alignment; CPU oracle is generic.
    state = np.zeros((2, 1000, 4, 4, 4), dtype=np.float32)
    q, k, _, g, beta, slots, lengths = args
    values = np.zeros((2, 4, 5, 4, 4), dtype=np.float32)
    plan = build_native_candidate(
        FakeArray(state),
        *(FakeArray(x) for x in (q, k, values, g, beta)),
        slots,
        lengths,
        binding=BINDING,
        epoch=8,
        enabled=True,
    )
    assert calls[0][1]["grid"] == (2 * 4 * 4 * 4, 1, 1)
    assert calls[0][0]["compile_options"] == {"math_mode": "safe"}
    assert plan.receipt()["contract_revision"] == BINDING.contract_revision
    with pytest.raises(ValueError, match="stale"):
        plan.fold(
            np.array([5, 0, 2, 0], dtype=np.int64),
            current_binding=BINDING,
            current_epoch=9,
        )
    assert len(calls) == 1
    staged = plan.fold(
        np.array([5, 0, 2, 0], dtype=np.int64), current_binding=BINDING, current_epoch=8
    )
    assert staged.states.shape == (2, 4, 4, 4, 4)
    assert calls[1][1]["grid"] == (2 * 4 * 4 * 4, 1, 1)
    np.testing.assert_array_equal(staged.publish_mask, [True, False, True, False])
    assert staged.law == NATIVE_LAW and not plan.receipt()["native_observed_used"]
    assert "ulong" in metal_source() and "contract(off)" in metal_source()


def test_extracted_mlx_style_device_pointer_wrappers_compile_offline(tmp_path):
    """Compile only: neither MLX import nor any kernel dispatch is involved."""
    if shutil.which("xcrun") is None:
        pytest.skip("offline Apple Metal compiler unavailable")
    source = metal_source()
    header = source.split("// HEADER_BEGIN\n")[1].split("// HEADER_END")[0]
    text = (
        "#include <metal_stdlib>\nusing namespace metal;\n#pragma clang fp contract(off)\n"
        + header
    )
    for tag, types_and_names in (
        (
            "VERIFY",
            [
                ("float", n)
                for n in ["checkpoint", "query", "keys", "values", "decay", "beta"]
            ]
            + [
                ("long", "slots"),
                ("long", "lengths"),
                ("ulong", "dims"),
                ("out", "outputs"),
                ("out", "corrections"),
            ],
        ),
        (
            "FOLD",
            [("float", n) for n in ["checkpoint", "keys", "corrections", "decay"]]
            + [
                ("long", "slots"),
                ("long", "accepted"),
                ("ulong", "dims"),
                ("out", "successor"),
            ],
        ),
    ):
        body = source.split(f"// {tag}_BODY_BEGIN\n")[1].split(f"// {tag}_BODY_END")[0]
        args = []
        for index, (dtype, name) in enumerate(types_and_names):
            pointer = (
                f"device float* {name}"
                if dtype == "out"
                else f"device const {dtype}* {name}"
            )
            args.append(pointer + f" [[buffer({index})]]")
        args.append("uint3 thread_position_in_grid [[thread_position_in_grid]]")
        text += (
            f"\nkernel void scaffold_{tag.lower()}("
            + ",".join(args)
            + ") {\n"
            + body
            + "\n}\n"
        )
    path = tmp_path / "gdn-mlx-scaffold.metal"
    path.write_text(text)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.2",
            "-fno-fast-math",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "gdn.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cpu_array_protocol_is_never_invoked():
    class DeviceLike:
        def __array__(self, *_args, **_kwargs):
            raise AssertionError("device conversion attempted")

    state, args = fixture()
    with pytest.raises(ValueError, match="CPU NumPy"):
        OutputOnlyReplay(DeviceLike(), BINDING, enabled=True)
    pool = OutputOnlyReplay(state, BINDING, enabled=True)
    with pytest.raises(ValueError, match="CPU NumPy"):
        pool.verify(DeviceLike(), *args[1:], binding=BINDING, epoch=0)
    assert pool._pending is None


def test_null_padded_rows_do_not_read_checkpoint_or_leave_tape_values():
    state, args = fixture()
    q, k, v, g, beta, slots, lengths = args
    lengths[:] = [0, 5, 1, 2]  # a null slot is inert even with a positive valid length
    pool = OutputOnlyReplay(state, BINDING, enabled=True)
    tx = pool.verify(q, k, v, g, beta, slots, lengths, binding=BINDING, epoch=0)
    assert np.count_nonzero(tx.outputs[:, 0]) == 0
    assert np.count_nonzero(tx.outputs[:, 1]) == 0
    assert np.count_nonzero(tx.outputs[:, 2, 1:]) == 0
    assert np.count_nonzero(tx._corrections[:, 1]) == 0
    result = tx.commit(lengths, binding=BINDING, epoch=0)
    np.testing.assert_array_equal(result[:, 4], state[:, 4])
