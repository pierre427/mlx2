"""Observed verification geometry controls generated-state publication evidence."""
from types import SimpleNamespace

from mlx2.runtime.generate import GenerationBatch
from mlx2.runtime.hybrid_speculative import (
    record_self_mtp_verify_geometry,
    self_mtp_numerical_schedule,
)


def lane(requested=3, selected=3):
    return SimpleNamespace(num_draft=selected, _requested_num_draft=requested,
        numerical_schedule_changed=False, numerical_initial_geometry=None,
        numerical_last_geometry=None)


def test_actual_depth_and_width_transitions_are_monotonic_numerical_changes():
    state = lane()
    observe = lambda width, depth: record_self_mtp_verify_geometry(
        state, compute_width=width, draft_depth=depth, backend_mode="padded")
    observe(1, 3)
    assert not self_mtp_numerical_schedule(state)["numerical_schedule_changed"]
    observe(1, 3)
    assert not state.numerical_schedule_changed
    observe(1, 2)
    assert state.numerical_schedule_changed
    observe(1, 3)
    assert state.numerical_schedule_changed
    assert state.numerical_initial_geometry == ("self_mtp", 1, 4, "padded")
    assert state.numerical_last_geometry == state.numerical_initial_geometry


def test_actual_lower_depth_admission_is_not_declared_default_geometry():
    state = lane(requested=3, selected=2)
    record_self_mtp_verify_geometry(state, compute_width=4, draft_depth=2, backend_mode="padded")
    assert state.numerical_schedule_changed


def test_ordinary_handoff_records_actual_numerical_geometry_change():
    state = lane()
    record_self_mtp_verify_geometry(state, compute_width=1, draft_depth=3, backend_mode="padded")
    receipt = self_mtp_numerical_schedule(state)
    GenerationBatch._record_execution_width(receipt, 4)
    assert receipt["numerical_schedule_changed"]
    assert receipt["numerical_last_geometry"] == ("ordinary", 4, 1, "ordinary")


def test_ordinary_reference_adapter_without_receipt_keeps_running():
    GenerationBatch._record_execution_width(None, 1)
    plain = {}
    GenerationBatch._record_execution_width(plain, 2)
    assert plain == {"ordinary_compute_widths": [2], "observed_compute_widths": [2]}


def test_real_executor_depth_transition_and_ordinary_fallback_keep_schedule_evidence():
    from route_harness import tiny_qwen38_mtp

    from mlx2.runtime.generate import BatchGenerator

    model, _ = tiny_qwen38_mtp()
    batch = BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                           prefill_step_size=16,
                           self_mtp={"persistent": True, "num_draft": 2})
    uid = batch.insert([[1, 7, 3, 9, 2]], max_tokens=[20])[0]
    try:
        for _ in range(15):
            batch.next()
            state = batch._generation_batch.numerical_schedule_for_uid(uid)
            if state is not None and state["numerical_initial_geometry"] is not None:
                break
        else:
            raise AssertionError("no native verification executed")
        initial = state["numerical_initial_geometry"]
        assert initial[0:3] == ("self_mtp", 1, 3)
        assert not state["numerical_schedule_changed"]
        batch._generation_batch.set_num_draft(1)
        batch.next()
        changed = batch._generation_batch.numerical_schedule_for_uid(uid)
        assert changed["numerical_schedule_changed"]
        assert changed["numerical_last_geometry"][2] == 2
        assert changed["numerical_initial_geometry"] == initial

        # Invoke the same committed-boundary ownership transfer as a bounded
        # scheduler fallback, then execute the real ordinary continuation.
        native = batch._generation_batch
        packages = native._detach_packages([0])
        native._plain_ready.extend(packages)
        native._mark_plain_fallback(packages, reason="admission_plain")
        native._retire_plain_handoff_transactions(packages)
        assert native.numerical_schedule_for_uid(uid) == changed
        batch._migrate_plain_fallbacks()
        assert native.numerical_schedule_for_uid(uid) is None
        for _ in range(30):
            _, responses = batch.next()
            terminal = next((row for row in responses if row.finish_reason), None)
            if terminal is not None:
                break
        else:
            raise AssertionError("ordinary continuation did not terminate")
        receipt = terminal.mtp_receipt
        assert receipt["route"] == "ordinary_after_mtp_fallback"
        assert receipt["numerical_schedule_changed"]
        assert receipt["numerical_initial_geometry"] == initial
        assert receipt["numerical_last_geometry"] == ("ordinary", 1, 1, "ordinary")
    finally:
        batch.close()


def test_real_executor_terminal_depth_cap_is_reported_as_changed():
    from route_harness import tiny_qwen38_mtp

    from mlx2.runtime.generate import BatchGenerator

    model, _ = tiny_qwen38_mtp()
    batch = BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                           self_mtp={"persistent": True, "num_draft": 3})
    uid = batch.insert([[1, 7, 3, 9, 2]], max_tokens=[3])[0]
    try:
        for _ in range(15):
            _, responses = batch.next()
            terminal = next((row for row in responses if row.finish_reason), None)
            if terminal is not None:
                break
        else:
            raise AssertionError("capped native verification did not terminate")
        assert terminal.uid == uid
        receipt = terminal.mtp_receipt
        assert receipt["stats"]["draft_proposed"] > 0
        assert receipt["numerical_initial_geometry"][2] < 4
        assert receipt["numerical_schedule_changed"]
    finally:
        batch.close()
