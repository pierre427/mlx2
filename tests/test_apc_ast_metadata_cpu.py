"""CPU contracts for the APCv2-owned AST metadata/debug memo."""

import threading

import pytest

from mlx2.runtime.apc_ast_metadata import APCASTMemoStore, shadow_checked_metadata
from mlx2.runtime.apc_v2 import APCSessionNotFound, APCv2


def _store(clock, *, debug=False, **limits):
    return APCASTMemoStore(
        lock=threading.RLock(),
        now_fn=lambda: clock[0],
        wall_time_fn=lambda: 1000 + clock[0],
        enabled=True,
        debug_enabled=debug,
        session_max_ttl_seconds=30,
        limits=limits or None,
    )


def _scope(character):
    return character * 64


def _identity(*scopes, kind="ragged_geometry", geometry="9", **changes):
    value = {
        "schema": "mlx2.apcv2.ast-memo.v1",
        "value_kind": kind,
        "apc_semantic_epoch": changes.pop("apc_semantic_epoch", 0),
        "scope_epochs": changes.pop(
            "scope_epochs", tuple((scope, 0) for scope in scopes)
        ),
        "layout_revision_sha256": "1" * 64,
        "target_fingerprint": "target@fingerprint",
        "drafter_fingerprint": "drafter@fingerprint",
        "source_build_device_sha256": "2" * 64,
        "policy_sha256": "3" * 64,
        "owner_semantic_contract_sha256": "4" * 64,
        "capability_manifest_sha256": "5" * 64,
        "cohort_geometry_sha256": geometry * 64,
    }
    value.update(changes)
    return value


def _value(second_length=4):
    lengths = [2, second_length]
    maximum = max(lengths)
    logical = sum(lengths)
    return {
        "schema": "mlx2.ast-ragged-geometry.v1",
        "query_lengths": lengths,
        "cu_query_lengths": [0, 2, logical],
        "right_padding": [maximum - 2, maximum - second_length],
        "token_to_lane": [0, 0] + [1] * second_length,
        "backend_mode": "padded",
        "logical_rows": logical,
        "physical_rows": len(lengths) * maximum,
        "padding_rows": len(lengths) * maximum - logical,
        "max_query_len": maximum,
    }


def _debug_record(**changes):
    value = {
        "lookup_input_sha256": "a" * 64,
        "candidate_backends_and_shapes": [{"backend": "padded", "shape": [2, 4]}],
        "selected_plan": {"backend": "padded", "shape": [2, 4]},
        "decline_reasons": [],
        "memo_outcome": "miss",
        "invalidation_cause": None,
        "shadow_comparison": None,
    }
    value.update(changes)
    return value


def _apc(**kwargs):
    return APCv2(
        layout_name="test-layout",
        max_size=2,
        max_bytes=1 << 20,
        **kwargs,
    )


def test_apcv2_memo_is_default_off_and_clear_empty_trie_still_advances_epoch():
    apc = _apc()
    try:
        before = apc.ast_memo_receipt()
        assert before["enabled"] is False
        assert apc.ast_memo_handle() is None
        apc.clear(release_memory=False)
        after = apc.ast_memo_receipt()
        assert after["apc_semantic_epoch"] == before["apc_semantic_epoch"] + 1
        assert apc.apc_stats["ast_memo"]["enabled"] is False
    finally:
        apc.close(release_memory=False)


def test_executor_entries_are_shareable_but_session_entries_are_tenant_isolated():
    store = _store([0.0])
    first, second = _scope("a"), _scope("b")
    assert store.register_scope(
        first, tenant="tenant-a", session_tag=("tenant-a", "one"), named=True
    )
    assert store.register_scope(
        second, tenant="tenant-b", session_tag=("tenant-b", "two"), named=True
    )
    static = _identity()
    assert store.put(static, _value())
    assert store.get(static) == _value()
    with pytest.raises(ValueError, match="static executor tier"):
        store.put(_identity(first), _value())
    # A scope cannot be rebound across tenants even if a caller knows its hash.
    assert not store.register_scope(
        first, tenant="tenant-b", session_tag=("tenant-b", "one"), named=True
    )


def test_composite_key_binds_every_participant_and_one_advance_invalidates_it():
    store = _store([0.0])
    first, second = _scope("a"), _scope("b")
    store.register_scope(
        first, tenant="tenant", session_tag=("tenant", "one"), named=True
    )
    store.register_scope(
        second, tenant="tenant", session_tag=("tenant", "two"), named=True
    )
    identity = _identity(first, second)
    assert store.debug_append(identity, _debug_record()) is None
    before = store.snapshot((first, second))["scope_epochs"]
    assert store.advance((second,), "transaction_commit")
    assert store.snapshot((first, second))["scope_epochs"] == [
        (first, 0),
        (second, 1),
    ]
    after = store.snapshot((first, second))["scope_epochs"]
    assert before == [(first, 0), (second, 0)]
    assert after == [(first, 0), (second, 1)]


def test_static_geometry_survives_transaction_epoch_and_session_tier_is_refused():
    store = _store([0.0])
    scope = _scope("6")
    store.register_scope(scope)
    static = _identity(kind="ragged_geometry")
    assert store.put(static, _value())
    with pytest.raises(ValueError, match="unsupported AST memo value kind"):
        store.put(_identity(scope, kind="segment_analysis"), _value())
    store.advance((scope,), "transaction_commit")
    assert store.get(static) == _value()


def test_shadow_mismatch_keeps_live_geometry_and_atomically_repairs_static_entry():
    store = _store([0.0])
    handle = store.handle()
    identity = _identity(kind="ragged_geometry")
    stale, live = _value(3), _value(4)
    assert handle.put(identity, stale)
    comparison = shadow_checked_metadata(handle, identity, live)
    assert comparison["outcome"] == "shadow_mismatch"
    assert comparison["prior_present"] is True
    assert comparison["match"] is False
    assert handle.get(identity) == live


def test_atomic_shadow_publish_refuses_an_epoch_that_changed_during_recompute():
    store = _store([0.0])
    handle = store.handle()
    stale_identity = _identity()
    # This models an APC semantic revision changing after the caller captured
    # its key but before live analysis reached compare-and-publish.
    store.clear("model_invalidation")
    comparison = handle.shadow_compare_put(stale_identity, _value())
    assert comparison["outcome"] == "decline"
    assert store.snapshot()["executor_entries"] == 0


@pytest.mark.parametrize("cause", ["transaction_commit", "transaction_abort"])
def test_commit_and_abort_advance_only_after_explicit_close_signal(cause):
    store = _store([0.0])
    scope = _scope("c")
    store.register_scope(
        scope, tenant="tenant", session_tag=("tenant", "session"), named=True
    )
    assert store.put(_identity(), _value())
    assert store.get(_identity()) == _value()
    assert store.advance((scope,), cause)
    assert store.get(_identity()) == _value()
    assert store.snapshot((scope,))["invalidations"][cause] == 1


def test_tagged_publication_advances_and_session_delete_purges_before_lookup():
    store = _store([0.0])
    scope = _scope("d")
    tag = ("tenant", "session")
    store.register_scope(scope, tenant="tenant", session_tag=tag, named=True)
    assert store.put(_identity(), _value())
    assert store.advance_tag(tag, "tagged_publication")
    assert store.get(_identity()) == _value()
    assert store.purge_tag(tag, "session_end") == 1
    assert store.snapshot((scope,)) is None

    apc = _apc(ast_memo_enabled=True)
    called = []
    original = apc._ast_memo.purge_tag
    apc._ast_memo.purge_tag = lambda value, cause: called.append((value, cause)) or 0
    try:
        with pytest.raises(APCSessionNotFound):
            apc.delete_session("tenant", "missing")
        assert called == [(("tenant", "missing"), "session_end")]
    finally:
        apc._ast_memo.purge_tag = original
        apc.close(release_memory=False)


def test_ttl_payload_entry_and_byte_caps_evict_lru_without_unbounded_state():
    clock = [0.0]
    store = _store(
        clock,
        executor_entries=1,
        executor_bytes=1024,
        executor_ttl=5,
        session_entries=1,
        session_bytes=1024,
        session_global_entries=2,
        payload_bytes=512,
    )
    scope = _scope("e")
    store.register_scope(scope)
    assert store.put(_identity(geometry="8"), _value(3))
    assert store.put(_identity(geometry="9"), _value(4))
    assert store.get(_identity(geometry="8")) is None
    assert store.get(_identity(geometry="9")) == _value(4)
    clock[0] = 31.0
    assert store.snapshot((scope,)) is None
    assert store.snapshot()["scopes"] == 0


def test_scope_count_is_globally_bounded_and_ttl_removes_tag_membership():
    clock = [0.0]
    store = _store(clock, session_global_scopes=2)
    named, anonymous, refused = _scope("1"), _scope("2"), _scope("3")
    tag = ("tenant", "named")
    assert store.register_scope(
        named, tenant="tenant", session_tag=tag, named=True
    )
    assert store.register_scope(anonymous)
    assert not store.register_scope(refused)
    assert store.snapshot()["scopes"] == 2
    assert store.snapshot()["tag_bindings"] == 1
    clock[0] = 31.0
    assert store.snapshot()["scopes"] == 0
    assert store.snapshot()["tag_bindings"] == 0
    assert store.purge_tag(tag) == 0


def test_named_scope_binds_one_session_tag_and_anonymous_scope_forbids_tags():
    store = _store([0.0])
    scope = _scope("4")
    assert store.register_scope(
        scope, tenant="tenant", session_tag=("tenant", "one"), named=True
    )
    assert not store.register_scope(
        scope, tenant="tenant", session_tag=("tenant", "two"), named=True
    )
    with pytest.raises(ValueError, match="named scopes require"):
        store.register_scope(_scope("5"), tenant="tenant", named=True)
    with pytest.raises(ValueError, match="anonymous scopes forbid"):
        store.register_scope(_scope("6"), session_tag=("tenant", "six"))


def test_narrow_handle_failure_disables_and_clears_namespace_without_raising():
    store = _store([0.0])
    handle = store.handle()
    assert handle.put(_identity(), _value())
    original = store.advance

    def broken(*_args, **_kwargs):
        raise RuntimeError("synthetic lifecycle failure")

    store.advance = broken
    try:
        assert handle.advance((_scope("7"),), "transaction_abort") is False
    finally:
        store.advance = original
    receipt = store.snapshot()
    assert receipt["enabled"] is False
    assert receipt["executor_entries"] == 0
    assert receipt["stats"]["diagnostic_failures"] == 1


def test_malformed_keys_and_payloads_fail_closed_and_never_store_values():
    store = _store([0.0])
    scope = _scope("f")
    store.register_scope(scope)
    malformed = _identity(scope)
    malformed.pop("policy_sha256")
    with pytest.raises(ValueError, match="incomplete"):
        store.get(malformed)
    with pytest.raises(ValueError, match="incomplete or unsupported"):
        store.put(
            _identity(),
            {"schema": "mlx2.ast-ragged-geometry.v1", "input_ids": [1, 2]},
        )
    with pytest.raises(ValueError, match="incomplete or unsupported"):
        store.put(
            _identity(),
            {"schema": "mlx2.ast-ragged-geometry.v1", "proposal": [1, 2]},
        )
    with pytest.raises(ValueError, match="unsupported AST memo value kind"):
        store.put(_identity(kind="capability_manifest"), _value())
    assert store.snapshot()["session_entries"] == 0


def test_debug_is_separate_bounded_token_free_and_shadow_compare_has_no_callback():
    scope = _scope("7")
    off = _store([0.0], debug=False)
    off.register_scope(scope)
    assert off.debug_append(_identity(scope), _debug_record()) is None

    on = _store(
        [0.0],
        debug=True,
        debug_records=2,
        debug_scope_bytes=8192,
        debug_global_bytes=8192,
        debug_record_bytes=4096,
    )
    on.register_scope(scope)
    assert on.debug_append(_identity(scope), _debug_record()) == 0
    assert (
        on.debug_append(
            _identity(scope),
            _debug_record(
                memo_outcome="hit",
                shadow_comparison={
                    "outcome": "shadow_mismatch",
                    "prior_present": True,
                    "match": False,
                    "prior_sha256": "c" * 64,
                    "live_sha256": "b" * 64,
                },
            ),
        )
        == 1
    )
    assert on.debug_append(_identity(scope), _debug_record(memo_outcome="decline")) == 2
    records = on.debug_records((scope,))
    assert len(records) == 2
    assert all(
        "key_sha256" in record and "scope_sha256" in record for record in records
    )
    with pytest.raises(ValueError, match="incomplete or unsupported"):
        on.debug_append(
            _identity(scope),
            _debug_record(selected_plan={"backend": "padded", "shape": [2, 4], "input_ids": [1]}),
        )
    # There is no recompute callback API: diagnostics compare caller-supplied
    # live hashes and therefore cannot execute or influence serving.
    assert not hasattr(on, "shadow_recompute")


def test_layout_revision_binds_apc_key_layout_and_state_codec():
    apc = _apc(ast_memo_enabled=True)
    try:
        key = apc.key(
            "model",
            revision="r1",
            adapter="a",
            tokenizer_fingerprint="t",
            cache_layout_fingerprint="layout",
            semantic_fingerprint="semantic",
        )
        first = apc.ast_memo_layout_revision(key)
        assert len(first) == 64
        changed = apc.key(
            "model",
            revision="r2",
            adapter="a",
            tokenizer_fingerprint="t",
            cache_layout_fingerprint="layout",
            semantic_fingerprint="semantic",
        )
        assert apc.ast_memo_layout_revision(changed) != first
    finally:
        apc.close(release_memory=False)


def test_normal_and_indeterminate_close_purge_the_child_store():
    normal = _apc(ast_memo_enabled=True)
    normal.ast_memo_handle().put(_identity(), _value())
    normal.close(release_memory=False)
    assert normal.ast_memo_receipt()["session_entries"] == 0

    indeterminate = _apc(ast_memo_enabled=True)
    indeterminate.ast_memo_handle().put(_identity(), _value())
    indeterminate._metadata_failure = RuntimeError("synthetic")
    indeterminate.close(release_memory=False)
    assert indeterminate.ast_memo_receipt()["session_entries"] == 0
