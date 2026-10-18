import hashlib
import json
import struct

import mlx.core as mx
import numpy as np
import pytest
from route_harness import collect, make_engine, patch_host, tiny_qwen38_mtp

from mlx2.runtime.models.cache import KVCache, prompt_cache_payload
from mlx2.runtime.prefill_handoff import (
    MANIFEST_SCHEMA,
    OPERATION,
    OPERATION_REVISION,
    RECORD_SCHEMA,
    PrefillHandoffError,
    build_manifest,
    deserialize_handoff,
    load_records,
    make_plan,
    match_record,
    plan_digest,
    prefill_prefix,
    receipt,
    serialize_handoff,
    token_digest,
    validate_handoff,
)
from mlx2.serving import ServingEngine, trusted_prefill_handoff


def _cache(length=4):
    cache = KVCache()
    values = mx.arange(length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(values, values + 10)
    mx.eval(cache.state)
    return [cache]


def _blob(tokens=(1, 2, 3, 4, 5), **overrides):
    plan = make_plan(len(tokens), 2)
    cache = _cache(len(tokens) - 1)
    fields = {
        "artifact_fingerprint": "artifact",
        "tokenizer_digest": "tokenizer",
        "layout_fingerprint": "layout",
    }
    fields.update(overrides)
    manifest = build_manifest(
        cache=cache,
        token_ids=tokens,
        plan=plan,
        producer={"runtime_digest": "producer", "device_class": "cpu", "mlx": "test"},
        **fields,
    )
    expected = {
        **fields,
        "token_digest": token_digest(tokens[:-1]),
        "plan_digest": plan_digest(plan),
    }
    return serialize_handoff(cache, manifest), expected, manifest


def _rewrite_manifest(blob, mutate):
    offset = 8
    length = struct.unpack(">Q", blob[offset : offset + 8])[0]
    start, end = offset + 8, offset + 8 + length
    manifest = json.loads(blob[start:end])
    mutate(manifest)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return blob[:offset] + struct.pack(">Q", len(encoded)) + encoded + blob[end:]


def _rewrite_payload_metadata(blob, mutate, *, refresh_payload_digest):
    outer_length = struct.unpack(">Q", blob[8:16])[0]
    manifest_start, manifest_end = 16, 16 + outer_length
    manifest = json.loads(blob[manifest_start:manifest_end])
    payload = blob[manifest_end:]
    header_length = struct.unpack("<Q", payload[:8])[0]
    header_end = 8 + header_length
    header = json.loads(payload[8:header_end])
    mutate(header["__metadata__"])
    encoded_header = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode()
    encoded_header += b" " * (-len(encoded_header) % 8)
    rewritten_payload = (
        struct.pack("<Q", len(encoded_header))
        + encoded_header
        + payload[header_end:]
    )
    if refresh_payload_digest:
        manifest["payload_sha256"] = hashlib.sha256(rewritten_payload).hexdigest()
    encoded_manifest = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    return (
        blob[:8]
        + struct.pack(">Q", len(encoded_manifest))
        + encoded_manifest
        + rewritten_payload
    )


def _reason(blob, expected):
    with pytest.raises(PrefillHandoffError) as caught:
        deserialize_handoff(blob, expected=expected)
    return caught.value.reason


def test_round_trip_restores_bit_identical_arrays():
    blob, expected, _ = _blob()
    restored, _ = deserialize_handoff(blob, expected=expected)
    before, _ = prompt_cache_payload(_cache())
    after, _ = prompt_cache_payload(restored)
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(np.asarray(before[name]), np.asarray(after[name]))


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("token_digest", "bad", "token_digest_mismatch"),
        ("plan_digest", "bad", "plan_mismatch"),
        ("artifact_fingerprint", "bad", "artifact_mismatch"),
        ("tokenizer_digest", "bad", "tokenizer_mismatch"),
        ("layout_fingerprint", "bad", "layout_mismatch"),
    ],
)
def test_expected_identity_refusals(field, value, reason):
    blob, expected, _ = _blob()
    expected[field] = value
    assert _reason(blob, expected) == reason


def test_operation_revision_and_malformed_refusals():
    blob, expected, _ = _blob()
    changed = _rewrite_manifest(
        blob, lambda value: value.__setitem__("operation_revision", "old")
    )
    assert _reason(changed, expected) == "operation_revision_mismatch"
    assert _reason(b"wrong", expected) == "malformed"
    assert _reason(b"MLX2HO1\n" + (1 << 40).to_bytes(8, "big"), expected) == "malformed"


def test_tampered_array_bytes_refused():
    blob, expected, _ = _blob()
    tampered = bytearray(blob)
    tampered[-1] ^= 1
    assert _reason(bytes(tampered), expected) == "array_digest_mismatch"


def test_metadata_only_tamper_is_bound_by_payload_digest():
    blob, expected, _ = _blob()
    changed = _rewrite_payload_metadata(
        blob,
        lambda metadata: metadata.__setitem__("1.tampered", "yes"),
        refresh_payload_digest=False,
    )
    assert _reason(changed, expected) == "payload_digest_mismatch"


def test_disallowed_cache_class_token_is_refused_before_materialization():
    blob, expected, _ = _blob()

    def replace_class(metadata):
        key = next(key for key in metadata if key.startswith("2."))
        metadata[key] = "attacker.module:EvilCache"

    changed = _rewrite_payload_metadata(
        blob, replace_class, refresh_payload_digest=True
    )
    assert _reason(changed, expected) == "cache_class_disallowed"


def test_bfloat16_round_trip_is_bit_identical_and_tampering_is_refused():
    tokens = (1, 2, 3, 4, 5)
    plan = make_plan(len(tokens), 2)
    cache = KVCache()
    values = mx.arange(4, dtype=mx.float32).astype(mx.bfloat16).reshape(1, 1, 4, 1)
    cache.update_and_fetch(values, values + 10)
    mx.eval(cache.state)
    manifest = build_manifest(
        cache=[cache], token_ids=tokens, plan=plan,
        artifact_fingerprint="artifact", tokenizer_digest="tokenizer",
        layout_fingerprint="layout",
        producer={"runtime_digest": "producer", "device_class": "cpu", "mlx": "test"},
    )
    expected = {
        "artifact_fingerprint": "artifact", "tokenizer_digest": "tokenizer",
        "layout_fingerprint": "layout", "token_digest": token_digest(tokens[:-1]),
        "plan_digest": plan_digest(plan),
    }
    blob = serialize_handoff([cache], manifest)
    restored, _ = deserialize_handoff(blob, expected=expected)
    before, _ = prompt_cache_payload([cache])
    after, _ = prompt_cache_payload(restored)
    for name in before:
        np.testing.assert_array_equal(
            np.asarray(before[name].view(mx.uint16)),
            np.asarray(after[name].view(mx.uint16)),
        )
    tampered = bytearray(blob)
    tampered[-1] ^= 1
    with pytest.raises(PrefillHandoffError, match="array_digest_mismatch"):
        validate_handoff(bytes(tampered), expected=expected)


class _TinyModel:
    def __call__(self, tokens, *, cache):
        values = tokens.astype(mx.float32).reshape(1, 1, -1, 1)
        cache[0].update_and_fetch(values, values)
        total = mx.sum(cache[0].keys_and_values()[0], axis=(1, 2, 3))
        return mx.stack([total, total + 1, total + 2], axis=-1)[:, None, :]


def test_restored_prefix_final_token_logits_are_lossless_same_host():
    tokens = [2, 4, 6, 8, 10]
    plan = make_plan(len(tokens), 2)
    model = _TinyModel()
    reference = [KVCache()]
    prefill_prefix(model, reference, tokens, plan)
    want = model(mx.array([[tokens[-1]]]), cache=reference)
    prefix = [KVCache()]
    prefill_prefix(model, prefix, tokens, plan)
    manifest = build_manifest(
        cache=prefix, token_ids=tokens, plan=plan, artifact_fingerprint="artifact",
        tokenizer_digest="tokenizer", layout_fingerprint="layout",
        producer={"runtime_digest": "producer", "device_class": "cpu", "mlx": "test"},
    )
    expected = {
        "artifact_fingerprint": "artifact", "tokenizer_digest": "tokenizer",
        "layout_fingerprint": "layout", "token_digest": token_digest(tokens[:-1]),
        "plan_digest": plan_digest(plan),
    }
    restored, _ = deserialize_handoff(serialize_handoff(prefix, manifest), expected=expected)
    got = model(mx.array([[tokens[-1]]]), cache=restored)
    np.testing.assert_array_equal(np.asarray(want), np.asarray(got))


def _record(**changes):
    value = {
        "schema": RECORD_SCHEMA, "record_id": "r1", "operation": OPERATION,
        "operation_revision": OPERATION_REVISION, "verdict": "qualified",
        "artifact_fingerprint": "artifact",
        "producer": {"runtime_digest": "producer"},
        "receiver": {
            "runtime_digest": "receiver",
            "execution_digest": "execution",
        },
        "plan_chunks": [2], "max_prompt_tokens": 8,
        "gates": {gate: {"pass": True} for gate in ("G1", "G2", "G3", "G4", "G5")},
        "record_state": "final",
    }
    value.update(changes)
    return value


def test_match_record_all_bindings():
    plan = make_plan(5, 2)
    kwargs = {
        "artifact_fingerprint": "artifact",
        "producer_runtime_digest": "producer",
        "receiver_runtime_digest": "receiver",
        "receiver_execution_digest": "execution",
        "plan": plan,
    }
    record = _record()
    assert match_record([record], **kwargs) is record
    assert match_record([_record(operation="other")], **kwargs) is None
    assert match_record([_record(operation_revision="old")], **kwargs) is None
    assert match_record([_record(artifact_fingerprint="other")], **kwargs) is None
    assert match_record([_record(producer={"runtime_digest": "other"})], **kwargs) is None
    assert match_record([_record(receiver={"runtime_digest": "other"})], **kwargs) is None
    assert match_record([
        _record(receiver={"runtime_digest": "receiver", "execution_digest": "other"})
    ], **kwargs) is None
    assert match_record([_record(max_prompt_tokens=4)], **kwargs) is None
    assert match_record([_record(plan_chunks=[512])], **kwargs) is None


def test_load_records_enforces_final_and_provisional_gates(tmp_path):
    final = _record()
    final.pop("record_state")
    provisional = _record(
        record_id="provisional",
        provisional=True,
        gates={gate: {"pass": True} for gate in ("G1", "G2", "G3", "G4")},
    )
    provisional.pop("record_state")
    failed = _record(record_id="failed")
    failed.pop("record_state")
    failed["gates"]["G3"] = {"pass": False}
    for name, value in (("final", final), ("provisional", provisional), ("failed", failed)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))

    records = load_records(tmp_path)
    assert [(item["record_id"], item["record_state"]) for item in records] == [
        ("r1", "final")
    ]
    records = load_records(tmp_path, allow_provisional=True)
    assert {(item["record_id"], item["record_state"]) for item in records} == {
        ("r1", "final"),
        ("provisional", "provisional"),
    }


def test_provisional_receipt_is_matched_but_not_qualified():
    record = _record(record_state="provisional", provisional=True)
    value = receipt(
        manifest={"producer": {}, "plan_digest": "plan", "token_count": 2},
        record=record,
        observed_used=False,
        bytes_moved=10,
    )
    assert value["record_id"] == "r1"
    assert value["record_state"] == "provisional"
    assert value["qualified"] is False


def test_manifest_schema_constant_is_bound():
    _, _, manifest = _blob()
    assert manifest["schema"] == MANIFEST_SCHEMA


def test_engine_import_bypasses_apcv2_and_receipts_observed_use(monkeypatch):
    from mlx2.runtime.apc_v2 import APCv2

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, prefill_handoff="both")
    tokens = [index % (vocab - 2) + 1 for index in range(24)]
    body = {"tokens": tokens, "max_tokens": 2, "temperature": 0}
    try:
        planned = engine.prefill_handoff_plan(body)
        blob = engine.export_prefill_handoff(tokens, planned["plan"])
        expected = {
            field: planned[field]
            for field in (
                "artifact_fingerprint",
                "tokenizer_digest",
                "layout_fingerprint",
                "token_digest",
                "plan_digest",
            )
        }
        manifest = validate_handoff(blob, expected=expected)
        calls = {"lookup": 0, "store": 0}
        real_lookup, real_store = APCv2.lookup, APCv2.store

        def lookup(self, *args, **kwargs):
            calls["lookup"] += 1
            return real_lookup(self, *args, **kwargs)

        def store(self, *args, **kwargs):
            calls["store"] += 1
            return real_store(self, *args, **kwargs)

        monkeypatch.setattr(APCv2, "lookup", lookup)
        monkeypatch.setattr(APCv2, "store", store)
        result = collect(
            engine.submit(
                {
                    **body,
                    "_mlx2_prefill_handoff": trusted_prefill_handoff(
                        blob=blob, manifest=manifest, record=None,
                        bytes_moved=len(blob),
                    ),
                }
            )
        )
    finally:
        engine.close()
    assert "error" not in result
    assert calls == {"lookup": 0, "store": 0}
    receipt = result["receipt"]["prefill_handoff"]
    assert receipt["observed_used"] is True
    assert receipt["qualified"] is False


def test_engine_refuses_plain_dict_handoff(monkeypatch):
    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, prefill_handoff="both")
    try:
        with pytest.raises(ValueError, match="public requests cannot provide"):
            engine.submit({
                "tokens": [1, 2, 3],
                "max_tokens": 1,
                "_mlx2_prefill_handoff": {"manifest": {}},
            })
    finally:
        engine.close()


def test_export_validation_precedes_exclusive_model_operation():
    class Budget:
        prefill_transient_bytes = None

        @staticmethod
        def project(tokens):
            return tokens * 10

    class Adapter:
        tokenizer = type("Tokenizer", (), {"vocab_size": 8})()

        @staticmethod
        def cache_budget(*, mtp):
            assert mtp is False
            return Budget()

    engine = object.__new__(ServingEngine)
    engine.PREFILL_HANDOFF_EXPORT_WAIT_SECONDS = 0
    engine.adapter = Adapter()
    engine.mtp = False
    engine.max_context = 4
    engine.prefill_step = 2
    engine.prefill_step_autoscale = False
    engine._hard_reserve_gib = 0
    engine.moe_expert_streaming_policy = None
    engine.projection_workspace_reserve_gib = lambda: 0
    engine.admission_headroom_fn = lambda: lambda: 100
    called = []
    engine._exclusive_adapter_operation = lambda *_args, **_kwargs: called.append(True)

    cases = (
        ([1], make_plan(1, 2), "prompt_too_short"),
        ([1, 8], make_plan(2, 2), "invalid_token_ids"),
        ([1, 2, 3], make_plan(3, 1), "plan_chunk_mismatch"),
        ([1, 2, 3, 4, 5], make_plan(5, 2), "context_limit_exceeded"),
    )
    for tokens, plan, reason in cases:
        with pytest.raises(PrefillHandoffError) as raised:
            engine.export_prefill_handoff(tokens, plan)
        assert raised.value.reason == reason
    engine.max_context = 8
    engine.admission_headroom_fn = lambda: lambda: 1
    with pytest.raises(PrefillHandoffError) as raised:
        engine.export_prefill_handoff([1, 2], make_plan(2, 2))
    assert raised.value.reason == "memory_admission_refused"
    assert called == []


def test_export_token_bound_is_the_embedding_not_the_base_tokenizer_vocab():
    """Chat specials sit above a tokenizer's base ``vocab_size`` (Qwen3.6:
    248044 base, 248320 embedding rows); the live M3 export of a chat prompt
    was refused before this used the model config's vocab_size."""

    class Budget:
        prefill_transient_bytes = None

        @staticmethod
        def project(tokens):
            return tokens

    class Adapter:
        tokenizer = type("Tokenizer", (), {"vocab_size": 8})()
        model = type("Model", (), {"args": type("Args", (), {"vocab_size": 16})()})()

        @staticmethod
        def cache_budget(*, mtp):
            return Budget()

    engine = object.__new__(ServingEngine)
    engine.PREFILL_HANDOFF_EXPORT_WAIT_SECONDS = 0
    engine.adapter = Adapter()
    engine.mtp = False
    engine.max_context = 64
    engine.prefill_step = 2
    engine.prefill_step_autoscale = False
    engine._hard_reserve_gib = 0
    engine.moe_expert_streaming_policy = None
    engine.projection_workspace_reserve_gib = lambda: 0
    engine.admission_headroom_fn = lambda: lambda: 10**9
    called = []
    engine._exclusive_adapter_operation = lambda *_args, **_kwargs: called.append(True)

    engine.export_prefill_handoff([1, 12, 3], make_plan(3, 2))
    assert called == [True]
    with pytest.raises(PrefillHandoffError) as raised:
        engine.export_prefill_handoff([1, 16, 3], make_plan(3, 2))
    assert raised.value.reason == "invalid_token_ids"


def test_export_admission_counts_evictable_apc_and_reclaims_before_refusing():
    class Budget:
        prefill_transient_bytes = None

        @staticmethod
        def project(tokens):
            return 100

    class Adapter:
        tokenizer = type("Tokenizer", (), {"vocab_size": 8})()

        @staticmethod
        def cache_budget(*, mtp):
            return Budget()

    def engine_with(free, evictable, reclaimed):
        engine = object.__new__(ServingEngine)
        engine.PREFILL_HANDOFF_EXPORT_WAIT_SECONDS = 0
        engine.adapter = Adapter()
        engine.mtp = False
        engine.max_context = 64
        engine.prefill_step = 2
        engine.prefill_step_autoscale = False
        engine._hard_reserve_gib = 0
        engine.moe_expert_streaming_policy = None
        engine.projection_workspace_reserve_gib = lambda: 0
        engine.admission_headroom_fn = lambda: lambda: free[0]
        engine.apc = type("APC", (), {"unleased_resident_nbytes": staticmethod(lambda: evictable)})()
        engine._clear_allocator_cache_before_reject = lambda: reclaimed.append("clear")
        engine._settle_footprint_before_reject = lambda: reclaimed.append("settle")
        engine._exclusive_adapter_operation = lambda *_a, **_k: reclaimed.append("run")
        return engine

    reclaimed = []
    engine_with([60], 50, reclaimed).export_prefill_handoff([1, 2, 3], make_plan(3, 2))
    assert reclaimed == ["run"]  # 60 free + 50 evictable >= 100: no reclaim needed

    reclaimed, free = [], [10]
    engine = engine_with(free, 0, reclaimed)
    engine._clear_allocator_cache_before_reject = lambda: (reclaimed.append("clear"), free.__setitem__(0, 200))
    engine.export_prefill_handoff([1, 2, 3], make_plan(3, 2))
    assert reclaimed == ["clear", "run"]

    reclaimed = []
    with pytest.raises(PrefillHandoffError) as raised:
        engine_with([10], 0, reclaimed).export_prefill_handoff([1, 2, 3], make_plan(3, 2))
    assert raised.value.reason == "memory_admission_refused"
    assert reclaimed == ["clear", "settle"]


def test_export_admission_waits_for_headroom_before_refusing(monkeypatch):
    """Found live 2026-10-18: back-to-back exports on the 36 GiB M3 refused on
    the first headroom reading (Metal had not yet returned the previous
    export's pages), while ordinary admission waits."""
    import mlx2.serving as serving_module

    class Budget:
        prefill_transient_bytes = None

        @staticmethod
        def project(tokens):
            return 100

    class Adapter:
        tokenizer = type("Tokenizer", (), {"vocab_size": 8})()

        @staticmethod
        def cache_budget(*, mtp):
            return Budget()

    clock = [0.0]
    monkeypatch.setattr(serving_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(serving_module.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    readings = iter([10, 10, 10, 200])  # headroom recovers after ~0.75 s
    events = []
    engine = object.__new__(ServingEngine)
    engine.PREFILL_HANDOFF_EXPORT_WAIT_SECONDS = 15.0
    engine.adapter = Adapter()
    engine.mtp = False
    engine.max_context = 64
    engine.prefill_step = 2
    engine.prefill_step_autoscale = False
    engine._hard_reserve_gib = 0
    engine.moe_expert_streaming_policy = None
    engine.projection_workspace_reserve_gib = lambda: 0
    engine.admission_headroom_fn = lambda: lambda: next(readings)
    engine.apc = None
    engine._clear_allocator_cache_before_reject = lambda: events.append("clear")
    engine._settle_footprint_before_reject = lambda: events.append("settle")
    engine._exclusive_adapter_operation = lambda *_a, **_k: events.append("run")

    engine.export_prefill_handoff([1, 2, 3], make_plan(3, 2))
    assert events == ["run"]  # admitted after waiting; no reclaim, no refusal
    assert 0.5 <= clock[0] < 15.0

    clock[0] = 0.0
    events.clear()
    engine.admission_headroom_fn = lambda: lambda: 10
    with pytest.raises(PrefillHandoffError) as raised:
        engine.export_prefill_handoff([1, 2, 3], make_plan(3, 2))
    assert raised.value.reason == "memory_admission_refused"
    assert events == ["clear", "settle"]  # reclaim only after the bounded wait
    assert clock[0] >= 15.0
