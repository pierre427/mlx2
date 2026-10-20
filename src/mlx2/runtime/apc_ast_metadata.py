"""Bounded AST metadata owned by one APCv2 lifecycle.

This is not a prefix, tensor, result, or planner cache. It stores only
canonical JSON metadata and diagnostic records. Every state-bound key includes
the APC semantic epoch and ordered participating scope epochs. A hit is never
an execution attestation; callers must revalidate capability and layout.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, OrderedDict
from dataclasses import dataclass

SCHEMA = "mlx2.apcv2.ast-memo.v1"
DEBUG_SCHEMA = "mlx2.apcv2.ast-memo-debug.v1"

EXECUTOR_MAX_ENTRIES = 128
EXECUTOR_MAX_BYTES = 16 << 20
EXECUTOR_IDLE_SECONDS = 3600
SESSION_MAX_ENTRIES = 64
SESSION_MAX_BYTES = 2 << 20
SESSION_MAX_SCOPES_PER_TENANT = 64
SESSION_GLOBAL_MAX_SCOPES = 512
SESSION_GLOBAL_MAX_ENTRIES = 512
SESSION_IDLE_SECONDS = 1800
MAX_PAYLOAD_BYTES = 256 << 10
DEBUG_MAX_RECORDS = 64
DEBUG_MAX_BYTES_PER_SCOPE = 256 << 10
DEBUG_GLOBAL_MAX_BYTES = 16 << 20
DEBUG_IDLE_SECONDS = 600
DEBUG_MAX_RECORD_BYTES = 8 << 10

_VALUE_KINDS = frozenset({"ragged_geometry"})
_RAGGED_GEOMETRY_SCHEMA = "mlx2.ast-ragged-geometry.v1"
_RAGGED_GEOMETRY_FIELDS = frozenset(
    {
        "schema",
        "query_lengths",
        "cu_query_lengths",
        "right_padding",
        "token_to_lane",
        "backend_mode",
        "logical_rows",
        "physical_rows",
        "padding_rows",
        "max_query_len",
    }
)
_RAGGED_BACKENDS = frozenset({"padded", "native_ragged", "flattened"})
_INVALIDATION_CAUSES = frozenset(
    {
        "transaction_commit",
        "transaction_abort",
        "tagged_publication",
        "tagged_drop",
        "session_end",
        "anonymous_lane_end",
        "policy_change",
        "source_or_build_change",
        "target_or_drafter_change",
        "capability_change",
        "layout_change",
        "apcv2_clear",
        "apcv2_close",
        "model_invalidation",
        "ttl_expired",
    }
)
_DEBUG_FIELDS = frozenset(
    {
        "lookup_input_sha256",
        "candidate_backends_and_shapes",
        "selected_plan",
        "decline_reasons",
        "memo_outcome",
        "invalidation_cause",
        "shadow_comparison",
    }
)


def _digest(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _bounded_string(value, label):
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(f"{label} must be a bounded nonempty string")
    return value


def _bounded_integer_list(value, label, *, minimum=0, maximum=1 << 31):
    if (
        not isinstance(value, (tuple, list))
        or len(value) > 4096
        or any(
            type(item) is not int or item < minimum or item > maximum
            for item in value
        )
    ):
        raise ValueError(f"{label} must be a bounded integer sequence")
    return list(value)


def _validate_ragged_geometry(value):
    if not isinstance(value, dict) or set(value) != _RAGGED_GEOMETRY_FIELDS:
        raise ValueError("ragged geometry fields are incomplete or unsupported")
    if value["schema"] != _RAGGED_GEOMETRY_SCHEMA:
        raise ValueError("ragged geometry schema mismatch")
    lengths = _bounded_integer_list(value["query_lengths"], "query_lengths", minimum=1)
    if not lengths:
        raise ValueError("query_lengths must be nonempty")
    offsets = _bounded_integer_list(value["cu_query_lengths"], "cu_query_lengths")
    padding = _bounded_integer_list(value["right_padding"], "right_padding")
    lanes = _bounded_integer_list(value["token_to_lane"], "token_to_lane")
    expected_offsets = [0]
    for length in lengths:
        expected_offsets.append(expected_offsets[-1] + length)
    max_query_len = max(lengths)
    expected_padding = [max_query_len - length for length in lengths]
    expected_lanes = [
        index for index, length in enumerate(lengths) for _ in range(length)
    ]
    if offsets != expected_offsets:
        raise ValueError("cu_query_lengths does not match query_lengths")
    if padding != expected_padding:
        raise ValueError("right_padding does not match query_lengths")
    if lanes != expected_lanes:
        raise ValueError("token_to_lane does not match ragged geometry")
    backend = value["backend_mode"]
    if backend not in _RAGGED_BACKENDS:
        raise ValueError("unsupported ragged geometry backend")
    counts = {}
    for field in ("logical_rows", "physical_rows", "padding_rows", "max_query_len"):
        item = value[field]
        if type(item) is not int or item < 0 or item > 1 << 31:
            raise ValueError(f"{field} must be a bounded non-negative integer")
        counts[field] = item
    logical_rows = sum(lengths)
    expected_physical = (
        len(lengths) * max_query_len if backend == "padded" else logical_rows
    )
    if counts != {
        "logical_rows": logical_rows,
        "physical_rows": expected_physical,
        "padding_rows": expected_physical - logical_rows,
        "max_query_len": max_query_len,
    }:
        raise ValueError("ragged geometry row counts are inconsistent")
    return {
        "schema": _RAGGED_GEOMETRY_SCHEMA,
        "query_lengths": lengths,
        "cu_query_lengths": offsets,
        "right_padding": padding,
        "token_to_lane": lanes,
        "backend_mode": backend,
        **counts,
    }


def _validate_value(kind, value):
    if kind == "ragged_geometry":
        return _validate_ragged_geometry(value)
    raise ValueError("unsupported AST memo value kind")


def _validate_shape(value, label):
    result = _bounded_integer_list(value, label, minimum=1, maximum=1 << 20)
    if not result or len(result) > 4:
        raise ValueError(f"{label} must contain one to four positive dimensions")
    return result


def _validate_debug_plan(value, label, *, candidate):
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    if set(value) == {"backend", "shape"}:
        if value["backend"] not in _RAGGED_BACKENDS:
            raise ValueError(f"{label} backend is unsupported")
        return {
            "backend": value["backend"],
            "shape": _validate_shape(value["shape"], f"{label}.shape"),
        }
    fields = (
        {"backend", "logical_rows", "physical_rows", "max_query_len"}
        if candidate
        else {
            "backend_mode",
            "logical_rows",
            "physical_rows",
            "padding_rows",
            "max_query_len",
        }
    )
    if set(value) != fields:
        raise ValueError(f"{label} fields are incomplete or unsupported")
    backend_field = "backend" if candidate else "backend_mode"
    if value[backend_field] not in _RAGGED_BACKENDS:
        raise ValueError(f"{label} backend is unsupported")
    result = {backend_field: value[backend_field]}
    for field in fields - {backend_field}:
        item = value[field]
        if type(item) is not int or item < 0 or item > 1 << 31:
            raise ValueError(f"{label}.{field} must be a bounded integer")
        result[field] = item
    return result


def _validate_debug_record(record):
    if not isinstance(record, dict) or set(record) != _DEBUG_FIELDS:
        raise ValueError("AST debug record fields are incomplete")
    candidates = record["candidate_backends_and_shapes"]
    if not isinstance(candidates, (tuple, list)) or len(candidates) > 32:
        raise ValueError("AST debug candidates must be a bounded sequence")
    candidates = [
        _validate_debug_plan(value, "candidate", candidate=True)
        for value in candidates
    ]
    selected = record["selected_plan"]
    if selected is not None:
        selected = _validate_debug_plan(selected, "selected_plan", candidate=False)
    declines = record["decline_reasons"]
    if not isinstance(declines, (tuple, list)) or len(declines) > 32:
        raise ValueError("AST debug decline reasons must be a bounded sequence")
    declines = list(declines)
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_.:@/-"
    if any(
        not isinstance(item, str)
        or not item
        or len(item) > 128
        or any(character not in allowed for character in item)
        for item in declines
    ):
        raise ValueError("AST debug decline reasons must be bounded identifiers")
    outcome = record["memo_outcome"]
    if outcome not in {None, "hit", "miss", "decline", "shadow_mismatch"}:
        raise ValueError("invalid AST debug memo outcome")
    invalidation = record["invalidation_cause"]
    if invalidation is not None and invalidation not in _INVALIDATION_CAUSES:
        raise ValueError("invalid AST debug invalidation cause")
    shadow = record["shadow_comparison"]
    if shadow is not None:
        fields = {
            "outcome",
            "prior_present",
            "match",
            "prior_sha256",
            "live_sha256",
        }
        if not isinstance(shadow, dict) or set(shadow) != fields:
            raise ValueError("AST debug shadow comparison fields are incomplete")
        if shadow["outcome"] not in {"hit", "miss", "decline", "shadow_mismatch"}:
            raise ValueError("invalid AST debug shadow outcome")
        if type(shadow["prior_present"]) is not bool or type(shadow["match"]) is not bool:
            raise ValueError("AST debug shadow flags must be boolean")
        shadow = dict(shadow)
        for field in ("prior_sha256", "live_sha256"):
            if shadow[field] is not None:
                shadow[field] = _digest(shadow[field], f"shadow_comparison.{field}")
    return {
        "lookup_input_sha256": _digest(
            record["lookup_input_sha256"], "lookup_input_sha256"
        ),
        "candidate_backends_and_shapes": candidates,
        "selected_plan": selected,
        "decline_reasons": declines,
        "memo_outcome": outcome,
        "invalidation_cause": invalidation,
        "shadow_comparison": shadow,
    }


def _canonical_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _tag_digest(tenant, session_id):
    return hashlib.sha256(_canonical_bytes([str(tenant), str(session_id)])).hexdigest()


def _tenant_digest(tenant):
    return hashlib.sha256(str(tenant).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class APCASTMemoKey:
    schema: str
    value_kind: str
    scope_epochs: tuple[tuple[str, int], ...]
    apc_semantic_epoch: int
    layout_revision_sha256: str
    target_fingerprint: str
    drafter_fingerprint: str
    source_build_device_sha256: str
    policy_sha256: str
    owner_semantic_contract_sha256: str
    capability_manifest_sha256: str
    cohort_geometry_sha256: str

    @property
    def tier(self):
        return "session" if self.scope_epochs else "executor"

    def digest(self):
        return hashlib.sha256(
            _canonical_bytes(
                {name: getattr(self, name) for name in self.__dataclass_fields__}
            )
        ).hexdigest()


@dataclass(slots=True)
class _Scope:
    epoch: int
    tenant_sha256: str | None
    named: bool
    session_tag_sha256: str | None
    expires_at: float


@dataclass(slots=True)
class _Entry:
    payload: bytes
    nbytes: int
    expires_at: float


class APCASTMemoHandle:
    """Narrow interface passed to AST; it cannot reach APC prefix state."""

    __slots__ = ("_store",)

    def __init__(self, store):
        self._store = store

    def _call(self, method, fallback, *args, **kwargs):
        try:
            return getattr(self._store, method)(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - diagnostic path is fail-soft
            try:
                self._store.fail_closed(error)
            except BaseException:  # noqa: BLE001,S110 - never mask serving
                # A broken optional namespace must never replace a model result
                # or the original serving exception.
                pass
            return fallback

    def get(self, identity):
        return self._call("get", None, identity)

    def register_scope(self, scope):
        """Register one anonymous request scope; named scopes stay APC-owned."""

        return self._call("register_scope", False, scope)

    def put(self, identity, value):
        return self._call("put", False, identity, value)

    def shadow_compare_put(self, identity, live_value):
        fallback = {
            "outcome": "decline",
            "prior_present": False,
            "match": False,
            "prior_sha256": None,
            "live_sha256": None,
        }
        return self._call("shadow_compare_put", fallback, identity, live_value)

    def advance(self, scopes, cause, *, purge=False):
        return self._call("advance", False, scopes, cause, purge=purge)

    def debug_append(self, identity, record):
        return self._call("debug_append", None, identity, record)

    def snapshot(self, scopes=()):
        return self._call("snapshot", None, scopes)


def shadow_checked_metadata(handle, identity, live_value):
    """Compare memo metadata with a live recomputation and repair drift.

    The live value is always authoritative. The return is diagnostic only;
    callers must continue with ``live_value`` regardless of the outcome.
    """

    return handle.shadow_compare_put(identity, live_value)


class APCASTMemoStore:
    """One memory-only child store whose lifetime is exactly its APCv2 owner."""

    def __init__(
        self,
        *,
        lock,
        now_fn,
        wall_time_fn,
        enabled=False,
        debug_enabled=False,
        session_max_ttl_seconds=3600,
        limits=None,
    ):
        self._lock = lock
        self._now = now_fn
        self._wall_time = wall_time_fn
        self._enabled = bool(enabled)
        self._debug_enabled = bool(enabled and debug_enabled)
        self._session_ttl = min(
            SESSION_IDLE_SECONDS, max(1, int(session_max_ttl_seconds))
        )
        configured = {
            "executor_entries": EXECUTOR_MAX_ENTRIES,
            "executor_bytes": EXECUTOR_MAX_BYTES,
            "executor_ttl": EXECUTOR_IDLE_SECONDS,
            "session_entries": SESSION_MAX_ENTRIES,
            "session_bytes": SESSION_MAX_BYTES,
            "session_scopes_per_tenant": SESSION_MAX_SCOPES_PER_TENANT,
            "session_global_scopes": SESSION_GLOBAL_MAX_SCOPES,
            "session_global_entries": SESSION_GLOBAL_MAX_ENTRIES,
            "payload_bytes": MAX_PAYLOAD_BYTES,
            "debug_records": DEBUG_MAX_RECORDS,
            "debug_scope_bytes": DEBUG_MAX_BYTES_PER_SCOPE,
            "debug_global_bytes": DEBUG_GLOBAL_MAX_BYTES,
            "debug_ttl": DEBUG_IDLE_SECONDS,
            "debug_record_bytes": DEBUG_MAX_RECORD_BYTES,
        }
        if limits is not None:
            if not isinstance(limits, dict) or set(limits) - set(configured):
                raise ValueError("invalid AST memo test limits")
            configured.update(limits)
        if any(type(value) is not int or value <= 0 for value in configured.values()):
            raise ValueError("AST memo limits must be positive integers")
        self._limits = configured
        self._executor = OrderedDict()
        self._session = OrderedDict()
        self._executor_bytes = 0
        self._session_bytes = 0
        self._scopes = {}
        self._tag_scopes = {}
        self._semantic_epoch = 0
        self._debug = OrderedDict()
        self._debug_bytes = 0
        self._debug_sequence = 0
        self._stats = Counter()
        self._invalidation_counts = Counter()

    @property
    def enabled(self):
        return self._enabled

    @property
    def debug_enabled(self):
        return self._debug_enabled

    def handle(self):
        return APCASTMemoHandle(self)

    def fail_closed(self, error):
        """Disable this optional namespace without exposing payload data."""

        with self._lock:
            self._executor.clear()
            self._session.clear()
            self._executor_bytes = self._session_bytes = 0
            self._scopes.clear()
            self._tag_scopes.clear()
            self._debug.clear()
            self._debug_bytes = 0
            self._semantic_epoch += 1
            self._enabled = False
            self._debug_enabled = False
            self._stats["diagnostic_failures"] += 1
            self._stats[f"diagnostic_failure_{type(error).__name__}"] += 1

    def _drop_entry_locked(self, tier, key):
        table = self._session if tier == "session" else self._executor
        entry = table.pop(key, None)
        if entry is not None:
            if tier == "session":
                self._session_bytes -= entry.nbytes
            else:
                self._executor_bytes -= entry.nbytes

    def _drop_scope_entries_locked(self, scope):
        for key in tuple(self._session):
            if any(member == scope for member, _epoch in key.scope_epochs):
                self._drop_entry_locked("session", key)

    def _drop_scope_debug_locked(self, scope):
        for sequence, (scopes, payload, _expires_at) in tuple(self._debug.items()):
            if scope in scopes:
                self._debug_bytes -= len(payload)
                del self._debug[sequence]

    def _drop_scope_tags_locked(self, scope):
        for tag, members in tuple(self._tag_scopes.items()):
            members.discard(scope)
            if not members:
                del self._tag_scopes[tag]

    def _expire_locked(self, now):
        for tier, table in (("executor", self._executor), ("session", self._session)):
            for key, entry in tuple(table.items()):
                if entry.expires_at <= now:
                    self._drop_entry_locked(tier, key)
                    self._stats["evictions"] += 1
        for scope, state in tuple(self._scopes.items()):
            if state.expires_at <= now:
                self._drop_scope_entries_locked(scope)
                self._drop_scope_debug_locked(scope)
                self._drop_scope_tags_locked(scope)
                del self._scopes[scope]
                self._invalidation_counts["ttl_expired"] += 1
        for sequence, (_scopes, payload, expires_at) in tuple(self._debug.items()):
            if expires_at <= now:
                self._debug_bytes -= len(payload)
                del self._debug[sequence]

    def register_scope(self, scope, *, tenant=None, session_tag=None, named=False):
        if not self._enabled:
            return False
        scope = _digest(scope, "scope")
        tenant_sha256 = None if tenant is None else _tenant_digest(tenant)
        if bool(named) != (session_tag is not None):
            raise ValueError("named scopes require one session tag; anonymous scopes forbid it")
        if session_tag is not None and (
            not isinstance(session_tag, tuple)
            or len(session_tag) != 2
            or not isinstance(session_tag[1], str)
        ):
            raise ValueError("invalid APCv2 session tag")
        session_tag_sha256 = (
            None if session_tag is None else _tag_digest(*session_tag)
        )
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            state = self._scopes.get(scope)
            if state is None:
                if len(self._scopes) >= self._limits["session_global_scopes"]:
                    self._stats["declines"] += 1
                    return False
                if tenant_sha256 is not None:
                    count = sum(
                        candidate.tenant_sha256 == tenant_sha256
                        for candidate in self._scopes.values()
                    )
                    if count >= self._limits["session_scopes_per_tenant"]:
                        self._stats["declines"] += 1
                        return False
                state = self._scopes[scope] = _Scope(
                    0,
                    tenant_sha256,
                    bool(named),
                    session_tag_sha256,
                    now + self._session_ttl,
                )
            elif (
                state.tenant_sha256 != tenant_sha256
                or state.named != bool(named)
                or state.session_tag_sha256 != session_tag_sha256
            ):
                self._stats["declines"] += 1
                return False
            else:
                state.expires_at = now + self._session_ttl
            if session_tag is not None:
                self._tag_scopes.setdefault(session_tag_sha256, set()).add(scope)
            return True

    def purge_tag(self, session_tag, cause="session_end"):
        if not self._enabled:
            return 0
        digest = _tag_digest(*session_tag)
        with self._lock:
            scopes = tuple(self._tag_scopes.pop(digest, ()))
            self.advance(scopes, cause, purge=True)
            return len(scopes)

    def advance_tag(self, session_tag, cause):
        if not self._enabled or session_tag is None:
            return False
        digest = _tag_digest(*session_tag)
        with self._lock:
            scopes = tuple(self._tag_scopes.get(digest, ()))
            return self.advance(scopes, cause)

    def _key_locked(self, identity, *, debug=False):
        if not isinstance(identity, dict):
            raise TypeError("AST memo identity must be a mapping")
        required = {
            "schema",
            "value_kind",
            "apc_semantic_epoch",
            "scope_epochs",
            "layout_revision_sha256",
            "target_fingerprint",
            "drafter_fingerprint",
            "source_build_device_sha256",
            "policy_sha256",
            "owner_semantic_contract_sha256",
            "capability_manifest_sha256",
            "cohort_geometry_sha256",
        }
        if set(identity) != required:
            raise ValueError("AST memo identity fields are incomplete")
        if identity["schema"] != SCHEMA:
            raise ValueError("AST memo key schema mismatch")
        kind = identity["value_kind"]
        if kind not in _VALUE_KINDS:
            raise ValueError("unsupported AST memo value kind")
        semantic_epoch = identity["apc_semantic_epoch"]
        if type(semantic_epoch) is not int or semantic_epoch < 0:
            raise ValueError("AST memo semantic epoch must be a non-negative integer")
        if semantic_epoch != self._semantic_epoch:
            return None
        supplied_epochs = tuple(identity["scope_epochs"])
        if len(supplied_epochs) > 16:
            raise ValueError("AST memo scope epochs must be bounded")
        if kind == "ragged_geometry" and supplied_epochs and not debug:
            raise ValueError("ragged geometry belongs to the static executor tier")
        scope_epochs = []
        now = self._now()
        seen = set()
        for item in supplied_epochs:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("AST memo scope epoch is malformed")
            scope, expected_epoch = item
            scope = _digest(scope, "scope")
            if scope in seen or type(expected_epoch) is not int or expected_epoch < 0:
                raise ValueError("AST memo scope epochs must be ordered and unique")
            seen.add(scope)
            state = self._scopes.get(scope)
            if state is None or state.epoch != expected_epoch:
                return None
            state.expires_at = now + self._session_ttl
            scope_epochs.append((scope, expected_epoch))
        return APCASTMemoKey(
            SCHEMA,
            kind,
            tuple(scope_epochs),
            semantic_epoch,
            _digest(identity["layout_revision_sha256"], "layout_revision_sha256"),
            _bounded_string(identity["target_fingerprint"], "target_fingerprint"),
            _bounded_string(identity["drafter_fingerprint"], "drafter_fingerprint"),
            _digest(
                identity["source_build_device_sha256"], "source_build_device_sha256"
            ),
            _digest(identity["policy_sha256"], "policy_sha256"),
            _digest(
                identity["owner_semantic_contract_sha256"],
                "owner_semantic_contract_sha256",
            ),
            _digest(
                identity["capability_manifest_sha256"], "capability_manifest_sha256"
            ),
            _digest(identity["cohort_geometry_sha256"], "cohort_geometry_sha256"),
        )

    def _live_key_locked(self, key):
        return bool(
            isinstance(key, APCASTMemoKey)
            and key.apc_semantic_epoch == self._semantic_epoch
            and all(
                (state := self._scopes.get(scope)) is not None and state.epoch == epoch
                for scope, epoch in key.scope_epochs
            )
        )

    def get(self, identity):
        if not self._enabled:
            return None
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            key = self._key_locked(identity)
            table = (
                self._session
                if key is not None and key.scope_epochs
                else self._executor
            )
            entry = None if key is None else table.get(key)
            if entry is None:
                self._stats["misses"] += 1
                payload = None
            else:
                table.move_to_end(key)
                entry.expires_at = now + (
                    self._session_ttl
                    if key.scope_epochs
                    else self._limits["executor_ttl"]
                )
                payload = bytes(entry.payload)
                self._stats["hits"] += 1
        return None if payload is None else json.loads(payload)

    def put(self, identity, value):
        if not self._enabled:
            return False
        kind = identity.get("value_kind") if isinstance(identity, dict) else None
        safe = _validate_value(kind, value)
        payload = _canonical_bytes(safe)
        if len(payload) > self._limits["payload_bytes"]:
            with self._lock:
                self._stats["declines"] += 1
            return False
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            key = self._key_locked(identity)
            if key is None:
                self._stats["declines"] += 1
                return False
            table = self._session if key.scope_epochs else self._executor
            self._drop_entry_locked(key.tier, key)
            ttl = (
                self._session_ttl if key.scope_epochs else self._limits["executor_ttl"]
            )
            table[key] = _Entry(payload, len(payload), now + ttl)
            if key.tier == "session":
                self._session_bytes += len(payload)
            else:
                self._executor_bytes += len(payload)
            self._stats["stores"] += 1
            self._enforce_limits_locked(key.tier)
            return key in table

    def shadow_compare_put(self, identity, live_value):
        """Atomically compare and repair one metadata value.

        Live JSON validation, serialization, and hashing happen before the
        lock. The expected APC/scope epochs are part of ``identity``; if they
        changed while live metadata was computed, key resolution declines and
        nothing is published under the newer revision.
        """

        if not self._enabled:
            return {
                "outcome": "decline",
                "prior_present": False,
                "match": False,
                "prior_sha256": None,
                "live_sha256": None,
            }
        kind = identity.get("value_kind") if isinstance(identity, dict) else None
        safe = _validate_value(kind, live_value)
        payload = _canonical_bytes(safe)
        live_sha256 = hashlib.sha256(payload).hexdigest()
        if len(payload) > self._limits["payload_bytes"]:
            with self._lock:
                self._stats["declines"] += 1
            return {
                "outcome": "decline",
                "prior_present": False,
                "match": False,
                "prior_sha256": None,
                "live_sha256": live_sha256,
            }
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            key = self._key_locked(identity)
            if key is None:
                self._stats["declines"] += 1
                prior_payload = None
                outcome = "decline"
            else:
                table = self._session if key.scope_epochs else self._executor
                prior = table.get(key)
                prior_payload = None if prior is None else bytes(prior.payload)
                if prior is None:
                    self._stats["misses"] += 1
                    outcome = "miss"
                elif prior_payload == payload:
                    self._stats["hits"] += 1
                    self._stats["shadow_matches"] += 1
                    outcome = "hit"
                    table.move_to_end(key)
                    prior.expires_at = now + (
                        self._session_ttl
                        if key.scope_epochs
                        else self._limits["executor_ttl"]
                    )
                else:
                    self._stats["hits"] += 1
                    self._stats["shadow_drifts"] += 1
                    outcome = "shadow_mismatch"
                if outcome in {"miss", "shadow_mismatch"}:
                    self._drop_entry_locked(key.tier, key)
                    ttl = (
                        self._session_ttl
                        if key.scope_epochs
                        else self._limits["executor_ttl"]
                    )
                    table[key] = _Entry(payload, len(payload), now + ttl)
                    if key.tier == "session":
                        self._session_bytes += len(payload)
                    else:
                        self._executor_bytes += len(payload)
                    self._stats["stores"] += 1
                    self._enforce_limits_locked(key.tier)
        prior_sha256 = (
            None if prior_payload is None else hashlib.sha256(prior_payload).hexdigest()
        )
        return {
            "outcome": outcome,
            "prior_present": prior_payload is not None,
            "match": prior_payload == payload,
            "prior_sha256": prior_sha256,
            "live_sha256": live_sha256,
        }

    def _enforce_limits_locked(self, tier):
        if tier == "executor":
            while (
                len(self._executor) > self._limits["executor_entries"]
                or self._executor_bytes > self._limits["executor_bytes"]
            ):
                self._drop_entry_locked("executor", next(iter(self._executor)))
                self._stats["evictions"] += 1
            return
        while len(self._session) > self._limits["session_global_entries"]:
            self._drop_entry_locked("session", next(iter(self._session)))
            self._stats["evictions"] += 1
        for scope in tuple(self._scopes):
            while True:
                members = [
                    (key, entry)
                    for key, entry in self._session.items()
                    if any(member == scope for member, _epoch in key.scope_epochs)
                ]
                total = sum(entry.nbytes for _key, entry in members)
                if (
                    len(members) <= self._limits["session_entries"]
                    and total <= self._limits["session_bytes"]
                ):
                    break
                self._drop_entry_locked("session", members[0][0])
                self._stats["evictions"] += 1

    def advance(self, scopes, cause, *, purge=False):
        if not self._enabled:
            return False
        if cause not in _INVALIDATION_CAUSES:
            raise ValueError("unknown AST memo invalidation cause")
        scopes = tuple(dict.fromkeys(_digest(scope, "scope") for scope in scopes))
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            changed = False
            for scope in scopes:
                state = self._scopes.get(scope)
                if state is None:
                    continue
                self._drop_scope_entries_locked(scope)
                if purge:
                    self._drop_scope_debug_locked(scope)
                    del self._scopes[scope]
                    self._drop_scope_tags_locked(scope)
                else:
                    state.epoch += 1
                    state.expires_at = now + self._session_ttl
                changed = True
            if changed:
                self._invalidation_counts[cause] += 1
            return changed

    def clear(self, cause):
        if cause not in {"apcv2_clear", "apcv2_close", "model_invalidation"}:
            raise ValueError("invalid AST memo clear cause")
        with self._lock:
            receipt = {
                "executor_entries": len(self._executor),
                "session_entries": len(self._session),
                "scopes": len(self._scopes),
                "cause": cause,
            }
            self._executor.clear()
            self._session.clear()
            self._executor_bytes = self._session_bytes = 0
            self._scopes.clear()
            self._tag_scopes.clear()
            self._debug.clear()
            self._debug_bytes = 0
            self._semantic_epoch += 1
            if self._enabled:
                self._invalidation_counts[cause] += 1
            return receipt

    def debug_append(self, identity, record):
        if not self._debug_enabled:
            return None
        safe = _validate_debug_record(record)
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            key = self._key_locked(identity, debug=True)
            if key is None:
                return None
            scopes = tuple(scope for scope, _epoch in key.scope_epochs)
            sequence = self._debug_sequence
            self._debug_sequence += 1
        body = {
            "schema": DEBUG_SCHEMA,
            "sequence": sequence,
            "wall_time": self._wall_time(),
            "key_sha256": key.digest(),
            "scope_sha256": list(scopes),
            **safe,
        }
        payload = _canonical_bytes(body)
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            if not self._live_key_locked(key):
                return None
            if len(payload) > self._limits["debug_record_bytes"]:
                self._stats["declines"] += 1
                return None
            self._debug[sequence] = (scopes, payload, now + self._limits["debug_ttl"])
            self._debug_bytes += len(payload)
            self._stats["debug_records"] += 1
            self._enforce_debug_limits_locked(scopes)
            return sequence

    def _enforce_debug_limits_locked(self, scopes):
        for scope in scopes:
            while True:
                rows = [
                    (sequence, payload)
                    for sequence, (members, payload, _expires) in self._debug.items()
                    if scope in members
                ]
                nbytes = sum(len(payload) for _sequence, payload in rows)
                if (
                    len(rows) <= self._limits["debug_records"]
                    and nbytes <= self._limits["debug_scope_bytes"]
                ):
                    break
                _members, payload, _expires = self._debug.pop(rows[0][0])
                self._debug_bytes -= len(payload)
        while self._debug_bytes > self._limits["debug_global_bytes"]:
            _sequence, (_members, payload, _expires) = self._debug.popitem(last=False)
            self._debug_bytes -= len(payload)

    def debug_records(self, scopes=()):
        scopes = tuple(scopes)
        with self._lock:
            self._expire_locked(self._now())
            payloads = [
                bytes(payload)
                for members, payload, _expires in self._debug.values()
                if not scopes or any(scope in members for scope in scopes)
            ]
        return [json.loads(payload) for payload in payloads]

    def snapshot(self, scopes=()):
        scopes = tuple(scopes)
        if scopes and not self._enabled:
            return None
        with self._lock:
            self._expire_locked(self._now())
            epochs = []
            for scope in scopes:
                scope = _digest(scope, "scope")
                state = self._scopes.get(scope)
                if state is None:
                    return None
                epochs.append((scope, state.epoch))
            return {
                "schema": SCHEMA,
                "authority": "APCv2",
                "enabled": self._enabled,
                "debug_enabled": self._debug_enabled,
                "apc_semantic_epoch": self._semantic_epoch,
                "scope_epochs": epochs,
                "executor_entries": len(self._executor),
                "session_entries": len(self._session),
                "executor_bytes": self._executor_bytes,
                "session_bytes": self._session_bytes,
                "scopes": len(self._scopes),
                "tag_bindings": len(self._tag_scopes),
                "debug_records": len(self._debug),
                "debug_bytes": self._debug_bytes,
                "stats": dict(self._stats),
                "invalidations": dict(self._invalidation_counts),
            }
