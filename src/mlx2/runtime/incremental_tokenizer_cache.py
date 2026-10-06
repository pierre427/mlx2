"""Revision-bound incremental prompt tokenization.

The serving route is deliberately opt-in.  A selected adapter supplies a
renderer which accepts a private tokenizer snapshot.  Binding takes three
independent snapshots: one for rendering, one for full reference encoding,
and one for suffix encoding.  The adapter's ordinary tokenizer is never
redirected or modified.

The suffix proof is intentionally narrow.  Reuse starts only after a safe
AddedToken boundary, far enough behind the common-prefix edge that a newly
completed AddedToken cannot change the old match.  Unsupported tokenizer
geometry and origin-sensitive plain spans fail closed to the ordinary path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass

_CONTRACT = "mlx2.incremental-tokenizer-cache.v1"


@dataclass(frozen=True)
class PreparedPrompt:
    """Rendered input bound to one immutable tokenizer/template revision."""

    text: str
    plain_spans: tuple[tuple[int, int], ...]
    namespace: str
    revision: str
    epoch: int


@dataclass(frozen=True)
class _Entry:
    key: str
    revision: str
    epoch: int
    text: str
    ids: tuple[int, ...]
    marks: tuple[tuple[int, int], ...]
    plain_spans: tuple[tuple[int, int], ...]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _digest(value) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    # Comparing character-by-character in Python costs more than the suffix
    # encode at long context.  Chunk equality stays in the C string loop while
    # preserving character (not UTF-8 byte) offsets for tokenizer marks.
    index = 0
    chunk = 4096
    while index + chunk <= limit and (
        left[index : index + chunk] == right[index : index + chunk]
    ):
        index += chunk
    high = min(index + chunk, limit)
    while index < high and left[index] == right[index]:
        index += 1
    return index


def _span_disagreement_limit(old, new, default: int) -> int:
    for old_span, new_span in zip(old, new):
        if old_span != new_span:
            return min(default, old_span[0], new_span[0])
    if len(old) != len(new):
        extra = old[len(new) :] or new[len(old) :]
        return min(default, extra[0][0])
    return default


class IncrementalPromptTokenizerCache:
    """Bounded, exact, CPU tokenization candidate.

    ``max_entries == 0`` is the production default and does no tokenizer
    cloning.  A selected adapter must expose all of the following:

    * ``incremental_tokenizer_cache_supported is True``;
    * ``incremental_tokenizer_renderer_revision`` as a non-empty string;
    * ``render_incremental_prompt(frozen_hf_tokenizer, request) -> str``;
    * a ``TokenizerWrapper``-style ``tokenizer._tokenizer`` fast tokenizer.

    The optional ``incremental_plain_spans`` hook is accepted for future
    adapters, but non-empty spans currently fail closed: the frozen HF backend
    does not expose Strata's origin-sensitive special-token suppression.
    """

    def __init__(
        self,
        *,
        max_entries: int = 0,
        max_characters: int = 8 << 20,
        max_tokens: int = 1 << 20,
    ):
        for name, value in (
            ("max_entries", max_entries),
            ("max_characters", max_characters),
            ("max_tokens", max_tokens),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.max_entries = max_entries
        self.max_characters = max_characters
        self.max_tokens = max_tokens
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._characters = 0
        self._tokens = 0
        self._stats = Counter()
        self._epoch = 0
        self._epoch_used = False
        self._adapter = None
        self._source_wrapper = None
        self._renderer = None
        self._plain_spans = None
        self._render_tokenizer = None
        self._reference = None
        self._suffix = None
        self._added = {}
        self._max_added_length = 0
        self._revision = None
        self._validated = False
        self._refusal = None

    @property
    def enabled(self) -> bool:
        return bool(self.max_entries and self.max_characters and self.max_tokens)

    @property
    def selected(self) -> bool:
        with self._lock:
            return bool(self._revision is not None and self._refusal is None)

    def _reset_locked(self) -> None:
        self._entries.clear()
        self._characters = 0
        self._tokens = 0
        self._validated = False

    def _advance_epoch_locked(self) -> int:
        """Invalidate all in-flight lifecycle work without losing diagnostics."""

        self._epoch += 1
        self._epoch_used = False
        self._reset_locked()
        return self._epoch

    def _clear_binding_locked(self) -> None:
        self._adapter = None
        self._source_wrapper = None
        self._renderer = None
        self._plain_spans = None
        self._render_tokenizer = None
        self._reference = None
        self._suffix = None
        self._added = {}
        self._max_added_length = 0
        self._revision = None
        self._refusal = None

    def unbind(self) -> None:
        with self._lock:
            self._advance_epoch_locked()
            self._clear_binding_locked()

    def _refuse(
        self,
        reason: str,
        *,
        epoch: int | None = None,
        revision: str | None = None,
    ) -> bool:
        with self._lock:
            if (epoch is not None and epoch != self._epoch) or (
                revision is not None and revision != self._revision
            ):
                self._stats["stale_refusal_skips"] += 1
                return False
            self._reset_locked()
            self._refusal = reason
            self._stats[f"refused_{reason}"] += 1
        return False

    def bind(self, adapter) -> bool:
        """Bind immutable snapshots once, without changing ``adapter``."""

        with self._lock:
            bind_epoch = self._advance_epoch_locked()
            self._clear_binding_locked()
        if not self.enabled:
            return False
        if getattr(adapter, "incremental_tokenizer_cache_supported", None) is not True:
            return self._refuse("adapter_not_opted_in", epoch=bind_epoch)
        renderer_revision = getattr(
            adapter, "incremental_tokenizer_renderer_revision", None
        )
        renderer = getattr(adapter, "render_incremental_prompt", None)
        if (
            not isinstance(renderer_revision, str)
            or not renderer_revision
            or not callable(renderer)
        ):
            return self._refuse("renderer_contract", epoch=bind_epoch)
        wrapper = getattr(adapter, "tokenizer", None)
        if wrapper is None or getattr(wrapper, "_v1_encode_worker", None) is not None:
            return self._refuse("tokenizer_v1_selected", epoch=bind_epoch)
        if getattr(wrapper, "_chat_template", None) is not None:
            return self._refuse("callable_template", epoch=bind_epoch)
        tokenizer = getattr(wrapper, "_tokenizer", None)
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            return self._refuse("no_fast_backend", epoch=bind_epoch)

        # Serialisation is paid once at adapter bind, never on a request.  The
        # ordinary adapter remains live and independent of these snapshots.
        try:
            backend_json = backend.to_str()
            payload = json.loads(backend_json)
            if (
                payload.get("truncation") is not None
                or payload.get("padding") is not None
            ):
                return self._refuse("backend_padding_or_truncation", epoch=bind_epoch)
            model = payload.get("model") or {}
            if model.get("type") != "BPE":
                return self._refuse("non_bpe_model", epoch=bind_epoch)
            # Suffix reuse assumes deterministic encodes.  BPE dropout can
            # make both the cut prefix and tail vary between calls even when
            # the immutable tokenizer snapshot itself has not changed.
            if model.get("dropout") is not None:
                return self._refuse("bpe_dropout", epoch=bind_epoch)
            normalizer = payload.get("normalizer")
            if normalizer is not None and normalizer != {"type": "NFC"}:
                return self._refuse("normalizer", epoch=bind_epoch)
            added = {}
            added_contents = []
            for token in payload.get("added_tokens") or ():
                if any(token.get(flag) for flag in ("single_word", "lstrip", "rstrip")):
                    return self._refuse("added_token_boundary_flags", epoch=bind_epoch)
                if token.get("normalized", False):
                    return self._refuse("normalized_added_token", epoch=bind_epoch)
                content = token.get("content")
                if not isinstance(content, str) or not content:
                    return self._refuse("added_token_content", epoch=bind_epoch)
                added_contents.append(content)
                if token.get("special", False):
                    added[int(token["id"])] = content
            if not added:
                return self._refuse("no_safe_added_tokens", epoch=bind_epoch)

            from tokenizers import Tokenizer

            reference = Tokenizer.from_str(backend_json)
            suffix = Tokenizer.from_str(backend_json)
            render_tokenizer = copy.deepcopy(tokenizer)
            if render_tokenizer.backend_tokenizer.to_str() != backend_json:
                return self._refuse("renderer_clone_mismatch", epoch=bind_epoch)
        except Exception:  # noqa: BLE001 - optional route refuses any snapshot failure
            return self._refuse("snapshot_failed", epoch=bind_epoch)

        artifact = getattr(adapter, "identity", {}) or {}
        template_identity = {
            "chat_template": getattr(render_tokenizer, "chat_template", None),
            "special_tokens_map": getattr(
                render_tokenizer, "special_tokens_map_extended", None
            ),
            "renderer_revision": renderer_revision,
        }
        revision = _digest(
            {
                "contract": _CONTRACT,
                "adapter": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
                "artifact": artifact.get("fingerprint"),
                "backend_sha256": hashlib.sha256(backend_json.encode()).hexdigest(),
                "template": template_identity,
            }
        )
        with self._lock:
            if self._epoch != bind_epoch:
                self._stats["stale_bind_skips"] += 1
                return False
            self._adapter = adapter
            self._source_wrapper = wrapper
            self._renderer = renderer
            self._plain_spans = getattr(adapter, "incremental_plain_spans", None)
            self._render_tokenizer = render_tokenizer
            self._reference = reference
            self._suffix = suffix
            self._added = added
            # Marks remain special-only, but every AddedToken can change a
            # tokenization decision across the common-prefix edge.
            self._max_added_length = max(map(len, added_contents))
            self._revision = revision
            self._refusal = None
            self._stats["binds"] += 1
        return True

    def prepare(self, request: dict) -> PreparedPrompt | None:
        with self._lock:
            if self._revision is None or self._refusal is not None:
                return None
            source_wrapper = self._source_wrapper
            renderer = self._renderer
            tokenizer = self._render_tokenizer
            plain_hook = self._plain_spans
            revision = self._revision
            epoch = self._epoch
        if getattr(source_wrapper, "_v1_encode_worker", None) is not None:
            self._refuse(
                "tokenizer_v1_selected_after_bind",
                epoch=epoch,
                revision=revision,
            )
            return None
        text = renderer(tokenizer, request)
        if not isinstance(text, str):
            self._refuse("renderer_result", epoch=epoch, revision=revision)
            return None
        spans = () if plain_hook is None else tuple(plain_hook(request, text))
        last_end = 0
        normalized = []
        for span in spans:
            if (
                not isinstance(span, (tuple, list))
                or len(span) != 2
                or type(span[0]) is not int
                or type(span[1]) is not int
                or not last_end <= span[0] < span[1] <= len(text)
            ):
                self._refuse("plain_span_contract", epoch=epoch, revision=revision)
                return None
            last_end = span[1]
            normalized.append((span[0], span[1]))
        spans = tuple(normalized)
        namespace = _digest(
            {
                "revision": revision,
                "epoch": epoch,
                "rendered_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "plain_spans": spans,
            }
        )
        return PreparedPrompt(text, spans, namespace, revision, epoch)

    @staticmethod
    def _encode_full(prepared: PreparedPrompt, reference, added):
        encoding = reference.encode(prepared.text, add_special_tokens=False)
        ids = tuple(int(token) for token in encoding.ids)
        marks = []
        for index, (token, offset) in enumerate(zip(ids, encoding.offsets), 1):
            content = added.get(token)
            start, end = map(int, offset)
            if content is not None and prepared.text[start:end] == content:
                marks.append((end, index))
        return ids, tuple(marks)

    def _entry_key(self, prepared: PreparedPrompt) -> str:
        return _digest(
            {
                "revision": prepared.revision,
                "epoch": prepared.epoch,
                "text": prepared.text,
                "plain_spans": prepared.plain_spans,
            }
        )

    def _store(self, entry: _Entry) -> None:
        if len(entry.text) > self.max_characters or len(entry.ids) > self.max_tokens:
            with self._lock:
                self._stats["oversize_skips"] += 1
            return
        with self._lock:
            if (
                self._refusal is not None
                or entry.revision != self._revision
                or entry.epoch != self._epoch
            ):
                self._stats["stale_store_skips"] += 1
                return
            old = self._entries.pop(entry.key, None)
            if old is not None:
                self._characters -= len(old.text)
                self._tokens -= len(old.ids)
            self._entries[entry.key] = entry
            self._characters += len(entry.text)
            self._tokens += len(entry.ids)
            while (
                len(self._entries) > self.max_entries
                or self._characters > self.max_characters
                or self._tokens > self.max_tokens
            ):
                _, evicted = self._entries.popitem(last=False)
                self._characters -= len(evicted.text)
                self._tokens -= len(evicted.ids)
                self._stats["evictions"] += 1
            self._stats["stores"] += 1

    def _candidate(self, prepared: PreparedPrompt):
        with self._lock:
            entries = tuple(
                entry
                for entry in self._entries.values()
                if entry.revision == prepared.revision and entry.epoch == prepared.epoch
            )
            margin = max(self._max_added_length - 1, 0)
        best = None
        for entry in entries:
            common = _common_prefix_length(entry.text, prepared.text)
            limit = max(common - margin, 0)
            limit = _span_disagreement_limit(
                entry.plain_spans, prepared.plain_spans, limit
            )
            mark = None
            for candidate in entry.marks:
                if candidate[0] <= limit:
                    mark = candidate
                else:
                    break
            if mark is not None and (best is None or mark[0] > best[1][0]):
                best = (entry, mark)
        return best

    def tokenize(
        self, prepared: PreparedPrompt, ordinary_full
    ) -> tuple[list[int], dict]:
        """Tokenize ``prepared``; ``ordinary_full`` is used only for cold parity.

        It is deliberately evaluated without the cache lock.  Once the first
        exact parity gate succeeds, all selected results are produced by the
        two immutable, independent snapshots.
        """

        with self._lock:
            refused = (
                prepared.revision != self._revision
                or prepared.epoch != self._epoch
                or self._refusal is not None
            )
            validated = self._validated
            reference = self._reference
            suffix = self._suffix
            added = self._added
        if refused:
            return list(ordinary_full()), self._receipt(
                "ordinary_full",
                exact=True,
                tokenizer_revision=prepared.revision,
                lifecycle_epoch=prepared.epoch,
            )
        if prepared.plain_spans:
            with self._lock:
                self._stats["plain_span_fallbacks"] += 1
            return list(ordinary_full()), self._receipt(
                "plain_span_full",
                exact=True,
                tokenizer_revision=prepared.revision,
                lifecycle_epoch=prepared.epoch,
            )

        if not validated:
            ordinary = [int(token) for token in ordinary_full()]
            reference_ids, marks = self._encode_full(prepared, reference, added)
            if ordinary != list(reference_ids):
                self._refuse(
                    "ordinary_reference_mismatch",
                    epoch=prepared.epoch,
                    revision=prepared.revision,
                )
                return ordinary, self._receipt(
                    "refused_full",
                    exact=True,
                    tokenizer_revision=prepared.revision,
                    lifecycle_epoch=prepared.epoch,
                )
            with self._lock:
                if (
                    self._refusal is None
                    and prepared.revision == self._revision
                    and prepared.epoch == self._epoch
                ):
                    self._validated = True
                    self._stats["cold_validations"] += 1
                else:
                    self._stats["stale_validation_skips"] += 1
            self._store(
                _Entry(
                    self._entry_key(prepared),
                    prepared.revision,
                    prepared.epoch,
                    prepared.text,
                    reference_ids,
                    marks,
                    prepared.plain_spans,
                )
            )
            return ordinary, self._receipt(
                "ordinary_full_validation",
                exact=True,
                tokenizer_revision=prepared.revision,
                lifecycle_epoch=prepared.epoch,
            )

        candidate = self._candidate(prepared)
        if candidate is None:
            ids, marks = self._encode_full(prepared, reference, added)
            self._store(
                _Entry(
                    self._entry_key(prepared),
                    prepared.revision,
                    prepared.epoch,
                    prepared.text,
                    ids,
                    marks,
                    prepared.plain_spans,
                )
            )
            with self._lock:
                self._stats["bound_full"] += 1
            return list(ids), self._receipt(
                "bound_full",
                exact=True,
                tokenizer_revision=prepared.revision,
                lifecycle_epoch=prepared.epoch,
            )

        entry, (cut, reused_tokens) = candidate
        with self._lock:
            if entry.key in self._entries:
                self._entries.move_to_end(entry.key)
        suffix_encoding = suffix.encode(prepared.text[cut:], add_special_tokens=False)
        ids = entry.ids[:reused_tokens] + tuple(
            int(token) for token in suffix_encoding.ids
        )
        marks = tuple(mark for mark in entry.marks if mark[0] <= cut)
        suffix_marks = []
        for index, (token, offset) in enumerate(
            zip(suffix_encoding.ids, suffix_encoding.offsets), reused_tokens + 1
        ):
            content = added.get(int(token))
            start, end = map(int, offset)
            if (
                content is not None
                and prepared.text[cut + start : cut + end] == content
            ):
                suffix_marks.append((cut + end, index))
        self._store(
            _Entry(
                self._entry_key(prepared),
                prepared.revision,
                prepared.epoch,
                prepared.text,
                ids,
                marks + tuple(suffix_marks),
                prepared.plain_spans,
            )
        )
        with self._lock:
            if (
                self._refusal is None
                and prepared.revision == self._revision
                and prepared.epoch == self._epoch
            ):
                self._stats["incremental_hits"] += 1
                self._stats["reused_characters"] += cut
                self._stats["reused_tokens"] += reused_tokens
                self._epoch_used = True
            else:
                self._stats["stale_hit_skips"] += 1
        return list(ids), self._receipt(
            "incremental_hit",
            exact=True,
            tokenizer_revision=prepared.revision,
            lifecycle_epoch=prepared.epoch,
            reused_characters=cut,
            reused_tokens=reused_tokens,
        )

    def bound_full_tokens(self, prepared: PreparedPrompt) -> list[int]:
        """Independent immutable full reference used by tests and benches."""

        with self._lock:
            if (
                prepared.revision != self._revision
                or prepared.epoch != self._epoch
                or self._reference is None
            ):
                raise RuntimeError("prepared prompt is not bound to this cache")
            reference = self._reference
        return [
            int(token)
            for token in reference.encode(prepared.text, add_special_tokens=False).ids
        ]

    def is_current(self, prepared: PreparedPrompt) -> bool:
        """Whether ``prepared`` still names this exact lifecycle generation."""

        with self._lock:
            return bool(
                prepared.revision == self._revision and prepared.epoch == self._epoch
            )

    def host_hit_receipt(self, prepared: PreparedPrompt) -> dict | None:
        with self._lock:
            if (
                self._refusal is not None
                or prepared.revision != self._revision
                or prepared.epoch != self._epoch
            ):
                self._stats["stale_host_hit_skips"] += 1
                return None
            self._stats["host_prompt_cache_hits"] += 1
            self._epoch_used = True
        return self._receipt(
            "host_prompt_cache_hit",
            exact=True,
            tokenizer_revision=prepared.revision,
            lifecycle_epoch=prepared.epoch,
        )

    def _receipt(
        self,
        action: str,
        *,
        exact: bool,
        tokenizer_revision: str | None = None,
        lifecycle_epoch: int | None = None,
        **extra,
    ) -> dict:
        with self._lock:
            current_revision = self._revision
            current_epoch = self._epoch
            refusal = self._refusal
        revision = tokenizer_revision or current_revision
        epoch = current_epoch if lifecycle_epoch is None else lifecycle_epoch
        selected = (
            revision is not None
            and revision == current_revision
            and epoch == current_epoch
            and refusal is None
        )
        return {
            "schema": _CONTRACT,
            "implemented": True,
            "qualified": False,
            "selected": selected,
            "observed_used": selected
            and action in {"incremental_hit", "host_prompt_cache_hit"},
            "serving_qualified": False,
            "action": action,
            "exact": bool(exact),
            "tokenizer_revision": revision,
            "lifecycle_epoch": epoch,
            "refusal": refusal,
            **extra,
        }

    def clear(self) -> None:
        with self._lock:
            self._advance_epoch_locked()
            self._stats["clears"] += 1

    def status(self) -> dict:
        with self._lock:
            return {
                "schema": _CONTRACT,
                "implemented": True,
                "qualified": False,
                "selected": self._revision is not None and self._refusal is None,
                "observed_used": self._epoch_used,
                "serving_qualified": False,
                "default": "off",
                "tokenizer_revision": self._revision,
                "lifecycle_epoch": self._epoch,
                "source_mutation_policy": "isolated_by_immutable_bound_snapshots",
                "ordinary_reference": "independent_bound_full_snapshot",
                "refusal": self._refusal,
                "entries": len(self._entries),
                "characters": self._characters,
                "tokens": self._tokens,
                "bounds": {
                    "entries": self.max_entries,
                    "characters": self.max_characters,
                    "tokens": self.max_tokens,
                },
                **dict(self._stats),
            }
