# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
import contextlib
import copy
import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .batch_admission import AdmissionState, LinearStateCost, StateBudget
from .models import cache
from .models.cache import (
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
    TokenBuffer,
    record_state_checkpoints,
)
from .sample_utils import LaneRNG, draw_key
from .state_boundaries import BoundaryPurpose, StateBoundary
from .multi_lora import bind_lora_rows, clear_lora_rows

DEFAULT_MAX_TOKENS = 100
MTP_STARVED_BOUNDARIES_BEFORE_PLAIN = 8
# A lane deferred this many consecutive times by a live segmented width lock
# continues as ordinary decode beside the locked cohort instead of waiting for
# the cohort to drain (which took the whole cohort's decode: 78 s measured).
WIDTH_LOCK_DEFERRALS_BEFORE_PLAIN = 4
CACHE_STATE_EVAL_INTERVAL = 256
ALLOCATOR_RECLAIM_STEP_INTERVAL = 512
# Self-MTP emits several tokens per step, and each cycle frees verify-width KV
# and activation buffers the pool keeps. Measured on Qwen3.8-27B, 64K prompt,
# K=3 (3.15 tokens/step): the 512-step clear let the MLX cache reach 70 GiB
# over a 29 GiB working set; clearing every 256 emitted tokens held it at
# 1.6-10.5 GiB with no decode regression (runs varied 21-27 tok/s). Cf.
# Ollama v0.34.2.
ALLOCATOR_RECLAIM_MTP_TOKEN_INTERVAL = 256
_COUNTER_MAX = (1 << 63) - 1
_PERSISTENT_DECODE_INPUTS = "_mlx2_persistent_decode_inputs"
generation_stream = mx.new_thread_local_stream(mx.default_device())


def _bump_bounded_counter(counters, key, amount=1):
    counters[key] = min(
        _COUNTER_MAX, int(counters.get(key, 0)) + int(amount)
    )


def _crossed_counter_interval(previous: int, current: int, interval: int) -> bool:
    """Return whether monotonically increasing work crossed a boundary.

    A quotient transition cannot miss a cleanup point if a future scheduler
    changes a counter by more than one.  Today the decode-step counter advances
    once per round, preserving the established 512-round cleanup cadence.
    """
    if isinstance(previous, bool) or isinstance(current, bool):
        raise TypeError("token counters must be integers")
    if not isinstance(previous, int) or not isinstance(current, int):
        raise TypeError("token counters must be integers")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("interval must be a positive integer")
    if previous < 0 or current < previous:
        raise ValueError("token counters must be monotone and non-negative")
    return current // interval > previous // interval


def interior_checkpoint_positions(
    prompt_tokens: int, *, count: int, min_stride: int
) -> Tuple[int, ...]:
    """Return budgeted absolute prompt positions on a power-of-two lattice.

    Positions are independent of token content and prefill chunking.  The final
    prompt token remains the generation boundary, so candidates are strictly
    below ``P - 1``; the existing ``P - 1`` checkpoint is retained separately.
    Design references: omlx#3456 and vllm#52244.
    """
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (
        prompt_tokens, count, min_stride
    )):
        raise TypeError("interior checkpoint geometry must use integers")
    if prompt_tokens < 0 or count < 0 or min_stride < 1:
        raise ValueError("invalid interior checkpoint geometry")
    boundary = prompt_tokens - 1
    values = []
    position = min_stride
    while position < boundary:
        values.append(position)
        position *= 2
    return tuple(values[-count:]) if count else ()


def _resolve_kv_bits(kv_bits, key_bits, value_bits):
    if kv_bits is None and key_bits is None and (value_bits is None):
        return (None, None)
    key_bits = kv_bits if key_bits is None else key_bits
    value_bits = kv_bits if value_bits is None else value_bits
    if key_bits is None or value_bits is None:
        raise ValueError(
            "Both key and value bits are required; set --kv-bits as a fallback or provide both --kv-key-bits and --kv-value-bits."
        )
    return (key_bits, value_bits)


def maybe_quantize_kv_cache(
    prompt_cache,
    quantized_kv_start,
    kv_group_size,
    kv_bits,
    kv_key_bits=None,
    kv_value_bits=None,
    kv_rotate=False,
):
    (key_bits, value_bits) = _resolve_kv_bits(kv_bits, kv_key_bits, kv_value_bits)
    if key_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if isinstance(c, CacheList):
            leaves = list(c.caches)
            maybe_quantize_kv_cache(
                leaves,
                quantized_kv_start,
                kv_group_size,
                kv_bits,
                kv_key_bits=kv_key_bits,
                kv_value_bits=kv_value_bits,
                kv_rotate=kv_rotate,
            )
            c.caches = tuple(leaves)
        elif hasattr(c, "to_quantized"):
            reason = getattr(c, "kv_quantization_unsupported", None)
            if reason:
                raise ValueError(
                    f"KV cache quantization is not available for {type(c).__name__}. {reason}"
                )
            reached_start = c.offset >= quantized_kv_start
            if isinstance(reached_start, mx.array):
                reached_start = bool(mx.all(reached_start).item())
            if not reached_start:
                continue
            symmetric = key_bits == value_bits and (not kv_rotate)
            if isinstance(c, (RotatingKVCache, BatchRotatingKVCache)) and (
                not symmetric
            ):
                raise ValueError(
                    f"Asymmetric or Hadamard-rotated KV quantization is not supported for {type(c).__name__} (sliding-window/rotating cache). Use a symmetric --kv-bits without --kv-rotate, or quantize only plain KVCache instances."
                )
            try:
                if symmetric:
                    prompt_cache[e] = c.to_quantized(
                        group_size=kv_group_size, bits=key_bits
                    )
                else:
                    prompt_cache[e] = c.to_quantized(
                        group_size=kv_group_size,
                        bits=kv_bits if kv_bits is not None else key_bits,
                        key_bits=key_bits,
                        value_bits=value_bits,
                        rotate=kv_rotate,
                    )
            except NotImplementedError as exc:
                detail = str(exc).strip()
                raise ValueError(
                    f"KV cache quantization is not available for {type(c).__name__}."
                    + (f" {detail}" if detail else "")
                ) from exc


def _right_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max((len(p) for p in prompts))
    return mx.array([p + [0] * (max_length - len(p)) for p in prompts])


@dataclass
class BatchStats:
    """
    An data object to hold generation stats.

    Args:
        prompt_tokens (int): The number of prompt tokens processed.
        prompt_tps (float): The prompt processing tokens-per-second.
        prompt_time (float): The time in seconds spent in prompt processing.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        generation_time (float): The time in seconds spent in generation .
        peak_memory (float): The peak memory used so far in GB.
        effective_quantized_kv_start (int, optional): The
          ``quantized_kv_start`` the generator's caches were built with, when
          ``kv_bits`` is set (``None`` otherwise). ``BatchGenerator`` only
          supports immediate quantization in the batched path, so this is
          always ``0`` when populated -- recorded here so a harness cannot
          conflate it with the delayed-start schedule of the non-batched
          entry points.
    """

    prompt_tokens: int = 0
    prompt_tps: float = 0
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0
    generation_time: float = 0
    peak_memory: float = 0
    effective_quantized_kv_start: Optional[int] = None


def _merge_caches(caches):
    batch_cache = []
    if not caches:
        return batch_cache
    for i in range(len(caches[0])):
        if hasattr(caches[0][i], "merge"):
            batch_cache.append(caches[0][i].merge([c[i] for c in caches]))
        else:
            raise ValueError(
                f"{type(caches[0][i])} does not yet support batching with history"
            )
    return batch_cache


def _extend_cache(cache_a, cache_b):
    if not cache_a:
        return cache_b
    if not cache_b:
        return cache_a
    for ca, cb in zip(cache_a, cache_b):
        ca.extend(cb)
    return cache_a


def _build_trie(sequences):
    """Build an Aho-Corasick trie from the provided sequences

    See https://en.wikipedia.org/wiki/Aho–Corasick_algorithm .
    """
    trie = {}
    for idx, seq in enumerate(sequences):
        node = trie
        try:
            for tok in seq:
                node = node.setdefault(tok, {})
            node["__match__"] = (tuple(seq), idx)
        except TypeError:
            node = node.setdefault(seq, {})
            node["__match__"] = ((seq,), idx)
    queue = deque()
    for key, child in trie.items():
        if key == "__match__":
            continue
        child["__fail__"] = trie
        queue.append(child)
    while queue:
        parent = queue.popleft()
        for key, child in parent.items():
            if key in ("__fail__", "__match__"):
                continue
            queue.append(child)
            fail = parent["__fail__"]
            while key not in fail and fail is not trie:
                fail = fail["__fail__"]
            child["__fail__"] = fail[key] if key in fail else trie
            if "__match__" not in child and "__match__" in child["__fail__"]:
                child["__match__"] = child["__fail__"]["__match__"]
    return trie


def _step_trie(node, trie, x):
    """One step in the Aho-Corasick trie."""
    while x not in node and node is not trie:
        node = node["__fail__"]
    if x in node:
        node = node[x]
    return node


class StopSequenceMatcher:
    """Detect stop sequences in a stream of tokens using an Aho-Corasick trie.

    Any matched sequence signals stop. Used by the batch generator for EOS and
    stop word detection.
    """

    def __init__(self, stop_sequences=None):
        self._trie = _build_trie(stop_sequences) if stop_sequences else {}

    def __deepcopy__(self, memo):
        new = object.__new__(StopSequenceMatcher)
        new._trie = self._trie
        return new

    def make_state(self):
        return self._trie

    @staticmethod
    def match(state, trie, x):
        """Advance by one token. Returns (new_state, matched)."""
        node = _step_trie(state, trie, x)
        return (node, node.get("__match__") is not None)


BATCH_UID_HOOK = None


class PromptProcessingBatch:
    """
    A batch processor for prompt tokens with support for incremental processing.

    This class handles batched prompt processing, managing KV caches and preparing
    tokens for generation. It supports extending, filtering, and splitting batches.
    """

    @dataclass
    class Response:
        uid: int
        progress: tuple
        end_of_segment: bool
        end_of_prompt: bool

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        caches: List[List[Any]],
        tokens: Optional[List[List[int]]] = None,
        prefill_step_size: int = 2048,
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        fallback_sampler: Optional[Callable[[mx.array], mx.array]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        stop_matchers: Optional[List[StopSequenceMatcher]] = None,
        max_tokens: Optional[List[int]] = None,
        prompt_trim_rollback_tokens: int = 0,
        prepared_prompt_cache: Optional[List[Any]] = None,
        prefill_inputs: Optional[List[Optional[dict]]] = None,
    ):
        self.model = model
        self.uids = uids
        self.prompt_cache = (
            list(prepared_prompt_cache)
            if prepared_prompt_cache is not None
            else _merge_caches(caches)
        )
        if prepared_prompt_cache is not None:
            for plane in self.prompt_cache:
                rows = getattr(getattr(plane, "offset", None), "shape", ())
                if rows and int(rows[0]) != len(uids):
                    raise ValueError(
                        "prepared prompt-cache batch does not match UID count"
                    )
        self.prompt_trim_rollback_tokens = max(0, int(prompt_trim_rollback_tokens))
        self._restart_prompt_rollback()
        self.tokens = tokens if tokens is not None else [[] for _ in uids]
        self.prefill_step_size = prefill_step_size
        self.samplers = samplers if samplers is not None else []
        self.fallback_sampler = fallback_sampler or (lambda x: mx.argmax(x, axis=-1))
        self.logits_processors = (
            logits_processors if logits_processors is not None else []
        )
        self.stop_matchers = (
            stop_matchers
            if stop_matchers is not None
            else [StopSequenceMatcher()] * len(uids)
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else [DEFAULT_MAX_TOKENS] * len(self.uids)
        )
        self.prefill_inputs = (
            [dict(value) if value is not None else None for value in prefill_inputs]
            if prefill_inputs is not None
            else [None] * len(self.uids)
        )
        if len(self.prefill_inputs) != len(self.uids):
            raise ValueError("prefill_inputs must have one entry per sequence")
        self.persistent_inputs = []
        for value in self.prefill_inputs:
            persistent = (
                value.pop(_PERSISTENT_DECODE_INPUTS, None)
                if value is not None
                else None
            )
            if persistent is not None and (
                not isinstance(persistent, dict)
                or set(persistent) != {"deep_concept_memory"}
            ):
                raise ValueError(
                    "persistent decode inputs must contain only deep_concept_memory"
                )
            self.persistent_inputs.append(persistent)
        if any(value is not None for value in self.persistent_inputs) and len(self.uids) != 1:
            raise RuntimeError(
                "persistent concept decode must run at an isolated B=1 boundary"
            )

    def __len__(self):
        return len(self.uids)

    @property
    def has_persistent_inputs(self):
        return any(value is not None for value in self.persistent_inputs)

    def _restart_prompt_rollback(self):
        if self.prompt_trim_rollback_tokens > 0:
            for c in self.prompt_cache:
                c.start_speculation(self.prompt_trim_rollback_tokens)

    def _stop_prompt_rollback(self):
        if self.prompt_trim_rollback_tokens > 0:
            for c in self.prompt_cache:
                c.stop_speculation()

    def extract_cache(self, idx: int) -> List[Any]:
        return [c.extract(idx) for c in self.prompt_cache]

    def extend(self, batch):
        persistent = any(value is not None for value in self.persistent_inputs)
        incoming_persistent = any(
            value is not None for value in batch.persistent_inputs
        )
        if (persistent or incoming_persistent) and self.uids:
            raise RuntimeError(
                "persistent concept decode cannot share a prompt batch"
            )
        if incoming_persistent and len(batch.uids) != 1:
            raise RuntimeError(
                "persistent concept decode must run at an isolated B=1 boundary"
            )
        if not any(self.samplers):
            self.samplers = [None] * len(self.uids)
        if not any(self.logits_processors):
            self.logits_processors = [[] for _ in range(len(self.uids))]
        samplers = batch.samplers if any(batch.samplers) else [None] * len(batch.uids)
        logits_processors = (
            batch.logits_processors
            if any(batch.logits_processors)
            else [[] for _ in range(len(batch.uids))]
        )
        self.uids.extend(batch.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, batch.prompt_cache)
        self.prompt_trim_rollback_tokens = max(
            self.prompt_trim_rollback_tokens,
            getattr(batch, "prompt_trim_rollback_tokens", 0),
        )
        self._restart_prompt_rollback()
        self.tokens.extend(batch.tokens)
        self.samplers.extend(samplers)
        self.logits_processors.extend(logits_processors)
        self.max_tokens.extend(batch.max_tokens)
        self.stop_matchers.extend(batch.stop_matchers)
        self.prefill_inputs.extend(batch.prefill_inputs)
        self.persistent_inputs.extend(batch.persistent_inputs)

    def _copy(self, deep: bool = True):
        new_batch = self.__class__.__new__(self.__class__)
        new_batch.model = self.model
        new_batch.uids = list(self.uids)
        new_batch.prompt_trim_rollback_tokens = self.prompt_trim_rollback_tokens
        new_batch.prompt_cache = (
            copy.deepcopy(self.prompt_cache) if deep else self.prompt_cache
        )
        new_batch.tokens = list(self.tokens)
        new_batch.prefill_step_size = self.prefill_step_size
        new_batch.samplers = list(self.samplers)
        new_batch.fallback_sampler = self.fallback_sampler
        new_batch.logits_processors = list(self.logits_processors)
        new_batch.stop_matchers = list(self.stop_matchers)
        new_batch.max_tokens = list(self.max_tokens)
        new_batch.prefill_inputs = list(self.prefill_inputs)
        new_batch.persistent_inputs = list(self.persistent_inputs)
        return new_batch

    def split(self, indices: List[int]):
        indices = sorted(indices)
        indices_left = sorted(set(range(len(self.uids))) - set(indices))
        if not indices_left:
            new_batch = self._copy(deep=False)
            self.prompt_cache = []
            self.filter([])
            return new_batch
        new_batch = self._copy()
        self.filter(indices_left)
        new_batch.filter(indices)
        return new_batch

    def filter(self, keep: List[int]):
        self.uids = [self.uids[idx] for idx in keep]
        if not keep:
            self.prompt_cache.clear()
        else:
            for c in self.prompt_cache:
                c.filter(keep)
        self.tokens = [self.tokens[idx] for idx in keep]
        if any(self.samplers):
            self.samplers = [self.samplers[idx] for idx in keep]
        else:
            self.samplers = [None] * len(keep)
        if any(self.logits_processors):
            self.logits_processors = [self.logits_processors[idx] for idx in keep]
        else:
            self.logits_processors = [[] for _ in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self.stop_matchers = [self.stop_matchers[idx] for idx in keep]
        self.prefill_inputs = [self.prefill_inputs[idx] for idx in keep]
        self.persistent_inputs = [self.persistent_inputs[idx] for idx in keep]
        self._restart_prompt_rollback()

    def prompt(self, tokens: List[List[int]]):
        """
        Process prompt tokens through the model.

        Args:
            tokens: List of token sequences to process.
        """
        if len(self.uids) != len(tokens):
            raise ValueError("The batch length doesn't match the number of inputs")
        if not tokens:
            return
        for sti, ti in zip(self.tokens, tokens):
            sti += ti
        lengths = [len(p) for p in tokens]
        max_length = max(lengths)
        padding = [max_length - l for l in lengths]
        max_padding = max(padding)
        totals = [len(st) for st in self.tokens]
        bases = [t - l for (t, l) in zip(totals, lengths)]
        if max_padding > 0:
            tokens = _right_pad_prompts(tokens, max_length=max_length)
            for c in self.prompt_cache:
                c.prepare(lengths=lengths, right_padding=padding)
        else:
            tokens = mx.array(tokens)
        processed = 0
        # A few low-level cache/checkpoint tests construct this batch with
        # ``object.__new__``.  Preserve the historical text-only behavior for
        # those callers while normal construction still validates one entry
        # per sequence in ``__init__``.
        prefill_inputs = getattr(self, "prefill_inputs", [None] * len(self.uids))
        media_inputs = [value for value in prefill_inputs if value is not None]
        if media_inputs and (len(self.uids) != 1 or len(media_inputs) != 1):
            raise RuntimeError("multimodal prefill must run at an isolated B=1 boundary")
        prefill_prefetch = getattr(self.model, "prefill_prefetch_hook", None)
        prefill_prefetch = prefill_prefetch() if callable(prefill_prefetch) else None
        while tokens.shape[1] > 0:
            n_to_process = (
                tokens.shape[1]
                if media_inputs and processed == 0
                else min(self.prefill_step_size, tokens.shape[1])
            )
            if BATCH_UID_HOOK is not None:
                BATCH_UID_HOOK(list(self.uids))
            kwargs = media_inputs[0] if media_inputs and processed == 0 else {}
            # Concurrent multi-LoRA: publish per-row adapter slots for exactly
            # this forward (no-op unless a manager is attached to the model).
            lora_rows = bind_lora_rows(self.model, self.uids)
            try:
                self.model(tokens[:, :n_to_process], cache=self.prompt_cache, **kwargs)
            finally:
                clear_lora_rows(lora_rows)
            if kwargs:
                self.prefill_inputs[0] = None
            if prefill_prefetch is not None and tokens.shape[1] > n_to_process:
                context_start = max(0, n_to_process - prefill_prefetch.context_len)
                prefill_prefetch(
                    np.asarray(
                        tokens[:, n_to_process : n_to_process + self.prefill_step_size]
                    ),
                    np.asarray(tokens[:, context_start:n_to_process]),
                )
            mx.eval([c.state for c in self.prompt_cache])
            processed += n_to_process
            record_state_checkpoints(
                self.prompt_cache,
                [b + min(processed, l) for (b, l) in zip(bases, lengths)],
            )
            mx.clear_cache()
            tokens = tokens[:, n_to_process:]
        if max_padding > 0:
            for c in self.prompt_cache:
                c.finalize()
            mx.eval([c.state for c in self.prompt_cache])
            mx.clear_cache()
        record_state_checkpoints(self.prompt_cache, totals, force=True)

    def generate(self, tokens: List[List[int]]):
        """
        Transition from prompt processing to generation.

        Args:
            tokens: Final tokens for each sequence to start generation.

        Returns:
            A GenerationBatch ready for token generation.
        """
        if any((len(t) > 1 for t in tokens)):
            self.prompt([t[:-1] for t in tokens])
        last_token = mx.array([t[-1] for t in tokens])
        self._stop_prompt_rollback()
        generation = GenerationBatch(
            self.model,
            self.uids,
            last_token,
            self.prompt_cache,
            self.tokens,
            self.samplers,
            self.fallback_sampler,
            self.logits_processors,
            self.stop_matchers,
            self.max_tokens,
            persistent_inputs=self.persistent_inputs,
        )
        self.uids = []
        self.prompt_cache = []
        self.tokens = []
        self.samplers = []
        self.logits_processors = []
        self.max_tokens = []
        self.prefill_inputs = []
        self.persistent_inputs = []
        return generation

    @classmethod
    def empty(
        cls,
        model: nn.Module,
        fallback_sampler: Callable[[mx.array], mx.array],
        prefill_step_size: int = 2048,
        prompt_trim_rollback_tokens: int = 0,
    ):
        return cls(
            model=model,
            fallback_sampler=fallback_sampler,
            prefill_step_size=prefill_step_size,
            uids=[],
            caches=[],
            tokens=[],
            samplers=[],
            logits_processors=[],
            max_tokens=[],
            stop_matchers=[],
            prompt_trim_rollback_tokens=prompt_trim_rollback_tokens,
        )


class GenerationBatch:
    """
    A batched token generator that manages multiple sequences in parallel.

    This class handles the generation phase after prompt processing, managing
    KV caches, sampling, and stop sequence detection for multiple sequences.
    """

    @dataclass
    class Response:
        uid: int
        token: int
        logprobs: mx.array
        finish_reason: Optional[str]
        prompt_cache: Optional[List[Any]]
        all_tokens: Optional[List[int]]
        from_draft: bool = False
        mtp_state: Optional[Tuple[List[Any], mx.array]] = None
        lane_rng: Optional[LaneRNG] = None
        rng_draws: int = 0
        mtp_receipt: Optional[dict] = None
        execution_width: int = 1

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        inputs: mx.array,
        prompt_cache: List[Any],
        tokens: List[List[int]],
        samplers: Optional[List[Callable[[mx.array], mx.array]]],
        fallback_sampler: Callable[[mx.array], mx.array],
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ],
        stop_matchers: List[StopSequenceMatcher],
        max_tokens: List[int],
        persistent_inputs: Optional[List[Optional[dict]]] = None,
        route_receipts: Optional[List[Optional[dict]]] = None,
        lane_rngs: Optional[List[Optional[LaneRNG]]] = None,
    ):
        self.model = model
        self.uids = uids
        self.prompt_cache = prompt_cache
        self.tokens = tokens
        self.samplers = samplers
        self.fallback_sampler = fallback_sampler
        self.logits_processors = logits_processors
        self.stop_matchers = stop_matchers
        self.max_tokens = max_tokens
        self.route_receipts = list(route_receipts or [None] * len(uids))
        self.lane_rngs = list(lane_rngs or [None] * len(uids))
        if len(self.route_receipts) != len(self.uids):
            raise ValueError("route_receipts must match generation batch width")
        if len(self.lane_rngs) != len(self.uids):
            raise ValueError("lane_rngs must match generation batch width")
        self.persistent_inputs = (
            list(persistent_inputs)
            if persistent_inputs is not None
            else [None] * len(self.uids)
        )
        if len(self.persistent_inputs) != len(self.uids):
            raise ValueError("persistent_inputs must have one entry per sequence")
        if any(value is not None for value in self.persistent_inputs) and len(self.uids) != 1:
            raise RuntimeError(
                "persistent concept decode must run at an isolated B=1 boundary"
            )
        if self.samplers and len(self.samplers) != len(self.uids):
            raise ValueError("Insufficient number of samplers provided")
        if self.logits_processors and len(self.logits_processors) != len(self.uids):
            raise ValueError("Insufficient number of logits_processors provided")
        self._current_tokens = None
        self._current_logprobs = []
        self._decode_steps = 0
        self._next_tokens = inputs
        self._next_logprobs = []
        self._token_context = [TokenBuffer(t) for t in tokens]
        self._num_tokens = [0] * len(self.uids)
        self._matcher_states = [m.make_state() for m in stop_matchers]
        if self.uids:
            self._step()

    def __len__(self):
        return len(self.uids)

    @property
    def has_persistent_inputs(self):
        return any(value is not None for value in self.persistent_inputs)

    def extend(self, batch):
        """Extend this batch with another generation batch."""
        persistent = any(value is not None for value in self.persistent_inputs)
        incoming_persistent = any(
            value is not None for value in batch.persistent_inputs
        )
        if (persistent or incoming_persistent) and self.uids:
            raise RuntimeError(
                "persistent concept decode cannot share a generation batch"
            )
        if incoming_persistent and len(batch.uids) != 1:
            raise RuntimeError(
                "persistent concept decode must run at an isolated B=1 boundary"
            )
        self.uids.extend(batch.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, batch.prompt_cache)
        self.tokens.extend(batch.tokens)
        self.samplers.extend(batch.samplers)
        self.logits_processors.extend(batch.logits_processors)
        self.max_tokens.extend(batch.max_tokens)
        self.stop_matchers.extend(batch.stop_matchers)
        self.persistent_inputs.extend(batch.persistent_inputs)
        if self._current_tokens is None:
            self._current_tokens = batch._current_tokens
            self._current_logprobs = batch._current_logprobs
        elif batch._current_tokens is not None:
            self._current_tokens = mx.concatenate(
                [self._current_tokens, batch._current_tokens]
            )
            self._current_logprobs.extend(batch._current_logprobs)
        if self._next_tokens is None:
            self._next_tokens = batch._next_tokens
            self._next_logprobs = batch._next_logprobs
        elif batch._next_tokens is not None:
            self._next_tokens = mx.concatenate([self._next_tokens, batch._next_tokens])
            self._next_logprobs.extend(batch._next_logprobs)
        self._token_context.extend(batch._token_context)
        self._num_tokens.extend(batch._num_tokens)
        self._matcher_states.extend(batch._matcher_states)
        self.route_receipts.extend(batch.route_receipts)
        self.lane_rngs.extend(batch.lane_rngs)

    def _residual_steer(self, inputs):
        """Assemble this step's per-lane residual steering, if any lane asks.

        A lane's logits processor may expose ``residual_steer(input_token)``
        returning ``(layer, vector)`` (the thinking guard's calibrated commit
        direction).  Lanes that do not steer get a zero row, so one batched
        forward serves steered and unsteered lanes together.  Models without
        ``residual_taps`` simply never steer.
        """
        if not any(self.logits_processors):
            return None, None
        taps = getattr(getattr(self.model, "model", None), "residual_taps", None)
        if taps is None:
            return None, None
        rows, layer, tokens = {}, None, None
        for index, processors in enumerate(self.logits_processors):
            for processor in processors or ():
                ask = getattr(processor, "residual_steer", None)
                if ask is None:
                    continue
                if tokens is None:
                    tokens = inputs.tolist()
                request = ask(tokens[index])
                if request is not None and (layer is None or request[0] == layer):
                    layer = request[0]
                    rows[index] = request[1]
        if not rows:
            return None, None
        width = next(iter(rows.values())).shape[-1]
        zero = mx.zeros((width,), dtype=next(iter(rows.values())).dtype)
        stacked = mx.stack([rows.get(index, zero) for index in range(len(self.logits_processors))])
        return taps, (layer, stacked[:, None, :])

    def _persistent_step_inputs(self):
        if not self.persistent_inputs or self.persistent_inputs[0] is None:
            return {}
        persistent = self.persistent_inputs[0]
        memory = persistent["deep_concept_memory"]
        schedule = memory.get("decode_values")
        if schedule is None:
            return persistent
        if not isinstance(schedule, mx.array) or schedule.ndim != 2:
            raise ValueError("concept decode_values must be a rank-2 device array")
        if self._decode_steps >= schedule.shape[0]:
            return {}
        stepped = dict(memory)
        stepped.pop("decode_values")
        gates = stepped.pop("decode_gates", None)
        if gates is not None:
            if (
                not isinstance(gates, (list, tuple))
                or len(gates) != schedule.shape[0]
            ):
                raise ValueError(
                    "concept decode_gates must match the capsule schedule"
                )
            stepped["gate"] = float(gates[self._decode_steps])
        stepped["values"] = schedule[
            self._decode_steps : self._decode_steps + 1
        ]
        return {"deep_concept_memory": stepped}

    def _step(self) -> Tuple[List[int], List[mx.array]]:
        """
        Perform a single generation step.

        Returns:
            Tuple of token list and logprobs list.
        """
        self._current_tokens = self._next_tokens
        self._current_logprobs = self._next_logprobs
        inputs = self._current_tokens
        if BATCH_UID_HOOK is not None:
            BATCH_UID_HOOK(list(self.uids))
        taps, steer = self._residual_steer(inputs)
        if steer is not None:
            taps.steer = steer
        lora_rows = bind_lora_rows(self.model, self.uids)
        try:
            kwargs = self._persistent_step_inputs()
            logits = self.model(inputs[:, None], cache=self.prompt_cache, **kwargs)
        finally:
            clear_lora_rows(lora_rows)
            if steer is not None:
                taps.steer = None
        logits = logits[:, -1, :]
        token_context = []
        if any(self.logits_processors):
            token_context = [
                tc.update_and_fetch(inputs[i : i + 1])
                for (i, tc) in enumerate(self._token_context)
            ]
            processed_logits = []
            for e in range(len(self.uids)):
                sample_logits = logits[e : e + 1]
                for processor in self.logits_processors[e]:
                    sample_logits = processor(token_context[e], sample_logits)
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)
        logits = logits.astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if any(self.samplers):
            groups = {}
            order = []
            for e in range(len(self.uids)):
                sample_sampler = self.samplers[e] or self.fallback_sampler
                if getattr(sample_sampler, "batch_groupable", False):
                    key = id(sample_sampler)
                else:
                    key = (e,)
                if key not in groups:
                    groups[key] = (sample_sampler, [])
                    order.append(key)
                groups[key][1].append(e)
            if len(groups) == 1:
                ((sample_sampler, rows),) = groups.values()
                sampled = sample_sampler(logprobs)
            else:
                all_samples = [None] * len(self.uids)
                for key in order:
                    (sample_sampler, rows) = groups[key]
                    if len(rows) == 1:
                        group_sampled = sample_sampler(logprobs[rows[0] : rows[0] + 1])
                    else:
                        group_sampled = sample_sampler(logprobs[mx.array(rows)])
                    for j, e in enumerate(rows):
                        all_samples[e] = group_sampled[j : j + 1]
                sampled = mx.concatenate(all_samples, axis=0)
        else:
            sampled = self.fallback_sampler(logprobs)
        self._next_tokens = sampled
        self._next_logprobs = list(logprobs)
        self._decode_steps += 1
        eval_targets = [self._next_tokens, self._next_logprobs, token_context]
        if self._decode_steps % CACHE_STATE_EVAL_INTERVAL == 0:
            eval_targets.append([c.state for c in self.prompt_cache])
        mx.async_eval(*eval_targets)
        mx.eval(inputs, self._current_logprobs)
        inputs = inputs.tolist()
        for sti, ti in zip(self.tokens, inputs):
            sti.append(ti)
        return (inputs, self._current_logprobs)

    def extract_cache(self, idx: int) -> List[Any]:
        return [c.extract(idx) for c in self.prompt_cache]

    def filter(self, keep: List[int]):
        """Filter the batch to keep only the specified indices."""
        self.uids = [self.uids[idx] for idx in keep]
        if not keep:
            self.prompt_cache.clear()
        else:
            for c in self.prompt_cache:
                c.filter(keep)
        self.tokens = [self.tokens[idx] for idx in keep]
        if self.samplers:
            self.samplers = [self.samplers[idx] for idx in keep]
        if self.logits_processors:
            self.logits_processors = [self.logits_processors[idx] for idx in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self.stop_matchers = [self.stop_matchers[idx] for idx in keep]
        self.persistent_inputs = [self.persistent_inputs[idx] for idx in keep]
        self._next_tokens = self._next_tokens[keep] if keep else None
        self._next_logprobs = [self._next_logprobs[idx] for idx in keep]
        self._token_context = [self._token_context[idx] for idx in keep]
        self._num_tokens = [self._num_tokens[idx] for idx in keep]
        self._matcher_states = [self._matcher_states[idx] for idx in keep]
        self.route_receipts = [self.route_receipts[idx] for idx in keep]
        self.lane_rngs = [self.lane_rngs[idx] for idx in keep]

    def next(self) -> List[Response]:
        """
        Generate the next batch of tokens.

        Returns:
            List of Response objects for each sequence in the batch.
        """
        if not self.uids:
            return []
        (tokens, logprobs) = self._step()
        width = len(self.uids)
        keep = []
        responses = []
        for i in range(len(self.uids)):
            receipt = self.route_receipts[i]
            if receipt is not None:
                ordinary_widths = set(receipt.get("ordinary_compute_widths", ()))
                ordinary_widths.add(width)
                receipt["ordinary_compute_widths"] = sorted(ordinary_widths)
                observed = set(receipt.get("observed_compute_widths", ()))
                observed.add(width)
                receipt["observed_compute_widths"] = sorted(observed)
            finish_reason = None
            self._num_tokens[i] += 1
            if self._num_tokens[i] >= self.max_tokens[i]:
                finish_reason = "length"
            (self._matcher_states[i], matched) = StopSequenceMatcher.match(
                self._matcher_states[i], self.stop_matchers[i]._trie, tokens[i]
            )
            if matched:
                finish_reason = "stop"
            if finish_reason is not None:
                responses.append(
                    self.Response(
                        uid=self.uids[i],
                        token=tokens[i],
                        logprobs=logprobs[i],
                        finish_reason=finish_reason,
                        prompt_cache=self.extract_cache(i),
                        all_tokens=self.tokens[i],
                        lane_rng=self.lane_rngs[i],
                        rng_draws=(
                            self.lane_rngs[i].draws
                            if self.lane_rngs[i] is not None else 0
                        ),
                        mtp_receipt=receipt,
                    )
                )
            else:
                keep.append(i)
                responses.append(
                    self.Response(
                        uid=self.uids[i],
                        token=tokens[i],
                        logprobs=logprobs[i],
                        finish_reason=None,
                        prompt_cache=None,
                        all_tokens=None,
                        mtp_receipt=receipt,
                    )
                )
        if len(keep) < len(self.uids):
            self.filter(keep)
        for response in responses:
            response.execution_width = width
        return responses

    @classmethod
    def empty(cls, model: nn.Module, fallback_sampler: Callable[[mx.array], mx.array]):
        return cls(
            model=model,
            fallback_sampler=fallback_sampler,
            uids=[],
            inputs=mx.array([], dtype=mx.uint32),
            prompt_cache=[],
            tokens=[],
            samplers=[],
            logits_processors=[],
            max_tokens=[],
            stop_matchers=[],
        )


@dataclass
class _PausedMTPGenerationLane:
    detached: Any
    initial_output: Optional[Any]
    stop_matcher: StopSequenceMatcher
    matcher_state: Any
    num_tokens: int
    handoff_receipt: Optional[dict] = None


def _segment_aware_live_tip_enabled(config: Optional[Mapping[str, Any]]) -> bool:
    if config is None:
        return False
    from .segmented_self_mtp import segmented_self_mtp_enabled

    explicit = (
        config.get("segment_aware_live_tip")
        if "segment_aware_live_tip" in config
        else None
    )
    return segmented_self_mtp_enabled(explicit)


def _segment_aware_cohort_size(config: Optional[Mapping[str, Any]]) -> int:
    """Return the max width admitted before a segmented cohort starts.

    This is an admission-window target, not permission to widen a live
    true-batched cohort.  The latter remains rejected by MTPGenerationBatch's
    width lock.

    The default of 2 is a floor for callers that supply no value: every serving
    adapter sets this to ``max_lanes`` in ``execution_config``, so served
    routes never reach it, and no qualification receipt in the tree records a
    cohort size of 2 (796 record 20, 421 record 4, 37 record 1).  The earlier
    note here -- "preserves the qualified N=2 policy" -- described ``num_draft``,
    which f7076a6 did qualify at 2; that commit's receipts carry no cohort size
    at all and ran at ``max_lanes`` 4.

    Operators still set this window: adapters bind it to ``max_lanes``, so
    ``--max-lanes`` is the lever, and it defaults to 4.  Receipts in the tree
    record windows of 20, 4 and 1, so narrow windows are shipped
    configurations rather than test-only states, and mechanisms keyed to this
    width (segmented prefill ordering among them) behave differently at 1 or 4
    than at 16.
    """
    value = config.get("segment_aware_cohort_size", 2) if config is not None else 2
    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("segment_aware_cohort_size must be an integer") from error
    if value < 1:
        raise ValueError("segment_aware_cohort_size must be positive")
    return value


def _segmented_async_qsa_promotion_enabled(config: Optional[Mapping[str, Any]]) -> bool:
    """Return the default-off first-cycle segmented-to-physical policy."""
    if config is None or not _segment_aware_live_tip_enabled(config):
        return False
    if "segment_aware_async_qsa_promotion" in config:
        return bool(config["segment_aware_async_qsa_promotion"])
    return os.environ.get("MLX_LM_SEGMENTED_ASYNC_QSA_PROMOTION", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _segmented_async_qsa_min_remaining_tokens(
    config: Optional[Mapping[str, Any]],
) -> int:
    """Return the known-output budget that stays segmented."""
    value = (
        config.get("segment_aware_async_qsa_min_remaining_tokens")
        if config is not None
        and "segment_aware_async_qsa_min_remaining_tokens" in config
        else os.environ.get("MLX_LM_SEGMENTED_ASYNC_QSA_MIN_REMAINING_TOKENS", "16")
    )
    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "segment_aware_async_qsa_min_remaining_tokens must be an integer"
        ) from error
    if value < 0:
        raise ValueError(
            "segment_aware_async_qsa_min_remaining_tokens must be non-negative"
        )
    return value


def _segmented_async_qsa_promotion_for_budget(
    config: Optional[Mapping[str, Any]], remaining_tokens: int, *, record: bool = False
) -> bool:
    """Admit physical promotion only when its destination can be reused."""
    if not _segmented_async_qsa_promotion_enabled(config):
        return False
    remaining_tokens = max(0, int(remaining_tokens))
    cutoff = _segmented_async_qsa_min_remaining_tokens(config)
    admitted = remaining_tokens > cutoff
    if record:
        from .segmented_self_mtp import note_segmented_self_mtp

        note_segmented_self_mtp("async_qsa_budget_checks")
        note_segmented_self_mtp(
            "async_qsa_budget_remaining_tokens_cumulative", remaining_tokens
        )
        note_segmented_self_mtp("async_qsa_budget_cutoff_tokens_cumulative", cutoff)
        note_segmented_self_mtp(
            "async_qsa_budget_promotions"
            if admitted
            else "async_qsa_budget_retained_segmented"
        )
    return admitted


def _prefetch_known_mtp_tail(model, history, prompt, config) -> int:
    """Asynchronously stage file-backed PLE rows for a known MTP tail.

    APC gives lane preparation both the committed prefix and its uncached tail.
    Qwen4 can therefore hash and stage those PLE rows before the target catch-up
    starts. This is a performance hint only: unsupported models, empty tails,
    and submission failures all fall back to the ordinary foreground lookup.
    """
    if not config.get("prefetch_known_tail_ple", False) or not prompt:
        return 0
    from . import round_levers as _lv

    _lv.bump("ple_tail_prefetch_requests")
    prefetch = getattr(model, "ple_prefetch_verify", None)
    if not callable(prefetch):
        _lv.bump("ple_tail_prefetch_declined")
        return 0
    try:
        tables = int(prefetch(list(history), list(prompt)))
    except Exception as error:
        _lv.bump("ple_tail_prefetch_failures")
        logging.warning("Known-tail PLE prefetch declined: %s", error)
        return 0
    if tables <= 0:
        _lv.bump("ple_tail_prefetch_declined")
        return 0
    _lv.bump("ple_tail_prefetch_tables", tables)
    return tables


def _close_segmented_detached(detached: Any, *, release_cache: bool) -> None:
    """Release a detached lane's ledger and, when discarded, COW owner pin."""
    first_error = None
    transaction = getattr(detached, "segment_transaction", None)
    if transaction is not None:
        try:
            transaction.close()
        except BaseException as error:
            first_error = error
        detached.segment_transaction = None
    if release_cache:
        close_target = getattr(detached.caches.target, "close", None)
        if callable(close_target):
            try:
                close_target()
            except BaseException as error:
                if first_error is None:
                    first_error = error
    if first_error is not None:
        raise first_error


def _note_copy_draft_round(stats: Dict[str, Any], proposal: Any) -> None:
    """Sync-free host counters for one committed copy-draft-capable round."""
    for span, accepted, decision in zip(
        proposal.copy_spans, proposal.accepted_lengths, proposal.copy_decisions
    ):
        if decision == "off":
            continue
        if span:
            _bump_bounded_counter(stats, "self_mtp_copy_rounds")
            _bump_bounded_counter(stats, "self_mtp_copy_proposed_tokens", span)
            _bump_bounded_counter(
                stats, "self_mtp_copy_accepted_tokens", min(accepted, span)
            )
            if decision == "probe":
                _bump_bounded_counter(stats, "self_mtp_copy_probe_rounds")
        elif decision == "declined":
            _bump_bounded_counter(stats, "self_mtp_copy_gate_declines")
        else:
            _bump_bounded_counter(stats, "self_mtp_copy_lookup_misses")


class MTPGenerationBatch:
    """Scheduler wrapper for Agent A's batched self-MTP transaction."""

    Response = GenerationBatch.Response

    def __init__(
        self,
        model: nn.Module,
        detached_lanes: Sequence[Any],
        initial_outputs: Sequence[Any],
        stop_matchers: Sequence[StopSequenceMatcher],
        *,
        prepared_caches: Optional[Any] = None,
        segmented_live_tip: bool = False,
        async_qsa_promotion: bool = False,
        arm_async_qsa_promotion: bool = True,
        async_qsa_prequeue: Optional[Any] = None,
        adaptive_depth_policy: Optional[Any] = None,
        ordinary_handoff_policy: Optional[Any] = None,
        scheduler_stats: Optional[Dict[str, Any]] = None,
        acceptance_logger: Optional[Any] = None,
        mtp_admission: Optional[
            Callable[
                [Sequence[Tuple[int, int, int, bool, float]]],
                Mapping[int, Union[int, str]],
            ]
        ] = None,
    ):
        if len(detached_lanes) != len(initial_outputs):
            raise ValueError("initial_outputs must have one entry per MTP lane")
        if len(detached_lanes) != len(stop_matchers):
            raise ValueError("stop_matchers must have one entry per MTP lane")
        from .hybrid_speculative import (
            BatchedSelfMTPState,
            SegmentedSelfMTPState,
            SelfMTPCachePair,
            attach_prebatched_self_mtp_lanes,
            attach_segmented_self_mtp_lanes,
            attach_self_mtp_lanes,
        )

        self.model = model
        self.segmented_live_tip = bool(segmented_live_tip)
        self.async_qsa_promotion = bool(async_qsa_promotion)
        if self.segmented_live_tip and prepared_caches is not None:
            raise ValueError("segmented B1 state cannot accept a physical B2 cache")
        if self.segmented_live_tip:
            self.state = (
                attach_segmented_self_mtp_lanes(model, None, list(detached_lanes))
                if detached_lanes
                else SegmentedSelfMTPState([], [], [], 0)
            )
        elif prepared_caches is not None:
            if not isinstance(prepared_caches, SelfMTPCachePair):
                raise TypeError("prepared_caches must be a SelfMTPCachePair")
            self.state = attach_prebatched_self_mtp_lanes(
                model, detached_lanes, prepared_caches
            )
        elif detached_lanes:
            self.state = attach_self_mtp_lanes(model, None, list(detached_lanes))
        else:
            self.state = BatchedSelfMTPState([], SelfMTPCachePair([], []), 0)
        self.stop_matchers = list(stop_matchers)
        self._matcher_states = [m.make_state() for m in stop_matchers]
        self._num_tokens = [0] * len(detached_lanes)
        self._initial_outputs = list(initial_outputs)
        self._paused: Dict[int, _PausedMTPGenerationLane] = {}
        # Lanes the admission controller queued for *memory*; every other
        # paused lane is waiting on the scheduler (width lock) and must not be
        # judged by the serving layer's memory-progress watchdog.
        self._memory_queued: set = set()
        # Consecutive width-lock deferrals per lane; bounded by
        # WIDTH_LOCK_DEFERRALS_BEFORE_PLAIN before the lane goes plain.
        self._width_lock_deferrals: Dict[int, int] = {}
        self._plain_ready: List[_PausedMTPGenerationLane] = []
        self.mtp_admission = mtp_admission
        self._base_mtp_admission = mtp_admission
        self.adaptive_depth_policy = adaptive_depth_policy
        self.ordinary_handoff_policy = ordinary_handoff_policy
        self.acceptance_logger = acceptance_logger
        self.scheduler_stats = scheduler_stats if scheduler_stats is not None else {}
        self._ordinary_handoff_latched = False
        self._adaptive_admitted_cap = min(
            (int(lane.lane.num_draft) for lane in detached_lanes), default=0
        )
        self._atomic_cohort_failure = None
        self._async_qsa_ticket = None
        self._async_qsa_receipt = None
        self._async_qsa_receipts_by_uid = {}
        self._optional_reclaim_deadline = 0.0
        self._optional_reclaim_retry_at = 0.0
        self._observed_widths_by_uid = {}
        self._segmented_compute_width_locked = False
        self._async_qsa_pending = self.async_qsa_promotion and self.segmented_live_tip
        if async_qsa_prequeue is not None:
            from .segmented_physical_promotion import SegmentedPhysicalPromotionDeclined
            from .segmented_self_mtp import note_segmented_self_mtp

            if not self._async_qsa_pending:
                async_qsa_prequeue.cancel_and_drain()
                raise ValueError(
                    "async QSA prequeue requires segmented async promotion"
                )
            try:
                self._async_qsa_ticket = async_qsa_prequeue.bind(
                    self.state, note=note_segmented_self_mtp
                )
            except SegmentedPhysicalPromotionDeclined as error:
                async_qsa_prequeue.cancel_and_drain()
                note_segmented_self_mtp("async_qsa_prequeue_declined")
                logging.info("Async QSA prequeue declined at bind: %s", error)
                self._arm_async_qsa_promotion()
            else:
                note_segmented_self_mtp("async_qsa_prequeue_bound")
        elif arm_async_qsa_promotion:
            self._arm_async_qsa_promotion()

    def _arm_async_qsa_promotion(self) -> None:
        """Queue immutable QSA-base formation once, before the first cycle."""
        if (
            not self._async_qsa_pending
            or self._async_qsa_ticket is not None
            or (not self.segmented_live_tip)
            or (not self.state.lanes)
        ):
            return
        if any(getattr(cache, "supports_shared_qsa_suffix", False)
               for pair in self.state.row_caches for cache in pair.target):
            # The short-output shared-suffix route intentionally keeps an
            # immutable base and private tails. Physical promotion targets the
            # ordinary QSA ABI and would duplicate that base. Preserve the
            # admitted copy-avoiding route; other cohorts may still promote.
            from .segmented_self_mtp import note_segmented_self_mtp
            reason = "shared_suffix_retains_segmented_owner"
            note_segmented_self_mtp("async_qsa_promotion_declined_shared_suffix")
            self._async_qsa_receipts_by_uid.update(
                (lane.uid, {"selected": False, "reason": reason})
                for lane in self.state.lanes
            )
            self._decline_async_qsa_promotion(reason)
            return
        from .segmented_physical_promotion import (
            SegmentedPhysicalPromotionDeclined,
            begin_segmented_physical_promotion,
        )
        from .segmented_self_mtp import note_segmented_self_mtp

        note_segmented_self_mtp("async_qsa_promotion_requests")
        reserve_tail = max((int(lane.num_draft) + 1 for lane in self.state.lanes))
        try:
            self._async_qsa_ticket = begin_segmented_physical_promotion(
                self.state,
                reserve_tail=reserve_tail,
                stream=mx.new_stream(mx.gpu),
                note=note_segmented_self_mtp,
            )
        except SegmentedPhysicalPromotionDeclined as error:
            self._async_qsa_pending = False
            note_segmented_self_mtp("async_qsa_promotion_declined")
            logging.warning("Async QSA promotion declined at queue: %s", error)
        except Exception as error:
            self._async_qsa_pending = False
            note_segmented_self_mtp("async_qsa_promotion_failures")
            logging.warning("Async QSA promotion failed at queue: %s", error)
        else:
            note_segmented_self_mtp("async_qsa_promotion_queued")

    def _decline_async_qsa_promotion(self, reason: str, *, strict: bool = False) -> None:
        if self._async_qsa_ticket is None and (not self._async_qsa_pending):
            return
        from .segmented_self_mtp import note_segmented_self_mtp

        ticket = self._async_qsa_ticket
        if ticket is not None:
            try:
                drain = getattr(ticket, "cancel_and_drain", None)
                if callable(drain):
                    drain()
            except BaseException as error:
                note_segmented_self_mtp("async_qsa_promotion_failures")
                logging.warning("Async QSA promotion drain failed: %s", error)
                if strict:
                    raise
            else:
                if getattr(ticket, "stream", None) is not None:
                    note_segmented_self_mtp("device_synchronizations")
        self._async_qsa_ticket = None
        self._async_qsa_pending = False
        note_segmented_self_mtp("async_qsa_promotion_declined")
        logging.info("Async QSA promotion retained segmented state: %s", reason)

    def __len__(self):
        return len(self.state.lanes)

    @property
    def uids(self):
        return [lane.uid for lane in self.state.lanes]

    @property
    def prompt_cache(self):
        if self.segmented_live_tip:
            return [cache for pair in self.state.row_caches for cache in pair.target]
        return self.state.caches.target

    @property
    def cache_nbytes(self):
        if self.segmented_live_tip:
            total = sum(
                (
                    cache.nbytes
                    for pair in self.state.row_caches
                    for cache in pair.target + pair.draft
                )
            )
        else:
            total = sum(
                (
                    cache.nbytes
                    for cache in self.state.caches.target + self.state.caches.draft
                )
            )
        total += sum(
            (
                cache.nbytes
                for paused in self._paused.values()
                for cache in paused.detached.caches.target
                + paused.detached.caches.draft
            )
        )
        if self.segmented_live_tip:
            view = getattr(self.state, "_segmented_caches", None)
            if view is not None:
                total += sum((cache.nbytes for cache in view.target + view.draft))
        if self._async_qsa_ticket is not None:
            total += int(self._async_qsa_ticket.reserved_bytes)
        return total

    @property
    def tokens(self):
        return [self._prefix_tokens(lane) for lane in self.state.lanes]

    @property
    def max_tokens(self):
        return [lane.max_tokens for lane in self.state.lanes]

    @staticmethod
    def _prefix_tokens(lane):
        values = lane.token_prefix.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        return [int(token) for token in values]

    def mtp_cycle_state(self):
        """Rows ``(uid, context, depth, resident, cache_gib, pending_gib)``.

        ``cache_gib`` is allocated bytes and sets the growth rate.
        ``pending_gib`` is lazy memory that is not allocated yet.
        """
        gib = float(1 << 30)
        lane_count = max(len(self.state.lanes), 1)
        pending_per_lane = (
            int(getattr(self._async_qsa_ticket, "pending_bytes", self._async_qsa_ticket.reserved_bytes)) / lane_count / gib
            if self._async_qsa_ticket is not None
            else 0.0
        )
        if self.segmented_live_tip:
            overlap = 0
            view = getattr(self.state, "_segmented_caches", None)
            if view is not None:
                overlap += sum((cache.nbytes for cache in view.target + view.draft))
            overlap_per_lane = overlap / lane_count
            rows = [
                (
                    lane.uid,
                    len(self._prefix_tokens(lane)) + 1,
                    lane.num_draft,
                    True,
                    (
                        sum((cache.nbytes for cache in pair.target + pair.draft))
                        + overlap_per_lane
                    )
                    / gib,
                    pending_per_lane,
                )
                for (lane, pair) in zip(self.state.lanes, self.state.row_caches)
            ]
        else:
            active_bytes = sum(
                (
                    cache.nbytes
                    for cache in self.state.caches.target + self.state.caches.draft
                )
            )
            contexts = [len(self._prefix_tokens(lane)) + 1 for lane in self.state.lanes]
            gib_per_token = active_bytes / lane_count / max(contexts, default=1) / gib
            rows = [
                (
                    lane.uid,
                    context,
                    lane.num_draft,
                    True,
                    gib_per_token * context,
                    pending_per_lane,
                )
                for (lane, context) in zip(self.state.lanes, contexts)
            ]
        for uid, paused in self._paused.items():
            context = len(self._prefix_tokens(paused.detached.lane)) + 1
            paused_gib = (
                sum(
                    (
                        cache.nbytes
                        for cache in paused.detached.caches.target
                        + paused.detached.caches.draft
                    )
                )
                / gib
            )
            merge_gib = 0.0
            if not self.segmented_live_tip:
                active = [row[1] for row in rows if row[0] not in self._paused]
                width = max([context] + active)
                rate = max([paused_gib / context] + [row[4] / row[1] for row in rows])
                merge_gib = (len(active) + 1) * width * rate if active else 0.0
            rows.append(
                (
                    uid,
                    context,
                    paused.detached.lane.num_draft,
                    True,
                    paused_gib,
                    merge_gib,
                )
            )
        return rows

    def set_num_draft(self, depths: Union[int, Mapping[int, int]]):
        if self.state.proposal_open:
            raise RuntimeError("cannot change MTP depth while a proposal is open")
        if isinstance(depths, int):
            depths = {uid: depths for uid in self.uids}
        unknown = set(depths) - set(self.uids) - set(self._paused)
        if unknown:
            raise KeyError(f"unknown MTP lane uids: {sorted(unknown)}")
        desired = {
            int(depths.get(lane.uid, lane.num_draft)) for lane in self.state.lanes
        }
        if len(desired) > 1:
            raise ValueError("adaptive per-lane self-MTP depth is excluded")
        for lane in self.state.lanes:
            if lane.uid in depths:
                depth = int(depths[lane.uid])
                if depth < 1:
                    raise ValueError("MTP draft depth must be positive")
                lane.num_draft = depth
        for uid, paused in self._paused.items():
            if uid in depths:
                depth = int(depths[uid])
                if depth < 1:
                    raise ValueError("MTP draft depth must be positive")
                paused.detached.lane.num_draft = depth

    def _detach_packages(self, indices: Sequence[int]):
        from .hybrid_speculative import detach_self_mtp_lanes

        indices = sorted(set((int(i) for i in indices)))
        if not indices:
            return []
        if self._async_qsa_ticket is not None:
            self._decline_async_qsa_promotion("membership changed before first cycle")
        old_matchers = self.stop_matchers
        old_states = self._matcher_states
        old_counts = self._num_tokens
        old_initial = self._initial_outputs
        (self.state, detached) = detach_self_mtp_lanes(self.model, self.state, indices)
        packages = [
            _PausedMTPGenerationLane(
                lane,
                old_initial[idx],
                old_matchers[idx],
                old_states[idx],
                old_counts[idx],
            )
            for (idx, lane) in zip(indices, detached)
        ]
        keep = [i for i in range(len(old_matchers)) if i not in set(indices)]
        self.stop_matchers = [old_matchers[i] for i in keep]
        self._matcher_states = [old_states[i] for i in keep]
        self._num_tokens = [old_counts[i] for i in keep]
        self._initial_outputs = [old_initial[i] for i in keep]
        return packages

    def _attach_packages(self, packages: Sequence[_PausedMTPGenerationLane]):
        if not packages:
            return
        from .hybrid_speculative import (
            attach_segmented_self_mtp_lanes,
            attach_self_mtp_lanes,
        )

        already_handed_off = [
            package for package in packages if package.handoff_receipt is not None
        ]
        if already_handed_off:
            self._plain_ready.extend(already_handed_off)
            packages = [
                package for package in packages if package.handoff_receipt is None
            ]
            if not packages:
                return
        if self._ordinary_handoff_latched:
            self._mark_plain_handoff(
                packages,
                decision={"reason": "cohort_already_handed_off"},
                projected_width=len(packages),
            )
            self._plain_ready.extend(packages)
            return

        if self.state.lanes:
            depths = {lane.num_draft for lane in self.state.lanes}
            if len(depths) != 1:
                raise RuntimeError("active self-MTP lanes have mixed draft depths")
            depth = depths.pop()
            for package in packages:
                package.detached.lane.num_draft = depth
        if (
            self.segmented_live_tip
            and self._segmented_compute_width_locked
            and self.state.lanes
        ):
            if self.ordinary_handoff_policy is not None:
                projected_width = (
                    len(self.state.lanes) + len(self._paused) + len(packages)
                )
                decision = self.ordinary_handoff_policy.decision(
                    width=projected_width,
                    width_locked=True,
                    adaptive=self.adaptive_depth_policy,
                )
                if decision is not None:
                    self._handoff_all_to_plain(
                        packages,
                        decision=decision,
                        projected_width=projected_width,
                    )
                    return
            from .segmented_self_mtp import note_segmented_self_mtp

            deferred = 0
            for package in packages:
                uid = package.detached.lane.uid
                if uid in self._paused:
                    raise ValueError(f"duplicate deferred self-MTP lane uid {uid}")
                count = self._width_lock_deferrals.get(uid, 0) + 1
                if count > WIDTH_LOCK_DEFERRALS_BEFORE_PLAIN:
                    # Bounded wait: decode this lane as ordinary beside the
                    # locked cohort rather than parking it for the cohort's
                    # whole lifetime.  The lock itself is untouched.
                    self._width_lock_deferrals.pop(uid, None)
                    self._plain_ready.append(package)
                    note_segmented_self_mtp("width_lock_plain_fallbacks")
                    continue
                self._width_lock_deferrals[uid] = count
                self._paused[uid] = package
                deferred += 1
            if deferred:
                note_segmented_self_mtp("live_width_change_deferrals", deferred)
            return
        for package in packages:
            self._width_lock_deferrals.pop(package.detached.lane.uid, None)
        attach = (
            attach_segmented_self_mtp_lanes
            if self.segmented_live_tip
            else attach_self_mtp_lanes
        )
        # An attestation is valid only before the prepared first output has
        # been emitted. Resumed or diverged rows never inherit this authority.
        for package in packages:
            package.detached.shared_qsa_prefix_id = (
                getattr(package.detached.lane, "_initial_shared_prefix_attestation", None)
                if package.initial_output is not None else None
            )
        state = None if self.segmented_live_tip and not self.state.lanes else self.state
        self.state = attach(
            self.model, state, [package.detached for package in packages]
        )
        self.stop_matchers.extend((package.stop_matcher for package in packages))
        self._matcher_states.extend((package.matcher_state for package in packages))
        self._num_tokens.extend((package.num_tokens for package in packages))
        self._initial_outputs.extend((package.initial_output for package in packages))
        self._arm_async_qsa_promotion()

    def _mark_plain_handoff(self, packages, *, decision, projected_width):
        receipt = {
            "selected": True,
            "engaged": True,
            "one_way": True,
            "reason": str(decision["reason"]),
            "projected_width": int(projected_width),
            "policy": self.ordinary_handoff_policy.as_dict(),
            "decision": dict(decision),
        }
        for package in packages:
            lane = package.detached.lane
            stats = dict(vars(lane.stats))
            stats["total_emitted"] = int(lane.stats.total_emitted)
            stats["draft_acceptance"] = float(lane.stats.draft_accepted) / max(
                int(lane.stats.draft_proposed), 1
            )
            lane_receipt = dict(receipt)
            lane_receipt["mtp_observed_compute_widths_before_handoff"] = sorted(
                self._observed_widths_by_uid.pop(lane.uid, set())
            )
            lane_receipt["num_draft_before_handoff"] = int(lane.num_draft)
            lane_receipt["committed_tokens_before_handoff"] = int(
                package.num_tokens
            )
            lane_receipt["prepared_initial_token_pending"] = (
                package.initial_output is not None
            )
            lane_receipt["prepared_initial_token_from_draft"] = (
                None
                if package.initial_output is None
                else bool(package.initial_output.from_draft)
            )
            lane_receipt["stats_before_handoff"] = stats
            lane_receipt["async_qsa_promotion_before_handoff"] = (
                self._async_qsa_receipts_by_uid.pop(lane.uid, None)
            )
            package.handoff_receipt = lane_receipt
            # Draft and segmented transaction ownership end here. The target
            # cache remains alive for exact ordinary continuation.
            _close_segmented_detached(package.detached, release_cache=False)

    def _handoff_all_to_plain(self, joining=(), *, decision, projected_width):
        """Detach admitted lanes at one closed boundary.

        Memory-queued lanes keep waiting for admission. They are marked for
        plain continuation and can never re-enter MTP when later admitted.
        """
        if self.state.proposal_open:
            raise RuntimeError("cannot hand off MTP while a proposal is open")
        ready = self._detach_packages(range(len(self.state.lanes)))
        waiting = []
        for uid, package in list(self._paused.items()):
            if uid in self._memory_queued:
                waiting.append(package)
            else:
                ready.append(package)
                self._paused.pop(uid)
        # Joining packages reached this seam only after the current admission
        # decision approved them. Do not charge or queue them a second time.
        ready.extend(joining)
        selected = [*ready, *waiting]
        self._ordinary_handoff_latched = True
        self._segmented_compute_width_locked = False
        self._width_lock_deferrals.clear()
        self._mark_plain_handoff(
            selected, decision=decision, projected_width=projected_width
        )
        self._plain_ready.extend(ready)
        _bump_bounded_counter(self.scheduler_stats, "mtp_ordinary_handoff_events")
        _bump_bounded_counter(
            self.scheduler_stats, "mtp_ordinary_handoff_lanes", len(selected)
        )
        reason = str(decision["reason"])
        _bump_bounded_counter(
            self.scheduler_stats, f"mtp_ordinary_handoff_{reason}"
        )

    def _maybe_handoff_active_cohort(self) -> bool:
        if (
            self._ordinary_handoff_latched
            or self.ordinary_handoff_policy is None
            or not self.state.lanes
        ):
            return False
        width = len(self.state.lanes) + len(self._paused)
        decision = self.ordinary_handoff_policy.decision(
            width=width,
            width_locked=False,
            adaptive=self.adaptive_depth_policy,
        )
        if decision is None:
            return False
        self._handoff_all_to_plain(decision=decision, projected_width=width)
        return True

    def release_ordinary_handoff_latch(self) -> bool:
        """Release safely once admitted migrated work is below the threshold."""
        if self.state.lanes or self._plain_ready:
            return False
        if any(
            package.handoff_receipt is None
            for package in self._paused.values()
        ):
            return False
        self._ordinary_handoff_latched = False
        return True

    @property
    def has_deferred_lanes(self) -> bool:
        """Whether fixed-width segmented admission is holding arrivals."""
        return bool(self._paused)

    def _apply_admission(self):
        if self.mtp_admission is None:
            return True
        ticket = self._async_qsa_ticket
        def partition_rows():
            rows = tuple(self.mtp_cycle_state())
            handed_off = {
                uid
                for uid, package in self._paused.items()
                if package.handoff_receipt is not None
            }
            return (
                tuple(row for row in rows if row[0] not in handed_off),
                tuple(row for row in rows if row[0] in handed_off),
            )
        preview = getattr(self.mtp_admission, "preview", None)
        def memory_constrained():
            native_rows, _ = partition_rows()
            if not native_rows:
                return False
            decision = preview(native_rows)
            if decision.stage == "full":
                return False
            ceiling = getattr(self.mtp_admission, "ceiling", None)
            if callable(ceiling):
                possible = ceiling(native_rows)
                return (decision.modes, decision.draft_depths) != (possible.modes, possible.draft_depths)
            return True
        if ticket is not None and callable(preview):
            if memory_constrained():
                from .segmented_self_mtp import note_segmented_self_mtp
                # The queued copy may already be resident. Settle before
                # charging it again as future allocation; do not let optional
                # promotion force APC eviction or a lane migration first.
                ticket.settle_for_admission()
                note_segmented_self_mtp("async_qsa_admission_settled")
                if memory_constrained():
                    note_segmented_self_mtp("async_qsa_promotion_declined_memory")
                    self._async_qsa_receipts_by_uid.update(
                        (lane.uid, {"selected": False, "reason": "memory_pressure_retains_segmented_owner"})
                        for lane in self.state.lanes
                    )
                    self._decline_async_qsa_promotion("memory_pressure_retains_segmented_owner", strict=True)
                    # Drop our local reference too before the callback reclaims.
                    ticket = None
                    self._optional_reclaim_deadline = time.monotonic() + 2.0
                    self._optional_reclaim_retry_at = 0.0
        if self._optional_reclaim_deadline and callable(preview):
            now = time.monotonic()
            if now >= self._optional_reclaim_retry_at:
                reclaim = getattr(self.mtp_admission, "reclaim", None)
                if callable(reclaim):
                    reclaim()
                self._optional_reclaim_retry_at = now + 0.25
            if not memory_constrained():
                self._optional_reclaim_deadline = 0.0
            elif now < self._optional_reclaim_deadline:
                # Host accounting can lag completed device reclamation. Yield
                # this poll without detaching lanes or evicting APC; no bytes
                # are credited until the live measurement actually recovers.
                notify = getattr(self.mtp_admission, "defer_optional", None)
                if callable(notify):
                    notify(tuple(self.mtp_cycle_state()))
                return False
            else:
                self._optional_reclaim_deadline = 0.0
        native_rows, handoff_rows = partition_rows()
        decisions = (
            dict(self.mtp_admission(native_rows) or {})
            if native_rows else {}
        )
        if handoff_rows:
            at_depth = getattr(self.mtp_admission, "at_depth", None)
            ordinary_admission = at_depth(0) if callable(at_depth) else (
                self.mtp_admission
            )
            ordinary = dict(ordinary_admission(handoff_rows) or {})
            decisions.update(
                {
                    uid: "queue" if value == "queue" else "plain"
                    for uid, value in ordinary.items()
                }
            )
        if self.adaptive_depth_policy is not None:
            admitted_depths = [
                int(value)
                for value in decisions.values()
                if isinstance(value, int)
                and not isinstance(value, bool)
                and int(value) > 0
            ]
            if admitted_depths:
                self._adaptive_admitted_cap = min(admitted_depths)
        if getattr(self.mtp_admission, "atomic_cohort", False) and native_rows:
            cohort_uids = tuple(row[0] for row in native_rows)
            depths = {
                decisions.get(uid)
                for uid in cohort_uids
                if isinstance(decisions.get(uid), int)
                and not isinstance(decisions.get(uid), bool)
                and decisions.get(uid) > 0
            }
            if any(
                not isinstance(decisions.get(uid), int)
                or isinstance(decisions.get(uid), bool)
                or decisions.get(uid) <= 0
                for uid in cohort_uids
            ) or len(depths) != 1:
                lanes = list(self.state.lanes) + [
                    package.detached.lane for package in self._paused.values()
                ]
                cohort = next(
                    (
                        getattr(lane, "_batch_cohort", None)
                        for lane in lanes
                        if getattr(lane, "_batch_cohort", None) is not None
                    ),
                    {},
                )
                self._atomic_cohort_failure = {
                    "uids": cohort_uids,
                    "cohort": dict(cohort),
                    "reason": (
                        "declared batch cohort could not remain wholly admitted "
                        "for segmented speculative decoding"
                    ),
                }
                return False
        drop = [
            i
            for (i, uid) in enumerate(self.uids)
            if decisions.get(uid) in ("queue", "plain")
        ]
        for package in self._detach_packages(drop):
            mode = decisions.get(package.detached.lane.uid)
            if mode == "plain":
                self._plain_ready.append(package)
            else:
                self._paused[package.detached.lane.uid] = package
                self._memory_queued.add(package.detached.lane.uid)
        self.set_num_draft(
            {uid: value for (uid, value) in decisions.items() if isinstance(value, int)}
        )
        joining = []
        for uid, value in decisions.items():
            if isinstance(value, int) and uid in self._paused:
                joining.append(self._paused.pop(uid))
                self._memory_queued.discard(uid)
            elif value == "plain" and uid in self._paused:
                self._plain_ready.append(self._paused.pop(uid))
                self._memory_queued.discard(uid)
        self._attach_packages(joining)
        return True

    def scheduler_waiting_uids(self):
        """Paused lanes waiting on the scheduler rather than on memory."""
        return [uid for uid in self._paused if uid not in self._memory_queued]

    def take_atomic_cohort_failure(self):
        failure = self._atomic_cohort_failure
        self._atomic_cohort_failure = None
        return failure

    def demote_oldest_paused_to_plain(self) -> int:
        """Move the oldest paused lane to plain decode and return its uid."""
        uid = next(iter(self._paused))
        self._plain_ready.append(self._paused.pop(uid))
        return uid

    def take_plain_fallbacks(self):
        (ready, self._plain_ready) = (self._plain_ready, [])
        self._normalize_empty_segmented_admission()
        return ready

    def _normalize_empty_segmented_admission(self):
        """Restore configured segmented admission at an ownership-free seam."""
        if not self.state.lanes and not self._paused and not self._plain_ready:
            self.mtp_admission = getattr(
                self, "_base_mtp_admission", self.mtp_admission
            )
        if (
            not self.async_qsa_promotion
            or self.state.lanes
            or self._paused
            or self._plain_ready
        ):
            return
        from .hybrid_speculative import SegmentedSelfMTPState

        epoch = int(getattr(self.state, "membership_epoch", 0))
        self.state = SegmentedSelfMTPState([], [], [], epoch)
        self.segmented_live_tip = True
        self._segmented_compute_width_locked = False
        self._async_qsa_pending = True
        self._async_qsa_receipt = None
        self._async_qsa_receipts_by_uid.clear()

    def close(self):
        first_error = None
        if self._async_qsa_ticket is not None:
            self._decline_async_qsa_promotion("batch closed before promotion")
        else:
            self._async_qsa_pending = False
        if self.segmented_live_tip:
            from .hybrid_speculative import close_segmented_self_mtp_state

            try:
                close_segmented_self_mtp_state(self.state)
            except BaseException as error:
                first_error = error
        for package in [*self._paused.values(), *self._plain_ready]:
            try:
                _close_segmented_detached(package.detached, release_cache=True)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        self._paused.clear()
        self._plain_ready.clear()
        if first_error is not None:
            raise first_error

    def extend(self, batch):
        if not isinstance(batch, MTPGenerationBatch):
            raise TypeError("MTPGenerationBatch can extend only another MTP batch")
        if self.segmented_live_tip != batch.segmented_live_tip:
            raise ValueError("cannot mix segmented and physical self-MTP batches")
        was_empty = not self.state.lanes and not self._paused and not self._plain_ready
        if was_empty:
            # The incoming cohort owns admission semantics for its lifetime.
            # Otherwise this long-lived empty batch would immediately replace
            # an atomic lower-k decision with its generic fewer-lanes policy.
            self.mtp_admission = batch.mtp_admission
            incoming_adaptive = batch.adaptive_depth_policy
            current_adaptive = self.adaptive_depth_policy
            if (
                current_adaptive is None
                or incoming_adaptive is None
                or current_adaptive.max_depth != incoming_adaptive.max_depth
                or current_adaptive.ewma_alpha != incoming_adaptive.ewma_alpha
                or current_adaptive.shrink_gate != incoming_adaptive.shrink_gate
                or current_adaptive.grow_gate != incoming_adaptive.grow_gate
                or current_adaptive.loss_rounds != incoming_adaptive.loss_rounds
                or current_adaptive.gain_rounds != incoming_adaptive.gain_rounds
                or current_adaptive.park_rounds != incoming_adaptive.park_rounds
                or current_adaptive.goodput_alpha
                != incoming_adaptive.goodput_alpha
                or current_adaptive.goodput_hysteresis
                != incoming_adaptive.goodput_hysteresis
                or current_adaptive.min_samples_per_depth
                != incoming_adaptive.min_samples_per_depth
                or current_adaptive.goodput_window
                != incoming_adaptive.goodput_window
                or current_adaptive.probe_interval
                != incoming_adaptive.probe_interval
                or current_adaptive.stale_rounds != incoming_adaptive.stale_rounds
            ):
                self.adaptive_depth_policy = incoming_adaptive
            self._adaptive_admitted_cap = batch._adaptive_admitted_cap
        if getattr(self, "acceptance_logger", None) is None:
            self.acceptance_logger = getattr(batch, "acceptance_logger", None)
        if self.segmented_live_tip and was_empty:
            # Output-budget policy belongs to the incoming cohort. A completed
            # short response must not permanently disable future promotion.
            self.async_qsa_promotion = batch.async_qsa_promotion
            self._async_qsa_pending = batch.async_qsa_promotion
        if self.async_qsa_promotion != batch.async_qsa_promotion:
            if not self.segmented_live_tip:
                # Physical destination.  With the same policy the only way the
                # flags differ is a cohort that already promoted (flag kept so
                # it re-arms segmented admission once empty) receiving a
                # short-output arrival whose budget gate returned False.  The
                # arrival joins a physical batch, where the flag governs
                # nothing, so the resident cohort's flag stays authoritative;
                # an arrival into an empty batch owns admission semantics.
                if was_empty:
                    self.async_qsa_promotion = batch.async_qsa_promotion
                    self._async_qsa_pending = False
                if batch._async_qsa_ticket is not None:
                    batch._decline_async_qsa_promotion(
                        "joined an already-physical cohort"
                    )
                from .segmented_self_mtp import note_segmented_self_mtp

                note_segmented_self_mtp("physical_join_flag_reconciled")
            else:
                if self._async_qsa_ticket is not None:
                    self._decline_async_qsa_promotion(
                        "short-output cohort joined before promotion"
                    )
                if batch._async_qsa_ticket is not None:
                    batch._decline_async_qsa_promotion(
                        "short-output cohort already owns admission"
                    )
                self.async_qsa_promotion = (
                    self.async_qsa_promotion and batch.async_qsa_promotion
                )
                self._async_qsa_pending = (
                    self.async_qsa_promotion and self.segmented_live_tip
                )
        packages = batch._detach_packages(range(len(batch)))
        packages.extend(batch._paused.values())
        batch._paused.clear()
        if self.mtp_admission is None:
            self._attach_packages(packages)
            return
        for package in packages:
            uid = package.detached.lane.uid
            if uid in self._paused or uid in self.uids:
                raise ValueError(f"duplicate self-MTP lane uid {uid}")
            self._paused[uid] = package
        self._apply_admission()

    def extract_cache(self, idx: int) -> List[Any]:
        if self.state.proposal_open:
            raise RuntimeError("cannot extract an MTP cache during a proposal")
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        packages = self._detach_packages(range(len(self)))
        result = copy.deepcopy(packages[idx].detached.caches.target)
        self._attach_packages(packages)
        return result

    def filter(self, keep: List[int]):
        keep = sorted(set(keep))
        drop = [i for i in range(len(self)) if i not in set(keep)]
        dropped_uids = [self.uids[i] for i in drop]
        for package in self._detach_packages(drop):
            _close_segmented_detached(package.detached, release_cache=True)
        for uid in dropped_uids:
            self._async_qsa_receipts_by_uid.pop(uid, None)
        self._normalize_empty_segmented_admission()

    def extract_uid(self, uid: int):
        if uid in self.uids:
            idx = self.uids.index(uid)
            return (self.extract_cache(idx), self.tokens[idx])
        if uid in self._paused:
            lane = self._paused[uid].detached
            return (copy.deepcopy(lane.caches.target), self._prefix_tokens(lane.lane))
        raise KeyError(uid)

    def remove_uids(self, uids):
        requested = set(uids)
        drop = [i for (i, uid) in enumerate(self.uids) if uid in requested]
        packages = self._detach_packages(drop)
        for package in packages:
            _close_segmented_detached(package.detached, release_cache=True)
        for uid in requested:
            package = self._paused.pop(uid, None)
            self._memory_queued.discard(uid)
            self._width_lock_deferrals.pop(uid, None)
            if package is not None:
                _close_segmented_detached(package.detached, release_cache=True)
        for uid in requested:
            self._async_qsa_receipts_by_uid.pop(uid, None)
            self._observed_widths_by_uid.pop(uid, None)
        self._normalize_empty_segmented_admission()

    @staticmethod
    def _finish_reason(
        token: int,
        count: int,
        maximum: int,
        matcher_state,
        matcher: StopSequenceMatcher,
    ):
        reason = "length" if count >= maximum else None
        (matcher_state, matched) = StopSequenceMatcher.match(
            matcher_state, matcher._trie, token
        )
        if matched:
            reason = "stop"
        return (matcher_state, reason)

    def _complete_responses(self, terminal_indices, last_response_by_index):
        packages = self._detach_packages(terminal_indices)
        for idx, package in zip(sorted(terminal_indices), packages):
            response = last_response_by_index[idx]
            lane = package.detached.lane
            response.prompt_cache = package.detached.caches.target
            response.all_tokens = self._prefix_tokens(lane)
            response.mtp_state = (package.detached.caches.draft, lane.seed_h)
            response.lane_rng = lane.rng
            response.rng_draws = lane.rng.draws if lane.rng is not None else 0
            stats = dict(vars(lane.stats))
            # ``vars`` hands out the live dicts; a delivered receipt must not
            # keep mutating behind the caller.
            for _hist in ("verify_span_hist", "verify_accept_hist"):
                if isinstance(stats.get(_hist), dict):
                    stats[_hist] = dict(stats[_hist])
            stats["total_emitted"] = int(lane.stats.total_emitted)
            stats["draft_acceptance"] = float(lane.stats.draft_accepted) / max(
                int(lane.stats.draft_proposed), 1
            )
            response.mtp_receipt = {
                "route": "segmented_self_mtp"
                if self.segmented_live_tip
                else "continuous_batched_self_mtp",
                "observed_compute_widths": sorted(
                    self._observed_widths_by_uid.pop(lane.uid, {1})
                ),
                "num_draft": int(lane.num_draft),
                "requested_num_draft": getattr(
                    lane, "_requested_num_draft", None
                ),
                "admission_stage": getattr(
                    lane, "_cohort_admission_stage", None
                ),
                "accept_rule": str(lane.accept_rule),
                "sampling_temperature": float(lane.sampling_temp),
                "verification": (
                    "fly"
                    if lane.fly_verification is not None
                    and lane.fly_verification.enabled
                    and not lane.logits_processors
                    and lane.sampling_temp <= 0
                    else "exact"
                ),
                "verification_parameters": (
                    lane.fly_verification.as_dict()
                    if lane.fly_verification is not None
                    and lane.fly_verification.enabled
                    and not lane.logits_processors
                    and lane.sampling_temp <= 0
                    else None
                ),
                "relaxed_accepts": int(lane.relaxed_accepts),
                **(
                    {"copy_draft": lane.copy_draft.receipt()}
                    if getattr(lane, "copy_draft", None) is not None
                    else {}
                ),
                "fly_disabled": bool(
                    lane.fly_verification is not None
                    and lane.fly_verification.enabled
                    and (lane.logits_processors or lane.sampling_temp > 0)
                ),
                "fly_disabled_reason": (
                    "logits_processors"
                    if lane.fly_verification is not None
                    and lane.fly_verification.enabled
                    and lane.logits_processors
                    else "sampling_or_block_verifier"
                    if lane.fly_verification is not None
                    and lane.fly_verification.enabled
                    and lane.sampling_temp > 0
                    else None
                ),
                "stats": stats,
                "async_qsa_promotion": self._async_qsa_receipts_by_uid.pop(
                    lane.uid, None
                ),
                "adaptive_depth": (
                    None
                    if self.adaptive_depth_policy is None
                    else {
                        "selected": True,
                        "max_depth": int(self.adaptive_depth_policy.max_depth),
                        "current": int(self.adaptive_depth_policy.current_depth),
                        "acceptance_ewma": self.adaptive_depth_policy.acceptance_ewma,
                        "policy": {
                            "ewma_alpha": float(
                                self.adaptive_depth_policy.ewma_alpha
                            ),
                            "shrink_gate": float(
                                self.adaptive_depth_policy.shrink_gate
                            ),
                            "grow_gate": float(
                                self.adaptive_depth_policy.grow_gate
                            ),
                            "loss_rounds": int(
                                self.adaptive_depth_policy.loss_rounds
                            ),
                            "gain_rounds": int(
                                self.adaptive_depth_policy.gain_rounds
                            ),
                            "park_rounds": int(
                                self.adaptive_depth_policy.park_rounds
                            ),
                            "goodput_alpha": float(
                                self.adaptive_depth_policy.goodput_alpha
                            ),
                            "goodput_hysteresis": float(
                                self.adaptive_depth_policy.goodput_hysteresis
                            ),
                            "min_samples_per_depth": int(
                                self.adaptive_depth_policy.min_samples_per_depth
                            ),
                            "goodput_window": int(
                                self.adaptive_depth_policy.goodput_window
                            ),
                            "probe_interval": int(
                                self.adaptive_depth_policy.probe_interval
                            ),
                            "stale_rounds": int(
                                self.adaptive_depth_policy.stale_rounds
                            ),
                        },
                        "counters": dict(self.adaptive_depth_policy.counters),
                        "cost_model": self.adaptive_depth_policy.diagnostics(),
                        "trace": list(self.adaptive_depth_policy.trace),
                    }
                ),
            }
            _close_segmented_detached(package.detached, release_cache=False)
        self._normalize_empty_segmented_admission()

    def _emit_initial(self):
        responses = []
        terminal = []
        last = {}
        for i, output in enumerate(self._initial_outputs):
            if output is None:
                continue
            self._initial_outputs[i] = None
            self._num_tokens[i] += 1
            (self._matcher_states[i], reason) = self._finish_reason(
                output.token,
                self._num_tokens[i],
                self.max_tokens[i],
                self._matcher_states[i],
                self.stop_matchers[i],
            )
            response = self.Response(
                uid=self.uids[i],
                token=output.token,
                logprobs=output.logprobs,
                finish_reason=reason,
                prompt_cache=None,
                all_tokens=None,
                from_draft=output.from_draft,
            )
            responses.append(response)
            if reason is not None:
                terminal.append(i)
                last[i] = response
        if terminal:
            self._complete_responses(terminal, last)
        return responses

    def _set_confidence_probes(self) -> None:
        """Attach this cycle's draft-confidence probe to every lane (or none)."""
        probe = None
        policy_probe = getattr(self.adaptive_depth_policy, "probe", None)
        if callable(policy_probe):
            probe = policy_probe()
        elif getattr(self, "acceptance_logger", None) is not None:
            probe = self.acceptance_logger.probe
        for lane in self.state.lanes:
            lane.confidence_probe = probe

    def _observe_draft_confidence(self, proposal) -> None:
        from .mtp_confidence import rows_from_proposal

        rows = rows_from_proposal(proposal, proposal._old_curs)
        if not rows:
            return
        _bump_bounded_counter(self.scheduler_stats, "mtp_confidence_feature_cycles")
        observe = getattr(self.adaptive_depth_policy, "observe_confidence", None)
        if callable(observe):
            observe(rows)
        logger = getattr(self, "acceptance_logger", None)
        if logger is not None:
            written = logger.record(rows)
            if written:
                _bump_bounded_counter(
                    self.scheduler_stats, "mtp_acceptance_log_records", written
                )

    def next(self) -> List[Response]:
        if any((output is not None for output in self._initial_outputs)):
            return self._emit_initial()
        if not self._apply_admission():
            return []
        if any((output is not None for output in self._initial_outputs)):
            return self._emit_initial()
        if self._maybe_handoff_active_cohort():
            return []
        if not self.state.lanes:
            self._segmented_compute_width_locked = False
        if not self.state.lanes and self._paused and (self.mtp_admission is None):
            deferred = list(self._paused.values())
            self._paused.clear()
            self._attach_packages(deferred)
            # A deferred arrival still owns its prepared first token; emit it
            # before the lane's first proposal round commits later tokens.
            if any((output is not None for output in self._initial_outputs)):
                return self._emit_initial()
        if not self.state.lanes:
            return []
        from .hybrid_speculative import (
            ZeroDepthFastUnavailable,
            advance_batched_self_mtp_zero,
            abort_batched_self_mtp,
            commit_batched_self_mtp,
            propose_batched_self_mtp,
        )

        cohort_width = len(self.state.lanes)
        if self.adaptive_depth_policy is not None:
            cohort_depth = self.adaptive_depth_policy.select(
                admitted_cap=self._adaptive_admitted_cap,
                width=cohort_width,
            )
            # This boundary is closed: admission has completed and no proposal
            # owns cache state.  Apply one depth to the whole physical cohort.
            # K=0 is the hybrid core's exact ordinary target round; it keeps
            # target/draft ownership intact for a later bounded re-entry probe.
            for lane in self.state.lanes:
                lane.num_draft = cohort_depth
        self._set_confidence_probes()
        zero_depth = all(
            min(lane.num_draft, max(lane.max_tokens - lane.ntoks - 1, 0)) == 0
            for lane in self.state.lanes
        )
        round_started = (
            time.perf_counter() if self.adaptive_depth_policy is not None else None
        )
        if zero_depth:
            from .hybrid_speculative import copy_draft_candidate_pending

            # A K=0 cohort still verifies a copied span when one exists.
            zero_depth = not copy_draft_candidate_pending(self.state)
        if zero_depth:
            try:
                proposal = advance_batched_self_mtp_zero(self.model, self.state)
            except ZeroDepthFastUnavailable:
                proposal = propose_batched_self_mtp(self.model, self.state)
        else:
            proposal = propose_batched_self_mtp(self.model, self.state)
        relaxed_accepts = sum(proposal.relaxed_accepts or ())
        true_batched_segmented = bool(
            proposal.true_batched
            or (
                getattr(self, "segmented_live_tip", False)
                and getattr(self.state, "_batched_state", None) is not None
            )
        )
        compute_width = (
            len(self.state.lanes)
            if true_batched_segmented or not self.segmented_live_tip
            else 1
        )
        for lane in self.state.lanes:
            self._observed_widths_by_uid.setdefault(lane.uid, set()).add(
                compute_width
            )
        try:
            emitted_counts = []
            terminal = []
            responses = []
            last = {}
            for i, outputs in enumerate(proposal.outputs):
                emitted = 0
                is_terminal = False
                for output in outputs:
                    emitted += 1
                    self._num_tokens[i] += 1
                    (self._matcher_states[i], reason) = self._finish_reason(
                        output.token,
                        self._num_tokens[i],
                        self.max_tokens[i],
                        self._matcher_states[i],
                        self.stop_matchers[i],
                    )
                    response = self.Response(
                        uid=self.uids[i],
                        token=output.token,
                        logprobs=output.logprobs,
                        finish_reason=reason,
                        prompt_cache=None,
                        all_tokens=None,
                        from_draft=output.from_draft,
                    )
                    responses.append(response)
                    last[i] = response
                    if reason is not None:
                        is_terminal = True
                        break
                emitted_counts.append(emitted)
                terminal.append(is_terminal)
            if not proposal.zero_fast_path:
                commit_batched_self_mtp(
                    self.state,
                    proposal,
                    emitted_counts=emitted_counts,
                    terminal=terminal,
                )
            if self.adaptive_depth_policy is not None:
                # Head-only evidence: copied spans never steer MTP depth.
                copy_spans = proposal.copy_spans or (0,) * len(
                    proposal.draft_depths
                )
                cost = {}
                if not any(copy_spans):
                    cost = {
                        "committed": sum(emitted_counts),
                        "elapsed_seconds": max(
                            time.perf_counter() - round_started, 1e-12
                        ),
                    }
                self.adaptive_depth_policy.observe(
                    sum(
                        depth
                        for (depth, copy) in zip(proposal.draft_depths, copy_spans)
                        if not copy
                    ),
                    sum(
                        accepted
                        for (accepted, copy) in zip(
                            proposal.accepted_lengths, copy_spans
                        )
                        if not copy
                    ),
                    width=cohort_width,
                    observed_compute_width=compute_width,
                    **cost,
                )
            if proposal.draft_features:
                self._observe_draft_confidence(proposal)
            if true_batched_segmented:
                self._segmented_compute_width_locked = True
            ticket = self._async_qsa_ticket
            if ticket is not None:
                if any(terminal):
                    self._decline_async_qsa_promotion(
                        "a lane terminated in the first segmented cycle"
                    )
                else:
                    from dataclasses import asdict
                    from .segmented_physical_promotion import (
                        SegmentedPhysicalPromotionDeclined,
                    )
                    from .segmented_self_mtp import note_segmented_self_mtp

                    try:
                        (self.state, receipt) = ticket.finish()
                    except SegmentedPhysicalPromotionDeclined as error:
                        self._decline_async_qsa_promotion(str(error))
                    except BaseException:
                        note_segmented_self_mtp("async_qsa_promotion_failures")
                        raise
                    else:
                        self._async_qsa_ticket = None
                        self._async_qsa_pending = False
                        self.segmented_live_tip = False
                        self._async_qsa_receipt = asdict(receipt)
                        self._async_qsa_receipts_by_uid.update(
                            (
                                (lane.uid, dict(self._async_qsa_receipt))
                                for lane in self.state.lanes
                            )
                        )
                        note_segmented_self_mtp("async_qsa_promotion_engaged")
                        note_segmented_self_mtp(
                            "async_qsa_promotion_reserved_bytes", receipt.reserved_bytes
                        )
                        note_segmented_self_mtp(
                            "async_qsa_promotion_patched_bytes", receipt.patched_bytes
                        )
                        note_segmented_self_mtp(
                            "async_qsa_promotion_wait_ns", receipt.stream_wait_ns
                        )
                        if getattr(ticket, "stream", None) is not None:
                            note_segmented_self_mtp("device_synchronizations")
        except BaseException as error:
            if self.state.proposal_open:
                abort_batched_self_mtp(self.state, proposal, cause=error)
            raise
        # Publish mechanism accounting only after the proposal/response commit
        # boundary.  An interrupted proposal must not become engagement proof.
        if proposal.zero_fast_path:
            _bump_bounded_counter(
                self.scheduler_stats, "self_mtp_zero_fast_rounds"
            )
            _bump_bounded_counter(
                self.scheduler_stats,
                "self_mtp_zero_draft_forwards_skipped",
                len(proposal.lane_uids),
            )
            _bump_bounded_counter(
                self.scheduler_stats,
                "self_mtp_zero_proposal_roundtrips_skipped",
                len(proposal.lane_uids),
            )
        if relaxed_accepts:
            _bump_bounded_counter(
                self.scheduler_stats, "fly_relaxed_accepts", relaxed_accepts
            )
        if proposal.copy_decisions:
            _note_copy_draft_round(self.scheduler_stats, proposal)
        terminal_indices = [i for (i, value) in enumerate(terminal) if value]
        if terminal_indices:
            self._complete_responses(terminal_indices, last)
        return responses

    @classmethod
    def empty(
        cls,
        model,
        *,
        mtp_admission=None,
        segmented_live_tip: bool = False,
        async_qsa_promotion: bool = False,
        adaptive_depth_policy: Optional[Any] = None,
        ordinary_handoff_policy: Optional[Any] = None,
        scheduler_stats: Optional[Dict[str, Any]] = None,
    ):
        return cls(
            model,
            [],
            [],
            [],
            mtp_admission=mtp_admission,
            segmented_live_tip=segmented_live_tip,
            async_qsa_promotion=async_qsa_promotion,
            adaptive_depth_policy=adaptive_depth_policy,
            ordinary_handoff_policy=ordinary_handoff_policy,
            scheduler_stats=scheduler_stats,
        )


def _share_qsa_indices_for_config(config: dict) -> bool:
    """Select shared QSA at the parent route, with an explicit opt-out."""
    return bool(config.get("share_qsa_indices", True))


class BatchGenerator:
    """
    A batch generator implements continuous batching.

    This class provides automatic management of prompt processing and generation
    batches, handling the transition between the two.

    It also allows for segmented prompt processing which guarantees that the
    generator will stop at these boundaries when processing an input.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        max_tokens: int = 128,
        stop_tokens: Optional[Sequence[Sequence[int]]] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        logits_processors: Optional[
            List[Callable[[mx.array, mx.array], mx.array]]
        ] = None,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        prefill_batch_window: Optional[int] = None,
        decode_priority_cadence: int = 1,
        adaptive_prefill: bool = False,
        adaptive_prefill_target_itl_ms: float = 1500.0,
        adaptive_prefill_max_defer_ms: float = 2000.0,
        adaptive_prefill_slices: Sequence[int] = (64, 128, 256, 512),
        max_kv_size: Optional[int] = None,
        kv_budget_bytes: Optional[int] = None,
        kv_cost: Optional[Tuple[float, float, Optional[int]]] = None,
        state_budget: Optional[StateBudget] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        stream=None,
        prompt_trim_rollback_tokens: int = 0,
        self_mtp: Optional[dict] = None,
        mtp_admission: Optional[
            Callable[
                [Sequence[Tuple[int, int, int, bool, float]]],
                Mapping[int, Union[int, str]],
            ]
        ] = None,
        scheduler_stats: Optional[Dict[str, Any]] = None,
        post_prefill_transform: Optional[Callable[..., Optional[dict]]] = None,
        decode_time_fairness: Optional[Mapping[str, Any]] = None,
        adaptive_mtp_depth: Optional[Mapping[str, Any]] = None,
        mtp_ordinary_handoff: Optional[Any] = None,
        fly_verification=None,
        apc_interior_checkpoints: Optional[Mapping[str, int]] = None,
        prefill_scheduling: Optional[Mapping[str, Any]] = None,
        memory_pressure_level: Optional[Callable[[], int]] = None,
        copy_draft=None,
        mtp_acceptance_log: Optional[Any] = None,
    ):
        if decode_priority_cadence < 1:
            raise ValueError("decode_priority_cadence must be positive")
        if adaptive_prefill and decode_priority_cadence != 1:
            raise ValueError(
                "adaptive_prefill cannot be combined with decode_priority_cadence"
            )
        if adaptive_prefill_target_itl_ms <= 0:
            raise ValueError("adaptive_prefill_target_itl_ms must be positive")
        if adaptive_prefill_max_defer_ms <= 0:
            raise ValueError("adaptive_prefill_max_defer_ms must be positive")
        adaptive_prefill_slices = tuple(
            sorted({int(value) for value in adaptive_prefill_slices})
        )
        if not adaptive_prefill_slices or adaptive_prefill_slices[0] <= 0:
            raise ValueError("adaptive_prefill_slices must contain positive values")
        if self_mtp is not None and decode_priority_cadence != 1:
            raise ValueError(
                "decode_priority_cadence is not supported with batched self-MTP"
            )
        if kv_bits is not None and quantized_kv_start != 0:
            raise NotImplementedError(
                "BatchGenerator only supports quantized_kv_start=0 with kv_bits set — delayed/threshold quantization is not implemented for the continuous-batching path."
            )
        self.model = model
        self.self_mtp = dict(self_mtp) if self_mtp is not None else None
        self.mtp_admission = mtp_admission
        if adaptive_mtp_depth is not None and self.self_mtp is None:
            raise ValueError("adaptive MTP depth requires a self-MTP route")
        self.adaptive_mtp_depth = (
            None if adaptive_mtp_depth is None else dict(adaptive_mtp_depth)
        )
        if mtp_ordinary_handoff is not None and self.self_mtp is None:
            raise ValueError("MTP ordinary handoff requires a self-MTP route")
        self.mtp_ordinary_handoff = mtp_ordinary_handoff
        if mtp_acceptance_log is not None and self.self_mtp is None:
            raise ValueError("MTP acceptance logging requires a self-MTP route")
        self.mtp_acceptance_logger = None
        if mtp_acceptance_log is not None:
            from .mtp_confidence import MTPAcceptanceLogger

            self.mtp_acceptance_logger = (
                mtp_acceptance_log
                if isinstance(mtp_acceptance_log, MTPAcceptanceLogger)
                else MTPAcceptanceLogger(
                    **(
                        {"path": mtp_acceptance_log}
                        if not isinstance(mtp_acceptance_log, Mapping)
                        else dict(mtp_acceptance_log)
                    )
                )
            )
        from .speculative_sampling import FLyVerificationPolicy

        self.fly_verification = FLyVerificationPolicy.from_value(fly_verification)
        if self.fly_verification.enabled and self.self_mtp is None:
            raise ValueError("FLy verification requires a self-MTP route")
        from .copy_draft import CopyDraftPolicy

        self.copy_draft = CopyDraftPolicy.from_value(copy_draft)
        if self.copy_draft.enabled and self.self_mtp is None:
            raise ValueError("self-MTP copy drafts require a self-MTP route")
        if self.self_mtp is not None:
            if not self.self_mtp.get("persistent", True):
                raise ValueError("batched self-MTP requires persistent_mtp=True")
            if self.self_mtp.get("window_size") is not None:
                raise ValueError("windowed MTP is not batchable")
            if self.self_mtp.get("rate_gate", False):
                raise ValueError("runtime rate gating is not batchable")
            if self.self_mtp.get("speculation_router") is not None:
                raise ValueError("adaptive per-lane MTP depth is not batchable")
            if max_kv_size is not None:
                raise ValueError("bounded (windowed) KV caches are not MTP batchable")
            if kv_bits is not None and (not self.self_mtp.get("allow_quantized_kv")):
                raise ValueError(
                    "quantized KV caches are not MTP batchable unless allow_quantized_kv is set in the self-MTP config"
                )
        self.max_tokens = max_tokens
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.logits_processors = logits_processors or []
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.prefill_batch_window = (
            1 if prefill_batch_window is None else prefill_batch_window
        )
        if self.prefill_batch_window < 1:
            raise ValueError("prefill_batch_window must be positive")
        self.decode_priority_cadence = decode_priority_cadence
        self.adaptive_prefill = bool(adaptive_prefill)
        self.adaptive_prefill_target_itl_ms = float(adaptive_prefill_target_itl_ms)
        self.adaptive_prefill_max_defer_ms = float(adaptive_prefill_max_defer_ms)
        self.adaptive_prefill_slices = tuple(
            (value for value in adaptive_prefill_slices if value <= prefill_step_size)
        ) or (min(adaptive_prefill_slices[0], prefill_step_size),)
        self._last_decode_completed_s = None
        self._last_decode_interval_ms = None
        self._last_decode_duration_ms = None
        self._prefill_ms_per_token_ewma = None
        self.scheduler_stats = scheduler_stats if scheduler_stats is not None else {}
        if self.copy_draft.enabled:
            # Mechanism counters are present (as zero) from the first scrape so
            # a harness can refuse an enabled arm that never copied.
            for key in (
                "self_mtp_copy_rounds",
                "self_mtp_copy_proposed_tokens",
                "self_mtp_copy_accepted_tokens",
                "self_mtp_copy_probe_rounds",
                "self_mtp_copy_gate_declines",
                "self_mtp_copy_lookup_misses",
            ):
                self.scheduler_stats.setdefault(key, 0)
        self.post_prefill_transform = post_prefill_transform
        self._post_prefill_receipts = {}
        # rm15: per-request prefill chunk sizes.  The chunk size is chosen per
        # round from measured timing and current concurrency, and a different
        # chunk size gives a different (still deterministic) answer, so a
        # receipt that does not record it cannot explain why the same prompt
        # answered differently.  A histogram, so it stays bounded by the number
        # of DISTINCT sizes rather than by the number of rounds.
        self._prefill_chunk_trace = {}
        from .adaptive_policy import DecodeTimeFairness

        self.decode_time_fairness = DecodeTimeFairness(
            **dict(decode_time_fairness or {})
        )
        from .adaptive_policy import PrefillOrder

        self.prefill_order = PrefillOrder.from_value(prefill_scheduling)
        if self.prefill_order.enabled:
            self._sync_prefill_order_stats()
        for key in (
            "prefill_rounds",
            "prefill_only_rounds",
            "decode_priority_release_rounds",
            "decode_priority_deferred_rounds",
            "adaptive_prefill_release_rounds",
            "adaptive_prefill_slack_deferred_rounds",
            "adaptive_prefill_deadline_forced_rounds",
            "adaptive_prefill_apc_priority_admissions",
            "mtp_short_prefill_interleaved",
            "short_prefill_overflow_admissions",
            "apc_interior_checkpoints_captured",
            "apc_interior_checkpoints_skipped_trimmable",
            "apc_interior_checkpoints_skipped_inexact",
            "prefill_chunk_rounds_recorded",
            "prefill_chunk_varied_requests",
        ):
            self.scheduler_stats.setdefault(key, 0)
        self.scheduler_stats.setdefault("adaptive_prefill_chunk_histogram", {})
        self.completion_batch_size = max(completion_batch_size, prefill_batch_size)
        if self.prefill_order.enabled and self.self_mtp is not None:
            # The self-MTP prefill admission window is a FIFO prefix of the
            # queue (``_next_mtp``: ``list(self._unprocessed_sequences)[:n]``),
            # and ``PrefillOrder`` chooses *inside* it.  A request can only be
            # overtaken while it is still a candidate, so the forced-bypass cap
            # needs a window of at least ``max_bypass + 1``; narrower than that
            # the served request never accumulates ``max_bypass`` consecutive
            # bypasses and the cap can never fire.  Measured on CPU with the
            # default cap of 3: the cap first fires at window 4 (window 2 and 3
            # give bypass_forced == 0).  Fail closed rather than serve a
            # fairness guarantee that is silently absent.
            #
            # The window is ``--max-lanes``; see _segment_aware_cohort_size's
            # docstring for why.  This guard is therefore tripped by a low
            # ``--max-lanes`` (default 4, i.e. exactly the boundary), or by a
            # harness constructing BatchGenerator directly.
            window = self.completion_batch_size
            if _segment_aware_live_tip_enabled(self.self_mtp):
                window = min(window, _segment_aware_cohort_size(self.self_mtp))
            if window <= self.prefill_order.max_bypass:
                raise ValueError(
                    "prefill_scheduling max_bypass="
                    f"{self.prefill_order.max_bypass} requires a self-MTP prefill "
                    f"admission window of at least "
                    f"{self.prefill_order.max_bypass + 1}, but this configuration "
                    f"admits at most {window} "
                    "(--max-lanes, which also sets segment_aware_cohort_size "
                    "through the adapter's execution_config); raise "
                    "--max-lanes or lower max_bypass"
                )
        self.max_kv_size = max_kv_size
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        self.apc_interior_checkpoints = dict(
            apc_interior_checkpoints or {"count": 0, "min_stride": 1}
        )
        # Host pressure gate for disposable rolling captures (P4); None means
        # the signal is unavailable and captures always proceed.
        self.memory_pressure_level = memory_pressure_level
        self.prompt_trim_rollback_tokens = max(0, int(prompt_trim_rollback_tokens))
        if state_budget is not None and (
            kv_budget_bytes is not None or kv_cost is not None
        ):
            raise ValueError(
                "state_budget cannot be combined with kv_budget_bytes/kv_cost"
            )
        if (
            state_budget is not None
            and type(state_budget.project) is not LinearStateCost
        ):
            raise ValueError(
                "BatchGenerator state_budget requires LinearStateCost; peak-only iterative policies such as StepStateCost must be driven by their model-native scheduler with actual resident request state"
            )
        if (
            state_budget is not None
            and state_budget.project.bytes_per_unit > 0
            and (state_budget.project.allocation_step_units is None)
        ):
            raise ValueError(
                "BatchGenerator budgeting over growing state requires allocation_step_units: shared batch caches allocate in steps at the cohort-max width, and unrounded admission can exceed the budget. Fixed-only state may omit the step."
            )
        if kv_budget_bytes is not None:
            if kv_cost is None:
                raise ValueError(
                    "kv_budget_bytes requires kv_cost=(fixed_bytes_per_row, bytes_per_token) measured for this model"
                )
            if len(kv_cost) < 3 or (kv_cost[1] > 0 and kv_cost[2] is None):
                raise ValueError(
                    "kv_cost must be (fixed_bytes, bytes_per_token, allocation_step_units) with a validated step whenever bytes_per_token > 0: unrounded per-token cost cannot budget shared stepped batch caches safely"
                )
            (fixed, per_token) = (kv_cost[0], kv_cost[1])
            if not (
                math.isfinite(kv_budget_bytes)
                and math.isfinite(fixed)
                and math.isfinite(per_token)
            ):
                raise ValueError("kv_budget_bytes and kv_cost must be finite")
            if kv_budget_bytes <= 0 or fixed < 0 or per_token < 0:
                raise ValueError("kv_budget_bytes must be positive and kv_cost >= 0")
        self.kv_budget_bytes = kv_budget_bytes
        self.kv_cost = kv_cost
        self.state_budget = state_budget
        if kv_budget_bytes is not None:
            step = kv_cost[2] if len(kv_cost) > 2 else None
            self.state_budget = StateBudget(
                kv_budget_bytes,
                LinearStateCost(
                    fixed, per_token, max_units=max_kv_size, allocation_step_units=step
                ),
            )
        self._stream = stream or generation_stream
        self._default_stop_matcher = StopSequenceMatcher(
            stop_tokens if stop_tokens else None
        )
        self._uid_count = 0
        self._prompt_batch = PromptProcessingBatch.empty(
            self.model,
            self.sampler,
            prefill_step_size=prefill_step_size,
            prompt_trim_rollback_tokens=self.prompt_trim_rollback_tokens,
        )
        if self.self_mtp is None:
            self._generation_batch = GenerationBatch.empty(self.model, self.sampler)
        else:
            self._generation_batch = MTPGenerationBatch.empty(
                self.model,
                mtp_admission=self.mtp_admission,
                segmented_live_tip=_segment_aware_live_tip_enabled(self.self_mtp),
                async_qsa_promotion=_segmented_async_qsa_promotion_enabled(
                    self.self_mtp
                ),
                ordinary_handoff_policy=getattr(
                    self, "mtp_ordinary_handoff", None
                ),
                scheduler_stats=self.scheduler_stats,
            )
        self._plain_fallback_batch = GenerationBatch.empty(self.model, self.sampler)
        self._starved_mtp_boundaries = 0
        self._unprocessed_sequences = deque()
        self._currently_processing = []
        self._mtp_states = {}
        self._mtp_lane_rngs = {}
        self._mtp_configs = {}
        # Declared HTTP batch cohorts are an atomic scheduling contract.  The
        # serving worker consumes these failures after ``next`` and releases
        # every request together; no lane from a rejected cohort may start a
        # smaller speculative decode batch.
        self._atomic_cohort_failures = []
        # UIDs in this set own mutable target + draft cache state produced by
        # bounded teacher-forced prefill.  Those bytes are already reflected
        # in the live free-memory reading; admission must charge only their
        # remaining growth before allowing the next slice.
        self._mtp_prefill_resident = set()
        self._mtp_prefill_projection_bytes = {}
        self._mtp_prefill_failures = []
        self._prompt_boundaries = {}
        self._interior_checkpoint_positions = {}
        self._interior_checkpoints = {}
        # Purpose of each planned position; absent positions are INTERIOR.
        self._state_boundary_purposes = {}
        # Rolling/junction snapshots awaiting immediate publication.
        self._state_checkpoints = []
        self._cache_capsule_pending = {}
        self._cache_capsule_by_uid = {}
        self._cache_capsule_receipts = {}
        self._prompt_tokens_counter = 0
        self._prompt_time_counter = 0
        self._gen_tokens_counter = 0
        self._steps_counter = 0
        device_info = mx.device_info()
        if (
            mx.metal.is_available()
            and "max_recommended_working_set_size" in device_info
        ):
            self._old_wired_limit = mx.set_wired_limit(
                device_info["max_recommended_working_set_size"]
            )
        else:
            self._old_wired_limit = None

    @property
    def stream(self):
        return self._stream

    def _fairness(self):
        policy = getattr(self, "decode_time_fairness", None)
        if policy is None:
            from .adaptive_policy import DecodeTimeFairness

            policy = self.decode_time_fairness = DecodeTimeFairness()
        return policy

    def _prefill_order(self):
        policy = getattr(self, "prefill_order", None)
        if policy is None:
            from .adaptive_policy import PrefillOrder

            policy = self.prefill_order = PrefillOrder()
        return policy

    def _bounded_prefill_chunks(self):
        """Whether prefill slices follow ``_adaptive_prefill_decision``."""
        order = getattr(self, "prefill_order", None)
        return bool(
            self.adaptive_prefill
            or self._fairness().enabled
            or (order is not None and order.one_slice_contention)
        )

    def close(self):
        if getattr(self, "_old_wired_limit", None) is not None:
            mx.synchronize(self._stream)
            mx.set_wired_limit(self._old_wired_limit)
            self._old_wired_limit = None
        generation_batch = getattr(self, "_generation_batch", None)
        if isinstance(generation_batch, MTPGenerationBatch):
            generation_batch.close()
        getattr(self, "_prompt_boundaries", {}).clear()
        getattr(self, "_interior_checkpoint_positions", {}).clear()
        getattr(self, "_interior_checkpoints", {}).clear()
        getattr(self, "_state_boundary_purposes", {}).clear()
        getattr(self, "_state_checkpoints", []).clear()
        getattr(self, "_post_prefill_receipts", {}).clear()
        getattr(self, "_prefill_chunk_trace", {}).clear()
        getattr(self, "_mtp_prefill_resident", set()).clear()
        getattr(self, "_mtp_prefill_projection_bytes", {}).clear()
        getattr(self, "_mtp_prefill_failures", []).clear()
        logger = getattr(self, "mtp_acceptance_logger", None)
        if logger is not None:
            logger.close()
        self.release_cache_capsules()

    def release_cache_capsules(self):
        """Release every pending/bound capsule while keeping the batch reusable."""
        seen = set()
        records = list(getattr(self, "_cache_capsule_by_uid", {}).values())
        records.extend(getattr(self, "_cache_capsule_pending", {}).values())
        for record in records:
            prepared = record["prepared"]
            if id(prepared) not in seen:
                seen.add(id(prepared)); prepared.close()
        getattr(self, "_cache_capsule_pending", {}).clear()
        getattr(self, "_cache_capsule_by_uid", {}).clear()
        getattr(self, "_cache_capsule_receipts", {}).clear()

    def __del__(self):
        if not sys.is_finalizing():
            self.close()

    @contextlib.contextmanager
    def stats(self, stats=None):
        stats = stats or BatchStats()
        if self.kv_bits is not None:
            stats.effective_quantized_kv_start = self.quantized_kv_start
        self._prompt_tokens_counter = 0
        self._prompt_time_counter = 0
        self._gen_tokens_counter = 0
        tic = time.perf_counter()
        try:
            yield stats
        finally:
            toc = time.perf_counter()
            total_time = toc - tic
            gen_time = total_time - self._prompt_time_counter
            stats.prompt_tokens += self._prompt_tokens_counter
            stats.prompt_time += self._prompt_time_counter
            stats.prompt_tps = stats.prompt_tokens / stats.prompt_time
            stats.generation_tokens += self._gen_tokens_counter
            stats.generation_time += gen_time
            stats.generation_tps = stats.generation_tokens / stats.generation_time
            stats.peak_memory = max(
                stats.peak_memory, mx.get_peak_memory() / 1000000000.0
            )

    def insert(
        self,
        prompts: List[List[int]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        stop_matchers: Optional[List[StopSequenceMatcher]] = None,
        mtp_states: Optional[List[Optional[Tuple[List[Any], mx.array]]]] = None,
        lane_rngs: Optional[List[Optional[LaneRNG]]] = None,
        self_mtp_configs: Optional[List[dict]] = None,
        apc_interior_positions: Optional[List[Optional[Sequence[int]]]] = None,
        prefill_inputs: Optional[List[Optional[dict]]] = None,
        state_boundaries: Optional[List[Optional[Sequence[StateBoundary]]]] = None,
    ):
        return self.insert_segments(
            [[p] for p in prompts],
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            stop_matchers,
            mtp_states,
            lane_rngs,
            self_mtp_configs,
            apc_interior_positions,
            prefill_inputs,
            state_boundaries,
        )

    def bind_cache_capsule(self, group, uid, prepared, expected_rows):
        """Bind one queued warm lane to an all-or-nothing prepared cache batch."""
        if self.self_mtp is not None:
            raise ValueError("cache capsules are not qualified with self-MTP")
        group = str(group)
        expected_rows = int(expected_rows)
        if expected_rows < 2:
            raise ValueError("cache capsule fanout requires at least two rows")
        pending_groups = getattr(self, "_cache_capsule_pending", None)
        if pending_groups is None:
            pending_groups = self._cache_capsule_pending = {}
        by_uid = getattr(self, "_cache_capsule_by_uid", None)
        if by_uid is None:
            by_uid = self._cache_capsule_by_uid = {}
        pending = pending_groups.setdefault(
            group,
            {"group": group, "prepared": prepared, "expected": expected_rows, "uids": []},
        )
        if pending["prepared"] is not prepared or pending["expected"] != expected_rows:
            raise ValueError("cache capsule group identity changed")
        if int(uid) in pending["uids"]:
            raise ValueError("duplicate cache capsule UID")
        pending["uids"].append(int(uid))
        if len(pending["uids"]) > expected_rows:
            raise ValueError("cache capsule group exceeded reserved rows")
        if len(pending["uids"]) == expected_rows:
            record = pending_groups.pop(group)
            record["uids"] = tuple(record["uids"])
            for member in record["uids"]:
                by_uid[member] = record
            return True
        return False

    def cancel_cache_capsule_group(self, group, reason="attachment_failed"):
        group = str(group)
        pending_groups = getattr(self, "_cache_capsule_pending", {})
        by_uid = getattr(self, "_cache_capsule_by_uid", {})
        record = pending_groups.pop(group, None)
        if record is None:
            record = next(
                (
                    candidate
                    for candidate in by_uid.values()
                    if candidate.get("group") == group
                ),
                None,
            )
        if record is None:
            return False
        self._decline_cache_capsule(record, reason)
        return True

    def _decline_cache_capsule(self, record, reason):
        by_uid = getattr(self, "_cache_capsule_by_uid", {})
        receipts = getattr(self, "_cache_capsule_receipts", None)
        if receipts is None:
            receipts = self._cache_capsule_receipts = {}
        uids = tuple(record["uids"])
        for uid in uids:
            by_uid.pop(uid, None)
            receipts[uid] = {
                "schema": "mlx2.cache-capsule.v1",
                "status": "fallback",
                "reason": str(reason),
                "rows": len(uids),
            }
        record["prepared"].close()
        self.scheduler_stats["cache_capsule_fallbacks"] = (
            self.scheduler_stats.get("cache_capsule_fallbacks", 0) + 1
        )

    def pop_cache_capsule_receipt(self, uid):
        return getattr(self, "_cache_capsule_receipts", {}).pop(int(uid), None)

    def _release_cache_capsule_uid(self, uid):
        uid = int(uid)
        by_uid = getattr(self, "_cache_capsule_by_uid", {})
        pending_groups = getattr(self, "_cache_capsule_pending", {})
        record = by_uid.pop(uid, None)
        if record is None:
            pending = next(
                (
                    (group, candidate)
                    for group, candidate in pending_groups.items()
                    if uid in candidate["uids"]
                ),
                None,
            )
            if pending is not None:
                group, record = pending
                pending_groups.pop(group, None)
                self._decline_cache_capsule(record, "pending_member_removed")
            return
        if not any(member in by_uid for member in record["uids"]):
            record["prepared"].close()

    def insert_segments(
        self,
        segments: List[List[List[int]]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        stop_matchers: Optional[List[StopSequenceMatcher]] = None,
        mtp_states: Optional[List[Optional[Tuple[List[Any], mx.array]]]] = None,
        lane_rngs: Optional[List[Optional[LaneRNG]]] = None,
        self_mtp_configs: Optional[List[dict]] = None,
        apc_interior_positions: Optional[List[Optional[Sequence[int]]]] = None,
        prefill_inputs: Optional[List[Optional[dict]]] = None,
        state_boundaries: Optional[List[Optional[Sequence[StateBoundary]]]] = None,
    ):
        """Queue prompts; ``state_boundaries`` plans exact prefill snapshots.

        ``apc_interior_positions`` is the INTERIOR-only alias of
        ``state_boundaries``; a lane's non-None ``state_boundaries`` entry
        wins.  With neither, a positive ``apc_interior_checkpoints.count``
        plans the default interior lattice.
        """
        uids = []
        max_tokens = max_tokens or [self.max_tokens] * len(segments)
        all_tokens = all_tokens or [[] for _ in segments]
        samplers = samplers or [None] * len(segments)
        logits_processors = logits_processors or [self.logits_processors] * len(
            segments
        )
        stop_matchers = stop_matchers or [self._default_stop_matcher] * len(segments)
        mtp_states = mtp_states or [None] * len(segments)
        lane_rngs = lane_rngs or [None] * len(segments)
        self_mtp_configs = self_mtp_configs or [{} for _ in segments]
        apc_interior_positions = (
            [None] * len(segments)
            if apc_interior_positions is None
            else apc_interior_positions
        )
        prefill_inputs = prefill_inputs or [None for _ in segments]
        state_boundaries = (
            [None] * len(segments) if state_boundaries is None else state_boundaries
        )
        for name, values in (
            ("mtp_states", mtp_states),
            ("lane_rngs", lane_rngs),
            ("self_mtp_configs", self_mtp_configs),
            ("apc_interior_positions", apc_interior_positions),
            ("prefill_inputs", prefill_inputs),
            ("state_boundaries", state_boundaries),
        ):
            if len(values) != len(segments):
                raise ValueError(f"{name} must have one entry per sequence")
        caches = caches or [None] * len(segments)
        for i in range(len(segments)):
            if caches[i] is None:
                caches[i] = self._make_new_cache()
            elif self.kv_bits is not None:
                maybe_quantize_kv_cache(
                    caches[i], self.quantized_kv_start, self.kv_group_size, self.kv_bits
                )
            if self.kv_bits is not None:
                for c in caches[i]:
                    leaves = c.caches if isinstance(c, CacheList) else (c,)
                    for leaf in leaves:
                        if not hasattr(leaf, "merge"):
                            raise ValueError(
                                f"kv_bits is set but the cache for this job quantizes to {type(leaf).__name__}, which does not support batching. Batched kv-quant currently requires rotating caches (set max_kv_size) or an unquantized generator."
                            )
            policy = getattr(
                self,
                "apc_interior_checkpoints",
                {"count": 0, "min_stride": 1},
            )
            total = len(all_tokens[i]) + sum(len(part) for part in segments[i])
            requested = state_boundaries[i]
            if requested is None and apc_interior_positions[i] is not None:
                requested = tuple(
                    StateBoundary(position, BoundaryPurpose.INTERIOR)
                    for position in apc_interior_positions[i]
                )
            if requested is None and int(policy.get("count", 0)) > 0:
                requested = tuple(
                    StateBoundary(position, BoundaryPurpose.INTERIOR)
                    for position in interior_checkpoint_positions(
                        total,
                        count=int(policy["count"]),
                        min_stride=int(policy["min_stride"]),
                    )
                )
            if requested or int(policy.get("count", 0)) > 0:
                from .apc_v2 import inspect_apc_capabilities

                capabilities = inspect_apc_capabilities(caches[i])
                if capabilities.interior_checkpoint_target:
                    requested = tuple(requested or ())
                    if not all(
                        isinstance(bound, StateBoundary) for bound in requested
                    ):
                        raise ValueError("invalid APC interior checkpoint positions")
                    passed = len(all_tokens[i])
                    requested = tuple(
                        bound for bound in requested if bound.position > passed
                    )
                    positions = tuple(bound.position for bound in requested)
                    if any(
                        isinstance(bound.position, bool)
                        or not isinstance(bound.position, int)
                        or bound.position >= total - 1
                        for bound in requested
                    ) or tuple(sorted(set(positions))) != positions:
                        raise ValueError("invalid APC interior checkpoint positions")
                    if positions:
                        uid = self._uid_count + i
                        self._interior_checkpoint_positions[uid] = deque(positions)
                        purposes = {
                            bound.position: BoundaryPurpose(bound.purpose)
                            for bound in requested
                            if bound.purpose != BoundaryPurpose.INTERIOR
                        }
                        if purposes:
                            self._state_boundary_purposes[uid] = purposes
                else:
                    _bump_bounded_counter(
                        self.scheduler_stats,
                        "apc_interior_checkpoints_skipped_trimmable",
                    )
        for seq, m, c, at, s, lp, sm, mtp_state, lane_rng, mtp_config, prefill_input in zip(
            segments,
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            stop_matchers,
            mtp_states,
            lane_rngs,
            self_mtp_configs,
            prefill_inputs,
        ):
            seq = list(seq)
            if len(seq[-1]) != 1:
                seq.append(seq[-1][-1:])
                seq[-2] = seq[-2][:-1]
            if self.self_mtp is not None and mtp_state is None and at:
                # A warm target prefix without its paired draft state cannot be
                # teacher-forced into a consistent self-MTP lane; the mismatch
                # would only surface inside ``next`` and take the whole batch
                # down.  Refuse at the request boundary instead.
                if not mtp_config.get("target_only_plain_fallback", False):
                    raise ValueError(
                        "self-MTP lane requires draft state for a warm target prefix; "
                        "use the target-only ordinary fallback"
                    )
            self._unprocessed_sequences.append(
                (self._uid_count, seq, m, c, at, s, lp, sm, time.monotonic(), prefill_input)
            )
            if self.self_mtp is not None:
                self._mtp_states[self._uid_count] = mtp_state
                self._mtp_lane_rngs[self._uid_count] = lane_rng
                self._mtp_configs[self._uid_count] = dict(mtp_config)
            uids.append(self._uid_count)
            self._uid_count += 1
        return uids

    def _make_new_cache(self):
        if self.max_kv_size is None:
            new_cache = cache.make_prompt_cache(self.model)
        else:
            new_cache = [
                RotatingKVCache(max_size=self.max_kv_size)
                if isinstance(ci, KVCache)
                else ci
                for ci in cache.make_prompt_cache(self.model)
            ]
        if self.kv_bits is not None:
            maybe_quantize_kv_cache(
                new_cache, self.quantized_kv_start, self.kv_group_size, self.kv_bits
            )
        return new_cache

    @staticmethod
    def _validate_mtp_config(config):
        if not config.get("persistent", True):
            raise ValueError("batched self-MTP requires persistent_mtp=True")
        if config.get("window_size") is not None:
            raise ValueError("windowed MTP is not batchable")
        if config.get("rate_gate", False):
            raise ValueError("runtime rate gating is not batchable")
        if config.get("speculation_router") is not None:
            raise ValueError("adaptive per-lane MTP depth is not batchable")
        if config.get("xtc_probability", 0.0) > 0.0:
            raise ValueError("stochastic XTC is not batchable with self-MTP")

    def _admit_mtp_joining(self, n: int) -> int:
        """Budget joining lanes' caches BEFORE ``_make_mtp_batch`` allocates.

        The admission callback sees the live rows plus the first ``n`` queued
        sequences (context, configured depth, retained cache bytes) and the
        gate admits the longest queue prefix whose lanes were approved (an MTP
        depth or plain; ``queue`` stops the prefix). A bounded-prefill
        continuation owns mutable cache already reflected in live headroom, so
        it reports that cache as resident and separately reserves its remaining
        projected growth. Fresh/APC joining lanes keep full-copy accounting.
        A lane approved only as plain is still prepared here — the
        merge-boundary admission pass then migrates it — while a queued lane
        allocates nothing this cycle.
        """
        if n <= 0:
            return n

        queued = list(self._unprocessed_sequences)
        first_uid = queued[0][0] if queued else None
        first_config = self._mtp_configs.get(first_uid, {})
        cohort = first_config.get("batch_cohort")
        cohort_uids = ()
        if cohort is not None:
            size = int(cohort["size"])
            key = (str(cohort["tenant_id"]), str(cohort["id"]), size)

            def belongs_to_cohort(sequence):
                value = self._mtp_configs.get(sequence[0], {}).get("batch_cohort")
                return value is not None and (
                    str(value["tenant_id"]),
                    str(value["id"]),
                    int(value["size"]),
                ) == key

            cohort_uids = tuple(
                sequence[0]
                for sequence in queued[:size]
                if belongs_to_cohort(sequence)
            )
            if len(cohort_uids) != size or n < size:
                self._record_atomic_cohort_failure(
                    tuple(sequence[0] for sequence in queued[:size]),
                    cohort,
                    "declared batch cohort could not fit one scheduler admission boundary",
                )
                return 0
            # A published cohort owns this boundary.  Do not merge later
            # ungrouped work into its strict-width speculative batch.
            n = size
            if any(
                self._mtp_row_lacks_draft_state(sequence)
                for sequence in queued[:size]
            ):
                # A member with a warm target prefix but no draft state can
                # decode only on the ordinary path, which the other members
                # cannot join without an ordinary prefill this route lacks
                # (a wholly target-only cohort is routed ordinary before
                # this).  Preparing it as a self-MTP lane would raise inside
                # ``next``, so fail the cohort closed before anything
                # allocates or prefills.
                self._record_atomic_cohort_failure(
                    cohort_uids,
                    cohort,
                    "declared batch cohort mixes a warm prefix without draft "
                    "state with self-MTP members",
                )
                return 0
        else:
            # Ungrouped work ahead of a cohort must not pull a proper prefix of
            # it into this boundary; the cohort waits to own the next one.
            for index, sequence in enumerate(queued[:n]):
                if self._mtp_configs.get(sequence[0], {}).get("batch_cohort"):
                    n = index
                    break
            if n > 0 and self._generation_batch_holds_cohort():
                # A declared cohort is an indivisible exact-width batch whose
                # admission stays all-or-nothing for its lifetime.  Ungrouped
                # work merged into it would be counted as a member and could
                # fail the cohort with it, so that work waits in the queue (a
                # scheduler wait) until the cohort drains.
                return 0

        if self.mtp_admission is None:
            return n
        from .apc_v2 import _walk_cache_entries

        rows = list(self._generation_batch.mtp_cycle_state())
        joining = []
        bound_violations = set()
        for sequence in list(self._unprocessed_sequences)[:n]:
            (uid, segments, _maximum, prompt_cache, history) = sequence[:5]
            config = dict(self.self_mtp or {})
            config.update(self._mtp_configs.get(uid, {}))
            prompt_len = sum((len(segment) for segment in segments))
            context = len(history) + prompt_len
            target_bytes = 0
            covered = 0
            for leaf in _walk_cache_entries(prompt_cache):
                target_bytes += int(getattr(leaf, "nbytes", 0))
                covered = max(covered, int(getattr(leaf, "offset", 0)))
            projected_target_bytes = target_bytes
            if covered > 0:
                projected_target_bytes = int(
                    target_bytes * (max(context, covered) / covered)
                )
            draft_bytes = 0
            draft_covered = 0
            mtp_state = self._mtp_states.get(uid)
            if mtp_state is not None:
                for leaf in mtp_state[0]:
                    draft_bytes += int(getattr(leaf, "nbytes", 0))
                    draft_covered = max(draft_covered, int(getattr(leaf, "offset", 0)))
            projected_draft_bytes = draft_bytes
            if draft_covered > 0:
                projected_draft_bytes = int(
                    draft_bytes * (max(context, draft_covered) / draft_covered)
                )
            else:
                layers = max(sum((1 for _ in prompt_cache)), 1)
                projected_draft_bytes = max(
                    draft_bytes, projected_target_bytes // layers
                )
            resident = uid in getattr(self, "_mtp_prefill_resident", ())
            if resident:
                current_bytes = target_bytes + draft_bytes
                measured_projection = projected_target_bytes + projected_draft_bytes
                projections = getattr(self, "_mtp_prefill_projection_bytes", {})
                projected_bytes = projections.get(uid)
                if projected_bytes is None:
                    cache_projection = getattr(
                        self.mtp_admission, "cache_projection_bytes", None
                    )
                    qualified_projection = (
                        int(cache_projection(context))
                        if cache_projection is not None
                        else 0
                    )
                    projected_bytes = max(
                        current_bytes, measured_projection, qualified_projection
                    )
                    projections[uid] = projected_bytes
                elif current_bytes > projected_bytes:
                    # An append-only cache exceeded its bound. Growing the
                    # reservation after granting ownership can strand a lane
                    # behind its own allocation, so reject this row instead.
                    # Rejection is terminal: the resident bytes only grow, so
                    # the row would fail this check at every later boundary.
                    # Surface it for the server to fail closed; never leave it
                    # queued with no memory or scheduler wait to explain it.
                    bound_violations.add(int(uid))
                    self._record_mtp_prefill_failure(
                        uid, current_bytes, projected_bytes
                    )
                    joining.append(
                        (
                            int(uid),
                            context,
                            int(config.get("num_draft", 1)),
                            False,
                            current_bytes / float(1 << 30),
                            0.0,
                        )
                    )
                    continue
                cache_gib = current_bytes / float(1 << 30)
                pending_gib = max(projected_bytes - current_bytes, 0) / float(
                    1 << 30
                )
            else:
                cache_gib = (
                    projected_target_bytes + projected_draft_bytes
                ) / float(1 << 30)
                pending_gib = 0.0
            step = int(config.get("prefill_step_size", self.prefill_step_size))
            if not resident and (covered <= 0 or context - covered > step):
                cache_gib = 0.0
            joining.append(
                (
                    int(uid),
                    context,
                    int(config.get("num_draft", 1)),
                    resident,
                    cache_gib,
                    pending_gib,
                )
            )
        admission_policy = self.mtp_admission
        if cohort_uids:
            admission_policy = getattr(admission_policy, "atomic", admission_policy)
        decisions = dict(admission_policy(tuple(rows + joining)) or {})
        if cohort_uids:
            # Integer decisions are speculative depths.  A plain/queue action
            # for even one member would turn the declared cohort into two
            # physical widths, so reject the entire group at this boundary.
            selected_depths = {
                decisions.get(uid)
                for uid in cohort_uids
                if isinstance(decisions.get(uid), int)
                and not isinstance(decisions.get(uid), bool)
                and decisions.get(uid) > 0
            }
            if any(
                uid in bound_violations
                or not isinstance(decisions.get(uid), int)
                or isinstance(decisions.get(uid), bool)
                or decisions.get(uid) <= 0
                for uid in cohort_uids
            ) or len(selected_depths) != 1:
                self._record_atomic_cohort_failure(
                    cohort_uids,
                    cohort,
                    "declared batch cohort was not wholly admitted for speculative decoding",
                )
                return 0
            selected_depth = selected_depths.pop()
            requested_depth = max(
                int(
                    self._mtp_configs[uid].get(
                        "num_draft", (self.self_mtp or {}).get("num_draft", 1)
                    )
                )
                for uid in cohort_uids
            )
            for uid in cohort_uids:
                self._mtp_configs[uid]["num_draft"] = selected_depth
                self._mtp_configs[uid]["cohort_admission_stage"] = (
                    "lower_k" if selected_depth < requested_depth else "full"
                )
                self._mtp_configs[uid]["requested_num_draft"] = requested_depth
        admitted = {
            row[0]
            for row in joining
            if row[0] not in bound_violations
            and (
                isinstance(decisions.get(row[0]), int)
                or decisions.get(row[0]) == "plain"
            )
        }
        # Queued prefills the memory controller refused stay in the prefill
        # queue; remember them so the serving watchdog can tell a memory wait
        # from a scheduler wait.  Every considered row is reclassified here.
        memory_queued = getattr(self, "_memory_queued_prefill", None)
        if memory_queued is None:
            memory_queued = self._memory_queued_prefill = set()
        for row in joining:
            if row[0] in admitted or row[0] in bound_violations:
                memory_queued.discard(row[0])
            elif decisions.get(row[0]) == "queue":
                memory_queued.add(row[0])
        # The memory policy ranks by cost, so its admitted set need not be a
        # queue prefix. Move that set to the front before preparation; otherwise
        # a rejected expensive head can prevent every cheaper row from running.
        if admitted:
            sequences = list(self._unprocessed_sequences)
            self._unprocessed_sequences = deque(
                [sequence for sequence in sequences if sequence[0] in admitted]
                + [sequence for sequence in sequences if sequence[0] not in admitted]
            )
        return len(admitted)

    def _record_atomic_cohort_failure(self, uids, cohort, reason):
        failures = getattr(self, "_atomic_cohort_failures", None)
        if failures is None:
            failures = self._atomic_cohort_failures = []
        normalized = tuple(int(uid) for uid in uids)
        if normalized and not any(item["uids"] == normalized for item in failures):
            failures.append(
                {
                    "uids": normalized,
                    "cohort": dict(cohort),
                    "reason": str(reason),
                }
            )
            self.scheduler_stats["atomic_cohort_admission_failures"] = (
                self.scheduler_stats.get("atomic_cohort_admission_failures", 0) + 1
            )

    def take_atomic_cohort_failures(self):
        failures = list(getattr(self, "_atomic_cohort_failures", ()))
        getattr(self, "_atomic_cohort_failures", []).clear()
        return failures

    def _mtp_prefill_resident_bytes(self, uid, prompt_cache):
        """Target + draft bytes a bounded self-MTP prefill currently holds."""
        from .apc_v2 import _walk_cache_entries

        total = sum(
            int(getattr(leaf, "nbytes", 0))
            for leaf in _walk_cache_entries(prompt_cache)
        )
        mtp_state = self._mtp_states.get(uid)
        if mtp_state is not None:
            total += sum(int(getattr(leaf, "nbytes", 0)) for leaf in mtp_state[0])
        return total

    def _record_mtp_prefill_failure(self, uid, current_bytes, projected_bytes):
        failures = getattr(self, "_mtp_prefill_failures", None)
        if failures is None:
            failures = self._mtp_prefill_failures = []
        uid = int(uid)
        if any(item["uid"] == uid for item in failures):
            return
        failures.append(
            {
                "uid": uid,
                "current_bytes": int(current_bytes),
                "projected_bytes": int(projected_bytes),
                "reason": (
                    "self-MTP prefill cache outgrew its admitted reservation "
                    f"({int(current_bytes)} > {int(projected_bytes)} bytes)"
                ),
            }
        )
        self.scheduler_stats["mtp_prefill_bound_violations"] = (
            self.scheduler_stats.get("mtp_prefill_bound_violations", 0) + 1
        )

    def _generation_batch_holds_cohort(self) -> bool:
        """Whether a declared batch cohort owns the live self-MTP batch."""
        batch = getattr(self, "_generation_batch", None)
        lanes = list(getattr(getattr(batch, "state", None), "lanes", ())) + [
            package.detached.lane
            for package in getattr(batch, "_paused", {}).values()
        ]
        return any(
            getattr(lane, "_batch_cohort", None) is not None for lane in lanes
        )

    def _mtp_row_lacks_draft_state(self, sequence) -> bool:
        """A queued target-only APC hit: warm target prefix, no draft state.

        ``insert`` accepts such a row only for the ordinary fallback;
        ``prepare_self_mtp_lane`` cannot teacher-force it into a self-MTP
        lane.
        """
        uid = sequence[0]
        return (
            bool(self._mtp_configs.get(uid, {}).get("target_only_plain_fallback"))
            and bool(sequence[4])
            and getattr(self, "_mtp_states", {}).get(uid) is None
        )

    def _record_mtp_lane_refusal(self, uid, reason):
        """Fail one queued lane through the bounded-prefill failure channel."""
        failures = getattr(self, "_mtp_prefill_failures", None)
        if failures is None:
            failures = self._mtp_prefill_failures = []
        uid = int(uid)
        if any(item["uid"] == uid for item in failures):
            return
        failures.append({"uid": uid, "reason": str(reason)})
        _bump_bounded_counter(self.scheduler_stats, "mtp_draftless_lane_refusals")

    def take_mtp_prefill_failures(self):
        """Transfer bounded-prefill lanes that can never be admitted."""
        failures = list(getattr(self, "_mtp_prefill_failures", ()))
        getattr(self, "_mtp_prefill_failures", []).clear()
        return failures

    def _make_mtp_batch(self, n: int):
        from .hybrid_speculative import prepare_self_mtp_lane
        from .adaptive_policy import CohortAdaptiveMTPDepth

        batch_admission = self.mtp_admission
        cohort_keys = []
        for sequence in list(self._unprocessed_sequences)[:n]:
            cohort = self._mtp_configs.get(sequence[0], {}).get("batch_cohort")
            cohort_keys.append(
                None
                if cohort is None
                else (
                    str(cohort["tenant_id"]),
                    str(cohort["id"]),
                    int(cohort["size"]),
                )
            )
        atomic_cohort = (
            bool(cohort_keys)
            and cohort_keys[0] is not None
            and len(set(cohort_keys)) == 1
            and len(cohort_keys) == cohort_keys[0][2]
        )
        if any(key is not None for key in cohort_keys) and not atomic_cohort:
            # Fail closed before popping: a declared cohort attaches exactly
            # as a whole, never as a proper subset or beside ungrouped work.
            raise RuntimeError(
                "declared batch cohort must be prepared as one whole batch"
            )
        draftless = [
            sequence
            for sequence in list(self._unprocessed_sequences)[:n]
            if self._mtp_row_lacks_draft_state(sequence)
        ]
        if draftless:
            # Admission routes these rows ordinary or fails their cohort, so
            # reaching here is a scheduler bug.  Refuse before popping and
            # fail the rows through the scheduler's failure channels: a raise
            # from ``prepare_self_mtp_lane`` would escape ``next`` and take
            # every other lane down with it.
            reason = "self-MTP lane has a warm target prefix without draft state"
            if atomic_cohort:
                self._record_atomic_cohort_failure(
                    [sequence[0] for sequence in list(self._unprocessed_sequences)[:n]],
                    self._mtp_configs.get(draftless[0][0], {}).get("batch_cohort")
                    or {},
                    reason,
                )
            else:
                for sequence in draftless:
                    self._record_mtp_lane_refusal(sequence[0], reason)
            return (None, [])
        sequences = [self._unprocessed_sequences.popleft() for _ in range(n)]
        if atomic_cohort:
            # Keep the same all-lanes uniform-depth policy at later decode
            # boundaries; otherwise the first cycle could re-expand to k2 and
            # detach one member after joining safely at k1.
            selected_depths = {
                int(self._mtp_configs[sequence[0]]["num_draft"])
                for sequence in sequences
            }
            if len(selected_depths) != 1:
                raise RuntimeError(
                    "declared batch cohort must use one admitted MTP depth"
                )
            selected_depth = selected_depths.pop()
            at_depth = getattr(batch_admission, "atomic_at_depth", None)
            batch_admission = (
                at_depth(selected_depth)
                if callable(at_depth)
                else getattr(batch_admission, "atomic", batch_admission)
            )
        detached = []
        initial = []
        stop_matchers = []
        progress = []
        for sequence in sequences:
            (
                uid,
                segments,
                maximum,
                prompt_cache,
                history,
                _,
                processors,
                matcher,
                _queued_at,
            ) = sequence[:9]
            getattr(self, "_mtp_prefill_resident", set()).discard(uid)
            getattr(self, "_mtp_prefill_projection_bytes", {}).pop(uid, None)
            prompt = [token for segment in segments for token in segment]
            config = dict(self.self_mtp or {})
            config.update(self._mtp_configs.pop(uid, {}))
            self._validate_mtp_config(config)
            processors = list(processors or [])
            prefix = mx.array(history, dtype=mx.uint32)
            prepare_processors = [
                lambda y, logits, processor=processor: processor(
                    mx.concatenate([prefix, y]), logits
                )
                for processor in processors
            ]
            _prefetch_known_mtp_tail(self.model, history, prompt, config)
            prompt_boundary = {}
            (lane, first) = prepare_self_mtp_lane(
                mx.array(prompt, dtype=mx.uint32),
                self.model,
                uid=uid,
                max_tokens=maximum,
                prompt_cache=prompt_cache,
                mtp_state=self._mtp_states.pop(uid, None),
                lane_rng=self._mtp_lane_rngs.pop(uid, None),
                num_draft=int(config.get("num_draft", 1)),
                sampling_temp=float(config.get("sampling_temp", 0.0)),
                sampling_top_p=float(config.get("top_p", 1.0)),
                sampling_top_k=int(config.get("top_k", 0)),
                sampling_min_p=float(config.get("min_p", 0.0)),
                accept_rule=config.get("accept_rule", "residual"),
                logits_processors=prepare_processors,
                prefill_step_size=int(
                    config.get("prefill_step_size", self.prefill_step_size)
                ),
                # The parent serving route selects shared QSA by default. Keep
                # an explicit False as the fail-safe opt-out for qualification
                # controls and bisects.
                share_qsa_indices=_share_qsa_indices_for_config(config),
                diagnostic_stages=config.get("_diagnostic_prepare_stages"),
                fused_gdn_catchup=bool(config.get("fused_gdn_catchup", False)),
                prompt_boundary_out=prompt_boundary,
                fly_verification=getattr(self, "fly_verification", None),
            )
            lane.lane.token_prefix = mx.array(history + prompt, dtype=mx.uint32)
            lane.lane.logits_processors = processors
            lane.lane._initial_shared_prefix_attestation = config.get("shared_prefix_attestation")
            lane.lane._cohort_admission_stage = config.get(
                "cohort_admission_stage"
            )
            lane.lane._requested_num_draft = config.get("requested_num_draft")
            lane.lane._batch_cohort = config.get("batch_cohort")
            copy_policy = getattr(self, "copy_draft", None)
            if copy_policy is not None and copy_policy.enabled:
                from .copy_draft import CopyDraftState

                # Index the *full* context, APC-restored history included, so
                # a prefix-cache hit still copies from its cached prompt.
                lane.lane.copy_draft = CopyDraftState(
                    copy_policy, list(history) + list(prompt) + [int(first.token)]
                )
            detached.append(lane)
            initial.append(first)
            stop_matchers.append(matcher)
            total = len(history) + len(prompt)
            progress.append(
                PromptProcessingBatch.Response(uid, (total, total), True, True)
            )
            if prompt_boundary:
                covered = int(prompt_boundary["covered_tokens"])
                full_prompt = history + prompt
                if covered <= len(full_prompt):
                    prompt_boundary["tokens"] = list(full_prompt[:covered])
                    self._prompt_boundaries[uid] = prompt_boundary
        configured_segmented = _segment_aware_live_tip_enabled(self.self_mtp)
        existing_rows = bool(self._generation_batch.mtp_cycle_state())
        segmented_join = configured_segmented and (
            not existing_rows or self._generation_batch.segmented_live_tip
        )
        return (
            MTPGenerationBatch(
                self.model,
                detached,
                initial,
                stop_matchers,
                mtp_admission=batch_admission,
                segmented_live_tip=segmented_join,
                async_qsa_promotion=_segmented_async_qsa_promotion_for_budget(
                    self.self_mtp,
                    min(
                        (
                            max(0, item.lane.max_tokens - item.lane.ntoks)
                            for item in detached
                        )
                    ),
                    record=True,
                ),
                arm_async_qsa_promotion=False,
                adaptive_depth_policy=(
                    None
                    if getattr(self, "adaptive_mtp_depth", None) is None
                    else CohortAdaptiveMTPDepth(
                        max_depth=max(
                            int(item.lane.num_draft) for item in detached
                        ),
                        **self.adaptive_mtp_depth,
                    )
                ),
                ordinary_handoff_policy=getattr(
                    self, "mtp_ordinary_handoff", None
                ),
                scheduler_stats=self.scheduler_stats,
                acceptance_logger=getattr(self, "mtp_acceptance_logger", None),
            ),
            progress,
        )

    def _next_interior_checkpoint(self, uid: int, covered: int):
        positions = getattr(self, "_interior_checkpoint_positions", {}).get(
            int(uid)
        )
        while positions and positions[0] <= int(covered):
            positions.popleft()
        return None if not positions else int(positions[0])

    def _boundary_purpose(self, uid: int, position: int) -> BoundaryPurpose:
        return getattr(self, "_state_boundary_purposes", {}).get(
            int(uid), {}
        ).get(int(position), BoundaryPurpose.INTERIOR)

    def _boundary_counter(self, purpose: BoundaryPurpose, event: str) -> str:
        name = "interior" if purpose == BoundaryPurpose.INTERIOR else purpose.name.lower()
        return f"apc_{name}_checkpoints_{event}"

    def _consume_state_boundary(self, uid: int, position: int) -> None:
        self._interior_checkpoint_positions[int(uid)].popleft()
        purposes = getattr(self, "_state_boundary_purposes", {}).get(int(uid))
        if purposes is not None:
            purposes.pop(int(position), None)

    def _skip_state_boundary_for_pressure(self, uid: int, position: int) -> bool:
        """Disposable rolling snapshots are the first thing pressure drops."""
        purpose = self._boundary_purpose(uid, position)
        probe = getattr(self, "memory_pressure_level", None)
        if purpose != BoundaryPurpose.ROLLING or probe is None:
            return False
        from .os_memory import PressureLevel

        try:
            level = int(probe())
        except Exception:  # noqa: BLE001 - a broken probe must not stop prefill
            return False
        if level < PressureLevel.WARN:
            return False
        _bump_bounded_counter(
            self.scheduler_stats, "apc_rolling_checkpoints_skipped_pressure"
        )
        self._consume_state_boundary(uid, position)
        return True

    def _record_state_checkpoint(self, uid: int, position: int, checkpoint) -> None:
        purpose = self._boundary_purpose(uid, position)
        checkpoint["purpose"] = purpose
        if purpose == BoundaryPurpose.INTERIOR:
            self._interior_checkpoints.setdefault(int(uid), []).append(checkpoint)
        else:
            self._state_checkpoints.append((int(uid), checkpoint))
        self._consume_state_boundary(uid, position)
        _bump_bounded_counter(
            self.scheduler_stats, self._boundary_counter(purpose, "captured")
        )

    def drain_state_checkpoints(self):
        """Transfer rolling/junction snapshots for immediate publication.

        Called once per serving loop after ``next``.  Interior snapshots stay
        with ``pop_interior_checkpoints`` and publish when the prompt ends.
        """
        drained = getattr(self, "_state_checkpoints", [])
        self._state_checkpoints = []
        return drained

    def _capture_mtp_interior_checkpoint(
        self, uid: int, history, prompt_cache, mtp_state
    ) -> bool:
        position = self._next_interior_checkpoint(uid, len(history) - 1)
        if position is None or position != len(history):
            return False
        if self._skip_state_boundary_for_pressure(uid, position):
            return False
        from .hybrid_speculative import capture_self_mtp_checkpoint

        lane_rng = self._mtp_lane_rngs.get(uid)
        checkpoint = capture_self_mtp_checkpoint(
            prompt_cache,
            mtp_state,
            rng_key=None if lane_rng is None else lane_rng.key,
            rng_draws=0 if lane_rng is None else int(lane_rng.draws),
        )
        if checkpoint is None:
            _bump_bounded_counter(
                self.scheduler_stats,
                self._boundary_counter(
                    self._boundary_purpose(uid, position), "skipped_inexact"
                ),
            )
            return False
        checkpoint["tokens"] = list(history)
        checkpoint["interior"] = True
        self._record_state_checkpoint(uid, position, checkpoint)
        return True

    def _advance_mtp_prefill(self, index: int, max_tokens: int):
        """Advance one queued self-MTP lane without blocking live decode."""
        from .hybrid_speculative import advance_self_mtp_prefill

        queued = list(self._unprocessed_sequences)
        sequence = queued[index]
        (
            uid,
            segments,
            maximum,
            prompt_cache,
            history,
            sampler,
            processors,
            matcher,
            _queued_at,
        ) = sequence[:9]
        prefill_input = sequence[9] if len(sequence) > 9 else None
        prompt = [token for segment in segments for token in segment]
        if len(prompt) <= 1:
            return self._release_prefilled_mtp_lane(index, [])
        config = dict(self.self_mtp or {})
        config.update(self._mtp_configs.get(uid, {}))
        self._validate_mtp_config(config)
        _prefetch_known_mtp_tail(self.model, history, prompt, config)
        next_checkpoint = self._next_interior_checkpoint(uid, len(history))
        if next_checkpoint is not None:
            max_tokens = min(max_tokens, next_checkpoint - len(history))
        tic = time.perf_counter()
        (remaining, prompt_cache, mtp_state, processed) = advance_self_mtp_prefill(
            mx.array(prompt, dtype=mx.uint32),
            self.model,
            prompt_cache=prompt_cache,
            mtp_state=self._mtp_states.get(uid),
            max_tokens=max_tokens,
            fused_gdn_catchup=bool(config.get("fused_gdn_catchup", False)),
        )
        toc = time.perf_counter()
        remaining = remaining.tolist()
        history = list(history) + prompt[:processed]
        self._record_prefill_chunk(uid, int(processed))
        self._mtp_states[uid] = mtp_state
        self._capture_mtp_interior_checkpoint(
            uid, history, prompt_cache, mtp_state
        )
        self._mtp_prefill_resident.add(uid)
        if uid not in self._mtp_prefill_projection_bytes:
            cache_projection = getattr(
                getattr(self, "mtp_admission", None),
                "cache_projection_bytes",
                None,
            )
            if cache_projection is not None:
                total = len(history) + len(remaining)
                projected = int(cache_projection(total))
                # The bytes this slice already allocated are a hard floor. An
                # adapter bound below them would make the very next boundary
                # read as an append-only bound violation, and the lane would
                # never be offered to admission again.
                resident = self._mtp_prefill_resident_bytes(
                    uid, prompt_cache
                )
                if resident > projected:
                    _bump_bounded_counter(
                        self.scheduler_stats,
                        "mtp_prefill_projection_floored_to_resident",
                    )
                    projected = resident
                self._mtp_prefill_projection_bytes[uid] = projected
        remaining_segments = (
            [remaining[:-1], remaining[-1:]] if len(remaining) > 1 else [remaining]
        )
        queued[index] = (
            uid,
            remaining_segments,
            maximum,
            prompt_cache,
            history,
            sampler,
            processors,
            matcher,
            toc,
            prefill_input,
        )
        self._unprocessed_sequences = deque(queued)
        self._prompt_tokens_counter += processed
        self._prompt_time_counter += toc - tic
        self._fairness().observe_prefill(
            processed,
            toc - tic,
            contended=self._has_active_decode(),
        )
        self.scheduler_stats["prefill_rounds"] += 1
        self.scheduler_stats["adaptive_prefill_release_rounds"] += 1
        histogram = self.scheduler_stats["adaptive_prefill_chunk_histogram"]
        key = str(processed)
        histogram[key] = int(histogram.get(key, 0)) + 1
        measured = (toc - tic) * 1000 / max(processed, 1)
        if self._prefill_ms_per_token_ewma is None:
            self._prefill_ms_per_token_ewma = measured
        else:
            self._prefill_ms_per_token_ewma = (
                0.75 * self._prefill_ms_per_token_ewma + 0.25 * measured
            )
        total = len(history) + len(remaining)
        progress = [
            PromptProcessingBatch.Response(uid, (len(history), total), False, False)
        ]
        if len(remaining) == 1:
            return self._release_prefilled_mtp_lane(index, progress)
        return (None, progress)

    def _release_prefilled_mtp_lane(self, index: int, progress):
        """Prepare one queued lane whose residual is its final prompt token.

        A declared ``batch_cohort`` member is held at that boundary instead:
        preparing it alone would start B1 decode while a sibling is still
        prefilling.  ``_next_mtp`` prepares the whole cohort in one
        ``_make_mtp_batch`` once no member needs another prefill slice.
        """
        queued = list(self._unprocessed_sequences)
        if self._mtp_configs.get(queued[index][0], {}).get("batch_cohort"):
            self.scheduler_stats["atomic_cohort_prefill_holds"] = (
                self.scheduler_stats.get("atomic_cohort_prefill_holds", 0) + 1
            )
            return (None, progress)
        if index:
            queued.insert(0, queued.pop(index))
            self._unprocessed_sequences = deque(queued)
        return self._make_mtp_batch(1)

    @staticmethod
    def _plain_sampler_for_mtp_lane(lane):

        def sample(logprobs):
            if lane.sampling_temp <= 0:
                return mx.argmax(logprobs, axis=-1)
            if lane.logprob_transform is not None:
                transformed = lane.logprob_transform(logprobs)
            else:
                transformed = logprobs / float(lane.sampling_temp)
            return mx.random.categorical(transformed, key=draw_key(lane.rng))

        sample.batch_groupable = False
        return sample

    def _migrate_plain_fallbacks(self):
        """Move plain-approved MTP lanes into the plain batch.

        Returns the responses this migration itself must emit: a lane that
        joined and was routed to plain at the merge boundary still carries its
        prepared-but-unemitted first token (``initial_output``, already
        counted in ``lane.ntoks``). It is delivered here exactly once, so the
        response neither drops that token nor double-subtracts it from the
        remaining budget; with ``max_tokens=1`` it is the lane's entire,
        terminal output.
        """
        if self.self_mtp is None:
            return []
        responses = []
        continuing = []
        for package in self._generation_batch.take_plain_fallbacks():
            lane = package.detached.lane
            handoff = package.handoff_receipt
            receipt = (
                None
                if handoff is None
                else {
                    "route": "ordinary_after_mtp_handoff",
                    "observed_compute_widths": list(
                        handoff["mtp_observed_compute_widths_before_handoff"]
                    ),
                    "ordinary_compute_widths": [],
                    "num_draft": int(handoff["num_draft_before_handoff"]),
                    "stats": dict(handoff["stats_before_handoff"]),
                    "mtp_ordinary_handoff": dict(handoff),
                }
            )
            if receipt is not None and (
                self._generation_batch.adaptive_depth_policy is not None
            ):
                adaptive = self._generation_batch.adaptive_depth_policy
                receipt["adaptive_depth"] = {
                    "selected": True,
                    "max_depth": int(adaptive.max_depth),
                    "current": int(adaptive.current_depth),
                    "counters": dict(adaptive.counters),
                    "cost_model": adaptive.diagnostics(),
                    "trace": list(adaptive.trace),
                }
            matcher_state = package.matcher_state
            initial = package.initial_output
            if initial is not None:
                (matcher_state, reason) = MTPGenerationBatch._finish_reason(
                    initial.token,
                    package.num_tokens + 1,
                    lane.max_tokens,
                    matcher_state,
                    package.stop_matcher,
                )
                response = MTPGenerationBatch.Response(
                    uid=lane.uid,
                    token=initial.token,
                    logprobs=initial.logprobs,
                    finish_reason=reason,
                    prompt_cache=None,
                    all_tokens=None,
                    from_draft=initial.from_draft,
                    mtp_receipt=receipt,
                )
                responses.append(response)
                if reason is not None:
                    response.prompt_cache = package.detached.caches.target
                    response.all_tokens = MTPGenerationBatch._prefix_tokens(lane)
                    response.mtp_state = (package.detached.caches.draft, lane.seed_h)
                    response.lane_rng = lane.rng
                    response.rng_draws = lane.rng.draws if lane.rng is not None else 0
                    _close_segmented_detached(package.detached, release_cache=False)
                    continue
            remaining = lane.max_tokens - lane.ntoks
            if remaining <= 0:
                _close_segmented_detached(package.detached, release_cache=True)
                continue
            continuing.append(
                (package, lane, matcher_state, remaining, receipt)
            )
        if continuing:
            plain = GenerationBatch(
                self.model,
                [lane.uid for (_, lane, _, _, _) in continuing],
                mx.array(
                    [lane.cur for (_, lane, _, _, _) in continuing],
                    dtype=mx.uint32,
                ),
                _merge_caches(
                    [
                        package.detached.caches.target
                        for (package, _, _, _, _) in continuing
                    ]
                ),
                [
                    MTPGenerationBatch._prefix_tokens(lane)
                    for (_, lane, _, _, _) in continuing
                ],
                [
                    self._plain_sampler_for_mtp_lane(lane)
                    for (_, lane, _, _, _) in continuing
                ],
                self.sampler,
                [lane.logits_processors for (_, lane, _, _, _) in continuing],
                [
                    package.stop_matcher
                    for (package, _, _, _, _) in continuing
                ],
                [remaining for (_, _, _, remaining, _) in continuing],
                route_receipts=[
                    receipt for (_, _, _, _, receipt) in continuing
                ],
                lane_rngs=[lane.rng for (_, lane, _, _, _) in continuing],
            )
            plain._matcher_states = [
                matcher_state for (_, _, matcher_state, _, _) in continuing
            ]
            self._plain_fallback_batch.extend(plain)
            for package, _, _, _, _ in continuing:
                _close_segmented_detached(package.detached, release_cache=True)
        return responses

    def mtp_cycle_state(self):
        if self.self_mtp is None:
            return []
        return self._generation_batch.mtp_cycle_state()

    def scheduler_waiting_uids(self):
        """Lanes that emit no progress only because the scheduler has not
        reached them: queued behind other prefills, or deferred by a live
        segmented width lock.  They are not memory-starved."""
        memory_queued = getattr(self, "_memory_queued_prefill", ())
        waiting = [
            row[0] for row in self._unprocessed_sequences if row[0] not in memory_queued
        ]
        if self.self_mtp is not None:
            waiting.extend(self._generation_batch.scheduler_waiting_uids())
        return waiting

    def set_mtp_num_draft(self, depths: Union[int, Mapping[int, int]]):
        if self.self_mtp is None:
            raise RuntimeError("BatchGenerator is not in self-MTP mode")
        self._generation_batch.set_num_draft(depths)

    def _find_uids(self, uids):
        uids = set(uids)
        results = {}
        for i, uid_i in enumerate(self._generation_batch.uids):
            if uid_i in uids:
                results[uid_i] = (2, i)
        if self.self_mtp is not None:
            active = set(self._generation_batch.uids)
            for uid_i, *_ in self._generation_batch.mtp_cycle_state():
                if uid_i in uids and uid_i not in active:
                    results[uid_i] = (2, -1)
        for i, uid_i in enumerate(self._plain_fallback_batch.uids):
            if uid_i in uids:
                results[uid_i] = (3, i)
        for i, uid_i in enumerate(self._prompt_batch.uids):
            if uid_i in uids:
                results[uid_i] = (1, i)
        for i, seq in enumerate(self._unprocessed_sequences):
            if seq[0] in uids:
                results[seq[0]] = (0, i)
        return results

    def extract_cache(self, uids):
        results = {}
        for uid, (stage, idx) in self._find_uids(uids).items():
            if stage == 0:
                results[uid] = self._unprocessed_sequences[idx][3:5]
            elif stage == 1:
                results[uid] = (
                    self._prompt_batch.extract_cache(idx),
                    self._prompt_batch.tokens[idx],
                )
            elif stage == 2:
                if self.self_mtp is not None:
                    results[uid] = self._generation_batch.extract_uid(uid)
                else:
                    results[uid] = (
                        self._generation_batch.extract_cache(idx),
                        self._generation_batch.tokens[idx],
                    )
            else:
                results[uid] = (
                    self._plain_fallback_batch.extract_cache(idx),
                    self._plain_fallback_batch.tokens[idx],
                )
        return results

    def pop_prompt_boundary(self, uid: int):
        """Transfer one committed prompt-boundary checkpoint to the server."""
        return self._prompt_boundaries.pop(int(uid), None)

    def pop_interior_checkpoints(self, uid: int):
        """Transfer exact budgeted interior checkpoints to the server."""
        getattr(self, "_interior_checkpoint_positions", {}).pop(int(uid), None)
        getattr(self, "_state_boundary_purposes", {}).pop(int(uid), None)
        return getattr(self, "_interior_checkpoints", {}).pop(int(uid), [])

    def pop_post_prefill_receipt(self, uid: int):
        """Transfer one optional state-transform receipt to the server."""
        return self._post_prefill_receipts.pop(int(uid), None)

    def _record_prefill_chunk(self, uid, width: int):
        """Note that ``uid``'s prefill advanced by ``width`` tokens this round.

        Sync-free and default-on: one dict update per request per round, no
        device work.  Bounded by the number of distinct widths a request can
        see, which is the number of adaptive slices plus the configured step.
        """
        if width <= 0:
            return
        trace = getattr(self, "_prefill_chunk_trace", None)
        if trace is None:
            trace = {}
            self._prefill_chunk_trace = trace
        entry = trace.get(int(uid))
        if entry is None:
            entry = {"widths": {}, "first": int(width), "last": int(width)}
            trace[int(uid)] = entry
        widths = entry["widths"]
        key = str(int(width))
        widths[key] = min(_COUNTER_MAX, int(widths.get(key, 0)) + 1)
        entry["last"] = int(width)
        stats = getattr(self, "scheduler_stats", None)
        if stats is not None:
            _bump_bounded_counter(stats, "prefill_chunk_rounds_recorded")

    def pop_prefill_chunk_trace(self, uid: int):
        """Transfer one request's prefill chunk histogram to the server.

        ``varied`` is the fact an auditor needs: the request's prefill did not
        run at one fixed chunk size, so its answer is reproducible only by
        replaying the same chunk schedule, not by replaying the prompt.
        """
        entry = getattr(self, "_prefill_chunk_trace", {}).pop(int(uid), None)
        if entry is None:
            return None
        widths = entry["widths"]
        # A single trailing short chunk is the arithmetic remainder of the
        # prompt length, not a scheduling decision, so it does not by itself
        # make a request's chunking load-dependent.
        scheduled = {w: c for w, c in widths.items()
                     if c > 1 or int(w) == entry["first"]}
        varied = len(scheduled) > 1
        stats = getattr(self, "scheduler_stats", None)
        if varied and stats is not None:
            _bump_bounded_counter(stats, "prefill_chunk_varied_requests")
        return {
            "schema": "mlx2.prefill-chunk-trace.v1",
            "configured_step": int(getattr(self, "prefill_step_size", 0)),
            "adaptive": bool(getattr(self, "adaptive_prefill", False)),
            "adaptive_slices": list(getattr(self, "adaptive_prefill_slices", ())),
            "widths": dict(widths),
            "rounds": int(sum(widths.values())),
            "first": entry["first"],
            "last": entry["last"],
            "varied": varied,
        }

    def remove(self, uids, return_prompt_caches=False):
        caches = {}
        if return_prompt_caches:
            caches = self.extract_cache(uids)
        keep = (
            set(range(len(self._unprocessed_sequences))),
            set(range(len(self._prompt_batch))),
            set(range(len(self._generation_batch))),
            set(range(len(self._plain_fallback_batch))),
        )
        found = self._find_uids(uids)
        memory_queued = getattr(self, "_memory_queued_prefill", None)
        if getattr(self, "_state_checkpoints", None):
            # An undrained snapshot has lost its owner (and with it the tenant
            # namespace it must be published under); drop it with the lane.
            removed = set(uids)
            self._state_checkpoints = [
                item for item in self._state_checkpoints if item[0] not in removed
            ]
        for uid in uids:
            self._prompt_boundaries.pop(uid, None)
            getattr(self, "_interior_checkpoint_positions", {}).pop(uid, None)
            getattr(self, "_interior_checkpoints", {}).pop(uid, None)
            getattr(self, "_state_boundary_purposes", {}).pop(uid, None)
            # Serving pops these at end_of_prompt; a lane removed before then
            # (cancelled, stalled or failed mid-prefill) would leave its
            # entry behind for the worker's lifetime.
            getattr(self, "_prefill_chunk_trace", {}).pop(uid, None)
            getattr(self, "_post_prefill_receipts", {}).pop(uid, None)
            if memory_queued is not None:
                memory_queued.discard(uid)
        for stage, idx in found.values():
            if idx >= 0:
                keep[stage].remove(idx)
        if len(keep[0]) < len(self._unprocessed_sequences):
            self._unprocessed_sequences = deque(
                (x for (i, x) in enumerate(self._unprocessed_sequences) if i in keep[0])
            )
            for uid in uids:
                self._mtp_states.pop(uid, None)
                self._mtp_lane_rngs.pop(uid, None)
                self._mtp_configs.pop(uid, None)
                self._mtp_prefill_resident.discard(uid)
                self._mtp_prefill_projection_bytes.pop(uid, None)
        if len(keep[1]) < len(self._prompt_batch):
            self._prompt_batch.filter(sorted(keep[1]))
            self._currently_processing = [
                x for (i, x) in enumerate(self._currently_processing) if i in keep[1]
            ]
        if self.self_mtp is not None:
            self._generation_batch.remove_uids(uids)
        elif len(keep[2]) < len(self._generation_batch):
            self._generation_batch.filter(sorted(keep[2]))
        if len(keep[3]) < len(self._plain_fallback_batch):
            self._plain_fallback_batch.filter(sorted(keep[3]))
        for uid in uids:
            self._release_cache_capsule_uid(uid)
        return caches

    @property
    def prompt_cache_nbytes(self):
        total = sum((c.nbytes for p in self._unprocessed_sequences for c in p[3]))
        total += sum(
            (
                int(getattr(leaf, "nbytes", 0))
                for state in self._mtp_states.values()
                if state is not None
                for leaf in state[0]
            )
        )
        total += sum(
            int(getattr(leaf, "nbytes", 0))
            for checkpoints in (
                *getattr(self, "_interior_checkpoints", {}).values(),
                [item[1] for item in getattr(self, "_state_checkpoints", ())],
            )
            for snapshot in checkpoints
            for leaf in list(snapshot.get("target_cache", ()))
            + list((snapshot.get("mtp_state") or ((), None))[0])
        )
        total += sum((c.nbytes for c in self._prompt_batch.prompt_cache))
        total += sum(
            (
                int(getattr(leaf, "nbytes", 0))
                for snapshot in getattr(self, "_prompt_boundaries", {}).values()
                for leaf in list(snapshot.get("target_cache", ()))
                + list((snapshot.get("mtp_state") or ((), None))[0])
            )
        )
        if self.self_mtp is None:
            total += sum((c.nbytes for c in self._generation_batch.prompt_cache))
        else:
            total += self._generation_batch.cache_nbytes
        total += sum((c.nbytes for c in self._plain_fallback_batch.prompt_cache))
        return total

    def _make_batch(self, n: int, indices=None):
        selected = (
            self._select_prefill_indices(n) if indices is None else list(indices)
        )
        if selected == list(range(n)):
            sequences = [self._unprocessed_sequences.popleft() for _ in range(n)]
        else:
            selected = set(selected)
            queued = list(self._unprocessed_sequences)
            sequences = [
                sequence for (i, sequence) in enumerate(queued) if i in selected
            ]
            self._unprocessed_sequences = deque(
                (sequence for (i, sequence) in enumerate(queued) if i not in selected)
            )
        uids = []
        caches = []
        tokens = []
        samplers = []
        logits_processors = []
        max_tokens = []
        stop_matchers = []
        for sequence in sequences:
            uids.append(sequence[0])
            caches.append(sequence[3])
            tokens.append(sequence[4])
            samplers.append(sequence[5])
            logits_processors.append(sequence[6])
            max_tokens.append(sequence[2])
            stop_matchers.append(sequence[7])
            self._currently_processing.append(
                [
                    sequence[1],
                    0,
                    sum((len(s) for s in sequence[1])),
                    sum((c.nbytes for c in sequence[3])) == 0,
                    len(sequence[4]) if sequence[4] else 0,
                    sequence[8] if len(sequence) > 8 else time.monotonic(),
                    sequence[9] if len(sequence) > 9 else None,
                ]
            )
        prepared_prompt_cache = None
        records = {
            id(record): record
            for uid in uids
            for record in [getattr(self, "_cache_capsule_by_uid", {}).get(uid)]
            if record is not None
        }
        if records:
            if len(records) != 1:
                for record in records.values():
                    self._decline_cache_capsule(record, "mixed_capsule_groups")
            else:
                record = next(iter(records.values()))
                if tuple(uids) == tuple(record["uids"]):
                    prepared = record["prepared"]
                    prepared_prompt_cache = prepared.prompt_cache
                    for uid in uids:
                        getattr(self, "_cache_capsule_receipts", {})[uid] = {
                            "schema": "mlx2.cache-capsule.v1",
                            "status": "engaged",
                            "rows": len(uids),
                            "planes": len(prepared_prompt_cache),
                            "capsule_planes": len(prepared.receipts),
                            "ordinary_planes": prepared.ordinary_planes,
                        }
                    self.scheduler_stats["cache_capsule_engaged"] = (
                        self.scheduler_stats.get("cache_capsule_engaged", 0) + 1
                    )
                    # Capsule storage is authoritative for this prompt batch;
                    # release the unused per-row APC branches immediately.
                    for cache in caches:
                        close = getattr(cache, "close", None)
                        if callable(close):
                            close()
                else:
                    self._decline_cache_capsule(record, "partial_scheduler_batch")
        return PromptProcessingBatch(
            model=self.model,
            uids=uids,
            caches=caches,
            tokens=tokens,
            prefill_step_size=self.prefill_step_size,
            samplers=samplers,
            fallback_sampler=self.sampler,
            logits_processors=logits_processors,
            stop_matchers=stop_matchers,
            max_tokens=max_tokens,
            prompt_trim_rollback_tokens=self.prompt_trim_rollback_tokens,
            prepared_prompt_cache=prepared_prompt_cache,
            prefill_inputs=[sequence[9] if len(sequence) > 9 else None for sequence in sequences],
        )

    def _prefill_chunk_length(self, segments):
        if len(segments) == 1 and len(segments[0]) == 1:
            return 0
        return min(len(segments[0]), self.prefill_step_size)

    def _admit_one_chunk_overflow(self, adaptive_chunk: int) -> bool:
        """Give a one-chunk request one extra slot beside full long prefills.

        With every prefill slot held by a multi-chunk prompt, a request that
        fits in a single chunk waited for a whole long prefill (omlx#3726).
        Admit it as one overflow lane; it finishes prefill in this round.

        Not while a post-prefill transform is installed: serving then runs
        prefill at B=1 so the transform sees one ready lane per boundary,
        and an overflow lane can become ready in the same round as the lane
        it overflowed.
        """
        if getattr(self, "post_prefill_transform", None) is not None:
            return False
        if len(self._prompt_batch) != self.prefill_batch_size:
            return False
        if (
            len(self._generation_batch) + len(self._prompt_batch)
            >= self.completion_batch_size
        ):
            return False
        step = (
            adaptive_chunk
            if BatchGenerator._bounded_prefill_chunks(self)
            else self.prefill_step_size
        )
        if any(
            sum(len(segment) for segment in sequence[0]) <= step
            for sequence in self._currently_processing
        ):
            return False
        for index, sequence in enumerate(self._unprocessed_sequences):
            if len(sequence) > 9 and sequence[9] is not None:
                continue
            if sum(len(segment) for segment in sequence[1]) <= step:
                break
        else:
            return False
        if self.state_budget is not None:
            # Budget the exact row being admitted, not the queue head.
            self._sync_budget_mutation()
            if not self._admit_states([self._candidate_admission_state(sequence)]):
                return False
        # Build exactly this row: the general selector could pick a media
        # request from the window instead.
        self._prompt_batch.extend(self._make_batch(1, indices=[index]))
        return True

    def _isolated_prefill_must_wait(self, sequence) -> bool:
        """Whether a request-private prefill row cannot be admitted now.

        A row carrying ``prefill_inputs`` must run its first prefill at an
        isolated B=1 boundary, so it waits while the prompt batch holds any
        other lane.  A persistent concept row also waits for the generation
        batch to drain, as the FIFO drain rule in ``_next`` does, because it
        may never share decode either.
        """
        if len(sequence) <= 9 or sequence[9] is None:
            return False
        if len(self._prompt_batch) > 0:
            return True
        payload = sequence[9]
        return (
            isinstance(payload, dict)
            and payload.get(_PERSISTENT_DECODE_INPUTS) is not None
            and len(self._generation_batch) > 0
        )

    def _select_prefill_indices(self, n: int):
        """Select a padding-efficient, starvation-bounded admission cohort."""
        if n <= 0:
            return []
        window = min(
            len(self._unprocessed_sequences), max(n, self.prefill_batch_window)
        )
        candidates = list(self._unprocessed_sequences)[:window]
        media = [
            index
            for index, sequence in enumerate(candidates)
            if len(sequence) > 9 and sequence[9] is not None
        ]
        if media:
            # Processor outputs carry per-request encoder inputs.  Keep their
            # first prefill isolated while ordinary text lanes continue to
            # batch normally before and after the boundary.  A row admitted
            # beside a prompt batch that still holds another lane's prefill
            # would not be isolated, so admission holds until that batch
            # drains; admitting text rows meanwhile could keep it busy and
            # starve the media row.
            if self._isolated_prefill_must_wait(candidates[media[0]]):
                return []
            return [media[0]]
        if window == n:
            return list(range(n))
        candidate_lengths = [
            self._prefill_chunk_length(sequence[1]) for sequence in candidates
        ]
        active_lengths = [
            self._prefill_chunk_length(sequence[0])
            for sequence in self._currently_processing
            if not (len(sequence[0]) == 1 and len(sequence[0][0]) == 1)
        ]
        selected = [0]
        selected_lengths = list(active_lengths)
        if candidate_lengths[0] > 0:
            selected_lengths.append(candidate_lengths[0])
        remaining = set(range(1, window))
        while len(selected) < n:

            def padding_after_adding(i):
                lengths = selected_lengths
                if candidate_lengths[i] > 0:
                    lengths = lengths + [candidate_lengths[i]]
                if not lengths:
                    return 0
                return max(lengths) * len(lengths) - sum(lengths)

            if getattr(self, "adaptive_prefill", False):

                def adaptive_cost(i):
                    residual = sum((len(segment) for segment in candidates[i][1]))
                    cached = len(candidates[i][4]) if candidates[i][4] else 0
                    return (residual, -cached, padding_after_adding(i), i)

                best = min(remaining, key=adaptive_cost)
                if best != min(remaining) and candidates[best][4]:
                    self.scheduler_stats[
                        "adaptive_prefill_apc_priority_admissions"
                    ] += 1
            else:
                best = min(remaining, key=lambda i: (padding_after_adding(i), i))
            selected.append(best)
            if candidate_lengths[best] > 0:
                selected_lengths.append(candidate_lengths[best])
            remaining.remove(best)
        return sorted(selected)

    def _queued_prefill_candidate(self, sequence):
        return self._prefill_order().candidate(
            sequence[0],
            sum(len(segment) for segment in sequence[1]),
            len(sequence[4]) if sequence[4] else 0,
            sequence[8] if len(sequence) > 8 else 0.0,
        )

    def _order_prefill_queue(self, n: int):
        """Move up to ``n`` ``PrefillOrder`` picks to the head of the queue.

        Returns ``(count, candidates)``; ``candidates`` describes the whole
        queue before the move so the caller can commit bypass counts for the
        rows it actually admits.  A media row keeps the isolation of
        ``_select_prefill_indices``: it is admitted only as the sole pick,
        and only once the batches it may not share have drained.
        """
        queued = list(self._unprocessed_sequences)
        candidates = [self._queued_prefill_candidate(seq) for seq in queued]
        order = self._prefill_order()
        pool = list(range(len(queued)))
        picks = []
        while pool and len(picks) < n:
            pick = pool[order.select([candidates[i] for i in pool])]
            pool.remove(pick)
            if len(queued[pick]) > 9 and queued[pick][9] is not None:
                if picks:
                    continue
                # When the row cannot run alone yet, admission holds this
                # round so the busy batch drains before its turn; picking
                # other rows instead could keep that batch busy forever.
                if not self._isolated_prefill_must_wait(queued[pick]):
                    picks.append(pick)
                break
            picks.append(pick)
        chosen = set(picks)
        self._unprocessed_sequences = deque(
            [queued[i] for i in picks]
            + [seq for (i, seq) in enumerate(queued) if i not in chosen]
        )
        return (len(picks), candidates)

    def _commit_prefill_order(self, served, candidates):
        self._prefill_order().commit(
            served,
            candidates,
            pending=[sequence[0] for sequence in self._unprocessed_sequences],
        )
        self._sync_prefill_order_stats()

    def _sync_prefill_order_stats(self):
        for key, value in self._prefill_order().counters.items():
            self.scheduler_stats[f"prefill_scheduling_{key}"] = int(value)

    def _one_slice_contended(self, bound: int) -> bool:
        """A long prefill runs while another prompt fits one bounded slice.

        ``bound`` is the slice being considered; a candidate fits only when
        its whole residual (less the final token kept for the generation
        boundary) is within ``bound`` *and* within its next interior
        checkpoint, because that boundary clamps its chunk too.
        """
        rows = []
        for index, sequence in enumerate(self._currently_processing):
            if len(sequence[0]) == 1 and len(sequence[0][0]) == 1:
                continue
            uid = self._prompt_batch.uids[index]
            covered = int(sequence[4] or 0) + int(sequence[1])
            rows.append((uid, sum(len(s) for s in sequence[0]), covered))
        for sequence in self._unprocessed_sequences:
            rows.append(
                (
                    sequence[0],
                    sum(len(s) for s in sequence[1]),
                    len(sequence[4]) if sequence[4] else 0,
                )
            )
        long_uids = {uid for (uid, remaining, _) in rows if remaining - 1 > bound}
        if not long_uids:
            return False
        for uid, remaining, covered in rows:
            if uid in long_uids:
                continue
            limit = bound
            boundary = self._next_interior_checkpoint(uid, covered)
            if boundary is not None:
                limit = min(limit, boundary - covered)
            if remaining - 1 <= limit:
                return True
        return False

    def _one_slice_bound(self, chunk: int) -> int:
        """Apply the stall bound to ``chunk`` under one-slice contention."""
        order = self._prefill_order()
        if not order.one_slice_contention:
            return chunk
        bound = self._fairness().stall_bound(chunk)
        if bound < chunk and self._one_slice_contended(bound):
            order.note_one_slice_clamp()
            self._sync_prefill_order_stats()
            return bound
        return chunk

    def _capped_tokens(self, tokens):
        if self.max_kv_size is not None:
            return min(tokens, self.max_kv_size)
        return tokens

    def _budget_admissible(self, n):
        """How many of the first n queued sequences fit the state budget.

        Shared batch caches (BatchKVCache) allocate every row at the
        cohort-max step-rounded width, so cost is NON-ADDITIVE: each prefix
        length is evaluated by recomputing the full cohort projection at
        final extents (via the policy's ``cohort_bytes``), floored by actual
        live bytes. No per-row resident credit is granted under stepped
        geometry — a supplied cache's bytes cannot reduce the shared width.
        """
        if self.state_budget is None or n <= 0:
            return n
        self._sync_budget_mutation()
        queued = list(self._unprocessed_sequences)[:n]
        return self._admit_states(
            [self._candidate_admission_state(seq) for seq in queued]
        )

    def _sync_budget_mutation(self):
        if (
            self.kv_budget_bytes is not None
            and self.state_budget.budget_bytes != self.kv_budget_bytes
        ):
            if not math.isfinite(self.kv_budget_bytes) or self.kv_budget_bytes <= 0:
                raise ValueError("kv_budget_bytes must be finite and positive")
            self.state_budget.budget_bytes = self.kv_budget_bytes

    def _candidate_admission_state(self, seq):
        new_tokens = sum((len(s) for s in seq[1]))
        history = len(seq[4]) if seq[4] else 0
        total = history + new_tokens + seq[2]
        existing = sum((c.nbytes for c in seq[3]))
        unverified = float(existing) if existing > 0 and history == 0 else 0.0
        return AdmissionState(
            seq[0],
            total,
            history,
            metadata={
                "phase": "queued",
                "prompt_units": new_tokens,
                "unverified_bytes": unverified,
            },
        )

    def _final_extent_states(self):
        """AdmissionStates of every admitted row at FINAL extent, for the
        shared-width cohort projection."""
        states = []
        gb = self._generation_batch
        for i in range(len(gb)):
            current = len(gb.tokens[i])
            final = current + max(gb.max_tokens[i] - gb._num_tokens[i], 0)
            states.append(
                AdmissionState(
                    gb.uids[i], final, current, metadata={"phase": "generation"}
                )
            )
        for i, seq in enumerate(self._currently_processing):
            history = seq[4] if len(seq) > 4 else 0
            final = history + seq[2] + self._prompt_batch.max_tokens[i]
            states.append(
                AdmissionState(
                    self._prompt_batch.uids[i],
                    final,
                    history + seq[1],
                    metadata={"phase": "prefill"},
                )
            )
        return states

    def _cohort_committed(self, candidate_states):
        """Projected committed bytes with the exact ``candidate_states``
        prefix admitted.

        The global cohort projection (all admitted rows + the selected
        prefix, at final extents, at the shared cohort-max rounded width)
        dominates both the current separate prompt/generation allocations
        and their eventual merge. Resident bytes of still-UNSELECTED queued
        rows are simultaneous with that future growth and are ADDED — never
        folded into a max — as are unverifiable supplied-cache bytes of
        selected candidates. The result is floored by admitted-batch actual
        live bytes (a stale wide allocation never assumed smaller than
        reality). State admission budget only; not a total process
        peak-memory guarantee (split/extend allocator transients are out of
        scope).
        """
        cost = self.state_budget.project
        cands = list(candidate_states)
        all_states = self._final_extent_states() + cands
        if hasattr(cost, "cohort_bytes"):
            projected = cost.cohort_bytes(all_states)
        else:
            projected = sum((self.state_budget.projected_bytes(s) for s in all_states))
        selected_unverified = sum(
            (s.metadata.get("unverified_bytes", 0.0) for s in cands)
        )
        selected_uids = {s.uid for s in cands}
        unselected_live = sum(
            (
                float(sum((c.nbytes for c in seq[3])))
                for seq in self._unprocessed_sequences
                if seq[0] not in selected_uids
            )
        )
        admitted_live = float(
            sum((c.nbytes for c in self._generation_batch.prompt_cache))
        ) + float(sum((c.nbytes for c in self._prompt_batch.prompt_cache)))
        return max(projected, admitted_live) + selected_unverified + unselected_live

    def _admit_states(self, states):
        """How many of ``states`` fit, in order — recomputing the full
        non-additive cohort cost for each exact prefix length."""
        states = list(states)
        admitted = 0
        for k in range(1, len(states) + 1):
            if self._cohort_committed(states[:k]) > self.state_budget.budget_bytes:
                break
            admitted = k
        return admitted

    def _demote_starved_mtp_lane(self):
        """Continue one paused lane as ordinary decode after bounded starvation.

        A paused lane retains its target and draft allocations.  If every MTP
        lane is paused and no ordinary fallback is running, waiting cannot free
        memory.  Count closed scheduler boundaries rather than wall time so the
        transition is deterministic and remains inside a transaction seam.
        """
        batch = self._generation_batch
        paused = getattr(batch, "_paused", None)
        starved = bool(
            paused
            and getattr(batch, "mtp_admission", None) is not None
            and not batch.state.lanes
            and not batch._plain_ready
            and len(self._plain_fallback_batch) == 0
        )
        if not starved:
            self._starved_mtp_boundaries = 0
            return None
        self._starved_mtp_boundaries += 1
        if self._starved_mtp_boundaries < MTP_STARVED_BOUNDARIES_BEFORE_PLAIN:
            return None
        self._starved_mtp_boundaries = 0
        uid = batch.demote_oldest_paused_to_plain()
        self.scheduler_stats["starved_mtp_plain_fallbacks"] = (
            self.scheduler_stats.get("starved_mtp_plain_fallbacks", 0) + 1
        )
        return uid

    def _admit_target_only_plain_fallbacks(self):
        """Start target-warm, draft-less lanes on the ordinary decode path."""
        capacity = self.completion_batch_size - len(
            self._generation_batch.mtp_cycle_state()
        ) - len(self._plain_fallback_batch)
        if capacity <= 0:
            return []
        queued = list(self._unprocessed_sequences)
        candidates = [
            index
            for index, sequence in enumerate(queued)
            if self._mtp_configs.get(sequence[0], {}).get(
                "target_only_plain_fallback", False
            )
        ]
        if not candidates:
            return []
        first_config = self._mtp_configs.get(queued[candidates[0]][0], {})
        cohort = first_config.get("batch_cohort")
        if cohort is None:
            indices = candidates[:capacity]
        else:
            cohort_key = (
                str(cohort["tenant_id"]),
                str(cohort["id"]),
                int(cohort["size"]),
            )

            def queued_cohort_key(index):
                value = self._mtp_configs.get(queued[index][0], {}).get(
                    "batch_cohort"
                )
                if value is None:
                    return None
                return (
                    str(value["tenant_id"]),
                    str(value["id"]),
                    int(value["size"]),
                )

            indices = [
                index
                for index in candidates
                if queued_cohort_key(index) == cohort_key
            ]
            if len(indices) != cohort_key[2] or len(indices) > capacity:
                return []
        for index in indices:
            sequence = queued[index]
            if sum(len(segment) for segment in sequence[1]) != 1:
                raise RuntimeError(
                    "target-only ordinary fallback requires a committed "
                    "prompt-boundary checkpoint"
                )
        uids = [queued[index][0] for index in indices]
        self._prompt_batch.extend(
            self._make_batch(len(indices), indices=indices)
        )
        for uid in uids:
            self._mtp_states.pop(uid, None)
            self._mtp_lane_rngs.pop(uid, None)
            self._mtp_configs.pop(uid, None)
        self.scheduler_stats["mtp_target_only_plain_fallbacks"] = (
            self.scheduler_stats.get("mtp_target_only_plain_fallbacks", 0)
            + len(uids)
        )
        return self._promote_ready_prompts(self._plain_fallback_batch)

    def _next_mtp(self):
        generation_responses = []
        prompt_responses = []
        had_decode_work = self._has_active_decode()
        decode_started = (
            time.perf_counter()
            if (self.adaptive_prefill or self._fairness().enabled)
            and had_decode_work
            else None
        )
        if (
            self._generation_batch.mtp_cycle_state()
            or self._generation_batch.has_deferred_lanes
        ):
            generation_responses.extend(self._generation_batch.next())
        lifecycle_failure = getattr(
            self._generation_batch, "take_atomic_cohort_failure", lambda: None
        )()
        if lifecycle_failure is not None:
            self.scheduler_stats["atomic_cohort_lifecycle_failures"] = (
                self.scheduler_stats.get("atomic_cohort_lifecycle_failures", 0) + 1
            )
            self._record_atomic_cohort_failure(
                lifecycle_failure["uids"],
                lifecycle_failure.get("cohort", {}),
                lifecycle_failure["reason"],
            )
        else:
            self._demote_starved_mtp_lane()
        generation_responses.extend(self._migrate_plain_fallbacks())
        if len(self._plain_fallback_batch) > 0:
            generation_responses.extend(self._plain_fallback_batch.next())
        handoff_policy = getattr(self, "mtp_ordinary_handoff", None)
        if (
            getattr(self._generation_batch, "_ordinary_handoff_latched", False)
            and handoff_policy is not None
            and len(self._plain_fallback_batch) < handoff_policy.max_mtp_width
        ):
            # Expire the cohort latch on width, not total service idleness.
            # A steady prefill queue must not make ordinary mode permanent.
            self._generation_batch.release_ordinary_handoff_latch()
        if decode_started is not None:
            decode_completed = time.perf_counter()
            self._last_decode_duration_ms = (decode_completed - decode_started) * 1000
            if self._last_decode_completed_s is not None:
                self._last_decode_interval_ms = (
                    decode_completed - self._last_decode_completed_s
                ) * 1000
            self._last_decode_completed_s = decode_completed
            self._fairness().observe_decode(
                decode_completed - decode_started
            )
            self._sync_decode_fairness_stats()
        if generation_responses:
            previous_tokens = self._gen_tokens_counter
            self._gen_tokens_counter += len(generation_responses)
            previous_steps = self._steps_counter
            self._steps_counter += 1
            if _crossed_counter_interval(
                previous_steps,
                self._steps_counter,
                ALLOCATOR_RECLAIM_STEP_INTERVAL,
            ) or _crossed_counter_interval(
                previous_tokens,
                self._gen_tokens_counter,
                ALLOCATOR_RECLAIM_MTP_TOKEN_INTERVAL,
            ):
                mx.clear_cache()
        prompt_responses.extend(self._admit_target_only_plain_fallbacks())
        occupied = len(self._generation_batch.mtp_cycle_state()) + len(
            self._plain_fallback_batch
        )
        n = min(self.completion_batch_size - occupied, len(self._unprocessed_sequences))
        if _segment_aware_live_tip_enabled(self.self_mtp):
            cohort_size = _segment_aware_cohort_size(self.self_mtp)
            n = min(n, max(0, cohort_size - occupied))
        n = self._budget_admissible(n)
        n = self._admit_mtp_joining(n)
        if n > 0:
            active_decode = self._has_active_decode()
            candidates = list(self._unprocessed_sequences)[:n]
            # ``prepare_self_mtp_lane`` consumes its entire residual prompt in
            # one call.  Keep that call bounded even when the server is idle or
            # the prompt has no APC history: otherwise a cold near-limit prompt
            # holds the serving loop until every target + draft prefill chunk
            # has completed, hiding progress and preventing cancellation and
            # watchdog checks.  A residual of ``step + 1`` tokens still needs
            # only one teacher-forced chunk because preparation retains the
            # final token for the generation boundary.
            candidate_steps = []
            incremental_indices = []
            for index, candidate in enumerate(candidates):
                config = dict(self.self_mtp or {})
                config.update(self._mtp_configs.get(candidate[0], {}))
                step = int(config.get("prefill_step_size", self.prefill_step_size))
                next_checkpoint = self._next_interior_checkpoint(
                    candidate[0], len(candidate[4])
                )
                if next_checkpoint is not None:
                    step = min(step, next_checkpoint - len(candidate[4]))
                candidate_steps.append(step)
                if (
                    next_checkpoint is not None
                    or sum(len(segment) for segment in candidate[1]) > step + 1
                ):
                    incremental_indices.append(index)
            incremental_prefill = bool(incremental_indices)
            order = self._prefill_order()
            adaptive_residual = incremental_prefill and (
                (
                    (self.adaptive_prefill or self._fairness().enabled)
                    and active_decode
                )
                # One-slice contention bounds a long slice on an idle server
                # too: the waiting short prompt is the contender.
                or order.one_slice_contention
            )
            (adaptive_defer, adaptive_chunk, deadline_forced) = (
                self._adaptive_prefill_decision(time.perf_counter())
                if adaptive_residual
                else (False, self.prefill_step_size, False)
            )
            if adaptive_residual and deadline_forced:
                adaptive_chunk = self._measured_adaptive_prefill_chunk()
                adaptive_chunk = self._fairness().cap(
                    adaptive_chunk, contended=True
                )
                self._sync_decode_fairness_stats()
            if adaptive_residual and adaptive_defer:
                return (prompt_responses, generation_responses)
            short_indices = [
                i for i in range(len(candidates)) if i not in incremental_indices
            ]
            has_cohort = any(
                self._mtp_configs.get(candidate[0], {}).get("batch_cohort")
                for candidate in candidates
            )
            prefill_candidates = (
                [self._queued_prefill_candidate(c) for c in candidates]
                if order.enabled
                else None
            )
            srpt_selected = None
            if (
                incremental_prefill
                and order.enabled
                and not deadline_forced
                and not has_cohort
            ):
                # SRPT over every candidate replaces the fixed alternation
                # below: a one-call prompt is always shorter than a multi-slice
                # residual, and the bypass cap (not a turn flag) bounds how
                # many of them may overtake a long prefill in a row.
                srpt_selected = order.select(prefill_candidates)
                if srpt_selected in short_indices:
                    self._commit_prefill_order(
                        [candidates[srpt_selected][0]], prefill_candidates
                    )
                    queued = list(self._unprocessed_sequences)
                    queued.insert(0, queued.pop(srpt_selected))
                    self._unprocessed_sequences = deque(queued)
                    self.scheduler_stats["mtp_short_prefill_interleaved"] = (
                        self.scheduler_stats.get("mtp_short_prefill_interleaved", 0) + 1
                    )
                    (batch, progress) = self._make_mtp_batch(1)
                    if batch is not None:
                        self._generation_batch.extend(batch)
                    prompt_responses.extend(progress)
                    generation_responses.extend(self._migrate_plain_fallbacks())
                    return (prompt_responses, generation_responses)
            elif (
                incremental_prefill
                and short_indices
                and not deadline_forced
                and not has_cohort
            ):
                # A request that fits in one chunk would otherwise wait for
                # every chunk of a long prefill ahead of it (omlx#3726: short
                # TTFT 8.86 s -> 0.73 s). Alternate the admission slot, long
                # prefill first, so both advance every other round. Declared
                # cohorts are admitted atomically and never interleave.
                short_turn = getattr(self, "_mtp_short_prefill_turn", False)
                self._mtp_short_prefill_turn = not short_turn
                if short_turn:
                    queued = list(self._unprocessed_sequences)
                    queued.insert(0, queued.pop(short_indices[0]))
                    self._unprocessed_sequences = deque(queued)
                    self.scheduler_stats["mtp_short_prefill_interleaved"] = (
                        self.scheduler_stats.get("mtp_short_prefill_interleaved", 0) + 1
                    )
                    (batch, progress) = self._make_mtp_batch(1)
                    if batch is not None:
                        self._generation_batch.extend(batch)
                    prompt_responses.extend(progress)
                    generation_responses.extend(self._migrate_plain_fallbacks())
                    return (prompt_responses, generation_responses)
            if incremental_prefill:
                if srpt_selected is not None:
                    selected = srpt_selected
                elif deadline_forced or len(incremental_indices) == 1:
                    selected = incremental_indices[0]
                else:
                    # Shortest residual among multi-slice prompts: the same
                    # ordering ``PrefillOrder`` applies when enabled, here
                    # without a bypass cap (disabled policy).
                    subset = [
                        prefill_candidates[i]
                        if prefill_candidates is not None
                        else self._queued_prefill_candidate(candidates[i])
                        for i in incremental_indices
                    ]
                    selected = incremental_indices[order.select(subset)]
                    if selected and candidates[selected][4]:
                        self.scheduler_stats[
                            "adaptive_prefill_apc_priority_admissions"
                        ] += 1
                if prefill_candidates is not None:
                    self._commit_prefill_order(
                        [candidates[selected][0]], prefill_candidates
                    )
                adaptive_chunk = min(adaptive_chunk, candidate_steps[selected])
                (batch, progress) = self._advance_mtp_prefill(selected, adaptive_chunk)
                if batch is not None:
                    self._generation_batch.extend(batch)
            else:
                (batch, progress) = self._make_mtp_batch(n)
                if batch is not None:
                    self._generation_batch.extend(batch)
            prompt_responses.extend(progress)
            generation_responses.extend(self._migrate_plain_fallbacks())
        return (prompt_responses, generation_responses)

    def _has_active_decode(self):
        if getattr(self, "self_mtp", None) is None:
            return len(self._generation_batch) > 0
        return bool(
            self._generation_batch.mtp_cycle_state()
            or self._generation_batch.has_deferred_lanes
            or len(self._plain_fallback_batch) > 0
        )

    def _has_prefill_work(self):
        if self._unprocessed_sequences:
            return True
        return any(
            (
                not (len(sequence[0]) == 1 and len(sequence[0][0]) == 1)
                for sequence in self._currently_processing
            )
        )

    def _should_defer_prefill(self):
        if (
            self.decode_priority_cadence == 1
            or not self._has_active_decode()
            or (not self._has_prefill_work())
        ):
            return False
        if self._steps_counter % self.decode_priority_cadence == 0:
            return False
        self.scheduler_stats["decode_priority_deferred_rounds"] += 1
        return True

    def _oldest_prefill_age_ms(self, now):
        queued = [
            sequence[8] for sequence in self._unprocessed_sequences if len(sequence) > 8
        ]
        active = [
            sequence[5]
            for sequence in self._currently_processing
            if len(sequence) > 5
            and (not (len(sequence[0]) == 1 and len(sequence[0][0]) == 1))
        ]
        service_times = active or queued
        return 0.0 if not service_times else (now - min(service_times)) * 1000

    def _mark_prefill_progress(self, now):
        """Restart the defer deadline for prompt rows that just made progress.

        A request's queued timestamp bounds its wait for first service.  Once
        admitted, the same field becomes the timestamp of its latest prefill
        slice so the deadline bounds the gap between slices instead of forcing
        every remaining slice after the request's original TTFT crosses it.
        """
        for sequence in self._currently_processing:
            if len(sequence) > 5:
                sequence[5] = now

    def _adaptive_prefill_decision(self, now):
        """Return ``(defer, chunk, deadline_forced)`` at a decode boundary."""
        contended = self._has_active_decode() and self._has_prefill_work()
        if not self._fairness().may_prefill(contended=contended):
            self._sync_decode_fairness_stats()
            return (True, 0, False)
        if (
            not self.adaptive_prefill
            or not self._has_active_decode()
            or (not self._has_prefill_work())
        ):
            chunk = self._fairness().cap(
                self.prefill_step_size, contended=contended
            )
            chunk = self._one_slice_bound(chunk)
            self._sync_decode_fairness_stats()
            return (False, chunk, False)
        forced = self._oldest_prefill_age_ms(now) >= self.adaptive_prefill_max_defer_ms
        if (
            not forced
            and self._last_decode_interval_ms is not None
            and (self._last_decode_interval_ms > self.adaptive_prefill_target_itl_ms)
        ):
            self.scheduler_stats["adaptive_prefill_slack_deferred_rounds"] += 1
            return (True, 0, False)
        chunk = self._measured_adaptive_prefill_chunk()
        if forced:
            self.scheduler_stats["adaptive_prefill_deadline_forced_rounds"] += 1
            chunk = self.adaptive_prefill_slices[0]
        chunk = self._fairness().cap(chunk, contended=contended)
        chunk = self._one_slice_bound(chunk)
        self._sync_decode_fairness_stats()
        return (False, chunk, forced)

    def _sync_decode_fairness_stats(self):
        for key, value in self._fairness().counters.items():
            self.scheduler_stats[f"decode_fairness_{key}"] = int(value)

    def _measured_adaptive_prefill_chunk(self):
        chunk = self.adaptive_prefill_slices[0]
        if self._prefill_ms_per_token_ewma is None:
            return chunk
        decode_ms = self._last_decode_duration_ms or 0.0
        budget_ms = max(0.0, self.adaptive_prefill_target_itl_ms * 0.75 - decode_ms)
        for candidate in self.adaptive_prefill_slices:
            if self._prefill_ms_per_token_ewma * candidate <= budget_ms:
                chunk = candidate
        return chunk

    def _capture_plain_interior_checkpoints(self):
        if not self._currently_processing:
            return
        from .cow_cache import snapshot_prompt_cache_descriptors

        for index, sequence in enumerate(self._currently_processing):
            uid = int(self._prompt_batch.uids[index])
            covered = int(sequence[4] or 0) + int(sequence[1])
            position = self._next_interior_checkpoint(uid, covered - 1)
            if position is None or position != covered:
                continue
            if self._skip_state_boundary_for_pressure(uid, position):
                continue
            target = self._prompt_batch.extract_cache(index)
            try:
                saved_target, _, receipt = snapshot_prompt_cache_descriptors(target)
            except Exception:
                for cache in target:
                    close = getattr(cache, "close", None)
                    if callable(close):
                        close()
                _bump_bounded_counter(
                    self.scheduler_stats,
                    self._boundary_counter(
                        self._boundary_purpose(uid, position), "skipped_inexact"
                    ),
                )
                self._consume_state_boundary(uid, position)
                continue
            mx.eval([cache.state for cache in saved_target])
            checkpoint = {
                "tokens": list(self._prompt_batch.tokens[index]),
                "target_cache": saved_target,
                "covered_tokens": covered,
                "committed_only": True,
                "interior": True,
                "snapshot_mode": "descriptor_cow",
                **receipt,
            }
            self._record_state_checkpoint(uid, position, checkpoint)

    def _promote_ready_prompts(self, destination=None):
        keep = []
        split = []
        for i, seq in enumerate(self._currently_processing):
            segments = seq[0]
            if len(segments) == 1 and len(segments[0]) == 1:
                split.append(i)
            else:
                keep.append(i)
        prompt_responses = []
        if split:
            last_inputs = [self._currently_processing[i][0][0] for i in split]
            progress = [(self._currently_processing[i][2],) * 2 for i in split]
            self._currently_processing = [self._currently_processing[i] for i in keep]
            ready = self._prompt_batch.split(split)
            exact_prompt_boundary = None
            if self.post_prefill_transform is not None:
                if len(ready.uids) != 1:
                    raise RuntimeError(
                        "post-prefill state transforms require an isolated B=1 boundary"
                    )
                uid = int(ready.uids[0])
                result = self.post_prefill_transform(
                    uid=uid,
                    model=self.model,
                    prompt_cache=ready.extract_cache(0),
                    cached_token_ids=tuple(ready.tokens[0]),
                )
                if result is not None:
                    receipt = dict(result.get("receipt") or {})
                    retained = result.get("retained_token_ids")
                    if retained is not None:
                        retained = [int(token) for token in retained]
                        if not retained:
                            raise RuntimeError(
                                "post-prefill state transform removed the entire cached prefix"
                            )
                        ready.tokens[0] = retained
                    replacement_cache = result.get("prompt_cache")
                    if replacement_cache is not None:
                        # The callback receives an extracted request-private
                        # cache. Publication is explicit: decode inherits it
                        # only when the transform returns it after committing.
                        # The transform edited an extracted single-row cache;
                        # decode needs the batch form back (extract/filter/extend).
                        ready.prompt_cache = _merge_caches([list(replacement_cache)])
                    exact_prompt_boundary = result.get("exact_prompt_boundary")
                    if exact_prompt_boundary is not None and (
                        not isinstance(exact_prompt_boundary, dict)
                        or not exact_prompt_boundary.get("committed_only")
                        or not exact_prompt_boundary.get("pre_transform_exact")
                        or not exact_prompt_boundary.get("tokens")
                        or exact_prompt_boundary.get("target_cache") is None
                    ):
                        raise RuntimeError(
                            "post-prefill exact boundary is incomplete or unauthoritative"
                        )
                    self._post_prefill_receipts[uid] = receipt
            for i, uid in enumerate(ready.uids):
                if exact_prompt_boundary is not None:
                    self._prompt_boundaries[uid] = exact_prompt_boundary
                    continue
                target_cache = ready.extract_cache(i)
                # The extracted rows are lazy slices of the whole ready batch.
                # Unevaluated, each published boundary would pin every row's
                # K/V while APCv2 accounts it as one row; scheduling the copy
                # now detaches the graph without a host sync.
                mx.async_eval([cache.state for cache in target_cache])
                self._prompt_boundaries[uid] = {
                    "tokens": list(ready.tokens[i]),
                    "target_cache": target_cache,
                    "committed_only": True,
                }
            gen_batch = ready.generate(last_inputs)
            for i, p in enumerate(progress):
                prompt_responses.append(
                    PromptProcessingBatch.Response(gen_batch.uids[i], p, True, True)
                )
            (self._generation_batch if destination is None else destination).extend(
                gen_batch
            )
        return prompt_responses

    def _next(self):
        if self.self_mtp is not None:
            return self._next_mtp()
        generation_responses = []
        prompt_responses = []
        if len(self._generation_batch) > 0:
            if self.adaptive_prefill or self._fairness().enabled:
                decode_started = time.perf_counter()
                generation_responses = self._generation_batch.next()
                decode_completed = time.perf_counter()
                self._last_decode_duration_ms = (
                    decode_completed - decode_started
                ) * 1000
                if self._last_decode_completed_s is not None:
                    self._last_decode_interval_ms = (
                        decode_completed - self._last_decode_completed_s
                    ) * 1000
                self._last_decode_completed_s = decode_completed
                self._fairness().observe_decode(
                    decode_completed - decode_started
                )
                self._sync_decode_fairness_stats()
            else:
                generation_responses = self._generation_batch.next()
            self._gen_tokens_counter += len(generation_responses)
            previous_steps = self._steps_counter
            self._steps_counter += 1
            if _crossed_counter_interval(
                previous_steps,
                self._steps_counter,
                ALLOCATOR_RECLAIM_STEP_INTERVAL,
            ):
                mx.clear_cache()
        if len(self._generation_batch) >= self.completion_batch_size:
            return (prompt_responses, generation_responses)
        # A persistent concept lane is deliberately a request-private B=1
        # route.  Do not start prefill work beside it: that work would later
        # have to merge into the persistent generation batch.
        if getattr(self._generation_batch, "has_persistent_inputs", False):
            return (prompt_responses, generation_responses)
        if self._should_defer_prefill():
            prompt_responses.extend(self._promote_ready_prompts())
            return (prompt_responses, generation_responses)
        (adaptive_defer, adaptive_chunk, _) = self._adaptive_prefill_decision(
            time.perf_counter()
        )
        if adaptive_defer:
            prompt_responses.extend(self._promote_ready_prompts())
            return (prompt_responses, generation_responses)
        n = min(
            self.prefill_batch_size - len(self._prompt_batch),
            self.completion_batch_size - len(self._generation_batch),
            len(self._unprocessed_sequences),
        )
        persistent_head = False
        if getattr(self._prompt_batch, "has_persistent_inputs", False):
            n = 0
        elif n > 0:
            persistent_positions = [
                index
                for index, sequence in enumerate(list(self._unprocessed_sequences)[:n])
                if len(sequence) > 9
                and isinstance(sequence[9], dict)
                and sequence[9].get(_PERSISTENT_DECODE_INPUTS) is not None
            ]
            if persistent_positions:
                first = persistent_positions[0]
                if first == 0:
                    persistent_head = True
                    # Wait for every ordinary lane to drain, then admit this
                    # request alone.  The queue remains FIFO.
                    n = (
                        1
                        if len(self._generation_batch) == 0
                        and len(self._prompt_batch) == 0
                        else 0
                    )
                else:
                    n = min(n, first)
        ordered = None
        if n > 0 and self._prefill_order().enabled and not persistent_head:
            (n, ordered) = self._order_prefill_queue(n)
        n = self._budget_admissible(n)
        if n > 0 and ordered is not None:
            # The ordered picks are the queue head, so the budget above
            # measured exactly the rows admitted here.
            served = [self._unprocessed_sequences[i][0] for i in range(n)]
            self._prompt_batch.extend(self._make_batch(n, indices=list(range(n))))
            self._commit_prefill_order(served, ordered)
        elif n > 0:
            indices = self._select_prefill_indices(n)
            if indices:
                self._prompt_batch.extend(
                    self._make_batch(len(indices), indices=indices)
                )
        elif self._admit_one_chunk_overflow(adaptive_chunk):
            self.scheduler_stats["short_prefill_overflow_admissions"] = (
                self.scheduler_stats.get("short_prefill_overflow_admissions", 0) + 1
            )
        prompt_responses.extend(self._promote_ready_prompts())
        prompts = []
        for i, seq in enumerate(self._currently_processing):
            response = PromptProcessingBatch.Response(
                self._prompt_batch.uids[i], 0, False, False
            )
            segments = seq[0]
            step_size = (
                adaptive_chunk
                if self._bounded_prefill_chunks()
                else self.prefill_step_size
            )
            if len(seq) > 6 and seq[6] is not None:
                step_size = len(segments[0])
            n = min(len(segments[0]), step_size)
            covered = int(seq[4] or 0) + int(seq[1])
            next_checkpoint = self._next_interior_checkpoint(
                self._prompt_batch.uids[i], covered
            )
            if next_checkpoint is not None:
                n = min(n, next_checkpoint - covered)
            prompts.append(segments[0][:n])
            self._record_prefill_chunk(
                self._prompt_batch.uids[i], len(prompts[-1])
            )
            segments[0] = segments[0][n:]
            if len(segments[0]) == 0:
                segments.pop(0)
                response.end_of_segment = True
            seq[1] += len(prompts[-1])
            response.progress = (seq[1], seq[2])
            prompt_responses.append(response)
        self._prompt_tokens_counter += sum((len(p) for p in prompts))
        if prompts:
            self.scheduler_stats["prefill_rounds"] += 1
            if len(self._generation_batch) == 0:
                self.scheduler_stats["prefill_only_rounds"] += 1
            elif self.decode_priority_cadence > 1:
                self.scheduler_stats["decode_priority_release_rounds"] += 1
        tic = time.perf_counter()
        self._prompt_batch.prompt(prompts)
        self._capture_plain_interior_checkpoints()
        for index, sequence in enumerate(self._currently_processing):
            if (
                len(sequence) > 6
                and sequence[6] is not None
                and self._prompt_batch.prefill_inputs[index] is None
            ):
                # The encoder payload is single-use.  Later prompt segments
                # return to the ordinary bounded prefill cadence.
                sequence[6] = None
        toc = time.perf_counter()
        if prompts and self.adaptive_prefill:
            self._mark_prefill_progress(toc)
        self._prompt_time_counter += toc - tic
        if prompts and self.adaptive_prefill and (len(self._generation_batch) > 0):
            width = max((len(prompt) for prompt in prompts))
            measured = (toc - tic) * 1000 / max(width, 1)
            if self._prefill_ms_per_token_ewma is None:
                self._prefill_ms_per_token_ewma = measured
            else:
                self._prefill_ms_per_token_ewma = (
                    0.75 * self._prefill_ms_per_token_ewma + 0.25 * measured
                )
            self.scheduler_stats["adaptive_prefill_release_rounds"] += 1
            histogram = self.scheduler_stats["adaptive_prefill_chunk_histogram"]
            key = str(width)
            histogram[key] = int(histogram.get(key, 0)) + 1
        if prompts:
            self._fairness().observe_prefill(
                max((len(prompt) for prompt in prompts), default=0),
                toc - tic,
                contended=len(self._generation_batch) > 0,
            )
            self._sync_decode_fairness_stats()
        return (prompt_responses, generation_responses)

    def next(self):
        """
        Get the next batch of responses.

        Returns:
            Tuple of prompt processing responses and generation responses.
        """
        with mx.stream(self._stream):
            result = self._next()
        self._observe_adaptive_mtp_responses(result[1])
        for response in result[1]:
            if response.finish_reason:
                self._release_cache_capsule_uid(response.uid)
        return result

    def _observe_adaptive_mtp_responses(self, responses):
        """Publish bounded engagement evidence from completed cohort receipts."""
        for response in responses:
            adaptive = (getattr(response, "mtp_receipt", None) or {}).get(
                "adaptive_depth"
            )
            if not adaptive or not adaptive.get("selected"):
                continue
            for name, value in adaptive.get("counters", {}).items():
                key = f"adaptive_mtp_{name}"
                self.scheduler_stats[key] = max(
                    int(self.scheduler_stats.get(key, 0)), int(value)
                )
            cost_model = adaptive.get("cost_model")
            if isinstance(cost_model, Mapping):
                # Six fixed width buckets and adapter-bounded native depths
                # keep this status tree finite. Prometheus applies its own
                # reviewed label bounds when exporting it.
                self.scheduler_stats["adaptive_mtp_cost_model"] = dict(cost_model)

    def next_generated(self):
        """
        Return only generated tokens ignoring batch generation responses.

        Returns:
            List of GenerationBatch.Response objects
        """
        with mx.stream(self._stream):
            while True:
                (prompt_responses, generation_responses) = self._next()
                if not generation_responses and prompt_responses:
                    continue
                self._observe_adaptive_mtp_responses(generation_responses)
                return generation_responses
