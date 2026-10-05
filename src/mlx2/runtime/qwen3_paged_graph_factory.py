"""Research-only shared native arena for a true packed Qwen3 B1/B2 graph."""

from __future__ import annotations

import os

from ..adapters.qwen3_paged_candidate import Qwen3PackedCandidate
from .paged_attention_plan import PAGE_SIZE
from .paged_kv_pool import PagedKVPool
from .paged_kv_token import PagedKVTokenOwner, TokenKVProfile
from .paged_kv_write import NativeWriteBackend, PagedKVWriteOwner
from .paged_native_atomic_owner import NativeAtomicRequestOwner
from .qwen3_paged_native_backend import NativeQwen3PagedBackend


def create_shared_qwen3_graph_pack(adapter, requests: tuple[tuple[str, int, int], ...],
                                   *, permit_candidate: bool = False,
                                   profile_host: bool = False):
    """Return independent owners and one candidate sharing a bounded arena.

    Each request tuple is ``(revision, prompt_tokens, max_tokens)``. This
    factory does not select a scheduler route or publish a performance price.
    Caller owns all request owners and their shared writer until full terminal
    retirement; a poisoned writer requires one-way failed-arena teardown.
    """
    if not permit_candidate:
        raise RuntimeError("shared Qwen3 graph requires explicit enablement")
    import mlx.core as mx

    if (type(requests) is not tuple or not 1 <= len(requests) <= 2 or
            any(type(item) is not tuple or len(item) != 3 or
                not isinstance(item[0], str) or not item[0] or
                type(item[1]) is not int or item[1] < 1 or
                type(item[2]) is not int or item[2] < 1 for item in requests)):
        raise ValueError("one or two exact revision and token bounds are required")
    model = adapter.model
    config = adapter.config
    if (config.get("model_type") != "qwen3" or config.get("quantization") or
            config.get("quantization_config") or config.get("sliding_window") or
            config.get("use_sliding_window") or config.get("rope_scaling") or
            getattr(model, "_target_verify_row_exact", False) or
            model.model.embed_tokens.weight.dtype != mx.float16 or
            model.args.num_experts or model.args.head_dim not in (128, 256)):
        raise ValueError("shared graph requires dense fp16 full-attention Qwen3")
    profile = TokenKVProfile(model.args.num_key_value_heads,
                             model.args.head_dim, "float16")
    layers = len(model.layers)
    capacity = sum(layers * ((prompt + maximum + PAGE_SIZE - 1) // PAGE_SIZE + 2)
                   for _, prompt, maximum in requests)
    plane_bytes = capacity * profile.page_bytes
    if plane_bytes > 12 * (1 << 30):
        raise MemoryError("shared graph exceeds 24 GiB paired arena cap")
    pool = PagedKVPool(capacity)
    native = NativeWriteBackend(plane_bytes, mx.default_stream(mx.gpu),
                                permit_candidate=True)
    writer = PagedKVWriteOwner(pool, native, page_bytes=profile.page_bytes,
                               permit_candidate=True)
    backend = NativeQwen3PagedBackend(writer, permit_candidate=True,
                                      profile_host=profile_host)
    owners = []
    try:
        for revision, _, _ in requests:
            public = tuple(PagedKVTokenOwner(writer, profile, permit_candidate=True)
                           for _ in range(layers))
            try:
                owners.append(NativeAtomicRequestOwner(
                    revision, public, {}, supported_planes=("kv",), enabled=True,
                    reuse_private_tail=(
                        os.environ.get("MLX2_PAGED_PRIVATE_TAIL_REUSE") == "1")))
            except BaseException:
                for layer in public:
                    layer.close()
                raise
    except BaseException:
        for owner in owners:
            owner.close()
            owner.reap_retired()
        raise
    candidate = Qwen3PackedCandidate(model, backend)
    candidate._research_staged_graph = True
    return tuple(owners), candidate
