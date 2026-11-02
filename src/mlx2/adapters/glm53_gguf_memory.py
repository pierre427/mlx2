"""Physical FP32 hybrid state envelope, including private AST successors.

Mechanism mined from flash_next_memory.py at f73a71d6d3404e125464c1567131825ed1cb3df4;
GLM dimensions and conservative private-tree copies are derived independently.
"""

from dataclasses import asdict, dataclass

# Sparse DSA workspaces (docs/design/glm53-sparse-dsa.md 6.1). One live
# gathered prefill block of 32 queries: gathered rows (2051 x 512) plus the
# (64 x 2051) scores and probabilities, fp32. sparse_prefill serializes blocks
# through the ids dependency, so one block is live at a time.
SPARSE_WIDTH = 2051
GATHERED_PREFILL_BLOCK_BYTES = 32 * (SPARSE_WIDTH * 512 + 2 * 64 * SPARSE_WIDTH) * 4
# One decode row group (<= 8 query rows) of gathered latent rows.
GATHERED_DECODE_GROUP_BYTES = 8 * SPARSE_WIDTH * 512 * 4
# fused_v2 (docs/design/glm53-speed-round2.md 1.1, 2.1): no gathered rows are
# materialized; one launch's partials are rows x ceil(2051 / 64) splits x 64
# heads x (512 + 2) fp32. Prompt blocks are 64 rows (PREFILL_BLOCK_V2, one live
# at a time); decode and verify rows go in one launch of at most
# max(8, tree rows + 1) rows.
FUSED_V2_ROW_BYTES = -(-SPARSE_WIDTH // 64) * 64 * (512 + 2) * 4
FUSED_V2_PREFILL_BLOCK_ROWS = 64
V2_PREFILL = tuple(f"gathered_v2_from_{n}" for n in (2049, 3072, 4096, 5120, 6144, 7168, 8192, 12288, 16384))
# Bytes per capacity token per indexed layer: latent 512 + raw indexer 256,
# plus the derived pooled keys (128 per 4 tokens) under incremental pools.
COMMITTED_FLOATS, POOLED_FLOATS = 768, 32


def _sparse_policy(kernel_policy):
    """(prefill law, decode law, pooled plane) for a kernel policy; a law is
    None (dense), "v1" (gathered ops / fused_decode_v1) or "v2" (fused_v2)."""
    if kernel_policy is None:
        return None, None, False
    if not isinstance(kernel_policy, dict):
        raise TypeError("GLM-5.3 cache budget kernel policy must be a mapping")
    prefill = kernel_policy.get("glm53_sparse_prefill", "dense_reference")
    decode = kernel_policy.get("glm53_sparse_attention", "mask_reference")
    pools = kernel_policy.get("glm53_indexer_pools", "recompute")
    if (
        prefill not in ("dense_reference", "gathered_v1_8k", "gathered_v1_16k", "gathered_v1_all", *V2_PREFILL)
        or decode not in ("mask_reference", "gathered_v1", "fused_decode_v1", "fused_v2")
        or pools not in ("recompute", "recompute_fixed_order", "incremental_v1", "incremental_v1_audit")
    ):
        raise ValueError("GLM-5.3 cache budget cannot price an unknown sparse policy")
    law = lambda value, dense, v2: None if value == dense else ("v2" if v2 else "v1")
    return (
        law(prefill, "dense_reference", prefill in V2_PREFILL),
        law(decode, "mask_reference", decode == "fused_v2"),
        pools in ("incremental_v1", "incremental_v1_audit"),
    )


def sparse_workspace_bytes(prefill, decode, tree_rows=0):
    """One live consumer workspace per phase, booked together (as before)."""
    prefill_bytes = {None: 0, "v1": GATHERED_PREFILL_BLOCK_BYTES,
                     "v2": FUSED_V2_PREFILL_BLOCK_ROWS * FUSED_V2_ROW_BYTES}[prefill]
    decode_bytes = {None: 0, "v1": GATHERED_DECODE_GROUP_BYTES,
                    "v2": max(8, tree_rows + 1) * FUSED_V2_ROW_BYTES}[decode]
    return prefill_bytes + decode_bytes


@dataclass(frozen=True)
class Glm53CacheBudget:
    recurrent_layers: int = 34
    indexed_layers: int = 11
    snapshot_copies: int = 2
    allocation_step: int = 256
    transient_gib_per_lane: float = 1.76
    # Gathered sparse attention workspace, booked on every prefill transient.
    sparse_workspace_bytes: int = 0
    # FP32 floats per capacity token per indexed layer (committed + derived).
    indexed_floats_per_token: int = COMMITTED_FLOATS
    # Capacity granularity of the indexed planes: the reference KVCache grows
    # by 256 rows; glm53_kv_append=rotate_v1 reserves glm53_kv_append.BASE_BLOCK
    # rows and keeps RING_BUFFERS resident copies of the committed planes.
    capacity_block: int = 256
    indexed_plane_copies: int = 1

    @classmethod
    def from_config(cls, config, *, mtp=False, tree_rows=0, kernel_policy=None, maximum=False):
        """``kernel_policy`` is the adapter's bound policy; ``maximum`` books
        every sparse workspace and the pooled plane (planning, before any
        policy is known)."""
        kinds = config["layer_types"]
        if (
            mtp
            or len(kinds) != 45
            or kinds.count("linear_attention") != 34
            or kinds.count("deepseek_sparse_attention") != 11
        ):
            raise ValueError(
                "GLM-5.3 memory envelope requires the exact ordinary trunk"
            )
        if type(tree_rows) is not int or not 0 <= tree_rows <= 32:
            raise ValueError("GLM-5.3 private tree row envelope outside 0..32")
        if maximum:
            # The larger of each phase's laws (v2 prefill > v1; v2 decode > v1).
            prefill, decode, pools = "v2", "v2", True
        else:
            prefill, decode, pools = _sparse_policy(kernel_policy)
        kv_append = "reference" if kernel_policy is None else kernel_policy.get("glm53_kv_append", "reference")
        if kv_append not in ("reference", "rotate_v1"):
            raise ValueError("GLM-5.3 cache budget cannot price an unknown KV append policy")
        from ..runtime.models.glm53_kv_append_geometry import BASE_BLOCK, RING_BUFFERS

        return cls(
            snapshot_copies=tree_rows + 2,
            sparse_workspace_bytes=sparse_workspace_bytes(prefill, decode, tree_rows),
            indexed_floats_per_token=COMMITTED_FLOATS + POOLED_FLOATS * pools,
            capacity_block=256 if kv_append == "reference" else BASE_BLOCK,
            indexed_plane_copies=RING_BUFFERS if kv_append == "rotate_v1" else 1,
        )

    @property
    def fixed_bytes(self):
        # A conv3 history of Q/K/V8192 and FP32 [64,128,128] vector state.
        return self.recurrent_layers * (3 * 24576 + 64 * 128 * 128) * 4 + 45 * 4096

    def snapshot_bytes(self, context_tokens):
        if type(context_tokens) is not int or context_tokens < 0:
            raise ValueError("GLM-5.3 context must be a nonnegative integer")
        block = self.capacity_block
        capacity = ((context_tokens + block - 1) // block + 1) * block
        # Latent512 and raw(key128, gate128); values have width zero. Pooled
        # keys (128 per 4-token pool) only when the pooled plane is attached.
        committed = COMMITTED_FLOATS * self.indexed_plane_copies
        floats = committed + self.indexed_floats_per_token - COMMITTED_FLOATS
        return self.fixed_bytes + self.indexed_layers * capacity * floats * 4

    def project(self, context_tokens):
        return (
            self.snapshot_copies * self.snapshot_bytes(context_tokens)
            + context_tokens * 16
        )

    def prefill_transient_bytes(self, context_tokens, chunk_rows):
        self.snapshot_bytes(context_tokens)
        if type(chunk_rows) is not int or chunk_rows < 0:
            raise ValueError("GLM-5.3 chunk rows must be nonnegative integer")
        rows = min(context_tokens, chunk_rows)
        # Bound active-layer HC, KDA projection glue, MoE intermediates and
        # dense selected mask/reference attention workspace separately, plus
        # one live gathered sparse block when a gathered policy is bound.
        return (
            rows * 4 * (12 * 4096 + 8 * 8192 + 8 * 2048 * 4 + 64 * 512 * 3)
            + rows * (context_tokens + 256) * 8
            + (self.sparse_workspace_bytes if rows else 0)
        )

    def as_dict(self):
        return {
            "schema": "glm53-gguf-cache-geometry-v1",
            **asdict(self),
            "fixed_bytes": self.fixed_bytes,
            "bound": "FP32-capacity-and-private-successor-copies; shared buffers not discounted",
        }
