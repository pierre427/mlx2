"""Physical FP32 hybrid state envelope, including private AST successors.

Mechanism mined from flash_next_memory.py at f73a71d6d3404e125464c1567131825ed1cb3df4;
GLM dimensions and conservative private-tree copies are derived independently.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Glm53CacheBudget:
    recurrent_layers: int = 34
    indexed_layers: int = 11
    snapshot_copies: int = 2
    allocation_step: int = 256
    transient_gib_per_lane: float = 1.76

    @classmethod
    def from_config(cls, config, *, mtp=False, tree_rows=0):
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
        return cls(snapshot_copies=tree_rows + 2)

    @property
    def fixed_bytes(self):
        # A conv3 history of Q/K/V8192 and FP32 [64,128,128] vector state.
        return self.recurrent_layers * (3 * 24576 + 64 * 128 * 128) * 4 + 45 * 4096

    def snapshot_bytes(self, context_tokens):
        if type(context_tokens) is not int or context_tokens < 0:
            raise ValueError("GLM-5.3 context must be a nonnegative integer")
        capacity = ((context_tokens + 255) // 256 + 1) * 256
        # Latent512 and raw(key128, gate128); values have width zero.
        return self.fixed_bytes + self.indexed_layers * capacity * 768 * 4

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
        # dense selected mask/reference attention workspace separately.
        return (
            rows * 4 * (12 * 4096 + 8 * 8192 + 8 * 2048 * 4 + 64 * 512 * 3)
            + rows * (context_tokens + 256) * 8
        )

    def as_dict(self):
        return {
            "schema": "glm53-gguf-cache-geometry-v1",
            **asdict(self),
            "fixed_bytes": self.fixed_bytes,
            "bound": "FP32-capacity-and-private-successor-copies; shared buffers not discounted",
        }
