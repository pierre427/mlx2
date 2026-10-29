"""First-class lossless packed MLX DeepSeek V4 text adapter.

The shared adapter owns the established serving/cache lifecycle. This entry
point requires self-contained MLX storage and exposes explicit optimized
profiles; it never falls back to a GGUF artifact or invents branched AST.
"""
from dataclasses import replace
from pathlib import Path

from .deepseek_v4_gguf import DeepSeekV4GGUFAdapter, DESCRIPTOR as COMMON_DESCRIPTOR
from .deepseek_v4_mlx_artifact import inspect_artifact

MODEL_TYPE = "deepseek4_mlx"
DESCRIPTOR = replace(COMMON_DESCRIPTOR, model_type=MODEL_TYPE,
    variant="vision-exp-packed-mlx-text",metadata={**COMMON_DESCRIPTOR.metadata,
    "execution":"mlx2.adapters.deepseek_v4_mlx.DeepSeekV4MLXAdapter",
    "storage":"lossless MLX safetensors; no GGUF source required",
    "scope":"ordinary B1 text and explicit optimized DSpark chain; unqualified"})


def optimized_policy(model_path, *, index_qat="fused_fp32", external=True):
    """Explicit candidate selection, not a qualified serving default."""
    if index_qat not in ("reference", "fused_fp32"):
        raise ValueError("unsupported optimized index QAT")
    result=dict(deepseek_packed="cooperative",deepseek_hc="fused_fp32",
        deepseek_hc_projection="fixed_fp32",deepseek_attention="packed_visible_rows_fp32",
        deepseek_embedding="file_rows",deepseek_kv="fp8_mixed",deepseek_rope_qat="fused_fp32",
        deepseek_dense_prefill="nax_fp16",deepseek_expert_prefill="segmented_nax_fp16",
        deepseek_index_qat=index_qat)
    if external:
        from .deepseek_v4_mlx_artifact import inspect_sidecar
        sidecar=Path(model_path).expanduser().resolve()/"dspark"
        inspect_sidecar(sidecar)
        result.update(draft_model=str(sidecar),num_draft=5,deepseek_verifier="causal_slab",deepseek_draft_confidence=.6)
    return result


class DeepSeekV4MLXAdapter(DeepSeekV4GGUFAdapter):
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "deepseek-v4-vision-exp-packed-mlx-ordinary-b1"

    def _inspect_model_artifact(self,model_path):
        artifact=inspect_artifact(model_path)
        if artifact["config"]["model_type"]!=MODEL_TYPE:
            raise ValueError("MLX adapter requires the dedicated deepseek4_mlx artifact")
        return artifact
