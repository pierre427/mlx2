"""Dense full-attention adapter's explicit native cohort construction."""

from .native_cohort import NativeCohortPreparation, live_identity


class Qwen3NativeCohort:
    def prepare(
        self,
        adapter,
        *,
        profile_path,
        manifest_path,
        mlx_wheel_path,
        tokens,
        environment,
        input_ids=(),
    ):
        from ..runtime.paged_b2_research_profile import (
            DIRECT_FENCE_FLAG,
            PACKED_FLAG,
            SCHEMA,
            SCHEMA_V3,
            SCHEMA_V4,
            SCHEMA_V5,
            SCHEMA_V6,
            SCHEMA_V7,
            SUSTAINED_FLAG,
            VECTOR_ROPE_FLAG,
            load_b2_research_profile,
            validate_combined_environment,
        )

        if any(
            environment.get(key) != "1"
            for key in (
                "MLX2_PAGED_Q1_SIMD_TILE",
                "MLX2_PAGED_GROUPED_Q1_WRITE",
                "MLX2_PAGED_PRIVATE_TAIL_REUSE",
            )
        ):
            raise ValueError("native B2 physical kernel flags are disabled")
        identity = live_identity(adapter, manifest_path, mlx_wheel_path)
        profile = load_b2_research_profile(
            profile_path,
            live_identity=identity,
            context_lengths=tuple(map(len, tokens)),
        )
        combined = validate_combined_environment(profile, environment)
        schema = profile["schema"]
        packed = schema in (SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6, SCHEMA_V7)
        sustained = schema in (SCHEMA_V4, SCHEMA_V5, SCHEMA_V6, SCHEMA_V7)
        vector_rope = schema == SCHEMA_V5
        direct_fence = schema in (SCHEMA_V6, SCHEMA_V7)
        for selected, flag, label in (
            (packed, PACKED_FLAG, "packed prefill"),
            (sustained, SUSTAINED_FLAG, "sustained decode"),
            (vector_rope, VECTOR_ROPE_FLAG, "vector Q1 RoPE"),
            (direct_fence, DIRECT_FENCE_FLAG, "direct grouped fence"),
        ):
            if selected and environment.get(flag) != "1":
                raise ValueError(f"native B2 {label} flag is disabled")
        return NativeCohortPreparation(
            profile,
            identity,
            profile,
            packed=packed,
            neutral_filters_required=schema == SCHEMA,
            options={
                "vector_rope": vector_rope,
                "direct_fence": direct_fence,
                "combined": profile["combined_optimizations"] if combined else {},
                "stripes": profile["q1_simd_stripes"] if combined else 4,
            },
        )

    def allocate(self, adapter, requests, preparation, *, cancelled):
        from ..runtime.qwen3_paged_graph_factory import create_shared_qwen3_graph_pack

        if cancelled():
            raise ValueError("native cohort cancelled before allocation")
        profile, options = preparation.profile, preparation.options
        owners, candidate = create_shared_qwen3_graph_pack(
            adapter,
            tuple(
                (revision, len(tokens), maximum)
                for _, revision, tokens, maximum in requests
            ),
            permit_candidate=True,
            profile_host=True,
        )
        candidate._serving_b2 = True
        candidate._b2_profile_id = profile["profile_id"]
        candidate._b2_packed_prefill = preparation.packed
        candidate._vector_q1_rope = options["vector_rope"]
        combined = options["combined"]
        candidate._defer_staged_q1_eval = combined.get("deferred_eval", False)
        candidate._defer_staged_q1_writes = combined.get("deferred_write_eval", False)
        candidate._b2_inline_metadata = combined.get("inline_metadata", False)
        candidate._b2_grouped_sampler = combined.get("grouped_sampler", False)
        candidate._b2_q1_stripes = options["stripes"]
        candidate._b2_stock_sdpa = combined.get("stock_sdpa", False)
        candidate.backend.direct_grouped_fence = options["direct_fence"]
        return owners, candidate, None
