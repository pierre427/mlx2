# Laguna XS 2.1 fused-MoE candidate qualification

Artifact fingerprint: `e397a0730641a6fd573706954789f165050411da1742de37347fbc6efaf5f97c`

The two shape-locked Metal candidates are implemented and retained behind
explicit benchmark-only selection:

- affine q8 expert-down projection with weighted top-8 reduction;
- sigmoid, correction-bias, normalized top-8 router.

`results-long.json` is the exact layer-1 benchmark (20 warmups, 100 measured
iterations per arm and width). All candidate arms passed BF16 allclose and hard
engagement gates at M=1, 2, 4, and 8. The isolated M=1 kernels were only about
1.05-1.06x faster, their combination was 0.96x, and widths 2-8 were neutral or
slower.

`full-model.json` is the decisive one-token, all-39-sparse-layer check (3
warmups, 12 measured iterations). Every requested kernel engaged 585 times with
zero fallbacks. Median latency regressed from 9.821 ms stock to 10.802 ms for
fused-down, 10.913 ms for fused-router, and 11.138 ms for both. The candidates
also failed the final-logit allclose and argmax gates (`max_abs` 3.59-3.69).

Decision: implemented, observed-used, but not qualified or selected. The
ordinary MLX reference route remains the default. Candidate admission is
fail-closed and restricted to the exact Laguna XS 2.1 geometry.
