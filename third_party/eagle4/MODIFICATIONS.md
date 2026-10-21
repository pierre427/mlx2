# Eagle 4 modification notice and change log

MODIFICATION NOTICE. The files listed below have been materially modified from
Eagle 4. This change log identifies the modifying person or entity, the date of
each material modification, the affected files, and a summary of the material
changes. This modified version is not an official Eagle 4 release, its
modifications have not been reviewed or approved by Eagle AI Corp, and it is
provided subject to the Eagle 4 Source-Available License in `LICENSE`.

This project includes or is derived from Eagle 4 technology developed by
Eagle AI Corp. It is not an official Eagle AI Corp project unless expressly
stated otherwise.

- Source: <https://github.com/eagle-hq/EAGLE-4>, revision
  `75436712e7415d352cf9f17d070019d07410786f` (`eagle_harness/models/eagle4_config.py`,
  `eagle_harness/models/low_rank_head.py`, `eagle_harness/sglang/models/llama_eagle4.py`,
  `docs/architecture.md`, `docs/new-model.md`).
- Modified by: Pierre Lamy, for the mlx2 project (mlx2 contributors).

## Change log

### 2026-10-21: MLX port of the dense Qwen3.8 drafter

- `src/mlx2/runtime/drafters/eagle4.py`: MLX port of the released dense Qwen3.8
  EAGLE-4 draft architecture (three normalized target taps projected into one
  Llama-style decoder layer whose q/k/v read concatenated normalized next-token
  embeddings and draft features), sliding-window draft state, low-rank
  candidate selection with exact candidate-row rescoring, an explicit chain
  proposal law and parent-first tree proposals for mlx2's AST verifier. Torch,
  Triton, CUDA and the engine plugins are replaced by MLX modules; latent MoE,
  NoPE, alternate fusion and runtime draft quantization fail closed.
- `src/mlx2/adapters/eagle4.py`: strict checkpoint configuration and tensor
  schema checks, payload-bound revision, target compatibility checks and MLX
  weight loading.
- `src/mlx2/adapters/external_drafter_formats.py` and `tests/test_eagle4_cpu.py`
  (original mlx2 code under Apache-2.0, listed for completeness):
  architecture-driven dispatch of `LlamaForCausalLMEagle4` artifacts to the
  files above, and CPU contract tests for them.

### 2026-10-21: model-free configuration module

- `src/mlx2/runtime/drafters/eagle4_config.py` (new, split from
  `runtime/drafters/eagle4.py`, which re-exports it): the checkpoint
  configuration and tensor schema, importable without MLX or model modules so
  artifact inspection can run before a target adapter pins its import-time
  environment.
- `src/mlx2/adapters/eagle4.py`: imports the configuration from the new module.

### 2026-10-21: proposal-tree policy

- `src/mlx2/runtime/drafters/eagle4.py`, `src/mlx2/runtime/drafters/eagle4_config.py`:
  the proposal-tree shape (`eagle4_tree`: children, depth and node cap) is an
  mlx2 policy, validated, bound into the drafter fingerprint and reported in
  route receipts. The default shape is unchanged.

mlx2 retains ownership of scheduling, target verification, exact target
sampling and acceptance, state commit, prefix caching, route selection and
receipts; those parts are original mlx2 code under the Apache License 2.0 and
are not Eagle Files. The upstream CUDA/ROCm engines, kernels and native
library are not included.
