# HySparse2 prototype (arXiv 2609.26368)

Status (2026-09-23): the model is **implemented and CPU-oracle verified** with
random weights. It is **not qualified and not selectable**: there is no adapter,
no registry entry and no route, and the registry still rejects
`model_type: hysparse2`. No checkpoint or reference code has been released.

## The paper in one paragraph

HySparse2 (Xiaomi LLM-Core, 2026-09-22) splits a 49-layer MoE into two halves.
The **self-decoder** has 12 SWA layers, 1 full-attention (FA) layer and 12 more
SWA layers. The **cross-decoder** is 4 blocks of 1 FA layer plus 5 token-sparse
(SA) layers. It shares KV at two levels:

- **KV Bridging.** Each cross FA layer builds its K/V with its own projections
  of the self FA layer's *input* hidden state.
- **KV Reuse.** SA layers own no K/V. They read their block FA layer's cache at
  the tokens that FA layer ranked highest by exact attention: the 128 most
  recent are forced, plus 1024 more by score. Selection is per token, not per
  block.

Every cross-decoder cache therefore comes from self-decoder states, so prefill
can stop after the self-decoder. Attention is MQA (64 query heads, 1 KV head,
head dim 256). FA and SA layers use NoPE. SWA layers use partial RoPE (64 dims),
and SWA and SA layers use sigmoid output gates and per-head sinks.

## Peer ecosystem (pr_index: 24 inference repos, plus HF and GitHub, 2026-09-23)

Nobody implements HySparse, HySparse2 or KV Bridging. The pr_index has no
match, HF has no model and GitHub has no repo. MiMo-V2.6, released 2026-09-20
with day-0 support in sglang, vllm, llama.cpp, mlx-vlm, omlx and Rapid-MLX, is
the paper's *Hybrid SWA* baseline, not HySparse2. The adjacent mechanisms
exist, and these PRs shaped the prototype. The lessons applied:

| Peer | Lesson | Where it landed |
|---|---|---|
| YOCO reference (`microsoft/unilm`) | Prefill `[0, P-1)` with the cross-decoder skipped, then the last token as a decode step | mlx2's `BatchGenerator` loop already does this; laziness prunes the cross-decoder (bench arm `serving_loop`) |
| sglang#27183, mlx-vlm#2157 | Trimming rows in the forward pass means fixing RoPE offsets, masks and per-layer inputs | `SWAAttention.suffix` takes an absolute `start`; the RoPE-offset mutation fails 10 tests |
| vllm#56749 | `argpartition` top-k ties are nondeterministic across batch shapes | `select_tokens` breaks ties toward the lower position and returns ascending indices; the test uses an adversarial `argpartition` |
| sglang#40352 | Force the newest block to `+inf` | Forced local window is `+inf` in `select_tokens` |
| sglang#38755, TensorRT-LLM#19138 | Speculative verify breaks cross-layer sharing first; each row needs its own causal selection | Multi-row verify is tested against sequential decode |
| sglang#32771, TensorRT-LLM#19138 | Pass producer selections within the forward and key them per producer layer | Indices live only inside `_cross_decoder`, per block; nothing is cached |
| transformers#47290, vllm#48365 | Keep the cache topology explicit and fail loudly on unbound consumers | `make_cache` returns caches for self layers plus cross FA layers only; `ModelArgs` rejects unbridged layouts |
| vllm#40108, vllm#55559 | A consumer's KV dtype and quantization scales must follow its producer | Recorded for the quantized-KV slice; not implemented |

## What was built

- `src/mlx2/runtime/models/hysparse2.py`:
  - `ModelArgs` defaults to the paper layout.
  - `select_tokens` does forced window plus top-k with deterministic ties.
  - Layers: `SWAAttention`, `SelfFullAttention`, `CrossFullAttention` (bridge,
    plus explicit probabilities for selection) and `SparseAttention` (a gather
    followed by SDPA with sinks and a gate).
  - `HyperConnection` implements simplified mHC.
  - Two paths share the same weights: `Model.__call__`, the every-layer
    reference used for decode and verify, and `Model.prefill`, the early exit.
- `src/mlx2/runtime/models/hysparse2_cost.py` is an analytic KV and prefill-FLOP
  model for all three designs in the paper.
- `scripts/bench_hysparse2_prefill.py` measures GPU wall-clock at paper scale.
- `tests/test_hysparse2_model.py` has 29 CPU tests.

### Open details, all `ModelArgs` fields

- **Head-score aggregation for MQA selection:** `prob_sum` by default,
  `logit_max` as the alternative.
- **Bridge normalisation:** the bridge uses its own RMSNorm on the source
  layer's pre-norm input.
- **SWA head shape:** the same as FA.
- **mHC:** static pre/post stream weights with identity residual mixing.
- **MLP:** dense SwiGLU in place of the MoE.

Each is a guess until a checkpoint exists.

## Beyond the paper: a suffix bound on the self-decoder

The paper charges the self-decoder's FA layer full quadratic attention during
prefill ("the self-decoder performs full attention in only one layer"). But
everything that layer's *output* feeds is sliding-window: 12 SWA layers, then
the cross-decoder, which prefill evaluates on the last row only. Its *input*
feeds its own cache and the bridges, and those are per-token projections.

The requirement follows by working down from the top SWA layer:

- The top layer must leave `W` exact rows in its cache.
- Each layer below needs `W - 1` more rows than the layer above it.

So the FA layer's output is needed only for the last

    S = W + (n_tail - 1)(W - 1)   (= 1,525 for W = 128, n_tail = 12)

rows. `Model.prefill` therefore:

1. Runs the layers up to the FA layer on every row, writing their caches, the
   FA cache and the bridges.
2. Queries the FA layer on the last `S` rows only.
3. Runs each trailing SWA layer on its shrinking suffix with no history,
   drops the `W - 1` rows that saw a truncated window, and seeds the rotating
   cache.

It is exact. Tests compare the logits, every cache and 12 decode steps against
the reference path. Four targeted mutations break it, and each fails at least
one test.

Prefill attention becomes O(T·S) instead of O(T²), and the self-decoder half
after the FA layer drops out of prefill. The cost model reproduces the paper's
own ratios to 2%: 2.96× against 2.92×, and 5.09× against 5.02×. From the same
model:

| Prompt | Hybrid SWA | HySparse | HySparse2 (paper) | HySparse2 + suffix bound |
|---:|---:|---:|---:|---:|
| 32K | 13.6 | 13.0 | 6.2 | 2.6 GFLOPs/token |
| 128K | 31.7 | 23.0 | 9.4 | 2.6 |
| 1M | 200.8 | 117.0 | 39.5 | 2.6 |

Per-token prefill cost stops growing with context. KV sizes are unchanged:
2.69, 6.72 and 12.09 GB at 1M, which exactly reproduces the paper's figure.

The idea applies to any YOCO-style model whose self-decoder ends in SWA layers
after its last global layer. Check a layout before claiming it: Gemma 3n and
Gemma 4 were not checked.

## GPU measurement

The run is a paper-scale layout with random bf16 weights on an M5 Max. Details
are in `qualification/runs/hysparse2-prefill-20260923/`. The table shows
time to first-token logits:

| Prompt | mlx2 serving loop | paper exit | suffix bound | gain |
|---:|---:|---:|---:|---:|
| 4K | 0.88 s | 0.88 s | 0.53 s | 1.66× |
| 16K | 3.95 s | 4.06 s | 2.03 s | 2.00× |
| 64K | 21.3 s | 22.2 s | 9.06 s | 2.46× |
| 128K | 53.0 s | 55.7 s | 18.2 s | 3.05× |

- **mlx2 already gets the paper's exit.** The existing serving loop gets it for
  free through MLX laziness.
- **The suffix bound's time is linear in the prompt.** The FLOP model
  predicted 1.67× and 2.09× at 4K and 16K. It over-predicts at 64K and 128K,
  because long attention runs at a higher efficiency than the projections.
- **Exact on the GPU in fp32:** both exits match the serving loop to within
  1.5e-5 at 32K.

## Not done (next slices, in order)

1. **Serving adapter.** `HySparse2Adapter`, a registry resolver, a
   `ModelDescriptor` with `Capability` limited to ordinary decode, and an
   APCv2 `cache_layout` hash over the layer layout, the bridge sources and the
   selection parameters. The model and cache classes are already
   APCv2-classifiable (plain `KVCache` and `RotatingKVCache`; sparse layers own
   no cache).
2. **Suffix bound in serving.** `BatchGenerator._process_prompts` feeds chunks
   without the prompt's remaining length, so serving currently gets only the
   paper's exit (via laziness), not the suffix bound. It needs a model-level
   prefill hook that sees the whole B=1 prompt.
3. **Batched caches.** Left padding in the bridged and sparse paths, including
   selection indices relative to each row's start (the vllm#56749 follow-up
   bug).
4. **Speculative rollback.** Trim the bridged `KVCache`s together with the
   self-decoder caches.
5. **Fused token-sparse attention kernel.** The current gather materialises
   `(B·L, K, D)`, and the cross FA computes explicit probabilities.
6. **Quantized KV.** Consumers inherit the producer's dtype and scales.
7. **MTP.** The paper's one MTP layer is conditioned on the hidden state at the
   self/cross boundary (Section 3.5). Once prefill exits there, that hidden
   state is available early, so drafting could start without waiting for the
   cross-decoder. Not built.

No bug in shared or upstream code was found or fixed here, so there is no
upstream candidate.
