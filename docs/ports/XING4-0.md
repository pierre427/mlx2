# Xing4.0-29B-A4B port

## Current state

Xing4.0-29B-A4B (`XingChen-AGI/Xing4.0-29B-A4B`, revision
`baae3c3e813cad5f888f1f485cfff659c89076c5`) has a native mlx2 adapter with
ordinary, native self-MTP, and prompt-lookup routes. All five candidate routes
(6-bit ordinary, MTP depth 1 and prompt lookup; bf16 ordinary and MTP depth 1)
passed the full serving qualifier in `--qualification-mode` on 2026-09-18.
They remain **unselected** until those records are approved against the
committed revision (the run bound an uncommitted tree).

Artifacts (built by `scripts/convert_xing4_0.py`, tokenizer by
`scripts/xing4_0_tokenizer.py`):

| Artifact | Size | Notes |
|---|---|---|
| `~/mlx-models/Xing4.0-29B-A4B-mlx-bf16` | 60.6 GB, 16 shards | checkpoint bf16 |
| `~/mlx-models/Xing4.0-29B-A4B-mlx-6bit` | 24.9 GB, 6 shards | affine g64; embeddings and heads 8-bit; router, correction bias and mHC operands unquantized |

4-bit was not built (too lossy for this model, per the port brief).

## Architecture

- DeepSeek-V3 lineage: MLA in all 40 layers (32 heads, q-rank 768, kv-rank
  512, 128 nope + 64 rope, v 128), YaRN ×64 to 262,144 tokens, noaux_tc
  sigmoid router (64 experts, top-4, one shared expert, `n_group=1`), two
  leading dense layers.
- **mHC**: four residual streams per token mixed by learned
  pre/post/combine weights with a 20-iteration Sinkhorn projection; no plain
  residual. Embedding is copied into four streams; the output averages them.
  Operands are computed in fp32 as vLLM and SGLang do (HF rounds to bf16).
  On the GPU fused Metal kernels run it (see Performance work); the compiled
  and eager paths remain as fallback and reference.
- One embedded DeepSeek-V3 MTP layer (no mHC). Its embedding and output head
  are bit-identical to the trunk's in the released checkpoint and are shared;
  the converter records that decision and the loader refuses artifacts where
  the tensors and the recorded decision disagree.

## Cache, batching and speculation

mHC streams exist only inside one forward pass. The only cross-token state is
the MLA latent (normed 512-d latent + 64-d roped key per token per layer) in
plain `KVCache` rows, so APCv2 prefix reuse, COW branches, rollback/trim and
batched caches are the shared runtime's, unchanged. The cache budget charges
576 fp32 values per token per layer (~92 KB/token for 40 layers, plus one
layer for MTP).

Segmented continuous batching (`segment_aware_live_tip`) needed one generic
addition: `SegmentedBatchKVCache.row_views()` exposes each row's own history
and mask so MLA, which reads the latent as both key and value, attends per row
without storing the latent twice. Greedy self-MTP matches ordinary greedy
decode exactly for a starting cohort, for a lane joining a running cohort
(both batching modes), and for lanes resumed from an APCv2 hit with their MTP
sidecar on concurrent COW branches (`tests/test_xing_serving_mtp.py`).

Porting this surfaced a model-neutral scheduler bug, fixed here: a lane parked
by the segmented width lock emitted its second token before its prepared first
token once the resident cohort drained (`tests/test_segmented_late_join_order.py`,
reproduced on tiny Qwen4 as well).

Known shared-runtime limit (not Xing-specific): the MTP route serves an APCv2
entry only when the new prompt covers the entire stored boundary with its
sidecar, so a long prompt that diverges near its end re-prefills in full on
the MTP route while the ordinary route reuses the prefix.

## Tokenizer and output

The vendor ships a slow SentencePiece tokenizer with remote code. mlx2 builds
a fast `tokenizer.json` that reproduces it id for id (including the per-chunk
leading `▁`, digit splitting, byte fallback, unregistered `<_observation>`
encoded as text) and stamps the files that passed parity; serving loads it
without remote code and fails closed on a missing or stale stamp. The slow
reference is an explicit opt-in (`tokenizer_reference_fallback`). Parity:
254,983 strings and 634,966 decode checks, all identical; at full scale the
adapter's rendered prompts equal the HF reference's for all probe prompts.

Thinking defaults on, as in the vendor template; `enable_thinking=false` or
`reasoning_effort="none"` turns it off. `</think>` is one token (id 10), so
structured output defers its grammar past reasoning instead of forcing
thinking off. Tool calls use the vendor tag format and are parsed with
vLLM/SGLang semantics, with one deliberate deviation: a call whose payload is
structurally complete (every parameter pair closed, or a whole JSON object)
but is ended by the turn end instead of `</tool_call>` is accepted. The
reference parsers drop such a call as plain text; the model does emit it
(GPU: warm-prefix numerics flipped `</tool_call>` to `<_end>`). Truncated calls
still fail closed.

## Sampling defaults

From `generation_config.json` and the model card: general temperature 1.0,
top_p 0.95, repetition_penalty 1.05; coding/agent (`sampling_profile:
"coding"` or `"agent"`) temperature 0.8, top_p 0.95, repetition_penalty 1.05.
Applied only to fields a request leaves unset.

## Int8 NAX prefill

`--int8-prefill {off,mlp,all}` (default off, APPROXIMATE fidelity, own APCv2
namespace). Xing declares both scopes. Routed experts stay on stock kernels.

## Evidence

### Full-scale parity vs the HF torch reference (bf16, CPU)

Three chats × thinking on/off, last-64-position logits and 16 greedy tokens:

| Artifact | Top-1 agreement | Mean KL | Greedy match |
|---|---|---|---|
| bf16 | 100% | 0.003–0.012 | 16/16 on all six |
| 6-bit | 86–100% | 0.014–0.022 | 16/16 on five, 12/16 on one |

### Performance work

A component ablation on the 6-bit artifact showed the mHC was 53% of a B=1
decode step and 35% of a 2k prefill, far above its memory-bound cost. Two
changes followed:

- **Fused Metal mHC kernels** (`runtime/models/xing4_0_mhc_metal.py`): one
  threadgroup per token computes the RMS norm, 24 mixing logits, gates,
  Sinkhorn projection and collapse; a second kernel applies the stream update.
  In the trunk each update → next pre → RMS norm runs as one compiled span.
  fp32 accumulation; KL to the HF reference 4.5e-3 against 5.5e-3 for the
  compiled path. Each specialization is evaluated inside a fallback guard on
  first use, before any cache write; `MLX2_XING_MHC_KERNEL=0` disables it.
- **Heads-as-queries absorbed MLA**: all 32 heads share one latent head, so
  decode and verify attend `[B, 1, L*H, 512]` against the latent once
  (2.5x at B=4, 7.5x for a 2-token verify block).

Model step (6-bit, isolated benchmark): decode B=1 30.9 → 19.2 ms, B=4
48.2 → 28.5 ms, prefill 2k ~1.3x. The MLA prefill gate already picks the
faster path everywhere; a concatenated 192-dim SDPA and the absorbed path are
both slower for long prefill chunks.

### GPU serving (M5 Max, `qualification/runs/xing4-0-gpu-20260918`)

Greedy, thinking off, neutral penalties; single samples. "Before" is the
first campaign (compiled mHC), "after" the final code.

| Route | Single stream tok/s | 4-way aggregate tok/s | 7k-word prompt TTFT |
|---|---|---|---|
| 6-bit ordinary | 34.4 → **59.4** | 98.6 → **150.0** | 6.0 → 5.7 s (repeat 0.12 s, APCv2 hit) |
| 6-bit MTP depth 1 | 46.2 → **66.8** | 101.1 → **123.0** | 6.0 → 5.7 s |
| 6-bit MTP depth 2* | 48.8 → 73.6 | 92.7 → 121.4 | 7.7 → 5.8 s |
| 6-bit int8 prefill `mlp`* | 34.5 → 59.6 | 97.1 → 156.9 | 5.9 → 5.2 s |
| bf16 ordinary | 28.4 → **41.9** | 75.8 → **102.7** | 7.1 → 5.6 s |
| bf16 MTP depth 1 | 35.7 → **49.7** | 54.1 → 61.6 | 7.4 → 5.7 s |

\* May overlap another session's GPU work; the two final passes agree
within a few percent.

Readings: native MTP is the single-stream route; under 4-way load ordinary
decode now wins at both precisions, so a load-aware MTP policy is the next
routing lever. Int8 NAX prefill is correct and engages as declared but moves
TTFT little on Xing (routed experts stay on stock kernels).

Prompt lookup (6-bit): 584 proposed, 317 accepted, 39 rollbacks; outputs
correct under concurrency.

### Qualification (`--qualification-mode`, committed harness)

| Route | Checks |
|---|---|
| 6-bit native MTP depth 1 | 37/37 |
| 6-bit ordinary | 29/29 |
| 6-bit prompt lookup (batched verify) | 32/32 |
| bf16 native MTP depth 1 | 37/37 |
| bf16 ordinary | 29/29 |

Serving checks keep a thinking-by-default model's reasoning on and add a
thinking allowance to their answer budgets.  Xing reasons at length but not
pathologically: with 8,192 tokens and no guard every probe closed `</think>`
and answered, open-ended prompts taking ~1.2K reasoning plus ~1.1K answer
tokens with low repetition (`thinking-length/study.json`).  The adapter
therefore declares `thinking_allowance_tokens = 4096` (reported in
`/v1/status`; other models keep the harness default of 2,048), and
`mixed_warm` is judged against its own sequential warm-up instead of a fixed
30 s wall clock (Xing: 44-56 s concurrent against 57-90 s sequential).

Defects found and fixed on the way, each with a regression test:
structured-output pieces were isolated decodes (SentencePiece `▁▁`
undercounted whitespace and dead-ended bounded grammars); the segmented
scheduler emitted a parked lane's second token first; the reasoning probe
embedded a literal `</think>` control token; the warm-hit admission estimate
divided a trimmed long checkpoint's full buffers by its few shared template
tokens (hundreds of GiB projected, HTTP 429 on an idle server - the earlier
"unreproduced" PLD 429); and the batched prompt-lookup verify cache lacked the
per-row view MLA needs.

## Open items

- APCv2 on the MTP route serves only full-boundary hits with their sidecar; a
  partial hit could reuse the target cache with a cold draft cache (exact,
  since the target verifies every draft).
- Under load, ordinary decode beats MTP; a load-aware MTP depth policy.
- A custom MLA flash kernel (prefill attention is ~2x its compute floor) and a
  grouped W6A8 NAX expert kernel (no per-call requant) are the next levers.

