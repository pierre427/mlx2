> **2026-09-15 DFlash2 update:** The historical metadata-only report below predates the CPU implementation, now integrated into mlx2. See [MUSE-DFLASH2.md](MUSE-DFLASH2.md) for the implemented linear draft/verify, batched target/draft projections, exact segmented rollback, APCv2 pairing and CPU evidence. Tiny random-weight models were subsequently constructed and tested on explicit CPU; no production model weights or GPU were loaded. All routes remain unqualified.

# Muse Glimmer: port and pre-GPU report

**2026-09-15 — CPU/static work only. No model was loaded, no MLX tensor module
was imported, and no GPU test was run for this port.** This is an ordinary text
serving candidate. It is not yet a qualified or deployed mlx2 model.

## Model and source identity

The first target is `~/mlx-models/Muse-Glimmer-30B-mlx-4bit`,
native `muse_glimmer` with a nested `muse_glimmer_text` tower. The checked local
metadata describes 52 dense layers, hidden size 6,656, 32 query heads, two KV
heads, head dimension 128, and vocabulary 202,048. There are **39 sliding
attention layers with a 2,048-token window and 13 global attention layers**.
Local layers use RoPE; global layers use NoPE. Attention has an output gate,
scaleless Q/K normalization, centered block norms, and final logit softcapping.
There is **no native self-MTP head**.

Text tensor math was mined from local `mlx-lm-unified` revision
`1e2bc604f71d070bee970c3e7db8b60f7855599b`,
`mlx_lm/models/muse_glimmer.py`. Its repository license is **MIT**; the complete
license and original file notice are retained in
[`muse-glimmer.NOTICE`](../../provenance/muse-glimmer.NOTICE).
Changes and source hash are recorded in
[`muse-glimmer.json`](../../provenance/muse-glimmer.json).

The adapter accepts the local text tower from quantized or BF16 conversions.
Vision weights are discarded by the tensor model's existing sanitizer. Image
and video requests are rejected, and vision is not declared as a capability.
No weights were copied into this repository.

| Local artifact under `~/mlx-models` | Weight bytes on disk | Role |
|---|---:|---|
| `Muse-Glimmer-30B-mlx-4bit` | 21,347,999,171 | Initial target; affine group-64 |
| `Muse-Glimmer-30B-mlx-8bit` | 34,636,881,597 | Target candidate; separately qualify |
| `Muse-Glimmer-30B-mlx-bf16` | 59,553,427,071 | Reference artifact candidate |
| `Muse-Glimmer-30B-assistant` | 5,111,976,608 | Original external drafter |
| `Muse-Glimmer-30B-assistant-q4` | 1,437,872,138 | Quantized original drafter |
| `Muse-Glimmer-30B-assistant-q8` | 2,715,824,278 | Quantized original drafter |
| `Muse-Glimmer-30B-DFlash2` | 5,544,328,424 | DFlash2 external drafter; already present |

These sizes include any vision tensors in target shards; they are not resident
text-model memory measurements. Metadata and file presence were inspected, not
all weight contents. The initial Q4 artifact identity is
`df2301de1c0588378c756c1e9caa42dedeb84860ce6f21ae767d48a494fa99fe`.
Identity covers metadata/tokenizer bytes plus weight filenames, sizes and
modification times. It is not a full weight checksum or an upstream immutable
download revision.

## Implemented slice

- `MuseGlimmerAdapter` exposes the shared adapter lifecycle: artifact identity,
  environment, model, tokenizer, context capacity, cache layout, descriptor,
  profile name, prompt serialization, output parser, diagnostics and cleanup.
- The ordinary model uses **existing mlx2 `KVCache` and `RotatingKVCache`**.
  It declares a topology-bound layered cache layout. The shared engine remains
  **APCv2 only**; no legacy APC or old server is imported.
- Model-specific math stays in `runtime/models/muse_glimmer.py`; shared
  scheduling and cache lifecycle need no Muse model-name branch.
- Configuration/metadata inspection is GPU-free. Unsupported topology and
  unsupported activations fail closed. Weight loading is strict and local-only.
- The parser handles Muse `to=self`/`to=user` channels, ATEM tool calls, stop
  sequences, and markers split across streaming chunks. Tool arguments are
  normalized without mutating incoming history. `<|eom|>` continues the turn;
  `<|eot|>` stops it.
- Reasoning effort is mapped to native low/medium/high strength hints; minimal
  maps to low and xhigh/max/ultra to high. Explicit thinking-off selects the
  direct `to=user` generation header. These are prompt policies, not calibrated
  compute budgets, and remain model-generation qualification candidates.
- `muse-glimmer-apcv2-ordinary` is the candidate ordinary profile. Asking for
  native MTP fails explicitly. The descriptor has **no MTP/segmented-MTP claim**.

| Mechanism | Implemented in this candidate | GPU-qualified | Selected/deployed | Observed on Muse GPU |
|---|---|---|---|---|
| Ordinary target math/loading | Yes | No | No | No |
| Streaming/reasoning/ATEM tools | CPU parser and tokenizer verified | No | No | No |
| Shared continuous batch/scheduler contract | Adapter/cache interfaces wired | No | No | No |
| APCv2 layered COW/memory+disk reuse | Existing shared engine, Muse layout declared | No | No | No |
| DFlash2 speculative execution | **No** | No | No | No |
| Segmented ragged speculative rotating state | **No Muse-qualified implementation** | No | No | No |

## Gaps that must be shown before GPU testing

### Functionality required to complete ordinary serving

1. **Shared dispatch and qualification integration.** The GPU-free adapter
   registry now selects from artifact metadata and rejects requested self-MTP
   before model allocation. Common serving integration must consume this registry.
   Normal serving must require a receipt for this artifact, runtime, settings,
   and adapter. Flash-Next's receipt cannot qualify Muse. Common integration is
   owned by the orchestrator's serving agent, not duplicated here.
2. **Real tensor and state validation.** The original math has been ported, but
   target logits, greedy output, chunked prefill and cache restoration have not
   been tested in mlx2. Test rotary offsets at 2,047/2,048/2,049 tokens and beyond
   multiple wraps. A static interface match is insufficient evidence.
3. **Batch and cache lifecycle qualification.** Exercise cold/warm APCv2,
   immutable prefix forks, concurrent siblings, bounded rotating snapshots,
   disk spill/restore, B1/B2/B4 admission, lane removal, cancellation, and
   recovery with zero leaked leases. Validate different prefix lengths and
   right/left padding for both local and global attention.
4. **Serving feature qualification.** Verify real generated tools, tool-result
   round trips, reasoning channel separation, stops, seed reproducibility,
   context bounds and slow clients. CPU parser tests do not prove model quality.
5. **Memory admission calibration.** Shared policy currently originated in the
   self-MTP lane controller. Muse's dense weights, 39 bounded rotating caches,
   13 growing global caches and optional large drafter require measured costs.
   Reusing Flash-Next lane estimates is not a Muse qualification.

### Major component: DFlash2 exists, but mlx2 integration is missing

There is **no need to train or acquire a DFlash2 head just to begin this work**:
the local artifact already exists. Its config has five layers, hidden size
6,656, eight KV heads, block size 16, mask token 201818, grouped dynamic
convolutions (group 16, kernel 2), selector rank 256/top-16, and target taps
**1, 13, 25, 37, 49**. Its `model_type` is `qwen3`; `architectures` and
`dflash_config` identify it as an external `DFlash2DraftModel`, not a standalone
Qwen model or Muse self-MTP head.

Local implementation candidates were inspected in the separate
`~/Desktop/mlx-uag/worktrees/agnes-vlm-support` checkout at
`8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5`:
`mlx_vlm/speculative/drafters/dflash2/dflash2.py` and
`mlx_vlm/models/muse_glimmer/language.py`. They are **not copied or imported by
this port**, and no existing qualification is transferred automatically.

Required new integration:

- Target hidden-state taps and body-only prefill, plus explicit target
  verification hooks, must cross an adapter-owned boundary.
- Mine the DFlash2 projection/convolution/selector dependency closure and bind
  target/drafter/tokenizer revisions. Preserve an ordinary reference route.
- Provide an exact sampled verification contract using the **actual proposal
  distribution and position-specific RNG**; selector-adjusted proposals cannot
  be labeled native MTP. Greedy agreement alone does not establish sampled
  correctness.
- Pair target state, drafter state, committed tokens and RNG in APCv2 sidecars
  and disk codecs. Define prefix promotion, cancellation and failed-verification
  cleanup as revision-bound transactions.
- **Resolve per-lane rotating rollback before claiming batched speculation.**
  Current `BatchRotatingKVCache.rewind_rows` explicitly refuses different rewind
  lengths because its eviction cursor is shared. Divergent acceptance requires
  independent per-row ring/segment state or safe decomposition with an honest
  execution receipt. QSA private-delta storage does not solve this by name.
- If tree verification is selected, port and validate target masks, explicit
  positions, ancestor visibility and RoPE offsets. A linear draft baseline does
  not qualify tree decoding.

### Potential optimizations that remain unqualified

| Candidate | Missing evidence or implementation |
|---|---|
| DFlash2 vs original assistant, BF16/Q8/Q4 drafter | Modern integration above; compare acceptance, quality and total memory |
| Shared-prefix/segmented **dense** KV attention | Current QSA-specialized paths do not apply to Muse; need a dense/global and rotating cache consumer |
| Fused centered RMSNorm/residual/MLP or compiled decode | Inspect newer local Muse mechanisms, mine with provenance, then numerical and performance qualification |
| Chunked/body-only prefill | Ordinary path works structurally; skip unused vocabulary projection only with an adapter hook and exact checks |
| Prompt lookup/speculative retrieval | Not connected/qualified for Muse; requires exact target verification and rotary rollback gates |
| Global-layer KV8/other compressed cache | Codec, APCv2 restore and attention qualification required; BF16 remains baseline |
| Approximate compaction/recompute | No Muse qualification; cannot publish approximate state through exact operations |
| Longer contexts up to metadata capacity 131,072 | Capacity is a model declaration, not a served or qualified limit |
| Vision/video | Separate model-tower, preprocessing, cache namespace and API port; intentionally absent from text profile |

Qwen-specific QSA, PLE, GDN, MoE and native-MTP kernels are not applicable
optimizations for this dense attention-only target. Their absence is not a
missing Muse feature. General batching, state isolation, prefix reuse,
telemetry and admission are applicable and must be qualified.

## CPU/static verification performed

`python -m pytest tests/test_muse_glimmer_port.py tests/test_adapter_registry.py -q`:
**39 Muse tests and 11 registry tests passed**.
Checks cover import isolation, configuration/layout identity, target/drafter
rejection, MTP refusal, message immutability, modality rejection, every tested
channel split, ATEM types/undeclared calls, stops and malformed output.
Undefined-name lint passes. All tensor modules parse successfully.

Using the **local tokenizer only**, a 131-token tool-history round trip rendered
ATEM arguments, resolved the tool name from `tool_call_id`, honored low reasoning
strength and ended at the expected assistant header. The streaming detokenizer
preserved control tokens; `<|eot|>`=200008 and `<|eom|>`=200007. `mlx.core` remained
absent from `sys.modules`.

Four safetensors **headers only** were read: Q/gate packed weights are
`[4096, 832]`, K is `[256, 832]`, and the output head is `[202048, 832]`, consistent
with group-64 Q4 dimensions. This does not establish numerical correctness,
weight completeness beyond indexed file presence, or GPU performance.

The next GPU session must begin with this report reviewed by the user and with
the model-port CPU-only boundary explicitly lifted. No service or default
endpoint has been changed by this port.
