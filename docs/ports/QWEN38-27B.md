# Qwen3.8 27B: port and pre-GPU report

## Status — 2026-09-15

**Implemented candidate with a bounded ordinary GPU probe. No route is fully
qualified, selected, deployed, or installed as the default service.** The
ordinary probe loaded the base oQ4e artifact on macOS 26.7, passed cold/warm
APCv2, client normalization, streaming, tools, reasoning, sampling, B4 batching
and mixed warm requests, then failed closed at the 262K admission gate. After
the dense-family cache bound described below, a frozen-source rerun admitted
and completed the cold 262K request with swap at zero. Its immediate warm replay
then exposed a generic APCv2 disk-pressure race: restoring two neighboring
checkpoints could re-spill the preferred candidate and publish its empty
disk-only placeholder as a cache hit. The generic fix rejects empty placeholders,
validates returned COW segment counts, closes malformed branches, and fails
closed without leaking their leases. CPU regression coverage passes; the fixed
source still requires a fresh isolated GPU rerun. MTP has not been GPU-tested.

The adapter is `mlx2.adapters.qwen38_27b.Qwen3827BAdapter`. Metadata inspection
is available separately as `inspect_artifact(path)` without importing MLX.
The tensor module is `mlx2.runtime.models.qwen38_27b`.

This is the dense `qwen3_5` architecture: 64 layers, comprising 48 GDN recurrent
layers and 16 full GQA attention layers; hidden width 5120, 24 query heads and
4 KV heads of width 256. It has no QSA sparse index, PLE n-gram table, or MoE
experts. Flash-Next-specific QSA/PLE/MoE optimizations therefore do not apply.

## Artifacts available locally

All paths below are under `~/mlx-models`. Their indexed shard
files exist. This is metadata/inventory evidence, not an integrity scan of
all tensor payloads or numerical qualification.

| Directory | Embedded MTP | Candidate use |
|---|---|---|
| `Qwen3.8-27B-oQ4e-mtp` | 29 tensors; one head | Base-model ordinary/MTP qualification target |
| `Qwen3.8-27B-Uncensored-oQ4e-fp16-mtp` | 29 tensors; one head | Existing Hermes artifact parity target |
| `Qwen3.8-27B-MLX-4bit`, `-6bit`, `-8bit` | Absent | Ordinary only until a separately identified compatible head is integrated |
| `Qwen3.8-27B-CRACK-MLX-4bit`, `-8bit`, `-bf16` | Absent | Distinct weight variants; ordinary candidates, separate qualification |
| `Qwen3.8-27B-MTP-BF16`, `Qwen3.8-27B-CRACK-MTP-BF16` | Head-only directories | Not complete models; sidecar loading is not implemented in this port |

The ordinary conversions advertise `mtp_num_hidden_layers=1` despite omitting
MTP weights. Inspection uses the actual weight index and checks essential MTP
components. Headless artifacts expose an ordinary descriptor; requesting MTP
fails before any tensor import. Incomplete heads and unexpected topology fail
closed.

The existing Hermes launcher in Application Support selects the Uncensored
artifact with unified revision `69dcb538965a7b546e9d63fab473c2ad721e1d33` on
port 8283. That file was inspected; this port did not inspect or modify its
live process. The Uncensored artifact's local download metadata records HF
revision `4977e8a5be8a4008a5cbfc6f402f2b3e478ab943`. Its local model card
identifies `pyros-vault/Qwen3.8-27B-Uncensored-oQ4e-fp16-mtp`, derived from
`orcarouter/Qwen3.8-27B-Uncensored`. These are artifact-file claims, not a fresh
external repository check. The base oQ4e card identifies `Qwen/Qwen3.8-27B`;
no immutable upstream download revision was recovered for that local directory.

The two MTP artifacts are distinct weights/tokenizers and must never share a
qualification record. The FP16 suffix does not mean every MTP tensor is FP16:
its local card describes packed Q4 projections with FP16 auxiliary tensors.

Full local inventory and metadata fingerprints are in
`provenance/qwen38-27b-artifacts.json`. Fingerprints hash configuration,
index/tokenizer/template contents and shard name/size/mtime, **not complete
weight payloads**. They inherit that existing startup-identity limitation.

## Modern contracts included

- The sole cache engine remains **APCv2**. The adapter supplies the distinct
  `qwen38-27b-hybrid-layer-segments-v1` layout. Runtime/artifact/tokenizer/layout
  identities form the same revision-bound cache key used by serving.
- Target state uses current `KVCache` and rollback-capable `ArraysCache`;
  embedded MTP has its own KV state. Shared COW freezing, layer segments,
  target/draft publication, byte bounds, eviction and disk restoration are
  reused through the existing service, not reimplemented in the model.
- Shared continuous batching, scheduler, admission, cancellation and request
  lifecycle execute the model through the same adapter interface as Flash-Next.
- Admission uses a Qwen3.8-specific cache bound derived from 16 target GQA
  layers, the optional embedded MTP GQA layer, and 48 fixed GDN layers. It
  charges fp32 KV allocation capacity, two copies of recurrent/conv state for
  rollback, complete observed warm-cache copies plus projected growth, and a
  3.1 GiB dense forward workspace per MTP lane while preserving the common
  20 GiB host/driver reserve.
- Ordinary decode is retained even when an embedded MTP head exists.
- Converted convolution layouts and shifted RMSNorm weights are sanitized
  without double-shifting already converted MTP artifacts.
- Quantization respects per-module overrides, including mixed Q4/Q5 oQ4e.
  Strict tensor loading prevents silently missing model weights.
- Shared Qwen chat-template, tool-call history normalization, incremental
  reasoning/tool output parser, stop handling and streaming detokenizer are
  reused. Vision tensors are excluded from this text serving slice.
- `implemented`, `qualified`, `selected`, and `observed-used` remain separate.
  Descriptor declarations grant no selectable qualification.

## Gaps to resolve before claiming parity

### New mechanism: segmented ordinary-KV execution, GPU-unqualified

The initial port audit found that `build_segmented_batch_cache_group()` rejected
plain `KVCache`: Qwen3.8 27B's full attention and MTP head could only execute
segmented transactions row by row. A generic compute view is now implemented
in `runtime/segmented_plain_kv.py`; the shared segmented-cache builder now registers it.

It retains independent B1 KV ownership, passes per-row offsets to batched
RoPE/projection/trunk execution, appends only valid row tokens and supports
ragged rewind, extraction and membership changes. The generic
`bucketed_attention` seam reduces each private history independently, avoiding
a joined historical KV slab. **The attention reductions themselves remain
per-row SDPA calls**; this is not a new fused segmented-attention Metal kernel.

Nine CPU NumPy oracles execute the production class bodies and ordinary
`KVCache` implementation. They cover independent attention equivalence,
ragged prefix/verification lengths, zero-length rows, rollback then append,
B2→B1, atomic validation, snapshot isolation, stale state and sliding masks.
They cannot validate MLX/Metal numerics, graph ordering or performance. Native
state oracles, full-model acceptance/rollback and observed B2/B4 compute
counters remain mandatory before qualification.

### Common serving integration

The metadata registry in `adapters/registry.py` now resolves this adapter from
validated artifact architecture and checks requested MTP against actual head
presence before model construction. Model-name branching stays outside
scheduler/cache code. No Flash-Next qualification record may authorize this
model. The shared serving and qualification integration remains owned by the
parent task. The adapter
accepts `execution_policy={"num_draft": 1|2|3}` (default two) and exposes
`execution_config(max_lanes=..., prefill_step=...)`; unrelated Flash-Next
policy fields are rejected.

### Model optimizations that may transfer but remain unqualified

| Mechanism | Current candidate state | Required work/evidence |
|---|---|---|
| APCv2/layered COW/disk target-draft pairs | Implemented shared machinery | Cold/warm/divergent prefix, disk roundtrip, mutation isolation and lease release using this model |
| Ordinary continuous batching | Implemented via standard KV/recurrent batch caches | Ragged B1/B2/B4, mixed warm/cold prompts, cancellation and queue progress |
| Segmented MTP and transformed sampling | Per-row lineage candidate | Forced acceptance/rejection, seed behavior, committed state parity and EOS boundary tests |
| True batched segmented MTP | New plain-KV compute view; CPU oracle passed | Native model/state qualification, then observe batched target/draft counters and profile per-row SDPA cost |
| Packed GDN recurrence | Enabled candidate shared kernel | This model's head geometry, dtype, long-prefix rollback and output oracle |
| Native MLX GDN core prefill | Off | Model-specific recurrence parity and measured benefit; candidate toggle is not qualification |
| Dense GDN projection fusion | Source mechanism exists in unified; not mined here | Port guarded quartet fusion/parity probe or shared helper, then check model weights/dtypes and actual dispatch counters |
| Compiled decode | Off; erroneous inherited MoE qualification token removed | Dense-model static-cache/replay contract and numerical qualification first |
| KV quantization | Not selected | Explicit fidelity policy, attention/cache class support and long-context quality tests |
| Prefix-tail/PLD speculation and adaptive draft depth | Shared stack candidates | Model-specific proposer/acceptance and workload measurements; no benefit assumed |
| Long context | Artifact maximum 262144, no mlx2 context domain qualified | Admission-before-prefill, memory bounds and staged context evidence |
| Approximate compaction / active recompute | Not declared | Explicit approximate-state operation and qualification; do not publish as exact APC state |

### Product scope requiring additional development

This port is **text only**. Some source artifacts include image/video weights
and processor files; there is no mlx2 vision encoder/projector, multimodal
position handling, request rendering or multimodal cache identity here.
Vision is not declared and must fail closed. Adding it is a separate serving
slice. External sidecar MTP head merging also remains absent; use an embedded
head artifact for initial MTP work. No DFlash head is required for native
self-MTP on the two complete embedded-head artifacts.

## Qualification order after user-visible gap review

1. CPU metadata/syntax/contract checks (including the implemented shared plain-KV builder hook).
2. Explicitly authorized isolated GPU window; tiny model state oracles first.
3. Ordinary reference: real strict load, text, tools/reasoning, streaming,
   stops, cancellation, mixed batches, context limits and APCv2 disk/COW.
4. MTP: compare committed state to ordinary under forced accept/reject and
   transformed sampling; then true B2/B4 engagement, topology changes, warm
   target/draft pairs and lease cleanup.
5. Measure B1 and B2/B4 separately against the same artifact in optimized
   unified. Admit optimizations only with matching artifact/runtime/settings
   evidence and nonzero mechanism counters.
6. Produce qualification receipts before enabling production selection.

## Validation and provenance

CPU command:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_qwen38_27b_port.py -v
```

The full CPU suite passes after the cache-budget change. Adapter tests cover
import without MLX, artifact/MTP capability detection,
partial/missing/path-traversal rejection, identity invalidation, topology
validation, environment isolation, descriptor scope, profile fail-closed,
syntax, execution-policy isolation, converted/raw norm sanitization and rejection
of inherited FlashNext QSA memory geometry. Dedicated CPU tests cover the dense
family cache projection, 262K admission, ordinary/MTP workspace, warm copies,
growth, reserve preservation and fail-closed topology validation. APCv2 tests
also reproduce the exact two-neighbor forced-disk restore race and verify the
selected recurrent/KV layer order, offsets, malformed-topology miss reason and
COW lease release.
**Nine additional segmented plain-KV CPU oracle tests passed** via
`test_segmented_plain_kv_cpu.py`. A separate native MLX CPU check constructs a
tiny four-layer dense hybrid model and verifies that chunked ordinary decode
and chunked embedded-MTP replay match their single-pass counterparts while
advancing the MTP KV offset exactly. Eight local full artifacts passed
metadata inspection. The actual local base oQ4e tokenizer also rendered a
372-token tool-history roundtrip through the production tokenizer wrapper,
without importing MLX or mutating input history. These are not inference tests.

Mined class closure comes from local unified revision
`1e2bc604f71d070bee970c3e7db8b60f7855599b`. Exact source paths/hashes, modifications
and validation are in `provenance/qwen38-27b.json`. The source root license is
MIT; original copyright notices are retained in `provenance/qwen38-27b.NOTICE`.
Original mlx2 adapter/tests/docs carry no Apple copyright header. Bounded
ordinary pre-fix evidence is under
`qualification/runs/macos-26.7/qwen38-27b/ordinary/`; it is diagnostic evidence,
not a qualification receipt.
