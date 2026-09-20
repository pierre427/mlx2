# Architecture

`mlx2` keeps policy and request ownership separate from model tensor math.

1. **Descriptors** declare model capabilities and state topology.
2. **Qualification** binds a tested artifact, native runtime and serving settings.
3. **RoutePlanner** selects a declared, qualified capability profile.
4. **ServingEngine** owns bounded requests, cancellation, memory admission,
   prompt checkpoints and response receipts on one model-execution thread.
5. **FlashNextAdapter** loads the model, pins its execution policy, prepares
   prompts and interprets model-specific output conventions.
6. **Runtime mechanisms** execute ordinary/self-MTP batches and own segmented
   cache transactions, fused kernels and file-backed PLE state.

The original CPU control contracts remain independently testable. The serving
adapter exposes the tensor model to the mined batch executor; the earlier
`ExecutionAdapter.prefill_batch/decode_batch` protocol is a future narrow public
backend interface, not a claim that the present implementation uses those calls.

## State ownership

APCv2 is the only serving prefix-cache engine. Its radix index locates frozen
layer/plane segments; borrowed branches retain leases. Revision/model/tokenizer/
layout identities prevent cross-artifact reuse. Target and MTP draft state are
published/restored atomically. Ordinary decode publishes an equivalent
prompt-boundary target checkpoint.

Approximate state never enters APCv2. An approximate-KV lane quantizes a
request-private copy of its attention planes (a warm exact prefix is read,
never written); under self-MTP with `compose_mtp` only the target planes are
quantized and the draft cache stays exact, and segmented self-MTP reduces
those rows through `SegmentedBatchQuantizedKVCache`. Selection of an
approximate operation requires a measured fidelity report
(`runtime/kv_quant_fidelity.py`, see SERVING.md).

Speculative execution uses explicit propose/commit/abort transitions. Per-row
state remains isolated across batching, partial acceptance, cancellation and
reordering. State tests include forced acceptance vectors, stale lineage,
rollback, B2-to-B1 survivors and immutable shared prefixes.

Speculative depth policies are host-only controllers sampled at closed
cohort boundaries (`runtime/adaptive_policy.py`). They choose one depth per
physical cohort and never own cache state.

The optional draft-confidence probe is read-only with respect to the
transaction:
- its device features ride the existing accept-boundary evaluation;
- its greedy lookahead drafts are dropped before verification and trimmed
  with the other draft steps;
- so propose/commit/abort semantics and outputs are unchanged.

External drafters plug into one executor contract (`make_cache`,
`batch_caches`, `append_context`, `draft_distributions` returning the exact
proposal laws). The target model supplies `forward_with_taps`/`prefill_body`
features paired with exactly the tokens its cache consumed; tap `k` is the
output of decoder layer `k`, and a model may define the sentinel
`k == num_hidden_layers` as its final-norm hidden state (North, for an
EAGLE-1 head). Only target-backed positions enter a draft cache, so the draft
plane is always paired with the committed target boundary. Drafters that fuse
a feature with the following token opt in with `requires_context_tokens`.

## Scheduling and receipts

HTTP ingress and pending jobs are bounded. The generation thread coalesces a
small arrival window, then applies memory policy before allocating prompt
state. Live physical footprint floors MLX allocator accounting. The scheduler
honors the admitted subset even when cost ordering differs from arrival order;
it never overrides a budget rejection for a lone request.

Model-specific decisions stay behind the adapter or model capabilities. The
common request lifecycle does not branch on model names. Other models should
supply adapters and qualification evidence for these mechanisms rather than
bring another server/cache product into mlx2.

Concurrent multi-LoRA (default off) keeps this rule. Per-row adapter slots are
bound from batch uids at the ordinary prefill/decode forward seams and read by
wrapped Linear modules (`runtime/multi_lora.py`); the scheduler only pins and
releases slots at admission/finish. Adapter identity is part of the APCv2
namespace, never a model-name branch.

A qualified profile and an observed optimization are distinct. Receipts expose
actual compute widths, MTP acceptance, prefix reuse and route identity. Kernel
counters distinguish selected gates from executed paths. Ordinary decode remains
available as a reference using the same modern state lifecycle.

Speculative proposal sources compose inside one self-MTP verify transaction
rather than as competing routes. With the default-off copy-draft policy, each
lane chooses per round between the MTP head's drafts and a span copied from
its own indexed context. Both are verified by the same batched target forward
under the exact acceptance law. The existing pending-hidden replay lets the
draft head catch up after a copy, so no cache or rollback contract changes.
Per-source counters keep head acceptance, which drives adaptive depth,
separate from copy acceptance.

## Scope of the first model

The selected Flash-Next profile includes APCv2, layered cache segments,
continuous batching, persistent MTP, fused GDN, compiled/file-backed PLE and
pooled QSA keys. Optional indexed/shared-QSA paths retain their qualification
and geometry gates. Strict grammar is implemented behind its own qualification
capability. Compaction, vision and additional model families need separate
vertical slices and evidence.
