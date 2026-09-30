# Peer-informed correctness and memory sweep — 2026-09-29

This audit started at mlx2 `376b5f53efc3d0355d9a51a9040438461cff5609` in an
isolated checkout. The original working directory and running services were
left alone. CPG was not used. Validation used CPU tensors and import-free
metadata tests; no model serving, GPU benchmarks, or new route qualification
was performed.

## Reproduced defects and changes

| Area | Failure | Correction and regression coverage |
| --- | --- | --- |
| APCv2 parked sessions | Replacing a pinned persisted entry ignored a failed payload/manifest save and deleted the only durable old snapshot. | Keep the old COW owner and files until the replacement is durable. On failure restore the trie entry and byte accounting and return `stored=False`. Four fault-injection cases cover resident/disk-only entries, both save boundaries, crash recovery and a successful retry. |
| APCv2 ownership | Closing an empty or persistent cache did not revoke existing capsule generations; the closed object could accept work after releasing its directory lock. | Advance the generation on close, reject lookup/store, and refuse new capsule reservations. Test both persistent and nonpersistent owners. |
| Speculative lane admission | The verification cap charged anchors for queued lanes, unnecessarily disabling MTP under concurrency. Depth-zero selection did not enforce the row cap. | Charge selected lanes `1 + k` rows; queued lanes consume none. Apply the cap to depth-zero execution too. Tests use widths 3, 8 and 20 with a six-row budget, and a two-row plain budget. |
| Re-admission accounting | Hysteresis could hold lanes in the queue while receipts still included their primary rows and memory estimates. | Subtract held-lane costs and primary rows. Preserve existing bounded holds, margin, and re-entry behavior. |
| MoE conversion | `SwitchLinear.to_quantized` initialized and quantized a random full expert bank before overwriting it with the real bank. | Build the quantized module directly from the source parameters; freeze the resulting parameters. Test exact packed weights/scales/biases and forbid random allocation during conversion. |
| Streamed expert memory | An oversized working set retained unrelated cached experts during page-in; materialization failure could leave the cache above capacity. | Evict unrelated rows before reading, release consumed host payloads, and trim/update resident accounting in `finally`. Tests cover success and injected failure. The caller still retains every expert required for its forward. |
| Nemotron grouped routing | Zero-masking discarded groups could select them when all retained corrected scores were negative. Group top-2 was also unnecessary when all groups were retained. | Mask with negative infinity, bypass unnecessary group pruning, validate routing geometry, and preserve explicit zero scaling. Test batched selected experts/weights, invalid geometry, and all-groups/zero-scale behavior. |
| Vision workspace | Building every Gemma 3n/MiniCPM-o chunk lazily retained all tower graphs despite configured chunk bounds. | Evaluate each completed feature chunk before constructing the next. Tests verify execution order, final partial chunks, mixed shapes and original sample/image order. Final feature storage remains proportional to the inputs. |
| LTX video publication | Timeout left a partial destination; a competing writer could be overwritten or deleted after the initial existence check. | Render in a private same-filesystem directory, validate output, then publish by an exclusive hard link. Tests cover success, timeout cleanup and a competing writer. |

These are original fixes to the existing mlx2 implementations. No code was
copied from peer PRs. Existing source provenance and licenses remain in place.
The quantization, vision, and expert-streaming changes remove identifiable
allocations or retained work; this audit does not claim a measured GPU speedup.

## Peer patterns and local history

The live PR MCP index was queried on September 29. The exact PR heads and
states are recorded in [peer-pr-snapshot.json](peer-pr-snapshot.json). Draft
and unmerged PR descriptions were used as leads, not accepted performance or
correctness evidence for mlx2.

| Peer evidence | Pattern applied to mlx2 |
| --- | --- |
| [omlx #4081](https://github.com/jundot/omlx/pull/4081), draft | Ownership lifetimes during cache extraction/merge; temporary versus steady-state memory. This led to inspecting APC snapshot replacement and streamed expert overflow. |
| [omlx #3918](https://github.com/jundot/omlx/pull/3918), closed unmerged | Double charging resident KV versus incremental transient memory. Checked against local resident-cache and pending-byte accounting; fixed independent verification-row overcharging. |
| [omlx #4031](https://github.com/jundot/omlx/pull/4031), merged | Late joins after filtering, transfer versus history replay, and state ownership across ordinary handoff. Reviewed segmented joins, queued-lane recovery and handoff tests. |
| [omlx #4050](https://github.com/jundot/omlx/pull/4050), merged | Greedy selection and low-precision tie behavior; row-alignment constraints. Local ordinary and hybrid temperature/logprob paths already cast logits to float32. |
| [omlx #4047](https://github.com/jundot/omlx/pull/4047), draft; [sglang #41759](https://github.com/sgl-project/sglang/pull/41759), open | MoE verify-window geometry and padding bounds. Reviewed local sorted gathers, shape-locked fused kernels, and unfused expert families. Fixed grouped routing and conversion allocations. |
| [omlx #3955](https://github.com/jundot/omlx/pull/3955), open; [vLLM #55203](https://github.com/vllm-project/vllm/pull/55203), draft | Feature-cache budgets and processor-option identity. Reviewed local revision/processor/media/tower-digest identities and cold-feature admission; existing tests cover these contracts. |
| [sglang #41768](https://github.com/sgl-project/sglang/pull/41768), open | Media ordering under multiple outputs. Checked typed media, sample order, shape grouping and vision chunk output order. |

Local history concentrates on the same seams: cache ownership and headroom,
hybrid cache estimates (`95aef357`, `add54c32`), expert projection memory
(`b698a001`), and request-scoped speculation (`5c3837a1`, `baf3c1d3`). The
request-to-request latch reset is already implemented. The controller keeps
K=0 re-entry pending while admission allows no draft, performs bounded probes,
and uses width-specific learned state. A configured ordinary handoff is
explicitly one-way; it is not a temporarily parked MTP lane. Existing adaptive,
admission, serving, and segmented-MTP tests exercise these distinctions.

## Media scope and fusion opportunities

Image generation/editing is implemented through the pinned Qwen-Image
backend. Video generation uses the LTX adapter. Multimodal understanding
bridges include Gemma, MiniCPM-o, SmolVLM and Qwen families, with separate
qualification gates. Audio input and Nemotron diarization are present. No
music-generation implementation was found in `src` or `scripts`; historical
Music3 service references do not establish mlx2 support.

The following are **unimplemented candidates**, not selected optimizations:

1. Fuse expert down-projection and weighted top-k reduction for Nemotron,
   North/Cohere, Granite and GPT-OSS. Their expert counts, activation/bias
   contracts and output accumulation differ. Existing Qwen/Laguna kernels are
   shape-locked and cannot simply be enabled for them.
2. Fuse scatter-unsort with expert weighting/reduction on the sorted-gather
   route to avoid materializing a full unsorted `[rows, top_k, hidden]` output.
   Validate non-aligned tails, repeated expert IDs, dtypes and reduction order.
3. Extend existing fused verify-window kernels only for proven `(batch,
   tokens, quantization)` layouts. Flattening rows without preserving the
   ordinary-decode reference may change token math and does not establish
   multi-lane equivalence.

SwiGLU already uses a shapeless compiled function, so wrapping the same
elementwise operations in another compile call would not establish a new
fusion benefit. Candidate kernel work requires device parity, memory/occupancy
measurements and model/HTTP qualification before changing route defaults.

## Validation

`tests/test_bug_sweep_sep29.py` contains 26 new CPU regressions. Defect-focused
negative controls failed before the corresponding fixes. The suite also
repairs three existing test assumptions: fake promotion tickets must not
allocate real GPU streams; the tokenizer EOS fixture must supply the sidecar
verification boundary; and the CPU kernel-admission test accepts the valid
Metal-unavailable refusal reason.

The broad suite must run in two processes: metadata tests assert MLX has never
been imported, while tensor tests intentionally import it on CPU. Mixing them
in a single process produces import-guard failures. The first diagnostic run
also exposed those stale fixtures; its result is not counted as a clean pass.

Reproduction with the project's development Python:

```sh
python docs/research/bug-sweep-20260929/validate_cpu.py /tmp/mlx2-cpu-sweep
```

The driver writes a log and JUnit XML for each group. Its metadata group rejects
MLX imports; its runtime group uses CPU tensors, rejects GPU device/stream
creation, and disables Metal-gated test cases. A focused invocation is also
available:

```sh
python docs/research/bug-sweep-20260929/validate_cpu.py runtime tests/test_bug_sweep_sep29.py -q
```

Final integrated validation on parent `68c89ce9` passed
5,312 tests and 88 subtests, with 204 GPU/optional tests skipped.
No failures or errors remained. Final counts, source hashes and integration verification are recorded in
`validation.json` alongside this report. GPU skips are deliberate; CPU tests
do not establish model-level quality, kernel throughput, or serving-route
qualification.
