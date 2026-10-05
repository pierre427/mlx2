# Native paged KV arena candidate

This internal, default-off prototype targets the installed MLX
`0.32.2.dev20260919+39400a0d4` wheel. It keeps K and V byte planes inside
one opaque C++ owner. The optional Python extension accepts MLX arrays and an
explicit stream, but does not expose unrestricted writable arena views or a
serving route. Importing the extension allocates no arena and submits no GPU
work.

The host `PagedKVWriteOwner` validates page-local bounds and generation,
then pins the page before submission. `NativeWriteBackend` calls the native
write with a declared byte count and runs `mx.async_eval` on the returned
MLX dependency. The native write validates uint8 dtype, one-dimensional
shape, contiguity, equal and declared byte lengths, bounds, nonzero epoch,
and GPU stream. The caller must serialize all owner/pool operations under
one host lock and retain the dependency in every later reader graph.

The primitive binds the shared K/V arena descriptors through
`set_output_array` and sources through `set_input_array`. It queues terminal
status for the containing command buffer without calling Python from the
Metal callback. The host consumes terminal events and retires the ledger
lease; a failed or ambiguous submission poisons the owner. A write event
alone does not publish a cache revision or complete a later read. Such reads
need their own lease and dependency.

Failure retirement is fail-closed. A terminal callback reports success only
after dispatch returned and the Metal command buffer completed successfully.
If either condition fails, the host poisons the writer and quarantines every
generation touched by that epoch before releasing terminal ledger pins. Such
slots never reenter the free list for this pool, even when all host references
are gone. A failed attention-read callback follows the same rule and rejects
future public snapshots. Invalid or conflicting callback batches release no
pins. An ambiguous enqueue with no terminal callback retains its pins and
refuses teardown.

The one-way failed-arena teardown requires zero pending write and read epochs.
The native backend then synchronizes its explicit stream and drops its arena
capsule; the request owner can close public/private generations after readers
and branches drain. MLX graph objects may still retain their own native arena
references, so this is a logical backend teardown, not a proof that the
allocator immediately freed its bytes. No quarantined slot can be reused in
the old pool. The double-opt-in `inject_terminal_failure` write binding is
only for a bounded GPU fault-path probe: it runs a real Metal command buffer
but intentionally reports a failed terminal callback. It is not a device-fault
injection and cannot establish behavior under an actual Metal error status.
An internal 2026-10-03 injected-terminal gate passed on source `c67a8f83`:
one false terminal event poisoned the writer, quarantined the sole page,
blocked reuse, and allowed one-way teardown after terminal drain. The raw
machine-specific gate receipts are not part of the public source export. A
real Metal command-buffer error and actual native allocator destruction
remain unverified.

Build and CPU import/conversion check:

```sh
cmake -S native/paged_kv -B /tmp/mlx2-paged-kv-build \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DMLX_DIR="$PWD/.venv/lib/python3.12/site-packages/mlx/share/cmake/MLX" \
  -DMLX_LIBRARY="$PWD/.venv/lib/python3.12/site-packages/mlx/lib/libmlx.dylib" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/mlx2-paged-kv-build -j 2
PYTHONPATH=/tmp/mlx2-paged-kv-build:src \
  .venv/bin/python -m pytest -q \
  tests/test_paged_kv_native_binding_cpu.py tests/test_paged_kv_write_cpu.py
```

CMake requires both the imported `mlx` target and `MLX_LIBRARY` to point to
the wheel. The binding uses MLX's `mlx` nanobind domain and pins nanobind
v2.15.0 at `a47983000e9dda68177fc1fd426f7147d785c2c0`, matching MLX
source revision `39400a0d4`. An earlier v2.13.0 build imported but rejected
real `mlx.core.array` and `Stream` objects. Rebuild and revalidate after any
MLX upgrade. Installing the optional extension requires an explicit CMake
install step or placing its build directory on Python's import path.

The CPU checks prove compile, link, import, and actual MLX array/Stream
conversion. Native GPU writes, completion and failure retirement, append/read
visibility, memory lifetime, and serving safety remain unverified. No serving
route selects this candidate.

A bounded, default-off diagnostic read is available solely for controlled
GPU proof work.
It copies a requested span into new arrays, consumes the writer dependency,
and requires explicit enablement. The probe has not run and is not a serving
read path.

The opt-in `copy_page` primitive copies a bounded K/V span between distinct
physical pages of this arena. `PagedKVWriteOwner.submit_copy` pins both
generation handles before native submission and waits for a terminal callback
before releasing those pins. A failed callback poisons the owner. An ambiguous
enqueue retains both pins until a terminal event or explicit arena teardown;
request cancellation alone cannot recycle either page. The caller must finish
prior writes to the source and must publish the destination page table entry
only after successful terminal completion. The existing metadata-only
`PagedKVSequence.append` is not a safe shared-tail publication path by itself.

`scripts/research/varlen_native_cow_gpu_probe.py` is a short, default-off
same-stream copy/read/generation gate for a shared 17-byte prefix. It requires
`--execute-gpu` and a separate owned GPU lease. The source-bound native copy
gate passed internally on 2026-10-03; its raw machine-specific receipt is not
part of the public source export.
A real failed Metal command buffer, cross-stream ordering, shared-tail metadata
publication, and model-serving integration remain unverified.

The next default-off primitive, `attention_read_fp16`, addresses K/V directly
inside this opaque arena. It accepts a packed fp16 query plus validated host
span/page coordinates, returns one attention output, and reports terminal
events through `poll_read_completions` separately from write/COW events.
`src/mlx2/runtime/paged_attention_native.py` checks the packed host plan and
holds its reader page lease through the matching native event. This avoids a
capacity-sized arena copy or a writable Python arena view. C++ validates
query dimensions, row/table coverage, page IDs, GQA and arena geometry before
creating the lazy graph node. It currently admits dense fp16 d128/d256 with
causal or sliding masks; bf16, q8, QSA, sinks and model serving remain out of
scope. The extension compiles and imports against the pinned wheel. The
dry-default `scripts/research/varlen_native_attention_gpu_probe.py` passed a
short 65/63-token two-lane native write-to-attention read/parity/callback cell
under the project GPU lease. The raw source-bound receipt is not part of the
public source export. No route selects the primitive.

### Bounded Q1 B2 inline metadata candidate

`MLX2_PAGED_Q1_INLINE_METADATA=1` is a default-off native host optimization.
After the existing complete arena/span/page validation, exactly two one-row
spans with visible lengths 32 through 128 and two through six page IDs retain
eight immutable metadata vectors in the primitive. The graph retains query and
write dependency arrays; encoding copies metadata through `set_bytes` into the
original buffer slots. Every inline buffer is at most 24 bytes. Larger retained
page tables, other batch/query geometry, or any other flag value retain the
ordinary array metadata path. Ordinary and SIMD tile modes each have a distinct
inline shader library; only metadata pointer address spaces differ, preserving
all arithmetic and bounds. Page generations remain the caller's pinned lease.

`q1_metadata_dispatch_count(arena)` counts encoded inline read dispatches,
including those whose later command buffer may fail. Graph creation and
fallback reads leave this counter unchanged. This is an engagement counter,
not a completion or qualification counter.

The 2026-10-04 CPU slice passed 35 tests, a pinned-wheel native build, and offline
Metal compilation for both shader variants. It has **no GPU or performance
qualification**. See `provenance/paged-q1-inline-metadata-2026-10-04.json`.

The smallest GPU discriminator is one B2 Q1 read against the same seeded opaque
arena, one lane ending at 65 tokens and the other at 63 tokens (three page IDs),
with windows off, dimension 128 and unchanged write dependency. Compare flag
unset versus `1`, first with `MLX2_PAGED_Q1_SIMD_TILE` unset, then with it set to
`1`. Require exact output equality within each shader mode, matching successful
read terminal events, unchanged page generation/lease cleanup, and native
metadata counter delta zero versus one. Repeat with dimension 256 and a
windowed retained-page boundary before the full model discriminator. In the
28-layer model, each valid B2 ready step should produce a metadata counter delta
of 28. Controlled request timing with immutable artifact/binary identity and
thermal/swap conditions is a separate performance gate.

## Long Q1 split-KV research kernel (2026-10-04)

Default off: `MLX2_PAGED_Q1_SPLIT_KV=128` or `256` explicitly selects the
partition token count. Missing/`0` preserves the existing scalar/short tile.
Selection requires the already validated two-row/two-span Q1 layout, D128 or
D256, query heads 1..128, and both visible lengths in 1..8192, with at least one
length above128. Short reads retain their existing policy. An explicit long
candidate over8192 fails its geometry bound. Profile/adapter selection belongs
to the higher-level qualification work; this native implementation changes no
model math, scheduler, page validation, or page-generation ownership contract.

The partial stage runs one128-thread group per row/head/partition. Four SIMD
groups build FP32 online softmax maxima, denominators and value numerators,
then merge into FP32 scratch. The reduction stage uses one128-thread group
per row/head, rescales all partition numerators with their maxima and writes
FP16 attention. Empty partitions and failed dependency bytes initialize zero
contributions without skipping barriers. Relative token offsets avoid uint32
absolute-position wrap near the coordinate limit. Channel loads are coalesced
across SIMD lanes. Existing scalar and short tile source strings are unchanged.

| Visible tokens | Partition tokens | Partitions | Scratch bytes, B2/H24/D256 |
| --- | --- | --- | --- |
| 4096 | 128 | 32 | 1585152 |
| 4096 | 256 | 16 | 792576 |
| 8192 | 128 | 64 | 3170304 |
| 8192 | 256 | 32 | 1585152 |

Both kernels check actual pipeline SIMD width32, max threads>=128 and static
threadgroup memory against the device limit. Scratch is checked against device
max buffer length, registered as MLX output then input to retain the standard
producer/consumer barrier, added as a temporary and captured by the terminal
callback. The callback is installed before either dispatch and reports read
success only after both were encoded and the command buffer completed. MLX
continues to own command commit. Failed/partial encodings retain the same
read-epoch completion contract.

`q1_split_partial_dispatch_count` and `q1_split_reduce_dispatch_count` increment
after their corresponding encoded dispatch. Long split does not increment
short tile/stripe/inline counters; it uses the existing dynamic metadata arrays.
GPU receipts must show both positive split counters and terminal completion.
Graph construction or flag selection alone proves no dispatch or performance.

Validation is source/CPU/compiler only:
`python3 tests/test_paged_q1_splitkv_source_cpu.py -v` (7 checks),
`python3 tests/test_paged_q1_geometry_source_cpu.py -v` (4 preserved checks),
offline Metal3.2 compilation of all D128/D256 ×128/256 variants and a complete
native extension compile linked against the pinned wheel library. Numeric CPU
oracles cover63/64/65,127/128/129,4K/8K, empty tails, extreme scores, page edges,
permuted page IDs, GQA head mapping and noncontiguous queries. This is not Metal
runtime compatibility, GPU parity, serving qualification or performance evidence.
`provenance/paged-q1-splitkv-20261004.json` records source/build identity.

## Explicit BF16 storage candidate (2026-10-04)

`create_arena(plane_bytes, storage_dtype="float16")` preserves the existing
one-argument Python ABI and FP16 default. An explicit `"bfloat16"` creates an
immutable BF16 arena. `storage_dtype(arena)` reports the selected spelling;
`attention_read`/`gather_q1` are generic aliases for the preserved historical
`attention_read_fp16`/`gather_q1_fp16` entrypoints. There is no automatic dtype
inference, model cast, mixed-precision fallback or default-policy change.

Grouped K/V writes and query readers require tensors matching the arena dtype;
attention and gather outputs use that same dtype. Both storage formats are16
bits, so plane/page geometry, byte offset bounds, raw byte writes, page copy,
terminal callbacks, read epochs, temporary retention and dispatch counters are
unchanged. The host continues to own the dtype/identity of opaque byte writes.
It must bind the explicit arena dtype to the model/profile/cache identity.

BF16 scalar/tile/striped/split attention sources substitute native Metal
`bfloat` storage for `half`; accumulators and split scratch remain FP32. Dtype
enters every affected Metal library cache identity. BF16 grouped writes and
gathers use `ushort` storage copies, preserving every16-bit payload without a
numeric conversion, including special/NaN payloads. Original FP16 source
strings are returned byte-for-byte; storage source drift fails closed.

Installed MLX headers use native `bfloat` and the current offline Metal3.2
compiler accepts it. Twenty BF16 storage variants compile offline, including
D256 and both long split partition sizes. Five new stdlib/compiler/source
checks and eleven preserved geometry/split checks pass; the full native
extension build also passes. No binary was imported, no MLX runtime/model
was loaded and no GPU operation occurred. This proves source/build feasibility,
not device compatibility, BF16 arithmetic parity, model serving or performance.
The root gate must bind BF16 profile/storage, prove unchanged BF16 boundaries,
check actual dispatch counters and terminal completion, then compare output,
logits/tokens and complete-request performance. FP16 defaults remain in force.
`provenance/paged-native-bf16-storage-20261004.json` binds the implementation.
