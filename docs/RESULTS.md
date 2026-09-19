# Serving results

## Source closeout — 2026-09-16

Runtime source
`ad1d6f8d9bf715f1a32c5d12159fb129ec4b62e8066574448f48a784cb11429e`
passed 621 CPU tests, skipped 16 Metal-only tests and ran 59 parameterized
subtests. Focused suites cover structured output, compaction fidelity, atomic
parallel-sample admission, Qwen3.6 geometry, qualification routing and the
heterogeneous shared-QSA regression. Build, archive-attribution and static
checks are recorded in the closeout commit.

This is source verification. No full-model GPU run was performed for this
closeout. Qwen3.6 real-artifact load/replay, APCv2 HTTP, 32K/64K context,
two-lane native-MTP and fused-GDN trials remain useful bounded historical
candidate evidence, but their receipts bind earlier source hashes. No Qwen3.6
route is qualified, selected or deployed. The Flash-Next 262K qualification
and five-round B1/B2/B4 comparison also remain outstanding.

## Paused qualification refresh — 2026-09-15

The modernized Flash-Next qualification and matched ordinary/MTP benchmark are
paused at the user's request. The last candidate requested a 262,144-token
context ceiling, four active/eight inflight requests and 16 GiB resident APC,
while preserving the 20 GiB service/driver reserve. These are candidate
settings, not a successful served-domain or deployment claim.

Source `873a986a504872b3558871e59996341e5540f077459bc1e48b00211f413a5ed4`
passed 392 CPU tests with 16 Metal tests skipped and 59 subtests. An independent
review reran 48 focused CPU tests for the final promotion-pressure repair.
The preceding source passed shorter API/batch checks but failed near-262K warm
reuse: both cold and repeated requests reported zero cached tokens. The repair's
live rerun was interrupted before completion. There is **no successful complete
new qualification profile or five-round benchmark** to report.

No historical rate below is a performance result for the changed runtime.
Mechanism evidence and the failure/repair boundary are tracked in
[Flash-Next parity](FLASHNEXT-PARITY.md) and the
[independent pressure review](experiments/PROMOTION-PRESSURE-REVIEW.md).

The current benchmark requires warm cache hits and evidence that each requested
B1/B2/B4 compute width actually appeared. It retains all observed widths and
request receipts, verifies unchanged runtime/artifact/settings across the run,
and reports output hashes for review. A concurrent HTTP request count alone
does not satisfy this check. Performance remains workload- and profile-specific.

## Historical initial 16K profile

The following measurements belong to source
`5195d36fabc652ef122148085ba7c6bad6dd19ec0ffa7baebe94ae6773e967c0`, the initial
16,384-token profile, and the local 128 GiB Apple Silicon host. They do not
qualify later source, a larger context, or newly selected mechanisms.

### Verification recorded at the time

- 102 tests and nine forced-acceptance subtests passed.
- Ordinary and MTP profiles passed HTTP cold/warm output, streaming, function
  calls and a tool round trip, reasoning, seeded repeatability, stop strings,
  concurrent requests and mixed warm prefixes.
- Both passed a 16,353-token prompt and a 16,352-token cache hit, over-limit
  rejection, streaming disconnect/cancellation, recovery and zero active APC
  leases after the workload.
- MTP counters recorded segmented target/draft forwards without full-prefix
  materialization. This did not prove that every requested HTTP concurrency
  ran at the corresponding compute width.
- APCv2 was the sole serving prefix cache. Optional indexed/shared-QSA routes
  were not credited as executed optimizations.

### Historical warm HTTP benchmark

Each row is the median of three runs: 160 generated tokens per request,
temperature zero, warm prompts, alternating 1/2/4 HTTP concurrency order.
Rates include HTTP wall time and queueing. Profiles ran sequentially with the
same four prompts and artifact; this was not a thermally controlled experiment.

| Concurrent HTTP requests | MTP2 aggregate tokens/s | Ordinary aggregate tokens/s | MTP / ordinary |
|---:|---:|---:|---:|
| 1 | 76.55 | 50.21 | 1.52x |
| 2 | 74.17 | 72.62 | 1.02x |
| 4 | 73.17 | 104.33 | 0.70x |

**Compute-width limitation:** the old harness did not require observed B2/B4.
Admission could select fewer lanes than the number of concurrent HTTP requests.
These rows therefore describe that historical endpoint workload, not qualified
true-B4 throughput. They are not a basis for selecting the current default
profile or attributing speed to a particular optimization.

### Archived evidence

The original qualification and benchmark records were preserved
byte-for-byte under
[`qualification/history/2026-09-15-initial-16k/`](../qualification/history/2026-09-15-initial-16k/):

- `flash-next-mtp2.json` and `flash-next-ordinary.json`
- `benchmark-mtp2.json` and `benchmark-ordinary.json`
- `qualified-startup.json`

These records retain their original identities, settings, timings and receipts.
See [serving](SERVING.md) for reproduction and the current qualification gate.

## Other model ports

Qwen3.8 27B and Muse Glimmer/DFlash2 are implemented CPU-tested candidates.
Their adapter/cache tests do not establish production-weight inference, GPU
performance, long-context memory capacity or deployed routes. Their pre-GPU
reports enumerate remaining work and transferable optimizations:
[Qwen3.8 27B](ports/QWEN38-27B.md), [Muse Glimmer](ports/MUSE-GLIMMER.md),
[Muse DFlash2](ports/MUSE-DFLASH2.md).
