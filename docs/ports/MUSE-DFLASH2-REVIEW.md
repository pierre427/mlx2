> Archived 2026-09-15 after the reviewed implementation was integrated into mlx2. This report preserves the original isolated review evidence and source hashes; it does not qualify the merged runtime or authorize GPU execution. Current status: [MUSE-DFLASH2.md](MUSE-DFLASH2.md).

# Independent Muse DFlash2 CPU review

Review target: `/tmp/mlx2-muse-dflash2-20260915`; main mlx2 worktree was not edited. No real model artifacts were loaded and no GPU tests were run. Tiny native tensor tests explicitly select MLX CPU before model construction; ring/cache oracles use NumPy production class bodies.

## Mechanism findings

- Selector proposal `q` is the actual normalized distribution used to sample the candidate path, including predecessor-dependent edge scores. The verifier uses target `p`, `min(1,p[x]/q[x])` acceptance, and normalized positive `p-q` on rejection. This is a deliberate distribution-aware extension of the mined DFlash2 selector, rather than the upstream exact-match protocol.
- Target inputs are anchor plus proposals. Exact lane transaction commits anchor plus accepted inputs; correction/bonus remains pending. Per-row snapshot/replay restores evicted ring state, with projections batched and attention computed per row.
- Target taps are post-layer, before final norm, consistent with the upstream Muse capture path. Body-only prefill avoids vocabulary projection.
- Fresh APC requests preserve their new seed. RNG continuation requires explicit opt-in. Whole-lane snapshots restore RNG/history/anchor/ready and target/draft state on a failed round.
- Normal external routing requires a distinct composite target/draft fingerprint, settings, declared external capability and four external mechanism checks. An ordinary receipt and an ordinary descriptor both fail the external gate.

## Concrete defects found and handed to owner

1. Commit failure after the cache transaction closed itself triggered an unconditional second abort, masking the original error and skipping lane RNG/state restoration. Independent injection reproduced stale/closed RuntimeError and advanced RNG. Owner corrected the closed guard and guaranteed lane restoration, with regression. Independent rerun preserves the original MemoryError and restores RNG plus draft offsets.
2. Snapshot allocation happened after setting the transaction-open flag but before entering try/finally. Snapshot failure could strand the lock and prevent close. Owner moved lock acquisition after successful snapshots and added a regression.
3. External receipts inherited ordinary pre-sampler logprob wording and stale adapter diagnostics. Owner corrected external transformed-target semantics and delegated qualification authority to the serving route.
4. Admission estimate initially counted 24 bytes per vocabulary position while four live float64 q/p arrays alone can require 32, before tensor/normalization temporaries. Owner increased reservation to four target-cache plus three draft-cache copies, 128 bytes per vocabulary position, explicit activation allowance, and a body-only prefill phase without vocabulary projection. This resolves the identified source undercount; measured peak calibration remains required.

5. The final serializable-empty draft cache helper initialized FP32 placeholder arrays, which promoted first BF16 appends to FP32. Independent CPU reproduction confirmed both plain and rotating caches were affected. Owner added a common append helper that returns empty projected arrays unchanged and discards placeholder storage before the first real append. BF16 zero-context and first S1/S3 append regression passes.

## Independent counterfactual checks

- `/tmp/mlx2-dflash-review.py`: first accepted EOS commits only the preceding anchor; malformed zero-q proposals consume no RNG; zero-p target support rejects to the residual; closed-commit failure reproduction.
- `/tmp/mlx2-dflash-warm-review.py`: cold versus restored paired cache agree for initial prefix lengths 1/3/9, greedy and temperature 0.8, new seed 99 (six exact generated-token comparisons).
- `/tmp/mlx2-dflash-gate-review.py`: ordinary evidence rejected; complete distinct external evidence accepted; ordinary descriptor rejected even with external receipt.
- `tests/test_segmented_rotating_kv_cpu.py`: 13 guarded tests, including 80 randomized B1..B4 rounds (200 lane-rounds), divergent accepted lengths, wrap, offset-only rewind counterexample, partial abort, membership changes and reference release.

## Remaining qualification boundary

CPU functional integration does not establish large-artifact loading, GPU correctness, memory admission calibration, speedup, full serving-domain qualification, or production selection. Exact whole-cache snapshots/replay and host probability arrays can be expensive. The target trunk and compatible draft projections batch; attention remains per row. Tree proposal verification, fused segmented attention, optimized sparse probability transport, and snapshot-free rollback are not demonstrated by this work.

## Final bounded verification

At isolated runtime source hash `e7ee2259510c26be88f213b515903bb5174ae7c1d81f47ed09c333f827d3851c` (2026-09-15 22:41 UTC), the targeted combined suite passes **83 tests**: external DFlash2 18, segmented ring 13, Muse adapter 39, registry 11, qualification 2. The command used isolated `PYTHONPATH`, `--noconftest`, and the first native test module selects MLX CPU before constructing tiny models. Separate ring oracle verification blocked MLX imports entirely. Undefined/export-symbol lint passes. No remaining reproducible correctness blocker was identified within this CPU scope.

Command:

```
PYTHONPATH=/tmp/mlx2-muse-dflash2-20260915/src ~/Desktop/mlx2/.venv/bin/python -m pytest --noconftest -q tests/test_external_dflash2_cpu.py tests/test_segmented_rotating_kv_cpu.py tests/test_muse_glimmer_port.py tests/test_adapter_registry.py tests/test_qualification.py
```

This source identity is an isolated review reference, not a production qualification receipt. Main-tree integration changes identity and must rerun its relevant checks.

Final CLI review also verifies mutually exclusive ordinary/external route intent before listener/model allocation; one-token/one-output sidecar disk serialization is covered. Owner confirmed source/tests frozen.


## Serving seam review reopened after root integration audit

Root found two integration defects that the earlier isolated executor review missed: the serving factory passed gross headroom although the executor assumed the 20 GiB reserve had already been subtracted; and a cohort larger than available memory deferred every ready lane even when one lane fit. The earlier CPU review did not establish these serving-factory and partial-fit contracts.

Independent baseline reproductions:

- `/tmp/mlx2-dflash-serving-seam-review.py` captures the actual callback passed by `ServingEngine._run`: 30 GiB gross incorrectly reached the executor as 30 GiB instead of 10 GiB net. The test uses a metadata-only adapter, a fake APC and no GPU memory calls, but does not replace the callback under test.
- `/tmp/mlx2-dflash-partial-fit-review.py` uses actual tiny target/draft context updates and proposals forced to reject at position zero. B1 fits (54,464-byte estimate; 98,035-byte budget), B2 does not. Eight old-code polls emitted nothing and both lanes remained at zero generated tokens. Forcing one emitted token per round makes first-fit starvation visible instead of allowing queued accepted chunks to hide it.

Final seam revision verified at source `fc5537106d8b1f0f95c47aa89a2684825b28161b3143a49bf1e5c48b80c7acc4` (2026-09-15 22:50 UTC): **93 targeted CPU tests pass**, plus undefined/export-symbol lint. Both independent scripts were rerun unchanged: factory callback now yields 10 GiB net, and partial-fit response order is `[0,1,0,1,0,1,0,1]` with four generated tokens per lane.

The actual factory test also covers gross 10 GiB clamping to zero, allocator synchronization/clear wiring and the unleased-eviction callback. Owner included the exact root-authored `APCv2.evict_oldest_unleased` helper in the isolated snapshot, with provenance; retain main's existing implementation during integration. A real APCv2 pressure test now drives the executor callback, evicts the unleased entry, preserves the active branch generation, refuses to evict the remaining leased entry, and permits retirement after lease close.

Reclamation is bounded to one initial allocator drain and at most two eviction attempts per scheduler poll, with remeasurement after changes. A lane-capacity limit alone does not evict checkpoints. Deliberate selection policy: if a smaller subset fits, progress that subset and preserve checkpoints; reclaim only when no subset fits. This policy does not promise maximizing batch width by evicting warm checkpoints.

External verification receipts no longer populate ordinary compute width; ordinary fallback/target execution does. Actual artifact/GPU/peak-memory/speed qualification remains outstanding. Lesson from this reopened review: trace the actual producer/consumer callback contract and test the assembled serving factory; an executor docstring plus safe standalone callbacks cannot prove the reserve boundary.
