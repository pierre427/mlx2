# Ecosystem probes, 2026-09-19

Checks run against peer-engine PRs and papers from 09-17 to 09-19. Everything
ran on CPU: tiny random-weight models, plus direct safetensors reads for the
35B norm probe. GPU checks followed the same day (see "GPU results"). Branch `claude/ecosystem-probes-20260919`.

## Findings

| Signal | Verdict | Action |
|---|---|---|
| omlx#3724: batched MTP + prefix cache, boundary emit corrupts every row | **Not impacted.** APC checkpoints are captured only during prefill, on a single-row cache (`generate.py` `_capture_mtp_interior_checkpoint`; `capture_self_mtp_checkpoint` refuses B>1). No checkpoint is taken mid-verify. Three MTP rows joining at steps 0/6/9 match their solo runs; checkpoint state diff is 0.0. | None. Optional hardening: shape-assert in `ArraysCache._blend_rows`. |
| omlx#3750 / MTPLX#511: MTP `pre_fc_norm` +1 shift ambiguity on 35B-A3B | **Impacted, fixed at load.** The oQ4e-mtp 35B artifact stores 4 of 7 MTP norms unshifted: `post_attention_layernorm`, `q_norm`, `k_norm`, `mtp.norm`. They are bit-identical to raw HF. The two `pre_fc` norms (0.27 / 0.49) are correct. oQ's `add_if_mean_lt_0_5` skipped raw norms with mean ≥ 0.5. Recorded 35B MTP acceptance is 0.341 (27B: 0.752, Flash-Next: 0.742). | `adapters/norm_repair.py`: byte-hash-keyed +1, idempotent. Runtime means now appear in adapter diagnostics. GPU: acceptance 0.358 → 0.740 (below). |
| vllm#57616: GDN + MTP records 0 prefix-cache hits | **Not impacted.** On both the qwen4_exp (Flash-Next runtime) and qwen38 hybrid, with MTP on: the extended prompt reuses 95/120 tokens and multi-turn reuses 103/110; the MTP sidecar is restored; warm output equals cold. | Mechanism assertion added: `tests/test_apc_hits_hybrid_gdn_self_mtp.py`. Minor: `apc_v2` counts a hit before serving rejects a sidecar-less one, so `mlx2_prefix_cache_hits_total` can over-report on MTP. |
| Ollama v0.34.2: buffer pool grows during MLX speculative decode | **PLD route impacted, fixed.** `pld.py` never called `mx.clear_cache`. Self-MTP clears every 512 cycles, i.e. every ~1.3-1.8K tokens at K=3, which is coarser than Ollama's 256 tokens. | PLD now clears after each prefill chunk and on the shared 512-step cadence. After the GPU probe (below), self-MTP also clears every 256 emitted tokens. |
| omlx#3726: new requests starved behind chunked prefill | **Impacted, fixed.** On self-MTP, a short request waited for the whole long prefill (first token at round 38 vs 38). On ordinary, it waited whenever two long prefills held both slots (41 vs 39). | Self-MTP alternates one-chunk admissions with long-prefill chunks. Ordinary admits one overflow lane for a one-chunk request. The short request's first token now comes at round 2-3. New counters: `mtp_short_prefill_interleaved`, `short_prefill_overflow_admissions`. |
| omlx#3703: left_padding on a warm singleton changes MTP output | **Related bug found on the plain route, fixed.** `ArraysCache.extend/merge` zero-filled the Qwen4 PLE n-gram history slot for a cold row, but the model reads an empty slot as all-EOS. A cold row that joined a warm batch diverged, and its interior checkpoints were wrong (diff 3.79). MTP joins were unaffected. | `ArraysCache._empty_slot` hook. `Qwen4ArraysCache` fills with EOS and fails closed when the id is unknown. `tests/test_qwen4_cold_join_ple_history.py`. |
| mlx-vlm#2271: lazy graph growth in decode | **Not impacted.** Pending graph nodes stay constant for self-MTP (physical and segmented), PLD and rotating replay, measured up to 950 tokens. | None. |
| sglang#40143: QSA pending index-K ring holds one compress group | **Not impacted.** mlx2 keeps an unbounded index-key ledger and pools only closed groups. The tail is derived per query. One T-token forward matches T single-token forwards, and rollback via `trim`, for T=1..9 at every offset mod 4 (max diff 4.8e-7). k≥4 and wider DFlash blocks are safe on this axis. | Invariant: every QSA rewind must go through `trim`/`trim_ragged`/`release_qsa_cycle`. A raw offset change leaves stale pooled blocks. |
| mlx#4529: `col_reduce_longcolumn` unsigned wrap | **Bug present in our fork** (`mlx-uag/mlx-src` @ 2a817ad94, the venv's editable build). mlx2 never reduces a negative-stride MLX view, so serving is not exposed. | Adopted on fork branch `claude/mlx-4529-col-reduce-negstride` (c9ff96be0) with a regression test. The test was checked on CPU only; it needs Metal to reach the kernel. |
| mlx-lm#1482: mixed-bit MLA derived projections | **Not impacted.** No K2 or Ling ports exist. Xing4.0 rebuilds `embed_q`/`unembed_out` from tensor shapes. A `kv_b_proj` bit override would fail loudly at strict load, not silently. | Optional: map `kv_b_proj` overrides onto derived modules in the Xing predicate. |
| mlx-vlm 0.7.1 slowdown (omlx#3750, Rapid#3551) | The `multimodal` extra pinned `mlx-vlm>=0.7.1,<0.8`, but the live multimodal smoke qualified on the 0.6.17 fork (653f1f13). | Pinned `mlx-vlm>=0.6.17,<0.7`. A/B before any bump. |
| llama.cpp#29110: multi-column Q4/Q8 matvec | The fork already has a multi-column kernel (qmv_wide). Fixed strided-batch re-reads and the tile cap instead (GPU results below). | `scripts/bench_qmm_multicolumn.py --i-own-the-gpu`: M=1..8 qmm on 27B- and A3B-class shapes, prints time(M)/time(1). |

## Papers

- **SwitchSD** (arXiv 2609.20186, Eisenstadt et al.) uses a linear probe after
  attention at mid depth (layer 14 on Llama-8B, F1 0.87, AUC > 0.99). Copy
  label: a verbatim run of ≥ 5 context tokens. It gates PLD vs EAGLE3 per
  step. Gains come from *avoiding* bad copies; only 3-9% of steps copy.
  **Implication:** mlx2's prose 0.70x is about 20 points route cost (PLD gives
  up compiled replay) and about 10 points bad proposals, so a perfect probe
  alone caps near 0.80x. The real design is copy-or-MTP gating inside the
  self-MTP loop. That needs the `serving.py` speculative-route exclusion lifted.
  Cheap first step: offline labels from `IndexedPromptLookup` replays, and a
  logistic probe on the final-norm hidden the MTP head already receives.
- **On-Demand Attention** (2609.20734): per-step, all-global-layer recall
  decision. Learned full-attention call rates are 42-58% on RULER-16K and 71%
  on LongBench. No learned rates at 32-128K; the headline 1.98x is a
  prescribed 12.5% schedule. Conflicts with compiled replay and the megakernel,
  and a masking variant saves nothing. Skip, unless it becomes an on-device
  block-skip inside the QSA indexed kernels.
- **DeepSeek-V4.1-Flash** (2609.19969): SWA bounded replay persists only global
  KV plus indexer keys, and replays the last 128 tokens on a miss. It is
  approximate, which is made safe by post-training. mlx2's exact counterpart is
  APC interior checkpoints. Worth borrowing: a RAM-tier TTL pool for
  SWA/GDN state with only full-attention KV on disk.
- **SiliconBench** (2609.19169): harness at github.com/WindChimeRan/SiliconBench,
  MIT. 3/9 engines passed (llama.cpp, vllm-metal, omlx). sglang and vllm-mlx
  broke their declared memory budgets; mlx_lm completed 2/100 at agent c=16.
  TP over TB5 RDMA gives 1.3x; PP over TCP gives 0.8x. **Candidate run:** agent
  split at c=1/8/16 against `mlx2.server --max-inflight 16`.
- **DSpark** (arXiv 2607.05147; landed in vLLM 09-15): Markov bias plus a
  per-position confidence head chooses verify length. DFlash2's
  `CandidateSelector` already covers the Markov part. The missing piece is the
  confidence head, which would replace the EWMA in
  `CohortAdaptiveMTPDepth`.

## Open after review

The two Codex review passes raised four issues, all fixed. Two leftovers:

- The pre-existing cohort split in incremental self-MTP prefill was fixed
  separately on main (`7daf236`). This branch is rebased on it.
- `Qwen4ArraysCache.ple_history_fill` is not serialized in `meta_state`, so an
  APC disk restore comes back without it. Harmless unless such a row is joined
  with a cold row that also lacks it. Cold rows come from `make_cache`, which
  sets the id, so that case fails closed. Persisting the id is a cache-format
  change and needs versioning.

## GPU results (M5 Max, 2026-09-19 afternoon)

| Check | Result | Evidence |
|---|---|---|
| 35B self-MTP with the norm repair | Draft acceptance 0.358 → **0.740**. Throughput B1 86.5 → **135.4**, B2 126.2 → 180.5, B4 154.1 → 216.6 tok/s (+57/43/41%). B1/B2 outputs byte-identical across arms. | `qualification/runs/qwen36-35b-a3b/mtp-norm-repair-20260919/` |
| 27B self-MTP allocator cache at 64K | The old 512-step clear let the MLX cache reach **70 GiB** over a 29 GiB working set (82 GiB with no clear). **Fixed:** self-MTP now also clears every 256 emitted tokens. Cache max is 10.5 GiB and decode is not slower. | `qualification/runs/qwen38-27b-mtp-allocator-reclaim-20260919/` |
| qmm M=1..16 (llama.cpp#29110) | The first warm-cache read ("~2x headroom at M=4") was wrong. Cache-cold, M=2..5 is already near the bandwidth floor. The real losses were (a) strided `(lanes, L, K)` verify inputs re-reading weights once per lane and (b) the qmv_wide tile cap of 5. Both are fixed in mlx fork `claude/qmv-wide-tile8` (`ad71bfaa7`, now in the shared venv): 2-lane verify matmuls +20-79%; 27B end to end +9% at 2 lanes and +5% at 4; 1 lane within noise; outputs identical. Follow-up: an M5 NAX small-M kernel (fork `39400a0d4`, in the shared venv) makes M=12-16 verify matmuls 2-3x faster; 27B +17.9% at 4 lanes. | `qualification/runs/qmm-verify-flatten-tile8-20260919/`, `qualification/runs/qmm-nax-small-m-20260919/` |
| Flash-Next plain late join, real weights | Outputs identical on main and fix. The real route never co-batched a cold row with a warm one mid-prefill in this setup. The fix remains a guard. | `qualification/runs/flash-next-plain-late-join-20260919/` |
| mlx#4529 on Metal, current build | `test_col_reduce_negative_stride` **fails** 6 subtests (the long-column shape (2,1024,16) for sum/max/min/mean/var, plus prod). The bug is confirmed on Metal. The patched fork build (`0.32.2.dev20260919+c9ff96be0`, scratch venv) passes all 14 `test_reduce` tests on Metal. | fork branch `claude/mlx-4529-col-reduce-negstride` |

## Watch items (no action)

MTPLX#508/#507 (Flash-Next M5 Max 44.7 → 87.0 tok/s, depth 3; they claim 102);
llama.cpp#29000 (Qwen4-Exp HC ops on Metal; possible second engine);
Rapid#3551 (same mlx-vlm 0.7 caution).
