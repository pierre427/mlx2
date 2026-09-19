# Segmented shared-prefix QSA Metal probe — 2026-09-17

Status: **prototype useful in a narrow decode lane; not qualified, selected, or observed-used**

## Scope and identity

- Initial source baseline: `ea22404` (`Qualify Qwen3.6 serving and atomic MTP
  batching`). A concurrent ASPIRE session advanced `main` to `d9260ac` during
  this run; final focused validation passed against that later worktree without
  modifying or claiming the concurrent changes.
- Device: Apple M5 Max, MLX `0.32.2.dev20260915+2a817ad94`.
- GPU serialization: matching CPG lease plus `/tmp/gpu.lock` and
  `/Users/Shared/mlxuag/gpu.lock`.
- The only live model process, `com.example.music3-api`, was unloaded before
  the run. Control-plane and MCP services were left running.
- The probe is isolated in `segmented_qsa_metal.py`; production serving does not
  import or select it.

The kernel assigns one SIMD group to a query head and shares each immutable-base
K/V load across B independent query rows. Row-private suffixes remain separate.
Pass one produces mergeable float32 online-softmax states over 32–128 token
partitions; pass two merges those states. An optional device index vector models
the equal ordered selected-block set required by exact-set QSA folding.

## Correctness

- 10/10 opt-in prototype Metal tests pass across B1/B2/B4/B8, Q1/Q4, ragged
  suffixes, the indexed selected-set path, and fail-closed unsupported batch.
- 6/6 existing Qwen4 private-delta and B2 exact-set Metal tests pass.
- The mechanism engagement word was `1` in every measured arm.
- General FP16 maximum absolute error was `6.103515625e-05`.
- Qwen4-shaped BF16 indexed maximum absolute error was `0.0009765625`.

These are tolerance-based results, not bit-exact qualification. The different
partition order is expected to change low bits.

## Performance evidence

The benchmark reports two references:

1. **end-to-end row reference**: gather/concatenate the shared selection for
   each row and invoke stock MLX SDPA per row;
2. **prepared reference**: gather the common base once outside the timed region,
   then invoke stock MLX SDPA per row. This is the harder lower bound.

For general FP16 geometry (`Hq=8`, `Hkv=2`, `D=128`) across 4K–131K prefixes:

- Q1 decode beat the prepared reference in 14/16 cells; median ratio `0.715`
  (about 28.5% lower attention latency).
- Q4 verify beat it in only 7/16 cells; median ratio `1.037`. The native SDPA
  query-tile path is already strong, so this probe should not target Q4.
- At 131K/Q1: B2 was `0.704 ms` vs `0.912 ms`, B4 `0.745 ms` vs `1.604 ms`,
  and B8 `1.926 ms` vs `2.990 ms` against the prepared reference.

For Qwen4-shaped BF16 indexed geometry (`Hq=12`, `Hkv=1`, `D=256`, 2,048
selected tokens from physical 16K/64K/131K bases):

- 6/9 cells beat the prepared reference; median ratio `0.898`.
- B4 won at all three physical widths. Ratios were `0.951`, `0.898`, and
  `0.379`; the last result is promising but must be treated as noisy until a
  model-bound repeated A/B confirms it.
- Median ratio against the gather-plus-row reference was `0.662`.
- Full-base D256 B8 exhibited severe register-pressure regressions, and Q4 was
  consistently poor. Those geometries should fail closed in any next probe.

Receipts:

- `qualification/experiments/segmented-qsa-metal-20260917/general-fp16.json`
  (`sha256:6bb9d0568b826dce6aa747cc4c9a7202d0f4d4c0da771a6e9913859199610cfc`)
- `qualification/experiments/segmented-qsa-metal-20260917/qwen4-bf16-indexed.json`
  (`sha256:5059e011ef1333a61272efe75c24ca19d9da655072f32a267f010d6c16c0c67f`)

## Verdict

There is a viable optimization here, but it is narrower than “shared QSA for
every batch.” The evidence supports a **B4, Q1 decode, equal ordered selected-set
candidate**. It does not support Q4 verification, arbitrary unequal selections,
or unrestricted B8/full-base execution.

Before production integration:

1. capture the exact-set proof hit rate on real Qwen4 batched decode; current
   qualification receipts contain zero exact-set-fold calls, so observed use is
   not established;
2. compare B4 against the existing B2 fold and ordinary private-delta kernel in
   the same model-bound request sequence, including selection and proof cost;
3. add current QSA block/tail/mask semantics rather than the probe's simplified
   visible-suffix contract;
4. gate to B4/Q1/D256/BF16 and require a nonzero mechanism receipt;
5. qualify output tokens and end-to-end layer/request latency before selection.

## Closeout

Both GPU lock directories were removed after clean tests. The
`com.example.music3-api` LaunchAgent was restored and verified listening on
`127.0.0.1:8600`. No prototype route was enabled.
