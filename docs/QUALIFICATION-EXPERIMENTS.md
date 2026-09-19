# Model qualification experiments

These runners collect evidence; they do not make a route selectable. A route is
qualified only after its complete report passes and a matching runtime-bound
qualification receipt is created through the normal serving qualification flow.

## Frozen inputs and identity

The experiment manifest is
[`qualification/four-model-experiments.json`](../qualification/four-model-experiments.json).
Its generic schema declares comparative arms for
Flash-Next, Qwen3.8 27B and Muse Glimmer, plus an explicit ordinary-only North
Mini Code lane. North records `arm_order_claim: none`; it does not imply an
optimized comparison. All models use distinct APCv2 directories, exact profile
and receipt gates, mechanism counter deltas, and safe activation commands.
Candidate servers run on port 8285 in qualification mode. The activation helper
stops only a PID whose exact command matches its own state file; it refuses a
reused or unrelated PID.

The frozen Spomin corpus is
[`qualification/corpora/spomin-20x20-long-multiturn-corpus-20260915.json`](../qualification/corpora/spomin-20x20-long-multiturn-corpus-20260915.json),
SHA-256 `5d7233f805b310e354cb628a0d0b1d12299cc90dbfeebb890eaf35dac966ad12`.
The clean port has no dependency on the former serving stack or cache code.
Its provenance is recorded in
[`provenance/spomin-20x20-harness.json`](../provenance/spomin-20x20-harness.json).

Every report binds the manifest, macOS build, host, Python, MLX, git revision,
server runtime hash, artifact, settings and profile. Resume refuses a changed
manifest, host/runtime/source identity, or server identity. Each completed cell
is written atomically; failed cells are retried rather than skipped.

Flash also binds each matrix arm to a passed `qualify_serving.py` receipt whose
runtime, artifact and complete settings equal the active server. Per-cell gates
require PLE, compiled PLE, pooled/scatter QSA, eager dispatch, fused MoE and
fused GDN counters to advance. The MTP receipt separately requires aggregate
advanced-feature probes for shared-QSA batching and asynchronous QSA promotion,
plus known-tail PLE, indexed QSA and private-delta execution. Those mechanisms
need distinct eligible concurrent cohorts, so the harness does not falsely
require both alternatives inside every sequential context cell.

## Build exact context prompts

Prompt calibration loads tokenizers only. It does not load tensor payloads or
use the GPU. The renderer is explicit: `qwen_direct` mirrors the Flash/Qwen
`enable_thinking=false` chat template; `muse_direct` mirrors Muse's high
reasoning-strength template followed by its direct-answer recipient suffix;
and `north_direct` uses North's native `reasoning=false`,
`reasoning_effort=none`, and `skip_thinking=true` controls.

```bash
.venv/bin/python scripts/build_context_prompts.py \
  --manifest qualification/four-model-experiments.json \
  --output-manifest /tmp/mlx2-qualification/four-model-experiments.generated.json \
  --prompt-dir /tmp/mlx2-qualification/context-prompts
```

The generated manifest freezes the path, SHA-256 and exact rendered token count
of every prompt. The current ladders are 2K, 8K, 32K, 64K, 131K and 262016 for
Flash/Qwen; 2K, 8K, 32K, 64K, 98304 and 131008 for Muse; and 2K, 8K, 32K,
64K, 131K, 262144 and 499936 for North. The runner rechecks the frozen prompt
hash immediately before every cell.

## Thermally controlled context ladder

```bash
.venv/bin/python scripts/run_qualification_matrix.py \
  --manifest /tmp/mlx2-qualification/four-model-experiments.generated.json \
  --suite context \
  --output qualification/context-ladder-macos-26.7.json \
  --resume
```

Comparative scheduling is run-major to avoid reloading a 20–127 GB service for every cell:
run 0 is arm A through contexts ascending then arm B descending; run 1 is B
ascending then A descending; run 2 repeats run 0. This preserves three runs per
model/context/arm, AB/BA/AB arm order, and a snake traversal with six model loads.
North has one ordinary arm, runs every context three times, and explicitly makes
no arm-order claim. Comparative models still require at least two arms.

Before every measured cell, the runner primes APCv2 outside the measured
interval, then requires three nominal, warning-free samples whose virtual
temperature range is at most 0.5 C. It uses a once-compiled Foundation probe,
`pmset -g therm`, and AppleSmartBattery temperature fields. The immediate
post-measure sample must also remain inside the thermal envelope. Raw thermal
state, warning text and raw temperature integers are preserved without copying
the rest of IORegistry.

## Separate width-20 stress suite

```bash
.venv/bin/python scripts/run_qualification_matrix.py \
  --manifest /tmp/mlx2-qualification/four-model-experiments.generated.json \
  --suite batch_stress \
  --output qualification/batch-stress-b20-macos-26.7.json \
  --resume
```

This suite intentionally has no thermal gate. For every model and route arm it
runs 20 rounds of 20 simultaneous warm APCv2 requests. A client barrier bounds
request start spread, and the result fails unless the receipt proves actual
compute width 20. HTTP concurrency alone is insufficient evidence.

## Frozen Spomin 20x20 suite

The historical `20x20` name means 20 domains times 20 cases, not 20 rounds at
width 20. Each model runs 400 cases as 20-request domain batches for both full
and compacted transcript arms, producing 800 paired rows. Domain arm order
alternates. This suite intentionally has no thermal gate.

```bash
for model in flash-next qwen38-27b muse-glimmer north-mini-code; do
  .venv/bin/python scripts/run_spomin_20x20.py \
    --manifest /tmp/mlx2-qualification/four-model-experiments.generated.json \
    --model "$model" \
    --output "qualification/spomin-20x20-${model}-macos-26.7.json" \
    --resume
done
```

Before live requests, all 400 client-side preparation receipts are reconciled
against the frozen source artifact with its historical Flash tokenizer for
token counts, selected segments, projected size and needle/query retention.
Each model's live request bodies are prepared separately with its own tokenizer.
The live gate requires all 20
requests to complete through the selected mlx2/APCv2 route and positive
multi-request batching evidence. It records actual widths but does not require
width 20; that stronger scheduler-capacity claim belongs to `batch_stress`.

The paired acceptance checks exact early/middle/late needles, sentinels,
compaction-specific loss, concept retention and lexical similarity. Compaction
is an exact HTTP transcript rebuild. It does not alter live KV, recurrent,
MTP, DFlash2 or APCv2 state.

## Monitoring

All reports expose `last_progress_at`, cell IDs, failures and the exact failed
cell manifest. A 15-minute supervisor should inspect that timestamp, the
harness-owned PID state at `/tmp/mlx2-qualification/server.json`, the active
server health/status and the current report. It should alert or resume only
when progress is stale; it must not weaken a failed receipt, thermal or width
gate.

## Host services, selected arms and the 2026-09-18 manifest refresh

- `host_services` (manifest top level) names resident launchd services that
  share the GPU, with the plist that restores each. The matrix runner refuses
  to start while one is loaded unless `--quiesce-host-services` is passed; with
  it, the services are booted out once for the run and restored at exit
  (normal exit, error or SIGTERM), and the report records both transitions.
  This replaces the per-arm `--bootout-label com.example.mlx2-flash-next`,
  which named a service that no longer exists and restored nothing. Taking the
  production server down stays an explicit operator decision per run.
- `selected_arm` (model level) names the arm a model is served with; it must be
  one of the model's arms. Qwen3.6-35B-A3B selects `mtp-artifact-ordinary`:
  native MTP measured 0.35–0.63x the same-artifact ordinary route at every
  width, so `mtp2` remains a reference arm.
- Muse-Glimmer gains a `prompt-lookup` arm that must observe batched target
  verification (`scheduler.pld_batched_rounds`).
- North-Mini-Code's arm now requires the served thinking configuration:
  adapter defaults in force, budget 512, alpha 0.2 and
  `thinking_steer.calibration.state == "calibrated"` — a North server that is
  not steering with a direction bound to its artifact does not qualify.

