# 2026-09-19 GPU quality campaign

This directory is a launch-ready, sequential campaign for the coordinator to
run under the lab GPU lease. The harness was built and validated on CPU only;
no server or model-weight load was performed while constructing it.

The runner preserves the `sanity-20x20-20260918` lifecycle: one owned server at
a time on `127.0.0.1:8297`, pinned source through `MLX2_CAMPAIGN_ROOT`, durable
`status.json` updates after every step, stage-local failures that do not stop
later stages, exact named-stage reruns, process-group shutdown, and conditional
quiesce/restore of `com.example.fn-uncensored-mlx-serve`. The service is
restored only if `launchctl print` showed it loaded before the campaign.

The evidence-backed triage of smoke pass 3, including classifications and the
exact rerun set, is in [PASS-3-TRIAGE.md](PASS-3-TRIAGE.md).
The latest non-Laguna smoke results and the pass-5 orchestrator verdict are in
[PASS-5-TRIAGE.md](PASS-5-TRIAGE.md).

## CPU preflight and dry-run

From the pinned clean worktree:

```bash
PYTHONPATH=src ~/Desktop/mlx2/.venv/bin/python \
  qualification/runs/quality-campaign-20260919/run_campaign.py \
  --phase all --dry-run
```

The dry-run sets MLX's default device to CPU before registry imports. It:

- verifies all nine local artifact paths through `mlx2.adapters.registry`;
- loads config plus tokenizer/processor only, never adapter weights;
- for Gemma 3n and MiniCPM-o only, verifies that `mlx_vlm` imports from the
  clean detached `/private/tmp/mlx-vlm-653f1f13` checkout at revision
  `653f1f13e238abb313fd45071bbd04b3de414635`; text-only stages retain the
  ordinary source-only `PYTHONPATH`;
- parses every generated server command with `mlx2.server.build_parser` and
  checks route compatibility, then invokes the same `ServingEngine` constructor
  argument validation used by real startup before any model is resolved;
- validates prompt-lookup, FLy, interior-checkpoint and Flash-Next policies
  with their real runtime policy types, plus each adapter's current allowlist;
- verifies the North layer-28 steering asset is bound to the exact artifact and
  that startup uses `--no-thinking-auto-calibration`;
- prints every server cycle, subprocess check and named feature check, and
  writes the same plan to `dry-run.json`.

The checked-in CPU result is:

```text
DRY-RUN PASS cpu_only=true models=9 policies=25 smoke=18 sanity=18 ladder=9
```

## GPU launch and reruns

The coordinator should export the clean source root and invoke the runner
inside the lab GPU lease wrapper. The campaign itself also refuses an existing
`/tmp/gpu.lock`, creates an ownership-stamped lock, and removes only its own
lock. Every non-dry campaign invocation replaces `preflight.json` with a fresh
receipt generated from `MLX2_CAMPAIGN_ROOT`; a failed refresh stops before any
stage rather than reusing an older source or harness identity.

```bash
export MLX2_CAMPAIGN_ROOT=/path/to/pinned-clean-mlx2

~/Desktop/mlx2/.venv/bin/python \
  qualification/runs/quality-campaign-20260919/run_campaign.py --phase smoke

~/Desktop/mlx2/.venv/bin/python \
  qualification/runs/quality-campaign-20260919/run_campaign.py --phase sanity

~/Desktop/mlx2/.venv/bin/python \
  qualification/runs/quality-campaign-20260919/run_campaign.py --phase ladder
```

Pass one or more exact stage names after the options to rerun only those
stages. Previous entries are retained in `status.json.history`:

```bash
~/Desktop/mlx2/.venv/bin/python \
  qualification/runs/quality-campaign-20260919/run_campaign.py \
  --phase smoke smoke-north-ordinary smoke-flash-next-mtp2
```

`summarize.py` writes `SUMMARY.md` with one Markdown table per phase:

```bash
~/Desktop/mlx2/.venv/bin/python \
  qualification/runs/quality-campaign-20260919/summarize.py
```

## Phase 1: feature smoke

There are 18 route stages. Each uses four server cycles with `--cache-dir` and
`--apc-persist-dir` naming the same persistent APCv2 directory, as required by
the serving startup contract:

1. **base** — full `scripts/qualify_serving.py`, the core real-server matrix,
   and both official SDKs through `scripts/sdk_smoke.py --url` under
   `~/evalplus-venv/bin/python`;
2. **opt-in** — restart with constrained-tool grammar and tolerant markers;
   only routes that declare compatible cache state enable interior APC
   checkpoints, while native-MTP and external-DFlash2 routes also enable FLy;
3. **persist-seed** — seed a session/prefix and stop with persistent flush;
4. **persist-rescan** — restart on the same `--apc-persist-dir` and require a
   nonzero rescan hit.

After the server is stopped, each route stage runs
`scripts/gpu_check_apc_prefetch.py --i-own-the-gpu --model-path ...`. Running
it outside the server avoids loading a second copy of the model concurrently.
The probe waits boundedly for any leased session entries to finish their
deferred park at the generation worker's next idle boundary and records the
wait in `timings.park_wait_seconds`. Its real ServingEngine/APCv2 lifecycle can
also be reproduced without model weights or Metal:

```bash
PYTHONPATH=src ~/Desktop/mlx2/.venv/bin/python \
  scripts/gpu_check_apc_prefetch.py --cpu --dir /private/tmp
```

Every Gemma 3n and MiniCPM-o stage repeats the pinned `mlx_vlm` checkout,
revision, cleanliness, detached-HEAD, and import-origin checks before starting.
The stage records that receipt in `runtime.json`; all of its server and check
processes inherit the pinned checkout on `PYTHONPATH`. No text-only stage does.
The runner also watches the owned server process and `/health` while a check is
active. Normal `503` loading responses (`error: null`) are allowed for the
per-model 2,400-second startup window, and draining/quiesced/suspended states
are not mistaken for engine errors. An exited server or non-null engine error
terminates the startup or active check immediately. The standalone APC probe
applies the same rule to its in-process ServingEngine worker and caps its
startup wait independently of request limits. The CPU suite also runs Ruff's
F-code checks over `scripts/` and this campaign directory, including an
explicit F821 undefined-name guard.

`feature_smoke.py` records every check as `PASS`, `FAIL`, or `SKIP` with a
reason and writes each raw HTTP/SSE reply under the stage's `raw/` directory.
Every check has a subprocess limit and its own daemon-thread deadline, so a
hung request fails without pinning the campaign. The named matrix covers:

- chat/completions, SSE, stops, byte logprobs, `n=2`, seeded sampling,
  JSON-object, strict local-`$ref` schema and regex grammar;
- tool auto/required/named/single-call policy and both opt-in tool levers;
- default thinking, state-aware/history budgets, reasoning effort and grammar
  deferral after thinking;
- Responses text/SSE, function roundtrip, durable lifecycle, continuation and
  reasoning/logprob includes;
- Anthropic text/SSE/system/tools/tool-result/signed-thinking roundtrips,
  token counting, a 32,000-token accepted cap and stop sequences;
- APC repeated hits, write suppression, sessions, interior checkpoints,
  persistence/rescan and loopback admin quiesce/suspend/resume equality;
- FLy engagement and fixed-prompt ordinary/speculative output comparison;
- real image and silent-WAV inputs for both multimodal adapters, plus text.

The ordinary baseline is always scheduled before a speculative sibling. Exact
greedy equality is reported as a fraction. A non-identical but nonempty answer
does not get silently called equal; it is retained for semantic review as the
documented batch-numerics case.

A smoke stage passes only when every applicable feature check, qualifier, SDK
matrix, persistent restart and APC-prefetch probe returns zero. Explicit SKIPs
are allowed only for undeclared model/route capabilities.

## Phase 2: 20x20 sanity

The 18 route stages run the frozen 20 rounds x 20 concurrent protocol with 20
lanes, 40 inflight requests, 32K context and the same screens/graders as the
2026-09-18 campaign. The single documented too-strict grader was corrected:
the Muse answer that fulfilled “six sentences” is now graded on six sentences
and a 24-word degeneration floor, not an unrelated 30-word verbosity quota.
All real correctness checks remain unchanged.

A stage passes with no HTTP/protocol/leak/empty/degeneration hard failures, a
healthy drained server, and at least 85% graded correctness, exactly as the
template defines. Both multimodal models use text-only prompts in this phase.

## Phase 3: context ladder

One production/default route runs per model: native MTP2 for Qwen3.6, Qwen3.8
and Flash-Next; native MTP1 for Xing; ordinary for Muse, North, Laguna, Gemma
3n and MiniCPM-o. Server `--max-context` is each model's effective maximum;
the requested ladder itself ends at 256K. Gemma 3n and MiniCPM-o therefore
stop at their 32K effective ceiling, while North starts with its 500K limit.

Requested lengths are 1K, 4K, 16K, 32K, 64K, 128K and 256K; each is run at
width 1 and width 4, cold then immediately warm. The prompt is calibrated
through the live `/v1/messages/count_tokens` tokenizer boundary, which covers
all heterogeneous adapters more accurately than the older Qwen/Muse/North-only
renderers. At the context ceiling it reserves output/framing headroom and
records both the requested rung and actual effective prompt tokens.

Each cell records streamed TTFT, receipt-or-TTFT prefill tok/s, steady-state
decode tok/s, peak byte telemetry selected from `/v1/status`, APC hit count,
output hash and a unique needle answer. A ladder stage passes only when every
stream terminates, every needle is recalled, and every warm request reports an
APCv2 hit.

## Route decisions from current main

- **Xing:** yesterday's “five routes” were three routes on the requested 6-bit
  artifact plus two on a separate bf16 artifact. This campaign includes the
  relevant 6-bit ordinary, MTP1 and prompt-lookup routes; the unrequested bf16
  pair is not mislabeled as 6-bit evidence.
- **North:** current main wins over the older guard record. Thinking, the
  512-token guard and alpha 0.2 steering are adapter defaults. The exact
  calibration binding is preflighted and auto-calibration is disabled.
- **Laguna:** only ordinary is selected. The fused-MoE candidates remain
  implemented/observed but unqualified and unselected after full-model
  correctness and latency regressions.
- **Multimodal:** both adapters are ordinary-only; Gemma 3n declares image and
  audio (plus video outside this requested matrix), and MiniCPM-o declares
  image/audio. No speculative flags are invented.

## Estimated GPU wall time

These are planning estimates, not performance claims; long-context admission,
model load time and failures can move them substantially.

| Phase | Estimated serialized wall time | Main driver |
|---|---:|---|
| Smoke | 10–18 hours | 72 server loads, full qualifiers, SDKs, persistence and 18 standalone APC probes |
| Sanity | 2–5 hours | 7,200 graded requests across 18 route stages |
| Ladder | 36–72 hours | 252 cold/warm width cells, including 128K/256K prefill on seven models |
| All | 48–95 hours | Sequential sum plus service transitions and failure headroom |

The estimates intentionally favor completing evidence over minimizing GPU
time. Named reruns prevent repeating passing stages.
