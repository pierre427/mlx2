# Qwen3.6-35B-A3B: native MTP versus same-artifact ordinary decode — 2026-09-18

## Decision

Native self-MTP is a **net throughput loss** on this artifact at every depth
and width measured, against the same-artifact ordinary route. The ordinary
APCv2 route is the Qwen3.6 candidate to select; no MTP uplift claim is
possible. The earlier "B14+ capacity block" was host state, not a defect: with
82 GiB headroom B14, B16 and B20 all admit, and the measured per-lane
speculative transient (1.73 GiB at B20) matches the controller's 1.76 GiB
constant.

Receipts: `qualification/runs/qwen36-35b-a3b/mtp-vs-ordinary-20260918/`
(`*-benchmark.json` = `scripts/benchmark_serving.py --rounds 5 --widths 1 2 4
--max-tokens 160`; `*-capacity-*.json` = concurrent cohorts, 200 tokens each,
temperature 0.7, Metal memory polled at 4 Hz).

## Setup

Artifact `Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp`,
candidate mode, `--max-context 32768 --max-lanes 20 --max-inflight 24
--cache-bytes 8 GiB`, dedicated cache dir, production `mlx-serve` quiesced.
Routes: `qwen36-mtp2.json` (k=2), `{"num_draft": 1}` (k=1),
`qwen36-mtp-artifact-ordinary.json --ordinary`. Source hash of the running
servers: the tree at `49d1de3` plus the uncommitted design-item changes of
this session (width-lock fallback, disconnect detection, float32 top_p,
tenant-scoped cache knob, block-manifest MAC); none touch the decode path.

## Benchmark medians (aggregate tok/s, 5 rounds)

| Width | Ordinary | MTP k=2 | k=2 / ord | MTP k=1 | k=1 / ord |
|---|---:|---:|---:|---:|---:|
| B1 | 130.0 | 80.4 | 0.62× | 82.1 | 0.63× |
| B2 | 208.7 | 117.1 | 0.56× | 118.3 | 0.57× |
| B4 | 287.2 | 144.5 | 0.50× | 165.8 | 0.58× |

Median draft acceptance: k=2 0.16–0.24, k=1 0.49.

## Capacity sweep (concurrent cohorts, aggregate tok/s, per-lane Metal delta)

| Width | Ordinary tok/s | MTP2 tok/s | ratio | MTP2 GiB/lane | Ordinary GiB/lane | Admitted |
|---|---:|---:|---:|---:|---:|---|
| 4 | 262.0 | 94.8 | 0.36× | 3.18 | 0.15 | 4/4 both |
| 8 | 350.4 | 123.3 | 0.35× | 2.04 | 0.12 | 8/8 both |
| 13 | 335.2 | 187.5 | 0.56× | 1.01 | 0.11 | 13/13 both |
| 16 | 390.7 | 206.7 | 0.53× | 0.81 | 0.00 | 16/16 both |
| 20 | 422.8 | 162.5 | 0.38× | 1.73 | 0.11 | 20/20 both (MTP staged via `fewer_lanes`) |

MTP k=1 at B8: 138.1 tok/s (0.39×).

## Why

The A3B model's ordinary decode step is cheap (about 3B active parameters), so
the fixed cost of a speculative round — drafting plus a `k+1`-row verify
forward — is not amortized by 1.2–1.5 accepted tokens per round. The lab's
09-03 step timings said the same thing in isolation (plain 7.2 ms; k=1 round
14.4 ms; k=2 round 16.2 ms); this is the serving-level confirmation with the
same-artifact ordinary reference the port report required before any uplift
attribution. The adaptive-depth controller's depth-zero park is the mechanism
that would let an MTP route degrade to ordinary, but selecting the ordinary
route outright is simpler and avoids the 1–3 GiB per-lane speculative
transient.

## Not measured

Long-context (>32K) behaviour, the 262K near-limit cells, thermal drift across
rounds, and output quality parity between routes (both routes are exact
target-law sampling by construction; no divergence was observed in the
qualifier's greedy checks).
