# rm10 — confidence-scheduled self-MTP depth (2026-09-19/20)

**Verdict: no-go.** The confidence arms ran 14.0/5.7/3.2% below fixed depth on
27B at 1/2/4 lanes, so the pre-registered "never more than 2% below fixed"
clause fails. Separability was never the problem: holdout top-1 AUC is
0.93-0.94 at the first drafted position on all three models, and the trained
head adds 1-2 points. `best-constant-*.json` combines the measured cycle table
with the measured conditional acceptance and finds the adapter cap
(`num_draft` 3) optimal at every width on 27B and 35B-A3B, 7-15% ahead of
depth 2, while one arm's run-to-run spread is about 3%. There is nothing for a
per-cycle policy to win.

The scheduler itself is **not** on `main`. What was folded is the measurement
apparatus that produced this verdict: the acceptance logger, the offline
trainer (`scripts/train_mtp_confidence.py`), `scripts/best_constant_depth.py`
and this directory's GPU harness.

## Read the cost profiles with care

`cost-27b.json`, `cost-35b.json`, `cost-27b-v2.json` and `cost-35b-v2.json`
are schema `mlx2.mtp_verify_cost.v1`. In that schema **only `cycle_table`,
`raw` and `probe_overhead` are measured**. The `m_points` / `costs` pair is a
*hardcoded literal* — the cold NAX 27B qkv-4bit curve — that the writer copied
into every profile it emitted, regardless of the model or the host. It is not
a measurement of anything in this directory, and above `m = 4` it is invented.

Two sessions read betas off those arrays before noticing. The writer now emits
schema `v2`, which drops the pair entirely and names every remaining array in
a `provenance` block. `scripts/best_constant_depth.py` reads only
`cycle_table` and prints a warning when handed a v1 file.

The `-v2` suffix in these filenames is the *second measurement pass*, not the
schema version; both passes wrote v1 files.

## Layout

- `profile-*.log`, `cost-*.json` — cycle-cost tables (`profile`).
- `accept-*.jsonl`, `collect-*.log` — acceptance logs (`collect`).
- `conf-*.json`, `conf-*-report.json`, `top1sts-*.json` — trained heads and
  holdout ECE/AUC (`scripts/train_mtp_confidence.py`).
- `best-constant-*.json` — `scripts/best_constant_depth.py` output.
- `ab-27b/`, `ab-27b-v2/`, `ab-35b/` — interleaved serving A/Bs; `summary.json`
  carries the per-width medians and the width-1 greedy identity check.
- `diag-27b-b1.json` — per-arm cycle-time decomposition.
