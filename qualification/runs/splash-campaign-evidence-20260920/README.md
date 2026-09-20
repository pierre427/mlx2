# Splash port campaign — GPU evidence (2026-09-20)

Artifacts behind the verdicts recorded in
`mlx-uag/wiki/docs/experiments/splash-port-campaign-2026-09-19.md`.
They were produced in a session scratchpad and are landed here so the
verdicts cite files rather than assertions.

- `d3-27b-qualification/d3-qualification.json` — the num_draft=3 qualification
  for Qwen3.8-27B. Three interleaved reps, arm order reversed, every request
  mechanism-gated. Verdict: DOES NOT QUALIFY, keep num_draft: 2. Width 1 reads
  -0.2% with a 0.9% within-arm spread; width 2 is a 10.7% regression. tau rises
  10-20% in every cell (d2 2.1324 -> d3 2.3387 at w1) while per-proposal
  acceptance falls 0.566 -> 0.441, so the third draft forward costs roughly what
  its extra acceptance returns.
- `gate-logs-06-07-08-10/` — gate runs for items 06, 07, 08 and 10 on the
  rebased trees, with the route asserted per arm from each smoke receipt.
- `gate-logs-12/` — item 12's gate, 3/3 both arms.
- `gate-logs-first-audit/` — the first audited pass over all 14 specs.
- `THRESHOLD-DERIVATION-06-07-08-10.txt` — thresholds re-derived against
  post-fix admission, written before those runs rather than after.

Item 03's sweep, item 01's kernel A/B and item 10's prefill run are in their own
`qualification/runs/` directories. Item 05 is NOT on main: proven safe (logprobs
rows cannot reach the device top-32 path) but never qualified.

## Corrections to earlier summaries of this evidence

Verified against these files rather than against the session transcript:

- `gate-logs-first-audit/` previously landed as a dangling symlink; the content
  is now here. The other `latest` symlinks are resolved to real directories.
- The per-item gate tallies quoted in early summaries (06 32/0, 07 22/0,
  10 25/0) do not reconcile with these logs, which show hardening iterations
  (06 20/2, 07 18/0, 08 30/2 -> 46/2 -> 43/1, 10 21/0) and failing spec
  summaries along the way. Read the logs, not the summaries.
- The d3 run in `d3-27b-qualification/` gives width 1 -0.08% median, width 2
  -8.1% median / -10.7% mean, and width 4 **+11.4%**, which no summary
  mentioned. It reads opposite to `claude/rm16-lifted-depth-cap-20260920`
  (+12.4% at width 1) and the two were RECONCILED rather than left open: the
  runs disagree on acceptance, not on method. rm16's prompt holds per-proposal
  acceptance nearly flat from d2 to d3 (0.692 -> 0.663) so fewer rounds win;
  this run's falls from a lower base (0.566 -> 0.441) so the third draft
  forward cannot pay for itself. `num_draft` is therefore workload-dependent
  on this artifact, and 2 stays the default because it is the better choice
  when acceptance is low, where being wrong costs ~10% at width 2. Cite either
  throughput figure only with its acceptance beside it. Full reconciliation:
  `mlx-uag/wiki/docs/experiments/splash-port-campaign-2026-09-19.md`.
- `harness/` holds the queue runner and gate evaluator used for these runs.
  They lived only in a session scratchpad and are landed so the method is
  inheritable, not because they are production tooling.
