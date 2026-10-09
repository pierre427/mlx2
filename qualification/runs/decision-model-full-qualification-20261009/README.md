# Source-bound decision-model qualification summary

All four standalone decision families passed the production qualification and
promotion contract on the merged private-main source identity
`07178885cc7bd8f52fe7a93682528c32a0e9953f8ff7172b3bedb89b1aeadf0b`.
The approved qualification producer is
`a6addcf209f28f35f31934df9bf8787a17d916f173dfd62f786945f2cd9513bb`.

| Family | Artifact revision | Gates | Source-derived head/output parity | Near-limit default truncation | Promoted |
|---|---|---:|---:|---:|---:|
| Clef Flash 9B | `8f9beb2e63474d547c70810b41b468d3b4a84932` | 13/13 | exact | 16,384 / 16,384 tokens | yes |
| Decision 2.0 Lux 9B | `e0cd13890ba2995c8ad1a464ff0cbd817f1aed7a` | 13/13 | exact | 16,384 / 16,384 tokens | yes |
| pplx-decider v1 27B | `b01a5cbaca5391f73bd55103d4f27e8982cd5e60` | 13/13 | exact | 32,761 / 32,768 tokens | yes |
| JEV 9B MLX 8-bit | `c0eafb30e090cb94cff6a762bd2c67baba29f131` | 13/13 | exact | 16,376 / 16,384 tokens | yes |

Each run covered artifact and tokenizer identity, the HTTP contract, semantic
oracles, repeat and restart determinism, a context ladder, a fail-closed
over-limit request, successful default truncation at at least 90% of the model
context, route receipts, Prometheus reconciliation, stability, and a
source-derived prompt/head/output comparison over the same runtime trunk and
frozen weights. This is not an independently loaded trunk comparison. The
promoted server then loaded the final receipt, advertised `qualified`, served a
real request with `observed_used: true`, reproduced the frozen direct answer,
and exposed a validated `/metrics` scrape.

The qualification scrapes recorded 64 successful requests, 24 expected
refusals, zero failures, 68 serialized executions, and 231,669 input tokens in
aggregate. Per-model load, CPU, dispatch, token, near-limit, and peak-RSS
observations are in `summary.json` and the four `*-metrics.prom` files. These
are uncontrolled single-host qualification observations, not performance
results or a cross-model comparison.

The full receipts, evidence documents, and GPU-lock records remain private and
apply only to the merged-main source identity above. They are intentionally not
included in this independent public projection. The public source is
unqualified until it is rerun and issued its own receipts. `summary.json` and
the four bounded `*-metrics.prom` files publish only aggregate qualification
statistics and Prometheus observations. No long-running production endpoint
was deployed by this run.
