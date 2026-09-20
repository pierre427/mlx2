# rm04 GPU qualification, 2026-09-19/20 (Flash-Next MTP, Qwen3.8 27B MTP)

Harness: `scripts/qualify_interior_checkpoints.py` (run under the lab GPU
lock). Arms interleave across rounds; cold references are taken per arm *and*
round in the same server process; one request per workload is replayed cold
twice as a determinism control.

| File | What it is |
|---|---|
| `smoke/smoke2-before-pool-fix.json` | first working smoke: shared_system hits, RAG captured 4 / published 0 |
| `smoke/smoke3-after-pool-fix.json` | same smoke after the interior count pool fix: RAG hits |
| `flashnext-shared-rag.json` | 3 arms x 2 rounds, auto at headroom_fraction 0.25 (RAG starved) |
| `flashnext-rag-headroom1.json` | RAG at headroom_fraction 1.0 with the budget gauges |
| `flashnext-shared-rag-headroom50.json` | headroom 0.5; exposed the cross-process text comparison |
| `flashnext-shared-rag-final.json` | 3 arms x 2 rounds, per-round references: W1/W2 go |
| `flashnext-linear.json`, `flashnext-linear-repeat.json` | no-harm control before the continuation gate: +3.9%, +7.2% |
| `flashnext-all-gated.json` | **headline**: shared_system + rag + linear, auto at headroom 0.5 and min_uncached_fraction 0.5 |
| `flashnext-linear-branch.json` | aborted on admission backpressure (429), kept as the reason the harness now retries |
| `qwen38-27b-shared-rag.json` | 27B confirmation at scale 0.1 |
| `*-logs/` | per-arm server logs |
| `*.try*.log` | queue-wrapper job logs (`rc` in the matching `.done`) |

Every server in these runs imported this worktree: `main` (db25d0e) has no
`apc_interior_checkpoints` support at all, and each result carries
`apcv2.interior.max_entries` and the `apc_interior_budget_mib_last` gauge,
which exist only on this branch.
