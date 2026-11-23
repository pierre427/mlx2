# A0 pilot 2 (2026-11-21, M3)

Harness pilot, one seed, not a decision run. Source `b5eca3e13` (clean, via the
M3 sync stamp). Host: M3 Pro 36 GB, MLX 0.32.4 GPU. Weights stay on the M3 in
`~/ens-lab/evidence/pilot2-20261121/`; only receipts and logs are here.

Cores (site `early`, w=64, seed 0):

| Arm | Checkpoint | Steps | steps/s | Peak GB | Final train acc |
|---|---|---|---|---|---|
| c6 | `ens-a0-core-c6-early-w64-s0@c25654a60499` | 12000 | 7.18 | 3.31 | bus 0.701, ordinary 0.693 |
| c7 | `ens-a0-core-c7-early-w64-s0@e6f36f425f96` | 12000 | 7.19 | 3.25 | typed 0.635, ordinary 0.619 |
| c6r | `ens-a0-core-c6r-early-w64-s0@a6affd048f26` | 6000 from c7 | 12.8 | 2.39 | bus 0.232 (readers did not learn) |

Held results: `eval-*/receipt.json`. Per-family oracle ceilings:
`ceilings-20261121.txt`. The reading is in `wiki/docs/ens/log.md`.
