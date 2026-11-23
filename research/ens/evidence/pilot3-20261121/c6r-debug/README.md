# c6r debugging, 2026-11-21 (M3, frozen pilot-2 c7 `ens-a0-core-c7-early-w64-s0@e6f36f425f96`)

Scripts run with the pilot-3 code (sync stamp `c647d0ce0`). Outputs as printed:

- `c6r_closed_20261121.py`: closed-form ingress reader R = M^-1 (E[V(z)] - E[NEUTRAL]), clean codebook messages.
  - fit error 8.7e-06; max |T| / |e_neutral| = 1.61.
  - tau 0.5: program.ADD 0.047, graph.6-2 0.352, fst.3-1 0.896, graph.4-2 0.496.
  - tau 1.5 and 3.0: 1.000 on all four.
- `c6r_path_20261121.py`: on *jittered* messages the exact reader scores only 0.252 on the mixed training
  batch. ||R*|| = 53, because the inverse of a random square codebook amplifies channel dropout and noise.
- `c6r_init_20261121.py`: gradient training does not move at ingress or early, with full or 0.01 reader
  init, over 1000 steps at batch 128.
- `c6r_jitter_20261121.py`: 1500 steps at ingress, tau 1.5, reader init 0.01, Adam 1e-3, all-bus rows.
  The mixed-family ceiling is about 0.63 (c7 typed).

| Reader | Acc clean | Acc jittered | Last loss |
|---|---|---|---|
| trained without jitter | 0.308 | 0.310 | 4.04 |
| trained with jitter | 0.382 | 0.378 | 2.86 (still falling) |
| ridge reader (lam 0.01–10) | 0.458 | 0.384 | — |
