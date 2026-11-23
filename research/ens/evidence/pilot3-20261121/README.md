# A0 pilot 3 (2026-11-21, M3)

Re-scoped extension families on the pilot-2 cores, plus a diagnostic c6r at tau 1.5. One seed; dev only.

- **Source:** code at sync stamp `c647d0ce0`, the re-scope before the review fixes. The pilot-2 cores come
  from `b5eca3e13`. Their training-family digests are unchanged.
- **Cores:**
  - c6 `ens-a0-core-c6-early-w64-s0@c25654a60499`
  - c7 `ens-a0-core-c7-early-w64-s0@e6f36f425f96`
  - c6r-t1.5 `ens-a0-core-c6r-early-w64-s0@300544eff3c9` (tau 1.5 over base tau 0.5; bus acc 0.234 after
    6k steps, so it did not learn)
- **Families:** program.ADD.h1, graph.6-2.h1, fst.3-1.h1, graph.4-2.h1, under their c647d0ce0 definitions.
  `ff798ac67` later changed held_div and the digests, so these modules no longer load under the current code.
- **z-blind baselines** for these held rows are in `zblind-20261121.txt`.
- **c6r debugging:** `c6r-debug/`.
