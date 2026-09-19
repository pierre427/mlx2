# Qwen3.6 overnight campaign summary

Identity: `9b8ede62f5034c1f60d7f7e5b6034103322a749904c5e003c8b92716f0665d57`

This campaign does not install qualification receipts or change serving defaults.

| Phase | Step | Status | Reason |
|---|---|---|---|
| 0 | Full CPU/Metal-aware suite with bound preflight receipt | passed |  |
| 0 | Focused Qwen/APCv2/MTP/PLD/serving tests | passed |  |
| 0 | Git diff whitespace check | passed |  |
| 0 | Build distributions | passed |  |
| 0 | Validate qualification JSON inputs | passed |  |
| 0 | Candidate ports 8296-8298 are free | passed |  |
| 0 | All Phase 0 gates passed | passed |  |
| 0.5 | Immutable Qwen3.6 execution policies | passed |  |
| 0.5 | Identity-bound qualifier preflight receipt contract | passed |  |
| 0.5 | Tracker-required HTTP qualifier coverage | passed |  |
| 0.5 | Matrix per-cell fail-soft continuation | passed |  |
| 0.5 | Context arm alternation contract | passed |  |
| 0.5 | Activation helper process-group cleanup contract | passed |  |
| 0.5 | Qwen3.6 experiment manifest | passed |  |
| safety | External CPG GPU lease and matching gpu.lock | passed |  |
| 1 | Fused GDN ordinary round 1/5 | passed |  |
| 1 | Fused GDN ordinary round 2/5 | passed |  |
| 1 | Fused GDN ordinary round 3/5 | passed |  |
| 1 | Fused GDN ordinary round 4/5 | passed |  |
| 1 | Fused GDN ordinary round 5/5 | passed |  |
| 1 | Fused GDN mtp-artifact round 1/5 | passed |  |
| 1 | Fused GDN mtp-artifact round 2/5 | passed |  |
| 1 | Fused GDN mtp-artifact round 3/5 | passed |  |
| 1 | Fused GDN mtp-artifact round 4/5 | passed |  |
| 1 | Fused GDN mtp-artifact round 5/5 | passed |  |
| 2 | Product ordinary serving qualification | failed | command qualification failed |
| 2.5 | M artifact forced ordinary serving qualification | failed | command qualification failed |
| 3 | Native MTP2 serving qualification | failed | command qualification failed |
| 3.5 | PLD serving slice prerequisite | blocked | missing prerequisites: ['qualification/policies/qwen36-pld.json'] |
| 3.5 | Conditional PLD serving qualification | blocked | dependencies did not pass: ['pld-prerequisite'] |
| 4 | Alternating thermally controlled context ladder | failed | RuntimeError: no context arm has a passing serving and HTTP qualification step |
| 5 | Product ordinary max20 serving qualification | blocked | dependencies did not pass: ['ordinary-serving'] |
| 5 | M artifact forced ordinary max20 serving qualification | blocked | dependencies did not pass: ['mtp-artifact-ordinary-serving'] |
| 5 | Native MTP2 max20 serving qualification | blocked | dependencies did not pass: ['mtp2-serving'] |
| 5 | Conditional PLD max20 serving qualification | blocked | dependencies did not pass: ['pld-prerequisite', 'pld-serving'] |
| 5 | Independent 20 rounds x actual B20 | failed | RuntimeError: no batch_stress arm has a passing serving and HTTP qualification step |
| 6 | Matched performance report | passed |  |
| cleanup | Unconditional owned-process cleanup and handoff | passed |  |

## Counts

```json
{"blocked": 6, "failed": 5, "passed": 27}
```
