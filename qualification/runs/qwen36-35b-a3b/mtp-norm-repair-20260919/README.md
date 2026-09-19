# Qwen3.6-35B-A3B self-MTP: norm-repair A/B (2026-09-19, M5 Max, Metal)

Artifact: `Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp`.
Server: `mlx2.server --max-context 32768 --max-lanes 20 --max-inflight 24
--qualification-mode` (self-MTP, num_draft 2). Driver: `scripts/benchmark_serving.py
--rounds 3 --widths 1 2 4 --max-tokens 160`, greedy. Both arms ran back to back
under `cpg_job.py --lock`.

- **base**: main `7daf236`. It loads the four unshifted MTP norms as stored.
- **fix**: `claude/ecosystem-probes-20260919` `7de929f`. `adapters/norm_repair.py`
  adds the missing +1.

| | pooled draft acceptance | B1 tok/s | B2 tok/s | B4 tok/s |
|---|---|---|---|---|
| base | 0.358 | 86.5 | 126.2 | 154.1 |
| fix | **0.740** | **135.4** | **180.5** | **216.6** |
| ratio | 2.07x | 1.57x | 1.43x | 1.41x |

Output identity:
- At B1 and B2, every output hash is identical across arms. Verify is lossless.
- At B4, both arms already differ from their own B1 warmup output, as the
  09-18 run did. One of the four prompts has a different hash across arms,
  because the fix changes the per-cycle verify shapes and so moves B4 batch
  rounding. That is batch-shape noise, not a correctness change.

`summary.txt` is the output of `analyze_ab.py` on the two JSON files.
