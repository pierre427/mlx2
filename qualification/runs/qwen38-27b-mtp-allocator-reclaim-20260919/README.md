# Qwen3.8-27B self-MTP allocator cache at 64K (2026-09-19, M5 Max, Metal)

`scripts/probe_mtp_cache_memory.py --model Qwen3.8-27B-oQ4e-mtp --prompt-file
flash-next-65536.txt --context 65536 --gen 4096 --num-draft 3`. Single request,
3.15 tokens per MTP cycle. Each run was under `cpg_job.py --lock`.

| Arm | File | Max MLX cache during decode | Decode tok/s |
|---|---|---|---|
| Old default: clear every 512 steps (≈1,610 tokens) | `mem-27b-r512-t0.jsonl` | **70.4 GiB** | 21.4 |
| No periodic clear | `mem-27b-r0-t0.jsonl` | 82.4 GiB | 22.7 |
| Probe-side clear every 256 tokens | `mem-27b-r0-t256.jsonl` | 1.6 GiB (sampled every 256) | 23.6 |
| **New default**: 512 steps **or** 256 emitted tokens | `mem-27b-shipped-fix-s100.jsonl` | **10.5 GiB** (mean 5.9, sampled every 100) | 27.4 |

Active memory peaks at 28.8 GiB in every arm. With the old default, the
freed-buffer pool reached about 2.4× the working set. That is the Ollama
v0.34.2 failure mode. The 256-token clear did not slow decode. Run-to-run
variance in decode speed is visible across arms (for example, prefill time
ranged 96-138 s), so these rates show there is no regression. They do not
measure a speedup.
