# APCv2 persistence CPU qualification — 2026-09-18

These runs measure the production APCv2 persistence path using synthetic
hybrid checkpoints with four target KV planes, four recurrent `ArraysCache`
planes (two arrays each), two MTP draft KV planes, and an MTP tail-hidden
sidecar. MLX was set to `mx.cpu` before allocation. No Metal test, real model
load, or serving process was run.

The requested base was `271b21e`. The measured harness and payload-read fix are
at `08e605986e1b66937263fce941df816861290acb`; every JSON records that commit
and the benchmark script SHA-256.

## Machine and scratch disk

- Apple Silicon `arm64`, 137,438,953,472 bytes (128 GiB) physical memory
- macOS 26.7 (build 25G229), Python 3.12.13
- `/private/tmp` on `/System/Volumes/Data`, APFS SSD (`disk3s5`, Apple Fabric)
- APFS container: 3,996,276,899,840 bytes; about 767.3 GB free at campaign start
- Hard guard: abort before every allocation when `vm_stat` free plus
  speculative memory is below 24 GiB or the next allocation would cross that
  floor. Two early attempts aborted safely at 23.75 and 16.74 GiB; their
  generated persistence directories were removed by `finally` cleanup.

## Multi-GiB timing results

All rows persist about 8 GiB. Spill time includes safetensors serialization,
file and directory fsync, one SHA-256 computation per payload, and manifest
publication. Restore values are per-entry means. The no-verification arm is an
explicit benchmark-only bypass; production restore always verifies SHA-256.

| Entry GiB × N | Spill total | Spill GiB/s | Manifest total | Rescan | Restore no SHA | Restore + SHA | SHA pass/entry | Resume enqueue | Prefetch service | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 × 16 | 4.275 s | 1.871 | 34.30 ms | 5.60 ms | 18.11 ms | 218.62 ms | 200.05 ms | 0.51 ms | 217.22 ms | 4.14 GiB |
| 1 × 8 | 4.323 s | 1.851 | 17.45 ms | 3.22 ms | 33.07 ms | 434.53 ms | 401.68 ms | 0.39 ms | 434.68 ms | 5.44 GiB |
| 2 × 4 | 4.245 s | 1.885 | 8.56 ms | 2.01 ms | 66.43 ms | 868.56 ms | 803.57 ms | 0.31 ms | 866.01 ms | 10.42 GiB |
| 4 × 2 | 4.084 s | 1.959 | 5.04 ms | 1.55 ms | 136.72 ms | 1,733.67 ms | 1,596.36 ms | 0.30 ms | 1,728.07 ms | 11.76 GiB |

The restore arms ran without an OS cache purge, with the benchmark-only
no-verification arm first. The isolated SHA timer accounts for essentially the
entire verified-minus-unverified difference; these are warm-filesystem CPU
costs, not a cold-storage latency claim.

Every large run observed exactly three digest calls per entry (target, draft,
aux), zero digest failures, zero restore failures, and zero startup-rescan
payload reads. Prefetch restored one entry and the subsequent session-tagged
lookup recorded a prefetch hit.

## Rescan scaling

Small entries are 0.01 GiB each. The read trap raises if startup rescan opens a
`.safetensors` payload for reading; every row observed zero calls and zero
payload bytes.

| Entries | Payload GiB | Rescan wall | Internal rescan | Payload reads |
|---:|---:|---:|---:|---:|
| 1 | 0.010 | 0.986 ms | 0.586 ms | 0 |
| 8 | 0.080 | 3.059 ms | 2.284 ms | 0 |
| 16 | 0.160 | 5.448 ms | 4.213 ms | 0 |
| 32 | 0.320 | 11.087 ms | 8.923 ms | 0 |
| 64 | 0.640 | 18.698 ms | 14.029 ms | 0 |

The 64-entry wall time is 18.9× the one-entry time for 64× the manifests, and
the 8/16/32/64 ladder is approximately linear. No rescan payload read or
quadratic trend was observed.

## Production issue found and fixed

The first 0.5 GiB × 16 run exposed increasing park latency: 0.59 seconds for
the first entry, 4.67 seconds at the maximum, and 2.74 seconds mean. Profiling
showed `block_file_paths()` decoding each raw safetensors payload in full to
decide that it was not a JSON block manifest. Pin accounting repeated that
probe for previously parked entries, and restore performed the same redundant
payload pass.

Commit `08e6059` adds a bounded binary-prefix discriminator before JSON parsing
and a regression test that forbids whole-file `read_text()` and `read_bytes()`
on raw safetensors. After the fix, the 0.5 GiB park range was 0.257–0.278
seconds (0.267 seconds mean), and the complete 8 GiB spill was 4.275 seconds.
Startup rescan behavior was unchanged and remained metadata-only.

## Exact commands

```bash
PYTHONPATH=src ~/Desktop/mlx2/.venv/bin/python scripts/benchmark_apc_persistence.py --dir /private/tmp --entry-gib 0.5 --entries 16 --max-total-gib 8 --output qualification/runs/apc-persistence-cpu-20260918/entry-0.5gib-n16.json
PYTHONPATH=src ~/Desktop/mlx2/.venv/bin/python scripts/benchmark_apc_persistence.py --dir /private/tmp --entry-gib 1 --entries 8 --max-total-gib 8 --output qualification/runs/apc-persistence-cpu-20260918/entry-1gib-n8.json
PYTHONPATH=src ~/Desktop/mlx2/.venv/bin/python scripts/benchmark_apc_persistence.py --dir /private/tmp --entry-gib 2 --entries 4 --max-total-gib 8 --output qualification/runs/apc-persistence-cpu-20260918/entry-2gib-n4.json
PYTHONPATH=src ~/Desktop/mlx2/.venv/bin/python scripts/benchmark_apc_persistence.py --dir /private/tmp --entry-gib 4 --entries 2 --max-total-gib 8 --output qualification/runs/apc-persistence-cpu-20260918/entry-4gib-n2.json

PYTHONPATH=src ~/Desktop/mlx2/.venv/bin/python scripts/benchmark_apc_persistence.py --dir /private/tmp --entry-gib 0.01 --entries 1 --max-total-gib 8 --rescan-only --output qualification/runs/apc-persistence-cpu-20260918/rescan-0.01gib-n1.json
PYTHONPATH=src ~/Desktop/mlx2/.venv/bin/python scripts/benchmark_apc_persistence.py --dir /private/tmp --entry-gib 0.01 --entries 8 --max-total-gib 8 --rescan-only --output qualification/runs/apc-persistence-cpu-20260918/rescan-0.01gib-n8.json
PYTHONPATH=src ~/Desktop/mlx2/.venv/bin/python scripts/benchmark_apc_persistence.py --dir /private/tmp --entry-gib 0.01 --entries 16 --max-total-gib 8 --rescan-only --output qualification/runs/apc-persistence-cpu-20260918/rescan-0.01gib-n16.json
PYTHONPATH=src ~/Desktop/mlx2/.venv/bin/python scripts/benchmark_apc_persistence.py --dir /private/tmp --entry-gib 0.01 --entries 32 --max-total-gib 8 --rescan-only --output qualification/runs/apc-persistence-cpu-20260918/rescan-0.01gib-n32.json
PYTHONPATH=src ~/Desktop/mlx2/.venv/bin/python scripts/benchmark_apc_persistence.py --dir /private/tmp --entry-gib 0.01 --entries 64 --max-total-gib 8 --rescan-only --output qualification/runs/apc-persistence-cpu-20260918/rescan-0.01gib-n64.json
```
