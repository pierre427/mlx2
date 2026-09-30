# mlx2 qualification results, 2026-09-29

The 2026-09-28–29 campaign was paused at the user's request. These are public summaries of source-bound private receipts, one page per M5-staged model. No raw prompts, model files, local paths, process details, or credentials are published.

## Scope

- M5 Max 128 GB: 34/34 staged models passed ordinary served smoke with expected defaults. The eligible 20×20 queue recorded 22 passes and four failures; eight roster entries were outside that queue.
- Flash-Next also has a separately retained contaminated 20×20 attempt, although it was excluded from the later eligible queue. This is why five model pages show a nonpassing 20×20 receipt while the queue itself reports four failures.
- M3 Pro 36 GB: six locally staged models passed smoke and 400/400 in 20×20. The other artifacts were absent on that host, not load failures.
- Qwen3.6 passed the complete M5 three-repetition context ladder. Qwen3.8 was partial. Muse's M5 ladder was deliberately interrupted after nine passing cells. Five M3 models passed a separate single-stream 1K/4K basic ladder.
- At the campaign pause, M3 North qualified eight applicable exercised feature operations. Other M3 feature attempts were partial or contaminated; no M5 per-model feature run had begun. The Flash follow-up below is later work.

## Flash-Next follow-up

Both staged Flash variants received focused M5 memory and feature checks after the pause. Their model pages distinguish combined feature observations, isolated APCv2 rolling recovery, optional kernel checks, and any 20×20 result. The new default avoids gate/up fusion's observed swap while retaining file-backed PLE. The M3 does not hold either large Flash artifact.

The Flash-Next page also reports the later opt-in TensorFold row-kernel and gated MTP/prompt-copy candidate when its thermal ladder has completed. It is a candidate measurement, not a default-route promotion or a full TensorFold executor qualification.

## Muse-Glimmer follow-up

The [Muse-Glimmer 4-bit page](muse.md) records the later M5 and M3 feature
qualification, lane-matmul comparisons, and an external DFlash2 candidate.
Its [single chart](muse-performance.png) plots the mixed 20×20 rates and
single-stream thermal ladders on separate axes.
Its family default is ordinary decode with the M5 lane wrapper off; the
full TensorFold fused executor has no Muse implementation. APCv2 rolling
recovery passed on both hosts. The M3 external-draft route remains an open gate.

A later [current-main comparison](upstream-main-comparison.md) records why the staged Flash-Next checkpoint did not load in TensorFold or omlx, plus a thermally controlled three-engine speed control using the shared Qwen3.8 27B 4-bit checkpoint.

## Nemotron 3.5 Lightning follow-up

The [Lightning result page](nemotron35-lightning.md) records M5 ordinary smoke, passing capped ordinary 20×20 runs, cooled three-repetition ordinary ladders through 262K, native-MTP candidate tests, and nine applicable feature engagement checks across a combined run and two isolated reruns. Eight-lane native-MTP batching is not qualified: the candidate verifier ran serial rows and its bounded eight-lane speculative round was slower than ordinary. The M3 could not safely load this artifact; its CPU and memory feasibility receipts are recorded without an M3 throughput claim.

The 20×20 rate is median aggregate generated tokens per second across mixed domain rounds. Context-ladder decode is per stream, from thermally admitted measured runs. They are different measurements. A functional smoke, stress pass, feature implementation, feature qualification, route selection, and observed production use are distinct states.

## Model pages

| Model | M5 smoke | M5 20×20 | M3 staged |
|---|---|---|---|
| [flash-next](flash-next.md) | passed | passed | no |
| [flash-next-uncensored](flash-next-uncensored.md) | passed | passed | no |
| [gemma3n](gemma3n.md) | passed | error | no |
| [gemma4-26b-bf16](gemma4-26b-bf16.md) | passed | not run | no |
| [gemma4-26b-q8](gemma4-26b-q8.md) | passed | not run | no |
| [gemma4-31b-bf16](gemma4-31b-bf16.md) | passed | not run | no |
| [gemma4-31b-q8](gemma4-31b-q8.md) | passed | not run | no |
| [laguna](laguna.md) | passed | error | no |
| [minicpmo](minicpmo.md) | passed | not run | no |
| [muse](muse.md) | passed | passed | yes |
| [muse-8bit](muse-8bit.md) | passed | passed | no |
| [muse-bf16](muse-bf16.md) | passed | passed | no |
| [muse-cyber-4bit](muse-cyber-4bit.md) | passed | passed | no |
| [muse-cyber-bf16](muse-cyber-bf16.md) | passed | passed | no |
| [muse-original](muse-original.md) | passed | passed | no |
| [nemotron](nemotron.md) | passed | not run | no |
| [nemotron35-lightning](nemotron35-lightning.md) | passed | passed, capped profile | no safe load |
| [north](north.md) | passed | passed | yes |
| [north-8bit](north-8bit.md) | passed | passed | no |
| [qwen36](qwen36.md) | passed | passed | yes |
| [qwen36-27b-8bit](qwen36-27b-8bit.md) | passed | passed | no |
| [qwen36-27b-heretic-4bit](qwen36-27b-heretic-4bit.md) | passed | passed | yes |
| [qwen36-heretic-4bit](qwen36-heretic-4bit.md) | passed | passed | no |
| [qwen36-ud-q8](qwen36-ud-q8.md) | passed | passed | no |
| [qwen38](qwen38.md) | passed | passed | no |
| [qwen38-crack-4bit](qwen38-crack-4bit.md) | passed | passed | yes |
| [qwen38-crack-8bit](qwen38-crack-8bit.md) | passed | passed | no |
| [qwen38-crack-bf16](qwen38-crack-bf16.md) | passed | passed | no |
| [qwen38-mlx-4bit](qwen38-mlx-4bit.md) | passed | passed | yes |
| [qwen38-mlx-6bit](qwen38-mlx-6bit.md) | passed | passed | no |
| [qwen38-mlx-8bit](qwen38-mlx-8bit.md) | passed | passed | no |
| [qwen38-uncensored-oq4e](qwen38-uncensored-oq4e.md) | passed | passed | no |
| [thinkingcap-27b](thinkingcap-27b.md) | passed | passed | no |
| [xing](xing.md) | passed | error | no |
| [xing-bf16](xing-bf16.md) | passed | contaminated | no |

## Scripts and provenance

The `scripts/` directory snapshots the smoke, 20×20, concurrency, thermal-ladder, feature, queue, and memory-probe scripts plus their execution policies. `scripts/manifest.json` records each original and published SHA-256. Two Python snapshots replace the run host's home-directory prefix with `Path.home()`; two policies use a literal `${HOME}` placeholder for the external draft path, and one local LaunchAgent label is anonymized. Other algorithm and test settings are unchanged. Restore the original qualification layout and set local model paths before running these historical snapshots; they also require the local MLX environment. The generated pages can be rebuilt from the private source-bound receipts with `build_model_pages.py --receipt-root <campaign-receipt-directory>`; `package_scripts.py` rebuilds the script snapshot from the private qualification source.

Private source receipt paths are recorded only as source commit IDs and artifact configuration hashes on the model pages. The private pause record is `qualification/runs/requal-20260928/PAUSED-20260929.md` at Forgejo commit `df0120ca`; it is intentionally not copied here.

## Open gates at pause

Flash-Next full-stress load swap; Laguna final-answer quality and thinking-marker leakage; Xing final-answer quality and BF16 swap; Gemma 3n supported-surface quality; Qwen3.8 M5 warm APCv2 misses and 262K admission; M3 multi-stream APCv2/admission limits; M3 Muse Fly repeated-prefix admission; remaining M5 performance and per-model feature qualification.
