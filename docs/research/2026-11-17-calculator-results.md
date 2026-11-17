# Calculator tile: completed M5 0.6B results

Status at 2026-11-17 13:51 America/Toronto: all requested 0.6B runs completed and the production service was restored. Latest diagnostic repair: **805/1,000**, comprising **800/800 literal expressions and 5/200 word problems**. This reused-test result is not held-out accuracy. Current prose overhead is **12.31%**. Details below preserve each historical source and protocol; production defaults and qualification are unchanged.

Checkpoint: `Qwen/Qwen3-0.6B@c1899de2`, full HF revision `c1899de289a04d12100db370d81485cdf75e47ca`, native bf16, MLX-LM 0.31.3 on Apple M5 Max. Accuracy runtime source: `6ef4a540403e903763183f02478eae3a8179d34f`. Greedy, thinking disabled, identical custom prompts, maximum 512 new tokens per arm. GSM8K uses all 1,319 vendored test questions, zero-shot; this is not a claim of vendor-harness equivalence. A token-budget truncation counts incorrect.

| Suite | Stock | GPU calculator |
| --- | ---: | ---: |
| Arithmetic: addition/subtraction/multiplication/division/multistep/powers | 82/100 | 100/100 |
| 12-digit operands and messy decimals | 8/100 | 90/100 |
| Noisy word problems: four fixed template families | 100/100 | 100/100 |
| GSM8K | 741/1,319 (56.18%) | 732/1,319 (55.50%) |

The original scorer recorded 731 calculator GSM8K answers. Rescoring preserved raw outputs with source `6f09fe1fb`: a trailing decorative seven-hash line had erased one valid final answer. Original rows remain unchanged. The original arm comparison helped 16 and hurt 26; the corrected scoring restores one of those hurt rows. GSM8K truncations: stock 21, calculator 24.

The independent noisy-word expression-only extraction prompt scored 20/100 by exact semantic evaluation, despite 100/100 final-answer solving under the ordinary solve prompt. Raw extraction responses actually include distractor numbers or omit necessary grouping; this is not just a formatting mismatch. These four synthetic template families do not establish general word-problem reliability. OOD here means extrapolation beyond the arithmetic probe's setup; no claim is made that numbers were absent from model pretraining.

## Throughput profile

Source `6fa76c177`, same numerical/controller code as the accuracy run. Six repetitions of four non-math prompts, alternating arm order, warm routes, fixed 256 generated tokens, including tokens after EOS. All 24 pairs produced exactly identical token IDs; zero arithmetic activations. Median stock 328.23 tokens/s, calculator 289.63 tokens/s. Median paired calculator/stock ratio 0.88266, approximately 11.73% slower. This is a fixed-work decode measurement; a natural-EOS completion measurement remains useful before calling it standard response throughput.

Die maxima before samples ranged 46.04–58.60 C, maximum after 59.72 C. Admission ceiling 60 C. Existing system swap was 4,570.25 MiB at entry; this was not a zero-swap host. Offline generation does not exercise APCv2 or serving batches. Both profiles held CPG/paired locks and restored the preexisting production service afterward.

The prefilter skipped CPU history materialization on 6,228/6,240 checks (99.81%). Nevertheless, this profile slowed down: a device filter and host flag read still cost work/synchronization. Avoiding text decoding alone is not proof of a faster generation loop.

Natural-EOS follow-up: all 48 samples finished normally before the 256-token cap; all 24 stock/calculator pairs were token-identical and no calculation activated. Stock median 325.41 tokens/s, calculator 286.36 tokens/s; median paired ratio 0.88134, approximately **11.87% slower**. Maximum die temperature before samples 56.20 C and after 57.32 C. This confirms similar overhead on ordinary completions, without counting post-EOS tokens. Exact source and raw measurements are in `v4-natural-performance/metadata.json` and `perf.jsonl`.

## Control repairs and focused evidence

Source `6f09fe1fb` rejects a literal-only LaTeX left side (`1800 =`) instead of forcing `1800` and destroying an algebraic equation. Version 4 also stops forcing a newline after every numerical result: observed failures dropped units or changed subsequent reasoning even when the computed number was correct. Version 3 retains its explicitly versioned behavior. Focused CPU regressions pass. The 261-case owned model probe covers all 200 arithmetic/OOD cases, the previously changed GSM8K answers and initial GSM8K controls. That targeted set is diagnostic, not an unbiased replacement GSM8K accuracy result.

The first repair at `6f09fe1fb` scored 99/100 arithmetic, 90/100 OOD and 42/61 selected GSM8K. It exposed a remaining continuation issue: after the tile correctly emitted 105 for `3^3+78`, the model appended `+78` and answered 183. Source `09505a7a0` closes the completed numerical right side against operator/digit extensions, including after spaces/tabs, while preserving units and allowing a newline to begin another formula. Final focused results: arithmetic 100/100, OOD 90/100, selected GSM8K 43/61 versus stock 34/61 (original tile 24/61). Stock token IDs matched the original profile on all 261 cases. This establishes targeted repair evidence, not a new full GSM8K score.

Focused CPU regressions after the final changes passed; see `focused-tests.xml` and its source-bound receipt. The natural-EOS profile is complete above. The subsequent full GSM8K run is recorded below.

Numerical GPU/GEMM checks remain 478 comparisons, zero mismatches; 426 successful answers and 52 matching refusals. There are 13 GPU trigger tests, including merged delimiter pieces and exhaustive cue flags for the exact checkpoint vocabulary. GPU roots/modulo still explicitly refuse; CPU reference supports bounded perfect roots/modulo. Full GPU residency has not been implemented: parsing, dispatch, final formatting and tokenization still involve the host.

Durable raw results, configured artifacts, source metadata and ownership receipts: `qualification/runs/calculator-latex-m5-20261117/`. Ongoing repair outputs: `output/research/calculator-m5-20261117/v4-repair-probe/`.

## Fresh full GSM8K calculator-only rerun

Runtime `0767a35278a0d52ce1f5d8cd790857e90e26d34d`, same Qwen checkpoint/prompt/greedy no-thinking settings, 512-token cap: **750/1,319 (56.86%)**, with 21 capped responses. All 1,319 scored rows contain only the calculator arm; a short non-math ordinary reference was the sole baseline check. Against the previously retained stock answers (741/1,319), this helped 10 and hurt 1, a net gain of 9 answers or 0.68 percentage points. The original unrepaired full calculator profile remains separately retained at 732/1,319. Numerical arithmetic was Metal: all 730 recorded calculator events report `gpu_metal`, with no CPU arithmetic fallback. Host parsing/tokenization/dispatch/formatting remain CPU-side.

The 21 cap failures were retried at a 4,096 total output-token ceiling, batch width four, source `3746f0cc2`. The existing run-on alarm and spiral detector were active over visible completion text. Eight run-on alarms requested a single bounded final-answer continuation; no spiral alarm fired. These traces never reached the spiral detector's 2,048-token minimum before other termination, so that detector was enabled but not exercised on a live spiral. The run-on alarm has a 1,024-token admission minimum and tests current repetition evidence, avoiding a stale early alarm. The continuation does not receive an oracle answer/candidate. Its nudge counts inside the 4,096-token bound, and its own maximum is 512 tokens. One response exhausted that conclusion allowance; it did not reach 4,096 total tokens.

The retry recovered **4/21**, yielding **754/1,319 (57.16%)** under a two-pass policy: 512 tokens, then guarded/batched 4,096-token retries only for cap failures. This is not a uniform full-set 4,096-token result, and gains cannot be attributed solely to the budget: batching and defenses also changed. Eight B1/B4 controls passed the exact functional oracle, with seven token-identical continuations. Per-lane calculators and detectors are isolated, and physical rows are rebound on every decode forward after compaction. Raw receipts record widths 1–4. All 27 retry arithmetic events executed as `gpu_metal`.

## Frozen hard1000 A/B

The new arithmetic-focused corpus is committed under `benchmarks/calculator_hard1000/20261117/`. The protocol and seed preceded generation, all 1,000 exact answers were cross-checked with SymPy, and no model/parser calls selected cases. Dataset SHA-256: `ce1303c8c57f62202bf7b5ce894001d2874a612f10866909a0395d2e8156d980`. Source `2fd3337209dd8403f782e9b77da9ba2cf56b94d1` evaluated every frozen question in both native width-four arms, with matched 4,096-token bounds and defenses and alternating arm order. The original calculator completed all1,000 pairs after a synchronization-only recovery of the final144. No corpus outcome-based prompt/parser/gate changes were made in that original arm. See its protocol/README for the intentionally limited synthetic scope and resource exclusions.


## Hard1000 completion, stall diagnosis and repair follow-up

The original calculator scored **124/1,000 (12.4%)**, versus stock **3/1,000 (0.3%)**:121 helped,0 hurt. All1,959 arithmetic events report `gpu_metal`. The first856 pairs used the original harness; the remaining144 used exactly the same four calculator source files from `2fd3337209dd8403f782e9b77da9ba2cf56b94d1`, with a repaired synchronization harness. These accuracy rows are combined by unique case ID; timing from the two profiles is not combined.

The initial run stalled on a GPU event at856 pairs. The same batch stalled on retry, and the host was rebooted. A native30-second watchdog subsequently captured a model-forward wait even with the cue filter off and callbacks serialized. The offline harness had inherited `MLX_METAL_FAST_SYNCH=1` without production's single-stream binding. Pinning normal Metal events before initialization and binding MLX-LM's actual generation stream allowed both the failed batch and the remaining144 questions to complete, with the GPU cue filter enabled. The runtime already documents a cross-stream GPU fence spin hazard for the old configuration. This diagnosis is strongly supported by the controlled recovery; it is not a captured GPU instruction trace. Production defaults are unchanged.

A separate renderer defect affected200 frozen questions: `\timesi` is an unknown TeX command rather than `\times i`. The original corpus/results remain immutable. `benchmarks/calculator_hard1000/20261117-latex-repair/` changes only those200 question strings and preserves all1,000 IDs, numbers, expressions and answers; the other800 question strings are identical. Do not call the original1,000-input result a clean valid-LaTeX benchmark.

The follow-up adds strict comma grouping and the opt-in `complete_literal_requests` gate. This gate parses the complete actual user expression/count, performs numerical work on the selected backend, and emits the exact answer plus EOS before a neural continuation can replace or round it. It receives no expected answer or evaluator expression. General word problems remain on the model's ordinary extraction path. This is a deterministic calculator request capability, not a claim that Qwen learned arithmetic. The revised corpus and mechanism have now been inspected; the follow-up is diagnostic, not untouched held-out accuracy. Retained stock answers can be compared directly only on the800 unchanged questions.

Focused CPU validation:411 passed. Owned numerical counting/request checks:88/88 exact GPU-vs-CPU comparisons,3/3 GPU cue checks, and explicit Unicode refusal on the ASCII GPU count route. The final results are recorded below. Evidence is retained under `qualification/runs/calculator-latex-m5-20261117/wrapup/`.

## Final repaired calculator-only result

At `fe509e6142154544e35f058f155ea09d49c1113d`, the corrected corpus (SHA-256 `e3275786fcdfef72d88428a90dea417b608279a6a2010f7d6c28a3e5967ee2b5`) scored **805/1,000**. All eight explicit-expression categories scored100/100: **800/800 literal expressions**. The remaining200 are difficult word problems: finance0/100 and rates/conversions5/100. Seven word-problem responses reached their bounds. All894 recorded arithmetic events were Metal. On the800 unchanged question strings, the retained comparison is stock3, original calculator122, repaired calculator605. The other200 strings changed, so their retained stock results are not a matched-input A/B. This is diagnostic repair evidence on an inspected synthetic set; no claim of fresh held-out generalization.

Actual model counting/list-sum smoke:8/8 correct. Six events used GPU integer reductions and two used exact Metal arithmetic. The tests include strawberry, case-sensitive versus case-insensitive counts, empty lists, and decimal sums. This proves the bounded literal feature, not arbitrary semantic counting.

A separate current-source natural-EOS prose profile has48 samples (four prompts, six repetitions, paired arm order): median stock323.067 tokens/s, calculator283.841 tokens/s; median paired ratio0.876906, **12.31% slower**. All24 pairs are token-exact, no arithmetic activated, and no response was truncated. Die maxima before/after samples were60.04/61.32 C. The nominal60 C admission sample can drift before the measured call; these are recorded temperatures, not a claim of a hard60 C ceiling throughout. The same solve-prompt wrapper was used in both arms and elicits a numeric final marker even for prose; this is a controlled non-math workload, not a representative chat benchmark. Normal Metal synchronization and one bound generation stream were enabled.

The ownership window ended successfully at18:51:14 UTC; the CPG lease closed, paired locks released, and the existing production health endpoint returned200/ok. The current fix has not reproduced the stalled batch; it does not prove every possible driver hang eliminated. GPU arithmetic is exact within its bounds, but extraction/formatting/tokenization still use the CPU. Word-problem extraction and ordinary-text overhead remain limitations.

Validation:411 focused CPU checks,88 GPU arithmetic/counting checks,3 GPU cue checks and8 model counting cases. The four new source/checkpoint ledger rows validate independently. The full historical feature-grid check encounters an unrelated pre-existing unreachable source revision (`0852...`); no broad preflight or historical repair was required.
