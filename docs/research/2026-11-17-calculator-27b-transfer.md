# Calculator tile transfer to Qwen3.8-27B

Checkpoint: `Qwen3.8-27B-BF16-q8-mlx2@2dd8d304` (the existing q8 artifact, 64 layers, hidden width5120). The 27B experiment is separate from the completed `Qwen/Qwen3-0.6B@c1899de2` measurements. No production default changed.

## What carries over

The bounded LaTeX parser, exact Metal arithmetic, GPU token-cue filter, literal counting and result-writing controller are reused. The gate's learned weights and read-layer binding are specific to the checkpoint. The previous 27B gate was bound to a particular neural-tile host layout; this experiment does not remove that binding or reuse the0.6B weights. Its baseline is the ordinary, uncarved 27B model.

The complete-literal-request shortcut had been connected only to the offline0.6B harness. Source `97e23aca5` wires the actual current user turn through the27B adapter into the shared calculator factory. It refuses to discard image parts or reuse an older user turn after a tool/assistant continuation. The option remains opt-in. Successful calculator plans now report their actual arithmetic execution backend in route counters. These repairs passed421 focused CPU checks; the final focused request/controller subset passed234.

## Calibration, separate from scored questions

The first bounded calibration at `97e23aca5` used160 positive training examples and12 negative examples per category, two English sequences per domain and candidate layers24/36. Neither gate met the existing validation bound, so no artifact was used. The process exited and its ownership window restored the service. This was a calibration rejection, not a GPU hang.

The retry at `72dd72b8d50a68bd0fcb7498e109460c7ef8c881` uses the historical27B recipe on the ordinary model:400 positive training examples,40 negatives per category and the full preexisting English splits; candidate layers24/36/48/56. Train/validation/test contain920/690/780 sequences and30010/21609/33148 labelled positions. Seeds and held-out templates follow the existing probe builder. The scored128 transfer questions are not used for fitting or threshold selection.

The selected gate reads layer24, with threshold0.9. Validation recall is100%, and the worst negative-group gate-open rate is0.9504%, within the unchanged1% bound. All285 held-out answer positions open the gate; the legacy parser recognizes225/285. Thus gate detection and expression extraction remain distinct. On the held-out teacher-forced set, nonmath negative groups have zero combined gate-plus-parser fires; two arithmetic nonanswer positions do fire (0.0792% of that group). These are probe diagnostics, not the live served result or a claim that the broader v4 parser cannot misfire.

Before calibration, the ordinary model answered the16-token-bounded reference prompt with exactly `4<|im_end|>`; logits were finite. The model loaded in6s with27.0GiB of active MLX allocations. This is a load observation, not a controlled performance benchmark.

## Frozen served A/B protocol

`benchmarks/calculator_transfer/20261117/` freezes128 cases before27B calibration:40 existing hard-corpus cases (first four in each category),20 basic math,16 large-number/decimal problems,8 noisy word problems,20 fixed GSM8K test questions,8 literal counting/list-sum cases,12 exact-copy negatives and4 prose prompts. There are112 numeric-answer cases,12 copy oracles and4 prose controls. These are reused diagnostic cases, not an untouched benchmark.

All stock batches run first, then all calculator batches, at width4. Each arm starts a fresh loopback server on port8391 with the ordinary route and matching default cache settings. No external draft, native MTP or neural carve tiles are enabled. Sequential and batched controls run before scored batches. The calculator artifact selects v4 parsing, GPU arithmetic, GPU cue filtering and complete-literal requests.

The common4096-token cap is accompanied by client-side run-on and spiral alarms using actual emitted token IDs. An alarm cancels the response and counts it incorrect; this protocol does not inject a conclusion or an expected answer. HTTP streams have a60-second read timeout; ownership and process groups are released in cleanup. Logprob-ID diagnostics perturb timing, so these wall times are not publishable throughput measurements. The two ordinary-reference controls that reach their128-token cap are retained as capped controls, not silently scored as correct.

Normal Metal events (`MLX_METAL_FAST_SYNCH=0`) and the generation stream binding are used. The GPU does the exact arithmetic; Python still parses, formats and tokenizes. This is not a GPU-only parser or a CUDA/vLLM implementation.

## Ordinary reference

The ordinary reference finished all 128 cases on source `72dd72b8d50a68bd0fcb7498e109460c7ef8c881`. It got **60/112** numeric answers and **12/12** copy checks. The run-on and spiral guards stopped 48 responses, including 39/40 hard cases. All 20 basic arithmetic cases and 19/20 GSM8K cases passed.

The calculator server rejected its startup arguments before loading. The harness passed the policy inline and the CLI expected a path to the policy JSON file. Source `30f7740ae` corrects the argument. The ordinary rows, startup error, checkpoint, probe report and ownership receipt are saved under `qualification/runs/calculator-27b-transfer-20261117/ordinary-reference/`.

The optimized comparison uses the local Qwen3.8-27B DFlash2 drafter with the same target, questions and width. The v4 calculator arm uses ordinary decoding. The experimental resident calculator has a separate speculative integration that requires its own served validation.

## Completed comparison

The optimized baseline ran DFlash2 with the `lane` drafter kernel and `q4g64` drafter quantization. It completed all scored questions at source `d40241494`. The corrected v4 calculator arm ran at `11b81c2bd` after fixing a second tokenizer unwrap that had prevented the first request from reaching the model. All eight sequential and batched controls passed.

| Case group | Ordinary stock | DFlash stock | v4 calculator |
|---|---:|---:|---:|
| Hard literal expressions | 0/32 | 0/32 | 32/32 |
| Hard word problems | 0/8 | 0/8 | 1/8 |
| Basic arithmetic | 20/20 | 20/20 | 20/20 |
| Large numbers and decimals | 5/16 | 7/16 | 16/16 |
| Noisy word problems | 8/8 | 8/8 | 8/8 |
| GSM8K sample | 19/20 | 19/20 | 19/20 |
| Literal counting and list sums | 8/8 | 8/8 | 8/8 |
| **Numeric total** | **60/112** | **62/112** | **104/112** |
| Exact-copy controls | 12/12 | 12/12 | 9/12 |

The calculator recovered 42 numeric cases that DFlash missed and preserved every numeric answer DFlash got right. The terminal receipt recorded 105 plans, with 96 Metal arithmetic evaluations and nine GPU integer reductions. The calculator pass includes control traffic in these process counters. Seven of the eight hard word problems hit the guards in the calculator arm. This diagnostic set shows a gain on explicit expressions and little change on the sampled word problems.

The v4 tile appended results to three requests that asked for exact copies of text ending in an equals sign. The subsequent copy guard in `51e93d18e` leaves the token scores untouched for explicit verbatim instructions. Focused CPU tests reproduce all three refusals. The table retains the failures from the served comparison.

The compressed results, controls, source identities and GPU ownership receipts are in [the run archive](../../qualification/runs/calculator-27b-transfer-20261117/). The diagnostic wall times include logprob serialization and the routes emit different amounts of text. They do not establish controlled decoder throughput.
