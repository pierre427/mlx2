# GPU viability and correctness of the 2026-09-18 integrations

Scope: does each newly connected mechanism engage on a real model, produce
correct output, and leave the server healthy. **Not** a performance
qualification and not a selection claim. One server at a time on port 8297,
`--qualification-mode`, 32K context, 4 lanes, 2 GiB APCv2;
`com.example.fn-uncensored-mlx-serve` quiesced for each window and restored
(`state: ready`) after it. Driver: `run_campaign.py`; per-stage state in
`status.json`; the superseded first pass is under `first-pass/`.

All passing receipts bind runtime source `5f2f75dfb521…`.

| Stage | Model | What ran | Result |
|---|---|---|---|
| `qwen36-ordinary` | Qwen3.6-35B-A3B | full qualifier, 28 checks | **pass** — includes `structured_output`, `structured_output_sampled` and the new `structured_output_thinking` (automaton engine, grammar deferred past `</think>`) |
| `qwen36-mtp2` | Qwen3.6-35B-A3B | full qualifier, 36 checks | **pass** |
| `qwen36-pld` | Qwen3.6-35B-A3B | full qualifier, 31 checks | **pass** |
| `muse-ordinary` | Muse-Glimmer-30B | full qualifier, 25 checks | **pass** |
| `muse-pld-rotating` | Muse-Glimmer-30B | probe + full qualifier, 29 checks | **pass** — `feature_prompt_lookup_rotating_replay` observed |
| `north-pld` / `north-pld-rotating` | North-Mini-Code | probe, knob off vs on | **pass** — 3/3 outputs byte-identical; 33 replay rounds, 6 rollbacks, 0 refusals, 0 rebuilds |
| `muse-pld` / `muse-pld-rotating` | Muse-Glimmer-30B | probe, knob off vs on | **pass** — 3/3 identical; 54 replay rounds, 13 rollbacks, 0 refusals, 0 rebuilds |
| `north-spomin` | North-Mini-Code | probe, 7 checks | **pass** — 13,278 → 10,206 tokens, answers "Lisbon", protected-prefix fact recalled |
| `muse-spomin` | Muse-Glimmer-30B | probe, 7 checks | **pass** — 13,225 → 10,153 tokens, same answers |
| `qwen36-approx-k8v4`, `qwen36-approx-q8` | Qwen3.6-35B-A3B | probe, 6 checks each | **pass** — needle recalled through quantized KV at 5.3K tokens, repeat identical, 3 quantized lanes batch |
| `qwen38-approx-k8v4` | Qwen3.8-27B | probe, 6 checks | **pass** |
| `north-ordinary`, `north-spomin-qualifier` | North-Mini-Code | full qualifier | **fail, unrelated** — see below |

What the Spomin and approximate-KV probes establish beyond "it ran": the
edited/quantized state is never published (`cached_tokens == 0` on the repeat,
`apcv2_store_skipped_approximate` advancing, APCv2 bytes unchanged), a short
prompt on the same server stays exact and warm-hits normally, and a compacted
lane decodes in the same batch as exact lanes.

## Defects this run found (fixed, suite 1020 passing)

1. **Spomin declined nearly every request after the first.** Any warm APCv2
   hit — in practice the shared chat-template prefix, ~110 tokens — was
   classified `shared_apcv2_state`. The boundary cache is an extracted
   request-private object and the backend only builds new arrays, so the
   decline was removed. First pass: 3 applied / 4 declined; after: 4 / 0.
2. **Rotating replay was unreachable on the only models with ring caches.**
   North and Muse did not declare `Capability.PROMPT_LOOKUP`, so
   `--prompt-lookup` refused to start. They declare it now; the route itself
   needed no change.

Probe-only correction: North writes a short preamble before answering, so the
12-token answer budget was raised to 96.

## Open

- **North full qualifier fails `shared_cohort_priming`** with and without
  Spomin: on the harness's ~32.6K-token `data data …` filler North emits two
  tokens and stops (`content: ""`). The 14 checks before it pass. This is the
  model on a degenerate prompt, present before this work; it blocks a North
  route receipt until the harness uses a realistic long prompt for it.
- Approximate KV with a warm *exact* prefix (`requantized_prefix_hits`) was not
  exercised: every lane on those servers is approximate, so nothing is stored.
- Flash-Next (127 GB) was not loaded: its Qwen4 Spomin backend still refuses
  hybrid state and it does not declare approximate KV.
- Nothing here measures throughput; rotating replay in particular is verified
  exact, not faster.
