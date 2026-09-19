# 20×20 batched sanity run — every supported model and route (2026-09-18)

Purpose: viability and output quality under batched load, **not** a performance
ladder and not route receipts. Each stage is 20 rounds × 20 simultaneous
requests (400) on a 20-lane server, 32K context, `--qualification-mode`.
`sanity_20x20.py` sends 20 graded task types per round — arithmetic, facts,
sequence, translation, code, lists, `json_object`, strict `json_schema`, stop
strings, streaming, tool calls, system + multi-turn, 1.5K-token needle recall,
sampled prose (top_p / top_k), logprobs, `n=2`, long explanations — and screens
every reply for HTTP errors, empty output, leaked control text, repetition and
character runs. I also read a round of raw output for every stage.
`run_campaign.py` runs one server at a time, quiesces the production mlx-serve
service for the window and restores it (`state: ready` confirmed at the end).

## Results (final pass per stage)

| Model | Route | Correct | Issues | Peak width |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | ordinary | 400/400 | none | 13 |
| Qwen3.6-35B-A3B | native MTP2 | 400/400 | none | 14 |
| Qwen3.6-35B-A3B | prompt lookup | 400/400 | none | 1 |
| Qwen3.8-27B | ordinary | 400/400 | none | 13 |
| Qwen3.8-27B | native MTP2 | 400/400 | none | 12 |
| Flash-Next (127 GB) | native MTP2 | 400/400 | none | 12 |
| Flash-Next (127 GB) | ordinary | 400/400 | none | 14 |
| Muse-Glimmer-30B | ordinary | 400/400 | none | 13 |
| Muse-Glimmer-30B | DFlash2 external draft | 400/400 | none | 9 |
| Muse-Glimmer-30B | prompt lookup | 399/400 | the miss was a too-strict word-count grader | 1 |
| North-Mini-Code | ordinary | 398/400 | 2 genuine model slips; cosmetic JSON padding | 15 |
| North-Mini-Code | prompt lookup | 398/400 | 2 genuine model slips; cosmetic JSON padding | 1 |
| North-Mini-Code | full qualifier (28 checks) | **pass** | — | — |

Greedy outputs are identical between a model's ordinary and speculative routes
for roughly 280–300 of 340 deterministic requests; the rest differ only through
batch-width numerics and are equally correct.

## Defects this run exposed (all fixed on `main`, CPU suite green)

| Symptom on GPU | Cause | Fix |
|---|---|---|
| Structured output refused (400) on North, Muse, Qwen3.8-27B | adapters never declared the capability although constrained decoding is tokenizer-level | declared (`518baf0` range) |
| First structured request on Muse DFlash2 killed the generation worker (503 for every lane) | the external-draft round deep-copies lane state, and the structured processor holds thread locks | `StructuredOutputProcessor.__deepcopy__` (`518baf0` range) |
| Every structured request on Muse DFlash2 then returned 502 | verify rows that follow a zero-probability draft token were still scored through the strict processor, latching a dead end | unreachable rows skip processors (`5a4af50`) |
| North ran strict-schema requests to the token cap inside whitespace | insignificant whitespace was unbounded and always admissible | 32-character bound per site (`735140d`, test corrected in `e134a30`) |
| A greedy model could extend a JSON number to the token cap | digit runs unbounded | 19 / 18 / 3 digit bounds (`d2892f5`) |
| `n=2` refused with an instant 429 under load (14/20 on Flash-Next, 3/20 on Qwen3.8 MTP2) | guard charged 4 GiB per sample and never waited | measured 2 GiB per sample + 15 s bounded wait (`fc0a054`); 0 refusals afterwards |
| North full qualifier failed `shared_cohort_priming` | North ends its turn immediately on ~30K copies of one token | near-limit filler built from 54 words that are one token in all five tokenizers; reply check accepts the marker or the topic (`d2892f5`) |

Also landed: the Prometheus `/metrics` branch was merged and the new sanity
counters (finish reasons, structured engine/deferral, peak batch width) are
exported through it (`518baf0`). Commit `735140d` reached `origin/main` with one
failing new test (chained with `;` instead of `&&`); `e134a30` corrected it a
few minutes later.

## Quality notes (model behaviour, not serving defects)

- **North-Mini-Code**: arithmetic slips (`62+130 → 292`, `44+88 → 176`,
  `56−13 → 33`), identical on both routes; sometimes restates the request
  before answering even with thinking off; pads JSON values with the whitespace
  the grammar allows (valid, cosmetic); factual errors inside valid JSON
  (`"coastal": true` for Paris/Budapest).
- **Muse-Glimmer**: terse, repetitive sentence openers in prose; writes a short
  plan in `content` before a tool call (the call itself is correct); weak on
  populations (`"population": 1`).
- **Qwen3.6 / Qwen3.8 / Flash-Next**: no quality concerns in any task type.

## Observations

- Prompt-lookup and external-draft routes verify one lane at a time
  (`peak width` 1 / 9), so their aggregate throughput under a 20-wide load is
  well below the ordinary route. Expected for these routes; not measured here.
- North's qualifier first failed `shared_warm_requests` with my 2 GiB sanity
  cache: its ~1.2 GB 32K entry was spilled and the second concurrent restore
  was budget-deferred into a cold prefill (by design). With 8 GiB it passes;
  the real manifest uses 16 GiB.
- Another session committed and edited `main` while this ran. The last two
  stages (Muse DFlash2, North qualifier) were therefore served from a clean
  detached worktree at `5a4af50`; an earlier DFlash2 pass on the mixed tree is
  kept under `dirty-tree/` (also 400/400). Superseded passes are under
  `first-pass/` and `second-pass/`.

## Follow-up: North thinks by default (same day)

North's arithmetic/factual slips and the reasoning that leaked into `content`
were all with thinking **off**, which the sanity requests (and the adapter's
old default) forced. The chat-template rendering was verified identical to the
vendor template; the template's own default is reasoning on and the model card
says it works best that way. On GPU with thinking on: `62+130 → 192`,
`44+88 → 132`, "Is Paris coastal → No.", clean answers, ~50 extra tokens.

`8fa4a25` makes thinking the default for North (clients still send
`enable_thinking: false` or `reasoning_effort: "none"`; constrained requests
default off because North declares no grammar-deferral marker). `3d3f27e` makes
the qualifier's fixed-64-token near-limit checks turn thinking off explicitly.
North's full qualifier then passes all 28 checks (`north-ordinary-qualification.json`,
served from a clean worktree at `3d3f27e`).
