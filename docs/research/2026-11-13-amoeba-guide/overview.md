# Amoeba: a model built from a core and loadable experts

Date: 2026-11-13. Audience: engineers who know what a transformer is but have not followed this work.
Companion: [pipeline.md](pipeline.md) gives every step in order, with scripts and checks.

**Status.** Everything here is research. No core, box, adapter or policy is qualified, and nothing is a
serving default. The work lives on unmerged branches (see [Where things live](#8-where-things-live)).
Results give a control where one was run; a dash in a control column means none was run. Where an early
claim was later corrected, this guide gives the corrected number and says so. An independent reviewer
checked this guide against the sources, and its corrections are applied.

## 1. The idea

A language model stores general language ability and topic knowledge in the same weights. Amoeba asks
whether the two can be separated and recombined:

1. **Core.** Keep the part of the model that does general English. Drop the rest.
2. **Experts.** Package each skill or topic (capital cities, arithmetic, French) as a small unit that can
   be loaded when needed.
3. **Gate and router.** Decide, per request and per token, which experts run. An expert that is not
   needed costs nothing.
4. **Ingest.** Take an expert from a different, stronger model and attach it to a small core.

The hoped-for results are a smaller always-resident model, better quality on chosen topics, and experts
that can be mixed and matched across cores.

Three of those four steps now work at small scale. The fourth works in one form and fails in another.
Section 3 gives the detail.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| Donor | The model a skill is taken from. |
| Host | The model that receives the skill. |
| Core | The subset of a model's MLP neurons kept for general English. It is a learned yes/no mask over neurons. |
| Box | A set of neurons outside the core that restores one skill when switched on. |
| Gate | A small per-token switch that turns a box on only where it is needed. |
| Router | A classifier on the core's internal state that picks which box a request needs. |
| Sink | The first token position. Many models use it as a parking place for attention, and it needs the full model. |
| Adapter (LoRA) | A small low-rank weight change added to the host. Here it is trained to copy a donor's answers. |
| Side stream | A second forward pass (a box, or the host plus an adapter) that runs beside the host with its own state. |
| Hitch | The join between a side stream and the host. Ours mixes output probabilities: `p = (1 − a·g)·p_host + a·g·p_side`. `g` is the gate (0 or 1) and `a` is a mixing weight chosen on validation data. |
| Path | Alternative expert content built from donor runs at one thinking-effort level. The only paths built so far are small weight changes (rank 16) on four late layers, not neuron sets. |
| Tile | The export format for a core or a box: packed weights plus a manifest. |
| English cost | Extra cross-entropy on English text, in nats per token, against the unmodified model. For side streams it is measured against the host alone. Lower is better; 0 means no damage. |

## 3. What works

Unless stated otherwise, the model is `Qwen/Qwen3-0.6B@c1899de2` (28 layers, 86,016 MLP neurons).

### 3.1 Splitting one model into a core and boxes

Three cores appear below. They were built by different recipes, so their sizes differ slightly.

| Name | Recipe | Used for |
|---|---|---|
| Ladder core | Dropped neurons replaced by their mean; 68.5% kept at +0.18 nats | The first size ladder, nesting, the 217-neuron box |
| Bisected core | Same replacement, one ranking cut to a target cost; 73.0% and 85.6% kept | Core sizes and byte savings |
| Export core | Dropped neurons set to zero, so it can be saved as a checkpoint; 74.3% kept | Skill boxes, gates, router, pruning, sink, served check |

| Finding | Result | Control |
|---|---|---|
| A learned core keeps English with far fewer neurons than chance (bisected core) | +0.18 nats with 73.0% of neurons kept; +0.05 nats with 85.6% | A random ranking needs 93.9% and 99.0% for the same cost. A random core of the same shape costs +0.94 |
| Cores nest (ladder core) | 96–98% of a smaller core lies inside the next larger one, so one ranking gives every size | 46–69% for random cores |
| The core sheds topic knowledge first (bisected core) | Capitals accuracy is 0.05 at the 73% core and 0.80 at the 86% core, equal to the full model | – |
| Boxes restore skills on top of the core (export core) | Capitals 0.53, arithmetic 0.43, French 0.48 with box sizes chosen on the fit split. With sizes chosen on the reporting split, which is optimistic: 0.62 / 0.49 / 0.62 | Full model 0.69 / 0.38 / 0.70; core alone 0.09 / 0.00 / 0.00 |
| A box works on countries it was never tuned on (ladder core) | 217-neuron capitals box: 21 of 91 fresh countries. That box size had been chosen on an older held-out split. The box chosen on the fit split (523 neurons) gets 16 of 91 | Full model 24 of 91; core 0; five random boxes 0 each |
| Per-token gates let several boxes coexist (export core) | Three boxes together, as capitals / arithmetic / French: 0.66 / 0.44 / 0.56 | The same boxes without gates: 0.53 / 0.22 / 0.46 |
| Gates stay closed on ordinary text | Open on 0.3% of English tokens (capitals and arithmetic boxes) | – |
| A router picks the right box for trained phrasings | On held-out requests in trained phrasings, routing is perfect and results equal the oracle | All boxes loaded with no router: 0.53 / 0.22 / 0.46 |
| Boxes can be pruned hard (capitals / arithmetic, gated) | Half the neurons: 0.69 / 0.50. One tenth (532 neurons): 0.59 / 0.23 | Unpruned 5,196 neurons: 0.69 / 0.51 |
| Boxes can be found without labels | The best of 64 co-firing clusters (209 neurons) adds +0.34 capitals when added and −0.28 when removed. The arithmetic box overlaps another cluster at 0.22 | Chance overlap 0.021. No random-cluster control was run for the add/remove test |
| The first position needs protection (export core) | English cost +0.611 with no protection, +0.209 with the sink rule | Keeping layers 0–2 whole: +0.226. Adding only three neurons: +0.243 |
| A saved checkpoint matches the masked model | Zero logit difference against the dense masked computation | Against the sink rule itself it differs: top-token agreement 0.82–0.96 |

Caveats on this table:

- **Router on new phrasings.** At layer 8 the router scores 0.868 on unseen phrasings and hard negatives,
  but the layer was chosen on that same set, so the figure is optimistic. On those phrasings it sends 14
  of 32 arithmetic prompts to "none".
- **Fresh countries.** The box chosen on the fit split is significantly below the full model (p = 0.008).
  The fit split does not pick the best box.
- **Pruning.** A 248-neuron capitals box trained in place scores 0.66, against 0.69 for the full box. Random
  neurons trained the same way score 0.59, so which neurons are kept matters little once they are trained.

Two findings about how the pieces behave:

- **The box carries a specific job.** At layer 20 the model must know which country is being asked about.
  That signal reads 0.98 in the full model, 0.75 in the core, 0.97 with the capitals box added, and 0.60
  in a random core. A review flagged this probe's scaling as slightly optimistic.
- **Model families differ.** SmolLM2-360M keeps its facts inside its English core (capitals 0.81 at a 78%
  core, base 0.91), and it shows no first-position effect. Screen each donor family before assuming the
  Qwen recipe applies.

### 3.2 Attaching an expert from another model

> **Cross-architecture mapping (topography), 2026-11-14.** A depth warp plus ridge maps translates one
> model's hidden state into another's coordinates, well enough that the host keeps working. This holds for
> three donors: Qwen 27B, Gemma-4-26B (a different family and tokenizer) and Flash-Next (MoE). Shape-only
> methods (Gromov-Wasserstein) fail, so alignment needs shared text. The first real transfer test
> (phase 1) and the five-step transfer pipeline are in `cross-model-remapping.md`.

The donor for the side box and the adapter is `mlx-community/Qwen3-4B-4bit@4dcb3d10`. The skill is
zero-shot addition, scored on 600 held-out sums.

| Finding | Result | Control |
|---|---|---|
| A fine-tuned copy of the host gives an exact box | On the same Qwen3-4B checkpoint, the box reproduces the fine-tune to 6.2e-5 in the logits (75.2% capitals, 100% currency, for a fictional atlas of 120 invented countries) | Random delta: 1.25% / 2.9% |
| A self-contained side box with an output hitch works | Qwen3-0.6B: 29.3% → 96.7%, English +0.0017 nats. Against the fair baseline (host with a newline first, 90.2%), the box reaches 98.2% | Random box of the same shape: 28.3% |
| The same box works on a host with a different tokenizer | SmolLM2-360M: 5.5% → 96.2%, aligned by token strings. With the newline: 61.7% → 97.7% | Random box: 5.2% |
| A gated adapter does the same job in 81 MB | Rank-64 LoRA trained on the donor's answers. Gated by its own confidence: 98.2%, English −0.0018 nats | Always on, the same adapter costs +0.215 nats |
| A gate on the host's own state is precise | No false firing on numbers in prose, dates or prices; at most 0.2% on subtraction and multiplication; 0.05% on English | Shuffled-label gate: host-level accuracy. Random gate: +0.43 nats English |
| With that gate, a closed expert costs almost nothing | Newline host, adapter behind the host-side gate: 97.8%, English +0.0000, 83.5 tokens/s against 85.6 for the host alone | Side stream always running: 21.8 (box), 34.1 (adapter) |
| Donor boxes can be located in large models | For one problem, a working box is 21–45% of each of four donors, from 26B to about 121B parameters | Random boxes of the same shape all fail |

Caveats on this table:

- **Two gates, two trade-offs.** The confidence gate needs the side stream to run all the time (about 34
  tokens/s). The host-side gate allows lazy running, but on the raw host it misses the phrasing "The sum
  of a and b is" (42 of 300), because the host has already gone off course. No single arm has the best
  accuracy, the lowest English cost and host speed together.
- **Was the donor needed?** No adapter was trained on the correct answers alone, or on the host's own
  outputs. For addition, correct answers are easy to generate, so this result does not show that the
  stronger donor mattered.
- **The exact box copies damage too.** It reproduces the fine-tune's loss of real capitals (98.4% →
  51.6%), and keeping the skill needs most of a 13-layer stack.
- **Donor box location** is one problem, searched on a coarse grid of sizes, and the box computation is
  not bit-exact for two of the four donors. Nothing has been moved into a host.

The side box is the one true mix-and-match result so far: the same expert, unchanged, was attached to
two different host models with different tokenizers. An adapter is cheaper but belongs to one host.

### 3.3 Reading and mapping tools

| Tool | What it tells you | Key number |
|---|---|---|
| Neuron atlas | Stable ids, co-firing clusters ("virtual experts") and tags for every MLP neuron of a dense model | 896 virtual experts for Qwen3-0.6B. An address book, not a measured claim |
| Topic heads | Which attention heads serve a topic | Capitals: five heads in layers 14–20. Arithmetic: a layer-20 cluster. French: layers 0–2 |
| Expert token signatures | What a mixture-of-experts expert does, read from weights alone; matches experts across models with a shared vocabulary | 4 tight matches over 2,473 experts. Useful for triage only |
| Residual ablation overlay | Whether one direction in the hidden state controls a behaviour | On Qwen3.5-4B, keyword refusals go from 100% to about 0% at KL 0.04, and a random direction does nothing. About half the answers become soft refusals |
| Map shape | Whether the attribution map of an answer predicts that it is right | AUC 0.79 against 0.74 for controls alone |

### 3.4 Thinking effort

The model is `Qwen3.8-27B-oQ4e-mtp@fb827b65`, a quantized 27B, on 200 GSM8K problems.

| Finding | Result |
|---|---|
| Low effort matches the highest effort on the quantized model | Low 0.96 at 308 tokens; xhigh 0.95 at 475 tokens |
| Forcing a conclusion on a runaway thought is safe | It broke 0 of 26 runs that were already correct, removed every capped run, and saved 27–33% of tokens |
| A path can repair a small model's runaway thinking | Qwen3.5-4B with thinking on: 0.375. With a "low" path distilled from the 27B: 0.880 at 358 tokens |

Caveats:

- **Full precision changes the first row.** On the unquantized 27B, xhigh scores 0.978 against 0.945 for
  low (6 fixes, 0 breaks, p about 0.03). Low effort is the better choice only on the quantized checkpoint.
- **The path result is a repair, not a gain.** The 4B answers 0.905 with thinking off. The 0.375 baseline
  ran under a 4,096-token thinking budget, and 122 of its 200 runs never finished. A rerun with a larger
  budget is queued.

### 3.5 Prompt format

Models without a begin-of-sequence token treat the first prompt token badly. One newline at position 0
lifts Qwen3-0.6B from 29.3% to 90.2% on zero-shot addition, and the 4B donor from 342 to 398 of 400.

In a screen of 14 simple skills, no skill was absent in the 0.6B model and present in the 4B. Each
apparent gap did one of three things:

- it closed with the newline or a two-shot prompt (3-digit subtraction: 5% → 87–95%);
- the 4B was weak too (word reversal: 15%);
- it was a partial gap (3-digit × 2-digit multiplication: host 38% at best, donor 73%, or 88% two-shot).

Rule: a raw zero-shot score is not evidence that a skill is missing.

### 3.6 The same pipeline on a 27B model (in progress, 2026-11-14)

Host `Qwen3.8-27B-BF16-q8-mlx2@2dd8d304`: Pierre's clean BF16, quantized to q8 on NVMe. On 1,488
held-out positions it costs −0.00002 nats against BF16 and agrees on 97.6% of top-1 tokens. Branch
`claude/amoeba-27b-80bc6a`, run note `docs/research/amoeba-27b/RUN_NOTE.md`.

| Step | Result |
|---|---|
| Core | 52.7% of MLP neurons kept at +0.18 nats (random ranking 82.2%), 63.4% at +0.05 (random 93.5%). A random core with the same per-layer counts costs +1.25. The exported core keeps 53.7%: English +0.160 to +0.168, capitals 0.46 against 0.94 |
| First position | No effect. Unlike Qwen3-0.6B, the 27B needs no protection for position 0, so every layer is tiled |
| Atlas | 2,048 virtual experts (32 per layer). A knowledge pool of 516,060 neurons (46.3%), split into 64 clusters with no labels |
| Cluster functions | Each cluster was added to the core and removed from the full model, against a random twin with the same per-layer counts, on 32 held items per task. Three clusters carry a skill clearly: C32 (digits) arithmetic +0.72 / −0.75; C60 French +0.53 / −0.44; C22 (" capital", " is") capitals +0.25 / −0.22. English cost is within 0.016 nats of the twin. 53 of 64 show nothing on the six tasks tested |
| Boxes | Carved on frozen weights; sizes chosen on the fit split, scored on held-out data. Arithmetic: 685 neurons, 0.32 → 1.00. French: 1,289 neurons, 0.42 → 0.96. Capitals, completed on all 103 facts: 8,140 neurons; 0.97 on trained facts in new wording, but only 0.575 on 40 unseen countries, against 0.95 for the full model. No box adds more than 0.002 nats of English cost |
| Tile precision (window 5) | A q16 core gains nothing over q8 and costs 5.4 GB more. Boxes at q4 score the same as at q8, and all five together are about 0.1 GB. A q4 core saves another 5.2 GB, for +0.012 nats and 4–7 points on the boxed tasks |
| Memory and speed (B1) | Full q8: 27.0 GB, 17.4 tok/s. Core q8 tiles: 21.8 GB, 20.8 tok/s. Core q4 plus boxes q8: 16.6 GB, 21.6 tok/s. The q8 core saves 31% of the MLP bytes, less than the 46% of neurons dropped, because q8 packs the down projection in 64-neuron groups and the core drops none of them whole |
| Served 20×20 (window 6, dev port) | Full model: 400/400 at B1–B4, decode 18.4 tok/s per request at B1. Core alone: 314–347 of 400, 15–30% faster at every width. **Core plus boxes: 377–386 of 400.** French and capitals go from 0/20 to 20/20, addition from 4/20 to 18/20, multi-turn recall stays 20/20. **Not yet fixed:** core plus boxes decodes at about 10 tok/s, because five unfused boxes per layer add about 2,000 small GPU operations per token; the fix is one grouped projection per layer. Multiplication stays at 7/20, because the arithmetic box was carved on addition only |
| Topic map | 42 of the 64 clusters labelled against taxonomy v1.2 (428 tested topics), with random-twin and shuffled-label controls; 23 of 64 pass the functional check. The co-activation puzzle stage placed 35% of the unlabelled clusters' neurons; 9.8% of all neurons are still unexplained. FN-vs-27B donor screen: FN better on 16 topics, the 27B on 35. See [topic-map.md](topic-map.md) |

**Caveat on the core (2026-11-14):** the core was carved on TinyStories, repo markdown and donor text. So
code, shell, git and inference-stack knowledge stayed in the core *by construction*, and 117 registry rows say
so. Pierre's direction: carve future cores on graded, high-quality prose (corpus v2: 3,786 texts in 24
level × register cells, built; 585 of them Artemis-31B-generated creative texts, labelled and withheld), as **nested tiers**:
- T1, a grade-school core;
- T2, a high-school extension carved with T1 fixed;
- T3, a carved bf16 residue for deep thinking, switched on at xhigh effort or by token entropy;
- T4, expert boxes;
- T5, a post-graduate English box for poetry, lyrics and screenplay.

Each tier is scored on a ladder: GSM8K and ARC for grade school, high-school MMLU and ARC-Challenge, then
MMLU-Pro and GPQA-diamond. The plan is in `docs/research/amoeba-27b/GRADED_CORES_PLAN.md` (approved).

**Partition rule (Pierre, 2026-11-14):** no neuron is discarded. Every carve ends with the core, the named
boxes and remainder boxes holding every neuron exactly once. With every gate open, the parts must reproduce
the full model, and each export checks this. Box "pruning" is withdrawn as a method.

What is new compared with 0.6B: the clusters were tested against random twins, which the 0.6B run never
did, and they still isolate real skills. A gain of 0.03 is one item in 32, so labels with a gain under
about 0.2 are weak.

#### Update, 2026-11-15

| Step | Result |
|---|---|
| **Fused tiles** (one grouped projection per layer, gates as a mask; branch `claude/amoeba-27b-spec-tiles`, not yet merged) | Unsettled ABBA ×2, 512-token outputs (window 11c). Decode tok/s per request, B1–B4: fused core plus boxes 22.3 / 21.0 / 18.9 / 17.6 against full model 19.1 / 17.9 / 15.9 / 14.6, i.e. 17–21% faster. Peak 35 GB against 45 GB. Fused against unfused at B1 greedy: 28/32 identical, 2 near ties, 2 failed near-ties on French. |
| **Tiles plus AST/DFlash2 tree speculation** | The tiled arm is the fastest of all. Total tok/s B1–B4: 42.7 / 88.7 / 101.2 / 101.0 against full AST 41.1 / 75.8 / 89.4 / 92.4, i.e. +4 to +17%. Mean acceptance length is a little lower (3.3–4.2 against 3.6–4.5), and the cheaper verify more than pays for it. GSM8K acceptance run: 6.51 against 8.38; median 88.0 against 89.1 tok/s at B1. Unsettled, not publishable. |
| **Chat GSM8K, thinking off** (100 questions, 512 tokens) | Full 0.87, core 0.27, core plus 5 boxes 0.34. The core loses multi-step maths, which is the case for the T3 reasoning residue. |
| **Six-problem xhigh re-probe** (q8, ×3 each) | xhigh at 8K: 10/18 at 4,294 mean tokens, 4 caps. Spiral detector plus conclude: **13/18 at 2,202 tokens, 0 caps**. 16K budget: 611 and 304 6/6, so those failures were budget. Problem 549 is a model error (q8 6/6 wrong, BF16 2/3 wrong). |
| **Remainder-box catalogue** | A math.arithmetic tile (7,238 neurons, C45+C49) takes multiplication from 0.09 to 0.75 (full model 0.81) and sums from 0.25 to 1.00. World capitals +0.22, French +0.34. |
| **Graded T1 (grade-school, lib driver)** | 61.3% of MLP kept at d_CE +0.175 on L1 prose. A random same-count core gets 0.97, and a random ranking needs 87.9%. At +0.05: 78.6% kept, capitals 0.87. Expository is the most expensive register. No sink effect. |
| **Graded T2 (high-school, T1 fixed)** | At +0.18 on L2 held: **76.2%** kept (random free units 0.234, random ranking 0.814), capitals 0.87. At +0.05: 91.5%. T1 ⊂ T2 holds. The technical register is the most expensive in every cut (0.395 at +0.18): that is the code-like text the plan expected to fall out. The spare-split export (69.2%) is under-sized, because two registers have no spare text, so the held-target cut is the better tier boundary. |
| **Carve-driver gradient check, all 64 layers** | Layer-wise against whole-graph rel L2 0.081 (gate 0.02 fails), minimum per-layer cosine 0.990 (passes), identical loss. bf16 noise grows with depth (1% at 4 layers, 8% at 64). The driver needs `--grad-tol` of about 0.1 for the 27B. Graded tiers use the proven `amoeba27b_lib` meanwhile. |

## 4. What did not work

These are recorded so nobody repeats them.

| Idea | Result |
|---|---|
| Packing several box neurons into one, using unused dimensions | Accuracy breaks at 2:1 (arithmetic 0.51 → 0.02). Pruning works; packing does not |
| Copying a box into a different model through linear maps | Qwen3-0.6B → Qwen3-4B: 98% on trained facts, 0% on held-out facts. A plain adapter and a random box also score 0% |
| Two exact boxes active together in one model | Each works alone; together they score 39% / 46% and damage existing knowledge |
| A reasoning box | Core plus box scores 0.110 on GSM8K, the same as the base model. An adapter on the full model reaches 0.36, and 0.32 with the model's own traces |
| Looping layers instead of thinking in tokens, with no training (Qwen3.5-4B) | Lowered the probability of the right answer on every arm |
| "The box beats the full model" | Withdrawn. The box had been chosen on the reporting split. On fresh data it is at or below the full model |
| The first served check (170 and 182 of 320) | Void. Its exports broke the first position |
| Escalating effort when the model looks unsure | Uncertainty predicts errors (AUC 0.86), but escalation costs more tokens than always using low effort |

## 5. What we think we can do with it

Each item is marked **shown**, **partly shown** or **not shown**.

### 5.1 Mix and match cores and experts

- **One core, many experts from the same model: partly shown** at 0.6B, for three lookup skills, with
  gates and a router. The router is only reliable on phrasings it was trained on.
- **One expert, several different hosts: shown** for one skill, with the side box and output hitch. The
  expert reads the text itself and keeps its own state, so the host's internals and tokenizer do not
  matter.
- **Experts distilled from another model: shown** for one skill, as a gated adapter. This is the cheapest
  form by a wide margin: 81 MB against 4.09 GB for the side box, with equal or better accuracy on most
  test sets. Whether the stronger donor was needed is not shown.
- **Several distilled experts on one host: partly shown** (round 3b, Qwen3-0.6B host, newline prefix).
  Four rank-64 adapters behind one router, all resident: English cost +0.0012 nats, 323 MB added, plain
  English at host speed when run lazily. Hours→minutes rose from 127 to 297 of 300 (donor 299), and
  sorting three numbers from 228 to 300. 3×2-digit multiplication did not move (118 → 126; donor 257),
  at any rank: the adapter bought a prompt, not a skill. The router opened the wrong adapter on 11% of
  the steps where it opened one, which sank one sorting phrasing to 103 of 300.
- **One joint adapter beats several separate ones.** A single rank-64 adapter trained on all four skills,
  behind the same router, matched or beat the four separate adapters at a quarter of the memory (that
  sorting phrasing 261 against 103; 5-digit hours 182 against 79; English +0.0001). Routing mistakes
  between skills stop mattering.
- **For these skills the donor was not needed.** Adapters trained on correct answers alone did as well
  or better than those trained on the donor's answers, and clearly better on the harder sets.
- **Copying a donor's neurons into a different model: not shown, and the first attempts failed.** We now
  think the working route joins models at the output, or trains the host against the donor's answers.
- **Swapping cores under a fixed set of experts: not shown.** Cores have only been built per model.

Our current picture of the architecture: the host stays whole or lightly reduced; each expert is a gated,
lazily run side stream; a router on the host's own state opens at most one expert per token.

### 5.2 Topic quality

- **Raising reliability on a skill the host half has: shown** once. With the gated adapter, addition on
  the newline host went from 90.2% to 97.8%, and 4-digit sums from 79% to 88%.
- **Restoring knowledge the core dropped: partly shown** for capitals, arithmetic and French. On fresh
  data a box brings capitals near the full model's level, not up to it.
- **Exceeding the donor: not shown**, and not expected from distillation.
- **Giving a small model a skill it lacks entirely: not shown.** We found no such skill between the 0.6B
  and 4B models among 14 simple skills.
- **Better multi-step reasoning: not shown.** A box did nothing; an adapter helped, mostly without the
  donor.

### 5.3 Efficiency

| Gain | Evidence today | Caveat |
|---|---|---|
| Fewer resident bytes | The bisected cores save 12% (73% core) and 6.4% (86% core) of Qwen3-0.6B. The exported core keeps 74.7% and saves about 11% | On the served 20×20 task check the exported core scores 197 of 320 against 263 for the full model. The saving is not free |
| Fewer bytes read per token | Boxes are small and run only where the gate opens: 1.3 MB for the 217-neuron capitals box; 28 MB, 5 MB and 21 MB for the gated capitals, arithmetic and French boxes | The MLP is 44% of this model's parameters. Attention and embeddings (664 MB of 1,192 MB) are untouched |
| Fewer tokens | On the quantized 27B, low effort saves 35% of tokens at equal accuracy. Forced conclusion saves 27–33% at xhigh | One benchmark (GSM8K). On the full-precision 27B, xhigh is more accurate than low |
| Host speed kept | A lazy expert behind a host-side gate leaves decode at 83.5 of 85.6 tokens/s | Measured in a plain one-lane loop, not on the serving stack. Needs the host-side gate |
| Cheaper precision for experts | Boxes tolerate 4-bit weights while the core stays at 8-bit: capitals 0.656 | The comparison is confounded by the first-position handling |
| Speculative decoding | The core alone drafts and core plus boxes verify: 2.6–5.1 tokens per verify step | A CPU proxy. A full-model verifier accepts about as many (2.7–4.9), so the boxes add nothing here; the core is simply a good drafter |

Our honest reading: on a 0.6B model the byte saving is small. Larger models spend more of their bytes in
the MLP (67% of parameters in Qwen3-4B), and mixture-of-experts models more still, so the saving should
grow with size. The first point supports it. On `Qwen/Qwen3-4B@1cfa9a72` (commit `265bbff14`, confirm
branch), matching +0.18 nats keeps 64.8% of MLP neurons (random ranking 83.4%; random core of the same
shape +1.61) and saves 23.5% of resident bytes, against 12.0% on the 0.6B. At +0.05 nats it keeps 79.5%
(random 92.9%) and saves 13.7%. Capitals survive better too: 0.56 at the 65% core. The confirmation
checks added `mlx-community/Qwen3-1.7B-4bit@3b1b1768`: 72.6% kept (random 88.5%), so between 0.6B and
1.7B the fraction barely moves. A 4-bit control at 0.6B leaves it unchanged (73.0%).

The 27B run (section 3.6) gives the third and largest point: **52.7%** kept at +0.18 nats and 63.4% at
+0.05 (random rankings 82.2% and 93.5%). On that checkpoint the MLP is 60.7% of the q8 file, so the
exported 53.7% core would drop about 8.4 GB of 30 GB, roughly 28%, before tile overhead. **Measured:**
the q8 core tile set is 12.57 GB against 18.18 GB of MLP, saving 5.6 GB (about 19% of the file, 21.8 against
27.0 GB resident), and it decodes 20% faster. The gap to the estimate is q8's 64-neuron packing of the down
projection. A q4 core with q8 boxes reaches 16.6 GB. Each
point is one seed. The random arm also needs fewer neurons as size grows, so part of the fall is the
general redundancy of bigger models, not something the learned ranking found.

### 5.4 Deterministic tiles: exact tools in the tile grid (2026-11-15)

Neural boxes are weak at exactness (multiplication) and at facts about entities they were not fitted
on. A tile can instead be a tool. It sits on the same contract as a neural box: read the MLP input
at layer L and the token text, gate per token, and write a residual delta at L' or the next tokens.
It is then gated and served like a box (`runtime/amoeba_det_tiles.py`, adapter policy
`amoeba_det_tiles`, ordinary route only, unqualified). Both kinds are measured on CPU on
`Qwen/Qwen3-0.6B@c1899de2`; details are in pipeline.md Track G.

| Tile | What it adds | Evidence | Caveat |
|---|---|---|---|
| Calculator (probe gate + exact parser; force or constrain) | Multiplication 10 → **80** of 80 on the 27B items; 20×20 sanity multiplication 7 → **20** of 20 | 0% false fire on 10 number-heavy non-arithmetic types and on English. The parser does most of the selection on clean prompts (a random gate through it gets 42/80); the probe is the guard on non-answer positions | Parser v2 (opt-in) adds whole expressions: chained items 2 → **40** of 40 on the 0.6B, no new false fires. Mixed wordings 12 → 28 of 40; a residual push instead of forced tokens reaches only 57/80 |
| Capitals fact table (centred nearest key; closed-form value rows; null rows) | Unseen countries in a wording no row was solved on 0.65 → **0.85** (0.45 → 0.925 in the key wordings, in-sample for the values); trained facts 0.505 → 0.932 | 0 opens on unknown countries, other relations and English (Δ CE 0.000); 7.5% leak on deleted rows; shuffled values fall below the host | Trained facts in new wordings only 0.54, chat 0.23–0.33; other languages and multi-step questions never reach the key |

Our reading: tools fix what boxes cannot. They are exact where a rule exists, and a table grows one
row at a time where a box would have to be refitted. The cost is coverage: a tool fires only where
its recogniser or parser reaches. On the 27B (`Qwen3.8-27B-BF16-q8-mlx2@2dd8d304`, served on a dev port,
unqualified), the calculator takes core + boxes from 1 to **20** of 20 on 3-digit × 2-digit
multiplication and from 17 to **20** of 20 on the sanity set, with no fires on number-heavy or prose
prompts. The 27B fact table is safe (no English cost, no false opens) but narrow. In the wording its
keys were cut from, it lifts trained facts 0.47 → 0.58 and unseen countries 0.33 → 0.45. In any other
wording, chat included, it never opens. Served, it adds nothing over the host without the capitals
box (unseen 5 of 20, against 8 with the box). Keying on the entity mention is the next step.

## 6. Limits

- **Scale.** The core, box, gate and router results are on a 0.6B model. The 27B run has a core and
  a cluster catalogue so far; its boxes, gates, router and serving run are still in progress. Work on 4B-class models covers
  the exact box, the failed linear transfer, the side-box and adapter donor, residual ablation, layer
  looping and effort paths. Work on 26B and larger covers effort levels and box location only.
- **Skill type.** Boxes and adapters have been shown for single-step skills: lookups, addition,
  translation of single words. Nothing here improves reasoning.
- **Sample sizes.** Several results rest on 64 to 600 items. Intervals are in the evidence files.
- **Serving.** Nothing runs on the production serving stack. The tiled adapter, cache namespacing and
  scheduler changes are designs.
- **Distillation.** The cheapest working expert is an adapter trained on donor outputs. The plan's stop
  rules include "distillation in disguise" (K3) and "ordinary adapters do as well" (K7). Neither has been
  formally evaluated, but the adapter result points toward both.
- **Controls matter.** Three early claims were wrong because of a missing control: a box chosen on the
  test split, a broken first position, and a prompt-format effect. Some numbers in section 3.1 still
  carry the first problem and are marked. The pipeline guide lists the checks.

## 7. Open questions, in priority order

1. How much of the core's fall with size is real? (At +0.18 nats: 0.6B 73.0%, 1.7B 72.6%, 4B 64.8%,
   27B 52.7%, one seed each; the random arm falls too.) Repeat seeds, and add a dense model between
   4B and 27B.
2. Do the 27B boxes, gates and router reproduce the 0.6B results, and does a tiled 27B core serve
   faster or smaller on the real stack?
3. Can a host-side gate or router reach unseen phrasings? Recall is 66–80% today, and routing errors
   between skills were the main cost of loading several adapters.
4. Does a stronger donor add anything over training on correct answers, for a skill whose answers cannot
   be generated? For addition, sorting and unit conversion it did not.
5. Why did 3×2-digit multiplication not move at any adapter rank, when the donor scores 85%?
6. Is there any skill a stronger donor has that a well-prompted small host lacks? If not, experts are
   about reliability, not capability.
7. Can the serving stack run cores and boxes as tiles with real speed and byte savings?
8. Does anything short of full fine-tuning move reasoning?

## 8. Where things live

| What | Branch | Path |
|---|---|---|
| Research plan and results, sections 8b–8h | `claude/amoeba-plan-8f9e33` | `docs/research/2026-11-11-amoeba-rnn-plan.md` |
| Confirmation checks, section 8i | `claude/amoeba-confirm-checks-1ec328` | the same file |
| Core and box evidence | both of the above | `docs/research/amoeba-cone-width-2026-11-12/` |
| Tile format and serving design | `claude/amoeba-plan-8f9e33` | `docs/research/2026-11-12-amoeba-tiled-execution-design.md` |
| Cross-model obstacles and framework | `claude/amoeba-plan-8f9e33` | `docs/research/2026-11-12-amoeba-ingest-framework-research.md` |
| Box registry and neuron atlas | `claude/amoeba-plan-8f9e33` | `docs/research/amoeba-box-registry/`, `docs/research/amoeba-neuron-atlas/` |
| First cross-model experiments | `claude/amoeba-m0` | `research/amoeba/`, `docs/research/2026-11-1*-amoeba-*-results.md` |
| Donor box location in large models | `claude/quirky-poitras-2cdefc` | `docs/research/amoeba-cone-width-2026-11-12/transplant-boxes-decimal17-2026-11-13.md`; raw evidence is outside git |
| Side box, gated adapter, skill screen | in git (2026-11-18) | results in `docs/research/amoeba-sidecar-m3-20261113/`; code: that folder's `amoeba-lab-20261118.bundle` (the M3 `~/amoeba-lab` history to 3add7f4; `git clone` it) |
| Effort lanes | `claude/amoeba-lane-moves-3fcf76`, `-spirals-be1149`, `-latent-80a8f5` | their copies of the plan, section 8h |
| 27B core, atlas, clusters, boxes | `claude/amoeba-27b-80bc6a` | `docs/research/amoeba-27b/` (RUN_NOTE.md, DESIGN.md, evidence directories) |

All of these branches were merged to `main` on 2026-11-14 (`de51a6d6a`). The 27B branch is still
live and will need another merge when its run ends.
