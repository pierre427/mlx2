# Amoeba: how one model swallows another's skills

*Draft, 2026-11-15. Plain-English overview of the whole project: the goal, the method, what works today, and what does not yet. Each claim is marked **measured**, **built** or **design**. Evidence lives in `overview.md`, `pipeline.md`, `cross-model-remapping.md` and `topic-map.md`, in this folder.*

## 1. The idea

An amoeba eats by wrapping itself around food, holding it in a pocket (a food vacuole), breaking it down, and absorbing what is useful into its own body. It doesn't need the food to be shaped like itself.

We want a language model to do the same with other models. A **host** model should be able to take in a skill or a body of knowledge that a **donor** model has, and make it part of itself. That should work even when the two are built differently: a different family, a different tokenizer, dense or mixture-of-experts, a different attention design. The host stays one model. It just has more inside it.

Today a skill is locked inside the model that learned it. If this works, skills become parts that can move between models.

## 2. The host's anatomy

Before a model can swallow anything, it has to know its own body. So the first job is to carve the host into parts.

| Amoeba | Model | What it is | Status |
|---|---|---|---|
| **Nucleus** | **Core** | The always-on part that keeps ordinary language working. It's carved from the host's own MLP neurons on plain English prose, at graded levels: T1 grade school, T2 high school. | **measured**: on Qwen3.8-27B, T1 keeps 61% of MLP neurons, T2 76% |
| **Organelles** | **Tiles (boxes)** | Groups of the host's neurons that each carry one topic or skill: arithmetic, French, capitals, CWE, and about 80 more. Each opens only when needed. | **measured**: a 685-neuron box restores 3-digit addition from 0.32 to 1.00 |
| **Cytoplasm** | **Residual stream** | The shared hidden state that every layer reads from and writes to. Everything flows through it. | – |
| **Membrane** | **Router and gates** | Decide, per request or per token, which tiles open. | **built**: router v1 sends arithmetic, capitals and translation to the right tiles |
| **Whole body** | **Partition** | Every neuron lives in exactly one place: the core, one tile, or a shared tile. With every tile open, the parts reproduce the original model exactly. | **measured**: exact reproduction, checked on every export |

Two rules keep the body healthy:
- **Nothing is thrown away.** Pruning is withdrawn. A neuron the core doesn't need goes into a tile, not the bin.
- **Full model at 100% means core plus tiles at 100%.** Anything less is a routing or tile bug, not a loss to accept.

The carved body also runs **faster** than the original, once the tiles are fused into one grouped operation per layer. On the 27B (unsettled numbers):
- fused tiles decode 17–21% faster than the full model;
- with AST tree speculation, about 101 tok/s total against 92;
- 10 GB less memory.

## 3. Eating: six stages

### Stage 1. Sense the food: is there anything worth eating?
Before taking anything, check whether the donor actually knows something the host doesn't.
- **How:** score both models on the same probe texts for each topic, as bytes-per-token loss. Correct for each model's overall offset, then keep only topics where the donor clearly wins. Score a served model through prompt logprobs, so it doesn't need pausing.
- **Pitfall:** check prompt format first. Many apparent gaps vanish with a newline at position 0 or a two-shot prompt.
- **Status: measured.** Flash-Next beats the 27B on 16 topics (security, CTI, CWE, some low-resource languages); the 27B wins on 35. A topic taxonomy of 8,000+ IDs is the shared vocabulary across models.

### Stage 2. Engulf: wrap the useful part
Cut out the donor's neurons for that topic: its **donor box**.
- **How:** rank the donor's units by how much more they work on the topic than on other text (contrastive carving), and check against a random twin of the same shape.
- **Pitfall:** carving by raw contribution grabs general machinery, not the topic. A first CWE box built that way held 90% of the contribution mass and hurt everything when removed.
- **Status: measured.** The contrastive CWE box from Flash-Next is genuinely topic-specific. Removing it costs CWE five times more than removing its twin, and barely touches unrelated text.

### Stage 3. Hold it in a vacuole: keep it intact and use it as is
The safest first use is to leave the donor box **untouched, with its own attention**, running beside the host on the same text. Its predictions blend into the host's output through a gated "hitch" at the end.
- **Why it ignores architecture:** the two models only meet at the output, as text and token probabilities. Different tokenizers are handled by matching strings.
- **Cost:** the donor box stays resident and runs every step. When the host itself can decide when to open it, the box can sit idle the rest of the time.
- **Status: measured at small scale** (Qwen3-4B into Qwen3-0.6B: addition 29% → 97%, tokenizer mismatch no problem). The head-to-head against remapping is **built and queued**: decimal problems Gemma-26B → Qwen3.5-4B, and CWE Flash-Next → 27B.

### Stage 4. Digest: break it down into the host's own material
The vacuole works but keeps the donor around. To truly absorb the skill, it has to be expressed in the host's own terms. There are two ways.
- **Translate it (remapping).** Learn a map from the donor's hidden state to the host's: a depth warp plus per-layer ridge maps, fitted on text both models read. Then inject the donor box's output into the host while it runs.
  - **Status:** the maps are **measured** and work across families: Qwen, Gemma and Flash-Next all translate into the host's coordinates. The injection itself **has not worked yet**. At full strength it was 26–33× the host's own per-layer change and derailed the host. At gentle strength it did no better than a random box.
- **Rebuild it (distillation into a tile).** Grow a small new tile of fresh neurons inside the host, gated, and train only that tile to reproduce what the donor does on the topic. Every other host weight stays frozen.
  - **Status: built, being measured.** The first recipe was broken: it made the host overconfident everywhere. The fixed recipe, with a proper loss, real context, a drift penalty on other text and validation-based stopping, is queued for CWE.

**The blend-in rule** (Pierre): new material must not overpower the host's existing networks.
- Every injected write or new neuron is capped relative to the host's own activity at that layer and token: 0.25× its own change. A drift penalty keeps everything else the same.
- The aim is to squeeze the new skill in quietly, not to graft on a loud new organ.

### Stage 5. Absorb and integrate: make it part of the body
- Register the new tile in the host's tile grid under the same topic ID, with its source, revision and licence.
- If the host already had a weak version (a **stub**), cut the stub out and put the new tile in its place, copying shared neurons rather than moving them.
- Let the router and gates open it like any native tile.
- **Status: design**, with parts built: the stub recipe is tested on Qwen3-0.6B, the registry and router exist, and tiles serve fused and fast.

### Stage 6. Excrete: keep only what helps
Every absorbed part is tested against the full set of controls:
- a random box of the same shape;
- a shuffled map;
- English cost;
- off-topic interference;
- near-tie rules.

Anything that does not beat its controls is dropped. The donor itself can be dropped once the skill is digested.

## 4. Symbionts: parts no model has to learn

Some jobs are better done by exact machinery than by neurons. These plug into the same tile grid, with the same gate, read and write contract:
- **Calculator tile:** a hidden-state probe spots "a number result comes next" and the exact answer is written. It targets multiplication, where the carved model is weak (7/20).
- **Fact tables (PLE-style):** recogniser neurons form a key, and the stored row carries the answer. Facts become exact, patchable and sourced.
- **Later:** databases, retrieval, steering directions for behaviour.

**Status: being built** in its own session.

## 5. Where we are, honestly

| Stage | State |
|---|---|
| Know your own body (core, tiles, partition, fast serving) | **Works** on Qwen3.8-27B |
| Sense the food (donor-gap screen) | **Works** |
| Engulf (contrastive donor box) | **Works** |
| Hold in a vacuole (side stream) | **Works small**; the big-model bench is queued |
| Digest by translation (remapping) | Maps work; **injection does not yet** |
| Digest by rebuilding (distilled tile) | Fixed recipe **queued** |
| Absorb (stub swap, router) | Parts built |
| Symbionts | Being built |

**The big open question:** can the host use knowledge it never had? Its later layers may have no reader for a concept it never learned. That is why the vacuole (stage 3) and the distilled tile (stage 4b) are the most promising routes, and why a sink at the output, which turns a donor's knowledge directly into token predictions, is still on the table.

A planned spot-check on GSM8K problem 549 was dropped. The 27B only "failed" it because the old benchmark prompt format broke chat models. With chat few-shot prompts, now the harness default, it solves 549 and scores 56/60 on that stretch of the test set. A donor-transfer spot-check needs a real gap, so it will use a topic from the donor screen instead.

## 6. What it would mean

If every trained model can be carved like this:
- **Cores become portable bases.** Pick a strong English core and add the tiles you need.
- **Skills become parts.** A topic that one model knows well can be moved to another, tested, and swapped out when a better donor appears.
- **Knowledge becomes patchable.** Facts in tables, behaviours in directions, skills in tiles, each updated on its own.
- **Serving gets cheaper.** Only the tiles a request needs run, and the carved model is already faster than the original.

The goal stays the amoeba's: one body that keeps growing by taking in what others have learned, without caring what shape it came in.
