# Cross-model remapping: what we found and what it means (2026-11-14)

The question: one model (the **donor**) has an expert box we want, and another model (the **host**) should use it. The two models don't share coordinates. Their layers are numbered differently, their hidden states have different widths, and the axes don't line up. Can we warp the donor's internal space into the host's, the way you would stretch and rotate one map to lay it over another?

This note covers three experiments:

1. The hub and deformation pilot (phase 0), from the "Cross-architecture neural map extraction" session.
2. The earlier linear-map box copy.
3. The transplant-box sizes.

## 1. The hub pilot (phase 0)

**Setup.**
- Host: `Qwen3.5-4B-MLX-4bit@82134c29`.
- Donors: `Qwen3.8-27B-oQ4e-mtp@fb827b65` (same family) and `gemma-4-26B-A4B-it-MLX-8bit@0364a378` (different family and tokenizer). A Flash-Next run also exists, but its numbers are invalid, see below.
- Text: the donor's own worked solutions to one problem, the 16-digit period of 1/17. Four traces, about 3,000–7,000 paired token positions per donor.
- The host reads the same text, and tokens are paired where both tokenizers end at the same character.
- mlx2 `0b3f5a812`. Evidence: `~/amoeba-runs/hub-0b3f5a812/`. Plan: `docs/research/amoeba-cone-width-2026-11-12/hub-deformation-plan-2026-11-14.md`.

**Scoring.**
- Fit a map on three traces and test it on the held-out fourth, rotating through all four.
- On the held-out trace, replace the host's hidden state at one layer with the mapped donor state. Measure how far the host's next-token predictions move: KL against the unpatched host, and top-1 agreement.

**Controls.**
- The same map with shuffled pairs.
- Replacing the host state with its average.
- A ceiling: the host's own state squeezed through the same 256 dimensions.

### Results (means over the four held-out traces)

| Host layer ← donor layer | Similarity of the two spaces (CKA) | KL after patching: ridge map | Rotation only (Procrustes) | Shuffled control | Host's own 256-dim ceiling | Top-1 agreement (ridge) |
|---|---|---|---|---|---|---|
| **Qwen 27B donor** | | | | | | |
| L3 ← D3 | 0.39 | 0.058 | 0.153 | 4.89 | 0.064 | 97.8% |
| L8 ← D15 | 0.57 | 0.061 | 0.172 | 5.03 | 0.040 | 97.3% |
| L16 ← D35 | 0.72 | 0.093 | 0.118 | 4.84 | 0.039 | 96.2% |
| L24 ← D51 | 0.68 | 0.082 | 0.136 | 4.37 | 0.033 | 96.3% |
| **Gemma-4 26B donor** | | | | | | |
| L3 ← D2 | 0.71 | 0.034 | 0.064 | 6.74 | 0.059 | 98.9% |
| L8 ← D7 | 0.58 | 0.053 | 0.122 | 6.15 | 0.013 | 98.5% |
| L16 ← D27 | 0.66 | 0.109 | 0.185 | 5.62 | 0.019 | 96.5% |
| L24 ← D27 | 0.64 | 0.093 | 0.118 | 5.29 | 0.021 | 97.9% |

What this shows:

1. **Layers line up by relative depth, and the alignment can be found automatically.** A monotone warp fitted on layer similarity pairs host layer 8 of 32 with donor layer 15 of 64, and host 16 with donor 35. Middle layers are the most alike (CKA 0.66–0.72). For the 27B the early layers are the least alike (0.39). Gemma's last captured layer (27) serves the whole upper half of the host, because the donor was captured only at some layers.
2. **A simple stretch-and-rotate map (a ridge regression) is enough on this data.** The host keeps working with a donor's state in place of its own: KL 0.03–0.11 and 96–99% top-1 agreement. Shuffled pairs and the host average both wreck it, at KL 4–7. The mapped donor state comes within a few hundredths of a nat of what the host's own 256-dimensional summary of itself gives, so part of the remaining loss is the compression, not the map.
3. **Rotation alone is not enough.** Procrustes, which only rotates, is consistently worse: KL 0.06–0.19. The two spaces differ by stretching and scale as well as orientation.
4. **Shape alone does not align the models.** Two methods use only each model's internal geometry, with no paired text:
   - **Gromov-Wasserstein** matches points by the pattern of distances within each model. It recovered almost nothing: top-1 about 0.005, against 0.0017 for chance.
   - **Relative representations** describe each point by its similarity to shared anchors. They reached top-1 0.06–0.35.

   The geometry is not distinctive enough to find the correspondence by itself. **Alignment needs paired data: the same text read by both models.**
5. **A different family works as well as the same family.** Gemma-4, with a different architecture and tokenizer, aligns as well as the Qwen 27B. Character-level token pairing handles the tokenizer difference.
6. **The Flash-Next numbers in this run are invalid.** The ridge penalty was absolute and swamped Flash-Next's small-scale hidden states, which is why its ridge KL reads 0.6–4.9.

   **Fixed rerun (2026-11-14):** `~/amoeba-runs/hub-1f6f62bc2-r3/fn`, mlx2 `1f6f62bc2` (tree clean, schema v1: relative ridge penalty, chunked prefill, 24 GB memory cap). CPU, 63 minutes, under a memory watchdog that never tripped. Donor `Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP@25c2da42836b90dd…`. Means over four held-out traces:

   | Host layer ← donor layer | CKA | Stitch KL, ridge | Rotation only | Shuffled | Host 256-dim ceiling | Top-1 agreement (ridge) |
   |---|---|---|---|---|---|---|
   | L3 ← D3 | 0.63 | 0.086 | 0.140 | 6.51 | 0.037 | 96.2% |
   | L8 ← D15 | 0.69 | 0.192 | 0.228 | 6.34 | 0.036 | 93.1% |
   | L16 ← D23 | 0.70 | 0.315 | 0.355 | 6.29 | 0.133 | 91.0% |
   | L24 ← D43 | 0.64 | 0.593 | 0.733 | 5.01 | 0.139 | 84.7% |

   Flash-Next, a sparse MoE model, aligns far better than chance at every depth, so the earlier failure was the fitting bug. But it aligns **less well than the two other donors**, and it gets worse with depth: KL 0.09 → 0.59 and top-1 agreement 96% → 85%, where the 27B and Gemma held 0.03–0.11 and 96–99%. Part of the gap is compression: the host's own 256-dimension ceiling is also higher at depth (0.13–0.14). A Flash-Next transfer is therefore the harder case, best placed at early to mid layers.

### What phase 0 cannot say

**Overlaying the maps is not the same as transferring knowledge.** The host was reading the donor's own correct solution, so after patching it is copying, not computing. A map that keeps the host's predictions intact on text it is reading says nothing yet about whether a donor's box can teach the host something it doesn't know. All of this is also one problem and four traces, so the maps are specific to long-division text.

## 2. The earlier warning: maps that memorise

On Qwen3-0.6B → Qwen3-4B, a box copied through linear maps scored 98% on the facts the map was fitted on and **0% on held-out facts**. A plain adapter and a random box also scored 0%.

The map had learned the trained facts, not a general translation. Any remap fitted on narrow data will do the same. Phase 0's good numbers are on held-out traces of the *same* problem, which is the easy case.

## 3. Donor boxes for a procedural skill are big

The transplant window measured how much of each donor is needed to keep solving 1/17. It counted the smallest box that still gives the right answer while a random box of the same shape fails:

| Donor | Box share of the donor | Parameters |
|---|---|---|
| Gemma-4 26B | 21.5% of experts | 4.9 B |
| Flash-Next | 28.3% | 34.2 B |
| Kimi Linear 48B | 33.2% of parameters | 15.7 B |
| Qwen3.8-27B (dense) | 44.8% of MLP neurons | 7.7 B |

Every random twin failed, so these boxes are specific. But they are **large**, spread through all depths, and only one problem was tested. By contrast, topic knowledge on the 27B fits in tiny boxes: arithmetic in 685 neurons, French in 1,289. Reasoning-style skills don't localise the way topics do.

## What it means for us

1. **Grafting a donor box is a coordinates problem we can solve, layer band by layer band.** The warp says where in the host a donor layer should write, and a ridge map per band translates it. Both come from text that both models read.
2. **The topic probe corpus is exactly the paired data we need.** For each topic in the cross-model map, the same probe texts give:
   - the host's stub (its own neurons for the topic);
   - the donor's box;
   - the warp and map between them.

   The topic ID is the join key; the warp and map are per pair of models and per layer band. This fits the stub-and-remap plan.
3. **Fit maps on broad data, never on the skill alone.** Otherwise we repeat the 0.6B → 4B result. The planned phase 2 does this: a shared dictionary across both models, fitted on a large corpus. A feature that decodes in both models is transferable; a feature found in only one model has nothing to map onto.
4. **Use topology as a check, not as the aligner.** Shape-only matching failed here. A persistent-homology check could still confirm that a mapped region keeps its structure, for example the same number of loops and clusters. That needs `ripser` or `gudhi` installed, which is a download and needs your OK.
5. **The decisive test hasn't run.** Phase 1 would inject the mapped donor box's output into the host *while the host generates*, on held-out problems. It would score accuracy against:
   - the host alone;
   - a shuffled map;
   - a map trained on a random donor, which controls for the map simply acting as an adapter.

   Until phase 1 runs, we have no transfer result. Fitting a ridge map is a regression with the models' weights frozen. A trained hub (phase 2) is closer to training, so it waits until after today's mapping-only rule.
6. **Where internal grafting pays off.** We already have a proven route: the gated side stream, which runs the donor box separately and mixes its output into the host's at the output. Grafting inside the host saves running the donor at all, but only when the donor box is small:
   - **Topic boxes** (hundreds to a few thousand neurons): promising.
   - **Procedural skills**, whose boxes are 20–45% of the donor: no saving yet. The side stream, or a gated adapter, remains the better tool for those.

## 4. Making it operational: the transfer pipeline (status 2026-11-14)

A donor box moves into a host in five steps. Each step has an owner and a state:

| Step | What | State |
|---|---|---|
| 1. Pick the topic | The **donor-gap screen** scores each topic's held probe texts on both models, as per-byte NLL, bare and with a neutral prefix. A topic is a candidate only if the donor leads in both conditions with 95% confidence. | Done. Gaps are centred on the whole-corpus gap per condition, because the neutral prefix costs the 27B more than FN. FN is better on 16 topics, mostly security, fish, maths word problems and low-resource languages. The 27B is better on 35. See [topic-map.md §6](topic-map.md#6-donor-screen-fn-against-27b). |
| 2. Find the host's stub | The **topic registry** (`mlx2.amoeba-topic-map.v1`) gives the host's neurons for that topic, per layer, with strength and status. The **stub recipe** cuts them out, copying neurons shared with other topics rather than moving them. | Registry built for the 27B: 42 of 64 clusters labelled, 9.8% of neurons unexplained. Stub recipe done and unit-tested on Qwen3-0.6B. |
| 3. Find the donor's box | Carve the donor's box for the same topic on the same probe texts: units must beat a random twin and the box must be complete. | Shared carving driver `research.amoeba.carve` is on main and loads MoE donors. Not yet run per topic. |
| 4. Translate coordinates | The **warp** (which donor layer feeds which host layer) and **ridge maps** per layer band, fitted on text both models read. Fit them on broad data, never on the skill alone (section 2). | Method proven in phase 0 on three donors. Fitted per problem today, not yet per topic. |
| 5. Inject and gate | Add the mapped box output to the host's hidden state where the box's gate opens, while the host generates. | **Phase-1 driver built**: `scripts/amoeba_hub_transfer.py`, branch `claude/quirky-poitras-2cdefc`. The size-matched controls were reviewed and fixed; CPU smoke next. |

**The first real transfer test (approved by Pierre, 2026-11-14):** one GPU slot of at most 2 hours, taken between the other sessions' windows.
- **Donor:** Gemma-4-26B, whose 826-expert decimal box is the most compact we have.
- **Host:** Qwen3.5-4B.
- **Items:** about 50 decimal-period problems, split train and test.
- **Method:** maps fitted on the train items, then the box injected while the host generates the test items.
- **Controls:** the host alone; a shuffled map; a random box of the same per-layer counts, rescaled to the real box's write size so the test is about direction, not size; and the donor itself as the ceiling.

It answers the question phase 0 couldn't: does a mapped donor box make the host *solve* problems it otherwise fails?

What would follow a pass: repeat it on a **topic** box (the case that matters for cross-model cores), picking a topic the donor-gap screen names, with the 27B as host.

## 5. Transfer results so far (2026-11-15)

**Short answer: no cross-model transfer has worked yet.** Every injection arm so far either hurts the host or does no better than its size-matched random control. The coordinates translate (section 1), but injecting a mapped donor box into a generating host has not made it solve anything it otherwise fails.

### Decimal problems: Gemma-4-26B box into Qwen3.5-4B
Setup: 21 held-out denominators, greedy decoding, and an exact oracle. Drivers on branch `claude/quirky-poitras-2cdefc`. Evidence under `~/amoeba-runs/hub-xfer-*`.

| Run | Arm | Correct |
|---|---|---|
| Phase 1, single point at full strength (alpha 1), 9bab05788 | host | 5/9 |
| | box @16 / @24 | 0/9, 0/8 |
| | shuffled map @16 / @24 | 0/8, 0/8 |
| | random box @16 / @24 | 0/8, 1/8 |
| Span over layers 8–28 (alpha 0.05 / 0.1 / 0.25), 409261e02 | host | 5/12 |
| | box | 8 / 6 / 4 of 12 |
| | random box | 7 / 8 / 4 of 12 |
| | shuffled map | 4 / 6 / 1 of 12 |

- At full strength any injection, real or control, pushes the host off its pattern-spotting strategy into long manual division, which runs out of tokens.
- Gentle span injection doesn't derail, but the real box doesn't beat the random box: 8/12 against 7/12 at 0.05, with 12 items. Stronger injection hurts every arm.
- The sink, span-plus-sink and quiet-channel slots (C, D) were built but not run, because B showed no strength at which the real box separates from its controls.

### Topic case: CWE, Flash-Next into Qwen3.8-27B (teacher-forced held-text NLL)
- **The donor box was not topic-specific.** At 0.9 contribution mass, FN's "CWE box" is general machinery. Switching it off costs CWE +0.54 nats/byte, but also threat actors +0.36, CVE +0.57 and ordinary text +0.22. Its twin costs ≤0.04 everywhere. A contrastive box is built (`claude/cwe-contrastive-box`: CWE contribution minus off-topic, coverage ≤ 0.5) and queued.
- **The first distillation recipe was broken.** It made held CWE worse in every trained arm, including self-distillation, which should be neutral. There were four causes: a renormalised top-20 loss that rewards sharpening, a context-free training window, step sizes far too large, and no negative data. Fixed in `claude/cwe-distill-v1` (top-20-plus-tail KL; self-gradient exactly 0 at init; prefix context; retain loss and gate L1; best-epoch selection) and queued.
- The mapped arms on the 27B (W2a) were killed once by an over-tight watchdog and are rerunning.

### The quiet-subspace idea
A partial CPU check, on 2 of 6 texts, found that the host's lowest-variance directions are **no quieter than a random subspace** of the same size: KL about 0.03 at 0.4× host RMS on layers 8 and 16. Only the top directions are louder. If the full rerun agrees, there is no special "dark" channel, and a sink at the output is the better bet.

### What this means
- The hub maps the coordinates, but a mapped write does not yet carry *usable* knowledge into a host that doesn't already know it. This is consistent with the "no downstream reader" concern.
- The two remaining live tests are both on CWE: a topic-specific donor box (W1c/W2e) and the fixed distillation (W2c). If distillation into a gated tile works and mapping does not, kill criteria K3/K7 apply: the cheapest working transfer is a distilled tile, not mapped neurons.

## Open decisions (from the hub plan)

1. ~~Phase 1 GPU windows~~: one donor and a slot of at most 2 hours, approved by Pierre on 2026-11-14. Done; result in section 5.
2. Installing `ripser` or `gudhi` for the topology check: still open.
3. Phase 2 storage: still open. Recommended: a layer subset (about 17 GB per donor), captured only if phase 1 shows transfer.
