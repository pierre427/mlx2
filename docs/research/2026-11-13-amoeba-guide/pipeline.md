# Amoeba pipeline: every step, in order

Date: 2026-11-13. Read [overview.md](overview.md) first for the idea, the vocabulary and the results.

This guide lists the stages needed to reproduce the work, in the order you would run them. Each stage
gives what it does, the script, the check that says it worked, and the pitfalls already hit.

Each stage carries one status, sometimes with a note in brackets:
- **measured**: run, and the result is recorded with whatever control was run;
- **built**: the code exists, with no measured result;
- **design**: written down only.

Scripts are in `scripts/` on branch `claude/amoeba-plan-8f9e33` unless a stage says otherwise. Items
marked *(confirm branch)* exist only on `claude/amoeba-confirm-checks-1ec328`.

There are four tracks:

| Track | Goal | Stages |
|---|---|---|
| A | Split one model into a core and boxes | A1–A13 |
| B | Attach an expert taken from another model | B1–B8 |
| C | Map what a model's parts do | C1–C4 |
| D | Control thinking effort | D1–D3 |

## 0. Before any run

These rules come from failures, including one kernel panic from memory starvation.

**Memory**
- Run `scripts/amoeba_memcheck.py` *(confirm branch)* before every model load. Its strict rule: no other
  process above 8 GB, and swap under 8 GB.
- The script takes `--mem-policy`:
  - `strict` is the default and is required for any load that peaks above 10 GB;
  - `small` is for 0.6B CPU probes only, one at a time. It records but waives the two strict rules, needs
    30 GB available, and aborts the run above 12 GB;
  - `waived:<avail>:<cap>:<floor>` needs the lab owner's explicit approval. It was used once, for a 4B run.
- Measure the physical footprint (`top -l 1 -pid <pid> -stats mem`), not only RSS. One process sat at
  about 3 GB RSS with a 109 GB footprint.
- In MLX code, set a cache limit and a memory limit, and clear the cache every step.
- Stream activation captures to disk. A 1.28M-token capture held in memory reached 36 GB.

**GPU and the production service**
- Wait for `/tmp/gpu.lock/owner.json` to clear. Never preempt another job.
- A GPU window is a chain of four wrappers, in this order:
  1. `run_with_service_paused.py --label com.example.mlx2-flash-next`
  2. `cpg_job.py run --lease --lock --lock-dir /tmp/<own>.lock --require-radio --heartbeat-s 60`
  3. `run_with_gpu_locks.py --cpg-owner-lock <same> --cpg-task <lease>`
  4. the job itself, or `amoeba_gpu_queue.py`
- Each wrapper needs more arguments than shown (run directory, label, output log, receipt path), and
  `cpg_job.py` lives in a separate checkout. Copy the working example *(confirm branch)*:
  `docs/research/amoeba-cone-width-2026-11-12/qwen3-0.6b-ab-sanity-sink-01a3cd6b6/receipts/launch.sh`.
- Pausing the production service on port 8282 needs the lab owner's go-ahead for each window.
- Development servers use ports 8391–8395, never 8282.
- Use a fresh output directory for every run. The scripts refuse a non-empty one.

**Measurement discipline**
- Choose every setting on a fit or validation split. Score the test split once.
- Put a control beside every number: a random box or core of the same shape, a shuffled-label gate, or
  the wrong donor.
- Name the checkpoint as `<artifact>@<fingerprint>` and the repository commit for every result.
- Validate the saved file, not the tensors in memory.
- Never edit donor weights in place inside a probe.
- Before calling a skill "missing", test the host with a newline at position 0 and with a two-shot prompt.
- Do not build test inputs from repeated strings.

**Devices**
- Most probes take `--device mps|cpu`. The core, skill-box, core-fraction, pruning and restate probes
  are CPU only.
- With `--device mps`, set `PYTORCH_ENABLE_MPS_FALLBACK=1`, and set the default device only after
  `from_pretrained`.
- In a worktree, run with `PYTHONPATH="$PWD/src"` so tests do not import the main checkout.

## Track A. Split one model into a core and boxes

Reference numbers are for `Qwen/Qwen3-0.6B@c1899de2`.

### A1. Build the corpus — built
- **What:** broad English text for learning the core: stories, repository prose and donor-written
  encyclopedic text.
- **Script:** `amoeba_build_corpus.py` *(confirm branch)*. It writes `corpus.json` under the lab's
  research-artifacts folder (`amoeba/corpus-2026-11-13/`).
- **Pitfall:** the first corpus was lost. Numbers from the rebuilt corpus are close to the earlier ones
  but not identical.
- **Pitfall (found at 27B, 2026-11-14):** carving the core on repository markdown keeps code, shell, git
  and inference-stack knowledge in the core *by construction*, so those topics never become box
  candidates. Carve on prose only. **Corpus v2** (topic session,
  `docs/research/amoeba-topics/core-corpus-v2-manifest.json`) has 3,786 texts in 24 level × register cells,
  graded grade school / high school / post-graduate; 585 are Artemis-31B-generated creative texts
  (source `artemis`, filterable). It is used for the nested T1/T2 cores in
  `docs/research/amoeba-27b/GRADED_CORES_PLAN.md`.
- **Driver:** new carves use the shared driver `research.amoeba.carve` on main. It loads dense and MoE
  donors and takes `--fixed` for nested tiers.

### A2. Learn the core mask — measured
- **What:** one yes/no mask over all MLP neurons, shared by every position. Train it so the masked model
  matches the full model on English (KL loss) while keeping as few neurons as possible. A ladder of
  penalty strengths gives a ladder of core sizes.
- **Script:** `probe_amoeba_english_core_cpu.py` with `--corpus`, `--lambdas`, `--steps 150`, `--lr 0.1`,
  `--ablation mean|zero` (default `mean`).
- **Output:** `masks_core<lam>.pt`, `english_core.json`.
- **Check:** extra English loss on held-out text, against a random core with the same number of neurons
  per layer. Reference: +0.18 nats at 68.5% kept, where a random core costs +1.16.
- **Pitfalls:**
  - Learn the mask jointly. Ranking neurons one at a time does not work.
  - Soft masks leak. Use hard masks with a straight-through gradient.
  - `mean` replaces a dropped neuron by its average, which is for analysis. A checkpoint you can save
    needs `zero`, which costs about six more points of kept neurons.

### A3. Choose the core size — measured
- **What:** cores nest, so train one ranking and cut it at any size.
- **Script:** `probe_amoeba_core_fraction_cpu.py` *(confirm branch)*; `--ranking` reuses a saved ranking.
- **Check:** bisect to a target English cost, and bisect a random ranking to the same cost.
- **Reference:** +0.18 nats at 73.0% kept (random ranking 93.9%; random core of the same shape +0.94);
  +0.05 at 85.6% (random ranking 99.0%).
- **Pitfall:** the size depends on the recipe. A2's ladder gives 68.5% at the same cost.
- **Size reference** (mean ablation, kept at +0.18 nats): 0.6B 73.0%, 1.7B 72.6%, 4B 64.8%, 27B 52.7%.
  Random rankings: 93.9%, 88.5%, 83.4%, 82.2%.

#### Running A2–A7 on a 27B model (`scripts/amoeba27b_*.py`, branch `claude/amoeba-27b-80bc6a`)
- **Use MLX, not torch.** A torch fp32 27B is about 108 GB. Learn on a q8 build checked against BF16
  (`amoeba27b_core.py` q8check: −0.00002 nats, 97.6% top-1 agreement).
- **Backprop one layer at a time** (`layerwise_value_and_grad`). The GDN recurrence keeps a state per
  token per layer: about 15–21 GB per 64-token sequence over 64 layers. `mx.checkpoint` saved nothing
  and doubled the time. Layer-wise backprop gives identical gradients and peaks at 29–33 GB at batch 6
  × 64 tokens, 3.6–11 s per step.
- **Guards:** a footprint watchdog plus a peak cap that stops training cleanly. Before taking the GPU,
  the window gate also waits until no process other than :8282 is over 8 GB. The first 27B core
  window was stopped by the watchdog at 110 GB, and it was right to stop it.
- **Twin control for clusters:** add each unnamed cluster to the core, and remove it from the full model,
  beside a random twin with the same per-layer counts (`amoeba27b_boxes.py` phase `clusters`).

### A4. Protect the first position — measured
- **What:** in Qwen3-0.6B, position 0 must see the full MLP in layers 0–2.
- **Scripts:**
  - `probe_amoeba_restate_cpu.py --part sink --sink-ks --sink-tol 0.01` *(confirm branch)*. On the
    exported core it gives +0.611 with no protection and +0.209 with the rule.
  - `probe_amoeba_tile_quant_cpu.py --sink-depths` scans the depth with boxes loaded: +0.441 at depth 0,
    +0.167 at depth 3.
  - `export_amoeba_tiled.py --sink-layers 3` applies the rule in tiles.
- **For a dense checkpoint,** which cannot treat one position differently, use one of two stand-ins:
  keep layers 0–2 whole (+0.226), or add back one chosen neuron in each of layers 0–2 (+0.243).
- **Check:** English cost at each depth. Test each model family: SmolLM2-360M and Qwen3.8-27B show no
  such effect. At 27B, use `--stand-in none`, because the lowest-cost stand-in wins only because it adds neurons.
- **Pitfall:** a check that looks only at the topic score will miss this. Always check English cost.

### A5. Learn a box — measured (sizes chosen on the reporting split; see pitfall)
- **What:** freeze the core. Learn a hard mask over the non-core neurons that restores one skill.
- **Script:** `probe_amoeba_skill_boxes_cpu.py` with `--core-masks`, `--skills`, `--box-lambdas`,
  `--box-steps 150`.
- **Output:** `skill_boxes.pt`, `skill_boxes.json`.
- **Check:** held-out exact match, English cost, and a random box of the same size.
- **Reference:** capitals 0.53, arithmetic 0.43, French 0.48 when the box size is chosen on the fit
  split. The full model scores 0.69 / 0.38 / 0.70 and the core alone 0.09 / 0.00 / 0.00.
- **Pitfall:** the run that produced the boxes used in A6–A8 chose sizes on the reporting split, which
  gave 0.62 / 0.49 / 0.62. That is optimistic and has not been rerun. The same mistake produced the
  withdrawn "box beats the full model" claim. Choose on the fit split.

### A6. Add a gate — measured
- **What:** a per-token switch for each box, computed from the hidden state. Train it on a mix of every
  skill, multi-turn recall and English, with a small penalty for being open on English.
- **Script:** the same, with `--gate-steps 200`, `--gate-penalty 0.05`, `--gate-lr 0.01`.
- **Check:** all boxes loaded together, gated against ungated, on the same boxes. Reference, as
  capitals / arithmetic / French: 0.66 / 0.44 / 0.56 gated against 0.53 / 0.22 / 0.46 ungated.
- **Pitfall:** an always-on box damages other behaviour. In the served check it cut multi-turn recall
  from 0.94 to 0.69.

### A7. Train the router — measured (layer chosen on the reporting set)
- **What:** a linear classifier on the core's mid-layer state that names the box a request needs, or
  none.
- **Script:** `probe_amoeba_router_cpu.py --layers`.
- **Check:** accuracy on phrasings never seen in training and on hard negatives, and end-to-end results
  against an oracle.
- **Reference:** on held-out requests in trained phrasings, routing is perfect and matches the oracle.
  On unseen phrasings layer 8 scores 0.868, and sends 14 of 32 arithmetic prompts to "none".
- **Pitfall:** the script picks the layer on the unseen-phrasing set and reports on it. Hold out a
  separate set for the choice.

### A8. Prune the box — **withdrawn as a method (Pierre, 2026-11-14)**
> **Partition rule:** no neuron is discarded. Every carve ends with the core, the named boxes and remainder
> boxes holding every neuron exactly once. The unsupervised clusters become the remainder boxes once the
> named boxes have taken their neurons. With every gate open, the parts must reproduce the full model
> exactly, and each export checks this and reports neuron coverage (it must be 1.0, with no duplicates).
> The importance scores below may only order neurons inside a box, or move them to another box, never
> delete them. Remainder boxes must be fused into one grouped projection per layer to serve at speed.
>
> The record below is kept for history.

- **What:** cut box neurons by importance.
- **Script:** `probe_amoeba_box_pack_cpu.py` with `--box-sets`, `--ratios`.
- **Check:** accuracy at each size against the unpruned box. No random-selection cut was run.
- **Reference,** capitals / arithmetic: half the neurons keeps 0.69 / 0.50; one tenth keeps 0.59 / 0.23.
- **Note:** a 248-neuron box trained in place scores 0.66, and random neurons trained the same way score
  0.59. Once neurons are trained, which ones you keep matters little.
- **Do not** try to merge several neurons into one. It breaks at 2:1.

### A9. Tune a box — measured (modest effect)
- **What:** fine-tune a box while protecting what the model already knows, by projecting each update
  away from the directions English uses.
- **Scripts:** `probe_amoeba_box_tune_cpu.py`; `amoeba_box_finetune.py` writes a new registered variant.
- **Check:** topic accuracy, plus the share of previously right answers that stay right.
- **Pitfalls:** project the optimizer's updates, not the gradients, and fit the projectors on the fit
  split only. Free tuning works on random neurons too, so it is ordinary fine-tuning.

### A10. Export — measured on CPU (the serving adapter is design)
- **Tiles:** `export_amoeba_tiled.py` writes `tiles.safetensors` and `manifest.json` (schema
  `mlx2.amoeba-tiles.v1`). Core tiles are packed kept neurons. Box tiles carry their gate.
- **Dense checkpoint:** `export_amoeba_core_checkpoint.py --masks --with-box --name`, plus `--verify`
  *(confirm branch)*. It zeroes the dropped neurons and keeps tensor shapes, so any server can load it.
- **Check:** largest logit difference against the masked reference. The dense checkpoint gives zero
  against the dense masked computation; the saved bf16 tile file gives 3.7e-3.
- **Pitfall:** "matches the masked computation" is not "matches the sink rule". A dense checkpoint agrees
  with the sink-rule model on the top token only 82–96% of the time.
- **Not built:** the serving adapter that loads tiles, cache namespacing by tile set, and the scheduler
  changes.

### A11. Register the box — built (in use)
- **What:** one JSON record per box with its provenance, a changelog and its variants.
- **Script:** `amoeba_box_registry.py` (`ingest-skill-boxes`, `log`, `verify`, `status`, `select`).
  `amoeba_box_ab.py` scores variants against one core and logs the result.
- **Rule:** selection needs a named approver. Scoring never changes a default.

### A12. Serve and compare — measured
- **What:** serve the base model, the core and the core plus box as separate arms, and run the 20×20
  sanity tasks.
- **Script:** `amoeba_ab_sanity.py --arm name=dir --port 8391 --rounds 20`.
- **Reference:** base 263 of 320; exported core with layers 0–2 whole (74.7% kept) 197; core plus box 193.
- **Reading:** this core loses about a quarter of these tasks. Report this next to any byte saving.

### A13. Optional extras
- **Unsupervised boxes** (measured): cluster co-firing neurons to find skills without labels.
- **Mixed precision** (measured, with a confound): `probe_amoeba_tile_quant_cpu.py --bits`. Boxes
  tolerate 4-bit; keep the core at 8-bit.
- **Box speculation** (measured on a CPU proxy only): `probe_amoeba_box_spec_cpu.py`. The core drafts,
  and the core plus boxes verify.
- **Box mixer** for prompts needing two skills (built): `probe_amoeba_box_mixer_cpu.py`.

## Track B. Attach an expert from another model

Three forms exist. Pick by what you have.

| Form | Use when | Cost | Status |
|---|---|---|---|
| Exact delta box | The donor is a fine-tune of the host | Large; needs most of a 13-layer stack | measured; two boxes interfere |
| Gated adapter | You can train on the donor's outputs | 81 MB at rank 64 | measured for one skill |
| Side box with output hitch | The donor has a different tokenizer or architecture, or the expert must serve several hosts | 4.09 GB for a 45% cut of a 4B donor | measured for one skill |

Copying donor neurons into a different host through fitted linear maps is not on this list. It failed:
98% on trained facts and 0% on held-out facts.

**Fourth form, under test: a mapped donor box (the hub route).** The donor box's output is translated into
the host's coordinates through a depth warp and ridge maps per layer band, fitted on text both models
read, then injected where the gate opens.
- Phase 0 showed the coordinates translate: stitch KL of 0.02–0.2 nats, against 4–7 for shuffled pairs, for
  Qwen 27B, Gemma-4-26B and Flash-Next donors into Qwen3.5-4B.
- Phase 1, the real transfer test on held-out problems with size-matched controls
  (`scripts/amoeba_hub_transfer.py`), runs in a GPU slot approved on 2026-11-14.
- Steps, owners and status: `cross-model-remapping.md` §4.
- Fit maps on broad data, or they memorise.

Side-box and adapter code is in `~/amoeba-lab/sidecar/` on the M3, preserved in git as
`docs/research/amoeba-sidecar-m3-20261113/amoeba-lab-20261118.bundle` (full history to 3add7f4), with results in
`docs/research/amoeba-sidecar-m3-20261113/`. Reference numbers use donor `mlx-community/Qwen3-4B-4bit@4dcb3d10`
and host `Qwen/Qwen3-0.6B@c1899de2`.

### B1. Screen for a real gap — measured
- **What:** find a skill where the donor is good and the host is not.
- **How:** score both zero-shot on an inventory split. For the host, also try a newline at position 0, a
  two-shot prompt, and a second phrasing. Fix the pass rule in code before running.
- **Check:** the gap must survive all three host conditions.
- **Reference:** among 14 simple skills, none passed (host at most 30% in all three conditions, donor at
  least 80%). Expect partial gaps. Examples, as host best against donor: 3-digit × 2-digit multiplication
  38% against 73%; sorting three numbers 76% against 97%; 2-digit × 2-digit multiplication 82% against 98%.

### B2. Fix splits and phrasings — measured
- Disjoint inventory, train, validation and test items. Treat a pair and its reverse as one item.
- At least two phrasings that never appear in training.
- A harder or out-of-range set, such as 4-digit sums for a 2–3 digit skill.
- For facts, report trained facts under new phrasings separately from held-out facts. An expert cannot
  learn a new fact from other facts.

### B3. Choose the first-position token — measured
- **What:** models without a begin-of-sequence token need a fixed first token. Choose it on validation.
- **Reference:** a newline works for the Qwen3 family. `<|endoftext|>` made the 4B donor answer " ?" and
  score 2 of 400.

### B4. Build the expert
- **Gated adapter — measured.** Run the donor once and cache its output distributions at the answer
  positions. Train a rank-64 LoRA on the host to match them. Save the weights. Set the learning rate from
  training loss only. Larger ranks overfit: rank 1024 fell to 17.7% on the harder set. Not yet run: the
  same adapter trained on correct answers alone, which would show whether the donor is needed.
- **Side box — measured.** Learn a hard mask over the donor's MLP neurons and attention heads, with KL to
  the full donor at the answer positions plus a size penalty. Let position 0 run the full donor and
  store its cache. Slice the kept weights into a compact model. The smallest working cut kept 44.7% of
  neurons and 40% of query heads. A 22% cut failed, and so did one 48% cut whose mask training
  collapsed, so 45% is a rough bound from a short fit and not a floor.
- **Vocabulary cut for a side box — measured.** Keep only the embedding and output rows the skill uses
  (14 in, 49 out for bare sums). This saves the embedding, 19% of the box. The box must abstain when the
  input contains a token outside its vocabulary.
- **Exact delta box — measured.** Subtract the host's MLP weights from the fine-tune's.

### B5. Join at the output — measured
- **What:** run the expert as a side stream with its own cache. Mix probabilities:
  `p = (1 − a·g)·p_host + a·g·p_side`, with gate `g` and mixing weight `a`.
- **Different tokenizers:** let the side box re-read the text as a string. Give each host token the side
  probability of the side token with the same string. No token ids cross between models.
- **Choose `a` and the gate threshold on validation,** with a cap on English cost (we used 0.01 nats).

### B6. Gate it — measured
- **Confidence gate:** open when the side stream's top probability passes a threshold. Simple and
  accurate, but the side stream must always run.
- **Host-side classifier gate:** a linear probe on the host's hidden state (layer 24 of Qwen3-0.6B).
  Train it with hard negatives: numbers in prose, dates, prices and other operations.
- **Check:** recall on answer tokens, false firing on English and on each hard negative, and two
  controls: a shuffled-label probe and a random gate at the same rate.
- **Reference:** no false firing on prose numbers, dates or prices. Recall 99.9% in the trained format,
  66–80% on unseen phrasings.
- **Pitfall:** a gate that reads the host's state cannot open if the host has already gone off course.
  Fix the prompt format first (B3).

### B7. Run it lazily — measured
- **What:** with a host-side gate, do not run the side stream until the gate opens. Then let it catch up
  on the text so far.
- **Check:** the same accuracy as always running, and host-level speed on plain English (83.6 against
  85.6 tokens/s). The accuracy match was checked on the first 100 test items only.
- **Pitfall:** when timing, force the side stream's output to be computed. MLX skips unused work, which
  made one timing run invalid.

### B8. Score with the full set of controls — design (a checklist; partly measured)
Every table should have these rows. The first five and the last have been run; the adapter-training
controls have not.

| Row | Why | Run so far |
|---|---|---|
| Host alone, raw and with the position-0 token | The fair baseline | yes |
| Host two-shot | Shows what a prompt alone buys | in the skill screen only |
| Donor alone | The ceiling | yes |
| Whole donor behind the same gate | Shows whether cutting the donor gained anything | yes |
| Random box, shuffled gate, random gate | Shows the content matters | yes |
| Adapter trained on correct answers, and on the host's own outputs | Shows whether the donor was needed | no |
| English cost, added bytes, tokens/s | The price | yes |

## Track C. Map what a model's parts do

Use these before cutting, to decide where to look.

- **C1. Neuron atlas — built.** `amoeba_neuron_atlas.py` gives every MLP neuron a stable id, a co-firing
  cluster and tags. Boxes refer to neurons by these ids.
- **C2. Topic heads — measured.** `probe_amoeba_head_map_cpu.py` ranks attention heads by how specific
  they are to a topic.
- **C3. Expert signatures and functional maps — measured, for mixture-of-experts models.** Read an expert
  through the vocabulary, or compare experts by their effect on outputs. Trust a match only above the
  row-permuted baseline.
- **C4. Donor box location — measured (one problem).** `probe_amoeba_unit_box.py` (branch
  `claude/quirky-poitras-2cdefc`) finds a small part of a large donor that still solves one problem, with
  a random twin as control. The search uses a coarse grid of sizes, and the result is not bit-exact for
  two of four donors. Moving that part into a host is not done.

## Track D. Control thinking effort

- **D1. Measure effort levels — measured.** `amoeba_adaptive_effort_ab.py` and
  `amoeba_adaptive_effort_analyse.py`. Run the same problems at each effort with the server in
  `--ordinary` mode, and record accuracy and tokens. On the quantized 27B, low effort matched xhigh at
  two thirds of the tokens; on the full-precision 27B, xhigh was more accurate. Test the precision you
  will serve. Drop requests that returned a transient 503; do not count them wrong.
- **D2. Conclude runaway thinking — measured (the client policy is built).** `amoeba_spiral_detector.py`
  and `amoeba_spiral_ab.py` (branch `claude/amoeba-lane-spirals-be1149`). When the detector fires, the
  client asks the model to conclude from the exact tokens so far. The opt-in client module
  `src/mlx2/spiral_conclude.py` has not yet run against a live server.
- **D3. Build an effort path — measured (a repair, not a gain).** `amoeba_effort_donor_traces.py`
  collects donor runs per effort. `amoeba_box_paths_4b.py --layers 4 --rank 16 --memory-limit-gb 40`
  trains a small weight change on the host against them. Register with the registry `path-*` commands.
  No path is selected. The baseline it repaired ran under a 4,096-token thinking budget; a rerun with a
  larger budget is queued.

## Track E. Serve tiles fast (learned 2026-11-14/15)

- **Fuse the tiles.** Per layer, use one grouped gate/up projection per weight format for the core and every box, one batched gate pass applied as a mask, one scatter and one down projection. Unfused, five boxes per layer added about 2,000 small GPU operations per token and halved decode speed (10 tok/s). Fused, the tiled 27B decodes 17–21% faster than the full model.
  - Branch `claude/amoeba-27b-spec-tiles`, policy `amoeba_tiles_layout: "fused"`.
- **Use the compiled `swiglu`,** not `silu·up`.
- **Install tiles before speculative routes,** so row-exact verify, lane matmul and the AST tree tiles cover the tile projections. Tiles plus AST/DFlash2 is the fastest 27B configuration measured: about 101 tok/s total at B3–B4.
- **Measure with long outputs.** 40-token answers and a 6 GiB MLX cache limit made AST look slower than plain decoding. Use 512-token outputs and a 16 GiB cache, and make the harness fail closed unless the route receipt shows tree tiles and verify rounds.
- **Thermal holds:** on a busy day, per-cell thermal admission never settles. Run ABBA-ordered unsettled comparisons for decisions, and keep settled runs for numbers you will quote (overnight).

## Track F. Cross-model transfer (status 2026-11-15)

See `cross-model-remapping.md` §4–5.
- The working pieces are the warp, the ridge maps, the donor-gap screen (centred per condition, BH-FDR) and prompt logprobs on :8282 for scoring a served model without pausing it.
- Not working yet: injection into a generating host. A carved donor box at 0.9 mass is general machinery, so carve contrastively, on topic minus off-topic text.
- For distillation into tiles, use a top-K-plus-tail KL; check that the self-distillation gradient is 0 at init; train with prefix context; add a retain loss on off-topic and ordinary text; select on a validation split.

## Track G. Deterministic tiles: calculator and fact table (2026-11-15)

Neural boxes are weak at exact computation (27B core + boxes: multiplication 7/20) and at facts about
entities they were not fitted on (capitals box: 0.575 on 40 unseen countries, full model 0.95).
Deterministic tiles plug exact tools into the same tile grid, on the neural boxes' contract:

- read the MLP input at layer L plus the token text;
- open or stay closed per token;
- write a delta to the MLP output at L' >= L, or force/constrain the next tokens.

A closed tile adds exactly zero. Code: `src/mlx2/runtime/amoeba_det_tiles.py` (schema
`mlx2.amoeba-det-tiles.v1`), adapter policy `amoeba_det_tiles` on Qwen3.8-27B (ordinary route only).
Experimental and unqualified; receipts carry the tile kind and `exact`.

### G1. Calculator — measured on CPU (Qwen3-0.6B)

- **Trigger = probe AND parser.** A logistic probe on the MLP input ("an arithmetic result comes
  next"), with layer and threshold chosen on val. The parser needs a complete `a OP b` followed by
  `=`, `equals` or `is equal to` at the end of the text, or a question ending in `?` whose answer
  slot is next ("What is 7 times 12?"). A bare `is` is not a cue.
  - Operators: + − × ÷ and their words, on integers and decimals.
  - Refusals: chained expressions (`2 + 3 * 4 =`), dates, times, years and ranges (`1999-2004`,
    unspaced `a-b` with a < b), phone-like groups, versions, operands glued to IDs, currencies,
    unspaced `x`, non-ASCII digits, and mixed units.
  - What stands before `a` is an allow-list (start of an untruncated text, a sentence or line
    break, a quote, GSM8K's `<<`, or lead-in words like *is*, *so*, *we need*, *computed*). A
    deny-list leaked: `2 \times 3 + 4 =`, `10 – 3 + 2 =`, `the square root of 16 + 9 =` and
    `20% of 50 + 1 =` all end in a valid `a OP b =` whose value alone is wrong. Gaps may not cross a
    newline (a `- 4` bullet is not subtraction), a one-sided `$` is refused (it may open LaTeX math),
    and an operand at the start of a cut context window is refused (found by the second review,
    fixed in the commit after the 0.6B reruns, which used only the allowed lead-ins).
  - The result is exact (`Fraction`). A non-terminating quotient is refused, and the host keeps its tokens.
- **Write:** `force` (canonical tokens) or `constrain` (any tokenization that keeps the text a prefix
  of the result). Both ban a digit after the result, also after a following `.` or `,`. Without
  that ban, `2090` became `20900` on 22 of 80 items. Masks are finite, so they compose with the
  thinking guard, and the tile aborts if an earlier processor already banned its token.
- **The parser carries the trigger; the probe guards it.** The probe alone fires on 3–4% of
  positions in dates, IDs, phones, prices, times and versions; the parser alone fires on 2.6% of
  non-answer positions inside arithmetic text. Together they fire on 0% of every negative type and
  of 23k English positions. Recall on teacher-forced test answer positions is 0.79; the misses are
  parser refusals (an unspaced `156-473=` reads as a range, and question forms without `?`).
- **Controls** fire at the real probe's rate on parse-positive val positions (0.36), through the
  same parser: a random gate, and a probe trained on shuffled labels with its own manifest. On
  clean arithmetic prompts a random gate already gets 42 of 80, because the parser does most of the
  selection. So the probe's value is in the guard, not in these accuracy numbers.
- **Results**, `Qwen/Qwen3-0.6B@c1899de2`, mlx2 `3065852f7`, whole-number scoring (`36` fails on
  `360` and `36.5`), evidence `docs/research/amoeba-det-tiles/calc-qwen3-0.6b/` (earlier runs kept
  as `-v1-unclosed` and `-v2-prereview`):

  | task | host | force | constrain | push ×2 (best) | random gate | shuffled-label probe |
  |---|---|---|---|---|---|---|
  | 27B multiplication items (80 held, 3-digit × 2-digit) | 10 | **80** | **80** | 57 | 42 | 10 |
  | 20×20 sanity multiplication (20) | 7 | **20** | **20** | 12 | 9 | 7 |
  | held-out wordings, mixed operators (40) | 12 | 28 | 28 | 23 | 18 | 20 |

- **Force or constrain beats a residual push.** The push (research only, toward the planned token
  at the last layer) reaches 57/80 at its best scale and is approximate. Serving refuses it.
- **Open:** the mixed-wording set dropped from 37 to 28 when the review tightened the parser (no
  bare `is`, `?` required, ranges refused); this is the price of 0% false fire. The calculator does
  not follow multi-step chains (GSM8K) beyond each single `a OP b =` it sees.
  A GSM8K check of the calculator has not been run. When it is, use the harness default from
  Forgejo main `507762087` (`scripts/spec_acceptance_gsm8k.py`, chat few-shot; the 27B scores
  56/60 there against 8/60 in raw completion, a format failure), and don't compare against
  completion-mode numbers.

### G1b. Calculator parser v2: whole expressions (CPU, Qwen3-0.6B)

- **What changed:** `calc-parse-v2` parses a whole expression with precedence, brackets, `^`/`**`
  and a leading minus, and evaluates it exactly. A lone `a OP b` still goes to v1 unchanged, and
  every v1 guard still applies around the expression. Tiles opt in through a `parser` field in the
  manifest; manifests without it are v1 with their original identity (and APCv2 namespace).
- **Extra refusals:** non-terminating fractions, exponents that aren't small integers, results
  over 60 digits, a spaced leading minus (a list bullet), a chain broken across a line,
  implicit multiplication, unbalanced brackets, word operators glued to numbers, nothing to compute
  (`So -3 =`), and any internal parser error (fail closed, never raise into decode).
- **Review:** a fresh-context review of the first version found six defects, all fixed before the
  run: a crash on `divided by`, wrong tails of expressions split across lines, times and ratios
  read as operands (`10:30 + 15 * 2`), unbounded powers, a changed identity for v1 manifests, and
  echo/range fires.
- **Results**, `Qwen/Qwen3-0.6B@c1899de2`, same probe and seed in both arms (L4, threshold 0.5),
  v1 at mlx2 `c668d1cf4`, v2 at `f2ffc21fb`; evidence `docs/research/amoeba-det-tiles/calc-qwen3-0.6b-parse{v1,v2}/`:

  | task | host | v1 force / constrain | v2 force / constrain | v2 random gate | v2 shuffled-label probe |
  |---|---|---|---|---|---|
  | chained expressions (40 held, new task) | 2 | 2 / 2 | **40 / 40** | 12 | 7 |
  | 27B multiplication items (80) | 10 | 80 / 80 | 80 / 80 | 42 | 10 |
  | 20×20 sanity multiplication (20) | 7 | 20 / 20 | 20 / 20 | 9 | 7 |
  | held-out wordings, mixed operators (40) | 12 | 28 / 28 | 28 / 28 | 18 | 20 |

  False fire is 0% on every negative type and on English in both arms; recall on teacher-forced
  answer positions is 0.79 in both. The probe was trained on single steps only and still opens on
  chained text.
- **Not run yet:** the 27B and GSM8K (chat few-shot, the harness default since `507762087`).
  Unqualified; v1 stays the default parser.

### G2. Fact table (PLE-style, hybrid) — measured on CPU (Qwen3-0.6B)

- **Recogniser:** Fisher-selected host units read at the answer position. The builder compares MLP
  neurons with MLP-input (residual) dimensions, and val chooses.
  - On the 0.6B, the country reaches the last position only late. Residual dimensions at layer 24
    score val top-1 0.98; neurons at 21 score 0.59.
  - **Pitfall:** centre the keys (subtract the mean recogniser vector). Otherwise the shared
    "capital" component dominates every cosine.
- **Table:** one row per (country, capital), plus null rows for the same countries under other
  relations (language, currency, continent, population, flag). "The official language of France"
  then lands on a null row and falls back.
- **Rows** are solved per fact in closed form: a minimum-norm first-order step on log p(first answer
  token) at the write layer, repeated 4 times, on 5 key wordings (2 iterations and 3 wordings on the 27B) (one of them chat). Every row's
  source and solve history is in `rows.jsonl`.
- **Threshold** (τ, margin) is chosen on val, on the fit countries only:
  - wrong row ≤ 2%;
  - false fire ≤ 1% on English, other relations and non-answer positions;
  - **absent entity ≤ 2%**: all of the prompt entity's rows masked out.
  - Without the absent-entity gate, deleted-row countries opened on a neighbour's row 22.5% of the
    time (0.68 against the host's 0.80). With it, 7.5% at test (0.75).
- **Streaming:** the table is 4.4 MB on the 0.6B (858 rows: 143 capital rows and 715 null rows, each
  with an f32 value row and a 256-dim key; null rows store zeros), so it stays resident.
  PLE-on-NVMe (`qwen4_ple_nvme.FileBackedShardedEmbedding`) serves rows by token id. A nearest-key
  table would need a host sync for the row id before the fetch, so it was not reused; it becomes
  worth it when tables reach GB scale.
- **Results**, `Qwen/Qwen3-0.6B@c1899de2`, mlx2 `3065852f7`; recogniser residual dims at L24,
  write at L27, τ 0.75, margin 0.05 (val: wrong row 0, absent entity 2.0%); evidence
  `docs/research/amoeba-det-tiles/facts-qwen3-0.6b/` (earlier runs kept as `-v1-no-absent-gate`
  and `-v2-prereview`). The host is the full 0.6B model.

  | category | host | tile | shuffled values | tile opens |
  |---|---|---|---|---|
  | trained facts, key wording (103) | 0.505 | **0.932** | 0.350 | 0.98 |
  | trained facts, new wordings (412) | 0.396 | 0.536 | 0.289 | 0.57 |
  | unseen countries, key wording (40) | 0.450 | **0.925** | 0.225 | 0.95 |
  | unseen countries, new wording (40) | 0.650 | **0.850** | 0.325 | 0.73 |
  | unseen countries, chat (40) / sanity-suite chat (40) | 0.175 / 0.225 | 0.225 / 0.325 | 0.125 / 0.125 | 0.40 / 0.38 |
  | French / German frames (20 each) | 0.10 / 0.55 | 0.10 / 0.55 | — | 0 / 0.10 |
  | multi-step ("country with the Eiffel Tower", 19) | 0.00 | 0.00 | — | 0 |
  | not in table: 10 extra countries | 0.20 | 0.20 | — | 0 |
  | not in table: 20 trained rows deleted (40 prompts) | 0.80 | 0.75 (full table 0.95) | — | 0.075 |
  | other relations (60) | — | — | — | 0 |
  | English Δ CE (60 seqs, 7,620 positions) | — | **0.000** (0 opens) | — | 0 |

  **Label honestly: "unseen countries" are unseen by the recogniser and calibration, not by the
  values.** Their rows are solved on the key wordings, so the key-wording row (0.925) is in-sample
  for the values. The new-wording row (0.85, a wording no row was solved on) is the clean
  out-of-sample number. A random gate at the tile's open rate scores 0.33 / 0.50 / 0.23 on trained
  new wordings / unseen new wording / unseen key wording, and shuffled values fall below the host,
  so the value rows carry the facts. With deleted rows the absent-entity gate holds at val (2.0%)
  but leaks 7.5% at test, costing 2 of 40 prompts against the host.
- **Reading:** the table fixes the neural box's weak spot, unseen entities. A row is added in closed
  form without refitting anything. Paraphrase robustness is partial (0.54 on trained facts in new wordings, 0.85 on unseen countries). Other
  languages and multi-step questions do not reach the key at all, so they fall back to the host
  rather than fail. To cover them, key them, or recognise the entity where it is mentioned.

### G3. 27B check (GPU, 2026-11-15/16)

`Qwen3.8-27B-BF16-q8-mlx2@2dd8d304`, mlx2 `4a49d1ca2`, one owned window (CPG lease, paired locks,
:8282 paused and healthy after). Host = core + boxes, fused (`...-tiles-core-boxes`). Evidence
`docs/research/amoeba-det-tiles/27b-4a49d1ca2/`. Unqualified.

- **Calculator build:** probe at L24, threshold 0.8 (val false fire ≤ 0.8% on every negative type).
  The probe alone fires far more on the 27B than on the 0.6B (prices 25%, code 18%, phones 14%,
  versions 12%); with the parser, 0% on every negative type and on English. Recall on teacher-forced
  test answer positions 0.79 (parser refusals, as on the 0.6B).

  | task (greedy, research driver) | core + boxes | force | constrain | random gate | shuffled-label probe |
  |---|---|---|---|---|---|
  | 27B multiplication items (40 held) | 3 | **40** | **40** | 26 | 25 |
  | 20×20 sanity multiplication (20) | 16 | **20** | **20** | 17 | 16 |
  | held-out wordings, mixed operators (40) | 14 | 32 | 32 | 20 | 24 |

- **Served** on dev port 8391 through the adapter policy (`--ordinary`, chat, thinking off,
  max 48 tokens), B1 and B4:

  | task | core + boxes | + calculator | core + boxes − capitals box | + fact table |
  |---|---|---|---|---|
  | 20×20 sanity multiplication (20) | 17 | **20** | 17 | 17 |
  | 3-digit × 2-digit (20) | 1 | **20** | 1 | 1 |
  | 20×20 capitals (20) | 20 | 20 | 19 / 18 (B1 / B4) | 19 / 18 |
  | unseen-country capitals, held-out wording (20) | 8 | 8 | 5 | 5 |

  Scores are the same at B1 and B4 except no-capitals 20×20 (19 vs 18). The calculator planned 40
  times, exactly once per arithmetic prompt, and never on the 5 number-heavy false-fire prompts or
  the prose prompts (process counters, receipts in `served/calc/rows-B*.json`). Every response
  carries the det-tile route receipt (`kind`, `exact`, unqualified). 2–5 of 87 outputs differ
  between B1 and B4 in every arm, the calculator-free ones included; these are host-route width
  drift, not classified (no logprob windows were captured).
- **Reading:** on the 27B the calculator takes multiplication from 1–3 correct to all correct, with
  0 false fires served. Removing the capitals box costs unseen countries 8 → 5 of 20; many misses are
  the host writing reasoning text ("The user is asking for the capital of …") despite thinking off.
- **Fact table** (second window, mlx2 `64f767e41`, 2026-11-16 05:30–06:00, same locks; evidence
  `docs/research/amoeba-det-tiles/27b-64f767e41/`, table tensors not committed). Host = core + boxes
  without the capitals box (fused). Two value-solve fixes made it run: reference softmax attention
  for short queries under the gradient (`3f162b733`) and tile gate counting suspended inside
  `value_and_grad` (`31bc4ba54`; the counters' `async_eval` is not allowed in a graph transformation).
  Recogniser: residual dims, read L48 (every read layer 40–58 separates the fit keys); write L59;
  key threshold 0.5, margin 0.1, chosen on validation (recall 0.59, 0 wrong rows, 0 false opens on
  other relations, interior positions, English and absent entities). Values solved in 2 iterations
  (1163 s).

  | set (research driver, greedy) | host | + table | shuffled values | table open rate |
  |---|---|---|---|---|
  | trained facts, row (key) wording (412) | 0.466 | **0.583** | 0.495 | 0.62 |
  | trained facts, new wording (412) | 0.481 | 0.481 | 0.481 | 0.005 |
  | unseen countries, row wording (40) | 0.325 | **0.45** | 0.35 | 0.53 |
  | unseen countries, new wording (40) | 0.425 | 0.425 | 0.425 | 0 |
  | unseen countries, chat / sanity chat (40) | 0.15 / 0.125 | same | same | 0 |
  | French / German (20) | 0.35 / 0.65 | same | same | 0.1 |
  | multi-step (19) | 0.47 | same | same | 0 |
  | deleted rows (40; host 0.625) | — | full table 0.70, rows deleted 0.625 | — | 0.025 on deleted |

  A random gate opening at the table's rate gives 0.35 on unseen countries in the row wording, below
  the table's 0.45. English: Δ CE 0.000, 0 open positions in 7,687. Served (last column of the table
  above, dev port 8391, B1 and B4) the table **never opened**. The counters show 0 open rows in 1,675
  forwards, so the route receipt reports `observed_used: false`, and every score equals the
  no-capitals arm.
- **Reading:** on the 27B the calculator takes multiplication from 1–3 correct to all correct, with
  0 false fires served. Removing the capitals box costs unseen countries 8 → 5 of 20; many misses are
  the host writing reasoning text ("The user is asking for the capital of …") despite thinking off.
  The fact table works only in the wording its keys were cut from: +12 points on trained facts and
  +12.5 on unseen countries there. It is safe everywhere (no English cost, no false opens, shuffled
  values gain little), but its key does not match any other wording on the 27B. That includes the
  chat prompts it is served on. On the 0.6B the same build reached 0.85 on unseen countries in a new
  wording. So the 27B table does not yet replace the capitals box (8/20 served), let alone the full
  model (0.95). Next: key on the entity mention rather than the question-final position, or add
  chat-wording keys, and set the threshold on chat validation prompts.

## Process lessons (2026-11-14/15)

- **Queue by gate files, tolerating pid changes.** Waiters must accept a pid that changes when a chain restarts: require the file's pid to be dead for several minutes. Don't filter gate files by timestamp, and never assume a tag exists before its owner writes it.
- **Watchdogs:** use footprint, not RSS, and Pierre's 80 GB per-process limit. A 60 GB cap killed a valid window.
- **Stop early when the answer is clear** (phase 1 stopped at 44 of 120 min), but make every driver write a receipt on SIGTERM.
- **Gradient parity across drivers:** compare each against an fp32 reference, never against each other. bf16 noise in layer-wise against whole-graph gradients grows with depth: 1% at 4 layers, 8% at 64.
- **Carving corpus decides residency.** A core carved on repo text keeps code by construction; carve on graded prose.

## Checklist before you report a result

1. Is the checkpoint named with its fingerprint, and the commit recorded?
2. Was every choice made on fit or validation data, and not on the set you report?
3. Is there a random or shuffled control in the same table?
4. Did you measure English cost, and not only the topic score?
5. Did you test the host with a position-0 token and a two-shot prompt?
6. Did you validate the saved artifact?
7. Does the write-up say what was not run?
8. Is the status stated truthfully: implemented, qualified, selected, or observed in use?
