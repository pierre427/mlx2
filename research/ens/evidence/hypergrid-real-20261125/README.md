# Count hypergrid → collapse on a real corpus (2026-11-25)

**Corpus:** aMoEba category box `sci.physics.fluids`, from amoeba-corpus
`builds/category-boxes-8ec12cb93` @ `806ac64`. The box has two registers, arXiv
abstracts and academic papers. The `fit` split is training data (2.0M word tokens),
`val` is used for tuning, and `held` is scored (391K tokens).

**Run details:** word level with a 10K vocabulary (3.0% unknown words on held), numpy
on CPU, no model checkpoint. The mlx2 tree is at `ecbb1739a`; the script is untracked.
Script: `hypergrid_real.py`; full numbers: `results_fluids.json`; log: `run_fluids.log`.

## Arms
- **A: hypergrid.** Raw trigram counts with Kneser-Ney-style interpolated backoff.
- **B: collapse.** Left/right context counts → PPMI → SVD (rank 100) → k-means merge
  into k word classes → class trigram × P(word | class). No gradients.
- **C: A+B.** A and B blended with one global weight, tuned on val.
- **D: MLP.** emb 64 + hidden 256 + softmax, Adam, best of 4 epochs on val.
- **A+D.** A and D blended the same way.

## Results: held NLL, nats per word

`unseen bigram` means the (previous word, word) pair never occurs in fit: 45,585 positions.
`unseen trigram` means the pair was seen but never with this first word: 118,926 positions.

| arm | stored numbers | all | unseen bigram | unseen trigram |
|---|---|---|---|---|
| A hypergrid | 1.37M | 4.583 | 10.358 | 5.835 |
| B collapse k=64 | 79K | 5.610 | **8.701** | 6.267 |
| B collapse k=256 | 286K | 5.435 | 9.133 | 6.153 |
| B collapse k=1024 | 542K | 5.176 | 9.476 | 6.029 |
| C A+B k=1024 (λ=0.8) | — | 4.489 | 9.629 | 5.660 |
| D MLP | 3.24M | 4.491 | 8.943 | 5.396 |
| A+D (λ=0.45) | — | **4.362** | 9.072 | 5.417 |

Example merged classes from the SVD+merge step, with no supervision:
- At k=1024, {velocity, heat, pressure, energy, temperature, density, mass, momentum,
  concentration} form one class.
- At k=1024, {increases, becomes, remains, varies, reaches, exceeds, dominates} form one class.

## Findings
1. **The toy result partly holds.** On word pairs the corpus never showed, the collapsed
   model is the best arm: 8.70 nats at k=64, against 8.94 for the MLP and 10.36 for raw
   counts. It does this with 79K stored numbers and no gradient descent.
2. **Hard classes lose too much on ordinary text.** Collapse alone scores 5.2–5.6 overall
   against 4.58 for raw counts, because putting every word of a class at its class
   frequency throws away word identity. The same knob as in the toy shows up here:
   smaller k generalizes better on unseen pairs and fits ordinary text worse.
3. **Memory plus collapse beats either alone** (4.489), and it matches the MLP (4.491)
   while needing no training loop. Memory plus MLP is better still (4.362). The trained
   model's continuous compression keeps more than hard clustering does.
4. **One global blend weight wastes the collapse.** With λ=0.8 on the memory, the blend's
   unseen-pair score falls back toward the raw counts (8.70 → 9.42–9.63). The right mix
   trusts memory where its context was seen often and the collapse where it was not,
   which is a per-context gate.

## What this points to
- **Soft collapse instead of hard merge.** Keep the SVD embeddings continuous, for
  example as a low-rank log-bilinear model read straight off the counts. Alternatively,
  keep several clusterings at different k.
- **A per-context gate between memory and collapse.** In ENS terms this is the bus/gate:
  route to the exact-count memory when the context is well attested, and to the
  compressed tile when it is novel.
- **Then the bus-query version:** count over quantized gate query vectors instead of words.

---

# Donor-steered grid (2026-11-25 evening, M5 GPU window 19:07:35–19:44:11)

**Question.** Can a donor model's logits on the same text boost the grid's connectivity?

**Method.** Each donor runs forward-only over the fluids box and records its top-32
next-token log-probs per position (`donor_dump.py`, mlx-lm-unified `1104ced`, mlx 0.32.4).
The grid is rebuilt on that donor's own tokens (`donor_grid.py`). Scores are held NLL in
**nats per donor token**, so they are not comparable to the word-level table above.
Every arm is scored on the same held positions. Splits: fit ≈ 0.8M tokens, val 60K,
held 150K. Exception: the 0.6B fit split is 0.63M tokens, cut by an early time cap
before documents were packed into full windows.

Arms:
- **A_kn:** raw trigram counts with KN-style backoff (the hypergrid).
- **A_add:** the same counts with an additive estimator that accepts fractional counts.
  This is the fair baseline for the soft arms.
- **S_soft:** counts plus α × the donor's top-32 probabilities at every fit position.
- **S_promote:** soft counts only for alternatives the text did not use, where the
  donor gives p ≥ 0.1.
- **G_gate:** A_kn and S_soft blended with a weight per context-count bucket, tuned on val.

| donor (checkpoint dir) | donor | A_kn | A_add | S_soft (best α) | S_promote α=1 | G_gate | gap closed by gate |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-BF16-microscaling-source | 2.846 | 5.211 | 5.725 | 5.038 (1) | 5.546 | **4.888** | 14% |
| Qwen3-4B | 2.452 | 5.088 | 5.609 | 4.927 (1) | 5.389 | **4.794** | 11% |
| Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit | 2.008 | 5.108 | 5.614 | 4.799 (4) | 5.360 | **4.664** | 14% |

"Gap closed by gate" is (A_kn − G_gate) / (A_kn − donor).

Unseen-bigram positions (about 10–11% of held tokens):

| donor | donor | A_kn | S_soft | G_gate |
|---|---|---|---|---|
| 0.6B | 4.71 | 10.30 | 9.92 | 9.37 |
| 4B | 4.27 | 10.37 | 10.05 | 9.51 |
| 35B-A3B | 3.32 | 10.53 | 9.68 | 9.14 |

Stored numbers: the hard grid holds 0.75–0.97M; the soft grid holds 12.6–16.9M (about
17× denser); the promote grid holds 1.2–1.5M.

## Findings
1. **Donor steering helps every time.** Against the same estimator, soft counts gain
   0.69–0.82 nats per token. With the gate, they beat the best raw-count grid by
   0.29–0.44 nats, and the gain is largest on unseen bigrams (−0.9 to −1.4 nats).
2. **A better donor barely helps the grid.** Going from 0.6B to 35B-A3B improves the
   donor by 0.84 nats but the gated grid by only 0.22. In every case the grid captures
   just 11–14% of the gap to its donor. The bottleneck is the grid's key, not the donor:
   a two-token context averages the donor's distribution over every occurrence of that
   context, which throws away everything the donor knew from the other 2,000 tokens.
3. **Promoting only strong missing edges is compact** (1.5M numbers, against 17M for
   full soft counts) and beats its own baseline by about 0.25 nats. It does not reach
   the full soft grid. Most of the soft-count gain comes from reweighting many mid-
   probability edges, not from a few promoted ones.
4. **The gate leans on the donor-steered grid** (memory weight 0.2–0.5), trusting the
   raw counts more only for well-attested contexts.

## Implication for ENS
Donor logits work as a promoter, but a token-keyed grid can't hold what they carry. The
next key to try is the bus query itself: quantize the donor's hidden state at the gate
into a codebook and count (code → donor top-k) into a product-key style table. The key
then carries the donor's long context, and the counts carry its predictions.

## Caveats
- Nemotron-3.5-Lightning-30B-A3B was dropped. Its weights are on the USB HDD
  (/Volumes/2tb) and loaded at about 25 MB/s, so it could not finish inside the window.
  Qwen3.6-35B-A3B (internal SSD) replaced it. That checkpoint is an abliterated finetune;
  it is used here as a text predictor only.
- Documents were packed into 2,048-token windows, so the donor sees the previous
  document's tail near each document start. First tokens are not scored, but the donor
  numbers are slightly flattered.
- Single seed, one box. Not a performance run: no thermal gating, and host swap sat at
  about 11 GB.

---

# Longer token keys vs bus-query keys (Qwen3-0.6B, 2026-11-25 ~20:20)

**Donor dump.** `donor_hidden.py` on Qwen3-0.6B-BF16-microscaling-source ran in the GPU
window 20:17:26–20:19:05 at about 10.7K tok/s. A self-check confirmed the manual forward
matches `model()` exactly (max logit difference 0). The same fit/val/held token budgets
as the donor runs above. It saved top-32 logits plus two hidden states per position:
L14, the mid-network residual, and L28, the final normed state that the LM head reads.

**Analysis.** `donor_grid2.py`. The same soft-count rule as before (α=1), β tuned on val.
- **Code keys:** PCA-128 of the hidden state, then k-means with 1,024 or 4,096 codes, or a
  256×256 product key. The table holds P(next token | code), built from hard plus donor
  soft counts and backed off to the best soft token chain.
- **GATE:** the code table blended with 4-token KN counts, by code-count bucket.

| arm | stored entries | all | unseen bigram | gap closed* |
|---|---|---|---|---|
| donor (reference) | — | 2.846 | 4.854 | 100% |
| KN, 2-token context | 0.73M | 5.089 | 10.372 | 0% |
| KN, 3-token | 1.39M | 5.139 | 10.515 | −2% |
| KN, 4-token | 2.11M | 5.168 | 10.562 | −4% |
| SOFT, 2-token | 15.0M | 4.865 | 9.624 | 10% |
| SOFT, 3-token | 34.2M | 4.848 | 9.641 | 11% |
| SOFT, 4-token | 56.8M | 4.842 | 9.653 | 11% |
| CODE L14 km4096 | 3.1M | 4.694 | 9.254 | 18% |
| GATE L14 km4096 | — | 4.610 | 9.017 | 21% |
| CODE L28 km1024 | 1.4M | 4.398 | 8.169 | 31% |
| CODE L28 km4096 | 2.5M | 4.347 | 8.340 | 33% |
| CODE L28 pq256×256 | 4.8M | 4.499 | 8.641 | 26% |
| **GATE L28 km4096** | — | **4.263** | 8.151 | **37%** |

\*Gap closed is (KN 2-token − arm) / (KN 2-token − donor).

Adding the last token to the code key (`CODE+tok`) changed nothing: its β went to the top
of the grid, which means the table backs off fully. Every arm's numbers are in
`grid2_h06.json`.

## Findings
1. **Longer token strings don't help at 0.8M tokens.** Raw counts get worse as the context
   grows, because longer contexts are too sparse to estimate. With donor soft counts, 3- and
   4-token keys gain only 0.017–0.023 nats while storing 2.3–3.8× more. The token key is not
   just too short; it is the wrong kind of key.
2. **Bus-query keys beat token keys outright.** The best code table is 0.52 nats better than
   the best soft token chain (4.347 vs 4.842) and stores 23× fewer entries (2.5M vs 56.8M).
   Gated, it closes 37% of the gap to the donor, against about 10% for any token key.
3. **The final-layer state works much better than the mid-network one** (L28 37% vs L14 21%).
   The caveat matters: the L28 state almost directly determines the donor's logits, so its
   codes are close to quantized donor predictions. L14 is the honest number for an ENS gate
   that taps the residual stream mid-network, and it still beats every token key at a
   fraction of the storage.
4. **Plain k-means beat the 256×256 product key** at this data size. With 65K keys, each key
   sees only about 12 positions, so the product key needs much more text than 0.8M tokens.

---

# Donor as suggestion, not fact (L28 km4096 key, CPU, 2026-11-25 ~20:45)

**Question (Pierre).** Downweight the donor's contribution so it acts as a suggestion and
never overrides the new material by itself.

**Setup.** `donor_grid3.py`, same 0.6B dump and the same L28 k-means-4096 key as above.
Every arm backs off to the same donor-free KN 2-token chain, so the donor enters only
through the code table.
- **TEXT:** hard counts per code. The donor only chooses the key.
- **SOFT a:** hard counts plus a × donor top-k probabilities.
- **PRIOR ρ:** the donor's average distribution for each code acts as a prior worth ρ ×
  the text's own count for that code. With ρ ≤ 1 the donor never holds more evidence than
  the text.

"Override" compares each arm with TEXT on the 89,305 held tokens whose (code, token) pair
the text showed at least 3 times: the share of those tokens whose probability the donor
cut by more than 1.25×, and the worst single cut.

| arm | all | unseen bigram | text-seen ≥3 | gap closed* | seen tokens cut >1.25× | worst cut |
|---|---|---|---|---|---|---|
| TEXT (code key, no donor counts) | 4.579 | 9.104 | 2.168 | 23% | 0 | 1.00× |
| PRIOR ρ=0.1 | 4.452 | 8.648 | 2.151 | 28% | 0 | 1.07× |
| PRIOR ρ=0.25 | 4.394 | 8.466 | 2.132 | 31% | 0 | 1.18× |
| SOFT a=0.3 | 4.391 | 8.455 | 2.129 | 31% | 0 | 1.21× |
| **PRIOR ρ=0.5** | **4.345** | 8.317 | 2.108 | 33% | 1.0% | 1.36× |
| PRIOR ρ=1 | 4.301 | 8.179 | 2.079 | 35% | 9.2% | 1.73× |
| SOFT a=1 (earlier best) | 4.308 | 8.200 | 2.084 | 35% | 8.2% | 1.70× |
| PRIOR ρ=2 (donor > text) | 4.277 | 8.078 | 2.054 | 36% | 16.3% | 2.45× |
| SOFT a=3 | 4.275 | 8.048 | 2.046 | 36% | 18.6% | 3.06× |

\*Gap closed is measured from KN 2-token counts (5.089) to the donor (2.846). The extra
gate against KN 4-token counts adds at most 0.02 nats here, so it is left out of the table.

## Findings
1. **The donor's biggest contribution is the key, not its predictions.** With the L28 code
   as the key and only the text's own counts (TEXT), the table scores 4.579. That already
   beats the best donor-steered token key (4.842 for a 4-token soft chain). The donor decides
   where an observation is filed, and the text decides what is filed there.
2. **Capped donor evidence keeps most of the gain without overriding the text.** At ρ=0.25
   the donor never cuts a text-attested token by 1.25× or more (worst case 1.18×) and still
   delivers 60% of the available donor-count gain. At ρ=0.5 it delivers 77%, with 1% of
   attested tokens cut by more than 1.25× (worst case 1.36×).
3. **Treating the donor as fact buys little and overrides a lot.** Going from ρ=0.5 to ρ=2 or
   a=3 gains only another 0.07 nats, while cutting 16–19% of text-attested tokens by more
   than 1.25×, in the worst case by 2.4–3.1×.
4. **Where the donor helps most is where the text is silent.** On unseen bigrams the capped
   prior gains 0.6–0.8 nats over TEXT, while on text-attested tokens the gain is only
   0.04–0.06.

**Recommendation.** Use a capped prior with ρ = 0.25 for the strict "never overrides" reading
and ρ = 0.5 for a light one. In both, the donor routes and suggests, and the text decides.

---

# Does a counted tile add to the donor? And state merging (CPU, 2026-11-25 ~21:05)

`donor_grid4.py`, run on the same Qwen3-0.6B dump. Blend: P = λ·P_donor + (1−λ)·P_tile, where
P_donor of the true token is the donor's exact log-prob. λ is set per bucket (tile evidence
× donor confidence) and tuned on val. The bar is the aMoEba one: at least 0.05 nats below
the donor alone on held text.

| tile blended with donor (donor alone = 2.8460) | tile alone | donor + tile | Δ all | Δ unseen bigram | Δ bigram seen ≥3 |
|---|---|---|---|---|---|
| KN 4-token, text only | 5.168 | 2.8075 | −0.039 | +0.029 | −0.051 |
| code L14, text only | 4.904 | 2.8073 | −0.039 | +0.025 | −0.051 |
| code L14, ρ=0.5 | 4.706 | 2.8086 | −0.037 | +0.021 | −0.048 |
| code L28, text only | 4.575 | **2.8004** | **−0.046** | +0.010 | −0.055 |
| code L28, ρ=0.5 | 4.341 | 2.8007 | −0.045 | +0.004 | −0.054 |
| merged state R4 + token (no donor in tile) | 5.069 | 2.8112 | −0.035 | +0.025 | −0.046 |

The tuned donor weight was about 0.90–0.93 in every blend.

**State merging, no donor anywhere.** Each round merges (previous state, token) keys with
at least 8 occurrences by the similarity of their next-token rows. Rare keys fall back to
their token's state. K = 4096.

| round | keys merged | held key coverage | state alone | state + token |
|---|---|---|---|---|
| 1 | 14,487 | 53% | 5.046 | 5.070 |
| 2 | 15,648 | 49% | 5.047 | 5.065 |
| 3 | 15,266 | 47% | 5.048 | 5.065 |
| 4 | 15,204 | 47% | 5.051 | 5.069 |

For reference, KN with a 2-token context scores 5.089 and the L28 code key with text-only
counts scores 4.575.

## Findings
1. **Every tile adds to the donor, but none clears the bar.** The best tile, the L28 code
   table with text-only counts, gains 0.046 nats against the 0.05 target. It stores 1.4M entries
   and needs no gradient descent.
2. **The gain is verbatim memory of the box, not new generalization.** Even the plain
   4-token count table gains 0.039, and every gain is concentrated where the text has
   attestation (−0.05 on bigrams seen 3+ times). On unseen bigrams every tile makes the blend
   slightly worse (+0.004 to +0.029). The donor already knows what the code table generalizes,
   so in the blend the capped donor prior adds nothing. The tile's job is to remember what is
   specific to this box.
3. **The better key helps a bit.** The L28 key is 0.007 nats better than plain counts and
   also hurts unseen bigrams least. The L14 key, the realistic gate tap, adds nothing over
   plain counts in the blend.
4. **State merging, in this form, does not grow a useful memory.** One round gains 0.04 over
   KN 2-token counts, and further rounds add nothing. Only about half of held keys are covered,
   so most positions fall back to their token's state and the recurrence never builds longer
   history. This is a negative result for this simple version, not for the idea: it starves on
   0.8M tokens and a count-threshold merge.

## Assessment
This is the cache / kNN-LM effect at a small scale, and kNN-LM gains are known to grow with
datastore size. Our datastore covers 0.8M of the box's 2M fit words, about 40%. The tile route
is on track if that scaling holds here. The bottom-up crystal route has not yet produced a key
that beats the donor's.

---

# Two papers → hidden states → mix → inject into Qwen3-0.6B (2026-11-25 20:50, GPU 20 s)

**Papers.** Two held fluids-box introductions, chosen as the most similar pair by TF-IDF:
- cb0240527, autoencoder reduced-order models of wakes: 1,486 tokens.
- cb0240685, neural-network generalization for fluid flows: 1,412 tokens.

They share many citations (Murata et al., Srinivasan et al., Milano & Koumoutsakos, Fukami et al.).

**Script.** `kv_mix.py`, run on Qwen3-0.6B-BF16-microscaling-source, mlx-lm-unified `1104ced`.
Greedy decoding with thinking off; the 10 questions are in the script, and every answer is
in `kv_mix_answers.json`. The "hidden states" are each layer's KV cache. Arms:
- **none:** no papers.
- **text:** both papers as ordinary text in one prompt.
- **parallel:** each paper encoded alone, so neither ever attends to the other. The caches are
  concatenated, and paper 2's keys are RoPE-shifted to sit after paper 1. The only mixing is the
  question attending over both.
- **pooled:** each paper's cache mean-pooled into 128 slots (about 11× compression), with keys
  re-rotated to the slot positions.
- **blended:** slot j is the average of paper 1's slot j and paper 2's slot j, so 128 slots hold
  both papers fused.

The first attempt stopped at its own RoPE self-check, which was a bug in the test, not in the
shift. The fixed check passes at 8e-7.

**Grading.** I graded every answer against the papers: 2 = correct and specific to the papers,
1 = partly right, 0 = wrong, invented or generic. This is a single reader and not blind.

| Q | topic | none | text | parallel | pooled | blended |
|---|---|---|---|---|---|---|
| 1 | linear AE ≡ POD | 0 | 0 | 1 | 0 | 0 |
| 2 | shared citation for activations (Murata, laminar cylinder wake) | 0 | 1 | **2** | 0 | 0 |
| 3 | Srinivasan (LSTM, nine-equation shear flow) | 0 | 2 | 2 | 0 ("social media") | 0 ("social media") |
| 4 | Wang LSTM vs Hasegawa CNN-LSTM | 1 | 1 | 1 | 0 | 1 |
| 5 | generalization findings and limitations | 0 | 1 | 1 | 0 | 0 |
| 6 | first AE in fluids (Milano & Koumoutsakos) | 0 | 2 | 2 | 0 ("UCSD") | 0 ("UCSD") |
| 7 | why CNN-AE; CNN tasks in paper 2 | 0 | 2 | 2 | 0 | 0 |
| 8 | hierarchical AE vs open questions | 0 | 1 | 1 | 0 | 0 |
| 9 | linear tools (POD, DMD, resolvent) | 0 | 2 | 2 | 0 | 0 |
| 10 | Tang drag reductions (5.7–38.7%) + laminar limit | 0 | 2 | 2 | 0 ("15–20%") | 0 ("15–20%") |
| | **total / 20** | **1** | **14** | **16** | **0** | **1** |

## Findings
1. **Papers encoded separately and mixed only at question time work as well as reading them
   together** (16/20 vs 14/20). The model connected facts across the two papers, such as which
   citation they share, even though neither paper's states ever saw the other. This is the useful
   result for ENS: tiles can be encoded independently and combined at query time through attention.
2. **Averaging hidden states destroys the content.** Chunk-mean pooling to 128 slots per paper
   scored 0/20, and averaging the two papers into one memory scored 1/20. Both are no better than
   no papers at all. They are also worse in kind: for Srinivasan they produced confident off-domain
   answers ("social media and mental health", "UCSD") that the no-paper arm did not.
   Mean-pooled keys and values are not a memory the model can read. They act as noise that knocks
   it off its priors.
3. **Mixing works through attention, not arithmetic.** Concatenation keeps each token's state
   intact and lets the question choose. Averaging blurs states that were never meant to be
   combined. Compressing a paper to a few slots would need learned compression, such as trained
   gist or memory tokens, not averaging.
4. **Caveats.** A 0.6B model, one greedy run per cell, one paper pair, and grading by me. The
   text arm also misattributed some facts between the papers. Small models often do.

---

# Thinking vs no-thinking: logits and neuron firing (Qwen3-0.6B, GSM8K, 2026-11-25 ~21:30)

**Setup.** `reason_probe.py` + `reason_analyze.py` on Qwen3-0.6B-BF16-microscaling-source,
the first 80 GSM8K test problems from `benchmarks/gsm8k/test.jsonl`, greedy decoding. The user
message is identical in both modes ("Give only the final answer as a number").
- **No-think:** the template's empty think block, then a forced "The answer is ".
- **Think:** greedy thinking, capped at 1,800 tokens, then a forced "\n\nThe answer is ".
  22 of the 80 traces hit the cap and were force-closed.

A teacher-forced capture pass re-runs every sequence and records, at every position, the
logits, all 28 layers' MLP activations (silu(gate)·up, 3,072 neurons), residual norms, and a
logit lens at the answer token. The capture's self-check against `model()` gives a max logit
difference of 0.

**Attempts.** The first two runs failed: a bf16→numpy crash, then a design flaw where
"no-think" still wrote out its working in the answer and was graded truncated. The final run
(r3, GPU 21:27:32–21:29:11) reuses the saved greedy thinking traces from run 2 and forces an
immediate answer in both modes.

**Accuracy.**
- No-think: 4/80 (5%). Its first digit is right 25% of the time, the same as always guessing
  "1" (26%).
- Think: 62/80. That is 52/58 (90%) for traces that closed on their own and 10/22 (45%) for
  force-closed ones; the median closed trace is 673 tokens.

The group where reasoning helps (think right, no-think wrong) has 49 problems, counting only
closed traces.

## What differs

1. **The answer moment is a different act.**
   - After thinking, the answer token has entropy 0.02 nats and a top-1 probability of 0.997:
     the model is copying a number it already wrote.
   - Without thinking, the same token has entropy 1.25–1.44 nats and a top-1 probability of
     0.5–0.6. That is a guess.

2. **No-think's confidence carries no information; think's does.**
   - No-think's answer entropy does not separate right from wrong (AUC 0.50).
   - After thinking, entropy predicts correctness (AUC 0.77). Answers with entropy below 0.2
     (88% of problems) are 84% correct; the rest are 30% correct.

   So the logits flag a failed reasoning chain, but they can't flag a failed direct guess.

3. **No evidence of hidden knowledge without thinking.** In the logit lens on wrong no-think
   answers, the correct first digit is top-ranked at the last layer in only 17 of 76 cases.
   That is about chance for a first-digit guess. In the middle layers (11–22) the correct digit
   is never consistently ahead. The model is not computing the answer silently and then losing
   it; without thinking, it never computes it.

4. **The two modes differ in the middle layers, not the early or late ones.** At the answer
   token of the same problem, comparing the two modes:

   | layers | MLP-activation cosine | residual cosine |
   |---|---|---|
   | 1–8 | 0.91–0.97 | about 0.97–0.99 |
   | 12–20 | 0.49–0.64 | down to 0.78 |
   | 24–28 | about 0.86 | about 0.88 |

   The pattern is the same for problems that both modes get right. So layers 12–20 reflect
   *what is in context* (a worked solution versus the bare question); they do not track
   correctness. They are where the model reads the trace, while early layers encode the token
   and late layers format the number.

5. **Thinking spreads its firing out; answering concentrates it.** The share of activation
   mass in the top 1% of neurons is 0.075 during thinking, against 0.093–0.098 at answer tokens
   and 0.087 on the prompt. Mean firing during thinking is lower in layers 1–12 and higher at
   layer 20. The most thinking-specific neurons sit in layers 21 and 28 (for example L21 #456
   and L28 #2097, with think/answer activation ratios of 2.1× and 3.0×, as natural logs of
   2.08 and 2.97).

   The phase profiles still have cosine 0.88–0.97, so thinking does not recruit a separate
   circuit; it reweights the same neurons.

6. **A "reasoning neuron" predictor can't be tested here.** Only 4 no-think answers are
   correct, so the held-out AUCs (0.49–0.91) rest on 2 positives and mean nothing.

## Caveats
One model at 0.6B, one greedy sample per problem, 80 problems. Neuron identities are
correlational. Activation magnitude is |silu(gate)·up| and attention is not analyzed. The
force-closed traces are a mixture of completed and truncated reasoning.

---

# Thinking probe 2: run-on controls, "is it just search?", ablations vs random, selectivity (2026-11-25 21:45)

**Setup.** `reason_probe2.py` + `reason_analyze2.py` on Qwen3-0.6B-BF16-microscaling-source,
the first 64 GSM8K test problems, greedy decoding. Thinking runs under mlx2's `ThinkingGuard`
(mlx2 `089ac4e63`, `src/mlx2/thinking_guard.py`). The thinking budget is 2,400 tokens (it was
an 1,800 cap before); the guard keeps its defaults (soft release at 80%, run-on CUSUM tau 2.5,
6-gram, 2-nat ramp) and Qwen3 declares no nudge.

Every arm is answered after a forced "The answer is ". Neuron sets come from the previous run
(`neuron_sets.json`), layers 12–28 only:
- **sel:** think-selective, by the thinking vs direct-answer activation ratio.
- **act:** the highest-firing neurons while thinking.
- **rnd:** random, excluding sel256.

The sel64 and act64 sets overlap by only 2.7%. Ablation zeroes the neurons' MLP activations.
GPU ran 21:45:44–21:51:24, peak 2.0 GB, no swap growth. Two self-checks run before anything
else: the manual forward against `model()` (within one bf16 step, same argmax) and attention
summing to about 1.

## Run-on controls
- **The budget never binds.** The run-on alarm fires first on 42 of 64 traces, at a median of
  481 tokens and as early as 155. The other 22 close on their own.
- **The alarm costs accuracy.** On the 42 tripped problems, accuracy falls from 67% (unguarded,
  median 1,735 thinking tokens) to 45%. Problems that close naturally stay at 95%.
- **Overall:** 62.5% against 76.6% unguarded on the same problems.

This confirms the note in `spiral_conclude.py` that the server-side alarm "trips far too early
on ordinary math reasoning", here on Qwen3-0.6B. A longer budget or context changes nothing
while the alarm fires first.

## Is the gain just extra compute or search space? No: it is written content.

| what preceded "The answer is" | accuracy | answer entropy |
|---|---|---|
| true no-think | 6.2% | 1.37 |
| empty think block | 4.7% | 1.51 |
| same-length dots | 4.7% | 1.92 |
| another problem's real trace | 0.0% | 0.47 |
| first 25% of the trace | 10.9% | 1.20 |
| first 50% | 26.6% | 0.83 |
| first 75% | 53.1% | 0.45 |
| full trace, sentences shuffled | **62.5%** | 0.49 |
| full trace | **62.5%** | 0.40 |

- **Extra token positions alone buy nothing:** dots score the same as an empty think block.
- **A generic "reasoning mode" buys nothing:** another problem's trace scores 0%, and the model
  confidently answers with that problem's numbers in 45% of cases.
- **Order doesn't matter at answer time:** shuffled sentences score exactly as well as the
  original. The answer is read out of the computed numbers in the trace.
- **Accuracy grows as the result gets written:** problems the full trace gets right are already
  right at 15% / 40% / 82% of the trace.

So for the readout, thinking works as a scratchpad the model computes into and then copies
from. Generating the trace still needs ordered reasoning; shuffling only tests reading it.

## Confidence as a predictor
1. **Full-trace entropy is a strong correctness signal** (AUC 0.94), and it is calibrated:

   | entropy | problems | accuracy |
   |---|---|---|
   | < 0.01 | 21 | 95% |
   | 0.01–0.05 | 15 | 87% |
   | 0.05–0.2 | 8 | 75% |
   | 0.2–0.7 | 4 | 25% |
   | ≥ 0.7 | 16 | 0% |

2. **It enables early exit.** Reading out at 75% of the trace, 32 of 64 answers are confident
   (entropy < 0.05). Those are 94% right and always match the full-trace answer, which would
   save 25% of thinking tokens on half of the problems. At 50%: 10 confident, 90% right, also
   always matching.
3. **It measures "found a number to copy", not truth.** With another problem's trace, 45% of
   problems get a confidently wrong answer. With no trace, confidence is useless (AUC 0.53).
4. **Attention gives a second signal.** At layer 20, the best head puts 29% of its attention on
   the gold number's tokens in the trace, and that attention predicts correctness (AUC 0.92).

## What do the thinking-selective neurons contribute, against random?

| ablated during readout (trace fixed) | accuracy | right→wrong | Δ gold first-token logprob |
|---|---|---|---|
| none | 62.5% | – | – |
| sel64 (think-selective) | 64.1% | 0 | −0.08 |
| rnd64 × 3 seeds | 60.9–62.5% | 0–1 | −0.06 to +0.02 |
| sel256 | 64.1% | 0 | −0.03 |
| rnd256 | 64.1% | 0 | −0.12 |
| **act64 (high-firing)** | **29.7%** | **22** | **−1.60** |

- **Reading the trace:** think-selective neurons are no more needed than random ones. The
  high-firing core is essential.
- **Generating the trace** (20 problems, thinking itself generated under ablation):
  - sel64: 45%, median 443 tokens
  - rnd64: 30%, median 423 tokens
  - unablated: 45%, median 442 tokens

  At n = 20 that is noise-level, but removing think-selective neurons does not hurt reasoning
  either. They correlate with thinking (their mean digit-vs-word log ratio is −0.19, leaning
  slightly toward word tokens) rather than doing its work.

## How selective is thinking? Not very.
- **Participation ratio** (1 = all neurons equally used) while thinking matches prompt reading
  layer by layer, about 0.84 at layer 1 down to 0.02 at layer 28. Late layers are highly
  concentrated in both modes.
- **Cross-problem overlap** of each problem's top-1% neurons is higher while thinking in layers
  24–28 (Jaccard 0.58–0.75 against 0.42–0.70 for prompt reading). Late-layer thinking uses a
  more fixed, problem-independent set.
- **Digit- or operator-selective neurons are rare**: at most 2.5% of a layer, mostly late layers.
  Word-selective neurons reach 8.9% only at layer 1.
- **Attention routing** at the answer token: layers 12–20 read the trace (22–40% of attention),
  layer 20 carries the strongest copy head, and layers 22–28 return to the prompt (70–75%).

**Conclusion.** Thinking is not a separate circuit or a large search over hypotheses. The same
high-firing core writes intermediate results into context, and middle-layer copy heads
(around layer 20) read the result back out. The extra reach comes from written intermediate
state, not from extra compute positions or special neurons.

## Caveats
One model at 0.6B, 64 problems (20 for ablated generation), greedy decoding, zero-ablation
only, and the readout scored at a forced answer. The guard's early trips cut traces and reduce
the number of correct cases.

# 8. Second family: Gemma 4 26B-A4B (2026-11-26)

**Setup.** I ran `gemma_probe.py` and then `gemma_analyze.py`.
- **Checkpoint:** `gemma-4-26B-A4B-it-MLX-8bit` (config sha256 `ea476b39447d`). It was loaded from an NVMe-staged copy with sizes and mtimes preserved; the copy is deleted now.
- **Code:** mlx-lm-unified `1104ced`. The guard is mlx2 `src/mlx2/thinking_guard.py`, last changed at `fbf240e8f`, run from mlx2 HEAD `62f426c5a`.
- **Data and decoding:** the first 48 GSM8K test problems, greedy.
- **Model:** 30 layers. Each layer has a dense GeGLU MLP (2,112 wide) plus a 128-expert MoE that picks 8 experts.
- **Hooks:** they wrap each layer's `mlp` and `router` in place, so `model()` runs its own code.
- **Guard:** budget only, 1,600 tokens, with the run-on alarm off. The budget never bound, and all 48 traces closed naturally (median 394 tokens, p10 182, p90 1,288).
- **Neuron and expert sets:** chosen on even problems from layers 13–30 and ablated on the 24 odd problems, so the ablation results are held out.
- **Not run here:** attention-head analysis and ablated generation.
- **GPU window:** 04:25:40–04:35:14, peak 26.6 GiB. The lease recorded a swapout delta of 5.6 M pages, and total swap grew from 11 to 16 GB during the window. That is not a quality issue, but this run is not a clean performance window.

## Same patterns as Qwen3-0.6B, more sharply

| what preceded "The answer is" | Gemma 26B-A4B | Qwen3-0.6B (probe 2) |
|---|---|---|
| true no-think | 58.3% | 6.2% |
| empty think block | 52.1% | 4.7% |
| same-length dots | 52.1% | 4.7% |
| another problem's trace | **6.2%** | 0.0% |
| first 25% of trace | 75.0% | 10.9% |
| first 50% | 97.9% | 26.6% |
| first 75% | 100% | 53.1% |
| full trace, sentences shuffled | **97.9%** | 62.5% |
| full trace | **100%** | 62.5% |

1. **Again it is written content, not compute.** Dots equal an empty block, and both are slightly below true no-think. Another problem's trace drives accuracy down to 6%, with 71% of problems confidently wrong (Qwen: 45%). Shuffling sentences costs nothing.
2. **The result is written early.** Half the trace already gives 98%. Of the 20 problems no-think gets wrong, 10 are already right at 25% of the trace.
3. **Confidence works as an early-exit signal, and it is again fooled by a foreign trace.**
   - At 50% of the trace, 44 of 48 answers are confident (entropy < 0.05). All 44 are right and equal the full answer.
   - At 75%: 48 of 48, all right.
   - For entropy as a right/wrong signal, AUC is 0.94 at cut50 and 0.89 for no-think. The full trace is at 100%, so its AUC is undefined.
   - This is a ceiling regime. A harder set is needed to calibrate full-trace confidence on this model.
4. **Neurons: the same split.** On held-out problems:
   - The think-selective dense neurons (sel64, 64 per layer, layers 13–30) do nothing: 100% against 100%.
   - Two random sets also give 100%.
   - The high-firing core (act64) collapses accuracy to **4.2%**, with 23 of 24 right→wrong (Qwen: 29.7%).
   - sel64 and act64 overlap by only 3.4%, the same disjointness as in Qwen (2.7%).
5. **Experts: highly redundant.** Zeroing the 8 most-used experts per layer while thinking (eact8), the 8 most think-selective ones (esel8), or 8 random ones, in layers 13–30, leaves held-out accuracy at 100%. The router's other 7 picks per token cover for each one removed.
6. **Thinking is not a separate circuit.**
   - The dense participation ratio while thinking tracks prompt reading layer by layer (0.15 / 0.56 / 0.31 / 0.18 / 0.14 / 0.16 / 0.27 at layers 1/6/11/16/21/26/30).
   - Each problem's top-1% dense neurons overlap heavily across problems in both phases: Jaccard 0.59–0.79 while thinking, 0.71–0.84 while reading the prompt. Thinking is slightly *less* fixed than prompt reading here, whereas Qwen's late layers were more fixed while thinking.
   - Expert usage entropy is nearly identical in both phases (3.71 vs 3.75 nats, max 4.85), and so is the cross-problem expert top-8 overlap (0.54 vs 0.57).

**Conclusion across both families.** A 0.6B dense model and a 26B 8-of-128 MoE behave the same way. Thinking writes intermediate results into the context, and the answer is copied out of them. That copy does not depend on order, it is confidently misled by a foreign trace, and it is mostly done by mid-trace. A small high-firing dense core is essential; think-selective neurons and individual experts are not. Neither model shows a dedicated thinking circuit or a broad search.

**Caveats.**
- One run with 48 problems, and 24 for the ablations.
- Gemma is at ceiling with the full trace, so ablations can only show harm.
- Zero-ablation only, and readout at a forced answer.
- Gemma ran without the run-on alarm, while Qwen probe 2 ran with it, so think-length effects are not compared.
- The act64 collapse may be generic high-magnitude damage rather than anything specific to thinking. A no-think arm with act64 ablated would separate the two; it was not run.

## 9. Gemma follow-ups (2026-11-26): the no-think control, and a hard set that would not finish

**Run.** Same checkpoint as section 8: gemma-4-26B-A4B-it-MLX-8bit (staged NVMe copy, config sha256 ea476b39447d), mlx-lm-unified 1104ced, thinking guard from mlx2 fbf240e8f. Probe script `gemma_probe.py` (new `--nothink-control` mode, untracked, beside this README; mlx2 HEAD 4ff940143). Night unit 22:49–22:52 under gpuq lease `ens-donor-20261125-ens-gemma-night-20261126`.

### No-think control: act64 damage is generic, not specific to reading a trace

The same 48 problems as section 8, answered with **no thinking at all**, with each of section 8's sets (chosen on even problems' thinking) zeroed. Unablated no-think reproduces section 8 exactly (48/48 identical answers). Output: `gemma_ntctl.json`, `gemma_ntctl.analysis.txt`.

| zeroed (layers 13–30) | held-out odd, no think (n=24) | same set, with full trace (section 8) |
|---|---|---|
| nothing | 50.0% | 100% |
| sel64 (think-selective dense) | 50.0% (0 right→wrong) | 100% |
| **act64** (high-firing dense) | **0.0%** (12 of 12 right→wrong) | 4.2% |
| rnd64 ×2 | 58.3%, 41.7% | 100%, 100% |
| esel8 (think-selective experts) | 58.3% | 100% |
| **eact8** (most-used experts) | **25.0%** (6 right→wrong) | 100% |
| ernd8 | 45.8% | 100% |

Even problems show the same: act64 66.7% → 4.2%, eact8 66.7% → 12.5%.

1. **act64 is a general-purpose core, not a reasoning circuit.** Zeroing it destroys answering whether or not the model has thought. Section 8's caveat is resolved: the collapse is generic damage. So in Gemma no neuron set tested so far is specific to thinking. The think-selective set does nothing with or without a trace.
2. **The most-used experts matter only when there is nothing to copy.** eact8 is harmless with a trace (100%) but halves no-think accuracy. With a trace in context the answer is copied, and the router's other picks cover for the removed experts. Without a trace, the answer has to be computed, and the heavily used experts do part of that work. Expert redundancy (section 8, point 5) holds for copying, not for computing.

### Hard set: still incomplete

`gsm8k_hard64.jsonl` holds 64 GSM8K test problems (index ≥ 48, the ones with the most `<<…>>` steps, 6–8). 49 thinking traces are saved (`gemma_hard2.traces.json`, median 1,926 tokens). Every attempt to read them out was stopped by the +3 GB swap guard:
- 08:10 and 09:30: after about 19.5 min, after phase A.
- 12:06, reusing the saved traces: 61 s into phase B.
- 22:51, also reusing them: about 66 s into phase B.

The last two stopped at the same point, while readouts for problem 5 were running, with a jump of about 4 GB within 10 s on otherwise flat swap and about 40 GB of RAM free. That looks deterministic, not like host noise. Trace lengths there are not unusual (1,923 tokens; problems 1, 2, 4, 7 and 8 are just as long).

What the 6 completed readouts show (`gemma_hard3.json`, `gemma_hard4.json`):
- full trace 6/6, no-think 1/6, another problem's trace 0/6, half the trace 5/6
- held-out odd problems: act64 0/3, sel64 3/3

This is the same pattern as section 8, and this set does leave room below ceiling for no-think. **The no-think control cannot be run on this set:** no-think is already wrong on every odd problem, so the ablation has nothing to damage. That is why the control above runs on the easy set.

**Caveats.** One run per arm, and zero-ablation only. In the control the sets come from section 8's selection (thinking on even problems), so odd problems are held out with respect to problems, not with respect to thinking-free answering.
