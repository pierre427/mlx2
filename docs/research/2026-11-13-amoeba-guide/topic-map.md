# The topic map: what each model knows, and where it keeps it

Date: 2026-11-14. Companion to [overview.md](overview.md). Design and the exact rules are in
[docs/research/amoeba-topics/DESIGN.md](../amoeba-topics/DESIGN.md). The files are in
[docs/research/amoeba-topics/](../amoeba-topics/).

**Status.** This is research. Nothing here is qualified or used in serving. Every number names its
checkpoint and the mlx2 commit in the file it comes from.

## 1. Why a map

Amoeba splits a model into an English core and topic boxes, and later moves boxes between models.
Doing that topic by topic needs three things:

1. A list of topics that every model shares, with names that never change. That is the **taxonomy**.
2. For each model, which neurons carry which topic. That is the **registry**.
3. A way to cut a host model's own neurons for a topic, so that a stronger donor's box can be placed
   where the host kept that topic. That is the **stub**.

## 2. The taxonomy

`taxonomy.v1.json` (version 1.3) has 8,386 topic IDs: 466 written by hand, plus every ISO 639-3
language code.

- **Programming:** each language, such as `prog.python`, `prog.shell.zsh`, `prog.perl`, `prog.web.css` and
  `prog.gpu.metal`, with Python sub-areas.
- **Software engineering:** git, testing, CI, packaging, licences and similar.
- **Inference stack:** `ml.inference.kv_cache`, `ml.inference.quantization.microscaling`, speculative decoding,
  kernels, serving, mlx and mlx2.
- **Mathematics:** following MSC 2020 and arXiv.
- **Sciences, humanities, social sciences and engineering:** following CIP 2020, the US Department of
  Education scheme that Open Syllabus's 69 fields come from.
- **Cybersecurity:** built on NIST CSF 2.0 (Govern, Identify, Protect, Detect, Respond, Recover and their
  categories), plus CTI, vulnerability management, offence, defence, DFIR, GRC and security domains.
- **Text form**, such as children's stories, digits and markdown structure, and **in-context skills**.
- **Audience and register (v1.3):** debate, Socratic dialogue, encyclopedic and textbook prose, and reading
  levels from young children to researchers.

**Cross-references.** Besides CIP, MSC, arXiv and Linguist, nodes carry Dewey Decimal class numbers
(`ddc`, numbers only) and Usenet group names (`usenet`, such as `comp.lang.python` or `sci.logic`), at
Pierre's suggestion. They help most for register and audience, which the subject schemes leave out.

**Rules.** An ID such as `sci.physics.quantum` is permanent. New topics bump the minor version. Old ones
are deprecated, never deleted, and the builder refuses to drop an ID.

**Open Syllabus Galaxy** turned out to be a map of assigned *books*, with no public field list,
download or API. We use CIP, its source scheme, instead.

## 3. How a cluster gets a label

The probe corpus (v4, used for the labels below) has 17,694 short texts over 428 tested topics. v5
(2026-11-16, `topic-probes-cb94e8725`, v5.2) keeps all of them and adds Cosmopedia, PG-19 and Wikipedia text
for 47,500 texts over 503 tested topics, at 64 fit / 32 held per topic where the source allows. Cosmopedia
is model-generated, so 65 topics whose held text is mostly Cosmopedia carry a `held_synthetic` flag. All but
the downloads are from local sources:
- code from installed packages and system files;
- mlx2 and mlx-uag docs;
- macOS interface strings;
- the downloads Pierre approved: FLORES+ (213 languages), an arXiv sample, and the security set (KEV,
  NVD, ATT&CK, CWE, NIST SP 800-53, Sigma).

Every text is assigned to *fit* or *held* by its source file. Nothing chosen on fit is reported on fit.

For each cluster and topic, the labeller asks: does the cluster fire more on this topic's texts than
on the texts of its siblings? A sibling of Python is C, for example, not prose. The label has to beat
two controls:
- **random twins:** 32 neuron sets of the same size, layer by layer;
- **a shuffled-label null:** counted in source files, not texts, and corrected for every test made.

The held split has to confirm the result. The label is the finest topic that passes.

**The functional check** follows: remove the cluster from the full model, or add it to the core, and
see whether that topic's text gets harder or easier to predict, compared with removing or adding
random twins.

## 4. Qwen3.8-27B (`Qwen3.8-27B-BF16-q8-mlx2@2dd8d304`)

**Clusters.**
- **Before:** 11 of the 64 unnamed clusters had a label, and 53 showed nothing on six task probes.
- **Now:** **42 of 64 pass selectivity.** All 64 were tested functionally on their first label, and
  **23 pass**. Examples:

| Cluster | Topic | Removing it costs that topic, beyond its twins |
|---|---|---|
| C22 | (capitals cluster, from the boxes run) | +0.45 nats |
| C60 | French family (French, Arpitan, Mauritian and Haitian Creole) | +0.29 |
| C44, C32 | digits and arithmetic | +0.15, +0.11 |
| C43 | YAML, Sigma detection rules, deployment configs | +0.12 |
| C19 | relativity, PDEs, mathematical physics | +0.12 |
| C46 | plasma and planetary physics | +0.09 |
| C62 | children's stories | +0.08 |
| C21 | web templates | +0.07 |
| C28 | linear algebra | +0.07 |
| C24, C31 | physical chemistry; biology | +0.04, +0.03 |
| C29, C5 | CWE weaknesses; threat actors | +0.04, +0.03 |

**Selective but not needed.** Some clusters are selective yet fail the functional check: C0 (history),
C16 (CWE), C23 and C49 (arithmetic), and most of the broad language clusters. They fire on the topic,
but the model does not need them for it. In the registry they become core tags, with that reason.

**Shared neurons.** C1 fires on Chinese, Malay and physical units. Per neuron:
- **Chinese and Malay share a population.** 948 neurons are selective for both, against 398 if the
  two were independent, and the per-neuron selectivities correlate at r = 0.73. The model packs them
  side by side in the same neurons.
- **Units overlap only partly.**

Shared neurons get weighted memberships, and a stub *copies* them, never moves them.

**Languages and scripts.** Language-labelled units come in three kinds:
- **single-language:** 80;
- **family:** 12, for example Romance, Cyrillic and French-based creoles;
- **multi-script:** 43. These fire together on unrelated rare scripts (Chakma, Ol Chiki, Tibetan,
  Ethiopic, N'Ko), so they are probably byte-fallback token units. They are flagged as that, not
  read as knowledge of the language.

**The puzzle.** Neurons in the 22 unlabelled clusters were placed by what they fire together with.
Six rounds placed 35% of them, and every round held on the held split (98–100%). Over the whole model:

| Part | Share of all neurons |
|---|---|
| English core | 53.7% |
| Labelled pool, including 5.4% placed by the puzzle | 36.5% |
| Shared between topics | 0.8% |
| Unexplained, kept in the core with their co-activation profile | 9.8% |

**The registry rows** (topic → model → neurons):

| Status | Topics | Meaning |
|---|---|---|
| cluster | 33 | a verified cluster carries the topic |
| box | 3 | carved boxes: capitals, arithmetic, French |
| core_resident | 290 | neuron maps inside the core, each with the reason it was not split |
| cluster_unverified | 99 | units exist but the window budget did not reach them |
| subsumed | 11 | a finer topic carries it |
| absent | 42 | nothing above chance |

**Caveat: code-like topics are core-resident partly by construction.** Programming languages, git,
KV cache, quantisation, mlx2, KEV and quantum physics are mostly core-resident: their selective
neurons sit inside the core. But the 27B core was carved to keep repo-style markdown cheap, so this
is partly by construction. Those 117 rows carry `core_carving_corpus: includes repo_markdown`. A
prose-only core would be needed to see them as separable boxes. That is the next corpus, below.

**Gaps.**
- The probe corpus has no capitals texts. That is why C22 first labelled as Aymara; its capitals role
  comes from the boxes run.
- Many topics stay untested for lack of probe texts.

## 5. Qwen3-0.6B (`Qwen/Qwen3-0.6B@c1899de2`), CPU

- **The full pipeline runs on the CPU here.** Its clusters were formed on the probe corpus itself, so
  "32 of 32 labelled" is optimistic.
- **It finds language families without labels:**
  - Germanic (German, Norwegian, Danish);
  - Iberian Romance;
  - West Slavic;
  - Chinese, Japanese and Korean.
- **Removing them matters.** Removing the Germanic cluster costs German 1.28–1.38 nats more than a twin
  does, and off-topic text moves by 0.02 at most.
- **Registry:** 49 cluster, 257 core-resident, 96 unverified, 10 subsumed, 66 absent.

**Stub, German.** 169 of 559 German neurons carry 90% of the removal cost. Removing those 169:
- costs German +1.33 nats over a matched twin;
- costs off-topic text −0.005;
- is strongest in layers 14–27.

The split checks hold: coverage is unchanged, no neuron sits in two units, and the masks are exact.

## 6. Donor screen (FN against 27B)

**The question.** For which topics does FN (`Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP@25c2da42`)
predict text better than the 27B? Those topics are where a donor box could help the 27B.

**How it was measured.**
- **Texts:** 5,419 texts of up to 64 tokens, at most 12 per topic, from the probe corpus.
- **Score:** prediction cost in nats per byte, from each text's third token on.
- **Two conditions:** bare, and after a neutral prefix ("Here is a passage.").
- **Statistics:** bootstrap intervals over source files, and a Benjamini–Hochberg FDR correction over all 492 topics.
- **Where each side ran:**
  - 27B: in-process, during window B.
  - FN: on the production service on :8282, through its prompt-logprob option, with no pause.
- **Same tokens:** the two models share a tokenizer, so both see the same token ids.
- **Files:**
  - `donorgap-host-qwen38-27b-4a90b0cf9/`;
  - `donorgap-fn-server-6bf72b77d/`;
  - `donorgap-compare-3a6cfd92e/donor_gap.json`.

**A model-level offset, removed first.**
- **Overall,** FN is slightly cheaper than the 27B: 0.009 nats/byte bare, 0.058 with the prefix.
- **The prefix effect:** "Here is a passage." costs the 27B 0.045 nats/byte, and FN only 0.006.
- **Not a scoring artefact:** on Qwen3-0.6B, joining prefix and text as token ids or as text gives the same
  cost (1.116 both ways).
- **Why it matters:** the offset is the same for every topic. Without removing it, 92 topics look like FN
  wins, mostly from the prefix condition.
- **Centred results:** each topic's gap minus the whole-corpus gap. A topic counts only if it is beyond
  the 0.005 margin in both conditions, after FDR.

**FN better, beyond the overall gap (16 topics)**, by nats/byte in the bare condition:

| Area | Topics |
|---|---|
| Security | `swe.security` +0.23, CTI attack techniques +0.15, CWE +0.11, CSF Recover / recovery plans +0.10 |
| Shells and markup | fish +0.15, markup +0.07 |
| Maths word problems | `math.word_problems` +0.16 |
| Low-resource languages | Aymara +0.25, Pangasinan +0.15, Hausa +0.13, Fijian +0.13, Jingpho +0.13, Tok Pisin +0.11, Welsh +0.08 |

**27B better (35 topics):**
- **Inference stack:**
  - Metal −0.12;
  - prefill −0.08;
  - draft-model speculation −0.08;
  - inference memory −0.07;
  - MoE −0.07;
  - pruning −0.06.
- **Software engineering:** packaging −0.06; git commits −0.04.
- **Arithmetic:** addition −0.07.
- **Children's stories:** −0.03.
- **About twenty languages and scripts,** including:
  - Chakma −0.24 and CJK −0.22;
  - Makhuwa, Tamasheq, Kamba, Sena, Kimbundu and Standard Moroccan Tamazight;
  - Hindi, Telugu, Maithili, Odia and Turkish.

**Reading it.**
- **This is a screen, not a transfer result.** It says where FN predicts the text better. It does not say
  that a box moved from FN would carry that skill.
- **Quantisation differs:** FN is a 4-bit MoE, and the 27B is q8 dense.
- **Security and the 27B's own map:** CWE and threat actors already have 27B clusters that pass
  (C29, C5), and FN is still better on them. They are the first candidates for a donor box.
- **Inference-stack prose is the 27B's strength.** It is mostly core-resident in the 27B (see §4).
- **Small groups:** 27 topics had fewer than 8 source files and were not tested. Most topics have
  12 groups, so a single topic's interval is wide.

## 7. What is next

- **English core corpus v2 (built, 2026-11-15):** graded prose from grade school through high school to
  postgraduate and literary, across registers, without repo markdown: 3,786 texts in 24 level × register
  cells. 3,201 come from public sources. 585 are creative texts (song, poetry, drama, letters, dialogue)
  generated by `Artemis-31B-v1.2-q8-mlx2@d347ab4d` on mlx2 `b1d6e5c5f` in one approved window: 600 prompts,
  0 rejected, 15 haiku sequences under the 400-char floor dropped, peak 51 GB, 70 min. They are labelled
  `model-generated`, carry source `artemis`, and stay withheld in research-artifacts; filter on the source
  to leave them out. They fill the cells the public-domain sources left empty or thin: letters at every
  level, L2 dialogue and drama, L3 song, L1/L2 poetry. The 27B session will carve nested cores from it.
- **Functional checks still outstanding:** the 99 unverified 27B units, mostly subclusters and core
  maps.
- **Deferred (Pierre: wait):** mapping the T7 models on the M3. Which models is still open.

## Words

**Carving** means choosing which of a model's own neurons form a unit. No weights change.
