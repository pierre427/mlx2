# Amoeba topic map: design note (2026-11-14)

Status: research, unqualified, off by default. Written before any capture. Session "Cross-model topic
taxonomy and cluster labelling", branch `claude/amoeba-topics-20261114`.

## Terms

**Carving** (Pierre, 2026-11-14) means choosing which existing neurons form a unit, whether by mask search (the core, boxes) or by ranking (stubs). No weights change, and the word "training" is reserved for weight changes; none happen in this work.
- **Gate fitting** sets a unit's few per-layer switch parameters.
- An **uncarved** unit is a plain union of clusters or units, for example a roll-up.

## Goal

One topic map that every amoeba model shares:
1. It labels the 64 unnamed knowledge clusters of `Qwen3.8-27B-BF16-q8-mlx2@2dd8d304`. 53 of them
   showed nothing on the six task probes.
2. It records, per model and topic, where that model keeps the topic: a box, a cluster, a stub, or
   neurons left in the English core.
3. A donor's box for topic X can then be remapped onto the host's own neurons for X (the stub), topic by
   topic, for FN next.

## Open Syllabus Galaxy: what it gives, and what we use instead

Checked 2026-11-14:
- Galaxy (galaxy.opensyllabus.org) is a node2vec → UMAP map of the citation graph of about 1.1 M
  frequently assigned *works* (books and articles). It is coloured by field, a field being one that holds
  more than 50% of a work's assignments.
- The page is a Next.js client that queries an undocumented backend (`galaxy-api-prod.opensyllabus.org`:
  `/fg/`, `/bg/`, `/detail/`, `/search-<type>/`). There is no public field list, no download and no API
  documentation. The Open Syllabus terms make anonymised metadata available "by inquiry", with
  restrictions.
- **Galaxy cannot be used programmatically**, and we do not scrape its private API.
- The Open Syllabus field taxonomy (69 top-level fields in dataset v2.14) is derived from the US
  Department of Education's **CIP** (Classification of Instructional Programs), which is public domain.
  The academic branches here (`sci`, `math`, `eng`, `hum`, `soc`) use CIP 2020 series as their
  backbone, and each node carries its `cip` code. A later Open Syllabus field list, if Pierre obtains
  one, joins through those CIP codes as aliases (`refs.os_field`) without renumbering anything.

Other schemes fill the gaps. Each node keeps the external code as a reference, never as its ID:

| Branch | Backbone | Licence of the scheme |
|---|---|---|
| `lang.*` | ISO 639-3 (all 7,9xx codes, from the pycountry 26.2.16 copy of Debian iso-codes) | ISO 639-3 code tables: SIL terms of use; iso-codes LGPL-2.1 |
| `prog.*` | GitHub Linguist language names (curated subset, names only) | MIT |
| `math.*` | MSC 2020 top-level classes, plus arXiv `math.*` | MSC: CC BY-NC-SA 4.0 (codes cited only); arXiv taxonomy: public |
| `sci.*`, `eng.*`, `hum.*`, `soc.*` | CIP 2020 two- and four-digit series, plus arXiv physics/q-bio/q-fin/stat | public domain / public |
| `cs.*`, `swe.*` | ACM CCS 2012 top-level concepts, plus arXiv `cs.*` | ACM CCS: free to use with attribution |
| `ml.*` | our own, with arXiv `cs.LG`/`cs.CL`, written for the inference stack | ours |
| `sec.*` (v1.2, Pierre) | NIST CSF 2.0 functions and categories (`refs.csf`, e.g. GV.OC, PR.AA, DE.CM), plus practitioner branches: CTI (ATT&CK, actors, IOCs, STIX/TAXII, TLP, lifecycle, Diamond), vulnerability management (CVE, CWE, CVSS/EPSS, KEV, patching, SBOM), offensive, defensive (SOC, Sigma/YARA, SIEM), DFIR, GRC (ISO 27001, SOC 2, 800-53, PCI DSS, privacy, risk registers, audit), and domains, with `refs.see` cross-links to `swe.*`, `cs.*` | NIST CSWP 29 (public domain) |
| (no `struct.*`) | structural roles stay untagged until the puzzle stage explains them (Pierre, 2026-11-14) | — |
| `form.*`, `skill.*` | our own: text form (markdown structure, story openings, digits) and in-context skills. v1.3 adds debate, Socratic dialogue, encyclopedic and textbook prose, an audience axis (`form.audience.{young_children,children,middle_school,high_school,college,researchers}`), and `skill.deduction`, `skill.entailment` | ours |
| all branches (v1.3, Pierre 2026-11-15) | `refs.ddc`: Dewey Decimal class numbers; `refs.usenet`: Usenet Big-8 / k12 / alt group and hierarchy names. Cross-references only, never the backbone. They add register and audience nodes that CIP and MSC lack. `hum.philosophy.logic` (DDC 160, sci.logic) and `.argumentation` (168) are new | DDC: OCLC's; numbers only, no schedule captions copied. Usenet names: public |

## Taxonomy: `taxonomy.v1.json`, schema `mlx2.amoeba-topics.v1`

- It is built by `scripts/amoeba_topics_taxonomy.py` from a curated spec in the same script, plus the
  ISO 639-3 table. It is deterministic, and it refuses to drop or rename an existing ID (see Versioning).
- **ID:** dotted lower-case path, `[a-z0-9_]` segments, such as `prog.python`, `prog.shell.zsh`,
  `sci.physics.quantum`, `math.linear_algebra`, `lang.fra`, `swe.git`, `ml.inference.kv_cache`. The ID is
  the key: never renumbered, never reused.
- **Node:** `{id, name, parent, depth, kind: branch|leaf, aliases[], refs{cip, arxiv, msc, iso639_3,
  linguist, acm_ccs, os_field}, since, status: active|deprecated, replaced_by, notes}`.
- **Language leaves:** `lang.<iso639-3>` for every code, with `type`
  (living/extinct/historical/constructed/ancient) and `scope` (individual/macrolanguage). Macrolanguages
  are branches whose individual languages point at them through `refs.macro`. Their IDs stay flat (`lang.cmn`,
  not `lang.zho.cmn`), so a later change to a macrolanguage grouping never renames anything.
- **Versioning:**
  - Adding a node bumps the minor version (`1.1`).
  - Deprecating a node sets `status: deprecated` and `replaced_by`; the node is never deleted.
  - Moving a node to a new parent is a deprecation plus a new ID.
  - The builder loads the previous file and fails if any old ID is missing.
  - Registry rows name the taxonomy version they were scored against.
- **Existing task labels map in:**
  - `capitals` → `soc.geography.world_capitals`
  - `arithmetic` → `math.arithmetic.addition`
  - `french` → `lang.fra`, plus `skill.translation`
  - `recall` → `skill.in_context_recall`
  - `english` → `lang.eng`
  - the atlas corpus sources: `tinystories_train` → `form.narrative.childrens_story`,
    `repo_markdown` → `prog.markup.markdown`

## Probe corpus

- **Per leaf, short held-out texts:** 64-token windows of the host tokenizer, the same length as the
  atlas capture.
- **Split:** each text is assigned to `fit` (choose and label) or `held` (report) by the parity of
  sha256 of its *group* (source file, bundle, book or FLORES sentence block). A group never feeds both
  splits, so a label is never reported on the text it was chosen on.
- **Target:** 64 fit and 32 held texts per leaf from v5 (v4: 24 and 24). Leaves with fewer than 16 texts in
  either split are `untested`.
- **v5 (2026-11-16, build `topic-probes-e8c22f31a`, 46,074 texts, 496 tested topics), CPU only, no new
  downloads:**
  - the v4 build (`16e8232ad`) is carried forward (`--carry`): every v4 text, label and split is kept and
    selected first, so atlases fitted on v4 fit rows stay inside v5 fit;
  - FLORES+ quota raised: all 209 language boxes have 64/32;
  - one box id per language: Apple UI strings' macrolanguage codes fold into the FLORES+ individual box
    (`zho→cmn`, `ara→arb`, `msa→zsm`, `nor→nob`), recorded as `lang_box_aliases` in the manifest;
  - Cosmopedia v2 (local shard, synthetic): BISAC subject headings in its prompts label history,
    philosophy, religion, literature, art, music, film, law, psychology, business, lifestyle, chemistry,
    biology, earth science, medicine and engineering; its audience and format labels seed
    `form.audience.*`, `form.textbook`, `form.instructions` and `form.dialogue`;
  - core corpus v2: PG-19 windows for `hum.literature.fiction/poetry/drama`, Wikipedia leads for
    `form.encyclopedic`;
  - origin per text (`natural`, `lab` = this lab's AI-agent-written repos, `template`, `synthetic` =
    Cosmopedia and TinyStories). Held text should not be model-generated, so held is filled natural first
    and synthetic last; the split itself stays the group-hash parity, so no row moves. 65 topics still have
    mostly synthetic held text (Cosmopedia is their only local source) and are flagged `held_synthetic`;
    51 inference-stack topics are flagged `held_lab`. Box verdicts on flagged topics carry the flag;
  - **v5.2 (build `topic-probes-cb94e8725`, 47,500 texts, 503 tested topics)** carries v5.1
    (`e8c22f31a`) and extends thin branches from licensed local text with `--extra-roots`: the uv CPython
    builds' Tcl library, the mlflow R package (Apache-2.0), numpy f2py Fortran tests, Xcode Objective-C
    templates and SDK framework headers, Go `_test.go` files for `swe.testing`, and 800-53 guidance and
    assessment parts for `sec.grc.*`. Thin code topics take up to 3 windows per file. Two mislabels are
    fixed, in new and carried rows: Apple Rez `.r` files were labelled R, and MATLAB/Octave `.m` files
    Objective-C (15 carried rows dropped, 4 of them v4 fit rows);
  - the code-root sources are live directories, so rebuilds are not bit-identical (two builds minutes
    apart differed in 10 lab-doc windows); the manifest's hashes are the record.
- **Builder:** `scripts/amoeba_topics_corpus.py`.
  - Local sources only, by default:
    - code from site-packages, Homebrew, the Xcode SDK and toolchain headers, system Perl, `/etc` and
      `/usr/bin` shell scripts (dialect from the shebang);
    - the inference stack from the mlx2 `src/` and `docs/`, the mlx-uag wiki and the MLX sources;
    - UI strings for languages (`*.lproj` `.strings`/`.loctable`);
    - SciFact abstracts (biomedicine);
    - GSM8K (word problems);
    - BillSum (US legislation);
    - TinyStories;
    - SymPy/SciPy module docstrings where installed (math and physics subfields);
    - man pages (Unix tools, git).
  - The texts go to `research-artifacts/amoeba/topic-probes-<rev>/corpus.jsonl`, which is withheld
    from the public export: it holds third-party text.
  - The manifest goes to `docs/research/amoeba-topics/probe-corpus-manifest.json`: per leaf, the counts
    per split, sources, licence, labelling method (`path`, `shebang`, `extension`, `keyword`,
    `dataset_label`, `template`) and sha256 of each text.
- **Keyword labels:** used for inference-stack subtopics and biomedical subfields. A text gets a label
  only with at least 2 keyword hits and a 2× margin over the next leaf. The method is recorded on the
  text, so weak labels stay visible.
- **Two downloads** are asked of Pierre once, and nothing waits on them:
  - FLORES+ devtest (CC BY-SA 4.0, about 30 MB). It is parallel text, so language is the only variable.
  - A stratified arXiv-abstracts sample (metadata CC0, about 1 GB parquet → 25 MB sample).

  Without them, many `lang.*`, `sci.*` and `math.*` leaves stay `untested`. That is a coverage gap, not
  a negative result.

## Capture

`scripts/amoeba_topics_capture.py` has two backends with one interface. It captures each MLP's
pre-`down_proj` activation (`silu(gate)·up`) at positions ≥ 1, excluding position 0, which is the
prompt-format artifact.
- **`mlx27b`** (`amoeba27b_lib.Host`, GPU, queued window): the host's `mean.npz` from
  `core-696591553`. Per text and neuron it records the mean over positions of |act − mean|, streamed to
  per-layer fp16 memmaps `[S, N]`, the atlas layout. For S ≈ 5,000 this is about 11 GB, kept under
  research-artifacts, never in git.
- **`torch`** (HF, CPU, small models such as Qwen3-0.6B): the same statistic, from a `down_proj`
  forward pre-hook. The mean comes from the same texts.

## Selectivity, controls and the hierarchical label (CPU, no model)

`scripts/amoeba_topics_label.py`:
- **Neuron response:** `rel[s, n] = r[s, n] / mean_s r[·, n]`, so a large neuron does not dominate.
- **Cluster score per text:** the mean of `rel` over the cluster's neurons.
- **Effect:** Cohen's d of the cluster score on topic texts against contrast texts (below), computed
  on fit and reported on held, with AUC alongside.
- **Controls, both required:**
  1. *Random twin:* 32 random neuron sets with the cluster's per-layer counts, drawn from the knowledge
     pool, so they are outside the core. The cluster's d must exceed the twins' 95th percentile by 0.5.
  2. *Shuffled labels:* the label-shuffle null of d, taken as normal with standard error
     `sqrt(1/g_pos + 1/g_neg)`, where g counts distinct source *groups*, not texts. Windows from one file
     are correlated, so counting texts would understate the null spread. The resulting p is
     Bonferroni-corrected over every (cluster, node, contrast) test and must be below 0.01.
     - An empirical text-level permutation p (2,000 shuffles) is reported for passing nodes as
       information only.
     - Its floor of 1/2001 cannot survive a Bonferroni correction over ~20k tests, so it gates nothing.

  Held-split d must also be at least 0.8 and above the twins' held 95th percentile.
- **Label = the finest passing node** (revised after the synthetic test; the first draft descended from the
  roots). Every tested node is gated:
  - a root, against all other texts;
  - a non-root, against its *siblings* (the other children of the same parent, which keeps "code against
    prose" from passing as "Python against C") **and** against all other texts.
  - A strict top-down descent missed a cluster selective for one leaf when its root as a whole was not
    selective: Python neurons against all prose, when only a third of the programming texts are Python.
  - A passing node with a passing descendant gives way to the descendant.
  - Up to three labels are kept, ranked by **fit** d; held only confirms.
  - Clusters with no passing node stay `unlabelled`, with their strongest failed node and why it failed.
- **Subclusters:** re-cluster a cluster's neurons on their topic-conditioned profile: per-neuron d per
  tested leaf, centred and L2-normalised, then spherical k-means with k ∈ 2..6.
  - k is chosen **inside fit**: cluster on fit half A (halved by source group), assign half B's
    profiles, and take the largest k with agreement ≥ 0.8 and every subcluster ≥ 32 neurons.
  - The final assignment uses all of fit. Each subcluster is labelled by the same test, with twins drawn
    *inside the parent cluster* (same per-layer counts), so a subcluster must be more specific than a
    random part of its parent. Held then confirms.

## Puzzle stage: placing unlabelled neurons (Pierre, 2026-11-14; CPU, `scripts/amoeba_topics_puzzle.py`)

Pierre's idea: "Cluster neurons that can't be classified either stay in the core or are placed like a
puzzle, by what they co-activate with, with the gaps getting clearer as more gets classified."

- **Anchors:** neurons of clusters and subclusters that passed the selectivity gates, plus the core as
  one more anchor.
- **Candidates:** every neuron of an unlabelled cluster, taken one at a time.
- **Each round,** on fit texts:
  - A candidate joins topic t when its response correlates with t's anchor score at r ≥ 0.5, its own
    fit d for t is ≥ 1, and it follows t more than it follows the core.
  - Its confidence is `r · min(1, d/2)`.
  - When several topics come within 0.05 of the best, it gets a weighted multi-membership (`shared`).
  - New members anchor the next round.
- **Stop rule:** stop when a round adds < 1% of the candidates, when its held confirmation falls below
  0.7 (held confirmation = the share of new members with held d ≥ 0.5), or after 6 rounds.
- **Explain away:** regress each leftover neuron on all anchor scores plus the core score. The R²
  distribution is reported. The texts with the largest positive residual over the leftovers are listed
  as passages no probe covered, hints for new topics.
- **Leftovers stay in the core,** tagged with evidence and their top five co-activation partners:
  - `general`: the core score alone gives R² ≥ 0.5;
  - `shared`;
  - `unexplained`.

  Leftovers in the first and last four layers are also listed as `struct.detokenize` and
  `struct.output_shaping` *candidates*. That is layer evidence only, not a label, and they stay in the
  core.
- **Verification:** a propagated label counts only after the group-level, twin-controlled removal
  check. Each topic's propagated set is a `puzzle:<topic>` unit in window B. When it is unmeasured, it
  stays `cluster_unverified`.
- **Shared neurons:** the stub recipe copies them (`split_units(..., shared=)`) and never moves them.
- **Report per model:** the fractions that are classified, core by evidence (general), shared and
  unexplained.
- **q8 rounding fallback** (Pierre: "try and exhaust q8 first"):
  - Only after the q8 rounds are exhausted, a cheap triage compares q8 and BF16 activations of the
    unexplained neurons on a small sample: correlation, and on/off flips.
  - If rounding moved them, a forward-only BF16 re-probe of only those neurons re-runs selectivity and
    the puzzle, and every label records the precision it came from.
  - A BF16-only label notes that it may not hold in q8 serving.
  - This needs its own GPU window (~51–55 GB, :8282 paused), which needs a new approval.

## Roll-up of small units (Pierre, 2026-11-14)

Pierre: "for small boxes like that, a top level generic coding box might be the best place, as long as
there's no induced interference".
- Labels stay fine-grained.
- The plan also tests, per parent with at least two child units under 1,024 neurons, one merged unit
  (`rollup:<parent>`): the uncarved union of the members' units (no mask carving, no gate fitting; nothing changes weights).
- In the registry, a member moves into the roll-up (`status: box`, `in_box: rollup:<parent>`) only if
  **every** member's add gain inside the merged unit is at least its standalone unit's gain minus
  0.01 nats, **and** the merged unit does not worsen the off-topic control (add ≥ −0.01, removal ≤ 0.01).
- Otherwise the members keep their own units, and the verdict is recorded.

## Donor-gap screen (FN round; Pierre, 2026-11-14)

Pierre: "run say 200 topic prompts through, see which ones fn does better than 27b, those are your
candidates". The script is `scripts/amoeba_topics_donor_gap.py`.
- **Score:** both models score every held probe text, forward only, as NLL per *byte*, so different
  tokenizers compare. There are two conditions:
  - bare;
  - a neutral prefix, the position-0 and format control.
- **Candidates:** a topic where (host − donor) nats/byte has a 95% CI lower bound above the margin in
  **both** conditions. The CI comes from a bootstrap over source groups.
- **Answerable items:** topics with answerable items (the task sets, GSM8K) add accuracy with the
  two-shot control in the FN round.
- **27B side:** window B scores the 27B right after the functional check (`function+screen`).
- **FN side:** the mlx2 server exposes no prompt logprobs (no echo or prompt_logprobs in
  `openai_compat.py`), so FN cannot be scored through :8282 without a pause. It needs either a server
  feature (Pierre's call) or its own FN window at about 72 GB, and either needs a new approval.

## Per-topic neuron map is the primary record

Cluster IDs are arbitrary per model; the topic ID is the cross-model join key (coordinator's note).
- The registry is keyed by topic.
- Each row's neuron maps are the record:
  - cluster and subcluster members;
  - propagated members with weights;
  - the core tag.
- Cluster labels are a derived view, the `clusters` section of the registry.

## Functional check (GPU, queued, forward only)

Metric for any topic: mean next-token NLL on the topic's held texts, at positions ≥ 1.

The arms, on the 53.7% export core (zero ablation, no stand-in, as in `boxes-de6adef06`):
- core + cluster against core + twin;
- full − cluster against full − twin.

Each unit has 4 twins with the same per-layer counts. The pass gate:
- removal costs X at least 0.02 nats more than the twins' mean;
- the unit beats every twin;
- the effect is at least twice the unit's effect on the off-topic control.

The control is held texts outside every root branch the unit is labelled with.

Also measured:
- per-layer-band removal for labelled units and core maps (4–8 bands, against the first twin). This
  gives the per-layer strength that the stub needs.

Core maps:
- the core neurons with fit d ≥ 1 for a node, selected on **fit** only, because the functional check
  measures on held texts;
- capped at the 4,096 strongest by fit d;
- kept only when their count is at least max(8, 2 × the count of core neurons at fit d ≤ −1), a chance
  estimate from the opposite tail.

The run has a time budget and writes partial results that can be resumed (`--budget-min`,
`--skip-done`). An unmeasured unit is recorded as `cluster_unverified`, never as a failure.

## Registry: `registry/<artifact>@<fp8>.topics.json`, schema `mlx2.amoeba-topic-map.v1`

- **Header:** the model identity (`<artifact>@<fp8>` for display, plus the full 64-hex fingerprint for
  joins, as hub-align runs use it; shape), the taxonomy version, the
  inputs (core run, atlas-clusters dir, capture dir with its corpus sha, labelling run), and the mlx2
  commit.
- **One row per taxonomy node that was tested or is held by a box:**
  - `status`: `cluster` | `box` | `cluster_unverified` | `subsumed` (a descendant carries it) |
    `core_resident` | `absent`; topics without probe texts are untested and have no row;
  - `units`: refs to the box registry `box_id`, pool cluster `pool:C32`, or subcluster `pool:C32.s1`;
  - `neuron_map`: per-layer counts here, the indices in the companion `.topics.npz` (key `"<topic>"`
    → int32 `[n, 2]` (layer, neuron)), and the npz sha in the row;
  - `selectivity`: d and AUC, fit and held, the twin 95th percentile and shuffled p;
  - `functional`: the add/remove ΔNLL against the twin, the off-topic ΔNLL, and the per-band removal
    ΔNLL;
  - `reason_not_split` (core-resident rows only);
  - `hub_align` (optional): a list of refs to xarch-maps hub-alignment runs for a donor → host pair:
    `{run (dir), schema ("amoeba-hub-align/v1"), candidate (box problem id, e.g. decimal_17), donor, host,
    warp ("warp.json": donor_for_host, one donor layer per host layer)}`.
    - These refs are problem-scoped. A warp fit on one problem's traces is not a general layer map.
    - The coarse donor capture (layers 0–5, every 4th, and the last) makes the warp plateau.
    - Agreed with the xarch-maps session, 2026-11-14.
- **Core tags.**
  - When a topic passes selectivity, but no pool cluster carries it, or the cluster that carries it fails
    the functional gate, the topic is tagged on the core.
  - The core tag's neuron map is exactly the plan's core map: fit-selected and chance-controlled, with
    the held confirmation fraction recorded. Its functional result measures those same neurons.
  - `reason_not_split` is one of:
    - `no_pool_cluster` (selective neurons live in the core);
    - `cluster_failed_function`;
    - `below_size_floor` (< 256 neurons);
    - `overlaps_english_core`;
    - `subsumed_by:<topic>`.
  - These rows are what Pierre asked for: every unsplit topic still has its neural map.
- **Joins, not forks.**
  - Box registry records (`mlx2.amoeba-box.v1`) gain a `topic_ids` list (CLI `topic-tag`), and the
    topic row points back at the `box_id`.
  - Tile manifests gain an optional `topic_ids` per tile.
  - The cross-model join is the topic ID. The model-pair geometry (depth warp, hub maps) stays in the
    xarch-maps session's hub-alignment runs, referenced by path.

## Stub extraction (the donor remapping contract)

`scripts/amoeba_topics_stub.py`, `cut_stub(host_map, topic, …)`:
1. **Candidates.** The host's units for X or a descendant whose functional gate passed (from the
   registry), else X's core tag.
2. **Ranking.** Candidates are ranked by per-neuron **fit** selectivity. The stub grows until removing it
   costs X at least 90% of the full candidate set's removal ΔNLL (bisection on the ranked list, chosen
   on fit, reported on held).
3. **The stub record:**
   - per-layer neuron ids;
   - per-neuron selectivity;
   - per-layer-band functional strength;
   - `layer_profile`: the fraction of removal strength per layer, normalised;
   - the twin-controlled removal ΔNLL for X and off-topic;
   - the source rows' hashes.
4. **Split.** The stub's neurons are removed from the core (or their box) and become a gated
   stub box. Invariants, all unit-tested:
   - core ∪ boxes ∪ stub covers what it covered before;
   - no neuron is in two units;
   - the masks are bit-exact against their per-layer index lists.
5. **Remap** (the next session's work, for FN): the donor's box for X is placed on the host by the
   xarch-maps depth warp. Its per-layer budget follows the stub's `layer_profile`, and it is gated where
   the stub was gated.

A unit test and a CPU evidence run on Qwen3-0.6B check that removing the stub costs X much more than
a random twin does, and costs off-topic text little.

## Where each stage runs

| Stage | Where | Memory |
|---|---|---|
| Taxonomy, corpus, manifest | CPU, no model | < 2 GB |
| 0.6B capture, labels, stub test | CPU torch fp32, memcheck `small` | ~4 GB |
| 27B capture + functional check | GPU, one queued window (CPG lease + paired locks, `amoeba27b_window.sh`), behind the 27B session's windows 2 and 3 | q8 30 GB + ~3 GB |
| Labelling, subclusters, registry | CPU, reads memmaps | ~6 GB |

Every result names `<artifact>@<fingerprint>` and the mlx2 commit.
