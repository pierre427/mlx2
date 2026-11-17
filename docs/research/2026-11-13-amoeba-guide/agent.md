# `mlx2 amoeba auto`: a 4B-driven skill-ingestion harness

Experimental and unqualified. Every run is labelled **smoke test (models < 8B)** while its models are small.

```
mlx2 amoeba auto --goal "make the host better at Python programming" --host DIR --candidates D1 D2 --out NEW_DIR
mlx2 amoeba auto --self --candidates D1 D2 --out NEW_DIR      # ingest into the brain itself
mlx2 amoeba auto-status                                        # competence, lessons, open jobs
mlx2 amoeba auto-eval --brain OTHER_BRAIN_DIR                  # replay labelled decisions through another brain
```

The common options are `--max-steps N` (default 12; a one-option menu spends none), `--brain DIR` (default `~/mlx-models/Qwen3.5-4B-MLX-4bit`),
`--tracker PATH` and `--no-tracker`, plus the run flags shared by every amoeba command (`--ignore-gpu-queue REASON`,
`--gpu-window-owner`, `--allow-dirty`).

## Primary goal

Pierre, 2026-11-16: the harness exists to improve **its own ability to ingest expert skills**, and to improve how it
improves. The harness never claims an improvement. It measures one, in three ways:

1. **Oracle labels.** Every brain decision is labelled after the fact by a deterministic oracle: the set of
   acceptable options, taken from pipeline outcomes. The tracker keeps accuracy per stage per brain checkpoint, so
   `auto-status` shows whether the harness is choosing better over time.
2. **Lessons against a control arm.** At the end of each run the brain writes one sentence of advice from the
   run's deterministic outcome (no lesson when nothing deterministic happened). The lesson is tagged with the topics
   it is about.
   - When a decision has matching lessons (the best two by record plus the least-shown other one, so new lessons
     always get showings), it alternates per stage and brain between a **lessons** arm, which shows them, and a
     **control** arm, which withholds them. A decision with no matching lesson is arm `none`, outside the
     comparison, so control rows come from the same period, commit and brain as the lessons rows.
   - A lesson's record counts only its labelled showings, kept per stage and brain checkpoint.
   - A lesson is retired when, for one stage and brain, it has at least 3 labelled showings, that stage has at least
     5 labelled control decisions, and its rate is more than 0.1 below the control rate. Nothing is retired on a
     handful of labels, and a retired sentence written again does not come back as a new lesson. Retirement is
     global: a lesson shown to do harm on one stage and brain is not recalled anywhere.
   - Unlike mlx2-explore, which keeps lessons forever, a lesson here has to beat the same brain without it.
3. **Brain replacement by evidence.** `auto-eval --brain X` replays every labelled decision, with its stored prompt,
   menu and selection rule, through brain X. A choice outside the options the oracle observed counts as unknown,
   not wrong. It reports accuracy on known choices, coverage, and the recorded brains' accuracy, with the commit,
   X's fingerprint and the smoke-test label. A better
   brain is one that wins on this replay. One example is a future brain carrying an installed box.

`--self` aims ingestion at the brain itself:
- the host is the brain;
- the topics are the skills the harness uses: `ml.inference`, `ml.interpretability`, `ml.llm.tool_use`,
  `ml.llm.agents` and `prog.python`, filtered by the box-set minimums;
- donors must share the brain's tokenizer, because install pairs tokens by id;
- donors must be dense, because install refuses MoE box units.

These filters are deterministic, and dropped donors are listed with their reason. A passing box goes on to
`mlx2 amoeba install --routes side`, and the install verdict is recorded as a fact. It is **never adopted**: mlx2
has no loader for installed routes, so the brain stays the checkpoint it was. Of the local models, only the 4B's own
variants and two MoE models (Qwen3.6-35B-A3B and Sage-A1.5) share the brain's tokenizer. A real self-ingestion
therefore still needs a dense same-tokenizer donor larger than the brain.

## Where the brain decides, and where code decides

The brain never writes a choice. Each decision is a lettered menu that code builds. The brain child returns the
log-probability of each letter as the first answer token. Code then takes:
- the argmax for a choice;
- the full order for a ranking;
- or every option within 1 nat of the best, up to 3, for a topic subset.

Each menu is scored once per cyclic shift and the scores are averaged, so every option is scored under every
letter and the letter prior adds the same amount to each. A choice outside the menu cannot be produced. A one-option menu needs no brain call and is never labelled.
A malformed reply or a crashed brain child is retried once and then stops the run.

| Step | Brain | Code |
|---|---|---|
| scope | picks 1–3 topics from the shortlist | builds the shortlist with the box-set builder's minimums, then validates the ids |
| locate | ranks the donors for those topics (the host is never on the menu) | inspects the checkpoints and drops the host |
| assess | — | runs the plan; refusals drop pairs; runs the centred gap screen (`extract --gap-only`) |
| triage | ranks the surviving pairs | — |
| run | — | resumes extract (core, map, box, gate, bundle) and reads the verdict |
| install (`--self`) | — | install side route; verdict recorded, never adopted |
| recover | next pair / retry once (crash only) / stop and ask | gap-screen refusals drop the pair with no brain call |
| learn | one advisory sentence | dedups, tags, and credits or retires by oracle labels |

Free text (a one-line rationale per decision and the lesson) is logged. It never controls anything.

## Work tracker

`~/Library/Application Support/mlx2/research-artifacts/amoeba/agent/tracker.sqlite` (override it with
`--tracker` or `MLX2_AMOEBA_AGENT_DB`). The design comes from mlx2-explore (see docs/PROVENANCE.md). The tables are:
- **jobs:** one per (stage, host fingerprint, donor fingerprint, topic, profile, probe-corpus sha, taxonomy sha,
  pipeline epoch), plus the box bundle's manifest sha for an install.
  - A lease is renewed before each pipeline child (timeout 2 h). One not renewed for 3 h returns to the queue, so an
    interrupted run's work is picked up again.
  - Jobs that are done or refused are reused instead of rerun, which also gives the oracle counterfactuals from
    earlier runs. A reused extraction whose directory is gone is rerun.
  - Only a gap verdict, a resource-plan refusal or one of install's policy refusals (no box, MoE donor, tokenizer or
    vocabulary mismatch, no interior units, layer collision) is a terminal `refused`. Guard refusals, crashes and other exits
    are `failed`, and each run gets 2 attempts at a failed job.
  - An admission block returns the lease without spending an attempt.
  - `auto-status --reset KEY` sends a job back to the queue.
  - A reused gap screen never leads to writing into another run's directory: the extraction then starts fresh in
    this run's `--out`.
- **facts:** deterministic outcomes only (gap screens, box verdicts, install verdicts), each with the commit it ran
  from.
- **decisions:** the prompt, menu, scores, choice, the lessons shown, the oracle and correctness.
- **lessons:** text, tags, and the shown, helped, hurt and retired counts.
- **runs:** one row per run.

`<out>/ledger.jsonl`, `events.jsonl` and `auto_report.json` remain the per-run evidence. The report
(schema `mlx2.amoeba-auto.v2`) names the brain checkpoint, the mlx2 commit, every decision, the oracle labels, the
lesson and a competence snapshot.

## Oracle rules

A decision is labelled only when it could have been right and could have been wrong: at least one acceptable
option is known, and the first choice's own outcome is known (it is acceptable, or every one of its pairs has a
known result). An option that was never assessed is unknown, not
wrong. Knowledge is the gap verdict or a resource-plan refusal, from this run or the tracker. A crash, an
admission block, a guard refusal or another run's lease is not knowledge.

- **scope:** correct when the first topic has a candidate donor that leads the host.
- **locate:** correct when the first donor leads the host on a chosen topic.
- **triage:** correct when the first pair has the best known extraction verdict (pass > miss > covered > failed).
  It needs at least two known verdicts, since a single run's verdict cannot make a ranking wrong.

## Safety

These carry over from v1:
- Memory admission runs before each brain and pipeline child, and refuses below 20 GiB free or above 1 GiB swap
  growth.
- The output directory must be fresh.
- `--ignore-gpu-queue REASON` bypasses only the GPU-window guard, and is recorded with the lock holder in every
  receipt and ledger entry.
- No services are touched and no foreign processes are killed.

## Results

### Smoke 1 (v1, `0852ff1f1`, 2026-11-16)

The brain wrote JSON rankings, and both rankings included the host as a donor. The menu check refused both, and the
run stopped after 60.75 s. No ingestion ran. That failure is why v2 has the brain score letters instead of writing
names.

### Smoke 2 (v2, `4517148be`, then `56e44604b`), 2026-11-16

Same command, models and override. Brain `Qwen3.5-4B-MLX-4bit@82134c29`, host `Qwen3-0.6B@759383a5`.

- **`4517148be`:** stopped at the first memory admission after 31 s. It counted "Pages free" (0.7 GB). On macOS the
  free page count stays near zero, and the reclaimable file cache sits in inactive, purgeable and speculative
  pages, about 25 GB at the time. Admission now counts available memory.
- **`56e44604b`:** 58 s, rc 0. Every brain decision was valid and sensible, and each took about 4 s:
  - **locate:** Qwen3-4B first (−0.13 nats) over Qwen3-1.7B-4bit (−3.06).
  - **scope:** `prog.python` (−0.70) over `prog` (−2.39) and eight subtopics (−5.2 to −7.1); the subset kept one.
  - **recover (×2):** `next_pair` (−0.20) over retry (−2.3) and stop (−2.9).
- **Both gap screens were blocked, not run.** The research scripts' standing memory admission (no second large
  model process beside a foreign one over 8 GB, coordinator 2026-11-14) refused to start next to the production
  :8282 service (Flash-Next, 73 GB). `--ignore-gpu-queue` does not bypass that rule, and should not.
- **What that run exposed:** v2 treated the blocks as "donor does not lead", labelled the locate decision wrong,
  and stored a lesson written from them. Fixed in `9f2effc04`: admission blocks stop the run naming the process,
  and crashes are not knowledge. The bad label and lesson were cleared from the tracker by hand.
- **The independent review of `4517148be`** found 12 oracle, tracker, scoring and lesson defects, fixed in `f355c844e`.

No ingestion verdict exists yet: the gap screens cannot run while the :8282 service is resident.

### Smoke 3 (`27aec47f5`, owned GPU window, 2026-11-16 21:28)

Pierre gave this run 30 minutes of GPU time, so it ran as `--minutes 28` inside an owned window with :8282 paused;
:8282 was healthy afterwards. It used the same goal and models, `--max-steps 16`, and the persistent tracker.

- **Outcome:** refused, rc 0, after 6.6 minutes (393 s inside the harness).
- **scope:** `prog.python` at −0.31 nats, with `prog` at −2.82 and every subtopic between −5.1 and −6.4. Arm `none`:
  no lesson was available for this topic.
- **locate:** Qwen3-4B first (−0.39) over Qwen3-1.7B-4bit (−1.77).
- **assess:** both centred gap screens ran, and both refused (deterministic, no brain call).

| Donor vs Qwen3-0.6B, `prog.python` | Bare gap (nats/byte) | Centred gap |
|---|---|---|
| Qwen3-4B@546bcd98 | +0.083 [0.078, 0.089] | −0.008 [−0.014, −0.001] |
| Qwen3-1.7B-4bit@91e95023 | +0.025 [0.020, 0.029] | −0.014 [−0.020, −0.009] |

Both donors are better than the host on Python only by as much as they are better in general. Nothing
Python-specific is there to extract, so the screen refused both, which is correct.

- **Oracle:** no decision was labelled, because no option was acceptable (empty gold). This is the rule working, not
  a missing label.
- **Lesson:** one was written. It is advisory, tagged `prog.python`, and will be credited against the control arm
  in later runs.

No extraction ran, so there is still no box verdict. A useful next goal is one where some candidate is a topic
specialist (a coder or math fine-tune against a general host), not just a larger model of the same family.
