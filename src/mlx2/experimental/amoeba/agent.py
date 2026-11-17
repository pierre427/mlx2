"""Bounded decisions around the existing amoeba pipeline; experimental, unqualified.

No mined implementation: orchestration is original; ordinary decode is reused from child_gen_server.Generator; the
work tracker's design follows mlx2-explore (see agent_tracker.py). The brain exits before any pipeline model loads.

The brain never writes a choice. Every decision is a lettered menu that deterministic code builds; the brain child
returns the log-probability of each letter as its first answer token, averaged over every cyclic shift of the menu
(each option scored once under each letter, so a letter's prior does not decide), and code takes the argmax, the
ranking or the near-top subset. A choice outside the menu cannot be produced. The brain's free text (a one-line
rationale per decision, one lesson per run) is advisory: it is logged, and a lesson is recalled into later prompts
only while the decisions it was shown for do at least as well as the same stage's no-lesson arm
(agent_tracker.Tracker.label).

Primary goal (Pierre, 2026-11-16): improve the harness's own ability to ingest expert skills. Measured, not claimed:
- every decision is labelled after the fact by a deterministic oracle (the acceptable options, from pipeline
  outcomes) and the tracker keeps per-stage accuracy per brain checkpoint;
- ``mlx2 amoeba auto-eval --brain DIR`` replays the labelled decisions through another brain checkpoint, which is how a
  better brain (for example one carrying an installed box) is shown to decide better before it replaces this one;
- ``--self`` points ingestion at the brain itself: the host is the brain, the topics are the skills its job uses, the
  donors must share its tokenizer (install pairs tokens by id), and a passing box goes on to ``mlx2 amoeba install``.
  An install verdict is recorded; it is never adopted automatically (mlx2 has no loader for installed routes yet).
"""
from __future__ import annotations

import functools
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

from . import paths
from .inspection import inspect_checkpoint
from .progress import Progress
from .runner import gpu_window, require_gpu_window, stop_group
from .state import write_json

LABEL = "smoke test (models < 8B)"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SUBSET_NATS = 1.0            # subset mode: options within this many nats of the best one, up to the maximum
SELF_GOAL = ("improve this harness's own ability to ingest expert skills: inference engines, model internals, "
             "tool use and agents, Python")
SELF_TOPICS = ("ml.inference", "ml.interpretability", "ml.llm.tool_use", "ml.llm.agents", "prog.python")
VERDICT_RANK = {"pass": 3, "miss": 2, "covered": 1, "failed": 0}
DEADLINE = {"t": None}       # --minutes: epoch after which no action or child starts; running children are cut at it
DEADLINE_GRACE_S = 120       # a pipeline child gets this long past the deadline to checkpoint (--stop-at) before kill


def past_deadline():
    return DEADLINE["t"] is not None and time.time() >= DEADLINE["t"]


class Stop(Exception):
    pass


def memory_snapshot():
    try:
        pressure = subprocess.check_output(["memory_pressure"], text=True, timeout=15)
        swap = subprocess.check_output(["sysctl", "vm.swapusage"], text=True, timeout=15)
    except (subprocess.SubprocessError, OSError) as exc:
        raise Stop(f"memory admission refused: memory admission unavailable ({type(exc).__name__}: {exc})")
    page = re.search(r"page size of (\d+)", pressure)
    used = re.search(r"used = ([\d.]+)([MG])", swap)
    # available, not "Pages free": macOS keeps free pages near zero and holds reclaimable file cache as inactive
    # and speculative pages (the 2026-11-16 v2 smoke refused at 0.7 GB "free" with ~25 GB available)
    # (purgeable pages are already inside active/inactive: counting them again would overstate availability)
    counts = {k: re.search(rf"Pages {k}:\s*(\d+)", pressure) for k in ("free", "inactive", "speculative")}
    if not (page and used and all(counts.values())):
        raise Stop("memory admission unavailable")
    return {"free_gb": int(page[1]) * sum(int(m[1]) for m in counts.values()) / 2**30,
            "pages_free_gb": int(page[1]) * int(counts["free"][1]) / 2**30,
            "swap_gb": float(used[1]) / (1024 if used[2] == "M" else 1),
            "memory_pressure": pressure, "swapusage": swap}


def admit_memory(baseline):
    facts = memory_snapshot()
    if facts["free_gb"] < 20 or facts["swap_gb"] - baseline > 1:
        raise Stop("memory admission refused: " + json.dumps({k: facts[k] for k in ("free_gb", "swap_gb")}))
    return facts


PIPELINE_GRACE_S = 150      # > runner.stop_group's 120 s, so a pipeline child can stop its own model processes


def child(cmd, *, env=None, stdin=None, log=None, timeout=7200, grace_s=10):
    """Own only this child process group; always reap on timeout or interruption. Under a run deadline no child
    starts after it, and a running one is stopped (its whole group) DEADLINE_GRACE_S after it."""
    if DEADLINE["t"] is not None:
        left = DEADLINE["t"] - time.time()
        if left <= 0:
            raise Stop("deadline: the run's time budget is spent")
        timeout = min(timeout, left + DEADLINE_GRACE_S)   # then stop_group(grace_s) on the whole group
    p = subprocess.Popen([str(x) for x in cmd], stdin=subprocess.PIPE if stdin is not None else None,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env=env or paths.child_env(), start_new_session=True)
    try:
        out, err = p.communicate(stdin, timeout=timeout)
    except BaseException:
        # SIGTERM to the child's group, SIGKILL after grace_s. A pipeline child (mlx2 amoeba ...) runs its model
        # processes in sessions of their own and stops them on SIGTERM within 120 s; its grace must cover that
        stop_group(p, grace_s=grace_s)
        raise
    if log:
        Path(log).write_text(out + "\n" + err)
    return p.returncode, out, err


class Brain:
    """One brain child per call (one model per process; nothing stays loaded while a pipeline child runs)."""

    def __init__(self, model, baseline, audit, window_owner="", ignore_queue=""):
        self.model, self.baseline, self.audit = model, baseline, audit
        self.window_owner, self.ignore_queue = window_owner, ignore_queue

    def __call__(self, request: dict) -> dict:
        # every brain load is a model load: the GPU-window guard again (as each pipeline stage does), then memory
        require_gpu_window(False, self.window_owner, ignore_queue=self.ignore_queue)
        facts = admit_memory(self.baseline)
        self.audit("memory_admission", facts={k: facts[k] for k in ("free_gb", "swap_gb")})
        rc, out, err = child([sys.executable, "-m", __name__, "--brain-child", self.model],
                             stdin=json.dumps(request), timeout=900)
        if rc:
            raise RuntimeError(f"brain child rc={rc}: {err[-4000:]}")
        return json.loads(out)


def check_scores(reply, n):
    """A brain reply must score exactly the menu's letters with finite numbers; anything else is a brain failure."""
    if not isinstance(reply, dict) or not isinstance(reply.get("scores"), list) or len(reply["scores"]) != n:
        raise ValueError(f"expected {n} scores")
    s = reply["scores"]
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in s):
        raise ValueError("non-finite score")
    return [float(x) for x in s]


def select(scores, mode, maximum=None):
    """Code's choice from the brain's scores: indices, best first."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    if mode == "one":
        return order[:1]
    if mode == "rank":
        return order
    if mode == "subset":
        top = scores[order[0]]
        keep = [i for i in order if scores[i] >= top - SUBSET_NATS]
        return keep[:maximum or len(keep)]
    raise ValueError(mode)


def menu_prompt(stage, question, facts, options, lessons):
    lines = ["You help a program decide one step of a skill-ingestion run. The program runs every step, checks every "
             "result and decides pass or fail itself; you only answer the question below.",
             f"Step: {stage}. Question: {question}"]
    if lessons:
        lines.append("Lessons from earlier runs (advisory):")
        lines += [f"- {x['text']}" for x in lessons]
    lines.append("Facts: " + json.dumps(facts, sort_keys=True, default=str))
    lines.append("Options:")
    lines += [f"{LETTERS[i]}. {o['text']}" for i, o in enumerate(options)]
    lines.append("Answer with the letter of the best option only.")
    return "\n".join(lines)


class Agent:
    def __init__(self, out, max_steps, brain, pipeline, metadata=None, tracker=None, run_id=None):
        self.out = Path(out)
        if self.out.exists() and any(self.out.iterdir()):
            raise Stop(f"{self.out} is nonempty; refusing overwrite")
        if max_steps < 1:
            raise Stop("max-steps must be positive")
        self.out.mkdir(parents=True, exist_ok=True)
        self.steps, self.limit = 0, max_steps
        self.brain, self.pipeline = brain, pipeline
        self.metadata = metadata or {}
        self.tracker = tracker
        self.run_id = run_id or f"{self.out.name}-{int(time.time())}"
        self.records = []
        self.decided = {}          # stage -> [(decision id, chosen ids)]
        self.assessed = {}         # pair id -> {"pair", "leads", "verdict"}
        self.menus = {}            # stage -> the first menu's option ids
        self.last_result = None
        self.progress = Progress(self.out / "events.jsonl", quiet=True)

    def audit(self, event, **fields):
        rec = {"step": self.steps, "event": event, "ts": time.time(), "prompt": None,
               "raw_output": None, "parsed_choice": None, "facts": {},
               "gpu_window": {**gpu_window(), "ignore_gpu_queue": self.metadata.get("ignore_gpu_queue", "")}, **fields}
        self.records.append(rec)
        with (self.out / "ledger.jsonl").open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self.progress.emit("auto_" + event, **{k: v for k, v in rec.items() if k != "event"})

    def action(self, stage):
        if self.steps >= self.limit:
            raise Stop("max_steps")
        if past_deadline():
            raise Stop("deadline: the run's time budget is spent")
        self.steps += 1
        self.audit("action", stage=stage)

    def decide(self, stage, question, facts, options, mode="one", maximum=None, tags=()):
        """options: [{"id": ..., "text": ...}]; returns the chosen ids, best first."""
        if not options:
            raise Stop(f"{stage}: empty menu")
        if len(options) > len(LETTERS):
            raise Stop(f"{stage}: menu longer than {len(LETTERS)}")
        if len(options) == 1:      # nothing to choose: no brain call, no step, never labelled
            self.audit("decision", stage=stage, facts=facts,
                       parsed_choice={"mode": mode, "chosen": [options[0]["id"]], "scores": None, "only_option": True})
            self.menus.setdefault(stage, [options[0]["id"]])
            self.decided.setdefault(stage, []).append((None, [options[0]["id"]]))
            return [options[0]["id"]]
        self.action(stage)
        # the lesson arm alternates per stage, so a lesson is credited against decisions made without lessons
        # recall first: with no matching lesson the decision is arm "none" (outside the comparison); otherwise the
        # lessons / control arms alternate, and a control decision withholds the lessons it could have shown
        avail = self.tracker.recall(set(tags)) if (self.tracker and tags) else []
        if not avail:
            arm, lessons = "none", []
        else:
            arm = self.tracker.next_arm(stage, self.metadata.get("brain_checkpoint", ""))
            lessons = avail if arm == "lessons" else []
        prompt = menu_prompt(stage, question, facts, options, lessons)
        request = {"mode": "choose", "prompt": prompt, "n": len(options), "why_tokens": 48}
        for attempt in range(2):
            reply = None
            try:
                reply = self.brain(request)
                scores = check_scores(reply, len(options))
            except (ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
                self.audit("brain_failure", stage=stage, prompt=prompt, raw_output=str(reply), facts=facts,
                           error=str(exc)[-2000:])
                continue
            idx = select(scores, mode, maximum)
            chosen = [options[i]["id"] for i in idx]
            did = None
            if self.tracker:
                did = self.tracker.record_decision(self.run_id, stage, self.metadata.get("brain_checkpoint", ""),
                                                   prompt, {"mode": mode, "maximum": maximum,
                                                            "ids": [o["id"] for o in options]},
                                                   scores, chosen, [x["id"] for x in lessons], arm=arm)
            self.decided.setdefault(stage, []).append((did, chosen))
            self.menus.setdefault(stage, [o["id"] for o in options])
            self.audit("decision", stage=stage, prompt=prompt, raw_output=reply.get("why", ""),
                       parsed_choice={"mode": mode, "chosen": chosen, "arm": arm,
                                      "scores": dict(zip([o["id"] for o in options], scores)),
                                      "lessons": [x["id"] for x in lessons]}, facts=facts)
            return chosen
        raise Stop("stop_and_ask: brain failed twice")

    def write(self, stage, prompt, max_tokens=80):
        self.action(stage)
        try:
            reply = self.brain({"mode": "write", "prompt": prompt, "max_tokens": max_tokens})
            text = str(reply.get("text", "")).strip()
        except Exception as exc:  # noqa: BLE001 - advisory text only; the run's outcome never depends on it
            self.audit("brain_failure", stage=stage, prompt=prompt, error=str(exc))
            return ""
        self.audit("written", stage=stage, prompt=prompt, raw_output=text)
        return text

    def recover(self, facts, crash, retried):
        menu = [{"id": "next_pair", "text": "skip this donor/topic pair and try the next one"},
                {"id": "stop_and_ask", "text": "stop the run and ask the operator"}]
        if crash and not retried:
            menu.insert(1, {"id": "retry_same", "text": "retry the same step once (it crashed)"})
        return self.decide("recover", "The last step did not succeed. What should the program do?",
                           {k: str(facts.get(k))[-1500:] for k in ("status", "reason", "error") if facts.get(k)},
                           menu)[0]

    def perform(self, stage, pair):
        retried = False
        while True:
            self.action(stage)
            try:
                result = self.pipeline(stage, pair)
            except Exception as exc:  # noqa: BLE001 - a crash goes to the recovery menu
                result = {"status": "crash", "error": f"{type(exc).__name__}: {exc}"}
            self.audit("stage_result", stage=stage, facts={"pair": pair, **result})
            self.last_result = result
            if result["status"] in ("ok", "reused"):
                return result
            if result["status"] == "memory_stop":
                raise Stop(result["reason"])
            if result["status"] in ("refused", "busy"):
                return None        # deterministic refusal or another run's lease: nothing to choose
            action = self.recover(result, result["status"] == "crash", retried)
            if action == "stop_and_ask":
                raise Stop("stop_and_ask: " + json.dumps({k: result.get(k) for k in ("status", "reason", "error")},
                                                         default=str))
            if action == "next_pair":
                return None
            retried = True

    def run(self, goal, host, candidates, shortlist, *, self_mode=False):
        t0 = time.monotonic()
        outcome = {"status": "stopped"}
        interrupted = False
        try:
            cands = [c for c in candidates if c["checkpoint"] != host.get("checkpoint")]
            if not cands:
                raise Stop("no candidate donors other than the host")
            if not shortlist:
                raise Stop("no matching topics meet box-set minimums")
            # scope first: the topic decides which donor is right, and topic-tagged lessons apply to both choices
            topics = self.decide("scope", f"Which taxonomy topic best matches the goal: {goal}?", {},
                                 [{"id": n["id"], "text": f"{n['id']}: {n.get('name', '')}"} for n in shortlist],
                                 mode="subset", maximum=3, tags=[n["id"] for n in shortlist])
            from .cli import check_topic
            for topic in topics:
                check_topic(topic, paths.default_taxonomy())
            ranked = self.decide(
                "locate", f"Which donor model is most likely to know more than the host about {', '.join(topics)} "
                          f"(goal: {goal})?",
                {"host (receives the skill; not a donor)": host.get("checkpoint")},
                [{"id": c["checkpoint"], "text": _cand_text(c)} for c in cands], mode="rank", tags=topics)
            surviving = {}
            for candidate in ranked:
                for topic in topics:
                    pair = {"candidate": candidate, "topic": topic}
                    result = self.perform("assess", pair)
                    pid = f"pair{len(self.assessed)}"
                    last = self.last_result or {}
                    gap = (result or last).get("gap") or {}
                    # knowledge only from deterministic outcomes: a gap verdict or a plan refusal. A crash, a
                    # blocked admission, another run's lease or any other refusal says nothing about the donor
                    leads = (bool(gap["donor_leads"]) if gap.get("donor_leads") is not None else
                             (False if last.get("plan_refused") else None))
                    self.assessed[pid] = {"pair": pair, "leads": leads, "margin": gap.get("margin"),
                                          "status": last.get("status")}
                    if result and gap.get("donor_leads"):
                        surviving[pid] = {"pair": pair, "gap": {k: gap.get(k) for k in ("donor_leads", "margin",
                                                                                         "offset_nats_per_byte")}}
            if not surviving and all(p["leads"] is None for p in self.assessed.values()):
                outcome = {"status": "stopped", "reason": "no assessment reached a verdict: "
                           + json.dumps([p["status"] for p in self.assessed.values()])}
            elif not surviving:
                outcome = {"status": "refused", "reason": "no surviving donor/topic pairs"}
            else:
                ordered = self.decide(
                    "triage", "Which surviving donor/topic pair should be extracted first?", {},
                    [{"id": pid, "text": f"{s['pair']['candidate']} on {s['pair']['topic']}, gap screen "
                                         + json.dumps(s["gap"])} for pid, s in surviving.items()],
                    mode="rank", tags=topics)
                self.triage_pairs = {pid: s["pair"] for pid, s in surviving.items()}
                for pid in ordered:
                    result = self.perform("run", surviving[pid]["pair"])
                    if not result:
                        continue
                    verdict = ((result.get("bundle") or {}).get("donor_quality") or {}).get("verdict") \
                        or (result.get("bundle") or {}).get("verdict")
                    self.assessed[pid]["verdict"] = verdict
                    outcome = {"status": "completed", "pair": surviving[pid]["pair"], "verdict": verdict,
                               "out": result.get("out")}
                    if self_mode and verdict == "pass":
                        inst = self.perform("install", surviving[pid]["pair"])
                        outcome["install"] = (inst or {}).get("install")
                        outcome["adopted"] = False
                        outcome["adoption"] = ("not adopted: mlx2 has no loader for installed routes; the brain "
                                               "stays at " + self.metadata.get("brain_checkpoint", "?"))
                    break
                else:
                    outcome = {"status": "refused", "reason": "all extractions refused or failed"}
        except KeyboardInterrupt:
            interrupted = True
            outcome = {"status": "interrupted", "partial": outcome}
            self.audit("stop", facts=outcome)
        except (Stop, Exception, SystemExit) as exc:  # noqa: BLE001 - every exit is recorded
            outcome = {"status": "stopped", "reason": f"{type(exc).__name__}: {exc}", "partial": outcome}
            self.audit("stop", facts=outcome)
        finally:
            labels = self.label_decisions()
            lesson = None if interrupted else self.learn(goal, outcome)
            self.audit("outcome", facts={**outcome, "oracle_labels": labels, "lesson": lesson})
            report = {"schema": "mlx2.amoeba-auto.v2", "label": LABEL, **self.metadata, "goal": goal,
                      "self_mode": self_mode, "host": host, "candidates": candidates, "steps": self.steps,
                      "wall_s": time.monotonic() - t0, "decisions": self.records, "oracle_labels": labels,
                      "lesson": lesson, "outcome": outcome}
            if self.tracker:
                report["competence"] = self.tracker.competence()
            write_json(self.out / "auto_report.json", report)
        if interrupted:
            raise KeyboardInterrupt
        return outcome

    def label_decisions(self):
        """Deterministic oracle: the acceptable options for each decision, from what the pipeline found in this run
        and (with a tracker) in earlier runs on the same identities. A decision is labelled only when it could have
        been wrong and could have been right: at least one acceptable option is known, and the chosen option's own
        outcome is known. An option never assessed is unknown, not wrong; the oracle stores the known options, so a
        replay (auto-eval) scores a choice outside them as unknown too."""
        labels = []
        p = self.pipeline

        def lead(cand, topic):
            for a in self.assessed.values():
                if a["pair"] == {"candidate": cand, "topic": topic} and a["leads"] is not None:
                    return a["leads"]
            f = getattr(p, "known_lead", None)
            return f(cand, topic) if callable(f) else None

        def verdict(pair):
            for a in self.assessed.values():
                if a["pair"] == pair and a.get("verdict"):
                    return a["verdict"]
            f = getattr(p, "known_verdict", None)
            return f(pair["candidate"], pair["topic"]) if callable(f) else None

        def put(stage, did, chosen, gold, known, rule):
            if did is None or not gold or chosen[0] not in known:
                return
            ok = chosen[0] in gold
            labels.append({"stage": stage, "decision": did, "chosen": chosen, "gold": sorted(gold),
                           "known": sorted(known), "correct": ok, "rule": rule})
            if self.tracker:
                self.tracker.label(did, {"gold": sorted(gold), "known": sorted(known), "rule": rule}, ok)

        cands = self.menus.get("locate") or [c for c in getattr(p, "candidates", {})
                                             if c != getattr(p, "host", {}).get("checkpoint")]
        topics_all = self.menus.get("scope", [])
        chosen_topics = (self.decided.get("scope") or [(None, [])])[0][1]
        for did, chosen in self.decided.get("scope", []):
            res = {t: [lead(c, t) for c in cands] for t in topics_all}
            # an option's outcome is known when it is acceptable (one lead) or fully assessed (every lead known)
            known = {t for t, v in res.items() if any(x is True for x in v) or all(x is not None for x in v)}
            gold = {t for t, v in res.items() if any(x is True for x in v)}
            put("scope", did, chosen, gold, known, "the first topic has a candidate donor that leads the host")
        for did, chosen in self.decided.get("locate", []):
            res = {c: [lead(c, t) for t in chosen_topics] for c in cands}
            known = {c for c, v in res.items() if any(x is True for x in v) or all(x is not None for x in v)}
            gold = {c for c, v in res.items() if any(x is True for x in v)}
            put("locate", did, chosen, gold, known, "the first donor leads the host on a chosen topic (gap screen)")
        pairs = getattr(self, "triage_pairs", {})
        for did, chosen in self.decided.get("triage", []):
            vs = {pid: verdict(pr) for pid, pr in pairs.items()}
            vs = {pid: v for pid, v in vs.items() if v}
            if len(vs) < 2:        # one known verdict cannot make a ranking wrong
                continue
            best = max(VERDICT_RANK.get(v, -1) for v in vs.values())
            gold = {pid for pid, v in vs.items() if VERDICT_RANK.get(v, -1) == best}
            put("triage", did, chosen, gold, set(vs), "the first pair has the best known extraction verdict")
        return labels

    def learn(self, goal, outcome):
        """One advisory lesson per run, written by the brain from the deterministic outcome, tagged with the topics
        it is about; stored only with a tracker and kept only while it beats the no-lesson arm."""
        if not self.tracker or self.steps >= self.limit or past_deadline():
            return None
        known = [a for a in self.assessed.values() if a["leads"] is not None or a.get("verdict")]
        if not known:
            return None            # nothing deterministic happened: a lesson would only restate a crash
        facts = {"goal": goal, "outcome": {k: outcome.get(k) for k in ("status", "reason", "verdict", "pair")},
                 "assessed": [{k: a.get(k) for k in ("pair", "leads", "margin", "verdict")} for a in known]}
        text = self.write("learn", "A skill-ingestion run just ended. Facts: " + json.dumps(facts, default=str) +
                          "\nWrite one short sentence of advice for choosing donors and topics in future runs. "
                          "Only state what these facts support.", max_tokens=60)
        tags = {a["pair"]["topic"] for a in known}
        lid = self.tracker.add_lesson(text, tags, self.run_id) if text else None
        return {"id": lid, "text": text} if lid else None


def _cand_text(c):
    q = c.get("quantization") or {}
    gb = (c.get("weight_bytes") or 0) / 2**30
    bits = f"{q['bits']}-bit" if q.get("bits") else "bf16/fp16"
    return (f"{c['checkpoint']}: {c.get('kind', '?')} {c.get('model_type') or '?'}, {c.get('layers')} layers, "
            f"hidden {c.get('hidden')}, {bits}, {gb:.1f} GB of weights")


def shortlist_topics(goal, host, out, ids=None):
    """Reuse the actual box-set builder's tokenizer, split and window minimum gates. ``ids``: a fixed candidate list
    (self mode); otherwise keyword match over taxonomy names."""
    from .config import resolve
    cfg = resolve("quick", {}, {}, "")
    comp, data = cfg["compute"], cfg["data"]
    nodes = json.loads(paths.default_taxonomy().read_text())["nodes"]
    if ids:
        by = {n["id"]: n for n in nodes}
        picked = [by[i] for i in ids if i in by]
    else:
        words = {w for w in re.findall(r"[a-z]+", goal.lower()) if len(w) > 2} - {"the", "host", "make", "better"}
        matches = []
        for node in nodes:
            text = " ".join([node["id"], node.get("name", ""), *node.get("aliases", [])]).lower()
            score = len(words & set(re.findall(r"[a-z]+", text)))
            if score and node.get("status") != "deprecated" and not node.get("deprecated"):
                matches.append((score, node))
        picked = [n for _, n in sorted(matches, key=lambda x: (-x[0], x[1]["id"]))[:12]]
    selected = []
    for node in picked:
        dest = out / "shortlist" / node["id"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, paths.repo_root() / "scripts/amoeba_fn_boxset_build.py",
               "--corpus", paths.DEFAULT_PROBE_CORPUS, "--out", dest, "--topic", node["id"],
               "--model", host, "--seq-len", data["window_tokens"], "--prompt-min", data["gen_prompt_tokens"] + 16]
        for opt, name in (("min-fit", "box_min_fit"), ("carve-prompts", "box_carve_prompts"),
                          ("val-prompts", "box_val_prompts"), ("held-prompts", "box_held_prompts"),
                          ("min-carve", "box_min_carve"), ("min-val", "box_min_val"), ("min-held", "box_min_held")):
            cmd += ["--" + opt, comp[name]]
        rc, _, err = child(cmd, log=dest.parent / (node["id"] + ".log"))
        if rc:
            raise Stop("box-set shortlist failed: " + err[-2000:])
        boxes = json.loads((dest / "boxes.json").read_text())["boxes"]
        if boxes:
            selected.append({"id": node["id"], "name": node["name"], "counts": boxes[0]})
    return selected


# Bump when a pipeline change makes earlier gap/extract/install outcomes stale: the epoch is part of every job key.
PIPELINE_EPOCH = 1
# install.py SystemExit messages that are properties of the (box, donor, host) triple, not of the machine or run
INSTALL_POLICY_REFUSALS = ("nothing to install", "the donor is MoE", "tokenizers are not byte-identical",
                           "donor vocabulary", "no units outside the core", "collide under the relative-depth warp")


@functools.lru_cache(maxsize=8)
def _content_sha_cached(path, size, mtime):
    return paths.sha256_file(Path(path))[:16]


def _content_sha(path):
    st = Path(path).stat()
    return _content_sha_cached(str(path), st.st_size, st.st_mtime_ns)


class Pipeline:
    """Deterministic stages. With a tracker, each (host, donor, topic, profile, data) assessment, extraction and
    install is one job keyed by identities: a terminal job's result is reused instead of rerun (a reused extraction
    or install whose directory is gone is rerun), a failed one may be leased again."""

    def __init__(self, args, host, candidates, baseline, audit, tracker=None, run_id="", commit=None):
        self.a, self.baseline, self.audit = args, baseline, audit
        self.out = Path(args.out).expanduser().resolve()
        self.host = host
        self.candidates = {c["checkpoint"]: c for c in candidates}
        self.dirs = {}
        self.attempts = 0
        self.tracker, self.run_id, self.commit = tracker, run_id, commit or {}
        self.owner = f"{run_id}:{id(self)}"
        self._data_ident = None

    def data_ident(self):
        if self._data_ident is None:
            tax, corpus = paths.default_taxonomy(), Path(paths.DEFAULT_PROBE_CORPUS)
            self._data_ident = {"probe_corpus": _content_sha(corpus) if corpus.exists() else str(corpus),
                                "taxonomy": paths.sha256_file(tax)[:16] if tax.exists() else None,
                                "epoch": PIPELINE_EPOCH}
        return self._data_ident

    def key(self, stage, pair):
        from .agent_tracker import job_key
        c = self.candidates[pair["candidate"]]
        extra = {}
        if stage == "install":     # an install is of one box bundle: a re-extracted box is a new install
            d = self.dirs.get((pair["candidate"], pair["topic"]))
            man = Path(d) / "box_bundle/manifest.json" if d else None
            extra["box"] = paths.sha256_file(man)[:16] if man is not None and man.is_file() else None
            extra["install_report_version"] = 2  # earlier reports could score an incomplete route
        return job_key({"assess": "gap", "run": "extract", "install": "install"}[stage],
                       host=self.host["fingerprint"], donor=c["fingerprint"], topic=pair["topic"], profile="quick",
                       probes_per_node=2, **self.data_ident(), **extra)

    def beat(self):
        """Renew the lease before each child, so a stage of several children never outlives its lease."""
        if self.tracker is not None and getattr(self, "_leased", None):
            self.tracker.heartbeat(self._leased, self.owner)

    def _terminal(self, stage, candidate, topic):
        if self.tracker is None or candidate not in self.candidates:
            return None
        job = self.tracker.job(self.key(stage, {"candidate": candidate, "topic": topic}))
        return job if job and job["state"] in ("done", "refused") else None

    def known_lead(self, candidate, topic):
        """An earlier run's gap-screen outcome for this exact host, donor, topic and data: True / False / None. Only
        a gap verdict or a plan refusal is knowledge."""
        job = self._terminal("assess", candidate, topic)
        if not job:
            return None
        res = job["result"] or {}
        gap = res.get("gap") or {}
        if gap.get("donor_leads") is not None:
            return bool(gap["donor_leads"])
        return False if res.get("plan_refused") else None

    def known_verdict(self, candidate, topic):
        job = self._terminal("run", candidate, topic)
        dq = ((job or {}).get("result") or {}).get("bundle") or {}
        if not job:
            return None
        return (dq.get("donor_quality") or {}).get("verdict") or dq.get("verdict")

    @staticmethod
    def _has_output(stage, result):
        if stage == "assess":
            return True  # gap evidence is held in the tracker itself
        if not result.get("out"):
            return False
        root = Path(result["out"])
        try:
            if stage == "run":
                bundle = json.loads((root / "box_bundle/manifest.json").read_text())
                return isinstance(bundle, dict) and bundle == result.get("bundle")
            report = json.loads((root / "report/manifest.json").read_text())
            verdicts = {v["route"]: v["verdict"] for v in report["verdicts"]}
            return bool(verdicts.get("side") and verdicts["side"] != "failed"
                        and verdicts == (result.get("install") or {}).get("verdicts"))
        except (OSError, ValueError, TypeError, KeyError):
            return False

    def __call__(self, stage, pair):
        if self.tracker is None:
            return self._run(stage, pair)
        from .agent_tracker import TERMINAL
        k = self.key(stage, pair)
        job = self.tracker.enqueue(stage, k, pair, self.run_id)
        if job["state"] in TERMINAL:
            res = dict(job["result"] or {})
            if job["state"] == "done" and not self._has_output(stage, res):
                self.tracker.reset(k)
                self.audit("job_requeued", stage=stage, facts={"key": k, "reason": "output manifest missing or changed"})
            else:
                if res.get("out") and stage != "install":
                    self.dirs[(pair["candidate"], pair["topic"])] = Path(res["out"])
                self.audit("job_reused", stage=stage, facts={"key": k, "state": job["state"], "run": job["run_id"]})
                return {**res, "status": "reused" if job["state"] == "done" else "refused"}
        if not self.tracker.lease(k, self.owner, self.run_id):
            return {"status": "busy", "reason": f"job {k} is leased by another run or out of attempts this run"}
        self._leased = k
        try:
            res = self._run(stage, pair)
        except BaseException as exc:
            try:
                self.tracker.finish(k, self.owner, "failed", {"error": f"{type(exc).__name__}: {exc}"})
            except RuntimeError:   # the lease was lost: keep the original error
                pass
            raise
        finally:
            self._leased = None
        if res["status"] == "memory_stop":
            self.tracker.release(k, self.owner)
            return res
        state = {"ok": "done", "refused": "refused"}.get(res["status"], "failed")
        slim = {x: res.get(x) for x in ("status", "reason", "gap", "bundle", "install", "out", "plan_refused")}
        try:
            self.tracker.finish(k, self.owner, state, slim)
        except RuntimeError as exc:    # another run took the lease: keep this run's result, record nothing shared
            self.audit("lease_lost", stage=stage, facts={"key": k, "error": str(exc)})
            return res
        commit = self.commit.get("commit", "")
        if res.get("gap"):
            self.tracker.add_fact(k, "gap", res["gap"], self.run_id, commit)
        if res.get("bundle"):
            dq = res["bundle"].get("donor_quality") or {}
            self.tracker.add_fact(k, "box_verdict", {"verdict": dq.get("verdict") or res["bundle"].get("verdict"),
                                                     "pair": pair, "held": dq.get("export_held")},
                                  self.run_id, commit)
        if state == "done" and res.get("install"):
            self.tracker.add_fact(k, "install_verdict", res["install"], self.run_id, commit)
        return res

    def _extract_cmd(self, base, model, topic, dest, gap_only):
        return base + ["extract", "--model", model, "--against", self.a.host, "--topic", topic, "--out", dest,
                       *(["--gap-only"] if gap_only else []), "--profile", "quick", "--gen-backend", "local",
                       "--compute", "map_probes_per_node=2"]

    def _run(self, stage, pair):
        try:
            facts = admit_memory(self.baseline)
        except Stop as exc:
            return {"status": "memory_stop", "reason": str(exc)}
        self.audit("memory_admission", facts={k: facts[k] for k in ("free_gb", "swap_gb")})
        model = self.candidates[pair["candidate"]]["path"]
        base = [sys.executable, "-m", "mlx2.experimental.amoeba.cli"]
        env = paths.child_env({"MLX2_AMOEBA_AUTO_SWAP_BASELINE": str(self.baseline)})
        self.attempts += 1
        n = self.attempts
        pair_key = (pair["candidate"], pair["topic"])
        plan = None
        if stage == "assess":
            self.beat()
            rc, out, err = child(base + ["plan", "--model", model, "--profile", "quick", "--stream", "resident",
                                         "--json"], env=env)
            if rc:                     # rc 2 = the resource plan refuses: deterministic, this donor cannot run here
                return {"status": "refused" if rc == 2 else "crash", "plan_refused": rc == 2,
                        "reason": (out + err)[-4000:]}
            plan = json.loads(out)
            dest = self.out / f"pair-{n}"
            self.dirs[pair_key] = dest
            cmd = self._extract_cmd(base, model, pair["topic"], dest, True)
        elif stage == "run":
            dest = self.dirs[pair_key]
            if self.out in Path(dest).resolve().parents:
                cmd = base + ["resume", "--out", dest, "--stages", "core,map,box,box_gate,box_bundle"]
            else:                      # the gap screen was reused from another run: never write into its files
                dest = self.out / f"pair-{n}"
                self.dirs[pair_key] = dest
                cmd = self._extract_cmd(base, model, pair["topic"], dest, False)
        elif stage == "install":
            src = self.dirs[pair_key]
            dest = self.out / f"install-{n}"
            cmd = base + ["install", "--box", src / "box_bundle", "--donor", model, "--host", self.a.host,
                          "--out", dest, "--routes", "side"]
        else:
            raise ValueError(stage)
        cmd += ["--metrics-port", "0", "--quiet"]
        if DEADLINE["t"] is not None:    # the stages' own clean stop (no stage starts after it; a carve checkpoints)
            cmd += ["--stop-at", str(int(DEADLINE["t"]))]
        if self.a.ignore_gpu_queue:
            cmd += ["--ignore-gpu-queue", self.a.ignore_gpu_queue]
        if self.a.gpu_window_owner:
            cmd += ["--gpu-window-owner", self.a.gpu_window_owner]
        if self.a.allow_dirty:
            cmd += ["--allow-dirty"]
        self.beat()
        rc, out, err = child(cmd, env=env, log=self.out / f"pipeline-{stage}-{n}.log",
                             grace_s=PIPELINE_GRACE_S)
        blocked = _admission_block(dest)
        if rc and (blocked or "memory admission refused" in err):
            return {"status": "memory_stop", "reason": "pipeline admission: " + (blocked or err[-1000:]),
                    "out": str(dest)}
        receipts = {p.stem: json.loads(p.read_text()) for p in (dest / "receipts").glob("*.json")} \
            if (dest / "receipts").exists() else {}
        if stage == "install":
            man = dest / "report/manifest.json"
            inst = json.loads(man.read_text()) if man.exists() else None
            # terminal only for install's policy refusals, properties of the box, donor and host that no retry
            # changes; guards (GPU window, dirty tree, memory cap) and crashes are failures a later run may retry
            policy = rc and any(m in err for m in INSTALL_POLICY_REFUSALS)
            verdicts = {v["route"]: v["verdict"] for v in (inst or {}).get("verdicts", [])}
            complete = verdicts.get("side") and verdicts["side"] != "failed"
            status = "ok" if rc == 0 and complete else ("refused" if policy else "crash")
            reason = err[-4000:] if rc else ("" if complete else f"side route failed or missing; see {dest}")
            return {"status": status, "reason": reason, "out": str(dest),
                    "install": {"verdicts": verdicts, "manifest": str(man)} if inst else None}
        gap_path = dest / "gap/gap.json"
        gap = json.loads(gap_path.read_text()) if gap_path.exists() else None
        bundle_path = dest / "box_bundle/manifest.json"
        bundle = json.loads(bundle_path.read_text()) if bundle_path.exists() else None
        # refused (terminal, reused by later runs) only for the gap verdict; every other non-zero exit, including
        # guard refusals recorded in receipts, is a failure that a later run may retry
        status = "refused" if gap and gap.get("donor_leads") is False else ("ok" if rc == 0 else "crash")
        return {"status": status, "reason": err[-4000:] if rc else "", "plan": plan, "gap": gap,
                "receipts": receipts, "bundle": bundle, "out": str(dest)}


def _admission_block(dest):
    """The research scripts' own memory admission (scripts/amoeba27b_lib.admit: no second large model process beside
    a foreign one over 8 GB, coordinator rule 2026-11-14). It is a standing rule, not a GPU-queue wait, so
    --ignore-gpu-queue does not bypass it; the run stops and says which process blocked it."""
    logs = Path(dest) / "logs"
    if not logs.exists():
        return None
    for lg in sorted(logs.glob("*.log")):
        text = lg.read_text(errors="replace")
        if "admit: another large process is running" in text:
            waits = re.findall(r"admit: waiting, other large processes (.*)", text)
            return "another large process is running " + (waits[-1][:300] if waits else "")
    return None


def self_candidates(brain_info, candidates):
    """Self mode: a donor must share the brain's tokenizer (install pairs tokens by id) and be dense (install refuses
    MoE donors). Returns (kept, dropped with reasons): deterministic, never the brain's call."""
    from .cli import _same_tokenizer
    kept, dropped = [], []
    for c in candidates:
        if c["checkpoint"] == brain_info["checkpoint"]:
            dropped.append({"checkpoint": c["checkpoint"], "reason": "is the brain itself"})
        elif _same_tokenizer(c["path"], brain_info["path"]) is not True:
            dropped.append({"checkpoint": c["checkpoint"], "reason": "tokenizer differs from the brain's"})
        elif c.get("kind") == "moe":
            dropped.append({"checkpoint": c["checkpoint"], "reason": "MoE donor (install refuses MoE box units)"})
        else:
            kept.append(c)
    return kept, dropped


def run_auto(a):
    from .agent_tracker import Tracker
    from .cli import check_commit
    started = time.monotonic()
    commit = check_commit(a.allow_dirty)
    require_gpu_window(False, a.gpu_window_owner, ignore_queue=a.ignore_gpu_queue)
    if getattr(a, "minutes", 0):
        DEADLINE["t"] = time.time() + 60 * a.minutes
    self_mode = bool(getattr(a, "self_mode", False))
    if self_mode:
        a.host = a.brain
        a.goal = a.goal or SELF_GOAL
    if not a.goal:
        raise SystemExit("--goal is required (or --self)")
    # Inspection loads no weights. The first admission is before any model child.
    host = inspect_checkpoint(a.host)
    brain = inspect_checkpoint(a.brain)
    candidates = [inspect_checkpoint(p) for p in a.candidates]
    if len({c["checkpoint"] for c in candidates}) != len(candidates):
        raise Stop("duplicate candidate identity")
    dropped = []
    if self_mode:
        candidates, dropped = self_candidates(brain, candidates)
    tracker = None if a.no_tracker else Tracker(a.tracker or None)
    run_id = f"{Path(a.out).name}-{int(time.time())}"
    agent = Agent(a.out, a.max_steps, None, None,
                  {"brain_checkpoint": brain["checkpoint"], "brain_fingerprint": brain["fingerprint"],
                   "mlx2_commit": commit, "ignore_gpu_queue": a.ignore_gpu_queue, "status": "unqualified",
                   "tracker": str(tracker.path) if tracker else None, "run_id": run_id,
                   "dropped_candidates": dropped, "deadline": DEADLINE["t"]},
                  tracker=tracker, run_id=run_id)
    if tracker:
        tracker.start_run(run_id, mode="self" if self_mode else "goal", goal=a.goal, brain=brain["checkpoint"],
                          commit=commit.get("commit", ""), out=a.out)
    outcome = {"status": "stopped"}
    try:
        initial_memory = memory_snapshot()
        baseline = initial_memory["swap_gb"]
        agent.metadata["initial_memory"] = {k: initial_memory[k] for k in ("free_gb", "swap_gb")}
        agent.audit("start", facts={"host": host, "candidates": candidates, "dropped": dropped,
                                    "memory": agent.metadata["initial_memory"]})
        agent.brain = Brain(a.brain, baseline, agent.audit, a.gpu_window_owner, a.ignore_gpu_queue)
        agent.pipeline = Pipeline(a, host, candidates, baseline, agent.audit, tracker, run_id, commit)
        shortlist = shortlist_topics(a.goal, a.host, agent.out, ids=SELF_TOPICS if self_mode else None)
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        agent.audit("setup_failure", facts={"error": str(exc)})
        outcome = {"status": "stopped", "reason": str(exc)}
        write_json(agent.out / "auto_report.json", {**agent.metadata, "label": LABEL, "outcome": outcome,
                   "wall_s": time.monotonic() - started, "decisions": agent.records})
        if tracker:
            tracker.finish_run(run_id, outcome)
        return 2
    outcome = {"status": "interrupted"}
    try:
        outcome = agent.run(a.goal, host, candidates, shortlist, self_mode=self_mode)
    finally:
        if tracker:
            tracker.finish_run(run_id, outcome)
    report = json.loads((agent.out / "auto_report.json").read_text())
    report["total_wall_s"] = time.monotonic() - started
    write_json(agent.out / "auto_report.json", report)
    print(json.dumps(outcome, indent=1, default=str))
    return 0 if outcome["status"] in ("completed", "refused") else 2


def run_status(a):
    from .agent_tracker import Tracker
    t = Tracker(a.tracker or None)
    for key in getattr(a, "reset", []) or []:
        print(json.dumps({"reset": key, "ok": t.reset(key)}))
    print(json.dumps({"tracker": str(t.path), "competence": t.competence(),
                      "lessons": [{k: x[k] for k in ("id", "text", "shown", "helped", "hurt", "retired")}
                                  for x in t.lessons()],
                      "open_jobs": [{k: j[k] for k in ("key", "kind", "state", "attempts")}
                                    for j in t.jobs() if j["state"] not in ("done", "refused")]}, indent=1))
    return 0


def run_eval(a):
    """Replay every oracle-labelled decision through another brain checkpoint: the same prompts and menus (the
    lessons shown then are part of the stored prompt), the same selection rule. Prints accuracy per stage for the new
    brain beside the recorded one. No pipeline runs; only the brain child loads."""
    from .agent_tracker import Tracker
    from .cli import check_commit
    commit = check_commit(getattr(a, "allow_dirty", False))
    require_gpu_window(False, a.gpu_window_owner, ignore_queue=a.ignore_gpu_queue)
    t = Tracker(a.tracker or None)
    info = inspect_checkpoint(a.brain)
    baseline = memory_snapshot()["swap_gb"]
    brain = Brain(a.brain, baseline, lambda *x, **k: None, a.gpu_window_owner, a.ignore_gpu_queue)
    rows = [d for d in t.decisions(labelled_only=True) if (d["oracle"] or {}).get("known")]
    rows = rows[-a.limit:] if a.limit else rows
    per = {}
    for d in rows:
        opts = d["options"]
        reply = brain({"mode": "choose", "prompt": d["prompt"], "n": len(opts["ids"]), "why_tokens": 0})
        idx = select(check_scores(reply, len(opts["ids"])), opts["mode"], opts.get("maximum"))
        first = opts["ids"][idx[0]]
        gold, known = set(d["oracle"]["gold"]), set(d["oracle"]["known"])
        s = per.setdefault(d["stage"], {"n": 0, "candidate_known": 0, "candidate_correct": 0, "recorded": {}})
        s["n"] += 1
        rec = s["recorded"].setdefault(d["brain"], {"n": 0, "correct": 0})
        rec["n"] += 1
        rec["correct"] += int(d["correct"])
        if first in known:     # a choice the oracle never observed is unknown, not wrong
            s["candidate_known"] += 1
            s["candidate_correct"] += int(first in gold)
            # paired comparison on the same rows: the recorded decision and the candidate's
            s["paired_recorded_correct"] = s.get("paired_recorded_correct", 0) + int(d["correct"])
            agree = first == d["chosen"][0]
            s["paired_agree"] = s.get("paired_agree", 0) + int(agree)
    for s in per.values():
        s["candidate_accuracy_on_known"] = (round(s["candidate_correct"] / s["candidate_known"], 3)
                                            if s["candidate_known"] else None)
        s["coverage"] = round(s["candidate_known"] / s["n"], 3)
        s["paired_recorded_accuracy"] = (round(s.get("paired_recorded_correct", 0) / s["candidate_known"], 3)
                                         if s["candidate_known"] else None)
        s["paired_agree_rate"] = (round(s.get("paired_agree", 0) / s["candidate_known"], 3)
                                  if s["candidate_known"] else None)
    print(json.dumps({"label": LABEL, "status": "unqualified", "mlx2_commit": commit,
                      "candidate_brain": info["checkpoint"], "candidate_fingerprint": info["fingerprint"],
                      "tracker": str(t.path), "labelled_decisions": len(rows), "per_stage": per}, indent=1))
    return 0


# brain child -------------------------------------------------------------------------------------------------------

def _letter_ids(gen, n):
    ids = []
    for ch in LETTERS[:n]:
        enc = gen.encode(ch)
        if len(enc) != 1:
            raise SystemExit(f"brain tokenizer splits the label {ch!r}; menu scoring needs single-token labels")
        ids.append(enc[0])
    return ids


def _first_token_logprobs(gen, rendered):
    import mlx.core as mx
    ids = gen.encode(rendered)
    make = getattr(gen.model, "make_cache", None)
    cache = make() if callable(make) else None
    logits = gen.model(mx.array([ids]), cache=cache)[:, -1, :].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(lp)
    return lp[0]


def _render(gen, prompt):
    return gen.t.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)


def _reorder(prompt, order):
    """The same prompt with its option lines in another order, relettered."""
    head, _, rest = prompt.partition("\nOptions:\n")
    body, _, tail = rest.rpartition("\nAnswer with")
    lines = body.split("\n")
    texts = [ln.split(". ", 1)[1] for ln in lines]
    new = [f"{LETTERS[j]}. {texts[i]}" for j, i in enumerate(order)]
    return head + "\nOptions:\n" + "\n".join(new) + "\nAnswer with" + tail


def brain_choose(gen, prompt, n, why_tokens):
    """Average letter log-probabilities over all n cyclic shifts of the menu, mapped back to the options: every
    option is scored once under every letter, so the letter prior adds the same amount to each option."""
    letter = _letter_ids(gen, n)
    sums = [0.0] * n
    passes = [[(j + s) % n for j in range(n)] for s in range(n)]
    for order in passes:
        lp = _first_token_logprobs(gen, _render(gen, _reorder(prompt, order)))
        for pos, opt in enumerate(order):
            sums[opt] += float(lp[letter[pos]].item())
    scores = [s / len(passes) for s in sums]
    why = ""
    if why_tokens:
        best = max(range(n), key=lambda i: scores[i])
        why, _ = gen.complete(_render(gen, prompt) + f"{LETTERS[best]}. Reason:", why_tokens, 0.0, 0)
        why = why.split("\n")[0].strip()
    return {"scores": scores, "why": why, "passes": len(passes)}


def brain_child(model, request):
    from .child_gen_server import Generator
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        gen = Generator(model, False)
        if request["mode"] == "choose":
            out = brain_choose(gen, request["prompt"], int(request["n"]), int(request.get("why_tokens", 0)))
        elif request["mode"] == "write":
            text, _ = gen.complete(_render(gen, request["prompt"]), int(request.get("max_tokens", 80)), 0.0, 0)
            out = {"text": text.strip().split("\n")[0]}
        else:
            raise SystemExit(f"unknown brain mode {request['mode']}")
    print(json.dumps(out), end="", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--brain-child":
        brain_child(sys.argv[2], json.loads(sys.stdin.read()))
    else:
        raise SystemExit("internal brain child only; use mlx2 amoeba auto")
