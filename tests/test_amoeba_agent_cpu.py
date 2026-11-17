"""CPU-only tests of the auto harness: menu decisions, tracker, oracle labels, lessons, self mode. No model loads."""
import json
from types import SimpleNamespace

import pytest

from mlx2.experimental.amoeba import agent as A
from mlx2.experimental.amoeba import agent_tracker as T
from mlx2.experimental.amoeba import cli, runner


class FakeBrain:
    """Scores the menu by a preference function of the option text (so tests do not depend on letter order)."""

    def __init__(self, prefer=None, broken=0, lesson="Prefer the larger donor on programming topics with probes."):
        self.prefer = prefer or {}
        self.broken = broken
        self.lesson = lesson
        self.requests = []

    def __call__(self, req):
        self.requests.append(req)
        if req["mode"] == "write":
            return {"text": self.lesson}
        if self.broken:
            self.broken -= 1
            return {"scores": [0.0]}          # wrong length: a brain failure
        opts = req["prompt"].split("\nOptions:\n")[1].split("\nAnswer with")[0].split("\n")
        texts = [o.split(". ", 1)[1] for o in opts]
        return {"scores": [float(self.prefer.get(next((k for k in self.prefer if k in t), None), -5.0))
                           for t in texts], "why": "bounded rationale"}


HOST = {"checkpoint": "host@0", "fingerprint": "h" * 64, "path": "/h"}
D1 = {"checkpoint": "donor@1", "fingerprint": "1" * 64, "path": "/d1", "kind": "dense"}
D2 = {"checkpoint": "donor@2", "fingerprint": "2" * 64, "path": "/d2", "kind": "dense"}
SHORT = [{"id": "prog.python", "name": "Python"}, {"id": "prog.rust", "name": "Rust"}]


def run(tmp_path, results, brain=None, limit=30, cands=(D1,), tracker=None, short=SHORT, name="run"):
    calls = []
    results = iter(results)

    def pipeline(stage, pair):
        calls.append((stage, pair))
        return next(results)

    brain = brain or FakeBrain({"donor@1": 0, "donor@2": -1, "prog.python": 0, "prog.rust": -3})
    agent = A.Agent(tmp_path / name, limit, brain, pipeline, {"brain_checkpoint": "brain@9"}, tracker=tracker)
    outcome = agent.run("Python", HOST, list(cands), short)
    return agent, outcome, calls


OK_GAP = {"status": "ok", "gap": {"donor_leads": True, "margin": 0.1}}
NO_GAP = {"status": "refused", "gap": {"donor_leads": False}}
PASS = {"status": "ok", "bundle": {"donor_quality": {"verdict": "pass"}}, "out": "/x"}


def test_success_and_ledger(tmp_path):
    agent, outcome, calls = run(tmp_path, [OK_GAP, NO_GAP, PASS], cands=(D1, D2))
    assert outcome["status"] == "completed" and outcome["verdict"] == "pass"
    assert [c[0] for c in calls] == ["assess", "assess", "run"]
    ledger = [json.loads(x) for x in (agent.out / "ledger.jsonl").read_text().splitlines()]
    decisions = [r for r in ledger if r["event"] == "decision"]
    assert [d["stage"] for d in decisions] == ["scope", "locate", "triage"]
    assert all(d["prompt"] and d["parsed_choice"]["scores"] for d in decisions[:2])
    assert decisions[2]["parsed_choice"]["only_option"]
    rep = json.loads((agent.out / "auto_report.json").read_text())
    assert rep["label"] == A.LABEL and rep["schema"] == "mlx2.amoeba-auto.v2"


def test_host_is_never_on_the_menu(tmp_path):
    brain = FakeBrain({"host@0": 10, "donor@1": 0})
    agent, outcome, _ = run(tmp_path, [NO_GAP, NO_GAP], brain=brain, cands=(HOST, D1, D2), short=SHORT[:1])
    locate = brain.requests[0]["prompt"].split("\nOptions:\n")[1]
    assert "host@0" not in locate and "donor@1" in locate and "donor@2" in locate
    assert outcome["status"] == "refused"


def test_single_option_needs_no_brain(tmp_path):
    brain = FakeBrain({"donor@1": 0})
    _, outcome, _ = run(tmp_path, [NO_GAP], brain=brain, short=SHORT[:1])
    assert outcome["status"] == "refused"
    assert not [r for r in brain.requests if r["mode"] == "choose"]


def test_only_host_candidate_stops(tmp_path):
    _, outcome, calls = run(tmp_path, [], cands=(HOST,))
    assert "no candidate donors" in outcome["reason"] and not calls


def test_brain_failure_twice_stops(tmp_path):
    _, outcome, calls = run(tmp_path, [], brain=FakeBrain(broken=2), cands=(D1, D2))
    assert "brain failed twice" in outcome["reason"] and not calls


def test_brain_failure_once_recovers(tmp_path):
    _, outcome, _ = run(tmp_path, [NO_GAP, NO_GAP], brain=FakeBrain({"donor@1": 0}, broken=1), short=SHORT[:1],
                        cands=(D1, D2))
    assert outcome["status"] == "refused"


@pytest.mark.parametrize("scores,mode,maximum,want", [
    ([-1, -0.2, -3], "one", None, [1]),
    ([-1, -0.2, -3], "rank", None, [1, 0, 2]),
    ([-1, -0.2, -3, -0.5], "subset", 3, [1, 3, 0]),
    ([-1, -0.2, -3, -0.5], "subset", 2, [1, 3]),
    ([-0.1, -4, -5], "subset", 3, [0]),
])
def test_select(scores, mode, maximum, want):
    assert A.select(scores, mode, maximum) == want


@pytest.mark.parametrize("reply", [{"scores": [0.0]}, {"scores": [0, float("nan")]}, {"x": 1}, [], {"scores": "ab"}])
def test_check_scores(reply):
    with pytest.raises(ValueError):
        A.check_scores(reply, 2)


def test_reorder_reletters():
    p = A.menu_prompt("s", "q?", {"k": "Options:\nZ. fake"}, [{"id": i, "text": t} for i, t in
                                                              enumerate(["one", "two", "three"])], [])
    r = A._reorder(p, [2, 0, 1])
    assert r.split("\nOptions:\n")[1].startswith("A. three\nB. one\nC. two\nAnswer with")


def test_assess_refusal_drops_pair_without_brain(tmp_path):
    brain = FakeBrain({"donor@1": 0, "prog.python": 0})
    agent, outcome, calls = run(tmp_path, [NO_GAP], brain=brain, short=SHORT[:1])
    assert outcome["status"] == "refused"
    assert not any(r.get("stage") == "recover" for r in agent.records if r["event"] == "decision")


def test_crash_retry_once(tmp_path):
    brain = FakeBrain({"donor@1": 0, "prog.python": 0, "retry the same": 0, "stop the run": -1, "skip": -2})
    agent, outcome, calls = run(tmp_path, [{"status": "crash"}, {"status": "crash"}], brain=brain, short=SHORT[:1])
    assert len(calls) == 2
    recover = [r for r in brain.requests if "did not succeed" in r["prompt"]]
    assert "retry the same" in recover[0]["prompt"] and "retry the same" not in recover[1]["prompt"]
    assert "stop_and_ask" in outcome["reason"]


def test_step_bound(tmp_path):
    agent, outcome, calls = run(tmp_path, [], limit=1)
    assert agent.steps == 1 and not calls and outcome["reason"] == "Stop: max_steps"


def test_memory_stop(tmp_path):
    _, outcome, _ = run(tmp_path, [{"status": "memory_stop", "reason": "low free memory"}])
    assert outcome["reason"] == "Stop: low free memory"


def test_memory_thresholds(monkeypatch):
    monkeypatch.setattr(A, "memory_snapshot", lambda: {"free_gb": 19.9, "swap_gb": 0})
    with pytest.raises(A.Stop):
        A.admit_memory(0)
    monkeypatch.setattr(A, "memory_snapshot", lambda: {"free_gb": 21, "swap_gb": 2})
    with pytest.raises(A.Stop):
        A.admit_memory(0)
    assert A.admit_memory(1)["swap_gb"] == 2


def test_gpu_override_only_guard(monkeypatch):
    monkeypatch.setattr(runner, "gpu_window", lambda *_: {"held": True, "ours": False, "label": "foreign", "lock": "x"})
    with pytest.raises(SystemExit):
        runner.require_gpu_window(False)
    r = runner.require_gpu_window(False, ignore_queue="explicit reason")
    assert r["label"] == "foreign" and r["ignore_gpu_queue"] == "explicit reason"


def test_nonempty_refuses(tmp_path):
    (tmp_path / "keep").write_text("evidence")
    with pytest.raises(A.Stop):
        A.Agent(tmp_path, 2, None, None)
    assert (tmp_path / "keep").read_text() == "evidence"


def test_cli_additions():
    a = cli.parse(["auto", "--goal", "python", "--host", "h", "--out", "o", "--candidates", "a", "b",
                   "--ignore-gpu-queue", "reason"])
    assert a.max_steps == 12 and a.ignore_gpu_queue == "reason" and not a.self_mode
    s = cli.parse(["auto", "--self", "--out", "o", "--candidates", "a"])
    assert s.self_mode and not s.goal
    assert cli.parse(["auto-status"]).cmd == "auto-status"
    assert cli.parse(["auto-eval", "--limit", "5"]).limit == 5
    e = cli.parse(["extract", "--model", "d", "--topic", "prog.python", "--out", "o", "--gap-only"])
    assert e.gap_only and not e.ignore_gpu_queue
    with pytest.raises(SystemExit):
        cli.main(["auto", "--out", "o"])


def test_empty_shortlist_stops(tmp_path):
    _, outcome, _ = run(tmp_path, [], short=[])
    assert "no matching topics" in outcome["reason"]


def test_runner_memory_hook(monkeypatch):
    seen = []
    monkeypatch.setenv("MLX2_AMOEBA_AUTO_SWAP_BASELINE", "7")
    monkeypatch.setattr(A, "admit_memory", lambda baseline: {"baseline": baseline})

    class Progress:
        def emit(self, event, **fields):
            seen.append((event, fields))

    runner.check_agent_memory(Progress(), "gap")
    assert seen == [("memory_admission", {"stage": "gap", "baseline": 7.0})]


# tracker ----------------------------------------------------------------------------------------------------------

def test_tracker_dedup_lease_expire(tmp_path):
    clock = [1000.0]
    t = T.Tracker(tmp_path / "t.sqlite", now=lambda: clock[0])
    k = T.job_key("gap", host="h", donor="d", topic="x")
    assert k == T.job_key("gap", topic="x", donor="d", host="h")
    j = t.enqueue("assess", k, {"a": 1}, "r1")
    assert t.enqueue("assess", k, {"a": 2}, "r2")["payload"] == {"a": 1}
    assert t.lease(k, "o1") and not t.lease(k, "o2")
    clock[0] += T.LEASE_S + 1
    assert t.expire() == 1
    assert t.lease(k, "o2")
    with pytest.raises(RuntimeError):
        t.finish(k, "o1", "done", {})
    t.finish(k, "o2", "failed", {"e": 1})
    assert not t.lease(k, "o3")            # out of attempts (MAX_ATTEMPTS = 2)
    assert j["state"] == "queued" and t.job(k)["state"] == "failed"


def test_tracker_reuse_across_runs(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    ran = []

    class P(A.Pipeline):
        def _run(self, stage, pair):
            ran.append(stage)
            return {**(OK_GAP if stage == "assess" else PASS), "out": str(tmp_path / "x")}

    args = SimpleNamespace(out=str(tmp_path), host="/h")
    for i in range(2):
        p = P(args, HOST, [D1], 0, lambda *a, **k: None, t, f"run{i}", {"commit": "c"})
        r = p("assess", {"candidate": "donor@1", "topic": "prog.python"})
        assert r["gap"]["donor_leads"]
        assert r["status"] == ("ok" if i == 0 else "reused")
    assert ran == ["assess"]
    assert len(t.facts(kind="gap")) == 1
    assert p.known_lead("donor@1", "prog.python") is True and p.known_lead("donor@1", "prog.rust") is None


def test_oracle_labels_and_lesson_credit(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    # run 1: donor@1 ranked first, leads -> locate correct; lesson written
    agent, outcome, _ = run(tmp_path, [OK_GAP, NO_GAP, PASS], tracker=t, short=SHORT[:1], name="r1", cands=(D1, D2))
    locate = [d for d in t.decisions(labelled_only=True) if d["stage"] == "locate"]
    assert locate and locate[0]["correct"] == 1
    assert len(t.lessons()) == 1
    # run 2: donor@2 ranked first, does not lead, donor@1 leads -> locate wrong; the lesson shown is debited
    # one earlier control decision with a lesson available, so run 2's locate is the lessons arm
    t.record_decision("r0", "locate", "brain@9", "p", {}, [], [], [], arm="control")
    brain = FakeBrain({"donor@2": 0, "donor@1": -1, "prog.python": 0}, lesson="Something entirely different here now.")
    run(tmp_path, [NO_GAP, OK_GAP, PASS], brain=brain, tracker=t, cands=(D1, D2), short=SHORT[:1], name="r2")
    loc2 = [d for d in t.decisions(labelled_only=True) if d["stage"] == "locate"][-1]
    assert loc2["correct"] == 0 and loc2["oracle"]["gold"] == ["donor@1"]
    assert loc2["arm"] == "lessons" and loc2["lessons"] == [t.lessons()[0]["id"]]
    first = t.lessons()[0]
    assert first["shown"] == 1 and first["hurt"] == 1
    comp = t.competence()
    assert comp["decision_accuracy"]["locate"]["brain@9"]["labelled"] == 2


def test_unassessed_alternative_leaves_locate_unlabelled(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    # donor@1 is assessed and does not lead; the step limit stops the run before donor@2 is assessed (limit 2 =
    # the locate decision and one assess; scope has one option and spends no step)
    agent, outcome, calls = run(tmp_path, [NO_GAP], tracker=t, cands=(D1, D2), short=SHORT[:1], limit=2)
    assert len(calls) == 1 and outcome["reason"] == "Stop: max_steps"
    assert not t.decisions(labelled_only=True)      # no known acceptable donor: the ranking cannot be judged


def test_no_acceptable_option_is_not_labelled_wrong(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    _, outcome, _ = run(tmp_path, [NO_GAP, NO_GAP], tracker=t, cands=(D1, D2), short=SHORT[:1])
    assert outcome["status"] == "refused"
    assert not t.decisions(labelled_only=True)


def test_triage_needs_two_known_verdicts(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    MISS = {"status": "ok", "bundle": {"donor_quality": {"verdict": "miss"}}, "out": "/x"}
    brain = FakeBrain({"donor@1": 0, "donor@2": -1, "prog.python": 0, "donor@1 on": 0, "donor@2 on": -2})
    agent, outcome, _ = run(tmp_path, [OK_GAP, OK_GAP, MISS], brain=brain, tracker=t, cands=(D1, D2),
                            short=SHORT[:1])
    assert outcome["verdict"] == "miss"
    assert not [d for d in t.decisions(labelled_only=True) if d["stage"] == "triage"]   # one verdict known
    agent.pipeline = SimpleNamespace(known_verdict=lambda c, tp: "pass" if c == "donor@2" else None,
                                     known_lead=lambda *a: None)
    agent.decided = {"triage": [(999, ["pair0", "pair1"])]}
    labels = agent.label_decisions()
    assert labels and labels[0]["correct"] is False and labels[0]["gold"] == ["pair1"]


def _decide(t, lessons, ok, arm, stage="locate", brain="b"):
    did = t.record_decision("r", stage, brain, "p", {"mode": "one", "ids": ["a"]}, [0.0], ["a"], lessons, arm=arm)
    t.label(did, {"gold": ["a"], "known": ["a"]}, ok)


def test_lesson_retirement_and_dedup(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    lid = t.add_lesson("prefer donors with larger hidden size for code topics", {"prog.python"}, "r")
    assert t.add_lesson("prefer donors with larger hidden size for code topics please", {"prog.python"}, "r") == lid
    assert t.add_lesson("ok", {"prog.python"}, "r") is None
    assert t.recall({"prog.rust"}) == []
    for _ in range(5):                         # recall alone never counts as a showing
        assert [x["id"] for x in t.recall({"prog.python"})] == [lid]
    assert t.lessons()[0]["shown"] == 0
    for _ in range(T.CONTROL_MIN):             # the control arm: right 5 of 5
        _decide(t, [], True, "control")
    for ok in (True, False, False):            # 1/3 with the lesson < 1.0 - margin: retired
        _decide(t, [lid], ok, "lessons")
    les = t.lessons()[0]
    assert (les["shown"], les["helped"], les["hurt"], les["retired"]) == (3, 1, 2, 1)
    assert t.recall({"prog.python"}) == []
    # the same sentence written again does not come back as a fresh lesson
    assert t.add_lesson("prefer donors with larger hidden size for code topics", {"prog.python"}, "r2") is None


def test_no_retirement_on_thin_control(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    lid = t.add_lesson("prefer donors with larger hidden size for code topics", {"prog.python"}, "r")
    _decide(t, [], True, "control")            # one control label: too few to compare against
    for _ in range(3):
        _decide(t, [lid], False, "lessons")
    assert t.lessons()[0]["retired"] == 0


def test_retirement_is_per_stage_and_brain(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    lid = t.add_lesson("prefer donors with larger hidden size for code topics", {"prog.python"}, "r")
    for _ in range(T.CONTROL_MIN):
        _decide(t, [], True, "control", stage="locate", brain="old")
    for _ in range(3):                         # bad showings for another brain: not judged against "old" control
        _decide(t, [lid], False, "lessons", stage="locate", brain="new")
    assert t.lessons()[0]["retired"] == 0


def test_recall_keeps_an_exploration_slot(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    texts = ["larger hidden size donors lead on code", "quantized donors still pass the gap screen",
             "the scientific subtopic has too few probes", "prefer dense checkpoints over sparse ones"]
    ids = [t.add_lesson(x, {"prog.python"}, "r") for x in texts]
    assert len(set(ids)) == 4
    for lid in ids[:3]:                        # three proven lessons
        for _ in range(2):
            did = t.record_decision("r", "locate", "b", "p", {"mode": "one", "ids": ["a"]}, [0.0], ["a"], [lid],
                                    arm="lessons")
            t.label(did, {"gold": ["a"], "known": ["a"]}, True)
    got = [x["id"] for x in t.recall({"prog.python"})]
    assert len(got) == T.RECALL_K and ids[3] in got      # the never-shown lesson still gets a slot


def test_arms_alternate_per_stage_and_brain(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    for _ in range(10):                        # decisions with no lesson available: outside the alternation
        t.record_decision("r", "locate", "b", "p", {}, [], [], [], arm="none")
    arms = []
    for _ in range(4):
        arm = t.next_arm("locate", "b")
        arms.append(arm)
        t.record_decision("r", "locate", "b", "p", {}, [], [], [], arm=arm)
    assert arms == ["control", "lessons", "control", "lessons"]
    assert t.next_arm("scope", "b") == "control" and t.next_arm("locate", "other") == "control"


def test_dedup_prefers_a_live_lesson(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    old = t.add_lesson("prefer donors with larger hidden size for code topics", {"x"}, "r")
    t.db.execute("UPDATE lessons SET retired=1 WHERE id=?", (old,))
    t.db.execute("INSERT INTO lessons(text, tags, source_run, created) VALUES(?,?,?,?)",
                 ("prefer donors with larger hidden size for code topics now", '["x"]', "r", 0))
    live = t.db.execute("SELECT id FROM lessons WHERE retired=0").fetchone()[0]
    assert t.add_lesson("prefer donors with larger hidden size for code topics", {"x"}, "r") == live


def test_self_candidates(tmp_path, monkeypatch):
    for d in ("brain", "same", "other", "moe"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "tokenizer.json").write_text("other" if d == "other" else "tok")
    mk = lambda n, kind="dense": {"checkpoint": f"{n}@0", "path": str(tmp_path / n), "kind": kind}
    kept, dropped = A.self_candidates(mk("brain"), [mk("brain"), mk("same"), mk("other"), mk("moe", "moe")])
    assert [c["checkpoint"] for c in kept] == ["same@0"]
    assert {d["checkpoint"]: d["reason"][:8] for d in dropped} == {"brain@0": "is the b", "other@0": "tokenize",
                                                                    "moe@0": "MoE dono"}


def test_self_mode_installs_but_never_adopts(tmp_path):
    calls = []
    res = iter([OK_GAP, PASS, {"status": "ok", "install": {"verdicts": {"side": "transfers"}}}])

    def pipeline(stage, pair):
        calls.append(stage)
        return next(res)

    agent = A.Agent(tmp_path / "s", 30, FakeBrain({"donor@1": 0, "prog.python": 0}), pipeline,
                    {"brain_checkpoint": "brain@9"})
    out = agent.run(A.SELF_GOAL, HOST, [D1], SHORT[:1], self_mode=True)
    assert calls == ["assess", "run", "install"]
    assert out["install"] == {"verdicts": {"side": "transfers"}} and out["adopted"] is False


def _eval(tmp_path, monkeypatch, capsys, prefer):
    monkeypatch.setattr(A, "inspect_checkpoint", lambda p: {"checkpoint": "cand@1", "fingerprint": "f" * 64})
    monkeypatch.setattr(A, "memory_snapshot", lambda: {"swap_gb": 0})
    monkeypatch.setattr(A, "require_gpu_window", lambda *a, **k: {})
    monkeypatch.setattr(cli, "check_commit", lambda *_: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(A, "Brain", lambda *a: FakeBrain(prefer))
    a = SimpleNamespace(tracker=str(tmp_path / "t.sqlite"), brain="/b", limit=0, gpu_window_owner="",
                        ignore_gpu_queue="")
    assert A.run_eval(a) == 0
    return json.loads(capsys.readouterr().out)


def test_eval_replays_labelled_decisions(tmp_path, monkeypatch, capsys):
    t = T.Tracker(tmp_path / "t.sqlite")
    run(tmp_path, [OK_GAP, NO_GAP, PASS], tracker=t, short=SHORT[:1], name="r1", cands=(D1, D2))
    t.close()
    same = _eval(tmp_path, monkeypatch, capsys, {"donor@1": 0, "donor@2": -3})
    assert same["label"] == A.LABEL and same["mlx2_commit"]["commit"] == "abc"
    loc = same["per_stage"]["locate"]
    assert loc["n"] == 1 and loc["candidate_correct"] == 1 and loc["recorded"]["brain@9"] == {"n": 1, "correct": 1}
    worse = _eval(tmp_path, monkeypatch, capsys, {"donor@2": 0, "donor@1": -3})
    assert worse["per_stage"]["locate"]["candidate_correct"] == 0           # a disagreeing replay is scored
    assert worse["per_stage"]["locate"]["candidate_known"] == 1


def test_eval_unobserved_choice_is_unknown(tmp_path, monkeypatch, capsys):
    t = T.Tracker(tmp_path / "t.sqlite")
    D3 = {"checkpoint": "donor@3", "fingerprint": "3" * 64, "path": "/d3", "kind": "dense"}
    # donor@1 leads and the run stops before donor@2 / donor@3 are assessed (step limit)
    brain = FakeBrain({"donor@1": 0, "donor@2": -1, "donor@3": -2, "prog.python": 0})
    run(tmp_path, [OK_GAP], brain=brain, tracker=t, short=SHORT[:1], name="r1", cands=(D1, D2, D3), limit=2)
    t.close()
    rep = _eval(tmp_path, monkeypatch, capsys, {"donor@3": 0})
    loc = rep["per_stage"]["locate"]
    assert loc["candidate_known"] == 0 and loc["coverage"] == 0 and loc["candidate_accuracy_on_known"] is None


def test_memory_snapshot_counts_reclaimable_pages(monkeypatch):
    mp = ("The system has 100 (10 pages with a page size of 1073741824).\nPages free: 1 \nPages purgeable: 2 \n"
          "Pages inactive: 20 \nPages speculative: 3 \n")
    sw = "vm.swapusage: total = 9216.00M  used = 512.00M  free = 1.00M  (encrypted)"
    monkeypatch.setattr(A.subprocess, "check_output", lambda cmd, **k: mp if cmd[0] == "memory_pressure" else sw)
    m = A.memory_snapshot()
    assert m["free_gb"] == 24 and m["pages_free_gb"] == 1 and m["swap_gb"] == 0.5     # purgeable not counted


def test_crashes_are_not_knowledge(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    agent, outcome, _ = run(tmp_path, [{"status": "crash"}, {"status": "crash"}], tracker=t, cands=(D1, D2),
                            short=SHORT[:1],
                            brain=FakeBrain({"donor@1": 0, "prog.python": 0, "skip": 0, "retry": -3, "stop": -4}))
    assert outcome["status"] == "stopped" and "no assessment reached a verdict" in outcome["reason"]
    assert not t.decisions(labelled_only=True)          # nothing to label locate against
    assert t.lessons() == []                             # no lesson from crashes


def test_admission_block_stops_and_keeps_the_attempt(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    dest = tmp_path / "pair-1" / "logs"
    dest.mkdir(parents=True)
    (dest / "gap.log").write_text("admit: waiting, other large processes [(1, 73.4, 'prod')]\n"
                                  "admit: another large process is running\n")
    assert "73.4" in A._admission_block(tmp_path / "pair-1")

    class P(A.Pipeline):
        def _run(self, stage, pair):
            return {"status": "memory_stop", "reason": "pipeline admission: x"}

    p = P(SimpleNamespace(out=str(tmp_path), host="/h"), HOST, [D1], 0, lambda *a, **k: None, t, "r", {})
    pair = {"candidate": "donor@1", "topic": "prog.python"}
    for _ in range(3):
        assert p("assess", pair)["status"] == "memory_stop"
    j = t.job(p.key("assess", pair))
    assert j["state"] == "queued" and j["attempts"] == 0


class StubGen:
    """A generator whose next-token distribution prefers one option text and, separately, the letter A."""

    def __init__(self, favourite, letter_bias=2.0):
        self.favourite, self.bias = favourite, letter_bias
        self.t = SimpleNamespace(apply_chat_template=lambda m, **k: "<u>" + m[0]["content"] + "<a>")
        self.rendered = []

    def encode(self, text):
        return [ord(text)] if len(text) == 1 else [0]

    def complete(self, prompt, n, temp, seed):
        return "because\nmore", 2


def test_brain_choose_cancels_letter_bias(monkeypatch):
    import math
    gen = StubGen("two")

    def lp(g, rendered):
        g.rendered.append(rendered)
        opts = rendered.split("\nOptions:\n")[1].split("\nAnswer with")[0].split("\n")
        logits = {}
        for line in opts:
            letter, text = line.split(". ", 1)
            logits[ord(letter)] = (3.0 if text == g.favourite else 0.0) + (g.bias if letter == "A" else 0.0)
        z = math.log(sum(math.exp(v) for v in logits.values()))
        return {k: SimpleNamespace(item=lambda v=v: v - z) for k, v in logits.items()}

    monkeypatch.setattr(A, "_first_token_logprobs", lp)
    prompt = A.menu_prompt("s", "q?", {}, [{"id": i, "text": t} for i, t in enumerate(["one", "two", "three"])], [])
    out = A.brain_choose(gen, prompt, 3, 4)
    assert out["passes"] == 3 and len(gen.rendered) == 3
    assert A.select(out["scores"], "rank") == [1, 0, 2] or A.select(out["scores"], "rank") == [1, 2, 0]
    assert abs(out["scores"][0] - out["scores"][2]) < 1e-9      # equal content, letter prior cancelled
    assert out["why"] == "because"


def test_letter_ids_refuse_split_labels():
    gen = SimpleNamespace(encode=lambda ch: [1, 2])
    with pytest.raises(SystemExit):
        A._letter_ids(gen, 2)


def _pipe(tmp_path, tracker=None):
    a = SimpleNamespace(out=str(tmp_path / "out"), host="/h", ignore_gpu_queue="", gpu_window_owner="",
                        allow_dirty=False)
    (tmp_path / "out").mkdir(exist_ok=True)
    return A.Pipeline(a, HOST, [D1], 0, lambda *x, **k: None, tracker, "run1", {"commit": "c"})


def _fake_child(monkeypatch, plan_rc=0, rc=0, err="", gap=None, receipts=None, log=None):
    def child(cmd, **k):
        if "plan" in cmd:
            return plan_rc, json.dumps({"fits": True}) if plan_rc == 0 else "no fit", ""
        dest = Path(str(cmd[cmd.index("--out") + 1]))
        dest.mkdir(parents=True, exist_ok=True)
        if gap is not None:
            (dest / "gap").mkdir(exist_ok=True)
            (dest / "gap/gap.json").write_text(json.dumps(gap))
        if receipts:
            (dest / "receipts").mkdir(exist_ok=True)
            (dest / "receipts/gap.json").write_text(json.dumps(receipts))
        if log:
            (dest / "logs").mkdir(exist_ok=True)
            (dest / "logs/gap.log").write_text(log)
        return rc, "", err
    monkeypatch.setattr(A, "child", child)
    monkeypatch.setattr(A, "admit_memory", lambda b: {"free_gb": 50, "swap_gb": 0})


from pathlib import Path  # noqa: E402
import re  # noqa: E402


@pytest.mark.parametrize("kw,status,plan_refused", [
    ({"plan_rc": 2}, "refused", True),
    ({"plan_rc": 1}, "crash", False),
    ({"gap": {"donor_leads": False}, "rc": 1}, "refused", None),
    ({"gap": {"donor_leads": True}}, "ok", None),
    ({"rc": 1, "receipts": {"error": "SystemExit: GPU stage outside a GPU window"}}, "crash", None),
    ({"rc": 1, "log": "admit: another large process is running\n"}, "memory_stop", None),
])
def test_pipeline_status_mapping(tmp_path, monkeypatch, kw, status, plan_refused):
    _fake_child(monkeypatch, **kw)
    r = _pipe(tmp_path)._run("assess", {"candidate": "donor@1", "topic": "prog.python"})
    assert r["status"] == status
    if plan_refused is not None:
        assert r["plan_refused"] is plan_refused


def test_guard_refusal_is_not_persisted_as_knowledge(tmp_path, monkeypatch):
    t = T.Tracker(tmp_path / "t.sqlite")
    _fake_child(monkeypatch, rc=1, receipts={"error": "SystemExit: GPU stage outside a GPU window"})
    p = _pipe(tmp_path, t)
    pair = {"candidate": "donor@1", "topic": "prog.python"}
    assert p("assess", pair)["status"] == "crash"
    assert t.job(p.key("assess", pair))["state"] == "failed" and p.known_lead("donor@1", "prog.python") is None


def test_plan_refusal_is_knowledge(tmp_path, monkeypatch):
    t = T.Tracker(tmp_path / "t.sqlite")
    _fake_child(monkeypatch, plan_rc=2)
    p = _pipe(tmp_path, t)
    p("assess", {"candidate": "donor@1", "topic": "prog.python"})
    assert p.known_lead("donor@1", "prog.python") is False


def test_attempts_count_per_run(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    k = T.job_key("gap", x=1)
    t.enqueue("assess", k, {}, "r1")
    for _ in range(T.MAX_ATTEMPTS):
        assert t.lease(k, "o", "r1")
        t.finish(k, "o", "failed", {})
    assert not t.lease(k, "o", "r1")
    assert t.lease(k, "o2", "r2")              # a later run gets its own attempts


def test_reused_extract_with_missing_dir_reruns(tmp_path, monkeypatch):
    t = T.Tracker(tmp_path / "t.sqlite")
    p = _pipe(tmp_path, t)
    pair = {"candidate": "donor@1", "topic": "prog.python"}
    k = p.key("run", pair)
    t.enqueue("run", k, pair, "old")
    t.lease(k, "o", "old")
    t.finish(k, "o", "done", {"status": "ok", "out": str(tmp_path / "gone")})
    ran = []
    monkeypatch.setattr(A.Pipeline, "_run", lambda self, s, pr: ran.append(s) or {"status": "ok", "out": "/y"})
    assert p("run", pair)["status"] == "ok" and ran == ["run"]


def test_keys_include_data_identity(tmp_path):
    p = _pipe(tmp_path)
    pair = {"candidate": "donor@1", "topic": "prog.python"}
    k1 = p.key("assess", pair)
    p._data_ident = {**p.data_ident(), "epoch": A.PIPELINE_EPOCH + 1}
    assert p.key("assess", pair) != k1


def test_brain_child_crash_is_retried(tmp_path):
    calls = []

    class Crashy(FakeBrain):
        def __call__(self, req):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("brain child rc=1")
            return super().__call__(req)

    agent, outcome, _ = run(tmp_path, [NO_GAP, NO_GAP], brain=Crashy({"donor@1": 0}), cands=(D1, D2),
                            short=SHORT[:1])
    assert outcome["status"] == "refused" and len(calls) == 2      # the locate decision: one crash, one retry
    fails = [r for r in agent.records if r["event"] == "brain_failure"]
    assert len(fails) == 1 and fails[0]["stage"] == "locate"


def test_interrupt_writes_report_and_reraises(tmp_path):
    def pipeline(stage, pair):
        raise KeyboardInterrupt

    agent = A.Agent(tmp_path / "i", 30, FakeBrain({"donor@1": 0}), pipeline, {"brain_checkpoint": "b"})
    with pytest.raises(KeyboardInterrupt):
        agent.run("Python", HOST, [D1, D2], SHORT[:1])
    rep = json.loads((agent.out / "auto_report.json").read_text())
    assert rep["outcome"]["status"] == "interrupted" and rep["lesson"] is None



def test_partly_assessed_choice_is_not_labelled_wrong(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    agent = A.Agent(tmp_path / "p", 30, None, None, {"brain_checkpoint": "b"}, tracker=t)
    did = t.record_decision("r", "locate", "b", "p", {"mode": "rank", "ids": ["donor@1", "donor@2"]}, [0, -1],
                            ["donor@1", "donor@2"], [], arm="control")
    agent.menus = {"locate": ["donor@1", "donor@2"], "scope": ["prog.python", "prog.rust"]}
    agent.decided = {"locate": [(did, ["donor@1", "donor@2"])], "scope": [(None, ["prog.python", "prog.rust"])]}
    agent.assessed = {"p0": {"pair": {"candidate": "donor@1", "topic": "prog.python"}, "leads": False},
                      "p1": {"pair": {"candidate": "donor@2", "topic": "prog.python"}, "leads": True}}
    agent.pipeline = SimpleNamespace(known_lead=lambda *a: None, known_verdict=lambda *a: None)
    assert agent.label_decisions() == []      # donor@1 on prog.rust is unknown: donor@1 may still be acceptable


def test_empty_recall_is_outside_the_comparison(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    agent, _, _ = run(tmp_path, [NO_GAP, NO_GAP], tracker=t, cands=(D1, D2), short=SHORT[:1])
    loc = [d for d in t.decisions() if d["stage"] == "locate"][-1]
    assert loc["arm"] == "none" and loc["lessons"] == []


def test_control_arm_withholds_available_lessons(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")
    lid = t.add_lesson("prefer donors with larger hidden size for code topics", {"prog.python"}, "r")
    run(tmp_path, [NO_GAP, NO_GAP], tracker=t, cands=(D1, D2), short=SHORT[:1], name="a")
    run(tmp_path, [NO_GAP, NO_GAP], tracker=t, cands=(D1, D2), short=SHORT[:1], name="b")
    locs = [d for d in t.decisions() if d["stage"] == "locate"]
    assert locs[0]["arm"] == "control" and locs[0]["lessons"] == []
    assert locs[1]["arm"] == "lessons" and lid in locs[1]["lessons"]
    assert "larger hidden size" in locs[1]["prompt"] and "larger hidden size" not in locs[0]["prompt"]


@pytest.mark.parametrize("err,status", [
    ("SystemExit: GPU stage outside a GPU window", "crash"),
    ("donor 9 GB + host 3 GB + work 2 GB = 14 GB > cap", "crash"),
    ("the donor is MoE: install reads dense MLP units only", "refused"),
    ("donor and host tokenizers are not byte-identical: every install route pairs tokens by id", "refused"),
    ("Traceback (most recent call last):\nKeyError: 'layer'", "crash"),
])
def test_install_refusal_classes(tmp_path, monkeypatch, err, status):
    _fake_child(monkeypatch, rc=1, err=err)
    p = _pipe(tmp_path)
    p.dirs[("donor@1", "prog.python")] = tmp_path / "x"
    assert p._run("install", {"candidate": "donor@1", "topic": "prog.python"})["status"] == status


def test_install_key_follows_the_box(tmp_path):
    p = _pipe(tmp_path)
    pair = {"candidate": "donor@1", "topic": "prog.python"}
    box = tmp_path / "x" / "box_bundle"
    box.mkdir(parents=True)
    (box / "manifest.json").write_text("one")
    p.dirs[("donor@1", "prog.python")] = tmp_path / "x"
    k1 = p.key("install", pair)
    (box / "manifest.json").write_text("two")
    assert p.key("install", pair) != k1


def test_interrupt_in_tracked_pipeline_fails_the_job(tmp_path):
    t = T.Tracker(tmp_path / "t.sqlite")

    class P(A.Pipeline):
        def _run(self, stage, pair):
            raise KeyboardInterrupt

    p = P(SimpleNamespace(out=str(tmp_path), host="/h"), HOST, [D1], 0, lambda *a, **k: None, t, "r", {})
    pair = {"candidate": "donor@1", "topic": "prog.python"}
    with pytest.raises(KeyboardInterrupt):
        p("assess", pair)
    assert t.job(p.key("assess", pair))["state"] == "failed"


def test_heartbeat_before_each_child(tmp_path, monkeypatch):
    clock = [0.0]
    t = T.Tracker(tmp_path / "t.sqlite", now=lambda: clock[0])
    beats = []
    monkeypatch.setattr(T.Tracker, "heartbeat", lambda self, k, o: beats.append(k))

    def child(cmd, **k):
        clock[0] += 7000
        return (0, json.dumps({}), "") if "plan" in cmd else (0, "", "")
    monkeypatch.setattr(A, "child", child)
    monkeypatch.setattr(A, "admit_memory", lambda b: {"free_gb": 50, "swap_gb": 0})
    p = _pipe(tmp_path, t)
    p("assess", {"candidate": "donor@1", "topic": "prog.python"})
    assert len(beats) == 2                     # plan and extract each renewed the lease first


def test_renewed_lease_survives_expiry(tmp_path):
    clock = [0.0]
    t = T.Tracker(tmp_path / "t.sqlite", now=lambda: clock[0])
    k = T.job_key("gap", x=2)
    t.enqueue("assess", k, {}, "r1")
    assert t.lease(k, "o1", "r1")
    for _ in range(3):                          # three children of 2 h each, the lease renewed before each
        t.heartbeat(k, "o1")
        clock[0] += 7200
    assert not t.lease(k, "o2", "r2")          # 6 h in, still o1's: the last renewal was 2 h ago
    clock[0] += T.LEASE_S
    assert t.lease(k, "o2", "r2")              # a dead run's lease does expire


def test_lost_lease_keeps_the_result(tmp_path, monkeypatch):
    t = T.Tracker(tmp_path / "t.sqlite")

    class P(A.Pipeline):
        def _run(self, stage, pair):
            return {**OK_GAP}

    p = P(SimpleNamespace(out=str(tmp_path), host="/h"), HOST, [D1], 0, lambda *a, **k: None, t, "r", {})
    monkeypatch.setattr(T.Tracker, "finish", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("not leased")))
    assert p("assess", {"candidate": "donor@1", "topic": "prog.python"})["gap"]["donor_leads"] is True


def test_lost_lease_on_failure_keeps_the_original_error(tmp_path, monkeypatch):
    t = T.Tracker(tmp_path / "t.sqlite")

    class P(A.Pipeline):
        def _run(self, stage, pair):
            raise ValueError("the real failure")

    p = P(SimpleNamespace(out=str(tmp_path), host="/h"), HOST, [D1], 0, lambda *a, **k: None, t, "r", {})
    monkeypatch.setattr(T.Tracker, "finish", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("not leased")))
    with pytest.raises(ValueError, match="the real failure"):
        p("assess", {"candidate": "donor@1", "topic": "prog.python"})


def test_policy_refusals_are_install_messages():
    src = (Path(A.__file__).parent / "install.py").read_text()
    exits = " ".join(re.findall(r"raise SystemExit\((.*?)\)\n", src, re.S))
    for m in A.INSTALL_POLICY_REFUSALS:
        assert m in exits, m


def test_deadline_stops_actions_and_children(tmp_path, monkeypatch):
    monkeypatch.setitem(A.DEADLINE, "t", A.time.time() - 1)
    with pytest.raises(A.Stop, match="deadline"):
        A.child(["true"])
    agent, outcome, calls = run(tmp_path, [OK_GAP], cands=(D1, D2), short=SHORT[:1])
    assert "deadline" in outcome["reason"] and not calls
    monkeypatch.setitem(A.DEADLINE, "t", A.time.time() + 1)
    t0 = A.time.time()
    monkeypatch.setattr(A, "DEADLINE_GRACE_S", 0)
    with pytest.raises(Exception):
        A.child(["sleep", "30"])               # cut at the deadline, its process group reaped
    assert A.time.time() - t0 < 20


def test_minutes_flag():
    assert cli.parse(["auto", "--goal", "g", "--host", "h", "--out", "o", "--minutes", "30"]).minutes == 30



def test_memory_snapshot_failure_is_a_memory_stop(monkeypatch):
    def boom(cmd, **k):
        raise A.subprocess.TimeoutExpired(cmd, 15)
    monkeypatch.setattr(A.subprocess, "check_output", boom)
    with pytest.raises(A.Stop, match="memory admission refused"):
        A.memory_snapshot()


def test_pipeline_children_get_a_grace_longer_than_the_inner_stop(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(A, "child", lambda cmd, **k: seen.setdefault("plan" if "plan" in cmd else "x", k)
                        and ((0, json.dumps({}), "") if "plan" in cmd else (0, "", "")))
    monkeypatch.setattr(A, "admit_memory", lambda b: {"free_gb": 50, "swap_gb": 0})
    p = _pipe(tmp_path)
    p._run("assess", {"candidate": "donor@1", "topic": "prog.python"})
    assert seen["x"]["grace_s"] == A.PIPELINE_GRACE_S > 120


def test_brain_checks_the_gpu_window_every_call(monkeypatch):
    calls = []
    monkeypatch.setattr(A, "require_gpu_window", lambda *a, **k: calls.append(k) or {})
    monkeypatch.setattr(A, "admit_memory", lambda b: {"free_gb": 50, "swap_gb": 0})
    monkeypatch.setattr(A, "child", lambda *a, **k: (0, json.dumps({"scores": [0, 0]}), ""))
    b = A.Brain("/m", 0, lambda *x, **k: None, "owner", "")
    b({"mode": "choose"})
    b({"mode": "choose"})
    assert len(calls) == 2


def test_extract_core_waits_for_the_gap_verdict():
    from mlx2.experimental.amoeba.state import EXTRACT_DEPENDS
    assert "gap" in EXTRACT_DEPENDS["core"]


def test_install_with_failed_route_can_be_retried(tmp_path, monkeypatch):
    tracker = T.Tracker(tmp_path / "t.sqlite")
    p = _pipe(tmp_path, tracker)
    pair = {"candidate": "donor@1", "topic": "prog.python"}
    p.dirs[tuple(pair.values())] = tmp_path / "extracted"

    def failed_route(cmd, **kwargs):
        dest = Path(cmd[cmd.index("--out") + 1])
        (dest / "report").mkdir(parents=True)
        (dest / "report/manifest.json").write_text(json.dumps({
            "verdicts": [{"route": "side", "verdict": "failed"}]}))
        return 0, "", ""

    monkeypatch.setattr(A, "child", failed_route)
    monkeypatch.setattr(A, "admit_memory", lambda _: {"free_gb": 50, "swap_gb": 0})
    assert p("install", pair)["status"] == "crash"
    assert tracker.job(p.key("install", pair))["state"] == "failed"
    assert not tracker.facts(kind="install_verdict")


def test_tracker_only_reuses_an_extraction_with_its_manifest(tmp_path, monkeypatch):
    tracker = T.Tracker(tmp_path / "t.sqlite")
    p = _pipe(tmp_path, tracker)
    pair = {"candidate": "donor@1", "topic": "prog.python"}
    k = p.key("run", pair)
    tracker.enqueue("run", k, pair, "old")
    tracker.lease(k, "old", "old")
    tracker.finish(k, "old", "done", {"status": "ok", "bundle": PASS["bundle"], "out": str(tmp_path)})
    monkeypatch.setattr(p, "_run", lambda *args: {**PASS, "out": str(tmp_path)})
    assert p("run", pair)["status"] == "ok"


def test_reusing_install_preserves_the_extracted_box(tmp_path):
    tracker = T.Tracker(tmp_path / "t.sqlite")
    p = _pipe(tmp_path, tracker)
    pair = {"candidate": "donor@1", "topic": "prog.python"}
    source = tmp_path / "donor"
    p.dirs[("donor@1", "prog.python")] = source
    k = p.key("install", pair)
    dest = tmp_path / "install"
    (dest / "report").mkdir(parents=True)
    (dest / "report/manifest.json").write_text(json.dumps({"verdicts": [{"route": "side", "verdict": "no gain"}]}))
    tracker.enqueue("install", k, pair, "old")
    tracker.lease(k, "old", "old")
    tracker.finish(k, "old", "done", {"out": str(dest), "install": {"verdicts": {"side": "no gain"}}})
    assert p("install", pair)["status"] == "reused"
    assert p.dirs[("donor@1", "prog.python")] == source
