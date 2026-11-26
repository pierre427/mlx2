"""Wording stress test for the H43 record ports (CPU only; no model)."""
from __future__ import annotations

import random

from research.ens import physics_bus as PB
from research.ens import physics_qa as QA
from research.ens import wording_eval as W


def test_wordings_are_new_balanced_and_well_formed():
    W._check_wordings()
    assert set(W.WORDINGS) == set(PB.RETRIEVAL + PB.REASONING)


def test_every_wording_builds_a_question():
    paper = lambda t, f, m, c: {"title": t, "field": f, "month": m, "concept": c}
    fields = list(QA.FIELD_NAME)
    A = paper("Alpha Beta Gamma", fields[0], QA.PF.MONTHS[0], "entanglement")
    B = paper("Delta Epsilon", fields[1], QA.PF.MONTHS[3], "renormalization")
    for task, ws in W.WORDINGS.items():
        for q, _ in ws:
            qa = QA.make_qa(task, [A, B], random.Random(0), {task: [q]})
            assert qa is not None, (task, q)
            prompt, answer, titles, meta = qa
            assert all(t in prompt for t in titles)
            assert prompt.index(A["title"]) < (prompt.index(B["title"]) if len(titles) > 1 else 10 ** 9)


def test_summary_pools_by_task_and_order():
    cells = {"w00.title_first.field": {"record.correct": 1.0, "swapped_record.correct": 0.1, "base.correct": 0.0},
             "w01.title_first.field": {"record.correct": 0.8, "swapped_record.correct": 0.1, "base.correct": 0.0},
             "w05.key_first.field": {"record.correct": 0.4, "swapped_record.correct": 0.2, "base.correct": 0.0}}
    s = W.summarize(cells)["field"]
    assert s["title_first"]["record"] == 0.9 and s["title_first"]["wordings"] == 2
    assert s["key_first"]["record"] == 0.4 and s["key_minus_title"] == -0.5


def test_default_eval_plan_is_unchanged():
    # the trained runs' cells are keyed "<set>.<task>" and seeded by hash_task(set, task); the wording option must
    # not change that default (resumed evals and earlier receipts stay comparable)
    import inspect
    src = inspect.getsource(PB.evaluate)
    assert '("train_questions", T_TRAIN), ("held_questions", T_HELD)' in src
    assert "hash_task(seed_name, task)" in src and "(name, temps, task, name)" in src


def test_decision_tasks_ask_which_one():
    # earlier / which_concept answers are parsed as "the first / second one" (physics_qa._decision); a yes/no
    # question would score every arm wrong (review)
    for task in ("earlier", "which_concept"):
        for q, _ in W.WORDINGS[task]:
            low = q.lower()
            assert not low.startswith(("was ", "is ", "did ", "does ")) and (" or " in low or "which" in low), q
