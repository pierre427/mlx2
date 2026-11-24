"""H45 loop port: forward semantics and resumable training (CPU)."""
import json

import mlx.core as mx
from mlx.utils import tree_flatten

from research.ens import common as C
from research.ens import families as F
from research.ens import loop_train as L
from research.ens.model import BusCall, Core, CoreConfig


def _core():
    mx.random.seed(0)
    return Core(CoreConfig(d=32, layers=6, heads=2, ffn=64, bus_width=8, loop_body=(2, 4)))


def _rows(n=6):
    fam = F.BY_ID["graph.6-2"]
    import random
    rng = random.Random(0)
    return [F.sample_row(fam, rng) for _ in range(n)]


def test_one_pass_equals_the_plain_core():
    core = _core()
    tokens, q, f, a, ans, zs = C.batch(_rows(), "bus")
    msg = mx.random.normal((tokens.shape[0], 8))
    want = core(tokens, BusCall("early", q, f, msg))
    got = core(tokens, BusCall("early", q, f, msg), passes=mx.ones((tokens.shape[0],), dtype=mx.int32))
    assert float(mx.max(mx.abs(want - got)).item()) == 0.0
    assert float(mx.max(mx.abs(core(tokens) - core(tokens, passes=mx.ones((tokens.shape[0],), dtype=mx.int32)))).item()) == 0.0


def test_per_row_passes_match_uniform_runs():
    core = _core()
    tokens, q, f, a, ans, zs = C.batch(_rows(), "bus")
    msg = mx.random.normal((tokens.shape[0], 8))
    mixed = mx.array([1, 2, 3, 1, 3, 2])
    got = core(tokens, BusCall("early", q, f, msg), passes=mixed)
    for k in (1, 2, 3):
        uni = core(tokens, BusCall("early", q, f, msg), passes=mx.full((6,), k))
        rows = [i for i in range(6) if int(mixed[i].item()) == k]
        d = mx.max(mx.abs(got[mx.array(rows)] - uni[mx.array(rows)]))
        assert float(d.item()) < 1e-5
    # max_passes (compiled callers) gives the same result
    same = core(tokens, BusCall("early", q, f, msg), passes=mixed, max_passes=3)
    assert float(mx.max(mx.abs(same - got)).item()) == 0.0
    # more passes change the output (the loop does something)
    two = core(tokens, BusCall("early", q, f, msg), passes=mx.full((6,), 2))
    one = core(tokens, BusCall("early", q, f, msg), passes=mx.full((6,), 1))
    assert float(mx.max(mx.abs(two - one)).item()) > 0


def test_answer_at_and_bus_before_loop():
    import random
    import pytest
    for fam in L.depth_families():
        r = F.sample_row(fam, random.Random(1))
        assert L.answer_at(fam, r, L.steps_of(fam)) == r.answer
    core = _core()
    tokens, q, f, a, ans, zs = C.batch(_rows(), "bus")
    with pytest.raises(ValueError):           # middle site (after block 3) is inside the loop body
        core(tokens, BusCall("middle", q, f, mx.zeros((6, 8))), passes=mx.full((6,), 2))
    plain = Core(CoreConfig(d=32, layers=6, heads=2, ffn=64, bus_width=8))
    with pytest.raises(ValueError):
        plain(tokens, passes=mx.full((6,), 2))


def test_steps_and_families():
    assert L.steps_of(F.BY_ID["graph.6-2"]) == 2
    assert L.steps_of(F.BY_ID["fst.2-2"]) == 2
    assert L.steps_of(F.BY_ID["program.ADD.MUL"]) == 2
    assert L.steps_of(F.BY_ID["perm.4"]) == 1
    assert max(L.steps_of(f) for f in L.train_mix()) == L.TRAIN_MAX_PASSES
    assert {L.steps_of(f) for f in L.depth_families()} == set(range(1, 7))
    for fam in L.depth_families() + list(L.EXTRA):
        import random
        F.layout(F.sample_row(fam, random.Random(0)), "bus")      # fits SEQ_LEN


def test_resume_is_identical_to_one_run(tmp_path):
    base = ["--arm", "loop", "--seed", "0", "--steps", "6", "--batch", "8", "--log-every", "3"]
    try:
        L.main(base + ["--out", str(tmp_path / "a")])
    except SystemExit as e:
        assert e.code == 0
    # stop after 2 updates (mid-warmup: Adam moments, step counter and both RNG streams in play), then after 2 more,
    # then run to the end: three launches must equal one
    for extra, code in ((["--stop-after", "2"], 3), (["--stop-after", "2"], 3), ([], 0)):
        try:
            L.main(base + ["--out", str(tmp_path / "b")] + extra)
        except SystemExit as e:
            assert e.code == code
    wa = dict(tree_flatten(mx.load(str(tmp_path / "a" / "core.safetensors"))))
    wb = dict(tree_flatten(mx.load(str(tmp_path / "b" / "core.safetensors"))))
    assert wa.keys() == wb.keys()
    for k in wa:
        assert float(mx.max(mx.abs(wa[k] - wb[k])).item()) == 0.0, k
    ra = json.loads((tmp_path / "a" / "receipt.json").read_text())
    rb = json.loads((tmp_path / "b" / "receipt.json").read_text())
    assert ra["checkpoint"] == rb["checkpoint"]
    _same_logs(ra, rb)


def test_sigterm_saves_and_resumes_identically(tmp_path, monkeypatch):
    import os
    import signal
    base = ["--arm", "loop", "--seed", "0", "--steps", "6", "--batch", "8", "--log-every", "3"]
    try:
        L.main(base + ["--out", str(tmp_path / "a")])
    except SystemExit as e:
        assert e.code == 0
    real, calls = L.build_batch, {"n": 0}

    def term_on_third(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            os.kill(os.getpid(), signal.SIGTERM)       # as the GPU-lock wrapper does
        return real(*a, **k)
    monkeypatch.setattr(L, "build_batch", term_on_third)
    try:
        L.main(base + ["--out", str(tmp_path / "b")])
    except SystemExit as e:
        assert e.code == 3
    assert (tmp_path / "b" / "resume" / "stream.pkl").exists()
    monkeypatch.setattr(L, "build_batch", real)
    try:
        L.main(base + ["--out", str(tmp_path / "b")])
    except SystemExit as e:
        assert e.code == 0
    wa = mx.load(str(tmp_path / "a" / "core.safetensors"))
    wb = mx.load(str(tmp_path / "b" / "core.safetensors"))
    assert all(float(mx.max(mx.abs(wa[k] - wb[k])).item()) == 0.0 for k in wa)
    assert signal.getsignal(signal.SIGTERM) is not L._on_term      # handler restored
    _same_logs(json.loads((tmp_path / "a" / "receipt.json").read_text()),
               json.loads((tmp_path / "b" / "receipt.json").read_text()))


def test_args_only_dir_is_reusable(tmp_path):
    base = ["--arm", "fixed", "--seed", "0", "--steps", "2", "--batch", "8", "--log-every", "1"]
    d = tmp_path / "c"
    d.mkdir()
    (d / "args.json").write_text(json.dumps(vars(L.build_parser().parse_args(base + ["--out", str(d)]))))
    try:
        L.main(base + ["--out", str(d)])
    except SystemExit as e:
        assert e.code == 0
    assert (d / "receipt.json").exists()


def test_autosave_survives_a_hard_kill(tmp_path, monkeypatch):
    """A crash after an autosave resumes from it; the finished run equals one uninterrupted run."""
    base = ["--arm", "loop", "--seed", "0", "--steps", "6", "--batch", "8", "--log-every", "3", "--save-every", "2"]
    try:
        L.main(base + ["--out", str(tmp_path / "a")])
    except SystemExit as e:
        assert e.code == 0
    real, calls = L.build_batch, {"n": 0}

    class Crash(Exception):
        pass

    def crash_on_fifth(*a, **k):
        calls["n"] += 1
        if calls["n"] == 5:
            raise Crash()                               # a hard kill: no handler, no clean save
        return real(*a, **k)
    monkeypatch.setattr(L, "build_batch", crash_on_fifth)
    import pytest
    with pytest.raises(Crash):
        L.main(base + ["--out", str(tmp_path / "b")])
    import pickle
    assert pickle.loads((tmp_path / "b" / "resume" / "stream.pkl").read_bytes())["it"] == 4
    monkeypatch.setattr(L, "build_batch", real)
    try:
        L.main(base + ["--out", str(tmp_path / "b")])
    except SystemExit as e:
        assert e.code == 0
    wa = mx.load(str(tmp_path / "a" / "core.safetensors"))
    wb = mx.load(str(tmp_path / "b" / "core.safetensors"))
    assert all(float(mx.max(mx.abs(wa[k] - wb[k])).item()) == 0.0 for k in wa)
    _same_logs(json.loads((tmp_path / "a" / "receipt.json").read_text()),
               json.loads((tmp_path / "b" / "receipt.json").read_text()))
    # a stale resume.old/ left by a kill during cleanup must not block the next save (review)
    (tmp_path / "b" / "resume.old" / "junk").mkdir(parents=True)
    L.save_resume(tmp_path / "b", L.Core(L.CoreConfig(d=32, layers=6, heads=2, ffn=64, bus_width=8, loop_body=(2, 4))),
                  type("O", (), {"state": {"step": mx.array(1)}})(), 1, [__import__("random").Random(0)], [], 0.0)
    assert (tmp_path / "b" / "resume" / "stream.pkl").exists() and not (tmp_path / "b" / "resume.old").exists()



def _same_logs(ra, rb):
    """Logs equal up to wall-clock fields."""
    strip = lambda log: [{k: v for k, v in r.items() if k != "elapsed_s"} for r in log]
    assert strip(ra["log"]) == strip(rb["log"])
