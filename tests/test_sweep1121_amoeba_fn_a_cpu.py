"""Bug sweep 2026-11-21, lane amoeba-fn-a (AMFN-1, 2, 3, 4, 5, 6 stats site, 8, 9). CPU only: no model, no GPU.

The shell tests run the real scripts from a copy of scripts/ with stub `python`, `git`, `curl` and `sleep` commands
(HOME points at a temp dir, so `~/Desktop/mlx2/.venv/bin/python` is the stub): only the scripts' own control flow
and exit codes are exercised."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
import test_amoeba_fn_langgate_cpu as L  # noqa: E402
import test_amoeba_p0_drivers_cpu as T  # noqa: E402
from test_amoeba_fn_langgate_cpu import toy_ops  # noqa: E402,F401  (fixture)
from test_amoeba_p0_drivers_cpu import toy  # noqa: E402,F401  (fixture)

ZSH = shutil.which("zsh")
shell = pytest.mark.skipif(ZSH is None, reason="zsh not installed")

STUB_PY = r"""#!/bin/sh
# stub python: STUB_RULES = "substring=rc;substring=rc" (first match on the joined argv wins, default 0);
# STUB_PRINT_PAT / STUB_PRINT: print STUB_PRINT when the argv contains STUB_PRINT_PAT.
args="$*"
echo "$args" >> "$STUB_LOG"
if [ -n "${STUB_PRINT_PAT:-}" ]; then case "$args" in *"$STUB_PRINT_PAT"*) echo "$STUB_PRINT";; esac; fi
old=$IFS; IFS=';'
for r in ${STUB_RULES:-}; do
  pat=${r%=*}; rc=${r##*=}
  case "$args" in *"$pat"*) exit "$rc";; esac
done
IFS=$old
exit 0
"""
STUB_GIT = r"""#!/bin/sh
case "$1 $2" in
  "rev-parse --short=9") echo abc123456;;
  "worktree add") mkdir -p "$5" && cp -R "$STUB_SNAP_SRC/." "$5/";;
  "worktree remove") rm -rf "$4";;
  "show "*) cat "$STUB_PLAN" 2>/dev/null;;
esac
exit 0
"""


@pytest.fixture
def sh(tmp_path):
    """A copy of scripts/*.sh in a fake checkout, a fake HOME and stub commands first on PATH."""
    repo, home, bin_ = tmp_path / "repo", tmp_path / "home", tmp_path / "bin"
    (repo / "scripts").mkdir(parents=True)
    for f in (ROOT / "scripts").glob("*.sh"):
        shutil.copy(f, repo / "scripts" / f.name)
    venv = home / "Desktop/mlx2/.venv/bin"
    venv.mkdir(parents=True)
    bin_.mkdir()
    for p, body in ((venv / "python", STUB_PY), (bin_ / "git", STUB_GIT),
                    (bin_ / "curl", "#!/bin/sh\necho '\"state\": \"ready\"'\nexit 0\n"),
                    (bin_ / "sleep", "#!/bin/sh\nexit 0\n")):
        p.write_text(body)
        p.chmod(0o755)
    na = tmp_path / "nighta"
    (na / "t2").mkdir(parents=True)
    (na / "t2/export_core.npz").write_text("x")
    (na / "t1").mkdir()
    env = {"HOME": str(home), "PATH": f"{bin_}:/usr/bin:/bin:/usr/sbin:/sbin", "STUB_LOG": str(tmp_path / "stub.log"),
           "NIGHTA_ROOT": str(na), "OUT_ROOT": str(tmp_path / "out")}
    (tmp_path / "out").mkdir()

    def run(script, *args, rules="", **extra):
        e = {**env, "STUB_RULES": rules, **{k: str(v) for k, v in extra.items()}}
        r = subprocess.run([ZSH, str(repo / "scripts" / script), *map(str, args)], env=e, capture_output=True,
                           text=True, timeout=120)
        return r
    return SimpleNS(tmp=tmp_path, repo=repo, home=home, out=tmp_path / "out", run=run, env=env)


class SimpleNS(types.SimpleNamespace):
    pass


def _touch(p, text="{}\n"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


CARVE = "research.amoeba.carve"
NOT_DONE = "stages_done=1"                      # the "is this box already bisected" check says no


# ----------------------------------------------------------------------------- AMFN-2: inner scripts

@shell
def test_amfn2_det_and_calc_window_inner_report_a_failed_phase(sh):
    runs = sh.tmp / "runs"
    assert sh.run("amoeba27b_det_window_inner.sh", runs).returncode == 0
    r = sh.run("amoeba27b_det_window_inner.sh", runs, rules="amoeba_det_facts_cpu.py=1")
    assert r.returncode == 1 and "phase served" in r.stdout              # the next phase still ran
    r = sh.run("amoeba27b_det_window_inner.sh", runs, rules="amoeba_det_calc_cpu.py=75")
    assert r.returncode == 75 and "phase facts" not in r.stdout          # a yield stops the window, unchanged
    assert sh.run("amoeba_fn_calc_window_inner.sh", runs, PHASES="fit").returncode == 0
    assert sh.run("amoeba_fn_calc_window_inner.sh", runs, PHASES="fit", rules="amoeba_det_calc_cpu.py=1").returncode == 1
    r = sh.run("amoeba_fn_calc_window_inner.sh", runs, PHASES="ab")      # nothing to run it on: not a success
    assert r.returncode == 1 and "ab skipped" in r.stdout


@shell
def test_amfn2_t12eval_and_calib_report_failed_evals_and_carves(sh):
    for f in ("build/manifest.json", "gen-held/gen.jsonl"):
        _touch(sh.out / f)
    assert sh.run("amoeba_fn_t12eval_inner.sh").returncode == 0
    r = sh.run("amoeba_fn_t12eval_inner.sh", rules="amoeba_fn_eval_masks.py=1")
    assert r.returncode == 1 and r.stdout.count("eval t") == 3           # every eval was still attempted
    p1 = sh.tmp / "part1"
    p1.mkdir()
    r = sh.run("amoeba_fn_calib_inner.sh", PART1_ROOT=p1, CONFIGS="A B")
    assert r.returncode == 0, r.stdout + r.stderr
    r = sh.run("amoeba_fn_calib_inner.sh", PART1_ROOT=p1, CONFIGS="B", rules=f"{NOT_DONE};{CARVE}=1")
    assert r.returncode == 1 and r.stdout.count("rc=1") == 3 and "calibration done" in r.stdout
    r = sh.run("amoeba_fn_calib_inner.sh", PART1_ROOT=p1, CONFIGS="A", rules="amoeba_fn_eval_masks.py=1")
    assert r.returncode == 1
    r = sh.run("amoeba_fn_calib_inner.sh", PART1_ROOT=p1, CONFIGS="B", rules=f"{NOT_DONE};{CARVE}=75")
    assert r.returncode == 75 and r.stdout.count("rc=75") == 1


@shell
def test_amfn2_part2_code_nightb_report_failed_carves_and_partitions(sh):
    _touch(sh.out / "build/boxes.json")
    _touch(sh.out / "onpolicy-gen/gen.jsonl")
    p1 = sh.tmp / "part1"
    (p1 / "box-old").mkdir(parents=True)
    for b in ("law", "med"):                                             # the carve driver creates its --out first
        (sh.out / f"box-{b}").mkdir()
    kw = dict(PART1_ROOT=p1, STUB_PRINT_PAT="x['box'] for x", STUB_PRINT="law med")
    r = sh.run("amoeba_fn_part2_inner.sh", **kw)
    assert r.returncode == 0, r.stdout + r.stderr
    r = sh.run("amoeba_fn_part2_inner.sh", rules=f"{NOT_DONE};{CARVE}=1", **kw)
    assert r.returncode == 1 and "box med rc=1" in r.stdout and "part 2 done" in r.stdout
    r = sh.run("amoeba_fn_part2_inner.sh", rules="amoeba_fn_box_partition.py=1", **kw)
    assert r.returncode == 1
    Path(sh.env["STUB_LOG"]).write_text("")
    r = sh.run("amoeba_fn_part2_inner.sh", rules=f"{NOT_DONE};{CARVE}=75", **kw)
    assert r.returncode == 75 and "--refused" not in Path(sh.env["STUB_LOG"]).read_text()     # a yield, not a refusal
    # code hierarchy: code_common carves, a language box does not
    kw = dict(STUB_PRINT_PAT="!= 'code_common'", STUB_PRINT="rust go")
    assert sh.run("amoeba_fn_code_inner.sh", **kw).returncode == 0
    r = sh.run("amoeba_fn_code_inner.sh", rules=f"{NOT_DONE};box-rust --model=1", **kw)
    assert r.returncode == 1 and "box go rc=0" in r.stdout
    assert sh.run("amoeba_fn_code_inner.sh", rules="amoeba_fn_box_partition.py=1", **kw).returncode == 1
    # night B
    for b in ("law", "med"):
        _touch(sh.out / f"mixed/{b}.jsonl")
    assert sh.run("amoeba_fn_nightB_inner.sh", BOXES="law med").returncode == 0
    r = sh.run("amoeba_fn_nightB_inner.sh", BOXES="law med", rules=f"{NOT_DONE};{CARVE}=1")
    assert r.returncode == 1 and "box med rc=1" in r.stdout
    assert sh.run("amoeba_fn_nightB_inner.sh", BOXES="law med", rules=f"{NOT_DONE};{CARVE}=75").returncode == 75


@shell
def test_amfn2_nighta_reports_a_failed_atlas_and_mix(sh):
    _touch(sh.out / "onpolicy-gen/gen.jsonl")
    far = int(time.time()) + 3600
    assert sh.run("amoeba_fn_nightA_inner.sh", STOP_AT=far).returncode == 0
    r = sh.run("amoeba_fn_nightA_inner.sh", STOP_AT=far, rules="atlas-run=1")
    assert r.returncode == 1 and "T2 rc=0" in r.stdout                   # T1/T2 still ran
    assert sh.run("amoeba_fn_nightA_inner.sh", STOP_AT=far, rules="atlas-run=75").returncode == 75
    assert sh.run("amoeba_fn_nightA_inner.sh", STOP_AT=far, rules="/out/t2 =3").returncode == 3
    r = sh.run("amoeba_fn_nightA_inner.sh", STOP_AT=far, rules="/mixed =1")      # the mixing heredoc
    assert r.returncode == 5 and "atlas rc" not in r.stdout


@shell
def test_amfn2_calc_trigger_and_boxes_report_the_second_run(sh):
    runs = sh.tmp / "runs"
    kw = dict(THINK_BEFORE="23:59", STUB_PRINT_PAT="rcpt-test1", STUB_PRINT="v4")     # the best-cue-arm pick
    a = (runs, 2, sh.tmp / "lock", sh.tmp / "rcpt-test1.json")
    r = sh.run("calc_trigger_gsm8k_inner.sh", *a, **kw)
    assert r.returncode == 0, r.stdout + r.stderr
    r = sh.run("calc_trigger_gsm8k_inner.sh", *a, rules="test2-think.arms.json=1", **kw)
    assert r.returncode == 1 and "gsm8k" in r.stdout
    r = sh.run("calc_trigger_gsm8k_inner.sh", *a, rules="test2.arms.json=3", **kw)
    assert r.returncode == 3                                             # test 2's own code is kept
    # box batch: the 40-step arm
    pc = sh.tmp / "probe.jsonl"
    pc.write_text('{"text": "a probe row"}\n')
    _touch(sh.out / "gen/gen.jsonl")
    _touch(sh.out / "boxset/boxes.json")
    kw = dict(PROBE_CORPUS=pc, PROBE_SHA=hashlib.sha256(pc.read_bytes()).hexdigest(), STEPS40_BOXES="law")
    r = sh.run("amoeba_fn_boxes_inner.sh", **kw)
    assert r.returncode == 0, r.stdout + r.stderr
    r = sh.run("amoeba_fn_boxes_inner.sh", rules="carves-expert-40=1", **kw)
    assert r.returncode == 1 and "40-step arm rc=1" in r.stdout
    assert sh.run("amoeba_fn_boxes_inner.sh", rules="carves-expert =4", **kw).returncode == 4


def _fake_carve(monkeypatch, tmp_path, fail):
    """research.amoeba.carve fakes for amoeba_fn_boxbatch.main: D.main refuses the boxes named in `fail`."""
    T._install(monkeypatch)
    D = sys.modules["research.amoeba.carve.__main__"]

    def main(argv, host_factory=None):
        d = Path(argv[argv.index("--out") + 1])
        if d.name.removeprefix("box-") in fail:
            raise SystemExit("gate refused")
        d.mkdir(parents=True, exist_ok=True)
        (d / "carve.json").write_text(json.dumps({"stages_done": ["bisect"]}))
    D.main = main
    C = types.ModuleType("research.amoeba.carve.carving")
    C.CleanStop = type("CleanStop", (SystemExit,), {})
    monkeypatch.setitem(sys.modules, "research.amoeba.carve.carving", C)
    model = tmp_path / "model"
    model.mkdir()
    H = sys.modules["research.amoeba.carve.host"]

    class Host(H.Host):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.adapter = types.SimpleNamespace(_tables={"ple": 1})

        def train_mode(self, on):
            pass
    H.Host = Host
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME", str(model / "ple_rows.bin"))
    bs = tmp_path / "boxset"
    bs.mkdir()
    (bs / "boxes.json").write_text(json.dumps({"boxes": [{"box": "law", "kind": "topic"}, {"box": "med", "kind": "topic"}]}))
    np.savez(tmp_path / "t2.npz", core=np.zeros((T.HOST_L, T.P.N), np.float32))
    monkeypatch.setattr(sys, "argv", ["boxbatch", "--model", str(model), "--boxset", str(bs), "--t2",
                                      str(tmp_path / "t2.npz"), "--out", str(tmp_path / "carves")])
    return T._load("amoeba_fn_boxbatch")


def test_amfn2_boxbatch_exits_nonzero_when_a_box_failed(tmp_path, monkeypatch):
    mod = _fake_carve(monkeypatch, tmp_path, fail={"law", "med"})
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert e.value.code == 1
    rows = [json.loads(x) for x in (tmp_path / "carves/verdicts.jsonl").read_text().splitlines()]
    assert [r["box"] for r in rows] == ["law", "med"] and all("gate refused" in r["error"] for r in rows)


def test_amfn2_boxbatch_still_returns_when_every_box_carved(tmp_path, monkeypatch):
    mod = _fake_carve(monkeypatch, tmp_path, fail=set())
    assert mod.main() is None
    assert len((tmp_path / "carves/verdicts.jsonl").read_text().splitlines()) == 2


# ----------------------------------------------------------------------------- AMFN-2: the day runner's markers

@shell
def test_amfn2_day_runner_marks_only_rc0_done(sh):
    me, q, src = sh.tmp / "me", sh.tmp / "queue", sh.tmp / "snapsrc"
    for d in (me, q, src / "scripts"):
        d.mkdir(parents=True)
    for name, rc in (("ok", 0), ("bad", 3), ("yield", 75)):
        (src / "scripts" / f"{name}.sh").write_text(f"echo inner {name}\nexit {rc}\n")
    plan = sh.tmp / "plan.md"
    plan.write_text("Review: passed ok\nReview: passed bad\nReview: passed yield\n")
    # the runner's checkout path is a constant: the test runs the real file with only that line repointed
    runner = sh.repo / "scripts/amoeba_fn_day_runner.sh"
    text = runner.read_text()
    line = next(x for x in text.splitlines() if x.startswith("ME="))
    runner.write_text(text.replace(line, f"ME={me}"))

    def go(stages):
        for f in q.iterdir():
            f.unlink()
        for i, n in enumerate(stages):
            (q / f"{i:02d}-{n}.stage").write_text(f"SHA=abc\nINNER=scripts/{n}.sh\n")
        (q / "END").write_text("")
        r = sh.run("amoeba_fn_day_runner.sh", QUEUE=q, STUB_SNAP_SRC=src, STUB_PLAN=plan)
        return r, sorted(f.name for f in q.iterdir() if f.suffix != ".stage" and f.name != "END")
    r, marks = go(["ok", "bad", "yield", "ok"])
    # after a yield the runner takes no further stage (a later one may need the yielded one's output)
    assert marks == ["00-ok.done", "01-bad.failed", "02-yield.yielded"], r.stdout + r.stderr
    assert r.returncode == 75 and r.stdout.count("inner yield") == 1     # requeued, not retried in a loop
    assert "rc=3" in (q / "01-bad.failed").read_text()
    r, marks = go(["ok", "bad"])
    assert r.returncode == 1 and marks == ["00-ok.done", "01-bad.failed"]
    r, marks = go(["ok"])
    assert r.returncode == 0 and marks == ["00-ok.done"]
    (q / "01-unreviewed.stage").write_text("SHA=abc\nINNER=scripts/ok.sh\n")
    r = sh.run("amoeba_fn_day_runner.sh", QUEUE=q, STUB_SNAP_SRC=src, STUB_PLAN=plan)
    assert r.returncode == 1 and (q / "01-unreviewed.failed").exists() and not (q / "01-unreviewed.done").exists()


# ----------------------------------------------------------------------------- AMFN-3: window wrappers

@shell
@pytest.mark.parametrize("rc", [0, 3, 75])
def test_amfn3_window_q_returns_the_job_code(sh, rc):
    r = sh.run("amoeba27b_window_q.sh", "tag", sh.tmp / "runs", "--", "true", QUIET_MIN=0,
               GATE_FILE=sh.tmp / "gate.pid", rules=f"run_with_service_paused.py={rc}")
    assert r.returncode == rc and "8282 healthy" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("name", ["amoeba27b_window_q.sh", "amoeba27b_window.sh", "amoeba27b_det_window.sh"])
def test_amfn3_wrappers_never_exit_zero_over_the_job_code(name):
    # the two older wrappers wait three real quiet minutes on the live GPU lock, so they are checked as text
    text = (ROOT / "scripts" / name).read_text()
    tail = text[text.index("run_with_service_paused.py --run"):]
    assert "jrc=$?" in tail and tail.count("exit $jrc") >= 2 and "exit 0" not in tail


# ----------------------------------------------------------------------------- AMFN-1: language-gate window

def test_amfn1_clock_deadline_takes_the_next_occurrence_for_a_daytime_start():
    import amoeba_fn_langgate as LG
    for hour in (7.5, 10, 14, 15, 22):
        now = time.mktime((2026, 11, 21, int(hour), int(hour % 1 * 60), 0, 0, 0, -1))
        dl = LG.clock_deadline("06:30", now)
        assert dl > now and time.localtime(dl)[3:5] == (6, 30) and dl - now < 24 * 3600
    late = time.mktime((2026, 11, 21, 6, 40, 0, 0, 0, -1))               # the cutoff has just passed
    assert LG.clock_deadline("06:30", late) == late


def test_amfn1_a_passed_cutoff_refuses_before_the_model_load(tmp_path, capsys):
    import amoeba_fn_langgate as LG
    out = tmp_path / "out"
    just_passed = time.strftime("%H:%M", time.localtime(time.time() - 300))
    with pytest.raises(SystemExit) as e:                                 # before any file of --src is read
        LG.main(["--src", str(tmp_path / "none"), "--model", str(tmp_path / "none"), "--out", str(out),
                 "--stop-new-at", just_passed])
    assert e.value.code == 75 and not out.exists() and "not started" in capsys.readouterr().out


@shell
def test_amfn1_inner_script_has_no_night_cutoff_by_day(sh):
    src = sh.tmp / "src"
    _touch(src / "partition/partition.json")
    _touch(src / "battery/battery.meta.json")
    _touch(sh.home / "mlx-models/fasttext-lid-176/lid.176.bin")
    date = sh.tmp / "bin/date"

    def at(hour, **env):
        date.write_text(f'#!/bin/sh\ncase "$1" in +%H) echo {hour:02d};; *) echo {hour:02d}:00:00;; esac\n')
        date.chmod(0o755)
        Path(sh.env["STUB_LOG"]).write_text("")
        r = sh.run("amoeba_fn_langgate_inner.sh", src, sh.tmp / "lg", **env)
        return r, Path(sh.env["STUB_LOG"]).read_text()
    r, log = at(10)
    assert r.returncode == 75 and "amoeba_fn_langgate.py" not in log    # night-only: not started
    r, log = at(10, NIGHT_ONLY=0)
    assert r.returncode == 0 and "amoeba_fn_langgate.py" in log and "--stop-new-at" not in log
    assert "--swap-growth-gb 8" in log
    r, log = at(10, NIGHT_ONLY=0, STOP_NEW_AT="13:00")
    assert "--stop-new-at 13:00" in log
    r, log = at(1)
    assert r.returncode == 0 and "--stop-new-at 06:30" in log and "--swap-growth-gb 2" in log


def test_amfn1_a_window_that_scored_no_cell_is_not_finished(tmp_path, toy_ops):
    import amoeba_fn_langgate as LG
    h = L._toy_host()
    src, model = L._fixture(tmp_path, h)
    r = LG.main(L._argv(src, model, tmp_path / "out", "--time-cap-min", "0"), host_factory=lambda log: h,
                lid_factory=lambda: (L.MockLID(), {"path": "mock", "sha256": "0", "bundled": False}))
    s = json.loads((tmp_path / "out/summary.json").read_text())
    assert r["status"] == LG.NO_CELLS != "finished" and s["status"] == LG.NO_CELLS and s["results"] == []


# ----------------------------------------------------------------------------- AMFN-4: battery path offset

def _all_open_calls():
    names = sorted(list(T.P.BOXES) + ["eng"])
    return [c for c in T.CALLS if c["sets"] == names and (c["mask"][T.P.LAYERS] > 0).all()]


def test_amfn4_path_offset_is_rescored_when_the_settings_change(toy, monkeypatch):  # noqa: F811
    T._battery(monkeypatch, toy, "bat", docs=3)
    assert len(_all_open_calls()) == 1
    assert json.loads((toy / "bat/path_offset.settings.json").read_text())["docs"] == 3
    T._battery(monkeypatch, toy, "bat", docs=3)                          # same settings: reused
    assert len(_all_open_calls()) == 0
    T._battery(monkeypatch, toy, "bat", docs=4)                          # other held documents, same names
    assert len(_all_open_calls()) == 1
    assert json.loads((toy / "bat/path_offset.settings.json").read_text())["docs"] == 4
    (toy / "bat/path_offset.settings.json").unlink()                     # an offset of unknown documents
    T._battery(monkeypatch, toy, "bat", docs=4)
    assert len(_all_open_calls()) == 1


# ----------------------------------------------------------------------------- AMFN-5: suff reference rows

def test_amfn5_reference_rows_are_the_latest_run_on_the_meta_held_specs(tmp_path):
    S = T._load("amoeba_fn_suff")
    old = {"docs": 80, "seed": 0, "partition": "P", "held": ["a=old.jsonl@256", "b=old-b.jsonl@256"]}
    new = dict(old, held=["a=new.jsonl@256", "b=new-b.jsonl@256"])
    newer = dict(new, docs=40)
    rows = [{"box": "a", "settings": old, "without": {"a": 9.0}, "random": {"a": 0.0}},
            {"box": "b", "settings": old, "without": {"b": 8.0}, "random": {"b": 0.0}},
            {"box": "b", "settings": new, "without": {"b": 0.4}, "random": {"b": 0.1}},
            {"box": "a", "settings": new, "without": {"a": 0.2}, "random": {"a": 0.1}}]
    (tmp_path / "battery.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ref, s = S.reference_rows(tmp_path, ["b=new-b.jsonl@256", "a=new.jsonl@256"])
    assert s == new and [r["box"] for r in ref] == ["b", "a"]
    assert S.necessity(ref[1]) == pytest.approx(0.1)
    ref, s = S.reference_rows(tmp_path)                                  # no meta: still the latest run's rows
    assert s == new
    with (tmp_path / "battery.jsonl").open("a") as fh:
        fh.write(json.dumps({"box": "a", "settings": newer, "without": {"a": 0.3}, "random": {"a": 0.0}}) + "\n")
    ref, s = S.reference_rows(tmp_path, new["held"])
    assert s == newer and len(ref) == 1                                  # main's docs / seed guard then refuses
    with pytest.raises(SystemExit, match="held specs"):
        S.reference_rows(tmp_path, ["a=other.jsonl@256"])


def test_amfn5_suff_ignores_stale_rows_of_an_earlier_battery_run(toy, monkeypatch):  # noqa: F811
    T._battery(monkeypatch, toy, "bat")
    bj = toy / "bat/battery.jsonl"
    rows = [json.loads(x) for x in bj.read_text().splitlines()]
    stale = [dict(r, settings=dict(r["settings"], held=["a=held/gone.jsonl@8"]),
                  core_plus={k: v + 1.0 for k, v in r["core_plus"].items()}) for r in reversed(rows)]
    bj.write_text("".join(json.dumps(r) + "\n" for r in stale + rows))
    _, rc = T._suff(monkeypatch, toy, "s")
    assert rc == 0
    summ = json.loads((toy / "s/summary.json").read_text())
    assert all(v["reproduces_battery_core_plus"] < 1e-12 for v in summ["boxes"].values())     # the current rows


# ----------------------------------------------------------------------------- AMFN-6 (stats) / AMFN-9

def test_amfn6_nan_is_never_served_yes():
    import amoeba_battery_stats as ST
    n = 12
    fin = lambda v: np.full(n, v) + np.linspace(-0.001, 0.001, n)       # noqa: E731
    nan = np.full(n, np.nan)
    own = lambda cp, ca: {"core_plus": cp, "core_alone": ca, "core_plus_rand": ca, "core_plus_rand_inexp": ca}  # noqa: E731
    assert ST.suff_box(own(fin(0.03), fin(0.9)), necessity_net=0.5)["served"] == "yes"
    for cp, ca in ((nan, nan), (nan, fin(0.3)), (fin(0.03), nan)):
        r = ST.suff_box(own(cp, ca), necessity_net=0.5)
        assert r["served"] == "no" and not r["useful"]
    assert ST.suff_box(own(fin(0.01), fin(0.02)), necessity_net=0.5)["served"] == "vacuous"


def test_amfn9_cf_is_censored_only_when_every_cell_is_under_the_bar():
    import amoeba_battery_stats as ST
    assert ST.cf_at([0.1, 0.2], [0.01, 0.02], 0.05) == (0.2, True)
    cf, cens = ST.cf_at([0.1, 0.2, 0.3], [0.2, 0.2, 0.01], 0.05)        # back under the bar at the last cell
    assert not cens
    cf, cens = ST.cf_at([0.1, 0.2, 0.3], [0.01, 0.07, 0.01], 0.05)
    assert not cens
    cf, cens = ST.cf_at([0.1, 0.2, 0.3], [0.01, 0.02, 0.03], 0.05, [False, True, False])
    assert cf == pytest.approx(0.3) and not cens                         # a flagged cell counts as above the bar
    cf, cens = ST.cf_at([0.1, 0.2, 0.3, 0.4], [0.01, 0.03, 0.07, 0.2], 0.05)
    assert cf == pytest.approx(0.25) and not cens


# ----------------------------------------------------------------------------- AMFN-8: Pareto resume

def test_amfn8_no_causal_resume_after_causal_cells_still_writes_the_held_verdict(toy, monkeypatch):  # noqa: F811
    T._pareto(monkeypatch, toy, "p")
    out = toy / "p"
    cells = [json.loads(x) for x in (out / "cells.jsonl").read_text().splitlines()]
    assert any(c["cell"] == "O-causal0@0.1" for c in cells) and (out / "attrib.npz").exists()
    # attempt 1 was killed after the causal grid: no held cells, no held verdict yet
    (out / "cells.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cells if not c["cell"].startswith("held:")))
    (out / "held_verdict.json").unlink()
    load = T._load

    def causal_choice(name):
        """The toy's best val order is O-reap-n: make the val choice the causal order (as on a model where it wins)."""
        mod = load(name)
        summarize = mod.summarize

        def forced(*a, **k):
            summ = summarize(*a, **k)
            if summ.get("choice"):
                summ["choice"] = dict(summ["choice"], order="O-causal0", schedule="base", comp="zero")
            return summ
        mod.summarize = forced
        return mod
    monkeypatch.setattr(T, "_load", causal_choice)
    _, rc = T._pareto(monkeypatch, toy, "p", "--no-causal")              # KeyError: 'O-causal0' before the fix
    assert rc == 0
    hv = json.loads((out / "held_verdict.json").read_text())
    assert hv["choice"]["cell"].startswith("held:O-causal0|base|zero@") and hv["choice"]["dce"] is not None
    assert json.loads((out / "status.json").read_text())["status"] == "finished"
