"""Build directory, stage receipts and resume.

    <out>/build.json             schema mlx2.amoeba-build.v1: model identity, inspection, plan, config, argv
    <out>/receipts/<stage>.json  schema mlx2.amoeba-stage.v1: status, times, input digest, outputs, summary
    <out>/events.jsonl           progress events (progress.py)
    <out>/logs/<stage>.log       every child's output
    <out>/<stage>/               the stage's files

A stage is done when its receipt says ``done`` and its input digest (the build config plus the output
digests of the stages it reads) still matches; resume skips it. A mismatch refuses instead of silently
reusing stale results (``--redo STAGE`` reruns it). The checkpoint fingerprint is re-checked on every
resume. Every receipt names ``<artifact>@<fingerprint>`` and the mlx2 commit.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

BUILD_SCHEMA = "mlx2.amoeba-build.v1"
STAGE_SCHEMA = "mlx2.amoeba-stage.v1"
# the map clusters the non-core units (s5.1.2), so the core is carved first
DEPENDS = {"core": (), "map": ("core",), "boxes": ("map", "core"), "partition": ("map", "core", "boxes"),
           "gates": ("boxes",), "bundle": ("map", "core", "boxes", "partition", "gates")}
# targeted extraction (mapping-workflow s11.1): one topic's box on a donor
# the core carve waits for the gap verdict: a donor the screen ruled out costs no GPU carve
EXTRACT_DEPENDS = {"gap": (), "core": ("gap",), "map": ("core",), "box": ("map", "core"), "box_gate": ("box",),
                   "box_bundle": ("gap", "map", "core", "box", "box_gate")}
# install an extracted box into another model (mapping-workflow s12.2-12.3)
INSTALL_DEPENDS = {"pair": (), "topo": ("pair",), "route_side": ("pair", "topo"), "route_tile": ("pair", "topo"),
                   "route_lora": ("pair", "topo"), "route_bridge": ("pair", "topo"),
                   "report": ("pair", "topo", "route_side", "route_tile", "route_lora", "route_bridge")}
KINDS = {"build": DEPENDS, "extract": EXTRACT_DEPENDS, "install": INSTALL_DEPENDS}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    tmp.replace(path)


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def process_started(pid: int) -> float | None:
    """When process ``pid`` started (epoch s, from ``ps``), or None when that cannot be read. From the elapsed
    time ([[dd-]hh:]mm:ss): ``lstart`` is local wall time, which repeats an hour when DST ends."""
    import subprocess
    try:
        out = subprocess.run(["ps", "-o", "etime=", "-p", str(int(pid))], capture_output=True, text=True, timeout=15,
                             check=False, env={**os.environ, "LC_ALL": "C", "LANG": "C"}).stdout.strip()
        if not out:
            return None
        days, _, rest = out.rpartition("-")
        secs = 0
        for part in rest.split(":"):
            secs = secs * 60 + int(part)
        return time.time() - (int(days or 0) * 86400 + secs)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def process_alive(pid, recorded_at: float | None = None) -> bool:
    """Whether the process that wrote a record (a lock, a ``running`` receipt) still runs. A pid the system has
    handed to a process born after the record was written is not that process."""
    try:
        pid = int(pid)
        os.kill(pid, 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        pass                                         # exists, owned by another user
    born = process_started(pid) if recorded_at else None
    return not (born is not None and born > float(recorded_at) + 2.0)


def group_alive(pgid, recorded_at: float | None = None) -> bool:
    """Whether a child process group recorded in the build lock still has a process (children run in their own
    sessions, so they outlive a killed parent). Another user's group (EPERM) counts as alive; a group id whose
    leader was born after the record is a recycled id."""
    try:
        pgid = int(pgid)
        if pgid <= 1:
            return False
        os.killpg(pgid, 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        pass
    born = process_started(pgid) if recorded_at else None
    return not (born is not None and born > float(recorded_at) + 2.0)


def lock_live_holder(held: dict) -> str | None:
    """What still runs for a lock record: its process, or a child group that process left behind; else None."""
    if process_alive(held.get("pid"), held.get("created")):
        return f"pid {held.get('pid')}, since {held.get('started')}"
    for g in held.get("groups") or []:
        if isinstance(g, dict) and group_alive(g.get("pgid"), g.get("t")):
            return (f"child process group {g.get('pgid')} of pid {held.get('pid')}, which is gone; the child still "
                    f"writes this build (stop it with `kill -TERM -{g.get('pgid')}`)")
    return None


def stage_commits(b) -> dict:
    """{stage: the mlx2 commit its receipt says it ran at}, for the manifests a later stage writes: a bundle made at
    one commit never restamps results made at another."""
    if not hasattr(b, "receipt"):
        return {}
    out = {}
    for st in KINDS[b.rec.get("kind", "build")]:
        r = b.receipt(st)
        c = (r or {}).get("mlx2_commit")
        if c is not None:
            # resumed inside the stage at another commit: its sub-step files come from each attempt
            mixed = isinstance(c, dict) and len(r.get("attempt_commits") or []) > 1
            out[st] = {**c, "attempt_commits": r["attempt_commits"]} if mixed else c
    return out


class Build:
    def __init__(self, out: Path):
        self.out = Path(out).expanduser().resolve()
        self.path = self.out / "build.json"
        self.rec = json.loads(self.path.read_text()) if self.path.exists() else None

    # ------------------------------------------------------------ creation / identity
    @classmethod
    def create(cls, out, rec: dict) -> "Build":
        out = Path(out).expanduser().resolve()
        if out.exists() and any(p.name not in (".DS_Store",) for p in out.iterdir()):
            raise SystemExit(f"{out} exists and is not empty: use `mlx2 amoeba resume --out {out}` or a fresh --out "
                             "(retries stay distinguishable)")
        out.mkdir(parents=True, exist_ok=True)
        b = cls(out)
        b.rec = {"schema": BUILD_SCHEMA, "created": now(), **rec}
        write_json(b.path, b.rec)
        return b

    def require(self) -> "Build":
        if self.rec is None:
            raise SystemExit(f"{self.out}: no build.json (start with `mlx2 amoeba build --model M --out {self.out}`)")
        if self.rec.get("schema") != BUILD_SCHEMA:
            raise SystemExit(f"{self.path}: schema {self.rec.get('schema')!r}, expected {BUILD_SCHEMA}")
        return self

    def save(self) -> None:
        write_json(self.path, self.rec)

    @property
    def cfg(self) -> dict:
        return self.rec["config"]

    @property
    def info(self) -> dict:
        return self.rec["inspection"]

    @property
    def plan(self) -> dict:
        return self.rec["plan"]

    @property
    def checkpoint(self) -> str:
        return self.info["checkpoint"]

    def stage_dir(self, stage: str) -> Path:
        d = self.out / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def log_path(self, stage: str) -> Path:
        p = self.out / "logs" / f"{stage}.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------ one run per build directory
    @property
    def lock_path(self) -> Path:
        return self.out / ".amoeba.lock"

    def acquire_lock(self) -> None:
        """One resume / stage command per build directory: a second one would re-enter a stage that is still
        running and delete its partial outputs. The lock of a dead process whose children are gone too is taken
        over (said on stderr)."""
        import fcntl
        guard = None
        try:                                         # one taker at a time checks and replaces a stale lock
            guard = os.open(self.out, os.O_RDONLY)
            fcntl.flock(guard, fcntl.LOCK_EX)
        except OSError:
            pass                                     # no flock on this filesystem: the rename below picks one winner
        try:
            self._acquire_lock()
        finally:
            if guard is not None:
                os.close(guard)

    def _acquire_lock(self) -> None:
        import sys
        me = {"pid": os.getpid(), "created": time.time(), "started": now(), "argv": sys.argv}
        for _ in range(5):
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                try:
                    raw = self.lock_path.read_text()
                    age = time.time() - self.lock_path.stat().st_mtime
                except FileNotFoundError:
                    continue                         # released between the failed create and this read
                except OSError:
                    raw, age = "", 0.0
                try:
                    held = json.loads(raw)
                except ValueError:
                    held = {}
                if not isinstance(held, dict):
                    held = {}
                mine = held.get("pid") == os.getpid()
                if mine and process_alive(os.getpid(), held.get("created")):
                    return                           # this process already holds it
                if not held and age < 10:
                    raise SystemExit(f"{self.out}: another mlx2 amoeba run is taking the build lock; retry")
                # a record with this pid from before this process was born is a dead holder's: only its children count
                live = lock_live_holder({**held, "pid": None} if mine else held) if held else None
                if live:
                    raise SystemExit(f"{self.out}: another mlx2 amoeba run holds this build ({live}): refusing, so "
                                     f"its partial outputs are not deleted under it. Wait for it or stop it "
                                     f"({self.lock_path})")
                # rename, not unlink: of several resumes that judged the same lock stale, one rename succeeds
                stale = self.lock_path.with_name(f"{self.lock_path.name}.stale.{os.getpid()}.{time.time_ns()}")
                try:
                    os.rename(self.lock_path, stale)
                except FileNotFoundError:
                    continue                         # another resume took it over first: look again
                if stale.read_text() != raw:         # that resume's new lock, not the record judged stale: put it back
                    try:
                        os.link(stale, self.lock_path)
                    except FileExistsError:
                        pass
                    except OSError:                  # no hard links here
                        if not self.lock_path.exists():
                            os.rename(stale, self.lock_path)
                    stale.unlink(missing_ok=True)
                    raise SystemExit(f"{self.out}: another mlx2 amoeba run is taking the build lock; retry")
                stale.unlink(missing_ok=True)
                print(f"{self.out}: taking over the stale build lock of pid {held.get('pid')} (started "
                      f"{held.get('started')}; that process is gone)", file=sys.stderr)
                continue
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(me) + "\n")
            return
        raise SystemExit(f"{self.out}: could not take the build lock ({self.lock_path}); retry")

    def note_child_group(self, pgid: int) -> None:
        """Add a child's process group to this process's lock record: the child runs in its own session and
        outlives a killed parent, and the lock stays held while any recorded group has a process."""
        try:
            held = json.loads(self.lock_path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(held, dict) or held.get("pid") != os.getpid():
            return
        groups = [g for g in held.get("groups") or [] if isinstance(g, dict) and group_alive(g.get("pgid"), g.get("t"))]
        write_json(self.lock_path, {**held, "groups": groups + [{"pgid": int(pgid), "t": time.time()}]})

    def release_lock(self) -> None:
        import sys
        try:
            held = json.loads(self.lock_path.read_text())
            if held.get("pid") != os.getpid():
                return
            live = [g.get("pgid") for g in held.get("groups") or []
                    if isinstance(g, dict) and group_alive(g.get("pgid"), g.get("t"))]
            if live:                                 # a child this run could not stop: the lock outlives the run
                print(f"{self.out}: child process group(s) {live} still running; the build lock stays until they "
                      f"end ({self.lock_path})", file=sys.stderr)
                return
            self.lock_path.unlink()
        except (OSError, ValueError, AttributeError):
            pass

    def refuse_if_running_elsewhere(self) -> None:
        """A receipt in ``running`` whose recorded process is alive: that run is still writing the stage's files
        (also a run from before the build lock existed). Refuse rather than delete them under it."""
        for st in KINDS[self.rec.get("kind", "build")]:
            r = self.receipt(st) or {}
            if r.get("status") != "running" or r.get("pid") == os.getpid():
                continue
            if process_alive(r.get("pid"), r.get("t0")):
                raise SystemExit(f"{self.out}: stage {st} is running in pid {r.get('pid')} (since {r.get('started')}): "
                                 "refusing to start a second run on this build directory")

    # ------------------------------------------------------------ receipts
    def receipt_path(self, stage: str) -> Path:
        return self.out / "receipts" / f"{stage}.json"

    def receipt(self, stage: str) -> dict | None:
        p = self.receipt_path(stage)
        return json.loads(p.read_text()) if p.exists() else None

    def inputs_digest(self, stage: str) -> dict:
        ups = {}
        deps = KINDS[self.rec.get("kind", "build")][stage]
        if stage == "map" and ((self.rec.get("stage_options") or {}).get("map") or {}).get("without_core"):
            deps = ()                                     # a map-only build (rolling capture): every unit is clustered
        for d in deps:
            r = self.receipt(d)
            # a stage that failed but lets the run go on (an install route whose child crashed) is a truthful
            # ``failed`` receipt the report still reads
            if not r or not (r.get("status") == "done" or (r.get("status") == "failed" and r.get("continue_past"))):
                raise SystemExit(f"stage {stage} needs stage {d} done first (receipt: "
                                 f"{(r or {}).get('status', 'missing')})")
            self.is_done(d)  # Check the dependency's inputs and actual output bytes too.
            ups[d] = r.get("outputs_digest")
        inputs = {"config": self.cfg, "fingerprint": self.info["fingerprint"], "inputs": self.rec.get("inputs"),
                  "upstream": ups, "stage_options": (self.rec.get("stage_options") or {}).get(stage)}
        return {"digest": digest(inputs), "upstream": ups}

    def is_done(self, stage: str) -> bool:
        r = self.receipt(stage)
        if not r:
            return False
        cur = self.inputs_digest(stage)["digest"]
        if r.get("inputs_digest") != cur:
            raise SystemExit(f"stage {stage}: started with inputs {r.get('inputs_digest')}, but the inputs are now {cur} "
                             f"(config, checkpoint or an upstream stage changed); rerun it with --redo {stage}")
        if r.get("status") != "done":
            return False
        from .paths import sha256_file
        def check_output(name, p, output):
            valid = p.is_dir() if output.get("dir") else (
                p.is_file() and p.stat().st_size == output.get("bytes")
                and sha256_file(p) == output.get("sha256"))
            if not valid:
                raise SystemExit(f"stage {stage}: output {name} ({p}) missing or changed; rerun with --redo {stage}")

        for name, output in (r.get("outputs") or {}).items():
            if not isinstance(output, dict) or "path" not in output:
                continue
            p = Path(output["path"])
            check_output(name, p, output)
            if stage == "box_bundle" and name == "manifest":
                # The receipt binds the index. The index binds the carried files,
                # including legacy single-file and current sharded weight packs.
                for rel, member in json.loads(p.read_text()).get("files", {}).items():
                    check_output(rel, p.parent / rel, member)
        return True

    def begin(self, stage: str, commit: dict) -> dict:
        ins = self.inputs_digest(stage)
        prev = self.receipt(stage) or {}
        # every commit an attempt of this stage ran at: a resume reuses the earlier attempts' sub-step files
        commits = list(prev.get("attempt_commits") or ([prev["mlx2_commit"]] if prev.get("mlx2_commit") else []))
        if commit not in commits:
            commits.append(commit)
        # the upstream continue-past failures this attempt reads (an install report over a crashed route), on the
        # receipt so that any later resume remakes this stage once that route has been retried
        failed_up = {}
        for d in ins["upstream"]:
            u = self.receipt(d) or {}
            if u.get("status") == "failed" and u.get("continue_past"):
                failed_up[d] = u.get("outputs_digest")
        r = {"schema": STAGE_SCHEMA, "stage": stage, "status": "running", "checkpoint": self.checkpoint,
             "fingerprint": self.info["fingerprint"], "mlx2_commit": commit, "profile": self.cfg["profile"],
             "inputs_digest": ins["digest"], "upstream": ins["upstream"], "started": now(), "t0": time.time(),
             "attempts": int(prev.get("attempts", 0)) + 1, "pid": os.getpid(), "attempt_commits": commits}
        if failed_up:
            r["failed_upstream"] = failed_up
        write_json(self.receipt_path(stage), r)
        return r

    def read_a_failure_since_retried(self, stage: str) -> list:
        """The upstream stages whose continue-past failure this stage's receipt read and which have since been
        retried to another result (done, or failed again differently). A retry that has not finished (stopped,
        interrupted, failed outright) is not listed: the stage keeps its receipt until the retry has a result."""
        out = []
        for d, dig in ((self.receipt(stage) or {}).get("failed_upstream") or {}).items():
            u = self.receipt(d) or {}
            still = u.get("status") == "failed" and u.get("continue_past")
            if u.get("status") == "done" or (still and u.get("outputs_digest") != dig):
                out.append(d)
        return out

    def end(self, r: dict, status: str, outputs: dict | None = None, summary: dict | None = None,
            error: str | None = None) -> dict:
        r = {**r, "status": status, "finished": now(), "wall_s": round(time.time() - r["t0"], 1)}
        if outputs is not None:
            r["outputs"] = outputs
            r["outputs_digest"] = digest(outputs)
        if summary is not None:
            r["summary"] = summary
        if error:
            r["error"] = error[-4000:]
        write_json(self.receipt_path(r["stage"]), r)
        return r

    def forget(self, stage: str) -> None:
        """--redo: archive the stage's receipt and its directory (every sub-step output), so it reruns from
        nothing; the old evidence stays under <out>/.redo/ and the receipts/ archive."""
        ts = int(time.time())
        p = self.receipt_path(stage)
        if p.exists():
            r = json.loads(p.read_text())
            write_json(p.with_name(f"{stage}.redo-{ts}.json"), r)
            p.unlink()
        d = self.out / stage
        if d.exists():
            dst = self.out / ".redo" / f"{stage}-{ts}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            d.rename(dst)


def file_outputs(paths: dict) -> dict:
    """{name: path} -> {name: {path, sha256, bytes}} for the files a stage hands downstream."""
    from .paths import sha256_file
    out = {}
    for k, p in paths.items():
        p = Path(p)
        if p.is_file():
            out[k] = {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        elif p.is_dir():
            out[k] = {"path": str(p), "dir": True}
        else:
            raise SystemExit(f"stage output {k}: {p} is missing")
    return out
