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
            if not r or r.get("status") != "done":
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
        for name, output in (r.get("outputs") or {}).items():
            if not isinstance(output, dict) or "path" not in output:
                continue
            p = Path(output["path"])
            valid = p.is_dir() if output.get("dir") else (
                p.is_file() and p.stat().st_size == output.get("bytes")
                and sha256_file(p) == output.get("sha256"))
            if not valid:
                raise SystemExit(f"stage {stage}: output {name} ({p}) missing or changed; rerun with --redo {stage}")
        return True

    def begin(self, stage: str, commit: dict) -> dict:
        ins = self.inputs_digest(stage)
        prev = self.receipt(stage) or {}
        r = {"schema": STAGE_SCHEMA, "stage": stage, "status": "running", "checkpoint": self.checkpoint,
             "fingerprint": self.info["fingerprint"], "mlx2_commit": commit, "profile": self.cfg["profile"],
             "inputs_digest": ins["digest"], "upstream": ins["upstream"], "started": now(), "t0": time.time(),
             "attempts": int(prev.get("attempts", 0)) + 1, "pid": os.getpid()}
        write_json(self.receipt_path(stage), r)
        return r

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
