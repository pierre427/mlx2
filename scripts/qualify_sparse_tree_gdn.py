#!/usr/bin/env python3
"""Native D128 recurrent gate for the sparse parent-restart tree GDN candidate.

Scope: NATIVE-ONLY, synthetic, recurrent-state gate for
``mlx2.adapters.qwen38_tree_gdn``. The conv window, cache commit, real model,
serving and performance all remain pending; no cache operation is performed.
Passing qualifies nothing beyond this recurrent comparison.

Per case (B1, D128, one case at a time): one sparse forward launch, then one
replay launch per row (every root-to-node accepted path). The reference is
the ordinary production M1 kernel, forced unpacked
(``gated_delta._gated_delta_kernel_impl(..., allow_packed=False)``), run one
token per call along each path from the entry state. Every forward readout
row and every replayed fp32 state is compared with the reference on FULL raw
storage bits (bf16 as uint16, fp32 as uint32) with dtype and shape; signed
zero counts as a difference. No allclose, no shared host step.

Fails closed (refused) on: no ``--i-own-the-gpu``; no explicit
``MLX2_INTAKE_SOURCE_COMMIT``; non-GPU default device or no Metal; missing
device/build identity; a source hash that changed between start and end;
any missing, partial or duplicate case, row or path; dtype or shape drift;
any non-finite value (matching or not); an all-zero (vacuous) reference;
any launch whose engagement was not confirmed on the
device; a confirmed-launch count different from the expected one. Bit
differences are counterexamples.

``--i-own-the-gpu`` is an operator acknowledgement only. Lease and lock
ownership come from the reviewed /tmp/mlx2-intake/stage3_gpu.py wrapper:

  MLX2_INTAKE_SOURCE_COMMIT=<sha> python /tmp/mlx2-intake/stage3_gpu.py sparse-tree-gdn-native \\
      .venv/bin/python scripts/qualify_sparse_tree_gdn.py --i-own-the-gpu --out <receipt>.json

Nothing in this module imports MLX at import time; the ordinary reference
module (which builds Metal kernels at import) is imported only after
admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SCHEMA = "mlx2.native.sparse-tree-gdn-recurrent-gate.v1"
SCOPE = ("native-only synthetic recurrent-state gate (B1, D128); conv window, cache commit, real model, "
         "serving and performance pending; no cache operations")
SOURCE_FILES = (
    "scripts/qualify_sparse_tree_gdn.py",
    "src/mlx2/adapters/qwen38_tree_gdn.py",
    "src/mlx2/runtime/models/gated_delta.py",
    "provenance/qwen38-tree-gdn-sparse-restart.json",
    "provenance/sparse-tree-gdn-native-qualifier.json",
    "scripts/paired_direct_ab.py",  # mlx_identity (build identity helper)
)
# Imported modules must be these files of THIS checkout (no sister-checkout imports).
MODULE_PATHS = {"candidate": "src/mlx2/adapters/qwen38_tree_gdn.py",
                "reference": "src/mlx2/runtime/models/gated_delta.py",
                "identity_helper": "scripts/paired_direct_ab.py"}
SHA256 = re.compile(r"[0-9a-f]{64}")
UPSTREAM = {
    "repository": "https://github.com/ddalcu/mlx-serve", "pull": 645, "state": "open",
    "head": "eea96c8738b953dcfdfeea5072e84993e93b3fbc", "base": "a453794f9e95b8588b9338464ca13599d28caf33",
    "updated_at": "2026-10-01T00:53:40Z", "file": "src/gdn_decode.zig",
    "file_blob": "74283c10085d02f28bbcc85773fd8991075c04f5",
    "file_sha256": "f416de892210819fc0d6f6e0def1d7b6f00d55aa4614e703c5fd49d7f70a5815",
    "future_conv_gate": "K1P conv rows now written at TL 1 and 2: a conv-window gate must include singleton and width-2 trees",
}
HEAD_DIM = 128
DTYPES = ("bfloat16", "float32")
GEOMETRIES = {"qwen38_27b": (16, 48), "ratio_2x": (2, 4), "ratio_1x2": (1, 2)}
WIDTH32 = (-1, 0, 1, 1, 3, 2, 5, 0, 7, 8, 4, -1, 11, 12, 6, 14, 15, 9, 17, 13,
           19, 20, 10, 22, 16, 24, 25, 18, 27, 21, 29, 23)
TOPOLOGIES = {
    "singleton": (-1,),
    "width2": (-1, 0),
    "chain": (-1, 0, 1, 2, 3, 4),
    "branch": (-1, 0, 0, 1, 1, 2, 5),
    "forest": (-1, 0, -1, 2, 0, -1, 5),
    "width32": WIDTH32,
    "short_after_wide": (-1, 0),  # runs immediately after width32 of the same dtype and geometry
}
BITS_DTYPE = {"bfloat16": "uint16", "float32": "uint32"}
EXPONENT = {"bfloat16": 0x7F80, "float32": 0x7F800000}
MAGNITUDE = {"bfloat16": 0x7FFF, "float32": 0x7FFFFFFF}
MAX_TIME_S = 7200.0


class Refused(RuntimeError):
    """The gate cannot run or cannot be evaluated; never a pass."""


# ================================================================ catalogue

def path_to(parents, node):
    path = [node]
    while parents[path[-1]] != -1:
        path.append(parents[path[-1]])
    return path[::-1]


def case_key(topology, dtype, geometry):
    return f"{topology}/{dtype}/{geometry}"


def catalogue(topologies=None, dtypes=None, geometries=None):
    """Ordered cases; short_after_wide always directly follows width32."""
    topologies = list(TOPOLOGIES) if not topologies else list(topologies)
    if "short_after_wide" in topologies and "width32" not in topologies:
        raise Refused("short_after_wide needs width32 in the same run")
    cases = []
    for geometry in (geometries or GEOMETRIES):
        for dtype in (dtypes or DTYPES):
            for topology in TOPOLOGIES:  # canonical order keeps width32 -> short_after_wide adjacent
                if topology in topologies:
                    hk, hv = GEOMETRIES[geometry]
                    cases.append({"case": case_key(topology, dtype, geometry), "topology": topology,
                                  "dtype": dtype, "geometry": geometry, "hk": hk, "hv": hv,
                                  "parents": list(TOPOLOGIES[topology])})
    return cases


def case_inputs(case, seed):
    """Deterministic finite synthetic inputs (numpy, fp32) for one case."""
    import numpy as np

    digest = hashlib.sha256(f"{seed}:{case['case']}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    width, hk, hv, d = len(case["parents"]), case["hk"], case["hv"], HEAD_DIM

    def unit(shape):
        x = rng.standard_normal(shape).astype(np.float32)
        return x / np.linalg.norm(x, axis=-1, keepdims=True).astype(np.float32)

    return {
        "q": unit((1, width, hk, d)) * np.float32(d ** -0.5),
        "k": unit((1, width, hk, d)),
        "v": rng.standard_normal((1, width, hv, d)).astype(np.float32),
        "g": rng.uniform(0.80, 0.999, (1, width, hv)).astype(np.float32),
        "beta": rng.uniform(0.05, 0.95, (1, width, hv)).astype(np.float32),
        "state": (rng.standard_normal((1, hv, d, d)) * 0.05).astype(np.float32),
    }


# ================================================================ evidence and evaluation

def inspect(evidence, *, dtype, shape):
    """Problems with one array's evidence (empty list when usable)."""
    import numpy as np

    if not isinstance(evidence, dict) or not {"dtype", "shape", "bits"} <= set(evidence):
        return ["evidence missing"]
    bits = evidence["bits"]
    if evidence["dtype"] != dtype:
        return [f"dtype {evidence['dtype']} != {dtype}"]
    if list(evidence["shape"]) != list(shape):
        return [f"shape {list(evidence['shape'])} != {list(shape)}"]
    if not isinstance(bits, np.ndarray) or str(bits.dtype) != BITS_DTYPE[dtype] or list(bits.shape) != list(shape):
        return ["raw bits unavailable or not the storage type"]
    nonfinite = int(np.count_nonzero((bits & EXPONENT[dtype]) == EXPONENT[dtype]))
    return [f"{nonfinite} non-finite values"] if nonfinite else []


def compare_bits(candidate, reference):
    """Full raw storage-bit comparison of two already-inspected arrays."""
    import numpy as np

    a, b = candidate["bits"], reference["bits"]
    differ = a != b
    zero_a = (a & MAGNITUDE[candidate["dtype"]]) == 0
    zero_b = (b & MAGNITUDE[reference["dtype"]]) == 0
    return {"equal": not bool(differ.any()), "mismatches": int(np.count_nonzero(differ)),
            "signed_zero_mismatches": int(np.count_nonzero(differ & zero_a & zero_b)),
            "candidate_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(b.tobytes()).hexdigest()}


def _hashes_ok(hashes):
    return isinstance(hashes, dict) and set(hashes) == set(SOURCE_FILES) and \
        all(isinstance(v, str) and SHA256.fullmatch(v) for v in hashes.values())


def evaluate_identity(identity, *, final=True):
    """Identity refusals. ``final=False`` is the preflight run before any case."""
    refusals = []
    if not isinstance(identity, dict):
        return ["identity missing"]
    if identity.get("ownership_acknowledged") is not True:
        refusals.append("no --i-own-the-gpu acknowledgement")
    if not re.fullmatch(r"[0-9a-f]{7,40}", str(identity.get("source_commit") or "")):
        refusals.append("no explicit MLX2_INTAKE_SOURCE_COMMIT")
    if identity.get("default_device") != "gpu":
        refusals.append("default device is not the GPU")
    device = identity.get("device_info")
    if not isinstance(device, dict) or not device.get("architecture"):
        refusals.append("device identity missing")
    mlx = identity.get("mlx")
    if not isinstance(mlx, dict) or not mlx.get("version") or \
            not SHA256.fullmatch(str(mlx.get("metallib_sha256") or "")):
        refusals.append("MLX build identity missing (version and 64-hex metallib sha256)")
    modules = identity.get("modules")
    for name, rel in MODULE_PATHS.items():
        path = modules.get(name) if isinstance(modules, dict) else None
        if not path or Path(path).resolve() != (ROOT / rel).resolve():
            refusals.append(f"{name} module is not {rel} of this checkout ({path})")
    before = identity.get("sources_before")
    if not _hashes_ok(before):
        refusals.append("source hashes missing or not 64-hex sha256")
    else:
        stages = [("at import", identity.get("sources_after_import"))]
        if final:
            stages.append(("during the run", identity.get("sources_after")))
        for when, later in stages:
            if not _hashes_ok(later):
                refusals.append(f"source hashes {when} missing or not 64-hex sha256")
            elif later != before:
                refusals.append(f"source changed {when}: {sorted(k for k in before if later[k] != before[k])}")
    if identity.get("source_commit_status") not in ("verified_git_revision", "explicit_label_unverified"):
        refusals.append("source commit status missing")
    refusals += [r for r in source_refusals(identity.get("source_commit"), before, identity.get("git"))
                 if not r.startswith("source hashes")]
    if identity.get("reference") != "gated_delta._gated_delta_kernel_impl(allow_packed=False)":
        refusals.append("ordinary reference is not the unpacked production kernel")
    return refusals


def _same_ints(got, want):
    """Equal lists of plain ints only (True == 1 and 1.0 == 1 do not count)."""
    return isinstance(got, list) and len(got) == len(want) and \
        all(type(g) is int and g == w for g, w in zip(got, want))


def evaluate(report, expected):
    """Verdict over a report; pure (numpy only, no MLX, no device)."""
    refusals, differences, comparisons = [], [], 0
    refusals += evaluate_identity(report.get("identity"))
    records = report.get("cases")
    if not isinstance(records, list) or not records:
        refusals.append("no case evidence")
        records = []
    refusals += [f"case entry {i} is not a dict" for i, r in enumerate(records) if not isinstance(r, dict)]
    keys = [r.get("case") for r in records if isinstance(r, dict)]
    want = [c["case"] for c in expected]
    for key in sorted({k for k in keys if keys.count(k) > 1}):
        refusals.append(f"duplicate case {key}")
    refusals += [f"missing case {k}" for k in want if k not in keys]
    refusals += [f"unexpected case {k}" for k in keys if k not in want]
    by_key = {c["case"]: c for c in expected}
    expected_total = 0
    for index, record in enumerate(records):
        case = by_key.get(record.get("case")) if isinstance(record, dict) else None
        if case is None:
            continue
        name, parents = case["case"], case["parents"]
        width, hv = len(parents), case["hv"]
        expected_total += 1 + width
        if [record.get(k) for k in ("topology", "dtype")] != [case["topology"], case["dtype"]] or \
                not _same_ints([record.get("hk"), record.get("hv")], [case["hk"], case["hv"]]) or \
                not _same_ints(record.get("parents"), parents):
            refusals.append(f"{name}: geometry or topology differs from the catalogue")
            continue
        if case["topology"] == "short_after_wide":
            previous = records[index - 1] if index else None
            if not isinstance(previous, dict) or \
                    previous.get("case") != case_key("width32", case["dtype"], case["geometry"]):
                refusals.append(f"{name}: did not run directly after width32")
        forward = record.get("forward") or {}
        if forward.get("engaged") is not True:
            refusals.append(f"{name}: forward engagement not confirmed")
        problems = inspect(forward.get("y"), dtype=case["dtype"], shape=(1, width, hv, HEAD_DIM))
        refusals += [f"{name}: forward y {p}" for p in problems]
        forward_ok = not problems
        nodes = record.get("nodes")
        if not isinstance(nodes, list):
            refusals.append(f"{name}: rows are not a list")
            nodes = []
        refusals += [f"{name}: row entry {i} is not a dict" for i, n in enumerate(nodes) if not isinstance(n, dict)]
        raw_ids = [n.get("node") for n in nodes if isinstance(n, dict)]
        refusals += [f"{name}: row id {i!r} is not a plain int" for i in raw_ids if type(i) is not int]
        ids = [i for i in raw_ids if type(i) is int]
        for node in sorted({i for i in ids if ids.count(i) > 1}, key=str):
            refusals.append(f"{name}: duplicate row {node}")
        refusals += [f"{name}: missing row {r}" for r in range(width) if r not in ids]
        refusals += [f"{name}: unexpected row {r}" for r in ids if r not in range(width)]
        if type(record.get("confirmed_launches")) is not int or record["confirmed_launches"] != 1 + width:
            refusals.append(f"{name}: confirmed launches {record.get('confirmed_launches')} != {1 + width}")
        for entry in nodes:
            node = entry.get("node") if isinstance(entry, dict) else None
            if type(node) is not int or node not in range(width):
                continue
            label = f"{name} row {node}"
            if not _same_ints(entry.get("path"), path_to(parents, node)):
                refusals.append(f"{label}: path is not the root-to-node path")
                continue
            if entry.get("replay_engaged") is not True:
                refusals.append(f"{label}: replay engagement not confirmed")
            pairs = (("readout", "candidate_y", "reference_y", case["dtype"], (1, 1, hv, HEAD_DIM)),
                     ("state", "replay_state", "reference_state", "float32", (1, hv, HEAD_DIM, HEAD_DIM)))
            for what, cand_key, ref_key, dtype, shape in pairs:
                problems = [f"candidate {p}" for p in inspect(entry.get(cand_key), dtype=dtype, shape=shape)]
                problems += [f"reference {p}" for p in inspect(entry.get(ref_key), dtype=dtype, shape=shape)]
                if problems:
                    refusals += [f"{label} {what}: {p}" for p in problems]
                    continue
                if what == "readout" and forward_ok and not (
                        entry[cand_key]["bits"] == forward["y"]["bits"][:, node:node + 1]).all():
                    refusals.append(f"{label}: candidate readout is not the forward's row")
                    continue
                if not (entry[ref_key]["bits"] & MAGNITUDE[dtype]).any():
                    refusals.append(f"{label} {what}: reference is all zero (vacuous evidence)")
                    continue
                result = compare_bits(entry[cand_key], entry[ref_key])
                entry.setdefault("comparisons", {})[what] = result
                comparisons += 1
                if not result["equal"]:
                    differences.append(f"{label} {what}: {result['mismatches']} raw-bit mismatches "
                                       f"({result['signed_zero_mismatches']} signed zero)")
    confirmed = sum(r["confirmed_launches"] for r in records
                    if isinstance(r, dict) and type(r.get("confirmed_launches")) is int)
    if records and confirmed != expected_total:
        refusals.append(f"confirmed launches {confirmed} != expected {expected_total}")
    verdict = "refused" if refusals else "counterexample" if differences else "pass"
    return {"verdict": verdict, "refusals": refusals, "differences": differences,
            "comparisons": comparisons, "expected_confirmed_launches": sum(1 + len(c["parents"]) for c in expected),
            "confirmed_launches": confirmed}


# ================================================================ running

def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() if (ROOT / name).exists() else None
            for name in SOURCE_FILES}


def git_identity():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    head = git("rev-parse", "HEAD")
    if head is None:
        return {"available": False}
    dirty = git("status", "--porcelain", "--", *SOURCE_FILES)
    if dirty is None:  # HEAD readable but status failed: unknown, never clean
        return {"available": True, "head": head, "status_ok": False, "dirty": None}
    return {"available": True, "head": head, "status_ok": True, "dirty": [line[3:] for line in dirty.splitlines()]}


def source_refusals(commit, sources, git):
    """Bound-source refusals; ``main`` applies them BEFORE the native backend exists.

    A frozen export without git keeps the commit as an explicit unverified
    label, but its file hashes are mandatory.
    """
    refusals = []
    if not _hashes_ok(sources):
        refusals.append("source hashes missing or not 64-hex sha256")
    if not isinstance(git, dict) or "available" not in git:
        refusals.append("git identity missing")
    elif git["available"]:
        if not str(git.get("head", "")).startswith(str(commit)):
            refusals.append("git HEAD does not match MLX2_INTAKE_SOURCE_COMMIT")
        if git.get("status_ok") is not True or not isinstance(git.get("dirty"), list):
            refusals.append("git status unavailable for the bound source files")
        elif git["dirty"]:
            refusals.append(f"bound source files are uncommitted: {git['dirty']}")
    return refusals


def admit(args, environ):
    """Pre-MLX admission: acknowledgement and explicit source commit."""
    if args.i_own_the_gpu is not True:
        raise Refused("refusing Metal execution without --i-own-the-gpu")
    commit = environ.get("MLX2_INTAKE_SOURCE_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        raise Refused("MLX2_INTAKE_SOURCE_COMMIT must name the bound source commit")
    return commit


def run_case(backend, case, seed, order):
    width = len(case["parents"])
    arrays = backend.arrays(case, case_inputs(case, seed))
    before = backend.confirmed()
    y, engaged = backend.forward(arrays, case["parents"])
    nodes = []
    for node in range(width):
        path = path_to(case["parents"], node)
        reference_y, reference_state = backend.reference(arrays, path)
        replay_state, replay_engaged = backend.replay(arrays, case["parents"], path)
        nodes.append({"node": node, "path": path,
                      "candidate_y": backend.row(y, node), "reference_y": reference_y,
                      "reference_state": reference_state, "replay_state": replay_state,
                      "replay_engaged": replay_engaged})
    record = {**{k: case[k] for k in ("case", "topology", "dtype", "geometry", "hk", "hv", "parents")},
              "order": order, "forward": {"y": y, "engaged": engaged}, "nodes": nodes,
              "confirmed_launches": backend.confirmed() - before}
    backend.release(arrays)
    return record


def run_gate(args, backend, cases, *, commit, sources_before=None, git=None, clock=time.monotonic):
    """Run cases one at a time. ``sources_before``/``git`` should be captured
    BEFORE the backend imported anything (``main`` does); the hashes are taken
    again right after import and at the end."""
    sources_before = source_hashes() if sources_before is None else sources_before
    git = git_identity() if git is None else git
    identity = {"ownership_acknowledged": args.i_own_the_gpu is True, "source_commit": commit,
                "ownership_note": ("operator acknowledgement only, not observed lock or CPG proof; this receipt "
                                   "must be paired with the parent wrapper's ownership receipt (lease + both flocks)"),
                # A git checkout can verify the label; a frozen export (the wrapper's source root) cannot, so
                # there the label stays an explicit, unverified claim and the file hashes are the binding.
                "source_commit_status": "verified_git_revision" if git.get("available") else "explicit_label_unverified",
                "sources_before": sources_before, "git": git, **backend.identity()}
    identity["sources_after_import"] = source_hashes()
    if evaluate_identity(identity, final=False):        # refuse before any native call
        identity["sources_after"] = source_hashes()
        report = {"identity": identity, "cases": []}
        return report, evaluate(report, cases)
    deadline = clock() + args.time_limit_s
    records = []
    for order, case in enumerate(cases):
        if clock() > deadline:
            break  # the missing cases make the verdict refused
        records.append(run_case(backend, case, args.seed, order))
    identity["sources_after"] = source_hashes()
    report = {"identity": identity, "cases": records}
    return report, evaluate(report, cases)


class NativeBackend:
    """Real Metal backend; constructed only after admission (never in CPU tests)."""

    def __init__(self):
        import mlx.core as mx

        if mx.default_device() != mx.gpu:
            raise Refused("default device is not the GPU")
        if not mx.metal.is_available():
            raise Refused("Metal is unavailable")
        from mlx2.adapters import qwen38_tree_gdn as T
        from mlx2.runtime.models import gated_delta as G  # builds Metal kernels: only now

        if getattr(G, "_gated_delta_kernel", None) is None:
            raise Refused("ordinary unpacked M1 kernel unavailable")
        self.mx, self.T, self.G = mx, T, G

    def identity(self):
        from scripts.paired_direct_ab import mlx_identity

        mx = self.mx
        return {"default_device": "gpu" if mx.default_device() == mx.gpu else str(mx.default_device()),
                "device_info": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                                for k, v in dict(mx.device_info()).items()},
                "mlx": mlx_identity(mx),
                "reference": "gated_delta._gated_delta_kernel_impl(allow_packed=False)",
                "environment": {k: os.environ.get(k) for k in ("MLX_GDN_PACKED", "MLX_GDN_CORE",
                                                               "MLX_ENABLE_TF32", "MLX2_INTAKE_SOURCE_ROOT")},
                "modules": {"candidate": self.T.__file__, "reference": self.G.__file__,
                            "identity_helper": sys.modules["scripts.paired_direct_ab"].__file__}}

    def arrays(self, case, host):
        mx = self.mx
        dtype = mx.bfloat16 if case["dtype"] == "bfloat16" else mx.float32
        out = {k: mx.array(host[k]).astype(dtype) for k in ("q", "k", "v")}
        out.update({k: mx.array(host[k]) for k in ("g", "beta", "state")})
        mx.eval(list(out.values()))
        return out

    def _evidence(self, x):
        import numpy as np

        mx = self.mx
        name = {mx.bfloat16: "bfloat16", mx.float32: "float32"}.get(x.dtype, str(x.dtype))
        view = {"bfloat16": mx.uint16, "float32": mx.uint32}.get(name)
        bits = np.array(x.view(view)) if view is not None else None
        return {"dtype": name, "shape": list(x.shape), "bits": bits}

    def confirmed(self):
        return self.T.status()["device_engagement_confirmed"]

    def forward(self, a, parents):
        y, flag = self.T.metal_sparse_tree_forward(a["q"], a["k"], a["v"], a["g"], a["beta"], a["state"],
                                                   list(parents), allow_unqualified_metal=True)
        self.mx.eval(y, flag)
        return self._evidence(y), self.T.confirm_device_engagement(flag)

    def row(self, y, node):
        return {"dtype": y["dtype"], "shape": [1, 1, *y["shape"][2:]],
                "bits": None if y["bits"] is None else y["bits"][:, node:node + 1].copy()}

    def reference(self, a, path):
        state, out = a["state"], None
        for row in path:
            out, state = self.G._gated_delta_kernel_impl(
                a["q"][:, row:row + 1], a["k"][:, row:row + 1], a["v"][:, row:row + 1],
                a["g"][:, row:row + 1], a["beta"][:, row:row + 1], state, None, allow_packed=False)
            self.mx.eval(out, state)
        return self._evidence(out), self._evidence(state)

    def replay(self, a, parents, path):
        state, flag = self.T.metal_replay_accepted_path(a["k"], a["v"], a["g"], a["beta"], a["state"],
                                                        list(parents), list(path), allow_unqualified_metal=True)
        self.mx.eval(state, flag)
        return self._evidence(state), self.T.confirm_device_engagement(flag)

    def release(self, arrays):
        arrays.clear()
        self.mx.clear_cache()


def summarize(report):
    """JSON-safe receipt: hashes and counts, never raw bits."""
    def clean(value):
        if isinstance(value, dict):
            if "bits" in value:
                bits = value["bits"]
                return {"dtype": value.get("dtype"), "shape": value.get("shape"),
                        "sha256": None if bits is None else hashlib.sha256(bits.tobytes()).hexdigest()}
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return clean(report)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--topology", nargs="+", choices=list(TOPOLOGIES))
    ap.add_argument("--dtype", nargs="+", choices=list(DTYPES))
    ap.add_argument("--geometry", nargs="+", choices=list(GEOMETRIES))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time-limit-s", type=float, default=3600.0)
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None, environ=None, backend_factory=NativeBackend):
    args = build_parser().parse_args(argv)
    environ = os.environ if environ is None else environ
    record = {"schema": SCHEMA, "scope": SCOPE, "upstream": UPSTREAM, "started": time.time()}
    try:
        if not (math.isfinite(args.time_limit_s) and 0 < args.time_limit_s <= MAX_TIME_S):
            raise Refused(f"--time-limit-s must be finite in (0, {MAX_TIME_S:g}]")
        cases = catalogue(args.topology, args.dtype, args.geometry)
        commit = admit(args, environ)                      # before any MLX import
        sources_before, git = source_hashes(), git_identity()   # before the backend imports source
        refused = source_refusals(commit, sources_before, git)
        if refused:                                         # pre-admission: no backend is built
            raise Refused("; ".join(refused))
        report, verdict = run_gate(args, backend_factory(), cases, commit=commit,
                                   sources_before=sources_before, git=git)
        record.update(verdict, report=summarize(report), cases=[c["case"] for c in cases],
                      requires="the parent wrapper's ownership receipt (canonical CPG lease + both flocks)")
    except Refused as error:
        record.update(verdict="refused", refusals=[str(error)], differences=[])
    record["finished"] = time.time()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(json.dumps({k: record.get(k) for k in ("verdict", "refusals", "differences")}, indent=1)[:4000])
    return 0 if record["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
