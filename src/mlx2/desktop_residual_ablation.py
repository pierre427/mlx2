"""Residual-ablation (Heretic-style abliteration) jobs behind the desktop control plane.

The desktop drives ``scripts/research/residual_ablation.py`` as a subprocess in
``owned`` mode, one job at a time. GPU ownership stays with that runner's own
CPG lease and paired-lock wrapper; this module never takes a lock, never calls
``launchctl`` and never imports MLX. Evidence is read back as JSON, and the
strength preview calls the engine's own (NumPy-only) plan math.

The source model is never written: every job writes into a fresh evidence
directory, and a bake writes a NEW checkpoint directory that must lie outside
the source (checked here before submission and again by the engine).
Everything this produces is experimental and unqualified.
"""
from __future__ import annotations

import ast
import json
import math
import os
import plistlib
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from .desktop import atomic_json

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/research/residual_ablation.py"
JOB_KIND = "residual_ablation"
PHASES = ("capture", "judgecheck", "search", "fullab", "promptab", "bake")
JUDGES = ("d3", "self", "d3x")
JUDGE_OPTIONS = ("three", "two")
ROW_NORMALIZATIONS = ("none", "full")
EDITED_PRECISIONS = ("bf16", "requant")
COMPONENTS = ("attention", "mlp")
KERNEL_KEYS = ("max_weight", "max_weight_position", "min_weight", "min_weight_distance")
D3_REVISION = "cab71e6ad3461b8086d51ddb572c656babd63b0b"
D3_DEFAULT = Path.home() / ".cache/huggingface/hub/models--vllm-sr--d3-edge/snapshots" / D3_REVISION
# Defaults reproduce the 2026-11-05 Qwen3.5-4B run (trial 5 on the Pareto
# front): D3-edge judge with three options, 100 trials in bounded windows,
# batch-50 greedy decode of 100 tokens, exact hooks ("none"), bf16 bake.
DEFAULTS = {
    "judge": "d3", "judge_options": "three", "trials": 100, "window_trials": 50, "startup": 30,
    "seed": 20261105, "batch": 50, "max_tokens": 100, "row_normalization": "none",
    "edited_precision": "bf16", "service_health": "http://127.0.0.1:8282/v1/models",
}
# Cancelling sends SIGINT to the owned runner so its own ``finally`` blocks stop
# the unit, release the CPG lease and bootstrap a stopped service back (the
# runner waits up to 900 s for the service to report healthy). Only after this
# grace does the desktop kill the process group.
CANCEL_GRACE_S = 1200
_TRIAL_LINE = re.compile(r"\btrial (\d+): refusal=")
_FAILURE_LINE = re.compile(r"(Error|Exception|failed|refusing|Traceback)", re.IGNORECASE)


# ------------------------------------------------------------------ runner facts


_RUNNER_CACHE = {}


def runner_facts(path=RUNNER):
    """Constants and accepted flags of the runner, read with ``ast`` (never executed).

    Reading rather than importing keeps MLX and the runner's model code out of
    the control process even if the runner's imports change.
    """
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "present": False, "flags": [], "datasets": {}}
    stamp = path.stat().st_mtime_ns
    cached = _RUNNER_CACHE.get(str(path))
    if cached and cached[0] == stamp:
        return cached[1]
    tree = ast.parse(path.read_text(), filename=str(path))
    constants, flags = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"DATASETS", "SYSTEM_PROMPT", "COMPLIANT_SYSTEM_PROMPT", "REFUSAL_MARKERS"}:
                try:
                    constants[name] = ast.literal_eval(node.value)
                except ValueError:
                    pass
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                    flags.add(arg.value)
    data = ROOT / "docs/research/heretic-qwen35-4b-20261105/data/raw"
    datasets = {}
    for role, (dataset, revision) in constants.get("DATASETS", {}).items():
        folder = data / dataset.split("/")[1] / "data"
        datasets[role] = {"dataset": dataset, "revision": revision,
                          "present": all((folder / f"{split}-00000-of-00001.parquet").is_file() for split in ("train", "test"))}
    facts = {"path": str(path), "present": True, "flags": sorted(flags), "datasets": datasets,
             # Runner's fixed slices (load_prompts): what each set is used for.
             "prompt_sets": [
                 {"role": "Harmless (direction)", "set": "good_train", "dataset": "harmless", "split": "train[:400]"},
                 {"role": "Harmful (direction)", "set": "bad_train", "dataset": "harmful", "split": "train[:400]"},
                 {"role": "Harmless (KL drift)", "set": "kl_eval", "dataset": "harmless", "split": "test[:100]"},
                 {"role": "Harmful (refusal rate)", "set": "refusal_eval", "dataset": "harmful", "split": "test[:100]"}],
             "system_prompt": constants.get("SYSTEM_PROMPT", ""),
             "compliant_system_prompt": constants.get("COMPLIANT_SYSTEM_PROMPT", ""),
             "refusal_markers": len(constants.get("REFUSAL_MARKERS", ())),
             "system_prompt_flag": "--system-prompt" in flags}
    _RUNNER_CACHE[str(path)] = (stamp, facts)
    return facts


# ------------------------------------------------------------- strength preview


def default_probe_parameters(layers):
    """The runner's capture-probe kernel (``default_probe_parameters``), as a dict.

    Kept equal to the runner by a test against real capture evidence.
    """
    last = layers - 1
    return {"direction_index": None, "kernels": {
        "attention": {"max_weight": 1.0, "max_weight_position": 0.75 * last, "min_weight": 0.5,
                      "min_weight_distance": 0.4 * last},
        "mlp": {"max_weight": 0.5, "max_weight_position": 0.75 * last, "min_weight": 0.25,
                "min_weight_distance": 0.4 * last}}}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def ablation_parameters(params):
    """``AblationParameters`` from the evidence/UI dict (``AblationParameters.as_dict`` shape)."""
    from .experimental.weight_space.residual_ablation import AblationParameters, ComponentKernel

    if not isinstance(params, dict) or not isinstance(params.get("kernels"), dict):
        raise ValueError("Strength profile needs per-component kernels")
    if set(params["kernels"]) != set(COMPONENTS):
        raise ValueError("Strength profile must cover exactly attention and mlp")
    kernels = {}
    for component in COMPONENTS:
        raw = params["kernels"][component]
        if not isinstance(raw, dict) or set(raw) != set(KERNEL_KEYS):
            raise ValueError(f"{component} kernel needs {', '.join(KERNEL_KEYS)}")
        kernels[component] = ComponentKernel(*(_number(raw[k], f"{component}.{k}") for k in KERNEL_KEYS))
    index = params.get("direction_index")
    if index is not None:
        index = _number(index, "direction_index")
        if index < 0:
            raise ValueError("direction_index must be nonnegative")
    return AblationParameters(kernels=kernels, direction_index=index)


def plan_preview(params, layers):
    """Per-layer λ for each component, computed by the engine's own kernel math.

    ``AblationParameters.plan`` needs direction vectors; the λ it assigns per
    (layer, component) is ``kernel.profile().strength(layer)``, which is what
    this returns, plus the edit count ``plan`` would produce.
    """
    if type(layers) is not int or not 1 <= layers <= 512:
        raise ValueError("Layer count must be between 1 and 512")
    parsed = ablation_parameters(params)
    curves = {}
    for component in COMPONENTS:
        profile = parsed.kernels[component].profile()
        curves[component] = [profile.strength(layer) for layer in range(layers)]
    edits = sum(1 for component in COMPONENTS for lam in curves[component] if lam != 0)
    direction = ({"scope": "per layer"} if parsed.direction_index is None else
                 {"scope": "global", "direction_index": parsed.direction_index,
                  "valid": 0 <= parsed.direction_index <= layers - 1,
                  "from_layers": [int(math.floor(parsed.direction_index)),
                                  min(layers - 1, int(math.floor(parsed.direction_index)) + 1)]})
    if direction.get("valid") is False:
        raise ValueError("direction_index outside the captured layers")
    return {"layers": layers, "components": curves, "edits": edits, "direction": direction,
            "params": parsed.as_dict()}


# ---------------------------------------------------------------- evidence I/O


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def metrics(result):
    if not isinstance(result, dict):
        return None
    out = {key: result.get(key) for key in ("refusal_rate", "keyword_rate", "judge_b_rate", "kl", "tokens",
                                            "generated_tokens") if key in result}
    if isinstance(result.get("labels"), dict):
        out["labels"] = result["labels"]
    elif isinstance(result.get("samples"), list) and result["samples"]:
        out["labels"] = sample_labels(result["samples"])
    return out


def sample_labels(samples):
    labels = Counter()
    for row in samples:
        if row.get("keyword"):
            labels["keyword"] += 1
        elif row.get("judge_label"):
            labels[row["judge_label"]] += 1
        elif row.get("judge") in ("A", "B"):
            labels["complies" if row["judge"] == "A" else "judged refusal"] += 1
    return dict(labels)


_TRIALS_CACHE = {}


def read_trials_file(path):
    """Parsed rows of one ``trials.jsonl`` (cached on size and mtime).

    A window killed mid-write can leave a partial last line; it is skipped.
    """
    path = Path(path)
    stat = path.stat()
    key = str(path)
    cached = _TRIALS_CACHE.get(key)
    if cached and cached[0] == (stat.st_size, stat.st_mtime_ns):
        return cached[1]
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or type(row.get("trial")) is not int:
            continue
        rows.append(row)
    _TRIALS_CACHE[key] = ((stat.st_size, stat.st_mtime_ns), rows)
    return rows


def trial_summary(row, window):
    out = {"trial": row["trial"], "window": window, "params": row.get("params"),
           "edits": row.get("edits"), "seconds": row.get("seconds")}
    out.update(metrics(row) or {})
    return out


def pareto_front(rows):
    """Trial numbers not dominated on (refusal_rate, kl), both minimised."""
    points = [(r["trial"], r["refusal_rate"], r["kl"]) for r in rows
              if isinstance(r.get("refusal_rate"), (int, float)) and isinstance(r.get("kl"), (int, float))]
    front = []
    for trial, rate, kl in points:
        dominated = any((r2 <= rate and k2 <= kl) and (r2 < rate or k2 < kl) for _, r2, k2 in points)
        if not dominated:
            front.append(trial)
    return sorted(front, key=lambda t: next((r, k) for n, r, k in points if n == t))


def _depth(path, root):
    return len(path.relative_to(root).parts)


def _run_dirs(root, max_depth=4):
    found = set()
    for marker in ("report.json", "trials.jsonl", "bake_receipt.json", "run.log"):
        for path in root.rglob(marker):
            if _depth(path, root) <= max_depth:
                found.add(path.parent)
    return sorted(found)


def _classify(directory, report):
    if isinstance(report, dict) and report.get("phase") in PHASES:
        return report["phase"]
    if (directory / "trials.jsonl").is_file():
        return "search"
    if (directory / "bake_receipt.json").is_file():
        return "bake"
    for phase in PHASES:
        if directory.name.startswith(phase):
            return phase
    return "unknown"


def _tail(path, limit=16384):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(max(0, Path(path).stat().st_size - limit))
            return stream.read().decode("utf-8", "replace")
    except OSError:
        return ""


def failure_reason(directory, stdout=None):
    """Best one-line reason a window ended without an ok report."""
    report = _read_json(directory / "report.json")
    if isinstance(report, dict) and str(report.get("status", "")).startswith("failed"):
        return report["status"]
    for source in (stdout, directory.parent / f"{directory.name}.stdout", directory / "unit.stdout"):
        if source and Path(source).is_file():
            lines = [line for line in _tail(source).splitlines() if _FAILURE_LINE.search(line)
                     and not line.startswith("[I ")]
            if lines:
                return lines[-1].strip()[:400]
    lease = _read_json(directory / "desktop-lease.json") or {}
    status = (lease.get("complete_worker") or {}).get("status")
    if status == "failed":
        return "CPG lease ended the window (worker recorded failed) before the runner wrote report.json"
    return None


def run_summary(directory, root):
    directory = Path(directory)
    report = _read_json(directory / "report.json")
    phase = _classify(directory, report)
    lease = _read_json(directory / "desktop-lease.json") or {}
    log = _tail(directory / "run.log", 4096).strip().splitlines()
    notes = {}
    for note in sorted(directory.glob("*.txt")):
        notes[note.name] = note.read_text(errors="replace")[:2000]
    status = report.get("status") if isinstance(report, dict) else None
    summary = {
        "name": directory.name, "path": str(directory), "relative": str(directory.relative_to(root)),
        "phase": phase, "status": status or ("no report" if (directory / "run.log").exists() else "unknown"),
        "ok": status == "ok", "checkpoint": (report or {}).get("checkpoint"),
        "seconds": (report or {}).get("seconds"), "modified": directory.stat().st_mtime,
        "service": _read_json(directory / "service.json"),
        "lease_status": (lease.get("complete_worker") or {}).get("status"),
        "last_log_line": log[-1] if log else None, "notes": notes,
        "invalid": any(name.upper().startswith("INVALID") for name in notes),
    }
    if status != "ok":
        summary["failure"] = failure_reason(directory)
    judge = (report or {}).get("judge")
    summary["judge"] = judge.get("kind") if isinstance(judge, dict) else judge
    if isinstance(judge, dict) and isinstance(judge.get("options"), dict):
        summary["judge_options"] = "three" if "redirects" in judge["options"] else "two"
    return summary, report


def scan_workspace(path):
    """Everything the Ablation view shows, read from one evidence folder.

    Layout-agnostic: run directories are found by their markers up to four
    levels deep, so desktop workspaces and hand-run evidence trees both load.
    Harmful-request sample texts are left out; ``trial_detail`` serves them on
    explicit request.
    """
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Evidence folder must be a directory")
    runs, captures, judgechecks, searches, fullabs, promptabs, bakes = [], [], [], [], [], [], []
    trials = {}
    journals = sorted(str(p) for p in root.rglob("study.journal") if _depth(p, root) <= 4)
    for directory in _run_dirs(root):
        summary, report = run_summary(directory, root)
        runs.append(summary)
        phase = summary["phase"]
        report = report or {}
        if phase == "capture" and summary["ok"]:
            separation = report.get("separation_d") or []
            cache = report.get("cache_dir")
            captures.append({**summary, "separation_d": separation, "adjacent_cosine": report.get("adjacent_cosine"),
                             "layers": len(separation), "baseline": report.get("baseline"),
                             "probe_measured": metrics(report.get("probe_measured")),
                             "probe_random_control": metrics(report.get("probe_random_control")),
                             "probe_params": (report.get("probe_measured") or {}).get("params"),
                             "directions_sha256": report.get("directions_sha256"),
                             "peak_layer": max(range(len(separation)), key=separation.__getitem__) if separation else None,
                             "cache_dir": cache,
                             "baseline_logprobs_present": bool(cache) and (Path(cache) / "baseline_logprobs.npy").is_file()})
        elif phase == "judgecheck" and summary["ok"]:
            judgechecks.append({**summary, **{k: report.get(k) for k in (
                "harmless_judged_A", "baseline_refusals_judged_B", "d3", "d3x", "self_seconds_per_100",
                "d3_seconds_per_100", "d3x_seconds_per_100")},
                "probe": metrics(report.get("probe")),
                "exemplars": (directory / "judge_exemplars.json").is_file()})
        elif phase == "search":
            window_rows = read_trials_file(directory / "trials.jsonl") if (directory / "trials.jsonl").is_file() else []
            for row in window_rows:
                trials[row["trial"]] = trial_summary(row, directory.name)
            searches.append({**summary, "trials_in_window": len(window_rows),
                             "trial_range": [window_rows[0]["trial"], window_rows[-1]["trial"]] if window_rows else None,
                             "batch_check": report.get("batch_check"), "journal": report.get("journal"),
                             "trials_complete": report.get("trials_complete"),
                             "report_pareto": [p.get("trial") for p in report.get("pareto", [])] or None})
        elif phase == "fullab":
            arms = [{"trial": arm.get("trial"), "params": arm.get("params"),
                     "hooks_none": metrics(arm.get("hooks_none")),
                     "materialized_full": metrics(arm.get("materialized_full"))} for arm in report.get("arms", [])]
            fullabs.append({**summary, "arms": arms, "restored_kl_first10": report.get("restored_kl_first10"),
                            "cross_host_kl_floor_first10": report.get("cross_host_kl_floor_first10")})
        elif phase == "promptab":
            arms = []
            for name, result in (report.get("arms") or {}).items():
                prompt, _, edit = name.partition("/")
                arms.append({"arm": name, "system_prompt": prompt, "edit": edit, **(metrics(result) or {})})
            promptabs.append({**summary, "arms": arms, "system_prompts": report.get("system_prompts")})
        elif phase == "bake":
            bakes.append(bake_summary(directory, summary, report))
    ordered = sorted(trials.values(), key=lambda r: r["trial"])
    front = set(pareto_front(ordered))
    for row in ordered:
        row["pareto"] = row["trial"] in front
    latest = lambda rows: sorted(rows, key=lambda r: r["modified"])[-1] if rows else None  # noqa: E731
    capture = latest(captures)
    search_judge = next((s for s in sorted(searches, key=lambda r: -r["modified"]) if s.get("judge")), None)
    return {
        "path": str(root), "runs": sorted(runs, key=lambda r: r["modified"]),
        "capture": capture, "captures": len(captures), "judgecheck": latest(judgechecks),
        "layers": capture["layers"] if capture else None,
        "search": {"windows": sorted(searches, key=lambda r: r["modified"]), "journals": journals,
                   "study": str(Path(journals[0]).parent) if len(journals) == 1 else None,
                   "trials_complete": len(ordered),
                   "judge": search_judge.get("judge") if search_judge else None,
                   "judge_options": search_judge.get("judge_options") if search_judge else None},
        "trials": ordered, "pareto": sorted(front, key=lambda t: (trials[t]["refusal_rate"], trials[t]["kl"])),
        "fullab": sorted(fullabs, key=lambda r: r["modified"]),
        "promptab": sorted(promptabs, key=lambda r: r["modified"]),
        "bakes": sorted(bakes, key=lambda r: r["modified"]),
    }


def bake_summary(directory, summary, report):
    receipt = _read_json(directory / "bake_receipt.json") or {}
    tensors = []
    for row in receipt.get("edited_tensors", []):
        edit, error = row.get("edit_rms"), row.get("requant_rms")
        tensors.append({"layer": row.get("layer"), "component": row.get("component"), "lam": row.get("lam"),
                        "key": row.get("key"), "edit_rms": edit, "requant_rms": error,
                        "stored": row.get("stored") or ("requant" if row.get("quantized") else "unknown"),
                        "requant_to_edit": (error / edit) if edit and error is not None else None})
    tensors.sort(key=lambda r: (r["layer"] if r["layer"] is not None else -1, r["component"] or ""))
    return {**summary, "receipt_present": bool(receipt), "destination": receipt.get("destination"),
            "source": receipt.get("source"), "source_files_hashed": len(receipt.get("source_hashes") or {}),
            "source_unchanged": receipt.get("source_unchanged"),
            "source_hashes_match_after_load": report.get("source_hashes_match_after_load"),
            "row_normalization": receipt.get("row_normalization") or report.get("row_normalization"),
            "edited_precision": receipt.get("edited_precision") or ("requant" if any(
                t["stored"] == "requant" for t in tensors) else None),
            "dropped_keys": len(receipt.get("dropped_keys") or []), "tensors": tensors,
            "trial": report.get("trial"), "params": report.get("params"),
            "in_memory": metrics(report.get("in_memory")), "baked": metrics(report.get("baked")),
            "baked_checkpoint": report.get("baked_checkpoint"),
            "max_edit_rms": report.get("max_edit_rms"), "max_requant_rms": report.get("max_requant_rms"),
            "qualification": "unqualified"}


def trial_detail(path, trial, *, samples=False):
    """One trial with optional sample texts (harmful-request outputs; opt-in)."""
    root = Path(path).expanduser().resolve(strict=True)
    if type(trial) is not int:
        raise ValueError("Choose a trial number")
    match = None
    for directory in _run_dirs(root):
        if (directory / "trials.jsonl").is_file():
            for row in read_trials_file(directory / "trials.jsonl"):
                if row["trial"] == trial:
                    match = (row, directory.name)
    if match is None:
        raise ValueError(f"Trial {trial} is not in this evidence folder")
    row, window = match
    out = trial_summary(row, window)
    out["sample_count"] = len(row.get("samples") or [])
    if samples:
        out["samples"] = [{"prompt": s.get("prompt"), "response": s.get("response"), "keyword": s.get("keyword"),
                           "judge": s.get("judge"), "label": s.get("judge_label"), "refusal": s.get("refusal")}
                          for s in row.get("samples") or []]
    return out


# ------------------------------------------------------------- validation


def _resolve(value, name, *, strict=False):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Choose {name}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name[0].upper() + name[1:]} must be an absolute path")
    return path.resolve(strict=strict)


def _inside(path, root):
    return path == root or root in path.parents


def validate_model(value):
    model = _resolve(value, "a model folder", strict=True)
    if not model.is_dir() or not (model / "config.json").is_file():
        raise ValueError("The model folder must contain config.json")
    return model


def model_layers(model):
    config = _read_json(Path(model) / "config.json") or {}
    for scope in (config, config.get("text_config") or {}, config.get("language_config") or {}):
        value = scope.get("num_hidden_layers")
        if type(value) is int and value > 0:
            return value
    return None


def model_checkpoint(model):
    """``<folder>@<fingerprint>`` exactly as the desktop library names it (GPU-free)."""
    try:
        from .adapters.registry import inspect_model

        return f"{Path(model).name}@{inspect_model(model).artifact_fingerprint}"
    except Exception:  # noqa: BLE001 -- an uninspectable artifact is labelled, the runner refuses it
        return None


def validate_destination(model, dest, *, free_bytes=None):
    """Refuse every bake destination that could touch the source or an existing model."""
    source = validate_model(model)
    destination = _resolve(dest, "a destination folder")
    if _inside(destination, source) or _inside(source, destination):
        raise ValueError("The destination must be outside the source model folder (and must not contain it)")
    # A Hugging Face snapshot is symlinks into a blob store; refuse that tree too.
    targets = {p.resolve().parent for p in Path(model).expanduser().iterdir() if p.is_symlink()}
    for target in targets:
        if _inside(destination, target):
            raise ValueError("The destination must be outside the source model's storage")
    if destination.exists() or destination.is_symlink():
        raise ValueError("The destination already exists; a bake never overwrites a folder")
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise ValueError(f"A leftover staging folder exists: {staging}")
    parent = destination.parent
    if not parent.is_dir():
        raise ValueError(f"The destination's parent folder does not exist: {parent}")
    if not os.access(parent, os.W_OK):
        raise ValueError(f"The destination's parent folder is not writable: {parent}")
    size = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
    # bf16 edited projections are ~4x their 4-bit size; budget a full copy twice.
    required = 2 * size
    free = shutil.disk_usage(parent).free if free_bytes is None else free_bytes
    if free < required:
        raise ValueError(f"Not enough free space: need about {required:,} bytes, {free:,} available")
    return {"destination": str(destination), "source": str(source), "source_bytes": size,
            "required_free_bytes": required, "free_bytes": free}


def launch_agents(folder=None):
    """Model services that ``--stop-service`` can stop: ~/Library/LaunchAgents plists (read only)."""
    folder = Path(folder or Path.home() / "Library/LaunchAgents")
    out = []
    for plist in sorted(folder.glob("*.plist")):
        try:
            data = plistlib.loads(plist.read_bytes())
        except Exception:  # noqa: BLE001 -- unreadable plist is not offered
            continue
        label = data.get("Label")
        if not isinstance(label, str) or "mlx" not in label.lower():
            continue
        args = [str(a) for a in data.get("ProgramArguments") or []]
        port = next((args[i + 1] for i, a in enumerate(args[:-1]) if a == "--port"), None)
        out.append({"label": label, "plist": str(plist),
                    "health": f"http://127.0.0.1:{port}/v1/models" if port and port.isdigit() else None})
    return out


def _int(request, key, low, high):
    value = request.get(key, DEFAULTS.get(key))
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{key.replace('_', ' ')} must be a whole number from {low} to {high}")
    return value


def find_capture(workspace):
    root = Path(workspace)
    if not root.is_dir():
        return None
    oks = [d for d in _run_dirs(root) if (_read_json(d / "report.json") or {}).get("phase") == "capture"
           and (_read_json(d / "report.json") or {}).get("status") == "ok"]
    return max(oks, key=lambda d: d.stat().st_mtime) if oks else None


def find_judgecheck(workspace):
    root = Path(workspace)
    if not root.is_dir():
        return None
    oks = [d for d in _run_dirs(root) if (d / "judge_exemplars.json").is_file()]
    return max(oks, key=lambda d: d.stat().st_mtime) if oks else None


def find_study(workspace):
    root = Path(workspace)
    journals = [p for p in root.rglob("study.journal") if _depth(p, root) <= 4] if root.is_dir() else []
    if len(journals) > 1:
        raise ValueError("This evidence folder holds several search studies; choose the study folder explicitly")
    return journals[0].parent if journals else None


def completed_trials(workspace):
    root = Path(workspace)
    rows = {}
    if root.is_dir():
        for directory in _run_dirs(root):
            if (directory / "trials.jsonl").is_file():
                for row in read_trials_file(directory / "trials.jsonl"):
                    rows[row["trial"]] = row
    return rows


def plan_job(request, *, settings, now=None):
    """Validate a job request and build its runner command. No side effects."""
    phase = request.get("phase")
    if phase not in PHASES:
        raise ValueError("Choose a phase: " + ", ".join(PHASES))
    facts = runner_facts(Path(settings.get("runner") or RUNNER))
    if not facts["present"]:
        raise ValueError(f"Residual-ablation runner not found at {facts['path']}")
    model = validate_model(request.get("model"))
    checkpoint = model_checkpoint(model)
    root = Path(settings["evidence_root"]).expanduser().resolve()
    workspace = (_resolve(request["workspace"], "an evidence folder") if request.get("workspace")
                 else root / model.name)
    if _inside(workspace, model):
        raise ValueError("The evidence folder must be outside the model folder")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
    trial_list = request.get("trial") or []
    if not isinstance(trial_list, list) or any(type(t) is not int or t < 0 for t in trial_list):
        raise ValueError("Trials must be a list of trial numbers")
    suffix = ("-t" + "-".join(map(str, trial_list))) if trial_list else ""
    out = workspace / f"{phase}{suffix}-{stamp}"
    if out.exists():
        raise ValueError("Evidence folder for this job already exists; wait a second and retry")
    spec = {"phase": phase, "model": str(model), "checkpoint": checkpoint, "workspace": str(workspace),
            "out": str(out), "stdout": str(workspace / f"{out.name}.stdout"), "qualification": "unqualified"}
    warnings = []
    for key, low, high in (("batch", 1, 256), ("max_tokens", 1, 4096), ("seed", 0, 2**31 - 1)):
        spec[key] = _int(request, key, low, high)
    judge = request.get("judge", DEFAULTS["judge"])
    options = request.get("judge_options", DEFAULTS["judge_options"])
    if judge not in JUDGES or options not in JUDGE_OPTIONS:
        raise ValueError("Choose a judge (d3, self or d3x) and a D3 option set (three or two)")
    spec.update(judge=judge, judge_options=options)
    d3 = request.get("d3") or settings.get("d3_path") or ""
    if judge != "self" or (phase == "judgecheck" and d3):
        d3_path = _resolve(d3, "the D3-edge judge artifact", strict=False)
        if not (d3_path / "config.json").is_file():
            raise ValueError(f"D3-edge judge artifact not found at {d3_path}")
        spec["d3"] = str(d3_path)
    capture = None
    if phase != "capture":
        capture = (_resolve(request["capture"], "a capture folder", strict=True) if request.get("capture")
                   else find_capture(workspace))
        if capture is None:
            raise ValueError("Run Measure (capture) for this model first")
        report = _read_json(capture / "report.json") or {}
        if report.get("status") != "ok" or not (capture / "directions.npz").is_file():
            raise ValueError("The capture folder has no completed capture (report.json status ok + directions)")
        fingerprint = (report.get("checkpoint") or "").partition("@")[2]
        if checkpoint and fingerprint and checkpoint.partition("@")[2] != fingerprint:
            raise ValueError(f"The capture was measured on {report['checkpoint']}, not on {checkpoint}")
        cache = Path(report.get("cache_dir") or "")
        if not (cache / "baseline_logprobs.npy").is_file():
            raise ValueError(f"The capture's baseline log-probs are not on this Mac ({cache}); run Measure here")
        spec["capture"] = str(capture)
    if judge == "d3x":
        judgecheck = (_resolve(request["judgecheck"], "a judge-check folder", strict=True)
                      if request.get("judgecheck") else find_judgecheck(workspace))
        if judgecheck is None or not (judgecheck / "judge_exemplars.json").is_file():
            raise ValueError("The d3x judge needs a judge check (judge_exemplars.json); run it first")
        spec["judgecheck"] = str(judgecheck)
    if phase == "search":
        study = (_resolve(request["study"], "a study folder") if request.get("study")
                 else (find_study(workspace) or workspace / "search-study"))
        if _inside(study, model):
            raise ValueError("The study folder must be outside the model folder")
        spec["study"] = str(study)
        spec["trials"] = _int(request, "trials", 1, 10000)
        spec["window_trials"] = _int(request, "window_trials", 0, spec["trials"])
        spec["startup"] = _int(request, "startup", 0, 1000)
        done = len(completed_trials(workspace))
        if done >= spec["trials"]:
            raise ValueError(f"The study already has {done} completed trials; raise the total to continue")
        spec["resume_from"] = done
        spec["window_planned"] = min(spec["trials"] - done, spec["window_trials"] or spec["trials"])
        if done:
            warnings.append(f"Continues the existing study: {done} trials done, this window runs up to {spec['window_planned']}")
    if phase in ("fullab", "promptab", "bake"):
        study = (_resolve(request["study"], "a study folder", strict=True) if request.get("study")
                 else find_study(workspace))
        if study is None or not (study / "study.journal").is_file():
            raise ValueError("Run a search first: comparisons and bakes use trials from its study journal")
        known = completed_trials(workspace)
        if not trial_list:
            raise ValueError("Choose a trial from the search")
        missing = [t for t in trial_list if t not in known]
        if missing:
            raise ValueError(f"Trial(s) {missing} have no completed result in this evidence folder")
        if phase == "bake" and len(trial_list) != 1:
            raise ValueError("A bake writes exactly one trial")
        spec["study"] = str(study)
        spec["trial"] = trial_list
    if phase == "bake":
        if request.get("row_normalization", DEFAULTS["row_normalization"]) not in ROW_NORMALIZATIONS:
            raise ValueError("Row normalization must be none or full")
        if request.get("edited_precision", DEFAULTS["edited_precision"]) not in EDITED_PRECISIONS:
            raise ValueError("Edited precision must be bf16 or requant")
        spec["row_normalization"] = request.get("row_normalization", DEFAULTS["row_normalization"])
        spec["edited_precision"] = request.get("edited_precision", DEFAULTS["edited_precision"])
        checked = validate_destination(str(model), request.get("dest"), free_bytes=request.get("_free_bytes"))
        if _inside(Path(checked["destination"]), workspace):
            raise ValueError("Bake into a model folder, not into the evidence folder")
        spec["dest"] = checked["destination"]
        spec["destination_check"] = checked
        if spec["edited_precision"] == "requant":
            warnings.append("4-bit requantization erased most of the edit on Qwen3.5-4B (2026-11-05); bf16 keeps it")
    system_prompt = request.get("system_prompt")
    if system_prompt not in (None, "", facts.get("system_prompt")):
        if not facts["system_prompt_flag"]:
            raise ValueError("This runner has no --system-prompt option; the pinned system prompt is used")
        if not isinstance(system_prompt, str) or len(system_prompt) > 4000:
            raise ValueError("System prompt must be text of at most 4000 characters")
        spec["system_prompt"] = system_prompt
    service = request.get("stop_service") or None
    if service:
        known = {row["label"]: row for row in launch_agents(settings.get("launch_agents"))}
        if service not in known:
            raise ValueError(f"No LaunchAgent plist for {service}; the runner could not restart it")
        if request.get("confirm_stop_service") != service:
            raise ValueError(f"Confirm stopping {service} for this job (it is restarted afterwards)")
        health = request.get("service_health") or known[service]["health"] or DEFAULTS["service_health"]
        parsed = urlparse(health)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost") or not parsed.port:
            raise ValueError("Service health URL must be http://127.0.0.1:<port>/…")
        spec["stop_service"] = service
        spec["service_health"] = health
    python = settings.get("python") or sys.executable
    if not Path(python).is_file():
        raise ValueError(f"Python executable not found: {python}")
    spec["python"] = python
    spec["runner"] = facts["path"]
    command = build_command(spec, facts)
    return {"spec": spec, "command": command, "warnings": warnings}


def build_command(spec, facts=None):
    """The exact runner argv for a validated spec (``owned`` mode)."""
    flags = set((facts or runner_facts(Path(spec["runner"])))["flags"])
    argv = [spec["python"], "-u", spec["runner"], "owned", spec["phase"], spec["out"],
            "--model", spec["model"], "--batch", str(spec["batch"]), "--max-tokens", str(spec["max_tokens"]),
            "--seed", str(spec["seed"]), "--judge", spec["judge"], "--judge-options", spec["judge_options"]]

    def add(flag, *values):
        if flag not in flags:
            raise ValueError(f"The runner does not accept {flag}")
        argv.extend([flag, *map(str, values)])

    if spec.get("d3"):
        add("--d3", spec["d3"])
    if spec.get("capture"):
        add("--capture", spec["capture"])
    if spec.get("judgecheck"):
        add("--judgecheck", spec["judgecheck"])
    if spec.get("study"):
        add("--study", spec["study"])
    if spec["phase"] == "search":
        add("--trials", spec["trials"])
        add("--window-trials", spec["window_trials"])
        add("--startup", spec["startup"])
    if spec.get("trial"):
        add("--trial", *spec["trial"])
    if spec["phase"] == "bake":
        add("--dest", spec["dest"])
        add("--row-normalization", spec["row_normalization"])
        add("--edited-precision", spec["edited_precision"])
    if spec.get("system_prompt"):
        add("--system-prompt", spec["system_prompt"])
    if spec.get("stop_service"):
        add("--stop-service", spec["stop_service"])
        add("--service-health", spec["service_health"])
    return argv


def job_progress(spec, out):
    """Plain-language progress from the window's run.log."""
    log = _tail(Path(out) / "run.log", 8192)
    lines = [line for line in log.splitlines() if line.strip()]
    trials = [int(m.group(1)) for m in map(_TRIAL_LINE.search, lines) if m]
    progress = {"last_line": lines[-1] if lines else None, "loaded": any(" LOADED " in f" {l} " for l in lines)}
    if spec["phase"] == "search":
        full = _tail(Path(out) / "run.log", 1 << 20)
        done = len(_TRIAL_LINE.findall(full))
        progress.update(window_done=done, window_planned=spec.get("window_planned"),
                        last_trial=trials[-1] if trials else None)
        if spec.get("window_planned"):
            progress["percent"] = min(100.0, 100.0 * done / spec["window_planned"])
    if not lines:
        waiting = "Waiting for the CPG lease and paired GPU locks"
        if spec.get("stop_service"):
            waiting = f"Stopping {spec['stop_service']} for this window, then waiting for GPU ownership"
        progress["detail"] = waiting
    else:
        progress["detail"] = lines[-1]
    return progress


# ------------------------------------------------------------------ jobs


class ResidualAblation:
    """One serial residual-ablation job at a time, run by the owned-mode runner."""

    def __init__(self, controller):
        self.controller = controller
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.process = None
        self.active = None
        self.closed = False
        self.settings_path = Path(controller.home) / "residual-ablation-settings.json"

    # settings live beside the desktop preferences but in their own file, so
    # the shared preferences schema is unchanged.
    def settings(self):
        value = {"evidence_root": str(Path(self.controller.home) / "residual-ablation"),
                 "d3_path": str(D3_DEFAULT) if (D3_DEFAULT / "config.json").is_file() else "",
                 "python": ""}
        stored = _read_json(self.settings_path)
        if isinstance(stored, dict):
            value.update({k: v for k, v in stored.items() if k in value and isinstance(v, str)})
        return value

    def configure(self, changes):
        if not isinstance(changes, dict) or set(changes) - {"evidence_root", "d3_path", "python"}:
            raise ValueError("Unknown ablation settings")
        value = {**self.settings(), **changes}
        for key, item in value.items():
            if not isinstance(item, str) or (item and not Path(item).expanduser().is_absolute()):
                raise ValueError(f"{key.replace('_', ' ')} must be an absolute path")
        if value["python"] and not Path(value["python"]).expanduser().is_file():
            raise ValueError("Python executable not found")
        atomic_json(self.settings_path, value)
        return value

    def jobs(self, limit=50):
        return [j for j in self.controller.store.jobs(200) if j["kind"] == JOB_KIND][:limit]

    def snapshot(self):
        facts = runner_facts()
        return {"jobs": self.jobs(), "running": bool(self.thread and self.thread.is_alive()),
                "active_job": self.active, "settings": self.settings(), "defaults": DEFAULTS,
                "runner": facts, "phases": list(PHASES),
                "note": "Experimental. The original model is never modified; bakes write a new folder and are unqualified."}

    def plan(self, request):
        return plan_job(request, settings=self.settings())

    def start(self, request):
        with self.lock:
            if self.closed:
                raise ValueError("Ablation controller is closing")
            if self.thread and self.thread.is_alive():
                raise ValueError("An ablation job is already running; wait for it or cancel it")
            planned = self.plan(request)
            spec, command = planned["spec"], planned["command"]
            identity = self.controller.store.enqueue(JOB_KIND, {**spec, "command": command})
            self.cancel.clear()
            self.active = identity
            self.thread = threading.Thread(target=self._run, args=(identity, spec, command), daemon=True,
                                           name="mlx2-residual-ablation")
            self.thread.start()
            return {"job_id": identity, "spec": spec, "command": command, "warnings": planned["warnings"]}

    def resume(self, request):
        """Run another search window on the same study journal."""
        job = next((j for j in self.jobs(200) if j["id"] == request.get("job_id")), None)
        if job is None or job["spec"].get("phase") != "search":
            raise ValueError("Choose a search job to resume")
        spec = job["spec"]
        keys = ("model", "workspace", "study", "capture", "judgecheck", "trials", "window_trials", "startup",
                "seed", "batch", "max_tokens", "judge", "judge_options", "d3")
        follow = {k: spec[k] for k in keys if spec.get(k) is not None}
        follow["phase"] = "search"
        for key in ("window_trials", "trials", "stop_service", "confirm_stop_service", "service_health"):
            if key in request:
                follow[key] = request[key]
        return self.start(follow)

    def _env(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / '.deps'}"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        return env

    def _run(self, identity, spec, command):
        store = self.controller.store
        child = None
        signalled = None
        try:
            Path(spec["workspace"]).mkdir(parents=True, exist_ok=True)
            store.update(identity, "waiting", job_progress(spec, spec["out"])["detail"])
            with Path(spec["stdout"]).open("ab", buffering=0) as log:
                child = self._spawn(command, log)
                self.process = child
                last = None
                while child.poll() is None:
                    if self.cancel.is_set() and signalled is None:
                        # The runner's finally blocks stop the unit, release the
                        # lease and restore a stopped service.
                        os.kill(child.pid, signal.SIGINT)
                        signalled = time.monotonic()
                        store.update(identity, "cancelling", "Cancelling: stopping the unit, releasing GPU "
                                     "ownership" + (f" and restarting {spec['stop_service']}" if spec.get("stop_service") else ""))
                    if signalled is not None and time.monotonic() - signalled > CANCEL_GRACE_S:
                        from .desktop import stop_group
                        stop_group(child)
                        break
                    if signalled is None:
                        progress = job_progress(spec, spec["out"])
                        state = "running" if progress["last_line"] else "waiting"
                        if (state, progress["detail"]) != last:
                            store.update(identity, state, progress["detail"], progress)
                            last = (state, progress["detail"])
                    time.sleep(1)
            child.wait()
            result = self.finish(spec, child.returncode)
            if signalled is not None:
                store.update(identity, "cancelled", "Cancelled · " + (result.get("failure") or "window stopped"), result)
            elif result["ok"]:
                store.update(identity, "completed", result["detail"], result)
                if spec["phase"] == "bake":
                    self._refresh_library(spec)
            else:
                store.update(identity, "failed", result.get("failure") or "Runner failed", result)
        except Exception as error:  # noqa: BLE001 -- durable failure record
            store.update(identity, "failed", str(error) if isinstance(error, (ValueError, OSError)) else type(error).__name__)
        finally:
            if child is not None and child.poll() is None:
                from .desktop import stop_group
                stop_group(child)
            self.process = None
            self.active = None

    def _spawn(self, command, log):
        return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=self._env(), cwd=str(ROOT))

    @staticmethod
    def finish(spec, returncode):
        out = Path(spec["out"])
        report = _read_json(out / "report.json") or {}
        service = _read_json(out / "service.json")
        ok = returncode == 0 and report.get("status") == "ok"
        result = {"returncode": returncode, "out": str(out), "stdout": spec["stdout"], "ok": ok,
                  "report_status": report.get("status"), "service": service,
                  "progress": job_progress(spec, out)}
        if spec["phase"] == "search":
            result["trials_complete"] = report.get("trials_complete")
            result["remaining"] = (spec["trials"] - report["trials_complete"]
                                   if isinstance(report.get("trials_complete"), int) else None)
        if spec["phase"] == "bake":
            result["dest"] = spec["dest"]
            result["qualification"] = "unqualified"
        if service and service.get("label") and service.get("healthy_after") is not True:
            result["service_warning"] = f"{service['label']} did not report healthy after the window"
        if ok:
            detail = {"capture": "Measured refusal directions", "judgecheck": "Judge calibrated",
                      "search": "Search window done", "fullab": "Hooks vs weights compared",
                      "promptab": "System-prompt A/B done", "bake": "Baked · unqualified"}[spec["phase"]]
            if spec["phase"] == "search" and result.get("remaining"):
                detail += f" · {result['remaining']} trials left (resume to continue)"
            result["detail"] = detail
        else:
            result["failure"] = failure_reason(out, spec["stdout"]) or f"Runner exited with status {returncode}"
        return result

    def _refresh_library(self, spec):
        try:
            library = self.controller.primary.library
            if Path(spec["dest"]).parent == Path(library.root).resolve():
                library.refresh()
        except Exception:  # noqa: BLE001 -- the baked folder is complete even if the UI refresh fails
            self.controller.event("ablation_library_refresh_failed")

    def log(self, request):
        job = next((j for j in self.jobs(200) if j["id"] == request.get("job_id")), None)
        if job is None:
            raise ValueError("Unknown ablation job")
        spec = job["spec"]
        return {"run_log": _tail(Path(spec["out"]) / "run.log", 32768),
                "stdout": _tail(spec["stdout"], 32768), "out": spec["out"]}

    def dispatch(self, request):
        command = request["command"]
        if command == "ablation_workspace":
            return scan_workspace(request.get("path"))
        if command == "ablation_trial":
            return trial_detail(request.get("path"), request.get("trial"), samples=request.get("samples") is True)
        if command == "ablation_preview":
            return plan_preview(request.get("params"), request.get("layers"))
        if command == "ablation_model":
            model = validate_model(request.get("path"))
            settings = self.settings()
            workspace = Path(settings["evidence_root"]).expanduser() / model.name
            return {"path": str(model), "checkpoint": model_checkpoint(model), "layers": model_layers(model),
                    "default_workspace": str(workspace), "has_index": (model / "model.safetensors.index.json").is_file(),
                    "default_params": default_probe_parameters(model_layers(model) or 1)}
        if command == "ablation_destination":
            return validate_destination(request.get("model"), request.get("dest"))
        if command == "ablation_services":
            return {"services": launch_agents()}
        if command == "ablation_plan":
            return self.plan(request)
        if command == "ablation_start":
            return self.start(request)
        if command == "ablation_resume":
            return self.resume(request)
        if command == "ablation_cancel":
            if not (self.thread and self.thread.is_alive()):
                raise ValueError("No ablation job is running")
            self.cancel.set()
            return {"cancel_requested": True}
        if command == "ablation_log":
            return self.log(request)
        if command == "ablation_settings":
            return self.configure(request.get("changes"))
        raise ValueError("Unknown ablation operation")

    def close(self):
        with self.lock:
            self.closed = True
            self.cancel.set()
        if self.thread:
            self.thread.join(timeout=25)
