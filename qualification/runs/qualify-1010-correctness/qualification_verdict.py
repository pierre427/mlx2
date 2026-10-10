#!/usr/bin/env python3
"""Correctness-only verdict for a frozen, planned context ladder.

Admission is an execution safety gate: a refused start leaves the ladder
pending. Throughput and environmental noise never decide qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

WARM_HIT_MIN_FRACTION = 0.90
RUNTIME_DIGESTS = ("source_sha256", "mlx_native_sha256")


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def identity(ladder):
    initial = ladder.get("initial")
    if not isinstance(initial, dict):
        initial = {}
    runtime = initial.get("runtime") or {}
    settings = initial.get("settings")
    return {
        "model": ladder.get("model"),
        "route": ladder.get("route"),
        "host": ladder.get("host"),
        "max_context": ladder.get("max_context"),
        "artifact": initial.get("artifact"),
        "runtime_source_sha256": runtime.get("source_sha256")
        if isinstance(runtime, dict)
        else None,
        "runtime_mlx_native_sha256": runtime.get("mlx_native_sha256")
        if isinstance(runtime, dict)
        else None,
        "settings": settings,
        "mtp": settings.get("mtp") if isinstance(settings, dict) else None,
    }


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _raw_request_problems(row, width, needles):
    problems = []
    requests = row.get("requests")
    if not isinstance(requests, dict):
        return ["raw cold/warm request evidence missing"]
    cold, warm = requests.get("cold"), requests.get("warm")
    cold_rows = cold.get("rows") if isinstance(cold, dict) else None
    warm_rows = warm.get("rows") if isinstance(warm, dict) else None
    if not isinstance(cold_rows, list) or not isinstance(warm_rows, list):
        return ["raw cold/warm request rows missing"]
    if len(cold_rows) != width or len(warm_rows) != width or len(needles) != width:
        return [f"request/needle row count does not match width {width}"]
    if not all(isinstance(needle, str) and needle for needle in needles):
        return ["retrieval needles malformed"]

    for lane, (cold_row, warm_row, needle) in enumerate(
        zip(cold_rows, warm_rows, needles)
    ):
        if not isinstance(cold_row, dict) or not isinstance(warm_row, dict):
            problems.append(f"lane {lane}: malformed request row")
            continue
        for label, request_row in (("cold", cold_row), ("warm", warm_row)):
            if request_row.get("done") is not True:
                problems.append(f"lane {lane}: {label} stream did not finish")
            content = request_row.get("content")
            if not isinstance(content, str):
                problems.append(f"lane {lane}: {label} content missing")
            elif needle not in content:
                problems.append(f"lane {lane}: {label} retrieval needle missing")
            if not _integer(request_row.get("prompt_tokens")):
                problems.append(f"lane {lane}: {label} prompt token count missing")
            if not _integer(request_row.get("cached_tokens")):
                problems.append(f"lane {lane}: {label} cached token count missing")
            if (
                not _integer(request_row.get("completion_tokens"))
                or request_row.get("completion_tokens", 0) < 1
            ):
                problems.append(f"lane {lane}: {label} completion token count missing")
            prompt_tokens = request_row.get("prompt_tokens")
            cached_tokens = request_row.get("cached_tokens")
            if (
                _integer(prompt_tokens)
                and _integer(cached_tokens)
                and cached_tokens > prompt_tokens
            ):
                problems.append(
                    f"lane {lane}: {label} cached tokens exceed prompt tokens"
                )
        if (
            isinstance(cold_row.get("content"), str)
            and isinstance(warm_row.get("content"), str)
            and cold_row["content"] != warm_row["content"]
        ):
            problems.append(f"lane {lane}: cold and warm answers differ")
        prompt_tokens, cached_tokens = (
            cold_row.get("prompt_tokens"),
            cold_row.get("cached_tokens"),
        )
        if (
            _integer(prompt_tokens)
            and _integer(cached_tokens)
            and cached_tokens > max(256, 0.01 * max(prompt_tokens, 1))
        ):
            problems.append(
                f"lane {lane}: cold request reused beyond fixed prompt preamble"
            )
        prompt_tokens, cached_tokens = (
            warm_row.get("prompt_tokens"),
            warm_row.get("cached_tokens"),
        )
        if (
            _integer(prompt_tokens)
            and _integer(cached_tokens)
            and cached_tokens < WARM_HIT_MIN_FRACTION * max(prompt_tokens, 1)
        ):
            problems.append(
                f"lane {lane}: warm APCv2 reuse below {WARM_HIT_MIN_FRACTION:.0%}"
            )
    return problems


def _functional_problems(row, width):
    if not isinstance(row, dict):
        return ["malformed measured run"]
    needles = row.get("needles")
    if not isinstance(needles, list):
        return ["run needle evidence missing"]
    problems = _raw_request_problems(row, width, needles)
    pre = row.get("thermal_pre")
    if (
        row.get("measured") is not True
        or not isinstance(pre, dict)
        or pre.get("stable") is not True
    ):
        problems.append("run lacks safe-admission evidence")
    functional_failures = row.get("functional_failures") or []
    if isinstance(functional_failures, list):
        problems.extend(str(problem) for problem in functional_failures)
    else:
        problems.append("malformed functional_failures evidence")
    return list(dict.fromkeys(problems))


def _noise(row, cell_name):
    if not isinstance(row, dict) or not row.get("contaminated"):
        return None
    details = row.get("contamination") or ["environmental noise"]
    if not isinstance(details, list):
        details = ["malformed contamination evidence"]
    return f"{cell_name} run {row.get('run')}: " + ", ".join(
        str(item) for item in details
    )


def verdict(ladder):
    if not isinstance(ladder, dict):
        return {
            "schema": "mlx2.correctness-qualification-verdict.v1",
            "qualified": False,
            "status": "failed",
            "failures": ["ladder report must be a JSON object"],
            "measurement_noise": [],
            "cells": [],
        }
    failures, noise, cells = [], [], []
    ident = identity(ladder)
    missing = [
        key
        for key in ("model", "route", "host", "artifact", "settings", "max_context")
        if ident[key] in (None, "", {})
    ]
    if missing:
        failures.append("ladder identity missing: " + ", ".join(missing))
    initial = ladder.get("initial")
    runtime = initial.get("runtime") if isinstance(initial, dict) else None
    invalid = [
        key
        for key in RUNTIME_DIGESTS
        if not isinstance(runtime, dict) or not _is_sha256(runtime.get(key))
    ]
    if invalid:
        failures.append(
            "runtime identity missing valid digests: "
            + ", ".join(f"runtime.{key}" for key in invalid)
        )
    if ladder.get("runs_per_cell") != 3:
        failures.append(
            f"runs_per_cell must be exactly 3, got {ladder.get('runs_per_cell')!r}"
        )

    if ladder.get("pending_admission"):
        return {
            "schema": "mlx2.correctness-qualification-verdict.v1",
            "rule": "functional evidence over the declared plan; admission refusal remains pending",
            "identity": ident,
            "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runs_per_cell": 3,
            "warm_hit_min_fraction": WARM_HIT_MIN_FRACTION,
            "qualified": False,
            "status": "pending_admission",
            "pending": ladder["pending_admission"],
            "failures": failures,
            "measurement_noise": noise,
            "cells": [],
        }

    expected_cells, observed_cells = ladder.get("expected_cells"), ladder.get("cells")
    if not isinstance(expected_cells, list) or not expected_cells:
        failures.append("declared expected_cells plan missing")
        expected_cells = []
    if not isinstance(observed_cells, list):
        failures.append("cells must be a list")
        observed_cells = []
    expected_keys = []
    for item in expected_cells:
        if (
            not isinstance(item, dict)
            or not _integer(item.get("requested_tokens"))
            or not _integer(item.get("width"))
            or item.get("width", 0) < 1
        ):
            failures.append(f"malformed expected cell: {item!r}")
            continue
        expected_keys.append((item["requested_tokens"], item["width"]))
    if len(set(expected_keys)) != len(expected_keys):
        failures.append("declared expected_cells contains duplicates")

    observed = {}
    for cell in observed_cells:
        if not isinstance(cell, dict):
            failures.append("malformed cell record")
            continue
        requested_tokens, width = cell.get("requested_tokens"), cell.get("width")
        if not _integer(requested_tokens) or not _integer(width) or width < 1:
            failures.append(
                f"malformed observed cell identity: {requested_tokens!r}, {width!r}"
            )
            continue
        key = (requested_tokens, width)
        if key in observed:
            failures.append(f"duplicate observed cell {key}")
        observed[key] = cell
    if set(observed) != set(expected_keys):
        failures.append(
            f"cell coverage mismatch: expected {sorted(expected_keys)}, observed {sorted(observed)}"
        )
    if ladder.get("finished_at") is None:
        failures.append("ladder did not finish")

    for requested_tokens, width in expected_keys:
        name = f"{requested_tokens}-w{width}"
        cell = observed.get((requested_tokens, width))
        if cell is None:
            continue
        problems = []
        runs, attempts = cell.get("runs"), cell.get("attempts") or []
        if not isinstance(runs, list) or not isinstance(attempts, list):
            problems.append("runs/attempts must be lists")
            runs, attempts = [], []
        if cell.get("error"):
            problems.append(f"cell error: {str(cell['error'])[:200]}")
        if len(runs) != 3:
            problems.append(f"{len(runs)}/3 valid measured runs completed")
        for index, row in enumerate(runs):
            problems.extend(
                f"run {index}: {problem}"
                for problem in _functional_problems(row, width)
            )
        for row in attempts + runs:
            item = _noise(row, name)
            if item:
                noise.append(item)
        cells.append({"cell": name, "passed": not problems, "problems": problems})
        failures.extend(f"{name}: {problem}" for problem in problems)

    qualified = (
        bool(expected_keys) and len(cells) == len(expected_keys) and not failures
    )
    return {
        "schema": "mlx2.correctness-qualification-verdict.v1",
        "rule": "declared plan coverage, three admitted runs, raw stream/needle evidence, cold/warm agreement, APCv2 reuse",
        "identity": ident,
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs_per_cell": 3,
        "warm_hit_min_fraction": WARM_HIT_MIN_FRACTION,
        "qualified": qualified,
        "status": "qualified" if qualified else "failed",
        "failures": failures,
        "measurement_noise": noise,
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ladder", type=Path)
    args = parser.parse_args()
    raw = args.ladder.read_bytes()
    result = verdict(json.loads(raw))
    result["ladder_file"] = {
        "path": str(args.ladder),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    print(json.dumps(result, indent=2))
    return (
        0
        if result["qualified"]
        else 2
        if result.get("status") == "pending_admission"
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
