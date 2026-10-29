"""Shared Uzu campaign measurements; historical fusion harness is in research/uzu."""

import os
import random
import statistics
import subprocess


def host_load():
    hits = []
    for line in subprocess.check_output(["ps", "-Ao", "pcpu=,pid=,command="], text=True).splitlines():
        fields = line.split(None, 2)
        if len(fields) != 3 or int(fields[1]) == os.getpid():
            continue
        cpu, command = float(fields[0]), fields[2]
        if cpu >= 60 or ("pytest" in command and cpu >= 5):
            hits.append({"pid": int(fields[1]), "cpu": cpu, "command": command[:140]})
    return hits


def paired_summary(baseline, candidate):
    """Unadjusted paired bootstrap interval; retained raw pairs are authoritative."""
    ratios = [a / b for a, b in zip(baseline, candidate)]
    rng = random.Random(1027)
    draws = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(5000))
    return {"baseline_median_seconds": statistics.median(baseline),
            "candidate_median_seconds": statistics.median(candidate),
            "paired_median_speedup": statistics.median(ratios),
            "paired_speedup_bootstrap95": [draws[125], draws[4874]],
            "pairs": len(ratios), "interval_scope": "per-cell unadjusted paired median bootstrap"}
