"""Shared atomic receipts and swap counter; historical probes are in research/uzu."""

import json
import subprocess


def atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def swapouts():
    text = subprocess.check_output(["vm_stat"], text=True)
    return int(next(line for line in text.splitlines() if line.startswith("Swapouts:")).split()[-1].rstrip("."))
