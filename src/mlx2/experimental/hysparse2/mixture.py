"""Deterministic fixed-token-length sampling with explicit source probabilities."""

import hashlib
import json
import math
from pathlib import Path


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Mixture:
    def __init__(self, path, *, tokenizer_sha256, sequence):
        import numpy as np

        path = Path(path)
        plan = json.loads(path.read_text())
        if plan.get("tokenizer_sha256") != tokenizer_sha256:
            raise ValueError("mixture tokenizer differs")
        self.values = []
        self.names = []
        weights = []
        self.receipt = []
        for source in plan["sources"]:
            weight = source["weight"]
            if (
                not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight <= 0
            ):
                raise ValueError("mixture weights must be finite and positive")
            name = source["id"]
            if name in self.names:
                raise ValueError("duplicate source ID")
            target = Path(source["path"]).expanduser()
            if not target.is_absolute():
                target = path.parent / target
            values = np.load(target, mmap_mode="r", allow_pickle=False)
            if (
                values.ndim != 1
                or values.dtype != np.uint32
                or len(values) < sequence + 2
            ):
                raise ValueError(f"invalid or too-short token source: {name}")
            actual = file_hash(target)
            if source.get("sha256") is not None and source["sha256"] != actual:
                raise ValueError(f"token source hash differs: {name}")
            self.names.append(name)
            self.values.append(values)
            weights.append(weight)
            self.receipt.append(
                {"id": name, "sha256": actual, "tokens": len(values), "weight": weight}
            )
        if not weights or not math.isfinite(sum(weights)):
            raise ValueError("empty or invalid mixture")
        self.weights = np.asarray(weights, dtype=np.float64) / sum(weights)
        self.sequence = sequence

    def sample(self, rng, batch):
        import numpy as np

        choices = rng.choice(len(self.values), size=batch, p=self.weights)
        rows = []
        for index in choices:
            values = self.values[index]
            start = rng.integers(0, len(values) - self.sequence - 1)
            rows.append(values[start : start + self.sequence + 2])
        return np.stack(rows), [self.names[i] for i in choices]
