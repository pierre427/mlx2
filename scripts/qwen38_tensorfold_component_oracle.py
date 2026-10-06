"""CPU-import-safe helpers for the Qwen3.8 TensorFold component oracle.

The native runner lives in ``qualify_qwen38_dflash_checkpoint_parity.py``.
This module contains the policy-arm and projection-capture mechanics so their
contracts can be tested without importing MLX or constructing Metal kernels.
"""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Iterator

PROJECTION_LAWS = ("common", "crossover")
FUSED_GDN_ARMS = ("on", "off")


def explicit_execution_policy(policy: dict, fused_gdn: str) -> dict:
    """Return a private policy copy with the GDN arm made explicit."""

    if fused_gdn not in FUSED_GDN_ARMS:
        raise ValueError(f"fused GDN arm must be one of {FUSED_GDN_ARMS}")
    result = copy.deepcopy(policy)
    result["fused_gdn"] = fused_gdn == "on"
    return result


def explicit_projection_policy(policy: dict, law: str) -> dict:
    """Bind q4 to one common lane law or retain the selected crossover.

    ``common`` changes only q4's crossover to one row.  It deliberately does
    not select lane ``exact`` globally, because the diagnostic question is
    the served q4 target rather than unrelated floating-point projections.
    """

    if law not in PROJECTION_LAWS:
        raise ValueError(f"projection law must be one of {PROJECTION_LAWS}")
    result = copy.deepcopy(policy)
    if law == "common":
        result["min_rows"]["q4"] = 1
        result["sources"]["min_rows.q4"] = "component-oracle:common"
    return result


def projection_targets(model) -> dict[int, tuple[int, str]]:
    """Map only the three requested projection boundaries per decoder layer."""

    targets: dict[int, tuple[int, str]] = {}
    for index, layer in enumerate(model.layers):
        inner = getattr(layer, "_layer", layer)
        if bool(getattr(inner, "is_linear", False)):
            mixer = inner.linear_attn.out_proj
            component = "gdn_projection"
        else:
            mixer = inner.self_attn.o_proj
            component = "attention_projection"
        targets[id(mixer)] = (index, component)
        targets[id(inner.mlp.down_proj)] = (index, "mlp_projection")
    return targets


class ProjectionCapture:
    """Temporarily intercept selected projection instances at class dispatch.

    Python resolves ``obj(...)`` through ``type(obj).__call__`` rather than an
    instance attribute.  The class wrapper therefore filters by object id and
    is restored exactly on exit.  No model module is replaced or re-parented.
    """

    def __init__(self, model):
        self.targets = projection_targets(model)
        self.records: dict[str, dict[tuple[int, str], tuple[object, object]]] = {}
        self._active: str | None = None
        self._originals: dict[type, object] = {}
        modules = {id(module): module for _, module in model.named_modules()}
        missing = set(self.targets) - set(modules)
        if missing:
            raise RuntimeError("projection targets are absent from named_modules")
        self._classes = {type(modules[key]) for key in self.targets}

    def __enter__(self):
        owner = self
        for cls in self._classes:
            original = cls.__call__
            self._originals[cls] = original

            def wrapped(module, *args, __original=original, **kwargs):
                output = __original(module, *args, **kwargs)
                target = owner.targets.get(id(module))
                if owner._active is not None and target is not None:
                    if not args:
                        raise RuntimeError(
                            "captured projection has no positional input"
                        )
                    route = owner.records.setdefault(owner._active, {})
                    if target in route:
                        raise RuntimeError(
                            f"projection {target} ran more than once in {owner._active}"
                        )
                    # The root is row zero for the tree and the sole row for
                    # ordinary decode.  Retain lazy device arrays until the
                    # native runner evaluates and summarizes both sides.
                    route[target] = (args[0][:, :1], output[:, :1])
                return output

            cls.__call__ = wrapped
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._active = None
        for cls, original in self._originals.items():
            cls.__call__ = original
        self._originals.clear()
        return False

    @contextlib.contextmanager
    def route(self, name: str) -> Iterator[None]:
        if self._active is not None:
            raise RuntimeError(f"capture route {self._active!r} is already active")
        if name in self.records:
            raise RuntimeError(f"capture route {name!r} already exists")
        self._active = name
        try:
            yield
        finally:
            self._active = None


def component_report(mx, capture: ProjectionCapture, compare, *, atol, rtol) -> dict:
    """Evaluate and compare tree-root and ordinary projection checkpoints."""

    tree = capture.records.get("tensorfold", {})
    ordinary = capture.records.get("ordinary", {})
    expected = set(capture.targets.values())
    missing = {
        "tensorfold": sorted(expected - set(tree)),
        "ordinary": sorted(expected - set(ordinary)),
    }
    extra = {
        "tensorfold": sorted(set(tree) - expected),
        "ordinary": sorted(set(ordinary) - expected),
    }
    rows = []
    for key in sorted(expected):
        if key not in tree or key not in ordinary:
            continue
        tree_input, tree_output = tree[key]
        ordinary_input, ordinary_output = ordinary[key]
        mx.eval(tree_input, tree_output, ordinary_input, ordinary_output)
        input_check = compare(mx, tree_input, ordinary_input, atol=atol, rtol=rtol)
        output_check = compare(mx, tree_output, ordinary_output, atol=atol, rtol=rtol)
        rows.append(
            {
                "layer": key[0],
                "component": key[1],
                "input": input_check,
                "output": output_check,
            }
        )

    def first(field: str, predicate):
        for row in rows:
            if predicate(row[field]):
                return {"layer": row["layer"], "component": row["component"]}
        return None

    return {
        "checkpoints": rows,
        "missing": missing,
        "extra": extra,
        "complete": not any(missing.values()) and not any(extra.values()),
        "first_input_exact_divergence": first("input", lambda item: not item["equal"]),
        "first_input_tolerance_failure": first("input", lambda item: not item["close"]),
        "first_output_exact_divergence": first(
            "output", lambda item: not item["equal"]
        ),
        "first_output_tolerance_failure": first(
            "output", lambda item: not item["close"]
        ),
    }


def fused_gdn_engagement_report(before: dict, after: dict, arm: str) -> dict:
    """Prove that the named ordinary GDN arm actually executed its policy."""

    if arm not in FUSED_GDN_ARMS:
        raise ValueError(f"fused GDN arm must be one of {FUSED_GDN_ARMS}")
    counters = ("decode_calls", "batch_decode_calls", "fallbacks")
    delta = {
        name: int(after.get(name, 0)) - int(before.get(name, 0))
        for name in counters
    }
    layers = int(after.get("layers", 0))
    enabled = bool(after.get("enabled"))
    if arm == "on":
        passed = (
            enabled
            and layers > 0
            and delta["decode_calls"] == layers
            and delta["batch_decode_calls"] == 0
            and delta["fallbacks"] == 0
        )
    else:
        passed = not enabled and all(value == 0 for value in delta.values())
    return {
        "arm": arm,
        "before": before,
        "after": after,
        "delta": delta,
        "expected_decode_calls": layers if arm == "on" else 0,
        "passed": passed,
    }
