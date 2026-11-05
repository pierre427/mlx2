"""Restricted, reversible LoRA mutation for an idle serving adapter.

Derived from the LoRALinear mechanism in mlx-lm-unified at revision
13eb83388750435bcc751d0b9e33857a38152544.  This serving port deliberately
supports only explicit Linear/QuantizedLinear keys, one active adapter, exact
tensor coverage, and reversible module replacement.
"""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

WEIGHT_PROGRAM_ARITHMETIC = "fp32_delta_round_once_v1"


def read_lora_config(path):
    path = Path(path).expanduser().resolve()
    config_path = path / "adapter_config.json"
    weights_path = path / "adapters.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError("LoRA directory requires adapter_config.json and adapters.safetensors")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict) or config.get("fine_tune_type", "lora") != "lora":
        raise ValueError("dynamic serving supports LoRA adapters only")
    parameters = config.get("lora_parameters")
    if not isinstance(parameters, dict):
        raise ValueError(  # noqa: TRY004 - artifact validation is one ValueError API
            "LoRA config requires lora_parameters"
        )
    allowed = {"rank", "scale", "dropout", "keys"}
    if set(parameters) - allowed:
        raise ValueError("unsupported LoRA parameters: " + ", ".join(sorted(set(parameters) - allowed)))
    rank = parameters.get("rank", 8)
    scale = parameters.get("scale", 20.0)
    dropout = parameters.get("dropout", 0.0)
    keys = parameters.get("keys")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 1024:
        raise ValueError("LoRA rank must be an integer from 1 to 1024")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise ValueError(  # noqa: TRY004 - artifact validation is one ValueError API
            "LoRA scale must be numeric"
        )
    if dropout != 0 and dropout != 0.0:
        raise ValueError("serving LoRA dropout must be zero")
    if not isinstance(keys, list) or not keys or any(
        not isinstance(key, str) or not key or key.startswith(".") for key in keys
    ):
        raise ValueError("serving LoRA requires a nonempty explicit keys list")
    if len(set(keys)) != len(keys):
        raise ValueError("LoRA keys must be unique")
    result = {"rank": rank, "scale": float(scale), "keys": tuple(keys)}
    # A leftover manifest cannot silently downgrade a compiled program to an
    # ordinary adapter when its config marker is removed.
    if "weight_program" in config or (path / "weight_program.json").exists():
        result["weight_program"] = config.get("weight_program")
    return path, result


def inspect_lora_program(path, config, *, base_checkpoint=None, payload=None,
                         require_base=True):
    """Validate compiled programs while preserving the legacy LoRA format."""
    if "weight_program" not in config:
        return None
    if require_base and not base_checkpoint:
        raise ValueError("weight program requires the exact loaded base checkpoint")
    from ..experimental.weight_space.programs import validate_program_artifact

    program = validate_program_artifact(
        path, expected_checkpoint=base_checkpoint, weights_payload=payload,
    )
    if program["numerical_policy"] != "unmerged_float32_factors":
        raise ValueError("unsupported weight program numerical policy")
    # Validation re-reads the config from disk; callers build modules and the
    # fingerprint from ``config``, so it must be the one that was validated.
    if (config["rank"] != program["rank"] or config["scale"] != 1.0
            or tuple(config["keys"]) != tuple(t["target"] for t in program["targets"])
            or config["weight_program"] != program):
        raise ValueError("LoRA config does not match the validated weight program")
    return program


def lora_fingerprint(payload, rank, scale, keys, weight_program=None):
    """Content identity; the historical identity is unchanged for plain LoRA."""
    digest = hashlib.sha256()
    digest.update(repr((rank, scale, tuple(sorted(keys)))).encode())
    digest.update(payload)
    if weight_program is not None:
        digest.update(json.dumps({
            "program_id": weight_program["program_id"],
            "checkpoint": weight_program["checkpoint"],
            "numerical_policy": weight_program["numerical_policy"],
            "runtime_arithmetic": WEIGHT_PROGRAM_ARITHMETIC,
        }, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


# Covered-call accounting under ``mx.compile``.  A wrapper counts each call by
# running a callback; a compiled replay never re-enters Python, so a site that
# traces covered modules compiles through ``compile_counting_calls``, which
# logs the callbacks its trace ran and runs them again on every call.
_CALL_TRACE = threading.local()


def count_covered_call(count):
    """Run ``count`` now, or log it for the compiled trace that is recording."""
    stack = getattr(_CALL_TRACE, "stack", None)
    if stack:
        stack[-1].append(count)
    else:
        count()


@contextmanager
def deferred_call_counts():
    """Hold covered-call counts until the block succeeds; drop them if it raises.

    For a compiled site made of several calls (the PLE head and tail) whose
    failure reruns the whole span eagerly: the eager rerun counts the calls,
    so the abandoned compiled part must not count them too.
    """
    stack = getattr(_CALL_TRACE, "stack", None)
    if stack is None:
        stack = _CALL_TRACE.stack = []
    recorder = []
    stack.append(recorder)
    try:
        yield
    finally:
        stack.pop()
    for count in recorder:
        count_covered_call(count)


def records_compiled_calls(module):
    """True when ``module`` or a child counts its calls (needs counted compile)."""
    return any(getattr(child, "records_compiled_calls", False)
               for _name, child in module.named_modules())


def compile_counting_calls(fn):
    """``mx.compile(fn)`` whose every call runs the call counts its trace ran.

    Counts apply after the compiled call returns, from the current state each
    callback reads, so a call counts exactly what the eager forward would and
    a failed trace counts nothing.  The replayed log is the latest trace's:
    key one of these per traced signature (shapes, dtypes, flags), as the
    compiled PLE chain does.  Not for ``shapeless`` traces.
    """
    import mlx.core as mx

    replay = [()]

    def traced(*args):
        stack = getattr(_CALL_TRACE, "stack", None)
        if stack is None:
            stack = _CALL_TRACE.stack = []
        stack.append([])
        try:
            result = fn(*args)
        finally:
            calls = tuple(stack.pop())
        _CALL_TRACE.traced = calls
        return result

    compiled = mx.compile(traced)

    def run(*args):
        outer = getattr(_CALL_TRACE, "traced", None)
        _CALL_TRACE.traced = None
        try:
            result = compiled(*args)
            calls = _CALL_TRACE.traced
        finally:
            _CALL_TRACE.traced = outer
        if calls is not None:
            replay[0] = calls
        for count in replay[0]:
            count_covered_call(count)
        return result

    run.counts_calls = True
    return run


def weight_program_receipt(program, *, selected=False, observed_used=False):
    if program is None:
        return None
    return {
        key: program[key]
        for key in ("program_id", "checkpoint", "source_revision", "construction",
                    "numerical_policy", "approximate")
    } | {
        "runtime_arithmetic": WEIGHT_PROGRAM_ARITHMETIC,
        "implemented": True, "qualified": False,
        "selected": selected, "observed_used": observed_used,
    }


@dataclass
class LoRASession:
    name: str
    path: str
    originals: dict
    weight_program: dict | None = None
    fingerprint: str | None = None
    # Called program-covered targets; one-element list shared with the
    # installed wrappers.  Requests snapshot it at admission and report the
    # delta, as multi-LoRA does per row.
    program_applications: list | None = None

    @property
    def program_application_count(self):
        return self.program_applications[0] if self.program_applications else 0

    def restore(self, model):
        from mlx.utils import tree_unflatten

        model.update_modules(tree_unflatten(list(self.originals.items())))


def install_lora(model, *, name, path, base_checkpoint=None):
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_flatten, tree_unflatten

    path, config = read_lora_config(path)
    payload = (path / "adapters.safetensors").read_bytes()
    program = inspect_lora_program(
        path, config, base_checkpoint=base_checkpoint, payload=payload,
    )
    applications = [0]

    def count_application():
        applications[0] += 1

    class ServingLoRALinear(nn.Module):
        # A program's covered calls are counted, so compiled sites that trace
        # this module must replay them (``compile_counting_calls``).
        records_compiled_calls = program is not None

        def __init__(self, linear):
            super().__init__()
            output_dims, packed_input_dims = linear.weight.shape
            input_dims = packed_input_dims
            if isinstance(linear, nn.QuantizedLinear):
                input_dims = packed_input_dims * 32 // linear.bits
            self.linear = linear
            self.scale = config["scale"]
            self.lora_a = mx.zeros((input_dims, config["rank"]))
            self.lora_b = mx.zeros((config["rank"], output_dims))

        def __call__(self, value):
            if program is not None:
                # Only a called covered target counts as program engagement.
                count_covered_call(count_application)
                y = self.linear(value)
                update = (value.astype(mx.float32) @ self.lora_a) @ self.lora_b
                return (y.astype(mx.float32) + self.scale * update).astype(y.dtype)
            update = (value @ self.lora_a) @ self.lora_b
            return self.linear(value) + (self.scale * update).astype(value.dtype)

    modules = dict(model.named_modules())
    resolve_key = getattr(model, "lora_module_key", lambda key: key)
    module_keys = tuple(resolve_key(key) for key in config["keys"])
    if any(not isinstance(key, str) or not key for key in module_keys):
        raise ValueError("LoRA module key mapping returned an invalid key")
    if len(set(module_keys)) != len(module_keys):
        raise ValueError("LoRA module key mapping is not one-to-one")
    originals = {}
    replacements = []
    for key, module_key in zip(config["keys"], module_keys):
        module = modules.get(module_key)
        if not isinstance(module, (nn.Linear, nn.QuantizedLinear)):
            raise ValueError(  # noqa: TRY004 - the selected key is incompatible
                f"LoRA key {key!r} is not a Linear/QuantizedLinear module"
            )
        originals[module_key] = module
        replacements.append((module_key, ServingLoRALinear(module)))
    model.update_modules(tree_unflatten(replacements))
    try:
        weights = mx.load(BytesIO(payload), format="safetensors")
        parameters = dict(tree_flatten(model.parameters()))
        expected = {
            f"{key}.{leaf}"
            for key in config["keys"]
            for leaf in ("lora_a", "lora_b")
        }
        if set(weights) != expected:
            missing = sorted(expected - set(weights))
            extra = sorted(set(weights) - expected)
            raise ValueError(f"LoRA tensor coverage mismatch; missing={missing}, extra={extra}")
        mapped_weights = []
        source_to_module = dict(zip(config["keys"], module_keys))
        for key, value in weights.items():
            source, leaf = key.rsplit(".", 1)
            module_key = source_to_module[source]
            target = f"{module_key}.{leaf}"
            if target not in parameters or tuple(value.shape) != tuple(parameters[target].shape):
                expected_shape = tuple(parameters[target].shape) if target in parameters else None
                raise ValueError(
                    f"LoRA tensor {key!r} has shape {tuple(value.shape)}, expected {expected_shape}"
                )
            mapped_weights.append((target, value))
        model.load_weights(mapped_weights, strict=False)
        mx.eval(model.parameters())
    except BaseException:
        model.update_modules(tree_unflatten(list(originals.items())))
        raise
    return LoRASession(
        name=name, path=str(path), originals=originals, weight_program=program,
        program_applications=applications if program is not None else None,
        fingerprint=lora_fingerprint(payload, config["rank"], config["scale"],
                                     config["keys"], program),
    )
