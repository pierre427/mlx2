"""Exact-artifact AST prepared-state, isolation and continuation oracle."""

import hashlib
import json
from pathlib import Path

import mlx.core as mx

from mlx2.adapters.glm53_ast_target import Glm53TargetTreeExecutor, clone, stamp
from mlx2.runtime.models.glm53_gguf import Glm53GGUFModel


def leaves(value):
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (tuple, list)):
        return [a for v in value for a in leaves(v)]
    return []


def compare_state(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError("layer topology differs")
    count = 0
    for layer_index, (a, b) in enumerate(zip(actual, expected, strict=True)):
        if a.meta_state != b.meta_state:
            raise AssertionError("state metadata/frontier differs")
        left, right = leaves(a.state), leaves(b.state)
        if len(left) != len(right):
            raise AssertionError("state plane count differs")
        for plane_index, (x, y) in enumerate(zip(left, right, strict=True)):
            if (
                x.shape != y.shape
                or x.dtype != y.dtype
                or not bool(mx.all(x == y).item())
            ):
                delta = (
                    float(mx.max(mx.abs(x - y)).item())
                    if x.size and x.shape == y.shape
                    else None
                )
                raise AssertionError(
                    f"prepared state differs at layer{layer_index}/plane{plane_index}: max_abs={delta}"
                )
            count += 1
    return count


def run(adapter, out, checkpoint, source_revision, *, context_tokens=None):
    result = {
        "checkpoint": checkpoint,
        "source_revision": source_revision,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "qualified": False,
        "status": "running",
        "variants": [],
    }
    tokens = [19, 2518, 6303, 6176, 13]
    parents = [-1, 0, 0, 1, 2]
    paths = [[0], [0, 1, 3], [0, 2, 4]]
    try:
        for seed_mode, layers in (
            (False, [5, 14, 24, 33, 42]),
            (False, [23, 27, 31, 35, 39, 43]),
            (True, [44]),
        ):
            m = Glm53GGUFModel(
                adapter.config,
                adapter.weights,
                identity=adapter.model.identity + ":oracle:" + str(seed_mode),
                math_policy="ordinary",
            )
            if seed_mode:
                m.external_feature_source = "mtp_seed"
            live = m.make_cache()
            prompt = context_tokens or [154822, 19, 2518, 13]
            for start in range(0, len(prompt), 32):
                z = m(mx.array([prompt[start : start + 32]]), live, logits_to_keep=1)
                mx.eval(z, [c.state for c in live])
            before = stamp(live)
            provider = Glm53TargetTreeExecutor(
                m,
                implementation=adapter.kernel_policy.get("glm53_ast_target", "serial"),
            )
            logits, features, transaction = provider.forward(
                m, tokens, parents, live, layers
            )
            if stamp(live) != before:
                raise AssertionError("tree changed live frontier")
            checks = []
            expected = None
            for path in paths:
                expected = clone(live)
                for row in path:
                    z, f = m.forward_with_taps(
                        mx.array([[tokens[row]]]), expected, layers
                    )
                    mx.eval(z, f, [c.state for c in expected])
                    if not bool(mx.all(z == logits[:, row : row + 1]).item()):
                        raise AssertionError("logits differ")
                    if not bool(mx.all(f == features[:, row : row + 1]).item()):
                        raise AssertionError("features differ")
                prepared = transaction.prepared_cache(path)
                checks.append(
                    {
                        "path": path,
                        "exact_state_arrays": compare_state(prepared, expected),
                    }
                )
            committed = transaction.commit_paths([paths[-1]])[0]
            z, f = m.forward_with_taps(mx.array([[19]]), committed, layers)
            rz, rf = m.forward_with_taps(mx.array([[19]]), expected, layers)
            mx.eval(
                z, f, rz, rf, [c.state for c in committed], [c.state for c in expected]
            )
            if not bool(mx.all(z == rz).item()) or not bool(mx.all(f == rf).item()):
                raise AssertionError("continuation differs")
            compare_state(committed, expected)
            _, _, aborted = provider.forward(m, tokens[:2], parents[:2], live, layers)
            aborted.abort()
            if stamp(live) != before:
                raise AssertionError("abort changed live frontier")
            result["variants"].append(
                {
                    "feature_source": m.external_feature_source,
                    "capture_layers": layers,
                    "context_tokens": len(prompt),
                    "logits": True,
                    "features": True,
                    "prepared_state": True,
                    "continuation": True,
                    "isolation": True,
                    "commit_abort": True,
                    "checks": checks,
                    "provider": provider.receipt(),
                }
            )
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        out.write_text(json.dumps(result, indent=2) + "\n")
    return result
