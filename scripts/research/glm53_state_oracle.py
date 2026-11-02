"""Exact-artifact AST prepared-state, isolation and continuation oracle.

``run`` with ``kernel_policy=None`` is the original oracle (ordinary math,
only ``glm53_ast_target`` reaches the executor). With ``kernel_policy`` (the
CLI's ``--execution-policy``) every ``glm53_*`` model-level policy value is
forwarded to the fresh ``Glm53GGUFModel`` the oracle builds, the cache is
primed through ``prefill_forward`` (so the prompt phase and
``glm53_sparse_prefill`` are exercised), and the candidate policy's
engagement counters must be non-zero afterwards; otherwise the oracle would
have validated the reference route (design section 12, P5 row).
"""

import argparse
import hashlib
import json
import sys
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


def same_words(x, y):
    """Bit-for-bit equality (signed zeros and NaN payloads included), shapes and dtypes too."""
    import numpy as np

    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.dtype == mx.bfloat16:
        x, y = x.view(mx.uint16), y.view(mx.uint16)
    a, b = np.array(x), np.array(y)
    if a.dtype.kind == "f":
        a, b = a.view(f"u{a.itemsize}"), b.view(f"u{b.itemsize}")
    return bool(np.array_equal(a, b))


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
                or not same_words(x, y)
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


SPARSE_KEYS = (
    "glm53_indexer_pools", "glm53_indexer_select", "glm53_sparse_attention",
    "glm53_sparse_prefill", "glm53_tree_rows_attention", "glm53_tree_frontier",
)


def model_kwargs(policy):
    """Glm53GGUFModel keywords for an adapter kernel policy (as the adapter builds it)."""
    hc = policy.get("glm53_hc", "reference")
    kwargs = {
        "math_policy": policy.get("glm53_math", "ordinary"),
        "hc_fused": hc in ("fused_fp32", "mega_v1"),
        "hc_mega": hc == "mega_v1",
        "shared_fold": policy.get("glm53_shared_fold", "off") == "kblock",
        "kda_group": policy.get("glm53_kda_group", "off") == "kblock",
        "mla_group": policy.get("glm53_mla_group", "off") == "kblock",
        "router": policy.get("glm53_router", "reference"),
        "kda_small": policy.get("glm53_kda_small", "reference"),
        "kda_fused": policy.get("glm53_kda_fused", "reference"),
    }
    for key in SPARSE_KEYS:
        if key in policy:
            kwargs[key.removeprefix("glm53_")] = policy[key]
    return kwargs


def run(adapter, out, checkpoint, source_revision, *, context_tokens=None, kernel_policy=None,
        prime_chunk=32):
    """``kernel_policy`` None keeps the original oracle; otherwise see the module docstring.

    Weight-level laws (dense/expert prefill, row kernels) are bound by the
    adapter's load; ``kernel_policy`` must therefore agree with
    ``adapter.kernel_policy`` on every non-model key, which is checked.
    """
    if kernel_policy is not None:
        weight_keys = ("glm53_dense_prefill", "glm53_expert_prefill", "glm53_row_kernel", "glm53_tree_batch")
        stray = [k for k in weight_keys if kernel_policy.get(k, adapter.kernel_policy.get(k))
                 != adapter.kernel_policy.get(k)]
        if stray:
            raise ValueError(f"kernel policy differs from the loaded weights on {stray}")
    result = {
        "checkpoint": checkpoint,
        "source_revision": source_revision,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "qualified": False,
        "status": "running",
        "kernel_policy": kernel_policy,
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
            if kernel_policy is None:
                m = Glm53GGUFModel(
                    adapter.config,
                    adapter.weights,
                    identity=adapter.model.identity + ":oracle:" + str(seed_mode),
                    math_policy="ordinary",
                )
            else:
                m = Glm53GGUFModel(
                    adapter.config,
                    adapter.weights,
                    identity=adapter.model.identity + ":oracle:" + str(seed_mode),
                    **model_kwargs(kernel_policy),
                )
                if kernel_policy.get("glm53_tree_batch", "off") == "kblock_v1":
                    m.tree_eval_every = 5  # as the adapter binds it
            if seed_mode:
                m.external_feature_source = "mtp_seed"
            live = m.make_cache()
            prompt = context_tokens or [154822, 19, 2518, 13]
            calls_before = dict(m.calls)
            for start in range(0, len(prompt), prime_chunk):
                piece = mx.array([prompt[start : start + prime_chunk]])
                if kernel_policy is None:
                    z = m(piece, live, logits_to_keep=1)
                else:
                    # The prompt phase: glm53_sparse_prefill is exercised.
                    z = m.prefill_forward(piece, live)[:, -1:]
                mx.eval(z, [c.state for c in live])
            before = stamp(live)
            provider = Glm53TargetTreeExecutor(
                m,
                implementation=adapter.kernel_policy.get("glm53_ast_target", "serial"),
            )
            logits, features, transaction = provider.forward(
                m, tokens, parents, live, layers
            )
            if not bool(mx.all(mx.isfinite(logits)).item()) or not bool(mx.all(mx.isfinite(features)).item()):
                raise AssertionError("tree logits/features are not finite")
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
                    if not same_words(z, logits[:, row : row + 1]):
                        raise AssertionError("logits differ")
                    if not same_words(f, features[:, row : row + 1]):
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
            if not all(bool(mx.all(mx.isfinite(v)).item()) for v in (z, f, rz, rf)):
                raise AssertionError("continuation is not finite")
            if not same_words(z, rz) or not same_words(f, rf):
                raise AssertionError("continuation differs")
            compare_state(committed, expected)
            _, _, aborted = provider.forward(m, tokens[:2], parents[:2], live, layers)
            aborted.abort()
            if stamp(live) != before:
                raise AssertionError("abort changed live frontier")
            engagement = None
            if kernel_policy is not None:
                from glm53_policy_ab import engagement as engaged

                engagement = engaged(
                    kernel_policy, calls_before, dict(m.calls),
                    context_tokens=len(prompt) + max(len(p) for p in paths),
                    prompt_tokens=len(prompt), decode_steps=1,
                    tree=kernel_policy.get("glm53_ast_target") == "layered",
                )
                if not engagement["engaged"]:
                    raise AssertionError(
                        f"candidate policy not engaged: missing {engagement['missing']}"
                    )
            result["variants"].append(
                {
                    "engagement": engagement,
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


def main(argv=None):
    """One-command oracle run under ``--execution-policy`` (GPU; loads the checkpoint)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from glm53_policy_ab import (
        CHECKPOINT,
        MODEL,
        atomic_write,
        claim_out_dir,
        host_record,
        load_adapter,
        needle_prompt,
        source_record,
    )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="new or empty evidence directory")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--execution-policy", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=2046,
                        help="realistic prompt length; 2046 makes a depth-3 path cross TOPK=2048")
    parser.add_argument("--prime-chunk", type=int, default=512)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--allow-revision-mismatch", action="store_true")
    parser.add_argument("--allow-dirty-src", action="store_true")
    args = parser.parse_args(argv)
    if not args.i_own_the_gpu:
        raise SystemExit("GPU run: pass --i-own-the-gpu under the GPU lease")
    out = claim_out_dir(args.out)
    source = source_record(args.source_revision, allow_mismatch=args.allow_revision_mismatch,
                           allow_dirty=args.allow_dirty_src)
    policy = json.loads(args.execution_policy.read_text())
    receipt = {"schema": "mlx2.glm53-state-oracle-run.v1", "checkpoint": CHECKPOINT, **source,
               "host": host_record(mx), "execution_policy": policy, "status": "running"}
    atomic_write(out / "receipt.json", receipt)
    adapter = None
    try:
        adapter, receipt["load_seconds"] = load_adapter(policy, args.model)
        ids = needle_prompt(adapter, args.context_tokens)
        result = run(adapter, out / "state-oracle.json", CHECKPOINT, args.source_revision,
                     context_tokens=ids, kernel_policy=adapter.kernel_policy, prime_chunk=args.prime_chunk)
        receipt.update(status=result["status"], context_tokens=len(ids))
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["host_after"] = host_record(mx)
        atomic_write(out / "receipt.json", receipt)
        if adapter is not None:
            adapter.close()
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
