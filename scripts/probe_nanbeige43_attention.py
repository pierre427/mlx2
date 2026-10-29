"""Bounded synthetic diagnostic. Run only inside the campaign's GPU lease.

Import run_probe() from an already-owned probe; it returns source-bound results
and does not claim model qualification. This module acquires no GPU ownership.
"""
import hashlib
import json
import subprocess


def run_probe():
    import mlx.core as mx

    from mlx2.adapters.nanbeige43_attention import LAW, attend, source_identity

    if mx.default_device() != mx.gpu:
        raise RuntimeError("this diagnostic requires an already-owned GPU window")
    results = []
    dispatch_trace = []
    for dtype in (mx.float32, mx.bfloat16, mx.float16):
        for prefix in (0, 257, 258, 259, 4096):
            mx.random.seed(4276)
            b, w, maximum = 2, 5, prefix
            prefixes = [prefix, max(0, prefix-251)]
            q = mx.random.normal((b, 56, w, 128)).astype(dtype)
            k = mx.random.normal((b, 8, maximum+w+256, 128)).astype(dtype)[:, :, :maximum+w]
            v = mx.random.normal((b, 8, maximum+w+256, 128)).astype(dtype)[:, :, :maximum+w]
            tree = [[True, False, False, False, False], [True, True, False, False, False],
                    [True, False, True, False, False], [True, True, False, True, False],
                    [True, False, True, False, True]]
            ancestors = mx.array([tree]*b)
            masks = []
            for index, length in enumerate(prefixes):
                head = mx.broadcast_to(mx.arange(maximum)[None] >= maximum-length, (w, maximum))
                masks.append(mx.concatenate([head, ancestors[index]], axis=1))
            reference = mx.fast.scaled_dot_product_attention(q, k, v, scale=128**-0.5, mask=mx.stack(masks)[:, None])
            candidate = attend(q, k, v, prefixes=prefixes, ancestors=ancestors, scale=128**-0.5, dispatch_trace=dispatch_trace)
            # Include all masked/empty leading split segments for short lane.
            mx.eval(candidate, reference)
            error = float(mx.max(mx.abs(candidate.astype(mx.float32)-reference.astype(mx.float32))).item())
            finite = bool(mx.all(mx.isfinite(candidate)).item())
            assert finite and error < (0.03 if dtype != mx.float32 else 3e-5), (dtype, prefix, error)
            results.append({"dtype": str(dtype), "prefixes": prefixes, "max_abs_error": error,
                            "finite": finite, "query_shape": list(q.shape)})
    assert len({json.dumps(row["partial_template"]) for row in dispatch_trace}) == 3
    assert len({json.dumps(row["combine_template"]) for row in dispatch_trace}) == 3
    fixture = {"seed": 4276, "geometries": [row["query_shape"]+row["prefixes"] for row in results]}
    return {"checkpoint": "synthetic-nanbeige43-gqa7-tree@sha256:"+hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "kernel_sha256": source_identity(), "law": LAW, "scope": "synthetic numerical diagnostic only; no model quality/qualification or performance claim",
            "results": results, "dispatch_trace": dispatch_trace,
            "distinct_partial_dispatch_keys": len({json.dumps(row["partial_template"]) for row in dispatch_trace}),
            "distinct_combine_dispatch_keys": len({json.dumps(row["combine_template"]) for row in dispatch_trace}),
            "expected_keys_per_stage": 3,
            "compilation_evidence_scope": "Three dtype families across15 changing-context dispatches; templates contain no prefix/split/span. Python factory cache is not a Metal compile counter."}


if __name__ == "__main__":
    raise SystemExit("Import run_probe() inside an owned campaign probe; do not run this directly.")


def run_model_isolation_probe(model, *, checkpoint, source_revision, implicit_attention=False, segmented_attention=False, shared_kv_rows=1, rope_kv_fused=False):
    """Same-shape peer-content isolation, mixed commit and ordinary continuation.

    The caller owns GPU lease/paired locks and the loaded exact checkpoint.
    This tests lane ownership, not B1/B2 numerical equivalence or answer quality.
    """
    import copy

    import mlx.core as mx

    from mlx2.adapters.nanbeige43_tree import NanbeigePackedTreeExecutor

    if mx.default_device() != mx.gpu:
        raise RuntimeError("this diagnostic requires an already-owned GPU window")
    if "@" not in checkpoint or not source_revision:
        raise ValueError("model probe requires exact checkpoint and source revision")
    def prefill(tokens):
        cache = model.make_cache()
        logits = model(mx.array([tokens], dtype=mx.uint32), cache)
        mx.eval(logits, [entry.state for entry in cache])
        return cache
    a = prefill([152367, 13]*16)
    b = prefill([152375, 13]*17)
    peer_changed = prefill([152385, 13]*17)
    inputs = [[152395, 13, 152367], [152375, 13, 152385]]
    other_inputs = [inputs[0], [152394, 13, 152385]]
    parents = [[-1, 0, 0], [-1, 0, 1]]
    provider = NanbeigePackedTreeExecutor.create(implicit_attention=implicit_attention, segmented_attention=segmented_attention, shared_kv_rows=shared_kv_rows, rope_kv_fused=rope_kv_fused)
    cases = []
    for peer, tokens in ((b, inputs), (peer_changed, other_inputs)):
        caches = [copy.deepcopy(a), copy.deepcopy(peer)]
        logits, features, transaction = provider.forward_many(model, tokens, parents, caches, (1, 25, 49),
                                                            cached=False, configured_limit=2)
        mx.eval(logits, features)
        assert provider.dispatch_metrics(2, transaction, cached=False).physical_packed
        before_offsets = [32, 34]
        assert all(item.offset == before_offsets[lane]+3 for lane, cache in enumerate(caches) for item in cache)
        transaction.commit_paths([[0, 2], [0]])
        assert all(item.offset == 34 for item in caches[0])
        assert all(item.offset == 35 for item in caches[1])
        next_logits = model(mx.array([[152375]], dtype=mx.uint32), caches[0])
        mx.eval(next_logits, [entry.state for entry in caches[0]])
        cases.append((logits[0], features[0], caches[0], next_logits))
    def same(left, right):
        return bool(mx.all(left == right).item())
    logits_equal = same(cases[0][0], cases[1][0])
    taps_equal = same(cases[0][1], cases[1][1])
    cache_equal = all(same(x, y) for first, second in zip(cases[0][2], cases[1][2], strict=True)
                      for x, y in zip(first.keys_and_values(), second.keys_and_values(), strict=True))
    next_equal = same(cases[0][3], cases[1][3])
    assert logits_equal and taps_equal and cache_equal and next_equal
    abort_caches = [copy.deepcopy(a), copy.deepcopy(b)]
    before = copy.deepcopy(abort_caches)
    logits, features, transaction = provider.forward_many(model, inputs, parents, abort_caches, (1, 25, 49),
                                                        cached=False, configured_limit=2)
    mx.eval(logits, features)
    transaction.abort()
    assert all(item.offset == prior.offset and all(same(x, y) for x, y in zip(item.keys_and_values(), prior.keys_and_values(), strict=True))
               for cache, reference in zip(abort_caches, before, strict=True)
               for item, prior in zip(cache, reference, strict=True))
    return {"checkpoint": checkpoint, "source_revision": source_revision,
            "provider": provider.receipt(), "scope": "same-shape B2 peer-content isolation; not B1/B2 equivalence or quality assessment",
            "prefixes": [32, 34], "tree_rows": 3, "captures": [1, 25, 49],
            "physical_packed": True, "same_shape_peer_content_logits_exact": logits_equal,
            "same_shape_peer_content_taps_exact": taps_equal,
            "divergent_commit_lane_a_kv_exact": cache_equal,
            "post_peer_retirement_lane_a_continuation_exact": next_equal,
            "abort_committed_state_exact": True}
