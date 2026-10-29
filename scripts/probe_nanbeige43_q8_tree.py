"""Parent-leased Q8 AST ownership probe; this module never acquires GPU itself."""
from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path

import mlx.core as mx

from mlx2.adapters.nanbeige43_cache import candidate_q8_cache
from mlx2.adapters.nanbeige43_quantized_tree import (
    NanbeigeQ8TreeExecutor,
    clone_q8_cache,
)
from mlx2.runtime.approximate_kv import prompt_cache_is_approximate


def _equal_prefix(actual, expected):
    checks=[]
    for a,b in zip(actual,expected,strict=True):
        if a.offset!=b.offset:return False
        for x,y in zip((*a.keys,*a.values),(*b.keys,*b.values),strict=True):
            checks.append(mx.array_equal(x[...,:a.offset,:],y[...,:b.offset,:]))
    return bool(mx.stack(checks).all().item())


def _dequantization_guard(original, *, embedding, hidden_size, kv_head_dim):
    """Permit selected embedding rows while rejecting history dequantization.

    Geometry alone is insufficient: a reshaped KV array can resemble an
    embedding row. The direct caller must also be this exact embedding's
    __call__. Counters describe Python API calls/attempts inside the scope.
    """
    bits, group = int(embedding.bits), int(embedding.group_size)
    expected_packed, expected_scales = hidden_size * bits // 32, hidden_size // group
    kv_packed = kv_head_dim * 8 // 32
    counts = {"embedding_calls": 0, "history_calls": 0, "unexpected_calls": 0}

    def guarded(*args, **kwargs):
        packed = args[0] if args else None
        shape = tuple(getattr(packed, "shape", ()))
        caller = inspect.currentframe().f_back
        try:
            embedding_caller = (
                caller.f_code.co_name == "__call__"
                and caller.f_locals.get("self") is embedding
            )
        finally:
            del caller
        parts = (
            packed,
            args[1] if len(args) > 1 else kwargs.get("scales"),
            args[2] if len(args) > 2 else kwargs.get("biases"),
        )
        valid_parts = all(hasattr(part, "shape") for part in parts)
        geometry = False
        if valid_parts and shape:
            scales, biases = parts[1:]
            geometry = (
                shape[-1] == expected_packed
                and tuple(scales.shape[:-1]) == tuple(biases.shape[:-1]) == shape[:-1]
                and int(scales.shape[-1]) == int(biases.shape[-1]) == expected_scales
                and math.prod(shape[:-1]) <= 32
                and kwargs.get("bits") == bits and kwargs.get("group_size") == group
                and kwargs.get("mode", "affine") == "affine"
            )
        if shape and shape[-1] == kv_packed:
            counts["history_calls"] += 1
            raise AssertionError("Q8 tree invoked KV-history dequantization")
        if embedding_caller and geometry:
            counts["embedding_calls"] += 1
            return original(*args, **kwargs)
        counts["history_calls"] += 1
        counts["unexpected_calls"] += 1
        raise AssertionError("Q8 tree invoked non-embedding/reshaped-history dequantization")

    return guarded, counts


def _bounded_continuation_window(tree_row, serial_row, tree_selected, serial_selected):
    """Only the already-selected comparison position; no generation or oracle."""
    if tree_row.ndim != 1 or tree_row.shape != serial_row.shape:
        raise ValueError("continuation evidence needs matching vocabulary rows")
    vocab = int(tree_row.shape[0])
    if not all(type(token) is int and 0 <= token < vocab for token in (tree_selected, serial_selected)):
        raise ValueError("continuation evidence selected token is outside the vocabulary")

    def arm(row, selected, alternate):
        row = row.astype(mx.float32)
        logprobs = row - mx.logsumexp(row)
        count = min(8, vocab)
        ids = mx.argpartition(-logprobs, count - 1)[:count]
        values = logprobs[ids]
        selected_values = logprobs[mx.array([selected, alternate], dtype=mx.int32)]
        mx.eval(ids, values, selected_values)
        top = sorted(
            ({"token_id": int(token), "logprob": float(value)}
             for token, value in zip(ids.tolist(), values.tolist(), strict=True)),
            key=lambda entry: (-entry["logprob"], entry["token_id"]),
        )
        own, other = map(float, selected_values.tolist())
        return {"selected_token_id": selected, "other_arm_selected_token_id": alternate,
                "selected_logprob": own, "other_arm_selected_logprob": other,
                "selected_minus_other_margin_nats": own-other,
                "other_selected_gap_from_top_nats": top[0]["logprob"]-other,
                "top_two_margin_nats": (top[0]["logprob"]-top[1]["logprob"] if len(top)>1 else None),
                "top8": top}

    return {"schema": "mlx2.single-position-continuation-window.v1",
            "normalization_dtype": "float32", "window_positions": 1,
            "tree": arm(tree_row, tree_selected, serial_selected),
            "serial": arm(serial_row, serial_selected, tree_selected),
            "equivalence_classification": "unclassified_without_functional_continuation_oracle",
            "qualitative_quality_verdict": None}


def _q8_format(cache):
    return (type(cache).__name__, cache.group_size, cache.key_bits, cache.value_bits,
            cache.rotate, cache.normalize,
            tuple((part.shape[-1], str(part.dtype)) for part in (*cache.keys, *cache.values)))


def _first_suffix_code_difference(tree_cache, serial_cache, start, keep):
    names = ("key_codes", "key_scales", "key_biases", "value_codes", "value_scales", "value_biases")
    checks = [mx.array_equal(a[..., start:start+keep, :], b[..., start:start+keep, :])
              for tree, serial in zip(tree_cache, serial_cache, strict=True)
              for a, b in zip((*tree.keys, *tree.values), (*serial.keys, *serial.values), strict=True)]
    flags = mx.stack(checks).tolist()
    for index, equal in enumerate(flags):
        if not equal:
            return {"layer": index//6, "component": names[index%6],
                    "scope": "first differing packed component in the identically forced committed suffix; not a root-cause verdict"}
    return None


def run(model, *, prompt_tokens, checkpoint, source_revision, adapter_fingerprint):
    """Called only inside the parent's CPG/paired-lock ownership window."""
    if not checkpoint or '@' not in checkpoint or not source_revision or not 1<=len(prompt_tokens)<=512:
        raise ValueError('Q8 probe requires exact checkpoint/source and bounded prompt')
    cache=candidate_q8_cache(model,allow_approximate=True)
    logits=model(mx.array([prompt_tokens],dtype=mx.uint32),cache);mx.eval(logits,[item.state for item in cache])
    source=clone_q8_cache(cache)
    source_before=clone_q8_cache(source)
    anchor=int(mx.argmax(logits[0,-1]).item())
    tokens=[anchor,((anchor+1)%int(model.args.vocab_size)),((anchor+2)%int(model.args.vocab_size)),((anchor+3)%int(model.args.vocab_size))]
    parents=[-1,0,0,2]
    provider=NanbeigeQ8TreeExecutor.create(adapter_fingerprint=adapter_fingerprint)
    dequantize=mx.dequantize
    guard, dequantization_calls = _dequantization_guard(
        dequantize, embedding=model.model.embed_tokens,
        hidden_size=int(model.args.hidden_size), kv_head_dim=int(model.args.head_dim),
    )
    try:
        mx.dequantize=guard
        values,taps,txn=provider.forward(model,tokens,parents,cache,(len(model.layers)-1,),cached=False)
        mx.eval(values,taps);assert mx.isfinite(values).all().item()
        txn.abort();assert _equal_prefix(cache,source)
        values,taps,txn=provider.forward(model,tokens,parents,cache,(len(model.layers)-1,),cached=True)
        mx.eval(values,taps)
        # Capture only the selected bounded suffix's packed codes, not logits.
        keep=[0,2,3];start=len(prompt_tokens)
        packed=[[(mx.take(part[...,start:start+len(tokens),:],mx.array(keep),axis=2)) for part in (*item.keys,*item.values)] for item in cache]
        mx.eval(packed);txn.commit_paths([keep])
        comparisons=[mx.array_equal(part[...,start:item.offset,:],expected) for item,parts in zip(cache,packed,strict=True) for part,expected in zip((*item.keys,*item.values),parts,strict=True)]
        assert mx.stack(comparisons).all().item()
    finally:mx.dequantize=dequantize
    next_token=((anchor+4)%int(model.args.vocab_size))
    actual=model(mx.array([[next_token]],dtype=mx.uint32),cache)
    reference=clone_q8_cache(source)
    for token in [tokens[index] for index in keep]+[next_token]:
        ordinary=model(mx.array([[token]],dtype=mx.uint32),reference)
    mx.eval(actual,ordinary)
    assert _equal_prefix(source,source_before)
    forced_prefix = [int(token) for token in prompt_tokens] + [tokens[index] for index in keep] + [next_token]
    expected_offset = len(forced_prefix)
    offsets_match = all(item.offset == expected_offset for item in (*cache, *reference))
    same_format = all(_q8_format(a) == _q8_format(b) for a,b in zip(cache,reference,strict=True))
    expected_format = all((item.group_size,item.key_bits,item.value_bits,item.rotate,item.normalize)==(64,8,8,False,False)
                          for item in (*cache,*reference))
    assert offsets_match and same_format and expected_format
    tree_selected = int(mx.argmax(actual[0,-1]).item())
    serial_selected = int(mx.argmax(ordinary[0,-1]).item())
    window = _bounded_continuation_window(actual[0,-1], ordinary[0,-1], tree_selected, serial_selected)
    window.update(shared_forced_token_prefix=forced_prefix,
                  forced_path_token_ids=[tokens[index] for index in keep],
                  final_forced_input_token_id=next_token,
                  consumed_prefix_tokens=expected_offset,
                  comparison_kind="tree-W4-derived Q8 state versus serial-W1-derived Q8 state",
                  cache_format_check={"matching":same_format,"expected_q8g64":expected_format,
                                      "matching_offsets":offsets_match,"offset":expected_offset,
                                      "layers":len(cache)},
                  first_packed_suffix_difference=_first_suffix_code_difference(cache,reference,len(prompt_tokens),len(keep)))
    return {'checkpoint':checkpoint,'source_revision':source_revision,
            'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'prompt_tokens':len(prompt_tokens),'tree_parents':parents,'committed_path':keep,
            'abort_committed_prefix_exact':True,'commit_packed_codes_exact':True,
            'full_history_dequantization_calls':dequantization_calls['history_calls'],
            'embedding_dequantization_calls':dequantization_calls['embedding_calls'],
            'dequantization_guard':dict(dequantization_calls, admission='exact embedding caller plus bounded selected-row feature geometry', scope='two target-tree forwards and suffix commit staging'),
            'approximate_cache_structurally_detected':prompt_cache_is_approximate(cache),
            'continuation_argmax_equal':tree_selected==serial_selected,
            'continuation_window':window,
            'continuation_max_abs_error_same_representation':mx.max(mx.abs(actual-ordinary)).item(),
            'provider':provider.receipt(),'qualification':False,
            'scope':'direct target ownership probe; not served AST/APCv2 or qualitative model quality evidence'}
