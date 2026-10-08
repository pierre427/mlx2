#!/usr/bin/env python3
"""Compare one real DFlash proposal across target-forward row partitions.

This is a direct-model research probe.  It obtains one proposal from the
configured DFlash drafter, then feeds the identical ``anchor + proposal``
input sequence to independent copies of the same target-cache boundary.  The
control evaluates one token row per target launch.  Candidate arms use one
fixed monolithic launch or the row partitions required by progressive
verification tiles.

No result is published to a live request, APCv2, or a serving route.  Real
Metal execution requires ``--i-own-the-gpu`` and must still be wrapped by the
repository's dual-lock launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

PROMPTS = (
    (
        "Explain why batching can increase aggregate accelerator throughput while "
        "a single autoregressive stream remains latency-bound."
    ),
    (
        "Write a Python function that parses an ISO-8601 duration into seconds, "
        "then give three tests."
    ),
)

IDENTITY_FILES = (
    "scripts/research/probe_dflash_split_shape.py",
    "scripts/research/progressive_dflash_verify.py",
    "scripts/paired_direct_ab.py",
    "src/mlx2/runtime/external_speculative.py",
    "src/mlx2/runtime/segmented_rotating_kv.py",
    "src/mlx2/runtime/models/muse_glimmer.py",
)


def source_identity() -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", *IDENTITY_FILES)),
        "files": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in IDENTITY_FILES
        },
    }


def progressive_input_chunks(anchor: int, proposals, tile: int):
    """Partition fixed inputs exactly as progressive proposal verification."""

    proposals = tuple(int(token) for token in proposals)
    if type(anchor) is not int or anchor < 0:
        raise ValueError("anchor must be a nonnegative token id")
    if not proposals or type(tile) is not int or not 1 <= tile <= len(proposals):
        raise ValueError("tile must be within the proposal length")
    fixed = (anchor, *proposals)
    chunks = []
    proposal_offset = 0
    input_offset = 0
    while proposal_offset < len(proposals):
        remaining = len(proposals) - proposal_offset
        final = remaining <= tile
        proposals_here = remaining if final else tile
        inputs_here = proposals_here + int(final)
        chunks.append(fixed[input_offset : input_offset + inputs_here])
        proposal_offset += proposals_here
        input_offset += inputs_here
    if input_offset != len(fixed) or any(not chunk for chunk in chunks):
        raise AssertionError("progressive partition did not cover fixed inputs once")
    return tuple(chunks)


def _parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in text.split(",") if value.strip())
    if not values:
        raise ValueError("need at least one integer")
    return values


def _array_digest(mx, value) -> dict:
    width = int(value.dtype.size)
    unsigned = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}
    if width not in unsigned:
        raise ValueError(f"unsupported array element width {width}")
    bits = np.ascontiguousarray(np.asarray(value.view(unsigned[width])))
    return {
        "dtype": str(value.dtype),
        "shape": [int(dim) for dim in value.shape],
        "sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
    }


def _logical_cache_payload(cache):
    payload = []
    for entry in cache:
        keys = getattr(entry, "keys", None)
        values = getattr(entry, "values", None)
        offset = getattr(entry, "offset", None)
        if keys is None or values is None or not isinstance(offset, int):
            raise TypeError(f"unsupported target cache {type(entry).__name__}")
        temporal = getattr(entry, "_temporal_order", None)
        if callable(temporal):
            keys = temporal(keys)
            values = temporal(values)
        else:
            keys = keys[..., :offset, :]
            values = values[..., :offset, :]
        payload.append(
            {
                "class": f"{type(entry).__module__}.{type(entry).__qualname__}",
                "offset": offset,
                "max_size": getattr(entry, "max_size", None),
                "keep": getattr(entry, "keep", None),
                "keys": keys,
                "values": values,
            }
        )
    return payload


def _load(args):
    import mlx.core as mx

    if args.tiny:
        mx.set_default_device(mx.cpu)
        from mlx2.adapters.muse_glimmer_config import ModelArgs
        from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
        from mlx2.runtime.drafters.dflash2_config import DFlash2Config
        from mlx2.runtime.models.muse_glimmer import Model

        mx.random.seed(8)
        target = Model(
            ModelArgs(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=4,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                vocab_size=128,
                sliding_window=8,
                max_position_embeddings=2048,
            )
        )
        draft = DFlash2DraftModel(
            DFlash2Config(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                vocab_size=128,
                num_target_layers=4,
                target_layer_ids=[0, 3],
                conv_kernel_size=2,
                conv_group_size=2,
                selector_rank=4,
                selector_top_k=4,
                block_size=8,
                mask_token_id=127,
                max_position_embeddings=2048,
                sliding_window=8,
                layer_types=["sliding_attention"] * 2,
            )
        ).bind(target)
        target.eval()
        if args.target_verify_row_exact:
            target.configure_target_verify_row_exact(True)
        mx.eval(target.parameters(), draft.parameters())
        prompts = [[1 + (index * 7 + step) % 120 for step in range(12)] for index in range(2)]
        return target, draft, "tiny", prompts, {
            "target": "tiny-random",
            "draft": "tiny-random",
            "draft_block_size": 8,
        }

    if not args.i_own_the_gpu:
        raise SystemExit("real Metal execution requires --i-own-the-gpu")
    from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter

    adapter = MuseGlimmerAdapter(
        args.target,
        execution_policy={
            "draft_model": args.draft,
            "num_draft": args.num_draft,
            "proposal_composition": False,
            "target_verify_row_exact": args.target_verify_row_exact,
        },
    )
    prompts = [
        adapter.prompt_tokens({"messages": [{"role": "user", "content": text}]})
        for text in PROMPTS
    ]
    return (
        adapter.model,
        adapter.draft_model,
        adapter.identity["fingerprint"],
        prompts,
        {
            "target": args.target,
            "draft": args.draft,
            "fingerprint": adapter.identity["fingerprint"],
            "draft_block_size": int(adapter.draft_model.config.block_size),
        },
    )


def _forward_arm(batch, base_cache, chunks):
    import mlx.core as mx
    from paired_direct_ab import state_digest

    cache = batch._copy_reference_cache(base_cache)
    logits_parts = []
    feature_parts = []
    elapsed = 0.0
    for chunk in chunks:
        owner = batch._target_owner([cache])
        transaction = owner.begin(lengths=[len(chunk)])
        started = time.perf_counter()
        try:
            logits, features = batch.model.forward_with_taps(
                mx.array([chunk], dtype=mx.int32),
                transaction.caches,
                batch.layers,
            )
            mx.eval(logits, features)
            cache = transaction.commit(accepted_lengths=[len(chunk)])[0]
        except BaseException:
            if not transaction.closed:
                transaction.abort()
            raise
        elapsed += time.perf_counter() - started
        logits_parts.append(logits)
        feature_parts.append(features)
    logits = logits_parts[0] if len(logits_parts) == 1 else mx.concatenate(logits_parts, axis=1)
    features = feature_parts[0] if len(feature_parts) == 1 else mx.concatenate(feature_parts, axis=1)
    mx.eval(logits, features)
    return {
        "chunks": [len(chunk) for chunk in chunks],
        "launches": len(chunks),
        "diagnostic_forward_s": elapsed,
        "logits": logits,
        "features": features,
        "logits_digest": _array_digest(mx, logits),
        "features_digest": _array_digest(mx, features),
        "argmax": np.asarray(mx.argmax(logits, axis=-1)).reshape(-1).astype(int).tolist(),
        "raw_cache_digest": state_digest(cache),
        "logical_cache_digest": state_digest(_logical_cache_payload(cache)),
        "cache_offsets": [int(entry.offset) for entry in cache],
    }


def _comparison(mx, reference, candidate) -> dict:
    logits_ref = np.asarray(reference["logits"].astype(mx.float32))
    logits_got = np.asarray(candidate["logits"].astype(mx.float32))
    taps_ref = np.asarray(reference["features"].astype(mx.float32))
    taps_got = np.asarray(candidate["features"].astype(mx.float32))
    raw_ref = reference["raw_cache_digest"]
    raw_got = candidate["raw_cache_digest"]
    logical_ref = reference["logical_cache_digest"]
    logical_got = candidate["logical_cache_digest"]
    result = {
        "logits_bits_equal": reference["logits_digest"] == candidate["logits_digest"],
        "features_bits_equal": reference["features_digest"] == candidate["features_digest"],
        "argmax_equal": reference["argmax"] == candidate["argmax"],
        "cache_offsets_equal": reference["cache_offsets"] == candidate["cache_offsets"],
        "raw_cache_digest_equal": raw_ref.get("status") == raw_got.get("status") == "complete"
        and raw_ref.get("sha256") == raw_got.get("sha256"),
        "logical_cache_digest_equal": logical_ref.get("status") == logical_got.get("status") == "complete"
        and logical_ref.get("sha256") == logical_got.get("sha256"),
        "logits_max_abs": float(np.max(np.abs(logits_ref - logits_got))),
        "features_max_abs": float(np.max(np.abs(taps_ref - taps_got))),
    }
    result["strict_equal"] = all(
        result[key]
        for key in (
            "logits_bits_equal",
            "features_bits_equal",
            "argmax_equal",
            "cache_offsets_equal",
            "raw_cache_digest_equal",
            "logical_cache_digest_equal",
        )
    )
    return result


def _public_arm(arm: dict) -> dict:
    return {key: value for key, value in arm.items() if key not in {"logits", "features"}}


def _verification_arm(batch, lane, base_cache, proposals, proposal_laws, rng_state, tile):
    import mlx.core as mx
    from paired_direct_ab import state_digest
    from research.progressive_dflash_verify import (
        PreparedStage,
        fixed_verify,
        progressive_verify,
    )

    from mlx2.runtime.speculative_sampling import RequestRNG

    def prepare_stage(cache, inputs, history):
        owner = batch._target_owner([cache])
        transaction = owner.begin(lengths=[len(inputs)])
        try:
            logits, features = batch.model.forward_with_taps(
                mx.array([inputs], dtype=mx.int32), transaction.caches, batch.layers
            )
            mx.eval(logits, features)
            target_laws = tuple(
                batch._target_law(
                    lane,
                    logits[0, position],
                    [*history, *inputs[: position + 1]],
                    True,
                    None,
                )
                for position in range(len(inputs))
            )
            return PreparedStage(target_laws, features, transaction)
        except BaseException:
            if not transaction.closed:
                transaction.abort()
            raise

    cache = batch._copy_reference_cache(base_cache)
    started = time.perf_counter()
    if tile is None:
        result = fixed_verify(
            proposals,
            proposal_laws,
            cache=cache,
            anchor=int(lane.anchor),
            history=tuple(lane.history),
            rng=RequestRNG(state=rng_state),
            prepare_stage=prepare_stage,
        )
    else:
        result = progressive_verify(
            proposals,
            proposal_laws,
            cache=cache,
            anchor=int(lane.anchor),
            history=tuple(lane.history),
            rng=RequestRNG(state=rng_state),
            verification_tile=tile,
            prepare_stage=prepare_stage,
        )
    elapsed = time.perf_counter() - started
    features = (
        result.feature_parts[0]
        if len(result.feature_parts) == 1
        else mx.concatenate(result.feature_parts, axis=1)
    )
    next_cache = batch._copy_reference_cache(result.cache)
    next_logits, next_features = batch.model.forward_with_taps(
        mx.array([[result.emitted[-1]]], dtype=mx.int32), next_cache, batch.layers
    )
    mx.eval(features, next_logits, next_features)
    return {
        "emitted": list(result.emitted),
        "accepted": result.accepted,
        "committed_inputs": list(result.committed_inputs),
        "rng_state": result.rng_state,
        "target_rows": result.target_rows,
        "target_launches": result.target_launches,
        "rejected": result.rejected,
        "diagnostic_forward_s": elapsed,
        "features_digest": _array_digest(mx, features),
        "raw_cache_digest": state_digest(result.cache),
        "logical_cache_digest": state_digest(_logical_cache_payload(result.cache)),
        "cache_offsets": [int(entry.offset) for entry in result.cache],
        "next_target_logits_digest": _array_digest(mx, next_logits),
        "next_target_features_digest": _array_digest(mx, next_features),
        "next_target_raw_cache_digest": state_digest(next_cache),
        "next_target_logical_cache_digest": state_digest(
            _logical_cache_payload(next_cache)
        ),
    }


def _verification_comparison(reference, candidate):
    # A rejected transaction may leave different bytes above the authoritative
    # offset in an append-only allocation. Logical K/V plus the next-target
    # oracle are the revision-bound state; keep the raw digest as a diagnostic.
    exact_fields = (
        "emitted",
        "accepted",
        "committed_inputs",
        "rng_state",
        "features_digest",
        "logical_cache_digest",
        "cache_offsets",
        "next_target_logits_digest",
        "next_target_features_digest",
        "next_target_raw_cache_digest",
        "next_target_logical_cache_digest",
    )
    equality = {f"{name}_equal": reference[name] == candidate[name] for name in exact_fields}
    equality["raw_cache_digest_equal"] = (
        reference["raw_cache_digest"] == candidate["raw_cache_digest"]
    )
    equality["strict_equal"] = all(
        equality[f"{name}_equal"] for name in exact_fields
    )
    return equality


def _run_prompt(target, draft, binding, prompt, args, tiles):
    import mlx.core as mx

    from mlx2.runtime.external_speculative import (
        ExternalDraftBatchGenerator,
        _block_row,
    )
    from mlx2.runtime.sample_utils import LaneRNG

    batch = ExternalDraftBatchGenerator(
        target,
        draft_model=draft,
        binding=binding,
        completion_batch_size=1,
        prefill_step_size=args.prefill_step,
        num_draft=args.num_draft,
        stop_tokens=[],
    )
    try:
        uid = batch.insert(
            [prompt],
            max_tokens=[args.num_draft + 2],
            sampling_configs=[{"sampling_temp": 0}],
            lane_rngs=[LaneRNG(args.seed)],
        )[0]
        lane = batch.lanes[uid]
        while lane.anchor is None:
            batch._prefill(lane)
        block = batch._propose([lane])[0]
        proposals, proposal_laws = _block_row(block, int(target.args.vocab_size))
        if len(proposals) != args.num_draft:
            raise RuntimeError("DFlash did not produce the requested proposal length")
        fixed_inputs = (int(lane.anchor), *map(int, proposals))
        base_cache = batch._copy_reference_cache(lane.cache)
        layouts = {
            "ordinary_rows": tuple((token,) for token in fixed_inputs),
            "fixed": (fixed_inputs,),
        }
        for tile in tiles:
            layouts[f"tile_{tile}"] = progressive_input_chunks(
                int(lane.anchor), proposals, tile
            )
        arms = {
            name: _forward_arm(batch, base_cache, chunks)
            for name, chunks in layouts.items()
        }
        reference = arms["ordinary_rows"]
        comparisons = {
            name: _comparison(mx, reference, arm)
            for name, arm in arms.items()
            if name != "ordinary_rows"
        }
        verification = None
        if args.verify_proposal:
            rng_state = lane.rng.snapshot()
            verification_arms = {
                "fixed": _verification_arm(
                    batch,
                    lane,
                    base_cache,
                    tuple(map(int, proposals)),
                    proposal_laws,
                    rng_state,
                    None,
                )
            }
            for tile in tiles:
                verification_arms[f"tile_{tile}"] = _verification_arm(
                    batch,
                    lane,
                    base_cache,
                    tuple(map(int, proposals)),
                    proposal_laws,
                    rng_state,
                    tile,
                )
            verification = {
                "arms": verification_arms,
                "comparisons_vs_fixed": {
                    name: _verification_comparison(verification_arms["fixed"], arm)
                    for name, arm in verification_arms.items()
                    if name != "fixed"
                },
            }
        return {
            "prompt_tokens": len(prompt),
            "anchor": int(lane.anchor),
            "proposal_tokens": list(map(int, proposals)),
            "arms": {name: _public_arm(arm) for name, arm in arms.items()},
            "comparisons_vs_ordinary_rows": comparisons,
            "proposal_verification": verification,
        }
    finally:
        batch.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target",
        default=str(Path.home() / "mlx-models/Muse-Glimmer-30B-mlx-4bit"),
    )
    parser.add_argument(
        "--draft",
        default=str(Path.home() / "mlx-models/Muse-Glimmer-30B-DFlash2"),
    )
    parser.add_argument("--num-draft", type=int, default=15)
    parser.add_argument("--tiles", default="3,4")
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--prefill-step", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=919)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--target-verify-row-exact", action="store_true")
    parser.add_argument("--verify-proposal", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    tiles = _parse_ints(args.tiles)
    target, draft, binding, prompts, identity = _load(args)
    block = int(draft.config.block_size)
    if not 1 <= args.num_draft < block:
        raise SystemExit(
            f"num_draft must be within 1..{block - 1} for trained block {block}"
        )
    if any(not 1 <= tile <= args.num_draft for tile in tiles):
        raise SystemExit("every tile must be within the proposal length")
    selected = prompts[: args.prompts]
    if not selected:
        raise SystemExit("need at least one prompt")
    results = [
        _run_prompt(target, draft, binding, prompt, args, tiles)
        for prompt in selected
    ]
    payload = {
        "schema": "mlx2.dflash-split-shape-probe.v1",
        "source": source_identity(),
        "identity": identity,
        "args": {
            "num_draft": args.num_draft,
            "tiles": list(tiles),
            "prompts": len(selected),
            "seed": args.seed,
            "tiny": args.tiny,
            "target_verify_row_exact": args.target_verify_row_exact,
            "verify_proposal": args.verify_proposal,
        },
        "state": {
            "probe_implemented": True,
            "serving_route_implemented": False,
            "qualified": False,
            "selected": False,
            "apcv2_publication": False,
            "performance_claim": False,
        },
        "results": results,
    }
    comparisons = [
        comparison
        for result in results
        for comparison in result["comparisons_vs_ordinary_rows"].values()
    ]
    verification_comparisons = [
        comparison
        for result in results
        if result["proposal_verification"] is not None
        for comparison in result["proposal_verification"][
            "comparisons_vs_fixed"
        ].values()
    ]
    payload["verdict"] = (
        "strict_equal"
        if comparisons
        and all(row["strict_equal"] for row in comparisons)
        and (
            not args.verify_proposal
            or verification_comparisons
            and all(row["strict_equal"] for row in verification_comparisons)
        )
        else "counterexample"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["verdict"] == "strict_equal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
