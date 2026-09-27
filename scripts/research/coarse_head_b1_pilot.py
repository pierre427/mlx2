#!/usr/bin/env python3
"""Research-only B1 matched MTP-head capture for oMLX #3958.

This deliberately emits a pilot receipt, not the 36-row v2 admission capture.
It never changes production source or the model artifact. Metal mode requires
an externally held GPU lease and host lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_coarse_head_preflight import (  # noqa: E402
    MLX2_CONTRACT_PATHS,
    OMLX_REV,
    teacher_forced_row_sha256,
)

MODEL = Path("~/mlx-models/Qwen3.8-27B-oQ4e-mtp")
SAMPLING = {"temperature": 0.8, "top_p": 0.95, "top_k": 20,
            "min_p": 0, "processors": [], "xtc_probability": 0,
            "accept_rule": "residual"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_receipt(model_path: Path | None) -> dict:
    receipt = {
        "mlx2_source_root": str(ROOT),
        "mlx2_source_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "mlx2_contract_sha256": {name: sha(ROOT / name) for name in MLX2_CONTRACT_PATHS},
        "capture_code_path": str(Path(__file__).resolve()),
        "capture_code_sha256": sha(Path(__file__)),
        "omlx_source_revision": OMLX_REV,
    }
    if model_path is not None:
        receipt["artifact_config_sha256"] = sha(model_path / "config.json")
        receipt["artifact_index_sha256"] = sha(model_path / "model.safetensors.index.json")
    return receipt


def tiny_model():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    cfg = TextModelArgs(
        model_type="qwen3_5", hidden_size=128, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=3,
        full_attention_interval=2, mtp_num_hidden_layers=1,
        partial_rotary_factor=0.5, rope_parameters=None,
        max_position_embeddings=1 << 20,
    )
    mx.random.seed(7)
    model = TextModel(cfg)
    model.eval()
    mx.eval(model.parameters())
    lang = getattr(model, "language_model", model)
    lang.lm_head = nn.QuantizedLinear.from_linear(
        lang.lm_head, group_size=64, bits=4)
    mx.eval(lang.lm_head.parameters())
    return model, mx.arange(16, dtype=mx.uint32), 128


def head_and_fc(model):
    import mlx.nn as nn
    lang = getattr(model, "language_model", model)
    head = lang.lm_head
    fc = lang.mtp.fc
    if (not isinstance(head, nn.QuantizedLinear)
            or getattr(head, "mode", "affine") != "affine"
            or int(head.bits) != 4 or int(head.group_size) != 64
            or type(fc) is not nn.Linear or "bias" in fc):
        raise RuntimeError("expected untied affine q4/group64 head and dense bias-free MTP fc")
    if int(head.weight.shape[0]) < 64 or int(fc.weight.shape[1]) % 64:
        raise RuntimeError("head or MTP fc is incompatible with the pinned coarse path")
    return lang, head, fc


def make_coarse_head(head):
    import mlx.core as mx
    import mlx.nn as nn
    rows = int(head.weight.shape[0])
    width = int(head.weight.shape[1]) * 32 // int(head.bits)
    if width % 128:
        raise RuntimeError("head width is not divisible by coarse group size 128")
    chunks = []
    for start in range(0, rows, 16384):
        stop = min(start + 16384, rows)
        weight = mx.dequantize(head.weight[start:stop], head.scales[start:stop],
                               head.biases[start:stop], group_size=64, bits=4)
        chunk = mx.quantize(weight, group_size=128, bits=3)
        mx.eval(*chunk)
        chunks.append(chunk)
    coarse = nn.QuantizedLinear(width, rows, bias=False, group_size=128, bits=3)
    coarse.weight, coarse.scales, coarse.biases = (
        mx.concatenate([chunk[i] for chunk in chunks], axis=0) for i in range(3))
    mx.eval(coarse.parameters())
    digest = hashlib.sha256()
    for part in (coarse.weight, coarse.scales, coarse.biases):
        raw = part.view(mx.uint16) if part.dtype == mx.bfloat16 else part
        digest.update(np.asarray(raw).tobytes())
    return coarse, digest.hexdigest()


def exact_rescore(head, post, ids):
    import mlx.core as mx
    return mx.quantized_matmul(
        post[:, -1, :], head.weight[ids], head.scales[ids], head.biases[ids],
        transpose=True, group_size=64, bits=4)[0].astype(mx.float32)


def cache_offsets(caches):
    offsets = [int(cache.offset) for cache in caches if hasattr(cache, "offset")]
    if not offsets:
        raise RuntimeError("cache group has no offset-bearing plane")
    if len(set(offsets)) != 1:
        raise RuntimeError(f"cache layer offsets diverged: {offsets}")
    return offsets[0]


def seed_digest(seed_h) -> str:
    import mlx.core as mx
    return hashlib.sha256(np.asarray(seed_h.astype(mx.float32)).tobytes()).hexdigest()


def assert_checkpoint(checkpoint, expected_offset: int, expected_seed: str):
    target = cache_offsets(checkpoint["target_cache"])
    draft = cache_offsets(checkpoint["mtp_state"][0])
    if (checkpoint.get("committed_only") is not True or
            checkpoint["covered_tokens"] != expected_offset or
            target != expected_offset or draft != target - 1 or
            seed_digest(checkpoint["mtp_state"][1]) != expected_seed):
        raise RuntimeError("committed target/draft checkpoint changed")


def clone_checkpoint(checkpoint):
    from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors
    target, sidecar, _ = snapshot_prompt_cache_descriptors(
        checkpoint["target_cache"], checkpoint["mtp_state"])
    return target, sidecar


def prepare(model, prefix, seed: int):
    from mlx2.runtime.hybrid_speculative import (
        capture_self_mtp_checkpoint, prepare_self_mtp_lane)
    from mlx2.runtime.sample_utils import LaneRNG
    detached, _ = prepare_self_mtp_lane(
        prefix, model, uid=0, max_tokens=8, prompt_cache=None, mtp_state=None,
        lane_rng=LaneRNG(seed), num_draft=3, sampling_temp=0.8,
        sampling_top_p=0.95, sampling_top_k=20, sampling_min_p=0.0,
        accept_rule="residual", logits_processors=[], prefill_step_size=512,
        share_qsa_indices=False)
    checkpoint = capture_self_mtp_checkpoint(
        detached.caches.target, (detached.caches.draft, detached.lane.seed_h))
    if checkpoint is None:
        raise RuntimeError("could not capture committed MTP checkpoint")
    expected_seed = seed_digest(checkpoint["mtp_state"][1])
    assert_checkpoint(checkpoint, int(prefix.size), expected_seed)
    return int(detached.lane.cur), checkpoint, expected_seed


def target_rows(model, checkpoint, anchor: int, count: int):
    import mlx.core as mx
    from mlx2.runtime.hybrid_speculative import (
        _finalize_self_mtp_cache_group, _mtp_backbone,
        _prepare_self_mtp_cache_group)
    target, _ = clone_checkpoint(checkpoint)
    forced = []
    logits = []
    for j in range(count):
        token = anchor if j == 0 else forced[-1]
        if cache_offsets(target) != checkpoint["covered_tokens"] + j:
            raise RuntimeError("target verify offset changed before row")
        _prepare_self_mtp_cache_group(target, (1,), (0,))
        try:
            head_h, _ = _mtp_backbone(model, mx.array([[token]], mx.uint32), target)
            row = model.logits(head_h)[0, -1].astype(mx.float32)
            mx.eval(row)
        finally:
            _finalize_self_mtp_cache_group(target)
        logits.append(np.asarray(row))
        if j + 1 < count:
            forced.append(int(mx.argmax(row).item()))
    return forced, logits


def draft_rows(model, checkpoint, anchor: int, forced: list[int],
               coarse=None, head=None):
    import mlx.core as mx
    from mlx2.runtime.hybrid_speculative import _finalize_self_mtp_cache_group, _prepare_self_mtp_cache_group
    _, (draft, seed_h) = clone_checkpoint(checkpoint)
    exact_rows, candidate_ids, candidate_scores = [], [], []
    inputs = [anchor, *forced]
    for j, token in enumerate(inputs):
        if cache_offsets(draft) != checkpoint["covered_tokens"] - 1 + j:
            raise RuntimeError("draft offset changed before row")
        _prepare_self_mtp_cache_group(draft, (1,), (0,))
        try:
            row_logits, post = model.mtp_step(seed_h, mx.array([[token]], mx.uint32), draft)
            row_logits = row_logits[0, -1].astype(mx.float32)
            if coarse is not None:
                approx = coarse(post[:, -1, :]).astype(mx.float32)
                ids = mx.argsort(-approx[0])[:64]
                scores = exact_rescore(head, post, ids)
                mx.eval(row_logits, post, ids, scores)
                candidate_ids.append(np.asarray(ids).astype(np.int32))
                candidate_scores.append(np.asarray(scores))
            else:
                mx.eval(row_logits, post)
            exact_rows.append(np.asarray(row_logits))
            seed_h = post
        finally:
            _finalize_self_mtp_cache_group(draft)
    return exact_rows, candidate_ids, candidate_scores


def capture(model, prefix, seed: int, head, fc, coarse):
    import mlx.core as mx
    import mlx.nn as nn
    lang = getattr(model, "language_model", model)
    anchor, dense_checkpoint, dense_seed = prepare(model, prefix, seed)
    forced, target = target_rows(model, dense_checkpoint, anchor, 3)
    assert_checkpoint(dense_checkpoint, int(prefix.size), dense_seed)
    dense, _, _ = draft_rows(model, dense_checkpoint, anchor, forced)
    assert_checkpoint(dense_checkpoint, int(prefix.size), dense_seed)
    lang.mtp.fc = nn.QuantizedLinear.from_linear(fc, group_size=64, bits=4)
    mx.eval(lang.mtp.fc.parameters())
    if not isinstance(lang.mtp.fc, nn.QuantizedLinear):
        raise RuntimeError("q4 MTP fc conversion did not engage")
    q4_anchor, q4_checkpoint, q4_seed = prepare(model, prefix, seed)
    if q4_anchor != anchor:
        raise RuntimeError("target sampled anchor changed with MTP fc quantization")
    q4, ids, scores = draft_rows(model, q4_checkpoint, anchor, forced,
                                  coarse=coarse, head=head)
    assert_checkpoint(q4_checkpoint, int(prefix.size), q4_seed)
    if not (len(dense) == len(target) == len(q4) == len(ids) == len(scores) == 3):
        raise RuntimeError("one or more model arms failed to execute every row")
    return anchor, forced, dense, q4, target, ids, scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tiny-cpu", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.tiny_cpu == args.i_own_the_gpu:
        parser.error("choose exactly one of --tiny-cpu or --i-own-the-gpu")
    if args.context < 3 or args.seed < 0:
        parser.error("context >= 3 and seed >= 0 required")
    if args.out.exists() or args.out.with_suffix(".json").exists():
        parser.error("output already exists; choose a new path")
    receipt = source_receipt(None if args.tiny_cpu else args.model)
    import mlx.core as mx
    if args.tiny_cpu:
        mx.set_default_device(mx.cpu)
        model, prefix, vocab = tiny_model()
    else:
        from mlx2.adapters.registry import resolve_adapter
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            raise RuntimeError("Metal GPU unavailable")
        adapter = resolve_adapter(args.model, mtp=True)(str(args.model))
        model = adapter.model
        vocab = int(adapter.tokenizer.vocab_size)
        encoded = adapter.tokenizer.encode(
            "Explain how an exact speculative draft is checked against the target model. ")
        if not encoded or max(encoded) >= vocab:
            raise RuntimeError("tokenizer yielded invalid prompt tokens")
        repeats = (args.context + len(encoded) - 1) // len(encoded)
        prefix = mx.array((encoded * repeats)[:args.context], mx.uint32)
    lang, head, fc = head_and_fc(model)
    coarse, coarse_digest = make_coarse_head(head)
    anchor, forced, dense, q4, target, ids, scores = capture(
        model, prefix, args.seed, head, fc, coarse)
    prefix_ids = np.asarray(prefix).astype(np.uint32)
    histories = [np.concatenate((prefix_ids, np.asarray([anchor, *forced[:j]], dtype=np.uint32)))
                 for j in range(3)]
    meta = {
        "context_length": np.full(3, prefix_ids.size, np.int64),
        "lane_id": np.zeros(3, np.int64),
        "seed": np.full(3, args.seed, np.int64),
        "draft_position": np.arange(3, dtype=np.int64),
        "target_offset": np.full(3, prefix_ids.size, np.int64),
        "draft_offset": np.full(3, prefix_ids.size - 1, np.int64),
    }
    row_digests = np.array([
        teacher_forced_row_sha256(tokens, **{key: int(val[j]) for key, val in meta.items()})
        for j, tokens in enumerate(histories)], dtype="S64")
    arrays = {
        "baseline_dense_fc_logits": np.asarray(dense, dtype=np.float32),
        "draft_exact_logits": np.asarray(q4, dtype=np.float32),
        "target_logits": np.asarray(target, dtype=np.float32),
        "coarse_candidate_ids": np.asarray(ids, dtype=np.int32),
        "candidate_exact_logits": np.asarray(scores, dtype=np.float32),
        "row_token_ids": np.concatenate(histories),
        "row_token_offsets": np.cumsum([0, *map(len, histories)], dtype=np.int64),
        **meta,
        **{name: row_digests for name in (
            "dense_fc_row_sha256", "q4_fc_full_row_sha256",
            "q4_fc_coarse_row_sha256", "target_row_sha256")},
    }
    if (any(not np.all(np.isfinite(arrays[name])) for name in
            ("baseline_dense_fc_logits", "draft_exact_logits", "target_logits",
             "candidate_exact_logits")) or
            any(not np.allclose(q4[j][ids[j]], scores[j], atol=0.125, rtol=0)
                for j in range(3))):
        raise RuntimeError("nonfinite logits or exact head rescore mismatch")
    if any(int(np.asarray(histories[j][-1])) != (anchor if j == 0 else forced[j-1])
           for j in range(3)):
        raise RuntimeError("recorded teacher-forced tokens differ from model inputs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    receipt.update({
        "schema": "mlx2-coarse-head-pilot-v1",
        "capture_kind": "matched_model_pilot" if args.i_own_the_gpu else "synthetic_test",
        "route": "self_mtp_draft",
        "alignment": "teacher_forced_same_prefix_and_draft_tokens",
        "sampling": SAMPLING,
        "artifact": str(args.model) if args.i_own_the_gpu else None,
        "capture_sha256": sha(args.out),
        "coarse_head_parameters_sha256": coarse_digest,
        "context_length": int(prefix_ids.size),
        "seed": args.seed,
        "anchor_token": anchor,
        "forced_draft_tokens": forced,
        "vocab_size": vocab,
        "mechanism_counts": {name: 3 for name in (
            "dense_fc_full_head_steps", "q4_fc_full_head_steps",
            "q4_fc_coarse_head_steps", "target_verify_steps",
            "committed_checkpoint_restore_checks")},
        "limits": ["B1 three-row pilot only; not the 36-row v2 admission capture",
                   "No serving, throughput, or quality qualification"],
    })
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"capture": str(args.out), "manifest": str(args.out.with_suffix('.json')),
                      "rows": 3, "coarse_head_parameters_sha256": coarse_digest}))


if __name__ == "__main__":
    main()
