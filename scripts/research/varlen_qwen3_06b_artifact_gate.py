"""Offline Qwen3-0.6B ordinary/paged parity gate; dry-run by default.

The opt-in GPU mode needs fresh GPU ownership, both lock guards, no foreign
service owner, and a matching native extension. It is a bounded research gate,
not serving qualification. Dry-run only reads local files and metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)
MLX_LM_SOURCE = Path.home() / "Desktop/mlx-uag/mlx-lm-unified"
CONFIG_SHA256 = "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
WEIGHTS_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
MLX_LM_REVISION = "1104ced19ed98800bdaf4ebcdca14bbdeb597c23"
QWEN3_SOURCE_SHA256 = "9facb9e667b7372771d4c68a54213eaa0b47fe0d0443c4f73328520300d3b66d"
POOL_PAGES = 128
ARENA_BYTES = POOL_PAGES * 8 * 128 * 2 * 64 * 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(snapshot: Path, mlx_lm_source: Path) -> dict:
    config_file = snapshot / "config.json"
    weight_file = snapshot / "model.safetensors"
    tokenizer_file = snapshot / "tokenizer.json"
    source_file = mlx_lm_source / "mlx_lm/models/qwen3.py"
    for path in (config_file, weight_file, tokenizer_file, source_file):
        if not path.is_file():
            raise FileNotFoundError(f"required offline artifact is missing: {path}")
    actual = {
        "config": sha256(config_file),
        "weights": sha256(weight_file),
        "qwen3_source": sha256(source_file),
    }
    if actual != {"config": CONFIG_SHA256, "weights": WEIGHTS_SHA256,
                  "qwen3_source": QWEN3_SOURCE_SHA256}:
        raise ValueError(f"pinned artifact/source hashes changed: {actual}")
    revision = subprocess.check_output(
        ["git", "-C", str(mlx_lm_source), "rev-parse", "HEAD"], text=True).strip()
    if revision != MLX_LM_REVISION:
        raise ValueError(f"offline mlx_lm source revision changed: {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(mlx_lm_source), "status", "--porcelain",
         "--untracked-files=no"], text=True)
    if dirty:
        raise ValueError("offline mlx_lm source has modified tracked files")
    config = json.loads(config_file.read_text())
    expected = {"model_type": "qwen3", "hidden_size": 1024,
                "num_hidden_layers": 28, "num_attention_heads": 16,
                "num_key_value_heads": 8, "head_dim": 128,
                "torch_dtype": "bfloat16"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint is not the pinned dense Qwen3 geometry")
    if config.get("rope_scaling") is not None or config.get("num_experts", 0):
        raise ValueError("unsupported checkpoint attention capability")
    from safetensors import safe_open

    with safe_open(str(weight_file), framework="pt", device="cpu") as safe:
        keys = safe.keys()
        dtypes = Counter(safe.get_slice(key).get_dtype() for key in keys)
        tensor_count = len(keys)
    if dtypes != {"BF16": tensor_count} or tensor_count < 300:
        raise ValueError("checkpoint weights are not the expected all-BF16 artifact")
    weight_bytes = weight_file.stat().st_size
    return {
        "schema": "mlx2.varlen-qwen3-06b-artifact-gate.v1",
        "snapshot": str(snapshot), "config_sha256": actual["config"],
        "weights_sha256": actual["weights"], "weight_bytes": weight_bytes,
        "mlx_lm_revision": revision, "qwen3_source_sha256": actual["qwen3_source"],
        "tensor_count": tensor_count, "weight_dtypes": dict(dtypes),
        "comparison_dtype": "float16 for both arms from one immutable cast model",
        "planned_cases": ["B1: 63-token prefill, two one-token continuations across 64",
                          "B2: 63/65-token ragged prefill, two one-token continuations"],
        "arena_bytes_two_planes": ARENA_BYTES,
        "source_file_cast_overlap_estimate_bytes": 2 * weight_bytes + ARENA_BYTES,
        "recommended_free_memory_bytes": 5 * (1 << 30),
        "gpu_time_cap_seconds": 300,
        "requires": ["fresh GPU FIFO ownership", "both GPU lock guards",
                     "no foreign service owner", "matching native extension build"],
    }


def gpu_gate(snapshot: Path, mlx_lm_source: Path, receipt: dict) -> dict:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(mlx_lm_source))
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load

    from mlx2.adapters.qwen3_paged_candidate import PackedLane, Qwen3PackedCandidate
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend, PagedKVWriteOwner
    from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend

    if not mx.metal.is_available():
        raise RuntimeError("Qwen3 artifact parity gate requires MLX GPU")
    mx.set_default_device(mx.gpu)
    stream = mx.default_stream(mx.gpu)
    model, _ = load(str(snapshot), lazy=True)
    if model.args.model_type != "qwen3" or len(model.layers) != 28:
        raise ValueError("offline loader did not return the expected Qwen3 model")
    # One immutable in-memory fp16 weight set serves both arms. No BF16 paged
    # read is claimed; reject any parameter that cannot be cast to fp16.
    model.apply(lambda p: p.astype(mx.float16))
    mx.eval(model.parameters())
    if any(p.dtype != mx.float16 for _, p in tree_flatten(model.parameters())):
        raise ValueError("both comparison arms require entirely fp16 weights")
    model.eval()

    class PagedModel:
        args = type("Args", (), {"model_type": "qwen3", "num_experts": 0,
                                  "rope_scaling": None, "head_dim": 128,
                                  "num_attention_heads": 16,
                                  "num_key_value_heads": 8})()

        def __init__(self, ordinary):
            self.ordinary = ordinary
            self.layers = ordinary.model.layers

        def paged_embed(self, tokens):
            return self.ordinary.model.embed_tokens(mx.array(tokens, dtype=mx.int32))

        def paged_project(self, index, hidden, counts, offsets):
            block = self.layers[index]
            attn = block.self_attn
            x = block.input_layernorm(hidden)
            q = attn.q_norm(attn.q_proj(x).reshape(-1, 16, 128))
            k = attn.k_norm(attn.k_proj(x).reshape(-1, 8, 128))
            v = attn.v_proj(x).reshape(-1, 8, 128)
            q_rows, k_rows = [], []
            begin = 0
            for count, offset in zip(counts, offsets):
                q_slice = q[begin:begin + count].transpose(1, 0, 2)[None]
                k_slice = k[begin:begin + count].transpose(1, 0, 2)[None]
                q_rows.append(attn.rope(q_slice, offset=offset)[0].transpose(1, 0, 2))
                k_rows.append(attn.rope(k_slice, offset=offset)[0].transpose(1, 0, 2))
                begin += count
            return (mx.contiguous(mx.concatenate(q_rows)),
                    mx.contiguous(mx.concatenate(k_rows)), mx.contiguous(v))

        def paged_finish_layer(self, index, hidden, attended):
            block = self.layers[index]
            projected = block.self_attn.o_proj(attended.reshape(-1, 16 * 128))
            x = hidden + projected
            return x + block.mlp(block.post_attention_layernorm(x))

        def paged_logits(self, hidden):
            return self.ordinary.model.embed_tokens.as_linear(
                self.ordinary.model.norm(hidden))

    def ordinary_last(tokens):
        output = model(mx.array([tokens], dtype=mx.int32))
        mx.eval(output)
        return output[0, -1]

    def compare(actual, reference, label):
        import numpy as np

        mx.eval(actual, reference)
        a = np.array(actual, dtype=np.float32)
        b = np.array(reference, dtype=np.float32)
        error = a - b
        nrms = float(np.sqrt(np.mean(error * error)) /
                     max(float(np.sqrt(np.mean(b * b))), 1e-12))
        top1 = int(np.argmax(a)) == int(np.argmax(b))
        if nrms > 0.02 or not top1:
            raise AssertionError(f"{label}: fp16 paged/ordinary logits differ: nrms={nrms}, top1={top1}")
        return {"label": label, "normalized_rms": nrms,
                "max_abs": float(np.max(np.abs(error))), "top1_equal": top1}

    results = []
    for name, prompts in (("B1", ((100,) * 63,)),
                          ("B2", ((100,) * 63, (101,) * 65))):
        pool = PagedKVPool(POOL_PAGES)
        profile = TokenKVProfile(8, 128, "float16")
        native = NativeWriteBackend(ARENA_BYTES // 2, stream, permit_candidate=True)
        writer = PagedKVWriteOwner(pool, native, page_bytes=profile.page_bytes,
                                   permit_candidate=True)
        backend = NativeQwen3PagedBackend(writer, permit_candidate=True,
                                           timeout_s=15)
        lanes = tuple(PackedLane(tokens, tuple(
            PagedKVTokenOwner(writer, profile, permit_candidate=True)
            for _ in range(28))) for tokens in prompts)
        candidate = Qwen3PackedCandidate(PagedModel(model), backend)
        logits, _ = candidate.forward(lanes, permit_candidate=True)
        reference = [ordinary_last(tokens) for tokens in prompts]
        row = 0
        for lane_index, tokens in enumerate(prompts):
            results.append(compare(logits[row + len(tokens) - 1], reference[lane_index],
                                   f"{name}-prefill-lane{lane_index}"))
            row += len(tokens)
        current = list(prompts)
        for step in (1, 2):
            next_tokens = [int(mx.argmax(ref).item()) for ref in reference]
            current = [seq + (token,) for seq, token in zip(current, next_tokens)]
            suffix = tuple(PackedLane((token,), lane.layers)
                           for lane, token in zip(lanes, next_tokens))
            logits, _ = candidate.forward(suffix, permit_candidate=True)
            reference = [ordinary_last(seq) for seq in current]
            for lane_index, ref in enumerate(reference):
                results.append(compare(logits[lane_index], ref,
                                       f"{name}-continuation{step}-lane{lane_index}"))
        expected_offsets = [[len(seq)] * 28 for seq in current]
        actual_offsets = [[owner.offset for owner in lane.layers] for lane in lanes]
        if actual_offsets != expected_offsets or writer.ledger.pending_count:
            raise AssertionError(f"{name}: page boundary publication or terminal read failed")
        for lane in lanes:
            for owner in lane.layers:
                owner.close()
        if pool.free_count != pool.capacity:
            raise AssertionError(f"{name}: pool did not retire after closure")
    return {**receipt, "mode": "gpu-artifact-parity", "gpu_executed": True,
            "comparisons": results, "all_page_boundaries_and_leases_closed": True,
            "serving_route_selected": False, "qualification": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--mlx-lm-source", type=Path, default=MLX_LM_SOURCE)
    parser.add_argument("--execute-gpu", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    receipt = preflight(args.snapshot, args.mlx_lm_source)
    if args.execute_gpu:
        signal.alarm(300)
        receipt = gpu_gate(args.snapshot, args.mlx_lm_source, receipt)
    else:
        receipt.update(mode="dry-run", gpu_executed=False, model_loaded=False,
                       qualification=False, serving_route_selected=False)
    receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
