"""Offline Qwen3-0.6B complete-request driver for the default-off price cell.

The factory loads one immutable fp16 model outside the clock.  Every call then
allocates, admits, executes, samples, synchronizes, and retires fresh request
state.  No result is a serving route or qualification claim.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from pathlib import Path

from varlen_pack_price_bench import sha256


def _resident_bytes() -> int:
    # macOS ru_maxrss is bytes and is a conservative process high-water mark.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class _PagedModel:
    def __init__(self, ordinary, mx):
        self.ordinary, self.mx = ordinary, mx
        self.layers = ordinary.model.layers
        self.args = type("Args", (), {"model_type": "qwen3", "num_experts": 0,
                                      "rope_scaling": None, "head_dim": 128,
                                      "num_attention_heads": 16,
                                      "num_key_value_heads": 8})()

    def paged_embed(self, tokens):
        return self.ordinary.model.embed_tokens(self.mx.array(tokens, dtype=self.mx.int32))

    def paged_project(self, index, hidden, counts, offsets):
        mx = self.mx
        attn = self.layers[index].self_attn
        x = self.layers[index].input_layernorm(hidden)
        q = attn.q_norm(attn.q_proj(x).reshape(-1, 16, 128))
        k = attn.k_norm(attn.k_proj(x).reshape(-1, 8, 128))
        v = attn.v_proj(x).reshape(-1, 8, 128)
        q_rows, k_rows = [], []
        begin = 0
        for count, offset in zip(counts, offsets):
            q_rows.append(attn.rope(q[begin:begin + count].transpose(1, 0, 2)[None],
                                    offset=offset)[0].transpose(1, 0, 2))
            k_rows.append(attn.rope(k[begin:begin + count].transpose(1, 0, 2)[None],
                                    offset=offset)[0].transpose(1, 0, 2))
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


class Qwen3RequestDriver:
    model_layers = 28

    def __init__(self, manifest: dict):
        source = Path(manifest["mlx_lm_source"]).resolve()
        revision = manifest["mlx_lm_revision"]
        digest = manifest["qwen3_source_sha256"]
        if (not source.is_dir() or type(revision) is not str or len(revision) != 40 or
                type(digest) is not str or len(digest) != 64 or
                subprocess.check_output(["git", "rev-parse", "HEAD"],
                                        cwd=source, text=True).strip() != revision or
                subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=source).strip() or
                sha256(source / "mlx_lm/models/qwen3.py") != digest):
            raise RuntimeError("external Qwen3 model source differs")
        sys.path.insert(0, str(source))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import mlx.core as mx
        from mlx.utils import tree_flatten
        import mlx_lm.models.qwen3 as loaded_qwen3
        from mlx_lm.utils import load

        if Path(loaded_qwen3.__file__).resolve() != source / "mlx_lm/models/qwen3.py":
            raise RuntimeError("imported Qwen3 model source differs")

        artifact = json.loads(Path(manifest["paths"]["artifact"]).read_text())
        model, _ = load(artifact["root"], lazy=True)
        if (model.args.model_type != "qwen3" or len(model.model.layers) != 28 or
                model.args.num_key_value_heads != 8 or model.args.head_dim != 128):
            raise RuntimeError("artifact did not load as reviewed dense Qwen3-0.6B")
        model.apply(lambda p: p.astype(mx.float16))
        mx.eval(model.parameters())
        if any(p.dtype != mx.float16 for _, p in tree_flatten(model.parameters())):
            raise RuntimeError("model weights are not all fp16")
        model.eval()
        self.model, self.mx = model, mx
        self._closed = False

    @staticmethod
    def _requests(reserved, prefill, contexts):
        if contexts != (63, 65, 129):
            raise ValueError("exact context identity differs")
        from varlen_pack_price_bench import CASES
        if reserved not in CASES or prefill not in (0, 1, 3, 17):
            raise ValueError("unreviewed pack shape")
        rows = [count for _phase, count in reserved]
        if prefill:
            rows.append(prefill)
        if not rows or any(type(count) is not int or count < 1 for count in rows):
            raise ValueError("invalid request rows")
        prompts = tuple((100 + index,) * length for index, length in enumerate(contexts))
        suffixes = tuple((200 + index,) * count for index, count in enumerate(rows))
        return prompts, suffixes

    def run_request(self, arm, reserved, prefill, contexts):
        if self._closed or arm not in ("ordinary", "paged"):
            raise RuntimeError("driver closed or invalid arm")
        prompts, suffixes = self._requests(reserved, prefill, contexts)
        if arm == "ordinary":
            return self._ordinary(prompts, suffixes)
        return self._paged(prompts, suffixes)

    def _ordinary(self, prompts, suffixes):
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache
        queue = list(enumerate(suffixes))
        output = []
        admitted = bool(queue) and sum(len(s) for s in suffixes) <= 64
        if not admitted:
            raise RuntimeError("ordinary request admission failed")
        caches = []
        try:
            for prompt in prompts:
                cache = make_prompt_cache(self.model.model)
                seed = self.model(mx.array([prompt], dtype=mx.int32), cache=cache)
                mx.eval(seed)
                caches.append(cache)
            for index, suffix in queue:
                # Ordinary cached continuation through all layers and the
                # same deterministic argmax sampler used in the paged arm.
                logits = self.model(mx.array([suffix], dtype=mx.int32),
                                    cache=caches[index])
                sample = mx.argmax(logits[0, -1]).item()
                output.append(int(sample))
            mx.synchronize(mx.default_stream(mx.gpu))
        finally:
            caches.clear()
            queue.clear()
        return {"arm": "ordinary", "admitted": admitted,
                "cache_transaction_published": True, "sampled_output": True,
                "synchronized": True, "request_state_released": True,
                "queue_state_released": not queue, "model_layers": self.model_layers,
                "output_token_ids": output, "peak_resident_bytes": _resident_bytes()}

    def _paged(self, prompts, suffixes):
        mx = self.mx
        from mlx2.adapters.qwen3_paged_candidate import PackedLane, Qwen3PackedCandidate
        from mlx2.runtime.paged_kv_pool import PagedKVPool
        from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
        from mlx2.runtime.paged_kv_write import NativeWriteBackend, PagedKVWriteOwner
        from mlx2.runtime.paged_native_atomic_owner import NativeAtomicRequestOwner
        from mlx2.runtime.paged_pack_scheduler import PagedPackDecision
        from mlx2.runtime.paged_request_transaction import (
            CandidateRequest, execute_paged_request,
        )
        from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend

        # The queue and private arena are created inside the measured call.
        queue = list(enumerate(suffixes))
        admitted = bool(queue) and sum(len(s) for s in suffixes) <= 64
        if not admitted:
            raise RuntimeError("paged request admission failed")
        profile = TokenKVProfile(8, 128, "float16")
        pool = PagedKVPool(512)
        native = NativeWriteBackend(pool.capacity * profile.page_bytes,
                                    mx.default_stream(mx.gpu), permit_candidate=True)
        writer = PagedKVWriteOwner(pool, native, page_bytes=profile.page_bytes,
                                   permit_candidate=True)
        backend = NativeQwen3PagedBackend(writer, timeout_s=10,
                                          permit_candidate=True)
        backend.reads = backend.terminals = 0
        native_read_completed = backend.read_completed

        def counted_read(use, queries, **kwargs):
            result = native_read_completed(use, queries, **kwargs)
            backend.reads += 1
            if use.state == "closed":
                backend.terminals += 1
            return result

        backend.read_completed = counted_read
        candidate = Qwen3PackedCandidate(_PagedModel(self.model, mx), backend)
        lanes = tuple(PackedLane(prompt, tuple(
            PagedKVTokenOwner(writer, profile, permit_candidate=True)
            for _ in range(self.model_layers))) for prompt in prompts)
        owners = []
        output = []
        published = False
        try:
            # All exact contexts are built from a cold request state, inside
            # the clock. The same 28 native layers supply each suffix read.
            seed_logits, _ = candidate.forward(lanes, permit_candidate=True)
            mx.eval(seed_logits)
            for index, suffix in queue:
                owner = NativeAtomicRequestOwner(
                    f"request-{index}", lanes[index].layers, {},
                    supported_planes=("kv",), enabled=True)
                owners.append(owner)
                request = CandidateRequest(index, f"request-{index}", len(suffix), ("kv",))
                decision = PagedPackDecision(True, "measured_admission", (), index,
                                             len(suffix), None, None)
                sampled = []

                def run(branch):
                    logits, _ = candidate.forward(
                        (PackedLane(suffix, branch.layers),), permit_candidate=True,
                        atomic_branch=branch)
                    sampled.append(int(mx.argmax(logits[-1]).item()))
                    return len(suffix)

                receipt = execute_paged_request(decision, request, owner, run,
                                                permit_candidate=True)
                if not receipt.published or not sampled:
                    raise RuntimeError("native cache transaction did not publish")
                output.extend(sampled)
            mx.synchronize(native.stream)
            if writer.pending_epochs or writer.ledger.pending_count:
                raise RuntimeError("native request work remained pending")
            published = True
        finally:
            # A transaction keeps former public views for reader safety.
            # This offline cell has no readers after synchronization, so it
            # explicitly retires every published and former layer owner. An
            # ambiguous native submission keeps its pins until process exit.
            seen = set()
            if not writer.pending_epochs and not writer.ledger.pending_count:
                for owner in owners:
                    for state in (*owner._retired, owner._public):
                        for layer in state.layers:
                            if id(layer) not in seen:
                                layer.close()
                                seen.add(id(layer))
                    owner.reap_quarantine()
                for lane in lanes:
                    for layer in lane.layers:
                        if id(layer) not in seen:
                            layer.close()
                            seen.add(id(layer))
            queue.clear()
        if pool.free_count != pool.capacity or writer.pending_epochs or writer.ledger.pending_count:
            raise RuntimeError("request cleanup retained native pages or epochs")
        return {"arm": "paged", "admitted": admitted,
                "cache_transaction_published": published, "sampled_output": bool(output),
                "synchronized": True, "request_state_released": True,
                "queue_state_released": not queue, "model_layers": self.model_layers,
                "output_token_ids": output, "peak_resident_bytes": _resident_bytes(),
                "paged_read_calls": backend.reads, "terminal_successes": backend.terminals,
                "pending_native_epochs": len(writer.pending_epochs),
                "retained_pages": pool.capacity - pool.free_count}

    def close(self):
        self._closed = True


def make_driver(manifest: dict) -> Qwen3RequestDriver:
    return Qwen3RequestDriver(manifest)
