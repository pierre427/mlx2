"""A Donor-shaped capture backend over Flash-Next (qwen4) for the calculator probe fit.

Research tooling only. The model is loaded through its mlx2 adapter with the native-MTP route's
adapter-owned defaults (the production kernel profile, PLE offloaded to NVMe), and
``forward(ids, record=)`` returns ``(logits, caps, {})`` like research.amoeba.donor.Donor, with
``caps[l] = (capture_l, None)``: the trunk's capture projection (the learned HC mixer) of the
hidden state after layer ``l``, which is exactly what the served ``layer_output_capture``
calculator reads (``amoeba_det_tiles.install_layer_output_reader``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
MODEL_FN = Path.home() / "mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP"


def production_adapter_policy():
    """The native-MTP route's adapter-owned defaults (kernels, decode laws), as serving passes them."""
    from mlx2.adapters.flash_next import FlashNextAdapter as F

    route = F.default_route_execution_policy["native_mtp"]
    return {k: v for k, v in route.items() if k in F.adapter_owned_default_policy_keys}


class BackendFN:
    def __init__(self, path=None, log=print, footprint_limit_gb=88.0):
        import time

        import amoeba27b_lib as A
        from mlx2.adapters.registry import inspect_model

        A.footprint_watchdog(footprint_limit_gb, log)
        self.path = Path(path or MODEL_FN)
        t0 = time.time()
        self.adapter = inspect_model(self.path).adapter_type(
            str(self.path), execution_policy=production_adapter_policy())
        if not getattr(self.adapter, "_tables", None) or not os.environ.get("MLX_QWEN4_PLE_NVME"):
            raise RuntimeError("Flash-Next must run with its PLE table offloaded (production profile)")
        self.trunk = self.adapter.model.language_model.model
        self.L = len(self.trunk.layers)
        self.d = int(self.trunk.args.hidden_size)
        fp = self.adapter.identity["fingerprint"]
        self.source_fingerprint = fp
        self.artifact = f"{self.path.name}@{fp.split(':')[-1][:8]}"
        self.host_tiles_sha256 = None
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download
        self._src_tok = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True))
        self._tok = self.adapter.tokenizer
        log(f"loaded {self.artifact} ({self.L} layers, d={self.d}) in {time.time() - t0:.0f}s, "
            f"MLX active {mx.get_active_memory() / 1e9:.1f} GB")

    def forward(self, ids, *, mlp_hook=None, record=(), stop_after=None, **_unused):
        if mlp_hook is not None:
            raise ValueError("the Flash-Next capture backend has no MLP hook (calculator reads layer outputs)")
        layers = tuple(sorted(record))
        sink = []
        hidden = self.trunk(ids, capture_layers=layers, hidden_sink=sink)
        caps = {l: (sink[i], None) for i, l in enumerate(layers)}
        if stop_after is not None:
            mx.eval(sink)
            return None, caps, {}
        lm = self.adapter.model.language_model
        logits = lm.model.embed_tokens.as_linear(hidden) if lm.args.tie_word_embeddings else lm.lm_head(hidden)
        return logits, caps, {}

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False)

    def reencode(self, ids):
        """Corpus ids are Qwen3-0.6B tokens: decode and re-encode with the host tokenizer."""
        return self.encode(self._src_tok.decode(list(ids)))
