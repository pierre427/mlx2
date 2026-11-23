"""A Donor-shaped capture backend over Hy4-flash for the calculator probe fit.

Research tooling only (GPU; nothing here runs in serving). The model loads through its mlx2 adapter
on the ordinary route with the served base policy (``{"engram_table_format": "iq4_nl"}``), so the
trunk numerics are the ones the DFly2 route verifies with (chain-packed verify is exact against
ordinary decode). ``forward(ids, record=)`` returns ``(logits, caps, {})`` like
research.amoeba.donor.Donor, with ``caps[l] = (capture_l, None)``: ``det_tile_capture`` (the f32
mean of the four hyper-connection streams) of layer ``l``'s output, read through the same
``Model.layer_output_readers`` hook the served calculator reads. A probe fitted on these features
is what ``read_point: layer_output_capture`` evaluates in serving, on the ordinary and the DFly2
routes alike.

Driver: ``scripts/amoeba_det_calc_cpu.py --backend hy4 --read-point layer_output_capture
--skip-generation`` (docs/hy4-flash/det-calculator.md has the GPU run sheet).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
MODEL_HY4 = Path.home() / "mlx-models/Hy4-flash-MLX-4.6bit-mlx2"
# The served base policy (GSM8K and the served check use the same one).
BASE_POLICY = {"engram_table_format": "iq4_nl"}
_SENTINEL = "⁣Q⁣"


class BackendHy4:
    def __init__(self, path=None, policy=None, log=print, footprint_limit_gb=104.0, wire=True, watchdog=True,
                 min_available_gb=15.0, available_counter="vm_stat", swap_growth_gb=2.0):
        from mlx2.adapters import hy_v4_flash as H

        self.watchdog = None
        if watchdog:
            import amoeba27b_lib as A

            # ~91 GB active once wired. Features stream to disk (det_calc.capture_to_disk), so the
            # process adds one sequence's captures, not the whole set. Also exits on low machine
            # memory (``available_counter``), swap growth or critical pressure.
            A.footprint_watchdog(footprint_limit_gb, log, min_avail_gb=min_available_gb,
                                 counter=available_counter, swap_growth_gb=swap_growth_gb)
            self.watchdog = A.WATCH
        self.path = Path(path or MODEL_HY4).expanduser()
        self.policy = dict(BASE_POLICY if policy is None else policy)
        if self.policy.get("draft_model") is not None:
            raise ValueError("the Hy4 capture backend loads the ordinary route (no drafter)")
        t0 = time.time()
        self.adapter = H.HYV4FlashAdapter(str(self.path), execution_policy=self.policy)
        if wire:
            from mlx2.runtime.weight_residency import wire_serving_weights

            # As serving does: pageable weights get compressed or swapped by prefill transients.
            self.weight_residency = wire_serving_weights()
        self.model = self.adapter.model
        self.L = len(self.model.model.layers)
        self.d = int(self.model.args.hidden_size)
        fp = self.adapter.identity["fingerprint"]
        self.source_fingerprint = fp
        self.artifact = f"{self.path.name}@{fp.split(':')[-1][:8]}"
        self.host_tiles_sha256 = None
        self.capture = H.DET_TILE_CAPTURE
        diag = self.adapter.diagnostics()
        # Recorded in the manifest: the adapter refuses the tile under other trunk numerics.
        self.fit_execution_policy = {
            "engram_table_format": self.adapter.engram_table_format,
            "engram_precision": self.policy.get("engram_precision", "transformers"),
            "latent_kv": self.adapter.latent_kv,
            "moe_top_k": self.policy.get("moe_top_k"),
        }
        self.receipt = {"checkpoint": diag["checkpoint"], "layout": self.adapter.layout,
                        "engram_table_format": diag["engram_table_format"],
                        "policy": self.policy, "capture": self.capture}
        self._tok = self.adapter.tokenizer
        # Positive templates in research/amoeba/det_calc.py use a chat string with {q}: Hy4's own
        # template, no_think (the served default).
        self.chat_format = self._chat_format()
        self._src_tok = None
        log(f"loaded {self.artifact} ({self.L} layers, d={self.d}, policy {self.policy}) in "
            f"{time.time() - t0:.0f}s, MLX active {mx.get_active_memory() / 1e9:.1f} GB")

    def _chat_format(self):
        """The rendered no_think chat prompt as a ``str.format`` template with ``{q}``, or None
        when the artifact has no chat template."""
        try:
            rendered = self.adapter.render_prompt({"messages": [{"role": "user", "content": _SENTINEL}]})
        except (ValueError, TypeError, OSError):
            return None
        if not isinstance(rendered, str) or rendered.count(_SENTINEL) != 1:
            raise RuntimeError("Hy4 chat template did not render the probe sentinel exactly once")
        return rendered.replace("{", "{{").replace("}", "}}").replace(_SENTINEL, "{q}")

    def forward(self, ids, *, mlp_hook=None, record=(), stop_after=None, **_unused):
        from mlx2.adapters.hy_v4_flash import det_tile_capture

        if mlp_hook is not None:
            raise ValueError("the Hy4 capture backend has no MLP hook (the calculator reads layer outputs)")
        layers = tuple(sorted(record))
        caps = {}
        readers = self.model.layer_output_readers
        if any(l in readers for l in layers):
            raise RuntimeError("a layer-output reader is already installed")

        def read(layer, streams, _ids, _prefix):
            caps[layer] = (det_tile_capture(streams), None)

        for l in layers:
            readers[l] = read
        try:
            hidden = self.model.hidden(ids, self.model.make_cache())
        finally:
            for l in layers:
                readers.pop(l, None)
        if stop_after is not None:
            mx.eval([v for v, _ in caps.values()])
            return None, caps, {}
        return self.model.logits(hidden), caps, {}

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False)

    def reencode(self, ids):
        """Corpus ids are Qwen3-0.6B tokens: decode and re-encode with the host tokenizer."""
        if self._src_tok is None:
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer

            self._src_tok = AutoTokenizer.from_pretrained(
                snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True))
        return self.encode(self._src_tok.decode(list(ids)))

    def close(self):
        """Free the model (the probe fit reads features from disk only)."""
        import gc

        adapter, self.adapter, self.model = self.adapter, None, None
        if adapter is not None:
            adapter.close()
        del adapter
        gc.collect()
        mx.clear_cache()
