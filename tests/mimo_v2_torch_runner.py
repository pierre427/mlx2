"""Run Xiaomi's PyTorch MiMo-V2 reference in an isolated interpreter (``python -I``).

Invoked by tests/test_mimo_v2_reference_cpu.py and tests/test_mimo_v2_real_layer_cpu.py as

    python -I tests/mimo_v2_torch_runner.py tiny  PKG CONFIG.json STATE.npz IDS.npy OUT.npz
    python -I tests/mimo_v2_torch_runner.py layer PKG CKPT_DIR LAYER H.npy OUT.npz

``PKG`` is a fresh temporary directory holding a copy of the checkpoint's
``modeling_mimo_v2.py`` / ``configuration_mimo_v2.py`` as package ``mimo_ref``;
the vendor files are never imported from the checkpoint directory. Everything
runs in float32 on the CPU with eager attention.

Captures per layer: the layer output, the dense down_proj input, the router's
(ids, weights), and each expert's outputs for the tokens routed to it.
In ``layer`` mode the weights are dequantized by tests/mimo_v2_numpy_ref.py
(independent NumPy) and routed experts are built one at a time on demand, so
the 384-expert layer never materializes (RSS stays a few GB).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _import_reference(pkg: Path):
    sys.path.insert(0, str(pkg))
    import torch  # noqa: F401
    from mimo_ref import configuration_mimo_v2 as C
    from mimo_ref import modeling_mimo_v2 as Mo
    return C, Mo


class Recorder:
    def __init__(self, model, torch):
        self.torch, self.out = torch, {}
        self.handles = []
        layers = model.model.layers
        for l, layer in enumerate(layers):
            self.handles.append(layer.register_forward_hook(self._save(f"h{l}")))
            mlp = layer.mlp
            self.handles.append(mlp.register_forward_pre_hook(self._save_in(f"x{l}")))
            if hasattr(mlp, "gate") and hasattr(mlp, "experts"):
                self.handles.append(mlp.gate.register_forward_hook(self._save_gate(l)))
                for e, ex in enumerate(mlp.experts):
                    self.handles.append(ex.register_forward_hook(self._save(f"y{l}_{e}")))
            else:
                self.handles.append(mlp.down_proj.register_forward_pre_hook(self._save_in(f"act{l}")))

    def _np(self, t):
        return t.detach().to(self.torch.float32).cpu().numpy()

    def _save(self, key):
        def hook(mod, inp, out):
            self.out[key] = self._np(out[0] if isinstance(out, tuple) else out)
        return hook

    def _save_in(self, key):
        def hook(mod, inp):
            self.out[key] = self._np(inp[0])
        return hook

    def _save_gate(self, l):
        def hook(mod, inp, out):
            self.out[f"ids{l}"] = out[0].detach().cpu().numpy().astype(np.int64)
            self.out[f"wts{l}"] = self._np(out[1])
        return hook


def masks(config, h, position_ids):
    """The reference's own mask mapping, built by transformers' mask functions.

    MiMoV2Model.forward was written against transformers 5.3 (``input_embeds``,
    ``cache_position``); 5.17 renamed the argument, so the mapping is built here
    with the current signature and passed in as the dict the forward accepts."""
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    kw = dict(config=config, inputs_embeds=h, attention_mask=None, past_key_values=None,
              position_ids=position_ids, allow_is_causal_skip=False)
    return {"full_attention": create_causal_mask(**kw),
            "sliding_window_attention": create_sliding_window_causal_mask(**kw)}


def tiny(pkg, cfg_path, state_path, ids_path, out_path):
    C, Mo = _import_reference(Path(pkg))
    import torch
    torch.manual_seed(0)
    cfg = json.loads(Path(cfg_path).read_text())
    config = C.MiMoV2Config(**cfg)
    config._attn_implementation = "eager"
    model = Mo.MiMoV2ForCausalLM(config).to(torch.float32).eval()
    state = {k: torch.from_numpy(v) for k, v in np.load(state_path).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    rec = Recorder(model, torch)
    ids = torch.from_numpy(np.load(ids_path).astype(np.int64))
    config.layer_types = ["sliding_attention" if x == 1 else "full_attention" for x in config.hybrid_layer_pattern]
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    with torch.no_grad():
        mapping = masks(config, model.model.embed_tokens(ids), pos)
        logits = model(input_ids=ids, attention_mask=mapping, use_cache=False).logits
    rec.out["logits"] = rec._np(logits)
    np.savez(out_path, **rec.out)


class LazyExpert:
    """Stands in for one routed expert: builds the vendor ``MiMoV2MLP`` from NumPy-dequantized
    weights when called, runs it and drops it (forward hooks still see its output)."""

    def __init__(self, Mo, config, ckpt, wm, prefix):
        self.Mo, self.config, self.ckpt, self.wm, self.prefix = Mo, config, ckpt, wm, prefix

    def module(self, torch):
        sys.path.insert(0, str(HERE))
        import mimo_v2_numpy_ref as R
        mlp = self.Mo.MiMoV2MLP(self.config, intermediate_size=self.config.moe_intermediate_size)
        for n in ("gate_proj", "up_proj", "down_proj"):
            name = f"{self.prefix}{n}.weight"
            _, _, packed = R.read(self.ckpt / self.wm[name], name)
            _, _, scale = R.read(self.ckpt / self.wm[name + "_scale"], name + "_scale")
            getattr(mlp, n).weight.data = torch.from_numpy(R.mxfp4_dequant(packed, scale))
        return mlp.to(torch.float32)


def layer(pkg, ckpt, l, h_path, out_path):
    C, Mo = _import_reference(Path(pkg))
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(HERE))
    import mimo_v2_numpy_ref as R
    ckpt, l = Path(ckpt), int(l)
    cfg = json.loads((ckpt / "config.json").read_text())
    wm = json.loads((ckpt / "model.safetensors.index.json").read_text())
    tp = int(wm.get("metadata", {}).get("tp_size", 1))
    wm = wm["weight_map"]
    for k in ("vision_config", "audio_config", "processor_config", "quantization_config"):
        cfg.pop(k, None)
    config = C.MiMoV2Config(**cfg)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        dec = Mo.MiMoV2DecoderLayer(config, l, attention_projection_layout=config.attention_projection_layout)
    p = f"model.layers.{l}."

    def get(name):
        dt, _, v = R.read(ckpt / wm[name], name)
        if name + "_scale_inv" in wm:
            _, _, s = R.read(ckpt / wm[name + "_scale_inv"], name + "_scale_inv")
            return R.fp8_dequant(v, s, (128, 128))
        return v.astype(np.float32)

    def assign(module, pname, value):
        setattr(module, pname, nn.Parameter(torch.from_numpy(np.ascontiguousarray(value, np.float32)),
                                            requires_grad=False))

    att = dec.self_attn
    H, K, hd, vhd = att.num_attention_heads, att.num_key_value_heads, att.head_dim, att.v_head_dim
    name = p + "self_attn.qkv_proj.weight"
    _, _, codes = R.read(ckpt / wm[name], name)
    _, _, scale = R.read(ckpt / wm[name + "_scale_inv"], name + "_scale_inv")
    assign(att.qkv_proj, "weight", np.concatenate(R.fused_qkv_dequant(codes, scale, H, K, hd, vhd, tp, (128, 128))))
    assign(att.o_proj, "weight", get(p + "self_attn.o_proj.weight"))
    if att.attention_sink_bias is not None:
        assign(att, "attention_sink_bias", get(p + "self_attn.attention_sink_bias"))
    assign(dec.input_layernorm, "weight", get(p + "input_layernorm.weight"))
    assign(dec.post_attention_layernorm, "weight", get(p + "post_attention_layernorm.weight"))
    mlp = dec.mlp
    if hasattr(mlp, "experts"):
        assign(mlp.gate, "weight", get(p + "mlp.gate.weight"))
        assign(mlp.gate, "e_score_correction_bias", get(p + "mlp.gate.e_score_correction_bias"))
        built = {}

        class Proxy(nn.Module):
            def __init__(self, lazy):
                super().__init__()
                self.lazy = lazy

            def forward(self, x):
                built[self.lazy.prefix] = built.get(self.lazy.prefix, 0) + 1
                return self.lazy.module(torch)(x)

        mlp.experts = nn.ModuleList([Proxy(LazyExpert(Mo, config, ckpt, wm, f"{p}mlp.experts.{e}."))
                                     for e in range(config.n_routed_experts)])
    else:
        for n in ("gate_proj", "up_proj", "down_proj"):
            assign(getattr(mlp, n), "weight", get(p + f"mlp.{n}.weight"))
    left = [n for n, t in list(dec.named_parameters()) + list(dec.named_buffers()) if t.is_meta]
    if left:
        raise SystemExit(f"unmaterialized reference tensors: {left[:5]}")
    dec.eval()

    # Masks and rotary as MiMoV2Model.forward builds them.
    config.layer_types = ["sliding_attention" if x == 1 else "full_attention" for x in config.hybrid_layer_pattern]
    h = torch.from_numpy(np.load(h_path).astype(np.float32))
    T = h.shape[1]
    cache_position = torch.arange(T)
    position_ids = cache_position.unsqueeze(0)
    swa = dec.attention_type == "sliding_window_attention"
    mask = masks(config, h, position_ids)[dec.attention_type]
    rot = Mo.MiMoV2RotaryEmbedding(config=config, is_swa=swa)
    pos = rot(h, position_ids)

    out = {}
    hooks = [dec.post_attention_layernorm.register_forward_hook(
        lambda m, i, o: out.__setitem__("x", o.detach().numpy()))]
    if hasattr(mlp, "experts"):
        def gate_hook(m, i, o):           # a forward hook must return None (else it replaces o)
            out["ids"], out["wts"] = o[0].numpy().astype(np.int64), o[1].numpy()

        hooks.append(mlp.gate.register_forward_hook(gate_hook))
        for e, ex in enumerate(mlp.experts):
            hooks.append(ex.register_forward_hook(
                (lambda e: lambda m, i, o: out.__setitem__(f"y{e}", o.detach().numpy()))(e)))
    else:
        hooks.append(mlp.down_proj.register_forward_pre_hook(lambda m, i: out.__setitem__("act", i[0].numpy())))
    with torch.no_grad():
        y = dec(h, attention_mask=mask, position_ids=position_ids, position_embeddings=pos,
                cache_position=cache_position)
    out["h"] = (y[0] if isinstance(y, tuple) else y).numpy()
    np.savez(out_path, **out)


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    {"tiny": tiny, "layer": layer}[mode](*args)
