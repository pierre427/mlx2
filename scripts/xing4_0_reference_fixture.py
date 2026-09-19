#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Regenerate the tiny Xing4.0 parity fixture from the Hugging Face reference.

Runs under a Python that has torch + transformers (the lab uses
``~/Desktop/mlx-uag/.venv/bin/python``). It imports the
upstream ``modeling_xing4_0.py`` / ``configuration_xing4_0.py`` from
``--reference-dir`` without copying them, builds a tiny random fp32 model,
and writes to ``--out``:

* ``config.json``        HF-style tiny config (model_type ``xing4_0``)
* ``weights.safetensors`` HF-named fp32 weights, including the MTP layer at
  ``model.layers.{num_hidden_layers}.*`` exactly as the real checkpoint lays
  it out (own ``embed_tokens`` equal to the trunk's, own ``shared_head.head``
  deliberately different so both sanitize branches are exercised)
* ``reference.safetensors`` input ids, logits, post-norm hiddens, MTP
  outputs and one standalone mHC evaluation
* ``manifest.json``      source hashes and library versions

HF ignores layer 40 (``_keys_to_ignore_on_load_unexpected``), so the MTP
reference is assembled here from the HF building blocks (attention, MoE,
RMSNorm, rotary) following the DeepSeek-V3 MTP definition used by vLLM
``deepseek_mtp`` and SGLang ``DeepseekModelNextN``:
``eh_proj(cat[enorm(embed(t_{i+1})), hnorm(h_i)]) -> residual block ->
shared_head.norm -> shared_head.head``, where ``h_i`` is the trunk's
stream-averaged, final-normed hidden (the tensor both engines hand the draft).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "tests" / "fixtures" / "xing4_0_tiny"

TINY_CONFIG = {
    "architectures": ["Xing4_0ForCausalLM"],
    "model_type": "xing4_0",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "first_k_dense_replace": 2,
    "hidden_act": "silu",
    "hidden_size": 64,
    "intermediate_size": 96,
    "kv_lora_rank": 32,
    "max_position_embeddings": 262144,
    "moe_intermediate_size": 16,
    "moe_layer_freq": 1,
    "n_group": 1,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 4,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 4,
    "num_key_value_heads": 4,
    "num_nextn_predict_layers": 1,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-06,
    "mhc_h_res_clamp_min": -30,
    "mhc_h_res_clamp_max": 30,
    "q_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 64,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    "routed_scaling_factor": 2.0,
    "scoring_func": "sigmoid",
    "tie_word_embeddings": False,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "v_head_dim": 16,
    "vocab_size": 128,
}

SEQ_A = 11
SEQ_B = 7
SEED = 20260918


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _import_reference(reference_dir: Path):
    package = types.ModuleType("xing4_0_reference")
    package.__path__ = [str(reference_dir)]
    sys.modules["xing4_0_reference"] = package
    configuration = importlib.import_module("xing4_0_reference.configuration_xing4_0")
    modeling = importlib.import_module("xing4_0_reference.modeling_xing4_0")
    return configuration, modeling


def _randomize(model: torch.nn.Module, generator: torch.Generator) -> None:
    """Deterministic, well-conditioned random fp32 weights."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            shape = param.shape
            if name.endswith("hc_fn"):
                value = torch.randn(shape, generator=generator) * 0.08
            elif name.endswith("hc_base"):
                value = torch.randn(shape, generator=generator) * 0.5
            elif name.endswith("hc_scale"):
                value = 0.5 + torch.rand(shape, generator=generator)
            elif param.ndim == 1:  # RMSNorm weights
                value = 1.0 + 0.1 * torch.randn(shape, generator=generator)
            elif name.endswith("mlp.gate.weight"):
                value = torch.randn(shape, generator=generator) * 0.3
            elif "embed_tokens" in name:
                value = torch.randn(shape, generator=generator)
            else:
                value = torch.randn(shape, generator=generator) / math.sqrt(shape[-1])
            param.copy_(value)
        for name, buffer in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buffer.copy_(torch.randn(buffer.shape, generator=generator) * 0.1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    reference_dir = args.reference_dir.resolve()
    configuration, modeling = _import_reference(reference_dir)

    # Upstream _init_weights references attributes (fn/base/scale) the mHC
    # module does not define; every tensor is overwritten below anyway.
    modeling.Xing4_0PreTrainedModel._init_weights = lambda self, module: None

    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    config = configuration.Xing4_0Config(**TINY_CONFIG)
    config._attn_implementation = "eager"
    model = modeling.Xing4_0ForCausalLM(config).float().eval()
    _randomize(model, generator)

    class SharedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = modeling.Xing4_0RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    class ReferenceMTPLayer(torch.nn.Module):
        """DeepSeek-V3 MTP block built from HF Xing4.0 modules (no mHC)."""

        def __init__(self):
            super().__init__()
            H = config.hidden_size
            self.embed_tokens = torch.nn.Embedding(config.vocab_size, H)
            self.enorm = modeling.Xing4_0RMSNorm(H, eps=config.rms_norm_eps)
            self.hnorm = modeling.Xing4_0RMSNorm(H, eps=config.rms_norm_eps)
            self.eh_proj = torch.nn.Linear(2 * H, H, bias=False)
            self.self_attn = modeling.Xing4_0Attention(config, layer_idx=0)
            self.mlp = modeling.Xing4_0MoE(config)
            self.input_layernorm = modeling.Xing4_0RMSNorm(H, eps=config.rms_norm_eps)
            self.post_attention_layernorm = modeling.Xing4_0RMSNorm(H, eps=config.rms_norm_eps)
            self.shared_head = SharedHead()

        def forward(self, hidden, tokens, rotary):
            S = tokens.shape[1]
            e = self.enorm(self.embed_tokens(tokens))
            h = self.hnorm(hidden)
            x = self.eh_proj(torch.cat([e, h], dim=-1))
            positions = torch.arange(S)[None]
            cos_sin = rotary(x, positions)
            mask = torch.full((S, S), torch.finfo(torch.float32).min).triu(1)[None, None]
            a, _ = self.self_attn(self.input_layernorm(x), cos_sin, mask)
            x = x + a
            x = x + self.mlp(self.post_attention_layernorm(x))
            post = self.shared_head.norm(x)
            return self.shared_head.head(post), post

    mtp = ReferenceMTPLayer().float().eval()
    _randomize(mtp, generator)
    with torch.no_grad():
        # DeepSeek-style checkpoints duplicate the trunk embedding into the
        # MTP layer; keep that equality, but give the MTP head its own
        # weights so the runtime must prove which head it uses.
        mtp.embed_tokens.weight.copy_(model.model.embed_tokens.weight)

    ids_a = torch.randint(0, config.vocab_size, (1, SEQ_A), generator=generator)
    ids_b = torch.randint(0, config.vocab_size, (1, SEQ_B), generator=generator)
    with torch.no_grad():
        out_a = model(input_ids=ids_a, use_cache=False)
        out_b = model(input_ids=ids_b, use_cache=False)
        hidden_a = model.model(input_ids=ids_a, use_cache=False).last_hidden_state
        hidden_b = model.model(input_ids=ids_b, use_cache=False).last_hidden_state
        mtp_logits_a, mtp_post_a = mtp(
            hidden_a[:, :-1], ids_a[:, 1:], model.model.rotary_emb
        )
        hc = model.model.layers[2].attn_hc
        mhc_input = torch.randn(
            (2, 3, config.hc_mult, config.hidden_size), generator=generator
        )
        mhc_post, mhc_comb, mhc_collapsed = hc(mhc_input)

    weights = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    weights = {k: v for k, v in weights.items() if "rotary_emb" not in k}
    mtp_index = config.num_hidden_layers
    for key, value in mtp.state_dict().items():
        weights[f"model.layers.{mtp_index}.{key}"] = value.detach().contiguous()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(out / "weights.safetensors"), metadata={"format": "pt"})
    reference = {
        "ids_a": ids_a.to(torch.int32),
        "ids_b": ids_b.to(torch.int32),
        "logits_a": out_a.logits.float(),
        "logits_b": out_b.logits.float(),
        "hidden_a": hidden_a.float(),
        "hidden_b": hidden_b.float(),
        "mtp_logits_a": mtp_logits_a.float(),
        "mtp_post_a": mtp_post_a.float(),
        "mhc_input": mhc_input.float(),
        "mhc_post": mhc_post.float(),
        "mhc_comb": mhc_comb.float(),
        "mhc_collapsed": mhc_collapsed.float(),
    }
    save_file(
        {k: v.contiguous() for k, v in reference.items()},
        str(out / "reference.safetensors"),
    )
    (out / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2, sort_keys=True) + "\n")
    manifest = {
        "generator": "scripts/xing4_0_reference_fixture.py",
        "seed": SEED,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "reference_sources": {
            name: _sha256(reference_dir / name)
            for name in ("modeling_xing4_0.py", "configuration_xing4_0.py")
        },
        "mtp_reference": "DeepSeek-V3 MTP (vLLM deepseek_mtp / SGLang NextN) from HF blocks",
        "mtp_seed_hidden": "trunk last_hidden_state: stream mean then model.norm",
        "dtype": "float32",
        "attn_implementation": "eager",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    sizes = {p.name: p.stat().st_size for p in sorted(out.iterdir())}
    print(json.dumps(sizes, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
