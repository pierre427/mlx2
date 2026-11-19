"""Model inspection without loading weights: identity, layout, unit count estimate, weight bytes.

The identity (fingerprint, adapter, family) comes from the adapter registry, the same resolution the
server and the carve host use. The layout comes from the artifact's config.json (standard keys only;
no model-name branches). Unit counts here are estimates for the resource plan; the exact per-layer
widths are the carve host's (recorded by the first stage that loads the model).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_MOE_EXPERT_KEYS = ("n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts")
_MOE_WIDTH_KEYS = ("moe_intermediate_size", "expert_intermediate_size", "ffn_hidden_size")


def artifact_name(path: Path) -> str:
    """The artifact's name: the directory name, or ``org/name`` for a Hugging Face snapshot directory."""
    parts = path.resolve().parts if not path.is_symlink() else path.parts
    for i, p in enumerate(parts):
        if p.startswith("models--") and i + 2 < len(parts) and parts[i + 1] == "snapshots":
            return p[len("models--"):].replace("--", "/")
    return path.name


def _text_config(cfg: dict) -> dict:
    for k in ("text_config", "llm_config", "language_config"):
        sub = cfg.get(k)
        if isinstance(sub, dict) and (sub.get("num_hidden_layers") or isinstance(sub.get("layers_block_type"), list)):
            out = {**cfg, **sub}
            if not out.get("num_hidden_layers"):         # hybrid trunks (Nemotron-H) count layers by block type
                out["num_hidden_layers"] = len(sub["layers_block_type"])
            return out
    return cfg


def _first(cfg, keys):
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, (int, float)) and v:
            return int(v)
    return None


def _moe_layers(cfg: dict, L: int, E: int | None) -> list[bool]:
    if not E:
        return [False] * L
    types = cfg.get("layers_block_type")                  # hybrid trunks name each layer's block (Nemotron-H)
    if isinstance(types, list) and len(types) == L:
        return [t == "moe" for t in types]
    pattern = cfg.get("hybrid_override_pattern")
    if isinstance(pattern, (str, list)) and len(pattern) == L:
        return [t == "E" for t in pattern]
    dense_first = int(cfg.get("first_k_dense_replace") or 0)
    only = set(cfg.get("mlp_only_layers") or [])
    step = int(cfg.get("decoder_sparse_step") or 1) or 1
    return [l >= dense_first and l not in only and (l + 1) % step == 0 for l in range(L)]


def _dense_layer(cfg: dict, l: int, moe: list) -> bool:
    """Layer l has a dense MLP: not MoE, and (hybrid trunks) not a Mamba or attention-only block."""
    if moe[l]:
        return False
    types = cfg.get("layers_block_type")
    if isinstance(types, list) and len(types) == len(moe):
        return types[l] == "mlp"
    pattern = cfg.get("hybrid_override_pattern")
    if isinstance(pattern, (str, list)) and len(pattern) == len(moe):
        return pattern[l] == "-"
    return True


def weight_bytes(path: Path) -> int:
    files = list(path.glob("*.safetensors")) or list(path.glob("*.npz"))
    return int(sum(f.resolve().stat().st_size for f in files if not f.name.startswith("._")))


_LAYER_KEY = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# the language model's decoder layers, not a vision / audio tower's
_NOT_DECODER = re.compile(r"(?:^|\.)(?:vision|visual|vit|image|audio|speech|mm_projector|multi_modal|encoder)[_.]")


def safetensors_index(path: Path) -> list:
    """[(key, file, begin, end)] for every tensor, from the shard headers alone (absolute byte ranges)."""
    import json as _json
    import struct
    out = []
    for f in sorted(path.glob("*.safetensors")):
        if f.name.startswith("._"):
            continue
        with f.resolve().open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = _json.loads(fh.read(n))
        base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            b, e = v["data_offsets"]
            out.append((k, str(f.resolve()), base + b, base + e))
    return out


# file-backed side tables a layer reads by row, never as a weight (Hy4-flash Engram tables, 31.7 GB each): not part of
# a streamed layer's bytes nor of the resident rest
_SIDE_TABLE = re.compile(r"\.engram_layer\.multi_head_embedding\.")


def layer_bytes(path: Path, L: int) -> dict:
    """Bytes per decoder layer (keys ``...layers.<i>....``), the rest (embeddings, head, norms, MTP) and the
    file-backed side tables read by row (``side_tables``)."""
    per = [0] * L
    experts = [0] * L
    rest = side = 0
    for k, _, b, e in safetensors_index(path):
        if _SIDE_TABLE.search(k):
            side += e - b
            continue
        m = None if _NOT_DECODER.search(k) else _LAYER_KEY.search(k)
        i = int(m.group(1)) if m else -1
        if 0 <= i < L and "mtp" not in k:
            per[i] += e - b
            if any(t in k for t in (".experts.", ".switch_mlp.", ".switch_glu.")) and "shared" not in k:
                experts[i] += e - b
        else:
            rest += e - b
    return {"per_layer": per, "max_layer": max(per) if per else 0, "rest": rest, "side_tables": side,
            "experts_per_layer": experts, "max_layer_experts": max(experts) if experts else 0,
            "layered": bool(per) and all(x > 0 for x in per)}


def inspect_checkpoint(model: str | Path) -> dict:
    path = Path(model).expanduser()
    if not (path / "config.json").exists():
        raise SystemExit(f"{path}: no config.json (not a model directory)")
    cfg = _text_config(json.loads((path / "config.json").read_text()))
    from mlx2.adapters.registry import inspect_model
    try:
        res = inspect_model(path)
        fp = str(res.artifact_fingerprint)
        family_only = None
    except ValueError as e:
        # no mlx2 adapter: a sweep family registered under the checkpoint's model_type may still map it (layer-major
        # capture only); its module names the checkpoint (checkpoint_fingerprint), as its own receipts do
        family_only = _family_without_adapter(path, cfg.get("model_type"), e)
        res = None
        fp = family_only["fingerprint"]
    L = _first(cfg, ("num_hidden_layers", "n_layer", "num_layers"))
    if not L:
        raise SystemExit(f"{path}: config.json has no layer count")
    d = _first(cfg, ("hidden_size", "d_model", "n_embd"))
    F = _first(cfg, ("intermediate_size", "ffn_dim", "n_inner"))
    E = _first(cfg, _MOE_EXPERT_KEYS)
    Fe = _first(cfg, _MOE_WIDTH_KEYS) or F
    moe = _moe_layers(cfg, L, E)
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    desc = getattr(res, "descriptor", None) if res is not None else None
    name = artifact_name(path)
    out = {
        # <artifact name>@<fingerprint[:8]> (AGENTS.md); for a Hugging Face snapshot directory the artifact is its
        # repository name, not the snapshot hash. The carve host names a run by its directory, so point the
        # command at a directory (or symlink) named after the artifact to make both names agree.
        "path": str(path), "artifact": name, "fingerprint": fp,
        "checkpoint": f"{name.split('/')[-1]}@{fp[:8]}", "directory_checkpoint": f"{path.name}@{fp[:8]}",
        "adapter": getattr(getattr(res, "adapter_type", None), "__name__", None),
        "family": getattr(desc, "family", None) if res is not None else cfg.get("model_type"),
        "model_type": cfg.get("model_type"), "no_adapter": family_only,
        "residual_streams": _first(cfg, ("hc_mult",)) or 1,      # hyper-connection streams the carry holds per token
        "layers": L, "hidden": d, "intermediate": F, "vocab": _first(cfg, ("vocab_size",)),
        "moe": {"experts": E, "expert_intermediate": Fe, "moe_layers": int(sum(moe)),
                "top_k": _first(cfg, ("num_experts_per_tok", "num_experts_per_token", "moe_top_k", "top_k"))} if E else None,
        "quantization": {"bits": q.get("bits"), "group_size": q.get("group_size"), "mode": q.get("mode")} if q else None,
        "weight_bytes": weight_bytes(path),
        "file_backed_ple": (path / "ple_rows.bin").exists(),   # an artifact feature, asserted at load when present
    }
    dense_units = sum(F or 0 for l in range(L) if _dense_layer(cfg, l, moe))
    out["units"] = {"dense_neurons": dense_units,
                    "moe_expert_units": sum(E for l in range(L) if moe[l]) if E else 0,
                    "moe_neuron_units": sum(E * Fe for l in range(L) if moe[l]) if E else 0,
                    "estimate": True}
    out["kind"] = "moe" if E and any(moe) else "dense"
    from mlx2.runtime.streamed_load import accepts_streaming_request
    out["adapter_streaming"] = bool(accepts_streaming_request(res.adapter_type)) if res is not None else False
    try:                                   # a layer-major family of the maps session's sweep capture, by registry
        from . import paths as _paths
        _paths.research_imports()
        from research.amoeba.sweep.family import FAMILIES
        # by the adapter family, else by the config model_type (hy_v4_flash: adapter family "hy-v4-flash";
        # kimi_linear, nemotron_h: families that stream an adapter's own model)
        out["sweep_family"] = next((f for f in (out["family"], out["model_type"]) if f in FAMILIES), None)
    except (ImportError, SystemExit):
        out["sweep_family"] = None
    try:
        out["layer_bytes"] = layer_bytes(path, L)
    except (OSError, ValueError, KeyError) as e:          # an unreadable header: no streaming plan
        out["layer_bytes"] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _family_without_adapter(path: Path, model_type, err) -> dict:
    from . import paths as _paths
    _paths.research_imports()
    from research.amoeba.sweep.family import FAMILIES
    if not isinstance(model_type, str) or model_type not in FAMILIES:
        raise SystemExit(f"{path}: {err}; and no sweep family is registered for model_type {model_type!r}")
    import importlib
    mod = importlib.import_module(FAMILIES[model_type].split(":")[0])
    fpf = getattr(mod, "checkpoint_fingerprint", None)
    if not callable(fpf):
        raise SystemExit(f"{path}: no mlx2 adapter, and sweep family {model_type!r} has no checkpoint_fingerprint(path) "
                         "to name the checkpoint")
    return {"family": model_type, "fingerprint": str(fpf(path)), "error": str(err),
            "note": "no mlx2 adapter: map capture through the sweep family only; carve stages refuse"}


def slug(checkpoint: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", checkpoint.split("@")[0].split("/")[-1].lower()).strip("-")
