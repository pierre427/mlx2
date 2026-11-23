"""Growth operators for the A0 core (H33 / H39 / H40 groundwork). Each returns a new Core and leaves the input core
unchanged. The function-preserving operators give, at the moment of growth, the same logits as before:

  insert_block(core, at)    a new block that starts as the identity (attention output and FFN down projections are
                            zero); bus sites after original block i >= at move to i + 1, so the bus stays after the
                            same original block. Exact.
  widen_ffn(core, new_ffn)  extra FFN hidden units whose down-projection columns are zero. Exact.
  widen_d(core, extra_heads) extra attention heads (head dim unchanged), so the residual width grows by extra_heads * hd.
                            Every write into the new dimensions (embeddings, attention output, FFN down, bus reader)
                            starts at zero; RMSNorm gains are scaled by sqrt(d_old / d_new) so the normed old dims are
                            unchanged; the LM head and the bus tap ignore the new dims at first. Exact: the RMSNorm
                            epsilon is scaled by d_old / d_new as well (stored as CoreConfig.norm_eps).
  remove_block(core, i)     pruning; not function-preserving (the growth controller judges it on validation).

With moments=True an operator maps an AdamW moment tree instead of weights (old entries kept, every new entry 0, no
gain rescaling); map_optimizer uses it so a grown core keeps its optimiser state instead of restarting AdamW.

The bus contract (CORE-001: no bus call equals the ordinary graph) holds for every grown core, as for any Core.
"""
from __future__ import annotations

import dataclasses

import mlx.core as mx
from mlx.utils import tree_flatten

from research.ens.model import Core, CoreConfig


def _clone(core: Core, cfg: CoreConfig) -> Core:
    new = Core(cfg)
    return new


def _copy_into(new: Core, old: Core, mapping):
    """Copy parameters by name; mapping(name, old_params, new_value) may return a custom tensor. A parameter the
    mapping does not handle must exist in the old core with the same shape: a silent fresh init would break
    function preservation (review 2026-11-22), so a shape mismatch raises. Parameters new by name keep their init
    only when the mapping returns them explicitly."""
    old_p = dict(tree_flatten(old.parameters()))
    new_p = dict(tree_flatten(new.parameters()))
    out = []
    for name, nv in new_p.items():
        ov = mapping(name, old_p, nv)
        if ov is None:
            ov = old_p.get(name)
            if ov is None or ov.shape != nv.shape:
                raise ValueError(f"growth: no rule for parameter {name} "
                                 f"({None if ov is None else ov.shape} -> {nv.shape})")
        out.append((name, ov))
    new.load_weights(out)
    return new


def _shift_sites(sites, at):
    return tuple((n, i + 1 if i >= at else i) for n, i in sites)


def insert_block(core: Core, at: int, *, moments: bool = False) -> Core:
    cfg = core.cfg
    if not 0 <= at <= cfg.layers:
        raise ValueError(f"insert position {at} outside 0..{cfg.layers}")
    if cfg.lora_rank:
        raise ValueError("insert_block on a LoRA core is not supported")
    new_cfg = dataclasses.replace(cfg, layers=cfg.layers + 1, sites=_shift_sites(cfg.sites, at))
    new = Core(new_cfg)

    def mapping(name, old_p, nv):
        parts = name.split(".")
        if parts[:2] == ["backbone", "blocks"]:
            i = int(parts[2])
            if i == at:                                         # the new block: identity at birth
                if moments or parts[3] in ("o", "down"):
                    return mx.zeros_like(nv)
                return nv
            src = i if i < at else i - 1
            return old_p[".".join(["backbone", "blocks", str(src)] + parts[3:])]
        return old_p.get(name)
    return _copy_into(new, core, mapping)


def widen_ffn(core: Core, new_ffn: int, *, moments: bool = False) -> Core:
    cfg = core.cfg
    if new_ffn <= cfg.ffn:
        raise ValueError("new_ffn must be larger")
    if cfg.lora_rank:
        raise ValueError("widen_ffn on a LoRA core is not supported")
    new = Core(dataclasses.replace(cfg, ffn=new_ffn))
    f0 = cfg.ffn
    init = 0.0 if moments else 0.02

    def mapping(name, old_p, nv):
        if name.endswith(".up.weight") and "lora" not in name:    # (2*ffn, d): [gate; up] halves
            ov = old_p[name]
            g, u = ov[:f0], ov[f0:]
            ng, nu = nv[:new_ffn], nv[new_ffn:]
            return mx.concatenate([g, ng[f0:] * init, u, nu[f0:] * init], axis=0)
        if name.endswith(".down.weight") and "lora" not in name:  # (d, ffn): new columns zero
            ov = old_p[name]
            return mx.concatenate([ov, mx.zeros((ov.shape[0], new_ffn - f0))], axis=1)
        return None
    return _copy_into(new, core, mapping)


def widen_d(core: Core, extra_heads: int = 1, *, moments: bool = False) -> Core:
    cfg = core.cfg
    hd = cfg.d // cfg.heads
    d0, d1 = cfg.d, cfg.d + extra_heads * hd
    H0, H1 = cfg.heads, cfg.heads + extra_heads
    if cfg.lora_rank:
        raise ValueError("widen_d on a LoRA core is not supported")
    if getattr(cfg, "reader_kind", "linear") != "linear":
        raise ValueError("widen_d supports the linear bus reader only (adapter readers have no exact rule yet)")
    # with eps scaled by d0/d1 too, rms_new^2 = (d0/d1) (mean_old + eps_old) exactly, so the gain correction is exact
    new = Core(dataclasses.replace(cfg, d=d1, heads=H1, norm_eps=cfg.norm_eps * d0 / d1))
    gain = 1.0 if moments else (d0 / d1) ** 0.5
    init = 0.0 if moments else 0.02

    def pad_rows(ov, n_rows, value=0.0):
        return mx.concatenate([ov, mx.full((n_rows - ov.shape[0],) + ov.shape[1:], value)], axis=0)

    def pad_cols(ov, n_cols, read=False):
        """New input columns. Reads of the new dims (read=True) start small and random: the new dims hold zeros at
        growth, so this keeps the function exact, while giving the zero-initialised writes a gradient. If reads
        and writes were both zero the new dims could never be used (review of the first version, 2026-11-22)."""
        extra = n_cols - ov.shape[1]
        fill = mx.random.normal((ov.shape[0], extra)) * init if read and init else mx.zeros((ov.shape[0], extra))
        return mx.concatenate([ov, fill], axis=1)

    def mapping(name, old_p, nv):
        ov = old_p.get(name)
        if ov is None:
            return None
        if name.endswith("n1.weight") or name.endswith("n2.weight") or name == "backbone.norm.weight":
            return pad_rows(ov * gain, d1, value=0.0 if moments else 1.0)   # old gains corrected; new dims read 1
        if name in ("backbone.embed.weight", "backbone.pos.weight"):
            return pad_cols(ov, d1)                              # (vocab, d): a write; new dims start at 0
        if name.endswith(".qkv.weight"):                          # (3d, d): q|k|v, each d rows grouped by head
            q, k, v = ov[:d0], ov[d0:2 * d0], ov[2 * d0:]
            nq, nk, nvv = nv[:d1], nv[d1:2 * d1], nv[2 * d1:]
            parts = []
            for old_part, new_part in ((q, nq), (k, nk), (v, nvv)):
                top = pad_cols(old_part, d1, read=True)          # old heads read the (zero) new dims
                parts.append(mx.concatenate([top, new_part[d0:] * init], axis=0))   # new heads
            return mx.concatenate(parts, axis=0)
        if name.endswith(".o.weight"):                            # (d, d): rows write residual, cols read heads
            top = pad_cols(ov, d1)                               # new heads' outputs are not written at first
            return mx.concatenate([top, mx.zeros((d1 - d0, d1))], axis=0)   # nothing written to new dims
        if name.endswith(".up.weight"):                           # (2ffn, d): read residual
            return pad_cols(ov, d1, read=True)
        if name.endswith(".down.weight"):                         # (d, ffn): write residual
            return pad_rows(ov, d1)
        if name == "backbone.head.weight":                        # (vocab, d): reads
            return pad_cols(ov, d1, read=True)
        if name.endswith(".tap.weight"):                          # (w, d): reads
            return pad_cols(ov, d1, read=True)
        if name.endswith(".reader.weight"):                       # (d, w): writes residual
            return pad_rows(ov, d1)
        return None
    return _copy_into(new, core, mapping)


def remove_block(core: Core, i: int, *, moments: bool = False) -> Core:
    cfg = core.cfg
    if not 0 <= i < cfg.layers or cfg.layers <= 1:
        raise ValueError("cannot remove that block")
    # a site after the removed block moves to after the block before it (index -1 = ingress); later sites shift down
    sites = tuple((n, j - 1 if j >= i and j >= 0 else j) for n, j in cfg.sites)
    new = Core(dataclasses.replace(cfg, layers=cfg.layers - 1, sites=sites))

    def mapping(name, old_p, nv):
        parts = name.split(".")
        if parts[:2] == ["backbone", "blocks"]:
            j = int(parts[2])
            src = j if j < i else j + 1
            return old_p[".".join(["backbone", "blocks", str(src)] + parts[3:])]
        return old_p.get(name)
    return _copy_into(new, core, mapping)


def map_optimizer(op, core: Core, opt, grown: Core, make_opt, *args, **kw):
    """A new optimiser for `grown` = op(core, *args, **kw) whose AdamW moments are the old ones mapped through the
    same operator (new entries 0); the step count is copied (MLX AdamW here runs without bias correction, so it does
    not enter the update). A fresh AdamW handicaps every grown candidate against the warm continuation (CPU smoke,
    2026-11-22). It builds scratch cores, so it consumes the global MLX random stream: callers that need the same
    candidate fills with and without mapping reseed before each candidate (grow_train does)."""
    from mlx.utils import tree_unflatten
    flat = dict(tree_flatten(opt.state))
    trees = {}
    for k in ("m", "v"):
        src = Core(core.cfg)
        src.load_weights([(n, flat[f"{n}.{k}"]) for n, _ in tree_flatten(core.parameters())])
        trees[k] = dict(tree_flatten(op(src, *args, moments=True, **kw).parameters()))
    new = make_opt()
    new.init(grown.trainable_parameters())
    state = tree_unflatten([(f"{n}.{k}", trees[k][n]) for n, _ in tree_flatten(grown.parameters()) for k in ("m", "v")])
    state["step"] = opt.state["step"]
    state["learning_rate"] = new.state["learning_rate"]
    new.state = state
    return new


def prune_ffn(core: Core, keep, *, moments: bool = False) -> Core:
    """Keep, in every block b, the FFN hidden units keep[b] (a list of indices, the same length k for every block,
    since CoreConfig.ffn is global). Not function-preserving: the removed units' contributions are dropped, which
    the pruner chooses to be the least important. moments=True maps an optimiser moment tree the same way."""
    cfg = core.cfg
    k = len(keep[0])
    if any(len(kb) != k for kb in keep) or len(keep) != cfg.layers or not 0 < k < cfg.ffn:
        raise ValueError("keep must give k < ffn unit indices for every block")
    if cfg.lora_rank:
        raise ValueError("prune_ffn on a LoRA core is not supported")
    new = Core(dataclasses.replace(cfg, ffn=k))
    f0 = cfg.ffn

    def mapping(name, old_p, nv):
        parts = name.split(".")
        if parts[:2] == ["backbone", "blocks"] and parts[3] in ("up", "down"):
            idx = mx.array(sorted(keep[int(parts[2])]))
            ov = old_p[name]
            if parts[3] == "up":                                  # (2*ffn, d): [gate; up] halves
                return mx.concatenate([ov[:f0][idx], ov[f0:][idx]], axis=0)
            return ov[:, idx]                                     # down (d, ffn)
        return None
    return _copy_into(new, core, mapping)

