"""ENS A0 tiny core: an ordinary decoder-only backbone plus a Neural ABI bus.

The backbone is a plain pre-norm causal transformer. The bus adds, at named
sites, a tap R (d -> w) and a reader U (w -> d) with a scalar gate. A module
consumes the tap view at the query position and returns a w-wide message. Only
the core adds the message to its private residual, at the field position, after
projecting the final site update onto an L2 ball of radius tau * ||h||
(spec v0.4 section 3.1, bounded profile, explicit zero rule at h = 0).

With no bus call the core runs the ordinary backbone graph: no tap, reader,
gate or bound nodes are built (structural bypass, CORE-001).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from research.ens import vocab as V
from research.ens.families import SEQ_LEN


@dataclass(frozen=True)
class CoreConfig:
    vocab: int = V.VOCAB_SIZE
    d: int = 256
    layers: int = 6
    heads: int = 4
    ffn: int = 1024
    seq_len: int = SEQ_LEN
    bus_width: int = 64
    tau: float = 0.5
    # bus applied after block index i; -1 = ingress (before block 0, on the embedding)
    sites: tuple = (("ingress", -1), ("early", 1), ("middle", 3), ("late", 5))
    reader_init_scale: float = 1.0
    reader_kind: str = "linear"      # "mlp": a small nonlinear adapter reader (the C6A retrofit control);
                                     # "mlp_state": the adapter also reads the field row's state (C6A-S)
    adapter_hidden: int = 256
    norm_eps: float = 1e-5           # RMSNorm epsilon; growth by widen_d scales it by d_old / d_new (exactness)
    lora_rank: int = 0               # >0: the C6L arm - low-rank updates on the blocks after lora_site, applied
    lora_site: str = ""              # only on the bus path (a tile is admitted); the ordinary graph is untouched

    def to_json(self) -> dict:
        d = asdict(self)
        d["sites"] = [list(s) for s in self.sites]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "CoreConfig":
        d = dict(d)
        d["sites"] = tuple(tuple(s) for s in d["sites"])
        return cls(**d)


class Block(nn.Module):
    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.heads = cfg.heads
        self.n1 = nn.RMSNorm(cfg.d, eps=cfg.norm_eps)
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.o = nn.Linear(cfg.d, cfg.d, bias=False)
        self.n2 = nn.RMSNorm(cfg.d, eps=cfg.norm_eps)
        self.up = nn.Linear(cfg.d, 2 * cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, cfg.d, bias=False)

    def __call__(self, h, lora=None):
        B, T, D = h.shape
        hd = D // self.heads
        x = self.n1(h)
        qkv = self.qkv(x) + (lora.qkv(x) if lora is not None else 0)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q, k, v = (t.reshape(B, T, self.heads, hd).transpose(0, 2, 1, 3) for t in (q, k, v))
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5, mask="causal")
        a = a.transpose(0, 2, 1, 3).reshape(B, T, D)
        h = h + self.o(a) + (lora.o(a) if lora is not None else 0)
        x = self.n2(h)
        g, u = mx.split(self.up(x) + (lora.up(x) if lora is not None else 0), 2, axis=-1)
        m = nn.silu(g) * u
        return h + self.down(m) + (lora.down(m) if lora is not None else 0)


class LoRA(nn.Module):
    """y = B(A x) * alpha / r, with B zero-initialised so the update starts at exactly 0."""

    def __init__(self, d_in, d_out, r, alpha=8.0):
        super().__init__()
        self.a = nn.Linear(d_in, r, bias=False)
        self.b = nn.Linear(r, d_out, bias=False)
        self.b.weight = mx.zeros_like(self.b.weight)
        self.scale = alpha / r

    def __call__(self, x):
        return self.b(self.a(x)) * self.scale


class BlockLoRA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        r = cfg.lora_rank
        self.qkv = LoRA(cfg.d, 3 * cfg.d, r)
        self.o = LoRA(cfg.d, cfg.d, r)
        self.up = LoRA(cfg.d, 2 * cfg.ffn, r)
        self.down = LoRA(cfg.ffn, cfg.d, r)


class Backbone(nn.Module):
    """The ordinary graph. Holds every weight the all-base path needs and nothing else."""

    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d)
        self.blocks = [Block(cfg) for _ in range(cfg.layers)]
        self.norm = nn.RMSNorm(cfg.d, eps=cfg.norm_eps)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)

    def embed_tokens(self, tokens):
        T = tokens.shape[1]
        return self.embed(tokens) + self.pos(mx.arange(T))

    def __call__(self, tokens):
        h = self.embed_tokens(tokens)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))


class SitePort(nn.Module):
    def __init__(self, cfg: CoreConfig):
        super().__init__()
        w, d = cfg.bus_width, cfg.d
        self.tap = nn.Linear(d, w, bias=False)
        self.reader = nn.Linear(w, d, bias=False)
        self.reader.weight = mx.random.normal((d, w)) * (cfg.reader_init_scale / w ** 0.5)
        self.gate = mx.array([1.0])                    # open, finite at onboarding (TRAIN-001)
        if cfg.reader_kind in ("mlp", "mlp_state"):
            # bias-free output: a zero message must write nothing, as with the linear reader (review)
            inp = w + (d if cfg.reader_kind == "mlp_state" else 0)
            self.state_reader = cfg.reader_kind == "mlp_state"   # plain attribute: chosen by config (review)
            self.adapter = nn.Sequential(nn.Linear(inp, cfg.adapter_hidden), nn.SiLU(),
                                         nn.Linear(cfg.adapter_hidden, d, bias=False))
            self.adapter.layers[-1].weight = self.adapter.layers[-1].weight * 0.01   # small first writes

    def read(self, msg, h_row=None):
        if "adapter" not in self:
            return self.reader(msg)
        # The first projection has a bias, so both adapters require an explicit
        # zero-message gate to keep ordinary training rows on the base path.
        present = mx.any(msg != 0, axis=-1, keepdims=True)
        inp = mx.concatenate([msg, h_row], axis=-1) if self.state_reader else msg
        return mx.where(present, self.adapter(inp), 0.0)


_EPS2 = 1e-24

Message = Union[mx.array, Callable[[mx.array], mx.array]]


@dataclass
class BusCall:
    """One exclusive writer at one site for a batch.

    message: (B, w) array, or a callable taking the tap view (B, w) at query_idx
    and returning (B, w). query_idx / field_idx: (B,) int positions.
    """
    site: str
    query_idx: mx.array
    field_idx: mx.array
    message: Message
    record: dict = field(default_factory=dict)        # filled with per-call diagnostics


def bounded_update(delta, h_row, tau: float):
    """Project delta (B, d) onto the L2 ball of radius tau * ||h_row||; zero when h_row = 0."""
    # Smooth norms: d||x||/dx is 0/0 at x = 0, which made a closed gate or a zero
    # reader return NaN gradients for every bus parameter (caught by TRAIN-001).
    h2 = mx.sum(h_row * h_row, axis=-1, keepdims=True)
    radius = mx.where(h2 > 0, tau * mx.sqrt(h2 + _EPS2), 0.0)           # explicit zero rule at h = 0
    norm = mx.sqrt(mx.sum(delta * delta, axis=-1, keepdims=True) + _EPS2)
    scale = mx.minimum(1.0, radius / norm)
    return delta * scale


def _gather_rows(h, idx):
    B = h.shape[0]
    return h[mx.arange(B), idx]


class Core(nn.Module):
    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg)
        self.ports = {name: SitePort(cfg) for name, _ in cfg.sites}
        self._site_after = {i: name for name, i in cfg.sites if -1 <= i < cfg.layers}
        if cfg.lora_rank > 0:
            after = dict(cfg.sites)[cfg.lora_site]
            if after + 1 >= cfg.layers:
                raise ValueError(f"lora_site {cfg.lora_site!r} has no blocks after it")
            # keys must not be bare digits: MLX tree utils read those as list indices on load
            self.lora = {f"b{i}": BlockLoRA(cfg) for i in range(after + 1, cfg.layers)}

    def __call__(self, tokens, bus: Optional[BusCall] = None, *, check_finite: bool = False):
        if bus is None:
            return self.backbone(tokens)               # structural bypass: the ordinary graph
        if bus.site not in self._site_after.values():
            raise KeyError(f"unknown or unreachable site {bus.site!r} for {self.cfg.layers} layers")
        bb = self.backbone
        h = bb.embed_tokens(tokens)
        if self._site_after.get(-1) == bus.site:
            h = self._apply(h, bus, check_finite)
        # the LoRA belongs to its own site's port: other ports (e.g. the untrained-port control) run without it
        lora = self.lora if ("lora" in self and bus.site == self.cfg.lora_site) else {}
        for i, blk in enumerate(bb.blocks):
            h = blk(h, lora.get(f"b{i}"))          # the tile's LoRA rides only on the bus path
            if self._site_after.get(i) == bus.site:
                h = self._apply(h, bus, check_finite)
        return bb.head(bb.norm(h))

    def _site_index(self, site):
        idx = {name: i for name, i in self.cfg.sites}.get(site)
        if idx is None or idx >= self.cfg.layers:
            raise KeyError(f"unknown or unreachable site {site!r} for {self.cfg.layers} layers")
        return idx

    def prefix(self, tokens, site):
        """The frozen state at a bus site before any write: embed and the blocks up to the site. With suffix(),
        prefix + suffix equals __call__ with a bus; a trainer can cache prefix() for a fixed row pool."""
        after = self._site_index(site)
        bb = self.backbone
        h = bb.embed_tokens(tokens)
        for i in range(after + 1):
            h = bb.blocks[i](h)                        # no LoRA here: it rides only on blocks after the site
        return h

    def suffix(self, h, bus: BusCall, *, check_finite: bool = False):
        """The bus write at the site and everything after it, from a prefix() state."""
        after = self._site_index(bus.site)
        h = self._apply(h, bus, check_finite)
        lora = self.lora if ("lora" in self and bus.site == self.cfg.lora_site) else {}
        for i in range(after + 1, self.cfg.layers):
            h = self.backbone.blocks[i](h, lora.get(f"b{i}"))
        return self.backbone.head(self.backbone.norm(h))

    def _apply(self, h, bus: BusCall, check_finite: bool):
        port = self.ports[bus.site]
        msg = bus.message
        if callable(msg) and getattr(msg, "reads_sequence", False):
            # a sequence-reading tile (attention tile, H19) sees the tap view of every position *before* the
            # field; the field and later positions are zeroed so it cannot read what it is about to write
            T = h.shape[1]
            before = (mx.arange(T)[None, :] < bus.field_idx[:, None]).astype(h.dtype)[..., None]
            view = h if getattr(msg, "reads_raw", False) else port.tap(h)   # raw: the full d-wide state
            msg = msg(view * before, before[..., 0])
        elif callable(msg):
            msg = msg(port.tap(_gather_rows(h, bus.query_idx)))
        if check_finite and not bool(mx.all(mx.isfinite(msg)).item()):
            raise ValueError("non-finite bus message refused")
        h_row = _gather_rows(h, bus.field_idx)
        delta = bounded_update(port.gate * port.read(msg, h_row), h_row, self.cfg.tau)
        T = h.shape[1]
        mask = (mx.arange(T)[None, :] == bus.field_idx[:, None]).astype(h.dtype)[..., None]
        return h + mask * delta[:, None, :]


def answer_logits(logits, ans_idx):
    return _gather_rows(logits, ans_idx)


def value_logits(logits, ans_idx):
    """Restrict answer logits to the 64 value tokens (every A0 answer is a value)."""
    return answer_logits(logits, ans_idx)[:, V.V0:V.V0 + V.N_VALUES]
