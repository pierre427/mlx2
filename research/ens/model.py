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

import math

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
    loop_body: tuple = ()            # H45 loop port: (start, end) blocks [start, end) may run k >= 1 times per row
                                     # (__call__ passes=); () = no loop port. Blocks before start run once, as do
                                     # blocks from end on.
    loop_gain: str = ""              # H45b (Pierre 2026-11-24): "" plain loop; "decay": round k's update scaled by
                                     # gamma^(k-1); "decaygain": also by (c_k / c_(k-1))^kappa, c = attention
                                     # concentration (sharpening). Needs uniform passes (max_passes).

    def to_json(self) -> dict:
        d = asdict(self)
        d["sites"] = [list(s) for s in self.sites]
        d["loop_body"] = list(self.loop_body)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "CoreConfig":
        d = dict(d)
        d["sites"] = tuple(tuple(s) for s in d["sites"])
        d["loop_body"] = tuple(d.get("loop_body", ()))
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

    def __call__(self, h, lora=None, conc=False, qmask=None):
        """conc=True (H45b): explicit attention, also returning each row's attention concentration, the mean over
        heads and query positions of exp(-entropy) of the attention distribution: (B,). 1 = every query attends one
        key; small = diffuse. qmask (B, T): the query positions that count (review 2026-11-24: padding queries were
        40-75% of the mean and could act as a learned halting signal)."""
        B, T, D = h.shape
        hd = D // self.heads
        x = self.n1(h)
        qkv = self.qkv(x) + (lora.qkv(x) if lora is not None else 0)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q, k, v = (t.reshape(B, T, self.heads, hd).transpose(0, 2, 1, 3) for t in (q, k, v))
        if conc:
            s = (q @ k.transpose(0, 1, 3, 2)) * hd ** -0.5
            causal = mx.triu(mx.full((T, T), -1e9, dtype=s.dtype), k=1)
            p = mx.softmax((s + causal).astype(mx.float32), axis=-1)
            ent = -mx.sum(p * mx.log(p + 1e-12), axis=-1)                    # (B, H, T)
            e = mx.mean(mx.exp(-ent), axis=1)                               # (B, T) over heads
            if qmask is None:
                c = mx.mean(e, axis=-1)
            else:
                w = qmask.astype(e.dtype)
                c = mx.sum(e * w, axis=-1) / mx.maximum(mx.sum(w, axis=-1), 1.0)
            a = (p.astype(v.dtype) @ v)
        else:
            a = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5, mask="causal")
        a = a.transpose(0, 2, 1, 3).reshape(B, T, D)
        h = h + self.o(a) + (lora.o(a) if lora is not None else 0)
        x = self.n2(h)
        g, u = mx.split(self.up(x) + (lora.up(x) if lora is not None else 0), 2, axis=-1)
        m = nn.silu(g) * u
        out = h + self.down(m) + (lora.down(m) if lora is not None else 0)
        return (out, c) if conc else out


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


GAMMA0_LOGIT = math.log(0.9 / 0.1)                     # H45b: gamma's neutral value 0.9
KAPPA_SCALE = 30.0                                      # H45b: kappa = KAPPA_SCALE * loop_kappa_raw


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
        if cfg.loop_gain:
            if cfg.loop_gain not in ("decay", "decaygain") or not cfg.loop_body:
                raise ValueError("loop_gain needs a loop body and is 'decay' or 'decaygain'")
            # gamma = sigmoid(off + GAMMA0_LOGIT): off 0 is gamma 0.9, so AdamW weight decay pulls toward the neutral
            # start (pure decay at 0.9), not toward gamma 0.5. kappa = KAPPA_SCALE * raw starts (and decays) at 0;
            # the scale lets kappa reach the ~10-30 needed (log-ratios of concentration are ~0.01-0.03 per round on
            # trained cores) at an AdamW step size of ~lr (review 2026-11-24). Names changed with this
            # parametrisation so an older checkpoint cannot load with a different meaning.
            self.loop_gamma_off = mx.array([0.0])
            if cfg.loop_gain == "decaygain":
                self.loop_kappa_raw = mx.array([0.0])

    def __call__(self, tokens, bus: Optional[BusCall] = None, *, check_finite: bool = False, passes=None,
                 max_passes: Optional[int] = None):
        if passes is not None:
            return self.looped(tokens, passes, bus, check_finite=check_finite, max_passes=max_passes)
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

    def kappa(self):
        return KAPPA_SCALE * self.loop_kappa_raw[0] if "loop_kappa_raw" in self else None

    def gamma(self):
        return mx.sigmoid(self.loop_gamma_off + GAMMA0_LOGIT)[0]

    def _looped_gain(self, tokens, P, bus, check_finite, return_gains):
        """H45b: every row runs P passes; pass k >= 2 applies its update scaled, h <- h + g_k (body(h) - h), with
        log g_k = (k - 1) log gamma [+ kappa log(c_k / c_(k-1))], c_k the attention concentration of the body blocks
        in pass k over the row's real query positions (not PAD). Pass 1 is the ordinary pass. Concentration is only
        computed for decaygain (decay uses the fused attention, as fixed6). c is one value per row over all its real
        queries, so a round's gain at position t depends on attention at later real positions: fine for answer-at-
        the-end tasks, not usable autoregressively as is. return_gains: also return g (P - 1, B) and c (P, B) (c is
        None for decay)."""
        start, end = self.cfg.loop_body
        bb = self.backbone
        use_c = "loop_kappa_raw" in self
        qmask = tokens != V.PAD
        h = bb.embed_tokens(tokens)
        if bus is not None and self._site_after.get(-1) == bus.site:
            h = self._apply(h, bus, check_finite)

        def body(x):
            cs = []
            for i in range(start, end):
                if use_c:
                    x, ci = bb.blocks[i](x, conc=True, qmask=qmask)
                    cs.append(ci)
                else:
                    x = bb.blocks[i](x)
            return x, (sum(cs) / len(cs) if use_c else None)
        for i in range(start):
            h = bb.blocks[i](h)
            if bus is not None and self._site_after.get(i) == bus.site:
                h = self._apply(h, bus, check_finite)
        if bus is not None and any(self._site_after.get(i) == bus.site for i in range(start, end)):
            raise ValueError("H45b writes the bus before the loop body")
        h, c_prev = body(h)
        log_gamma = mx.log(self.gamma())
        kappa = self.kappa()
        gains, concs = [], [c_prev]
        for k in range(2, P + 1):
            x, c = body(h)
            lg = (k - 1) * log_gamma * mx.ones((tokens.shape[0],))      # per row, also for pure decay
            if use_c:
                lg = lg + kappa * (mx.log(c + 1e-6) - mx.log(c_prev + 1e-6))
                c_prev = c
                concs.append(c)
            g = mx.exp(mx.clip(lg, -20.0, 2.0))          # bounded: a round can at most ~7x its update
            h = h + g[:, None, None] * (x - h)
            gains.append(g)
        for i in range(end, self.cfg.layers):
            h = bb.blocks[i](h)
        logits = bb.head(bb.norm(h))
        if not return_gains:
            return logits
        return logits, mx.stack(gains), (mx.stack(concs) if use_c else None)

    def looped(self, tokens, passes, bus: Optional[BusCall] = None, *, check_finite: bool = False,
               return_states: bool = False, max_passes: Optional[int] = None, return_gains: bool = False):
        """H45: run the loop body passes[b] >= 1 times for row b (passes: (B,) int array). The bus, if any, writes
        once, at its site, during the first pass only; its site must lie before the loop body ends. Rows with fewer
        passes keep the state they had after their last pass (per-row select), so a batch runs max(passes) passes.
        return_states: also return the final-norm state after every pass k = 1..P, (P, B, T, d), for halting tiles.
        These are the states of *all* rows after k passes (not each row's selected state), so callers that read them
        should use uniform passes."""
        if not self.cfg.loop_body:
            raise ValueError("passes= needs a core with a loop port (CoreConfig.loop_body)")
        start, end = self.cfg.loop_body
        if max_passes is None:                         # outside mx.compile; compiled callers pass max_passes
            P = int(mx.max(passes).item())
            if int(mx.min(passes).item()) < 1:
                raise ValueError("passes must be >= 1")
        else:
            P = max_passes                             # the caller guarantees 1 <= passes <= max_passes
        if bus is not None and self._site_index(bus.site) >= start:
            raise ValueError(f"bus site {bus.site!r} is not before the loop body; H45 writes once, before the loop")
        if "lora" in self:
            raise ValueError("looped(): LoRA cores are not supported")
        if self.cfg.loop_gain:
            if return_states:
                raise ValueError("return_states is for plain loop cores (halting tiles); gain cores return gains")
            if max_passes is None and int(mx.min(passes).item()) != P:      # no .item() under mx.compile
                raise ValueError("loop_gain cores run the same number of passes for every row")
            return self._looped_gain(tokens, P, bus, check_finite, return_gains)
        if return_gains:
            raise ValueError("return_gains needs a loop_gain core")
        bb = self.backbone
        h = bb.embed_tokens(tokens)
        if bus is not None and self._site_after.get(-1) == bus.site:
            h = self._apply(h, bus, check_finite)
        for i in range(end):
            h = bb.blocks[i](h)
            if bus is not None and self._site_after.get(i) == bus.site:
                h = self._apply(h, bus, check_finite)
        sel = h
        states = []

        def tail(x):
            for i in range(end, self.cfg.layers):
                x = bb.blocks[i](x)
            return x
        if return_states:
            states.append(bb.norm(tail(h)))
        for k in range(2, P + 1):
            for i in range(start, end):
                h = bb.blocks[i](h)
            sel = mx.where((passes >= k)[:, None, None], h, sel)
            if return_states:
                states.append(bb.norm(tail(h)))
        logits = bb.head(bb.norm(tail(sel)))
        return (logits, mx.stack(states)) if return_states else logits

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
