"""H56 + H56b: a depth-matched side tile beside a frozen donor (Pierre, 2026-11-24: "hold 2 tiles open. A frozen model
we do not write to at all. A new randomized tile with a neural net of some depth matching the donor ... read the hidden
state at each layer, then apply it to the tile, have the tile evolve the hidden state ... Remove the model's
contributions. Advance to next layer, rinse and repeat. We superimpose the model's state to the tile"; "if we can use a
bus intermediate hidden state projector, will allow for pluggable tiles and modules for hidden state
interoperability"; H56b: "allow the tile's neuron down dims to be able to align to the model's neurons if they are best
match"). Ledger rows H56 / H56b in wiki/docs/ens/hypotheses.md; the H51 failure it answers is the 2026-11-24 21:15 log
entry (the corpus text was memorised by the LoRA after the site; the memory carried 0.02 nats).

Design (Qwen3 first, research/ens/qwen3.py):
  aligned layers  lay = [0, m, 2m, ..., L] (m = --every; donor state h_l = the input of donor block l, h_L its output).
                  The tile has one block per group of m donor blocks (n_t = ceil(L / m)); tile block j reads at lay[j]
                  and its output s_{j+1} is aligned with lay[j+1].
  ladder in       z_l = P_l(unit_rms(h_l) - mu_l); position 0 (the attention sink) is not read: the tile sees a
                  fixed unit-RMS sink vector there (zeros blew up the backward, see SideTile.__init__). P_l: d -> w,
                  initialised to the top-w principal directions of the donor's centred states at that layer (scaled
                  so z has unit mean square over all w dimensions; per-dimension variance follows the eigenvalues),
                  mu_l the fixed per-layer mean (as H51's centring). in_0 = z_0 always;
                  in_j = alpha * z_{lay[j]} + (1 - alpha) * s_j for j >= 1 (alpha 1: teacher forced, every block
                  independent, trains in parallel; alpha 0: free running from the embedding ladder only).
  tile block      a Qwen3 block at width w (attention + SwiGLU MLP, causal, RoPE), --heads heads, MLP width 3w.
  write           one gate back: delta = U(s_{n_t}) (U: w -> d, zero at init) added at --write-site (default the input
                  of the last donor block; 'mid' = block L // 2; or an index), h <- h + bounded(gate * delta, h, tau)
                  as the bus (LB.bounded); --gate-per-layer writes U_j(s_{j+1}) at its aligned layer lay[j+1] (the
                  last one, aligned with the donor's output, after the last block at the head's input; no block is
                  written twice; a variant: more parameters, gradient through the whole donor). Position 0 is never written. With U
                  at zero the untrained tile is the original model bit for bit.
  two passes      pass 1 runs the original donor (no write) and keeps the states the tile reads, so the tile always
                  reads the ORIGINAL donor at every depth even when it writes at mid depth; pass 2 reruns the donor
                  from the first write site with the write. Pass 1 also gives the KL reference logits for free.
  no donor LoRA   DONOR_LORA = False. H51's lesson: a LoRA after the site soaks up the content, so side arms never
                  carry one; every donor block is called with lora=None (tested).
  bus seam        the tile reads the donor only through encode() (P_l) and writes only through U, so a later per-core
                  projector swap (portability, phase 2) replaces P / U without touching the tile blocks. Not built.

Phases (one global step counter: steps [0, steps_a) are phase A, then --steps of phase B):
  A  imitation on ordinary text (CT.text_chunks windows of --a-len tokens). Target, in state form: the projected next
     donor state, loss = mean over tile blocks of ||s_{j+1} - sg(z_{lay[j+1]})||^2 / ||sg(z_{lay[j+1]})||^2 (positions
     >= 1). At alpha 1 this is exactly predicting the projected per-layer update P(x_{l+1}) - P(x_l) (in = z_l); the
     state form is chosen because the same number at alpha 0 IS the free-running drift. P_l, U, gate are frozen in A
     (a trainable P would let the loss collapse by shrinking the bus). alpha schedule: 1 for the first --alpha-hold of
     A, linear 1 -> alpha_b over the middle, alpha_b for the last --alpha-hold, so phase A ends in the mode B uses
     (--anneal-to-zero: the old 1 -> 0). This anneal exists only to fix exposure bias in imitation (a block trained
     teacher-forced must also cope with the tile's own, imperfect upstream states); phase B's alpha is a separate
     setting. Phase A draws its batches from its own rng, so the phase-B batch stream is identical in every arm and
     equals correction's and H51's (same chunks, text rows and seed; tested). Logged: tf_err (alpha 1) and drift (alpha 0), per-layer relative state error on
     held ordinary text.
  B  corpus: next-token NLL on the H51 corpus chunks (CT.doc_chunks) + kl_weight x KL to the original on ordinary text
     rows (as H51, no routing loss), donor frozen, alpha held at --alpha-b (default 0.5: Pierre's design keeps
     superimposing the donor's state during training, "We superimpose the model's state to the tile as part of
     training for that step", so every ladder keeps reading the donor at depth). Trainable in B: tile blocks, P_0 ..
     P_{n_t - 1}, U, gate (P_{n_t} is an imitation target only).
     Variant --alpha-b 0, "free running": the tile then reads only the embedding ladder and is in effect a separate
     small model (~24.4M trainable on 0.6B) that must compute everything it writes itself: the H50 hidden-state
     injection risk. Explicit opt-in only.
Arms (--arm): correction (no phase A; B straight away, the ladder side-tuning control), imitate (A then B, Pierre's
design), and the H56b neuron-aligned arms (A then B) on the tile MLP's write (down) vectors:
  alignment across widths: a donor MLP down column c of donor block l (writing into h_{l+1}) is mapped into tile space
     as D_c = P_{l+1} W_down[:, c] / r_{l+1} (r = the mean RMS of h_{l+1}, since the ladder reads unit-RMS states),
     with stop-gradient on P. Tile and donor write vectors are then compared by cosine in tile space; --width d aligns
     at donor width with P a full PCA rotation.
  start state (align_init): hard / act / grow neurons start with their write vector set to a distinct donor column
     (the f columns with the largest ||D_c||) and tied, so ties start in place and the run measures whether they hold
     or break (from random init the best cosine is ~0.2 at w 256 and nothing would reach tau_on 0.5). Adam moments of
     a write vector are reset whenever its neuron ties or unties. tied_frac is logged at every match step; a warning is
     logged if it is 0 at the first match step.
  align_hard  every --match-every steps a tile neuron ties to its best donor column (greedy unique top-1 cosine, or
              --match hungarian with scipy) if cosine >= --tau-on; a tied neuron unties when the cosine of its shadow
              (the free weight, trained straight-through while tied) to its donor column drops below --tau-off
              (hysteresis: tau_off < tau_on). Tied neurons write exactly the donor-derived column (frozen in the
              forward); a newly tied neuron's shadow is reset to that column. Tied fraction is logged.
  align_soft  down vector = sum_k coef_ik D_{idx_ik} (K = --soft-k donor columns, support refreshed every --match-every
              steps to the top-K by |cosine| with a least-squares refit of the coefficients) + a rank --soft-rank free
              part. Supports start as K distinct random donor columns per neuron (seeded), coefficients scaled to a
              default-init column norm, the free part at 0.1x its default init (full-rank dictionary from step 0).
              Fewer trainable parameters (reported).
  align_act   (simpler first version, FLAG) as align_hard but ties by activation correlation between the tile neuron
              and donor neurons over the current batch's tokens (2 rows), thresholds on correlation.
  align_grow  (simpler first version, FLAG) starts with every tile neuron tied to a distinct donor column (largest
              ||D_c||); whenever the loss EMA improved by less than --grow-eps over a matching window, the --grow-frac of
              tied neurons whose shadows moved furthest are freed (never re-tied), their free weight starting at the
              donor column they replace. No grow decision while phase A's alpha anneals; at most --grow-max-frac
              (0.5) of all neurons are freed over a run; frees are logged (per step and total).
  Merge-back exactness (tile folded into donor weights) is a later step: merge_back() is the hook.
Equal parameters: at w 256, every 1, heads 4, MLP 768 on Qwen3-0.6B (d 1024, 28 layers): 28 tile blocks 23.9M + 29
  projectors 7.6M + U 0.26M = 31.7M stored vs the H51 tile's 31.87M. param_report() prints stored / trainable per
  phase / effective (minus tied columns). At the default alpha_b 0.5 (ladders active) the trainable count in B is
  31.48M (align_soft 26.60M), near H51; at the free-running alpha_b 0 it is ~24.4M (P_1.. unused, frozen).
Fairness of steps: --steps counts phase-B (corpus) steps only, so imitate and the aligned arms get --steps-a (2000)
  extra phase-A steps without corpus. A compute-matched correction is `--arm correction --compute-matched`: it adds
  round(steps_a x --a-cost-ratio) phase-B steps. A phase-A step (donor forward on 6 x 512 + the narrow tile's backward,
  no head, no donor backward) is estimated at ~0.5x a phase-B step: 12,000 + 2,000 x 0.5 = 13,000 steps (13,300 at a
  ratio of 0.65). Train receipts log phase_seconds and the measured a_cost_ratio; rerun the control with the measured
  ratio.
Kill measures (H56 / H56b): imitate-then-corpus no better than correction, or no better than LoRA only on recall;
  aligned arms neither faster at imitation nor better at recall than the free tile. Evaluation uses H51's nll_eval and
  579-question qa_eval unchanged (fields base / tile / lora_only; for a side tile lora_only = the tile's write zeroed,
  which equals base because side arms carry no donor LoRA: a built-in exactness check).
Memory: --accum K (1 .. batch rows) cuts every step into K contiguous micro-batches of nearly equal size
  (np.array_split), run one at a time: phase A over its rows in batch order, plus one extra no-gradient donor pass
  for the layer denominators; phase B over the corpus rows then the text rows. Every micro-batch uses the whole
  batch's denominators, so loss and gradient equal the default step (float order only); then one update (no gradient
  clipping, as the default step). K = rows is one row per micro-batch. H56b matching runs on the full batch either way. Eval batch sizes:
  --nll-batch (4 chunks x 512), --text-batch (16 x 128), --qa-batch (8).
Held-out tracking (--track-every N, default 0 = off): every N phase-B steps the log row carries the tile's mean
  next-token NLL at the run's alpha on a fixed sample of 8 trained-document and 8 held-out-document chunks (drawn once,
  seed 80000 + seed, from CT.nll_eval's chunk lists; the same chunks for every run and resume at one seed, different chunks at different seeds), the base model's NLL on them (computed at launch), gain_train,
  gain_held and gain_train_minus_held; the receipt gathers these rows in `track`. Batches of 4, no gradients. The
  history lives in the log, so it survives resumes.
Non-finite guard: a non-finite loss or gradient (before the update) or parameter (after it) writes nonfinite.json +
  nonfinite_tile.safetensors and exits 4 (not 3, so a runner does not retry it).
Resumable (autosave, SIGTERM -> save + exit 3, caught from the start of train() including the model load,
--max-steps-this-launch; phase A and B rng streams both saved), memory gate (--gate-gib), swap guard;
receipts name the base and tile checkpoints <artifact>@<fingerprint> and the source revision.

  python -m research.ens.side_tile train --corpus $ENS_PRIVATE_CORPUS/corpus --arm imitate --out <dir>
  python -m research.ens.side_tile eval --run <dir> [--questions $ENS_PRIVATE_CORPUS/eval]
  python -m research.ens.side_tile params [--model Qwen/Qwen3-0.6B]      (reads config.json only)
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import signal
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from research.ens import common as C
from research.ens import corpus_tile as CT
from research.ens import lm_bus as LB
from research.ens import physics_bus as PB
from research.ens import qwen3 as Q

RECIPE = "ens.h56.side_tile.v2"           # v2 (2026-11-25): fixed unit-RMS sink vector at position 0 (v1 fed zeros: NaN)
DONOR_LORA = False            # side arms never carry a LoRA on the donor (H51: the LoRA after the site took the content)
ARMS = {"correction": None, "imitate": None, "align_hard": "hard", "align_soft": "soft", "align_act": "act",
        "align_grow": "grow"}
H51_TILE_PARAMS = 31_866_881   # ens-h51-corpustile-qwen3-0.6b-site8-s0 trainable_params (evidence/h51-0p6b-20261124)
_TIED_MODES = ("hard", "act", "grow")


# ------------------------------------------------------------------ geometry
def aligned_layers(L: int, every: int) -> list[int]:
    n_t = -(-L // every)
    return [min(j * every, L) for j in range(n_t + 1)]


def write_sites(L: int, lay: list[int], site, per_layer: bool) -> list[tuple[int, int]]:
    """[(k, i)]: tile output s_k (1-based) is written at the input of donor block i (i = L: after the last block, the
    head's input). Per layer: s_{j+1} at its aligned layer lay[j+1], so the last one (aligned with the donor's output)
    lands at L and no block gets two writes."""
    n_t = len(lay) - 1
    if per_layer:
        return [(j + 1, lay[j + 1]) for j in range(n_t)]
    i = L - 1 if site == "last" else L // 2 if site == "mid" else int(site)
    if not 1 <= i <= L - 1:
        raise SystemExit(f"--write-site {site}: must be in 1..{L - 1}")
    return [(n_t, i)]


def donor_pass(model, tokens, keep, logits=False):
    """Pass 1, the original donor (no write, no LoRA): ({l: h_l for l in keep}, logits or None). h_l is the input of
    block l (h_L the last block's output). Stops after the deepest kept layer when no logits are wanted."""
    layers = model.model.layers
    L = len(layers)
    deepest = max(keep)
    h = model.model.embed_tokens(tokens)
    st = {}
    for i in range(L):
        if i in keep:
            st[i] = h
        if not logits and i >= deepest:
            return st, None
        h = layers[i](h)
    if L in keep:
        st[L] = h
    return st, (model.head(h) if logits else None)


def donor_mlp_act(blk, h):
    """A donor block's MLP hidden activation silu(gate) * up at input h (for align_act)."""
    h2 = h + blk.self_attn(blk.input_layernorm(h), "causal", None)
    x = blk.post_attention_layernorm(h2)
    return nn.silu(blk.mlp.gate_proj(x)) * blk.mlp.up_proj(x)


def _not_sink(T):
    return (mx.arange(T) > 0).astype(mx.float32)[None, :, None]


# ------------------------------------------------------------------ the tile
class Proj(nn.Module):
    def __init__(self, w0):
        super().__init__()
        self.weight = w0

    def __call__(self, x):
        return x @ self.weight.T


class TileMLP(nn.Module):
    """SwiGLU at tile width; the write (down) matrix is either free (down_proj), tied-capable (down_proj = the shadow,
    tie_mask / tie_idx fixed), or a soft dictionary mix (coef over soft_idx + a low-rank free part)."""

    def __init__(self, w, f, mode, soft_k=8, soft_rank=16):
        super().__init__()
        self.mode = mode
        self.gate_proj = nn.Linear(w, f, bias=False)
        self.up_proj = nn.Linear(w, f, bias=False)
        fixed = []
        if mode == "soft":
            self.coef = 0.02 * mx.random.normal((f, soft_k))
            self.free_a = nn.Linear(f, soft_rank, bias=False)
            self.free_b = nn.Linear(soft_rank, w, bias=False)
            self.soft_idx = mx.zeros((f, soft_k), dtype=mx.int32)
            fixed = ["soft_idx"]
        else:
            self.down_proj = nn.Linear(f, w, bias=False)
            if mode in _TIED_MODES:
                self.tie_mask = mx.zeros((f,))
                self.tie_idx = mx.zeros((f,), dtype=mx.int32)
                fixed = ["tie_mask", "tie_idx"]
        self._fixed_keys = fixed
        if fixed:
            self.freeze(recurse=False, keys=fixed)

    def __call__(self, x, w_eff=None, acts=None):
        a = nn.silu(self.gate_proj(x)) * self.up_proj(x)
        if acts is not None:
            acts.append(a)
        return a @ w_eff.T if w_eff is not None else self.down_proj(a)


class TileBlock(nn.Module):
    def __init__(self, c: Q.Qwen3Config, mode, soft_k, soft_rank):
        super().__init__()
        self.self_attn = Q.Attention(c)
        self.mlp = TileMLP(c.hidden_size, c.intermediate_size, mode, soft_k, soft_rank)
        self.input_layernorm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)

    def __call__(self, x, w_eff=None, acts=None):
        h = x + self.self_attn(self.input_layernorm(x), "causal", None)
        return h + self.mlp(self.post_attention_layernorm(h), w_eff, acts)


class SideTile(nn.Module):
    def __init__(self, cfg: Q.Qwen3Config, stats, donor_down=None, width=256, every=1, write_site="last",
                 gate_per_layer=False, heads=4, ffn_mult=3, align=None, soft_k=8, soft_rank=16, tau=0.5, alpha=0.0):
        super().__init__()
        L, d = cfg.num_hidden_layers, cfg.hidden_size
        if width > d:
            raise SystemExit(f"--width {width} > donor width {d}")
        if width % heads:
            raise SystemExit("--width must be a multiple of --heads")
        self._lay = aligned_layers(L, every)
        n_t = len(self._lay) - 1
        self._sites = write_sites(L, self._lay, write_site, gate_per_layer)
        self.width, self.tau, self.alpha, self.align = width, tau, float(alpha), align
        tc = Q.Qwen3Config(hidden_size=width, intermediate_size=ffn_mult * width, num_hidden_layers=n_t,
                           num_attention_heads=heads, num_key_value_heads=heads, head_dim=width // heads,
                           rms_norm_eps=cfg.rms_norm_eps, rope_theta=cfg.rope_theta, vocab_size=1,
                           tie_word_embeddings=True)
        if align in _TIED_MODES and donor_down is not None and tc.intermediate_size > donor_down[0].shape[1]:
            raise SystemExit("tile MLP wider than the donor's: unique ties impossible")
        self.blocks = [TileBlock(tc, align, soft_k, soft_rank) for _ in range(n_t)]
        self.P = [Proj(mx.array(stats["P"][j])) for j in range(n_t + 1)]
        self._mu = mx.array(stats["mu"])                 # (n_t + 1, d) fixed
        self._r = mx.array(stats["r"])                   # (n_t + 1,) fixed
        self.U = []
        for _ in self._sites:
            u = nn.Linear(width, d, bias=False)
            u.weight = mx.zeros_like(u.weight)           # the untrained tile writes exactly 0
            self.U.append(u)
        self.gate = mx.array([1.0])
        # Position 0 (the donor's attention sink) is never read from the donor; the tile sees a fixed unit-RMS vector
        # there instead of zeros. Zeros made position 0 exactly zero through every tile block (no biases), and each
        # tile RMSNorm's gradient at a zero input is 1/sqrt(eps) = 1000x: over 28 blocks the backward overflowed fp32
        # (NaN in blocks 0-7 at the first real update, Qwen3-0.6B, 2026-11-25).
        sink = np.random.default_rng(56).normal(size=(width,))
        self._sink = mx.array((sink / np.sqrt(np.mean(sink ** 2))).astype(np.float32))
        if align == "grow":
            self.grow_state = mx.zeros((3,))             # [loss EMA, EMA at last check, total freed] (fixed; saved)
        self._donor_down = donor_down                    # frozen donor down weights (d, F) per tile block: not params
        self.phase = None

    # ---------------------------------------------------------- bus seam
    def encode(self, j, h):
        """Donor state h at aligned layer index j -> tile space (the only way the tile reads the donor)."""
        ns = _not_sink(h.shape[1])
        x = (PB.unit_rms(h) - self._mu[j]) * ns
        return self.P[j](x) * ns + (1.0 - ns) * self._sink       # position 0: the fixed tile sink, never zeros

    def needed_layers(self, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        need = {self._lay[0]} | {i for _, i in self._sites}
        if alpha > 0:
            need |= set(self._lay[:-1])
        return need

    # ---------------------------------------------------------- donor-aligned write vectors (H56b)
    def dictionary(self, j, idx=None):
        """Donor block (lay[j+1] - 1)'s MLP down columns in tile space: sg(P_{j+1}) W_down[:, idx] / r_{j+1}."""
        wd = self._donor_down[j]
        if idx is not None:
            wd = mx.take(wd, idx, axis=1)
        return (mx.stop_gradient(self.P[j + 1].weight) @ wd.astype(mx.float32)) / self._r[j + 1]

    def down_eff(self, j):
        mlp = self.blocks[j].mlp
        if mlp.mode in _TIED_MODES:
            w = mlp.down_proj.weight
            dtied = self.dictionary(j, mlp.tie_idx)
            return w + mlp.tie_mask[None, :] * mx.stop_gradient(dtied - w)   # forward: donor column; grad: shadow
        if mlp.mode == "soft":
            f, k = mlp.soft_idx.shape
            dsub = self.dictionary(j, mlp.soft_idx.reshape(-1)).reshape(-1, f, k)
            return mx.sum(dsub * mlp.coef[None], axis=-1) + mlp.free_b.weight @ mlp.free_a.weight
        return None

    # ---------------------------------------------------------- forward of the tile alone
    def run(self, states, alpha=None, acts=None, on_input=None, on_output=None):
        """states {l: h_l} from donor_pass -> [s_1 .. s_{n_t}] (B, T, w) float32."""
        alpha = self.alpha if alpha is None else float(alpha)
        lay = self._lay
        s, outs = None, []
        for j, blk in enumerate(self.blocks):
            if j == 0:
                inp = self.encode(0, states[lay[0]])
            elif alpha >= 1.0:
                inp = self.encode(j, states[lay[j]])         # teacher forced: the tile's own state is not read
            elif alpha > 0.0:
                inp = alpha * self.encode(j, states[lay[j]]) + (1.0 - alpha) * s
            else:
                inp = s                                        # free running
            if on_input is not None:
                on_input(j, inp)
            s = blk(inp, self.down_eff(j), acts)
            if on_output is not None:
                s = on_output(j, s)
            outs.append(s)
        return outs

    def write_deltas(self, outs):
        """{donor block index: [delta (B, T, d) float32]} (position 0 zeroed)."""
        res = {}
        for (k, i), u in zip(self._sites, self.U):
            s = outs[k - 1]
            res.setdefault(i, []).append(u(s) * _not_sink(s.shape[1]))
        return res

    # ---------------------------------------------------------- phases
    def set_phase(self, phase):
        """A: tile blocks only. B: blocks, P_0 (P_1.. if alpha > 0), U, gate. Fixed arrays always frozen."""
        self.unfreeze()
        self.freeze()
        for blk in self.blocks:
            blk.unfreeze()
            if blk.mlp._fixed_keys:
                blk.mlp.freeze(recurse=False, keys=blk.mlp._fixed_keys)
        if phase == "B":
            self.P[0].unfreeze()
            if self.alpha > 0:
                for p in self.P[1:-1]:
                    p.unfreeze()
            for u in self.U:
                u.unfreeze()
            self.unfreeze(recurse=False, keys=["gate"])
        self.phase = phase

    def tied_fraction(self):
        if self.align in _TIED_MODES:
            return round(float(mx.mean(mx.stack([b.mlp.tie_mask for b in self.blocks])).item()), 4)
        return None

    def n_tied_entries(self):
        if self.align not in _TIED_MODES:
            return 0
        return int(sum(float(b.mlp.tie_mask.sum().item()) for b in self.blocks)) * self.width


def forward(model, tokens, tile=None, zero_values=False, states=None):
    """Logits (B, T, V) and None (the CT.forward contract, so CT.nll_eval / corpus_tile_eval.qa_eval take it as fwd).
    tile None = the original model; zero_values = the tile's write zeroed (= the original: no donor LoRA)."""
    if tile is None:
        return CT.forward(model, tokens, None)
    layers = model.model.layers
    first = min(i for _, i in tile._sites)
    if states is None:
        states, _ = donor_pass(model, tokens, tile.needed_layers() | {first})
    deltas = {} if zero_values else tile.write_deltas(tile.run(states))
    h = states[first]
    for i in range(first, len(layers)):
        for dl in deltas.get(i, ()):
            h = h + LB.bounded(tile.gate * dl, h.astype(mx.float32), tile.tau).astype(h.dtype)
        h = layers[i](h, "causal", None)                 # DONOR_LORA is False: the donor never carries a LoRA
    for dl in deltas.get(len(layers), ()):
        h = h + LB.bounded(tile.gate * dl, h.astype(mx.float32), tile.tau).astype(h.dtype)
    return model.head(h), None


def merge_back(tile, model):
    """H56b later step: fold tied tile neurons back into the donor's weights and check exactness (outputs within a
    near tie). Not built yet; the tied neurons' donor indices are tile.blocks[j].mlp.tie_idx."""
    raise NotImplementedError("H56b merge-back is a later step")


# ------------------------------------------------------------------ losses
def imitation_loss(tile, states, alpha, mask=None):
    """Mean over tile blocks of the relative state error ||s_{j+1} - z_{lay[j+1]}||^2 / ||z||^2 (positions >= 1)."""
    outs = tile.run(states, alpha)
    errs = []
    for j, s in enumerate(outs):
        z = mx.stop_gradient(tile.encode(j + 1, states[tile._lay[j + 1]]))
        m = _not_sink(s.shape[1])[..., 0] if mask is None else mask
        num = mx.sum(mx.sum((s - z) ** 2, axis=-1) * m)
        den = mx.maximum(mx.sum(mx.sum(z * z, axis=-1) * m), 1e-12)
        errs.append(num / den)
    return mx.mean(mx.stack(errs)), errs


def state_errors(tile, model, tokens, alpha):
    st, _ = donor_pass(model, tokens, set(tile._lay))
    mean, errs = imitation_loss(tile, st, alpha)
    return round(float(mean.item()), 4), [round(float(e.item()), 4) for e in errs]


def alpha_a(it, steps_a, hold=0.25, end=0.0):
    """Phase A anneal at step it (1-based): 1 for the first `hold` of A, linear to `end` over the middle, `end` for the
    last `hold` (train passes end = alpha_b, or 0 with --anneal-to-zero)."""
    u = it / max(1, steps_a)
    if u <= hold:
        return 1.0
    if u >= 1.0 - hold:
        return float(end)
    return 1.0 - (1.0 - end) * (u - hold) / max(1e-9, 1.0 - 2 * hold)


def annealing(it, steps_a, hold):
    u = it / max(1, steps_a)
    return it <= steps_a and hold < u < 1.0 - hold


# ------------------------------------------------------------------ H56b matching
def match_ties(score, mask, idx, tau_on, tau_off, method="greedy"):
    """score (f, F): tile neuron x donor column. A tied neuron stays tied while score[i, idx_i] >= tau_off; free neurons
    tie (unique donor columns, never stealing a kept tie) where score >= tau_on: greedy by descending score, or the
    Hungarian assignment (scipy) over the free rows and untaken columns. -> (mask, idx) numpy."""
    score = np.asarray(score, dtype=np.float64)
    f = score.shape[0]
    mask = np.asarray(mask, dtype=bool)
    idx = np.asarray(idx, dtype=np.int64)
    keep = mask & (score[np.arange(f), idx] >= tau_off)
    new_mask, new_idx = keep.copy(), np.where(keep, idx, 0)
    taken = set(idx[keep].tolist())
    rows = np.flatnonzero(~keep)
    if rows.size == 0:
        return new_mask, new_idx
    if method == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as e:
            raise SystemExit("--match hungarian needs scipy (not in this venv); use greedy") from e
        cols = np.array([c for c in range(score.shape[1]) if c not in taken])
        sub = score[np.ix_(rows, cols)]
        ri, ci = linear_sum_assignment(-sub)
        for a, b in zip(ri, ci):
            if sub[a, b] >= tau_on:
                new_mask[rows[a]], new_idx[rows[a]] = True, cols[b]
        return new_mask, new_idx
    r, c = np.nonzero(score[rows] >= tau_on)
    order = np.argsort(-score[rows][r, c], kind="stable")
    for t in order:
        i, j = rows[r[t]], int(c[t])
        if new_mask[i] or j in taken:
            continue
        new_mask[i], new_idx[i] = True, j
        taken.add(j)
    return new_mask, new_idx


def _normcols(a):
    return a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-12)


def _np(x):
    return np.array(x.astype(mx.float32))


def _corr(a, b):
    a = (a - a.mean(0)) / (a.std(0) + 1e-6)
    b = (b - b.mean(0)) / (b.std(0) + 1e-6)
    return a.T @ b / a.shape[0]


def _reset_moments(opt, j, path, sel, axis):
    """Zero the Adam moments (m, v) of the selected columns (axis 1) or rows (axis 0) of block j's MLP parameter at
    path (e.g. ("down_proj", "weight")), so a neuron that ties or unties starts its free weight with fresh moments."""
    if opt is None or not np.any(sel):
        return
    node = opt.state.get("blocks")
    try:
        node = node[j]["mlp"]
        for k in path:
            node = node[k]
    except (KeyError, IndexError, TypeError):
        return
    for k in ("m", "v"):
        if k in node:
            arr = np.array(node[k])
            if axis == 1:
                arr[:, sel] = 0
            else:
                arr[sel] = 0
            node[k] = mx.array(arr)


def align_init(tile, seed=0):
    """Start state of the aligned arms (called once on a fresh run; a resume reloads the saved state).
    hard / act / grow: every tile neuron's write vector starts as a distinct donor column in tile space (the f columns
      with the largest ||D_c||, i.e. the donor neurons best represented in the bus), tied, so the run measures whether
      ties hold or break rather than waiting for a random vector to reach tau_on (random best cosine is ~0.2 at w 256).
    soft: each neuron's support is K distinct donor columns drawn at random (seeded), coefficients N(0, 1/K) scaled so
      a column has the norm of a default Linear init column (sqrt(w / 3f)), and the low-rank free part at 0.1x its
      default init, so the dictionary dominates the write from the start (full rank, not rank 1)."""
    rng = np.random.default_rng(1_000 + seed)
    for j, blk in enumerate(tile.blocks):
        mlp = blk.mlp
        D = _np(tile.dictionary(j))
        w, F = D.shape
        if mlp.mode == "soft":
            f, k = mlp.soft_idx.shape
            idx = np.sort(np.stack([rng.choice(F, size=k, replace=False) for _ in range(f)]), axis=1)
            target = math.sqrt(w / (3 * f)) / max(1e-12, float(np.linalg.norm(D, axis=0).mean()))
            mlp.soft_idx = mx.array(idx.astype(np.int32))
            mlp.coef = mx.array((rng.normal(size=(f, k)) / math.sqrt(k) * target).astype(np.float32))
            mlp.free_b.weight = 0.1 * mlp.free_b.weight
            continue
        f = mlp.tie_mask.shape[0]
        pick = np.argsort(-np.linalg.norm(D, axis=0), kind="stable")[:f]
        mlp.tie_idx = mx.array(pick.astype(np.int32))
        mlp.tie_mask = mx.ones((f,))
        mlp.down_proj.weight = mx.array(D[:, pick])


def align_step(tile, a, model=None, tokens=None, opt=None, annealing=False):
    """Periodic H56b matching. -> summary dict for the log. opt: Adam moments of re-tied / freed write vectors are
    reset. annealing (phase A while alpha moves): align_grow makes no grow decision (the loss is not comparable)."""
    mode = tile.align
    if mode == "grow":
        ema, ref, total_freed = tile.grow_state.tolist()
        n_neurons = sum(b.mlp.tie_mask.shape[0] for b in tile.blocks)
        budget = int(a.grow_max_frac * n_neurons) - int(total_freed)
        stalled = (not annealing) and ref > 0 and (ref - ema) / ref < a.grow_eps and budget > 0
        freed = 0
        if stalled:
            for j, blk in enumerate(tile.blocks):
                mlp = blk.mlp
                mask = _np(mlp.tie_mask) > 0.5
                if not mask.any() or freed >= budget:
                    continue
                W, D = _np(mlp.down_proj.weight), _np(tile.dictionary(j, mlp.tie_idx))
                cos = np.sum(_normcols(W) * _normcols(D), axis=0)
                tied = np.flatnonzero(mask)
                n = min(int(math.ceil(a.grow_frac * tied.size)), budget - freed)
                drop = tied[np.argsort(cos[tied], kind="stable")[:n]]
                sel = np.zeros(mask.shape, bool)
                sel[drop] = True
                mask[drop] = False
                W[:, drop] = D[:, drop]                     # the free weight starts at the donor column it replaces
                mlp.down_proj.weight = mx.array(W)
                mlp.tie_mask = mx.array(mask.astype(np.float32))
                _reset_moments(opt, j, ("down_proj", "weight"), sel, 1)
                freed += n
        tile.grow_state = mx.array([ema, ema, total_freed + freed])
        return {"tied_frac": tile.tied_fraction(), "grow_freed": freed, "grow_freed_total": int(total_freed + freed),
                "grow_skipped_anneal": bool(annealing)}
    if mode == "soft":
        shares = []
        for j, blk in enumerate(tile.blocks):
            mlp = blk.mlp
            f, k = mlp.soft_idx.shape
            D = _np(tile.dictionary(j))
            idx_old, coef = np.array(mlp.soft_idx), _np(mlp.coef)
            dict_part = np.einsum("wfk,fk->wf", D[:, idx_old], coef)
            w_eff = dict_part + _np(mlp.free_b.weight) @ _np(mlp.free_a.weight)
            cos = np.abs(_normcols(w_eff).T @ _normcols(D))                  # (f, F)
            idx_new = np.sort(np.argpartition(-cos, k - 1, axis=1)[:, :k], axis=1)
            A = np.transpose(D[:, idx_new], (1, 0, 2))                       # (f, w, K)
            G = np.einsum("fwk,fwl->fkl", A, A) + 1e-6 * np.eye(k)[None]
            rhs = np.einsum("fwk,wf->fk", A, dict_part)
            coef_new = np.linalg.solve(G, rhs[..., None])[..., 0]
            mlp.soft_idx = mx.array(idx_new.astype(np.int32))
            mlp.coef = mx.array(coef_new.astype(np.float32))
            _reset_moments(opt, j, ("coef",), np.any(idx_new != idx_old, axis=1), 0)
            free = _np(mlp.free_b.weight) @ _np(mlp.free_a.weight)
            dp = np.einsum("wfk,fk->wf", D[:, idx_new], coef_new)
            shares.append(float((dp ** 2).sum() / max(1e-12, (dp ** 2).sum() + (free ** 2).sum())))
        return {"dict_share": round(float(np.mean(shares)), 4)}
    acts_t, acts_d = None, None
    if mode == "act":
        sub = tokens[:2]
        lay = tile._lay
        keep = set(lay) | {lay[j + 1] - 1 for j in range(len(tile.blocks))}
        st, _ = donor_pass(model, sub, keep)
        acts_t = []
        tile.run(st, tile.alpha if tile.phase == "B" else 1.0, acts=acts_t)
        acts_d = [donor_mlp_act(model.model.layers[lay[j + 1] - 1], st[lay[j + 1] - 1])
                  for j in range(len(tile.blocks))]
    for j, blk in enumerate(tile.blocks):
        mlp = blk.mlp
        W, D = _np(mlp.down_proj.weight), _np(tile.dictionary(j))
        if mode == "act":
            at = _np(acts_t[j][:, 1:]).reshape(-1, acts_t[j].shape[-1])
            ad = _np(acts_d[j][:, 1:]).reshape(-1, acts_d[j].shape[-1])
            score = _corr(at, ad)
            on, off = a.tau_on_act, a.tau_off_act
        else:
            score = _normcols(W).T @ _normcols(D)
            on, off = a.tau_on, a.tau_off
        mask, idx = _np(mlp.tie_mask) > 0.5, np.array(mlp.tie_idx)
        nm, ni = match_ties(score, mask, idx, on, off, a.match)
        newly = nm & ~(mask & (ni == idx))
        if newly.any():
            W[:, newly] = D[:, ni[newly]]                                    # shadow restarts at the donor column
            mlp.down_proj.weight = mx.array(W)
        _reset_moments(opt, j, ("down_proj", "weight"), newly | (mask & ~nm), 1)
        mlp.tie_mask = mx.array(nm.astype(np.float32))
        mlp.tie_idx = mx.array(ni.astype(np.int32))
    return {"tied_frac": tile.tied_fraction()}


# ------------------------------------------------------------------ world, stats
def world_from(model, tok, ckpt, docs, split, hashes=None):
    train_ids, held_ids = set(split["train"]), set(split["held_out"])
    if train_ids & held_ids:
        raise SystemExit("split.json: a document is both train and held out")
    return dict(docs=docs, split=split, model=model, cfg=model.config, ckpt=ckpt, tok=tok, train_ids=train_ids,
                held_ids=held_ids, hashes=hashes or {})


def build_world(args):
    """The H51 corpus (CT.load_corpus; same hashes), the frozen donor, its tokenizer. No anchors (H56 has none)."""
    mx.set_cache_limit(int(PB.CACHE_LIMIT_GIB * 2 ** 30))
    cdir = Path(args.corpus).expanduser()
    docs, split, _graph = CT.load_corpus(cdir)
    hashes = {n: CT.sha16(cdir / n) for n in ("corpus.jsonl", "split.json", "graph.json", "ref_df.json")
              if (cdir / n).exists()}
    path = Q.snapshot_dir(args.model)
    model, _cfg, ckpt = Q.load(path, dtype=mx.bfloat16, repo=args.model)
    model.freeze()
    return world_from(model, LB.Tok(path), ckpt, docs, split, hashes)


def ladder_stats(W, lay, width, seed, n_rows=256, n_tok=128, batch=16):
    """Per aligned layer: mu (mean unit-RMS state, positions >= 1), P0 (top-`width` principal directions, scaled so z
    has unit RMS per dimension), r (mean RMS of the raw state), retained variance. Half ordinary text, half corpus."""
    model, tok = W["model"], W["tok"]
    d = W["cfg"].hidden_size
    rows = CT.text_chunks(tok, "train", n=n_rows // 2, n_tok=n_tok, seed=seed + 7)
    heads = [w[:n_tok] for _, w, n in CT.doc_chunks(tok, W["docs"], W["train_ids"]) if n > n_tok]
    random.Random(seed + 8).shuffle(heads)
    rows = [r for r in rows if len(r) == n_tok] + heads[:n_rows - len(rows)]
    if not rows:
        raise SystemExit("ladder stats: no rows")
    S1 = {l: mx.zeros((d,)) for l in lay}
    S2 = {l: mx.zeros((d, d)) for l in lay}
    R = {l: 0.0 for l in lay}
    N = 0
    for i in range(0, len(rows), batch):
        T = mx.array(rows[i:i + batch])
        st, _ = donor_pass(model, T, set(lay))
        for l in lay:
            x = PB.unit_rms(st[l])[:, 1:].reshape(-1, d)
            raw = st[l][:, 1:].astype(mx.float32)
            S1[l] = S1[l] + mx.sum(x, axis=0)
            S2[l] = S2[l] + x.T @ x
            R[l] += float(mx.sum(mx.sqrt(mx.mean(raw * raw, axis=-1))).item())
            mx.eval(S1[l], S2[l])
        N += T.shape[0] * (T.shape[1] - 1)
    mu, P, r, retained = [], [], [], []
    for l in lay:
        m = np.array(S1[l], dtype=np.float64) / N
        cov = np.array(S2[l], dtype=np.float64) / N - np.outer(m, m)
        ev, vec = np.linalg.eigh((cov + cov.T) / 2)
        ev, vec = ev[::-1].clip(min=0), vec[:, ::-1]
        top = ev[:width].sum()
        P.append((vec[:, :width].T * math.sqrt(width / max(top, 1e-12))).astype(np.float32))
        mu.append(m.astype(np.float32))
        r.append(R[l] / N)
        retained.append(round(float(top / max(ev.sum(), 1e-12)), 4))
    return {"mu": np.stack(mu), "P": np.stack(P), "r": np.array(r, dtype=np.float32), "retained": retained,
            "n_tokens": N}


def make_tile(W, stats, a):
    model = W["model"]
    lay = aligned_layers(model.config.num_hidden_layers, a.every)
    align = ARMS[a.arm]
    donor_down = [model.model.layers[lay[j + 1] - 1].mlp.down_proj.weight for j in range(len(lay) - 1)] \
        if align else None
    return SideTile(model.config, stats, donor_down, width=a.width, every=a.every, write_site=a.write_site,
                    gate_per_layer=a.gate_per_layer, heads=a.heads, ffn_mult=a.ffn_mult, align=align,
                    soft_k=a.soft_k, soft_rank=a.soft_rank, tau=a.tau, alpha=a.alpha_b)


def param_report(tile):
    """Stored / trainable per phase / effective (B, minus tied columns) parameter counts."""
    def n(mods):
        return sum(v.size for m in mods for _, v in tree_flatten(m.parameters()))
    rep = {"blocks": n(tile.blocks), "P": n(tile.P), "U": n(tile.U), "gate": 1}
    rep["stored"] = C.n_params(tile)
    prev = tile.phase
    for ph in ("A", "B"):
        tile.set_phase(ph)
        rep[f"trainable_{ph}"] = C.n_params(tile, trainable=True)
    if prev:
        tile.set_phase(prev)
    rep["effective_B"] = rep["trainable_B"] - tile.n_tied_entries()
    rep["h51_tile_params"] = H51_TILE_PARAMS
    return rep


# ------------------------------------------------------------------ train
_FREE = {"max_seconds": None, "no_memory_gate": None, "chunk_tag": None, "gate_gib": None,
         "max_steps_this_launch": None}
STARTUP = {"args.json", "ladder_state.safetensors", "ladder_stats.json"}


def _tile_name(a):
    site = "perlayer" if a.gate_per_layer else f"site{a.write_site}"
    return (f"ens-h56-sidetile-{a.model.split('/')[-1].lower()}-{a.arm}-w{a.width}-e{a.every}-{site}"
            f"-ab{a.alpha_b:g}-s{a.seed}")


def _phase_seconds(log, steps_a):
    """{A, B, a_cost_ratio}: wall seconds per phase from the logged elapsed_s and the measured cost of an A step
    relative to a B step (feeds --a-cost-ratio for a compute-matched correction)."""
    el = [(r["step"], r["elapsed_s"]) for r in log if "elapsed_s" in r]
    if not el:
        return {}
    end = el[-1]
    at_a = next((e for st, e in el if st == steps_a), None)
    res = {"total": end[1]}
    if steps_a and at_a is not None and end[0] > steps_a:
        res |= {"A": at_a, "B": round(end[1] - at_a, 1),
                "a_cost_ratio": round((at_a / steps_a) / max(1e-9, (end[1] - at_a) / (end[0] - steps_a)), 3)}
    return res


def load_stats(run: Path):
    st = mx.load(str(run / "ladder_state.safetensors"))
    extra = json.loads((run / "ladder_stats.json").read_text())
    return {"mu": np.array(st["mu"]), "P": np.array(st["P"]), "r": np.array(st["r"])} | extra


def train(args, world=None) -> int:
    """TERM is caught from the start (before the memory gate and the model load); a TERM during loading is honoured at
    the first loop check: state saved, exit 3."""
    PB._TERM["requested"] = False
    old = signal.signal(signal.SIGTERM, PB._on_term)
    try:
        return _train(args, world)
    finally:
        signal.signal(signal.SIGTERM, old)


NONFINITE_EXIT = 4            # not 3: a runner must not retry a run that went non-finite


def _all_finite(arrays, tree) -> bool:
    flags = [mx.all(mx.isfinite(a)) for a in arrays] + \
        [mx.all(mx.isfinite(v)) for _, v in tree_flatten(tree) if mx.issubdtype(v.dtype, mx.floating)]
    return bool(mx.all(mx.stack(flags)).item()) if flags else True


def _nonfinite_stop(out: Path, tile, it, ph, loss, grads, log, where) -> int:
    """Write <out>/nonfinite.json (the step, which loss / gradients / parameters are non-finite, the last log rows)
    and the tile's current parameters (nonfinite_tile.safetensors), then stop with NONFINITE_EXIT. No resume state is
    written for this step, so the last autosave stays the last good state."""
    def bad(tree):
        return [k for k, v in tree_flatten(tree) if mx.issubdtype(v.dtype, mx.floating)
                and not bool(mx.all(mx.isfinite(v)).item())]
    diag = {"step": it, "phase": ph, "where": where, "loss_finite": bool(mx.all(mx.isfinite(loss)).item()),
            "loss": float(loss.item()) if bool(mx.all(mx.isfinite(loss)).item()) else None,
            "bad_grads": bad(grads), "bad_params": bad(tile.parameters()), "last_log": log[-5:],
            **C.source_revision(), **C.host_record()}
    mx.save_safetensors(str(out / "nonfinite_tile.safetensors"), dict(tree_flatten(tile.parameters())))
    C.write_json(out / "nonfinite.json", diag)
    print(f"NON-FINITE at step {it} ({where}); diagnostics in {out / 'nonfinite.json'}; exit {NONFINITE_EXIT}",
          flush=True)
    return NONFINITE_EXIT


def _saved_args(out: Path) -> dict:
    """A run's args.json; runs written before --accum existed ran without accumulation (accum 1)."""
    return {"accum": 1, "clip": 0.0, "track_every": 0} | json.loads((out / "args.json").read_text())  # pre-2026-11-25


TRACK_N, TRACK_BATCH, TRACK_KEYS = 8, 4, ("track_train", "track_held", "base_train", "base_held", "gain_train",
                                          "gain_held", "gain_train_minus_held")


def track_sample(tok, W, seed):
    """--track-every: a FIXED sample of TRACK_N trained-document and TRACK_N held-out-document chunks, drawn once with a
    fixed seed from the chunk lists CT.nll_eval uses (CT.doc_chunks over train_ids / held_ids). Deterministic, so a
    resume draws the same sample."""
    r = random.Random(80_000 + seed)
    tr = CT.doc_chunks(tok, W["docs"], W["train_ids"])
    he = CT.doc_chunks(tok, W["docs"], W["held_ids"])
    return r.sample(tr, min(TRACK_N, len(tr))), r.sample(he, min(TRACK_N, len(he)))


def track_nll(model, tile, sample):
    """Mean next-token NLL (positions >= 1, real tokens only, as CT.nll_eval) of the tile at its alpha (tile None: the
    base model) on the sample, in groups of TRACK_BATCH chunks, no gradients."""
    tot, cnt = 0.0, 0.0
    for i in range(0, len(sample), TRACK_BATCH):
        sub = sample[i:i + TRACK_BATCH]
        T = mx.array([w[:-1] for _, w, _ in sub])
        Y = mx.array([w[1:] for _, w, _ in sub])
        Wm = mx.array([[1.0] * (m - 1) + [0.0] * (CT.CHUNK - m + 1) for _, _, m in sub])
        lg, _ = forward(model, T, tile)
        ce = nn.losses.cross_entropy(lg.astype(mx.float32), Y, reduction="none")
        tot += float(mx.sum(ce * Wm).item())
        cnt += float(mx.sum(Wm).item())
    if cnt == 0:        # review 2026-11-25: an empty split must not read as "no memorisation" (0 - 0)
        raise ValueError("track sample has no tokens (empty trained or held split)")
    return tot / cnt


def _accum_imitation(tile, model, T, alpha, k):
    """Phase A with gradient accumulation: the rows cut into k contiguous groups of nearly equal size (np.array_split,
    in batch order), evaluated one at a time. The relative error of each
    layer is a ratio over the whole batch, so a first pass (no gradient; P is frozen in A, the targets are fixed)
    sums each layer's denominator over the batch, and every micro-batch then contributes num_i / den_total. Their sum
    equals the full-batch loss and gradient exactly (up to float summation order). Costs one extra donor forward."""
    lay, n_t = tile._lay, len(tile.blocks)
    groups = [(int(g[0]), int(g[-1]) + 1) for g in np.array_split(np.arange(T.shape[0]), k)]
    dens = [mx.array(0.0) for _ in range(n_t)]
    for a0, a1 in groups:
        st, _ = donor_pass(model, T[a0:a1], set(lay))
        msk = _not_sink(T.shape[1])[..., 0]
        for j in range(n_t):
            z = tile.encode(j + 1, st[lay[j + 1]])
            dens[j] = dens[j] + mx.sum(mx.sum(z * z, axis=-1) * msk)
        mx.eval(dens)
    dens = [mx.maximum(d, 1e-12) for d in dens]

    def micro(t, st):
        outs = t.run(st, alpha)
        msk = _not_sink(outs[0].shape[1])[..., 0]
        errs = [mx.sum(mx.sum((s - mx.stop_gradient(t.encode(j + 1, st[lay[j + 1]]))) ** 2, axis=-1) * msk) / dens[j]
                for j, s in enumerate(outs)]
        return mx.mean(mx.stack(errs))
    total, gsum = None, None
    for a0, a1 in groups:
        st, _ = donor_pass(model, T[a0:a1], set(lay))
        loss, g = nn.value_and_grad(tile, lambda t, st=st: micro(t, st))(tile)
        total = loss if total is None else total + loss
        gsum = g if gsum is None else tree_map(mx.add, gsum, g)
        mx.eval(total, gsum)
    return total, gsum


def _accum_corpus(tile, model, T, Y, Wm, ttok, keep, kl_weight, k):
    """Phase B with gradient accumulation. The step's rows in a fixed order, the corpus rows first, then the text
    rows, are cut into k contiguous groups of nearly equal size (np.array_split over that order; a group may hold
    corpus rows, text rows or both; k = rows is one row per micro-batch). NLL uses the whole batch's token count as its
    denominator and KL the whole batch's row x position count, so the sum of micro losses and gradients equals the
    full-batch step exactly (up to float summation order). Micro-batches run one at a time."""
    nc = T.shape[0]
    nt = 0 if ttok is None else ttok.shape[0]
    w_total = mx.maximum(mx.sum(Wm), 1.0)
    n_kl = 1 if ttok is None else nt * ttok.shape[1]
    total = nll_s = kl_s = gsum = None
    for g in np.array_split(np.arange(nc + nt), k):
        c0, c1 = min(int(g[0]), nc), min(int(g[-1]) + 1, nc)
        t0, t1 = max(int(g[0]) - nc, 0), max(int(g[-1]) + 1 - nc, 0)
        Tc = Yc = Wc = st_c = tt = st_t = tref = None
        if c1 > c0:
            Tc, Yc, Wc = T[c0:c1], Y[c0:c1], Wm[c0:c1]
            st_c, _ = donor_pass(model, Tc, keep)
        if t1 > t0:
            tt = ttok[t0:t1]
            st_t, tref = donor_pass(model, tt, keep, logits=True)
            tref = mx.stop_gradient(tref.astype(mx.float32))

        def micro(t, Tc=Tc, Yc=Yc, Wc=Wc, st_c=st_c, tt=tt, st_t=st_t, tref=tref):
            nll = kl = mx.array(0.0)
            if Tc is not None:
                logits, _ = forward(model, Tc, t, states=st_c)
                ce = nn.losses.cross_entropy(logits.astype(mx.float32), Yc, reduction="none")
                nll = mx.sum(ce * Wc) / w_total
            if tt is not None:
                tl = forward(model, tt, t, states=st_t)[0].astype(mx.float32)
                kl = mx.sum(mx.softmax(tref, -1) * (LB._log_softmax(tref, -1) - LB._log_softmax(tl, -1))) / n_kl
            return nll + kl_weight * kl, (nll, kl)
        (loss, (nll, kl)), gr = nn.value_and_grad(tile, micro)(tile)
        total = loss if total is None else total + loss
        nll_s = nll if nll_s is None else nll_s + nll
        kl_s = kl if kl_s is None else kl_s + kl
        gsum = gr if gsum is None else tree_map(mx.add, gsum, gr)
        mx.eval(total, gsum, nll_s, kl_s)
    return total, nll_s, kl_s, gsum


def _batch_sha(*arrays):
    import hashlib
    h = hashlib.sha256()
    for x in arrays:
        if x is not None:
            h.update(np.array(x).tobytes())
    return h.hexdigest()[:12]


def _train(args, world=None) -> int:
    out = Path(args.out)
    if args.match == "hungarian":
        try:
            import scipy.optimize  # noqa: F401
        except ImportError:
            raise SystemExit("--match hungarian needs scipy (not in this venv); use --match greedy") from None
    n_text_ = max(1, int(round(args.batch * args.p_text))) if args.p_text > 0 else 0
    if args.accum < 1:
        raise SystemExit("--accum must be >= 1")
    if args.accum > args.batch:
        raise SystemExit(f"--accum {args.accum}: at most the {args.batch} rows of a step")
    if (out / "train_receipt.json").exists():
        print("already trained", flush=True)
        return 0
    resuming = PB.resume_dir(out) is not None
    if not resuming:
        if out.is_dir() and {p.name for p in out.iterdir()} <= STARTUP:
            for nme in STARTUP:                          # a launch that stopped before its first autosave
                if (out / nme).exists():
                    (out / nme).unlink()
        out = C.fresh_out(args.out)
        C.write_json(out / "args.json", vars(args))
    elif _saved_args(out) | _FREE != vars(args) | _FREE:
        raise SystemExit("resume: arguments differ")
    if not args.no_memory_gate:
        PB.memory_gate(gib=args.gate_gib)
    W = world if world is not None else build_world(args)
    model, tok = W["model"], W["tok"]
    lay = aligned_layers(model.config.num_hidden_layers, args.every)
    if (out / "ladder_state.safetensors").exists():
        stats = load_stats(out)
    else:
        stats = ladder_stats(W, lay, args.width, args.seed, n_rows=args.stat_rows)
        mx.save_safetensors(str(out / "ladder_state.safetensors"),
                            {k: mx.array(stats[k]) for k in ("mu", "P", "r")})
        C.write_json(out / "ladder_stats.json", {"retained": stats["retained"], "n_tokens": stats["n_tokens"],
                                                 "layers": lay})
    guard = None if args.no_memory_gate else PB.SwapGuard()
    mx.random.seed(args.seed)
    tile = make_tile(W, stats, args)
    if tile.align:
        align_init(tile, args.seed)                      # deterministic; a resume reloads the saved state anyway
    steps_a = 0 if args.arm == "correction" else args.steps_a
    extra_b = int(round(args.steps_a * args.a_cost_ratio)) if (args.compute_matched and args.arm == "correction") else 0
    steps_b = args.steps + extra_b
    total = steps_a + steps_b
    alpha_end = 0.0 if args.anneal_to_zero else args.alpha_b
    params = param_report(tile)
    print("params", json.dumps(params), flush=True)
    chunks = CT.doc_chunks(tok, W["docs"], W["train_ids"])
    text = CT.text_chunks(tok, "train", n=2000, seed=args.seed)
    a_text = CT.text_chunks(tok, "train", n=2000, n_tok=args.a_len, seed=args.seed + 1) if steps_a else []
    held = mx.array(CT.text_chunks(tok, "held", n=4, n_tok=128, seed=3))
    if steps_a and not a_text:
        raise SystemExit("phase A: no ordinary-text windows of --a-len tokens")
    rng = random.Random(10_000 + args.seed)              # phase B stream: the same in every arm (and as H51)
    rng_a = random.Random(20_000 + args.seed)            # phase A stream, separate so B batches do not depend on A
    rngs = [rng, rng_a]
    n_text = max(1, int(round(args.batch * args.p_text))) if args.p_text > 0 else 0

    def phase_of(it):
        return "A" if it < steps_a else "B"

    def make_opt(ph):
        n_steps, lr = (steps_a, args.lr_a) if ph == "A" else (steps_b, args.lr)
        wu = max(1, min(200, n_steps // 20))
        o = optim.AdamW(learning_rate=optim.join_schedules(
            [optim.linear_schedule(1e-6, lr, wu), optim.cosine_decay(lr, max(1, n_steps - wu), lr * 0.1)], [wu]),
            weight_decay=0.0)
        tile.set_phase(ph)
        o.init(tile.trainable_parameters())
        return o

    it0, log, t_used, cur, opt = 0, [], 0.0, None, None
    if resuming:
        d = PB.resume_dir(out)
        it0 = pickle.loads((d / "stream.pkl").read_bytes())["it"]
        cur = phase_of(min(it0, max(0, total - 1)))
        opt = make_opt(cur)
        it0, log, t_used = PB.load_state(d, tile, opt, rngs)
        print(f"resumed at {it0}", flush=True)

    def nll_kl(t, T, Y, Wm, st_c, ttok, st_t, tref):
        logits, _ = forward(model, T, t, states=st_c)
        ce = nn.losses.cross_entropy(logits.astype(mx.float32), Y, reduction="none")
        nll = mx.sum(ce * Wm) / mx.maximum(mx.sum(Wm), 1.0)
        kl = mx.array(0.0)
        if ttok is not None:
            tl = forward(model, ttok, t, states=st_t)[0].astype(mx.float32)
            kl = mx.mean(mx.sum(mx.softmax(tref, -1) * (LB._log_softmax(tref, -1) - LB._log_softmax(tl, -1)), -1))
        return nll + args.kl_weight * kl, (nll, kl)

    track = None
    if args.track_every:
        tr_s, he_s = track_sample(tok, W, args.seed)
        track = {"train": tr_s, "held": he_s, "base_train": track_nll(model, None, tr_s),
                 "base_held": track_nll(model, None, he_s)}          # the base once per launch (deterministic)
    t0, it = time.time(), it0
    first_site = min(i for _, i in tile._sites)
    while it < total:
        ph = phase_of(it)
        if ph != cur:
            opt, cur = make_opt(ph), ph
            if tile.align == "grow":                 # a new loss: restart the EMA, keep the total freed
                tile.grow_state = mx.array([0.0, 0.0, tile.grow_state.tolist()[2]])
        n_launch = it - it0
        swap = guard.check(n_launch) if guard and (n_launch <= PB.SWAP_WARMUP_STEPS or n_launch % 10 == 0) \
            else None
        budget = args.max_steps_this_launch and n_launch >= args.max_steps_this_launch
        if PB._TERM["requested"] or swap or budget or (args.max_seconds and time.time() - t0 > args.max_seconds):
            if swap:
                log.append({"step": it, "swap_stop": swap})
            PB.save_state(out, tile.parameters(), opt, it, rngs, log, t_used + time.time() - t0)
            print(f"stopped at {it} (resume state saved){' - ' + swap if swap else ''}", flush=True)
            return 3
        if args.save_every and it > it0 and it % args.save_every == 0:
            PB.save_state(out, tile.parameters(), opt, it, rngs, log, t_used + time.time() - t0)
        it += 1
        match_now = bool(tile.align) and it % args.match_every == 0
        logging = it % args.log_every == 0 or it == total or it == steps_a or match_now
        if ph == "A":
            alpha = alpha_a(it, steps_a, args.alpha_hold, alpha_end)
            T = mx.array([a_text[rng_a.randrange(len(a_text))] for _ in range(args.batch)])
            if args.accum == 1:
                st, _ = donor_pass(model, T, set(lay))
                vg = nn.value_and_grad(tile, lambda t: imitation_loss(t, st, alpha))
                (loss, _errs), grads = vg(tile)
            else:
                loss, grads = _accum_imitation(tile, model, T, alpha, args.accum)
            rec = {"phase": "A", "alpha": round(alpha, 4), "imit": None}
            tokens = T
        else:
            pick = [chunks[rng.randrange(len(chunks))] for _ in range(args.batch - n_text)]
            T = mx.array([w[:-1] for _, w, _ in pick])
            Y = mx.array([w[1:] for _, w, _ in pick])
            Wm = mx.array([[1.0] * (n - 1) + [0.0] * (CT.CHUNK - n + 1) for _, _, n in pick])
            keep = tile.needed_layers() | {first_site}
            if args.accum == 1:
                st_c, _ = donor_pass(model, T, keep)
                ttok = st_t = tref = None
                if n_text:
                    ttok = mx.array([text[rng.randrange(len(text))] for _ in range(n_text)])
                    st_t, tref = donor_pass(model, ttok, keep, logits=True)
                    tref = mx.stop_gradient(tref.astype(mx.float32))
                vg = nn.value_and_grad(tile, lambda t: nll_kl(t, T, Y, Wm, st_c, ttok, st_t, tref))
                (loss, (nll, kl)), grads = vg(tile)
            else:
                ttok = mx.array([text[rng.randrange(len(text))] for _ in range(n_text)]) if n_text else None
                loss, nll, kl, grads = _accum_corpus(tile, model, T, Y, Wm, ttok, keep, args.kl_weight, args.accum)
            rec = {"phase": "B", "alpha": tile.alpha}
            if it - steps_a <= 3:
                rec["batch_sha"] = _batch_sha(T, ttok)
                logging = True
            tokens = T
            track_now = bool(track) and (it - steps_a) % args.track_every == 0
            logging = logging or track_now
        if not _all_finite([loss], grads):
            return _nonfinite_stop(out, tile, it, ph, loss, grads, log, "loss or gradient before the update")
        if args.clip > 0:       # 2026-11-25: unclipped, the real-model correction arm diverged (loss 2.4 -> 3.8 by step 200)
            grads, gnorm = optim.clip_grad_norm(grads, args.clip)
            rec["grad_norm"] = round(float(gnorm.item()), 4) if logging else rec.get("grad_norm")
        opt.update(tile, grads)
        mx.eval(tile.parameters(), opt.state)
        if not _all_finite([], tile.parameters()):
            return _nonfinite_stop(out, tile, it, ph, loss, grads, log, "parameter after the update")
        if tile.align == "grow":
            ema, ref, freed = tile.grow_state.tolist()
            lv = float(loss.item())
            tile.grow_state = mx.array([lv if ema == 0 else 0.98 * ema + 0.02 * lv, ref if ref else lv, freed])
        if match_now:
            rec |= align_step(tile, args, model, tokens, opt,
                              annealing=ph == "A" and annealing(it, steps_a, args.alpha_hold))
            if it == args.match_every and tile.tied_fraction() == 0:
                rec["warn"] = "tied_frac 0 at the first match step"
                print(f"WARNING step {it}: no tile neuron is tied", flush=True)
            mx.eval(tile.parameters(), opt.state)
            if not _all_finite([], tile.parameters()):           # review 2026-11-25: matching runs after the guard
                return _nonfinite_stop(out, tile, it, ph, loss, grads, log, "parameter after alignment matching")
        if ph == "B" and track and track_now:
            tt0 = time.time()
            a_tr, a_he = track_nll(model, tile, track["train"]), track_nll(model, tile, track["held"])
            g_tr, g_he = track["base_train"] - a_tr, track["base_held"] - a_he
            rec |= {"track_train": round(a_tr, 4), "track_held": round(a_he, 4),
                    "base_train": round(track["base_train"], 4), "base_held": round(track["base_held"], 4),
                    "gain_train": round(g_tr, 4), "gain_held": round(g_he, 4),
                    "gain_train_minus_held": round(g_tr - g_he, 4), "track_s": round(time.time() - tt0, 2)}
        if logging:
            tf, tf_l = state_errors(tile, model, held, 1.0)
            dr, dr_l = state_errors(tile, model, held, 0.0)
            rec |= {"step": it, "loss": round(float(loss.item()), 5), "tf_err": tf, "drift": dr,
                    "drift_last": dr_l[-1], "tied_frac": tile.tied_fraction(),
                    "elapsed_s": round(t_used + time.time() - t0, 1), **PB.mem_stats()}
            if ph == "A":
                rec["imit"] = rec["loss"]
            else:
                rec |= {"nll": round(float(nll.item()), 4), "kl": round(float(kl.item()), 5)}
            if ph == "A" or alpha_end != 0.0:
                rec["err_at_alpha_b"] = state_errors(tile, model, held, tile.alpha)[0]
            if it == steps_a:
                rec["end_of_A"] = {"tf_err_by_layer": tf_l, "drift_by_layer": dr_l}
            log.append(rec)
            print(rec, flush=True)
    PB.save_state(out, tile.parameters(), opt, it, rngs, log, t_used + time.time() - t0)
    wpath = out / "tile.safetensors"
    mx.save_safetensors(str(wpath), dict(tree_flatten(tile.parameters())))
    receipt = {"schema": "ens.h56.side_tile_train.v1", "recipe": RECIPE, "base_checkpoint": W["ckpt"],
               "tile_checkpoint": f"{_tile_name(args)}@{CT.sha16(wpath)[:12]}", "arm": args.arm,
               "align": tile.align, "donor_lora": DONOR_LORA, "steps_a": steps_a, "steps_b": steps_b,
               "compute_matched_extra_b": extra_b, "phase_seconds": _phase_seconds(log, steps_a),
               "corpus_hashes": W["hashes"], "label": "private_corpus: never publish",
               "aligned_layers": lay, "write_sites": tile._sites,
               "ladder_retained_variance": stats["retained"], "params": params,
               "trainable_params": params["trainable_B"], "n_train_chunks": len(chunks),
               "epochs": round(steps_b * (args.batch - n_text) / max(1, len(chunks)), 2),
               "tied_frac_end": tile.tied_fraction(), "args": vars(args), "log": log,
               **C.source_revision(), **C.host_record()}
    if args.track_every:
        receipt["track"] = [{k: r[k] for k in ("step", *TRACK_KEYS, "track_s")} for r in log if "gain_train" in r]
    C.write_json(out / "train_receipt.json", receipt)
    print("trained", receipt["tile_checkpoint"], flush=True)
    return 0


# ------------------------------------------------------------------ evaluation (H51's nll_eval and qa_eval, unchanged)
def load_run(run: Path, W, targs):
    tile = make_tile(W, load_stats(run), targs)
    weights = run / "tile.safetensors"
    receipt = json.loads((run / "train_receipt.json").read_text())
    if CT.sha16(weights)[:12] != receipt["tile_checkpoint"].rpartition("@")[2]:
        raise ValueError(f"{weights}: weights differ from the training receipt")
    tile.load_weights(str(weights))
    tile.freeze()
    tile.alpha = float(targs.alpha_b)
    return tile


def evaluate(args, world=None) -> int:
    run = Path(args.run)
    rec = json.loads((run / "train_receipt.json").read_text())
    targs = argparse.Namespace(**({"accum": 1} | rec["args"]))
    if not args.no_memory_gate:
        PB.memory_gate(gib=args.gate_gib)
    W = world if world is not None else build_world(targs)
    if W["ckpt"] != rec["base_checkpoint"] or W["hashes"] != rec["corpus_hashes"]:
        raise SystemExit("base checkpoint or corpus files differ from the training receipt")
    tile = load_run(run, W, targs)
    eb = {"nll_batch": args.nll_batch, "text_batch": args.text_batch, "qa_batch": args.qa_batch}
    held = mx.array(CT.text_chunks(W["tok"], "held", n=16, n_tok=128, seed=4))
    res = {"nll": CT.nll_eval(W, tile, targs, fwd=forward, chunk_batch=args.nll_batch, text_batch=args.text_batch),
           "eval_batches": eb,
           "state_error": {"teacher_forced": state_errors(tile, W["model"], held, 1.0),
                           "free_running": state_errors(tile, W["model"], held, 0.0),
                           f"alpha_{tile.alpha:g}": state_errors(tile, W["model"], held, tile.alpha)},
           "arm_names": {"lora_only": "tile write zeroed; equals base (side arms carry no donor LoRA)"},
           "tied_frac": tile.tied_fraction()}
    if args.questions:
        from research.ens import corpus_tile_eval as E
        res |= E.qa_eval(W, tile, targs, Path(args.questions).expanduser(), run, batch=args.qa_batch, fwd=forward)
    receipt = rec | {"schema": "ens.h56.side_tile_eval.v1", "eval": res,
                     "eval_source_revision": C.source_revision(), "eval_host": C.host_record()}
    C.write_json(run / "receipt.json", receipt)
    print("eval", json.dumps(res), flush=True)
    return 0


def params_only(args) -> int:
    """Parameter counts per arm for a donor config (config.json only; no weights loaded)."""
    path = Q.snapshot_dir(args.model)
    cfg = Q.Qwen3Config.from_hf(json.loads((path / "config.json").read_text()))
    print(json.dumps(count_params(cfg, args), indent=1))
    return 0


def count_params(cfg, a, arms=tuple(ARMS)):
    lay = aligned_layers(cfg.num_hidden_layers, a.every)
    n = len(lay)
    stats = {"mu": np.zeros((n, cfg.hidden_size), np.float32), "P": np.zeros((n, a.width, cfg.hidden_size), np.float32),
             "r": np.ones((n,), np.float32)}
    out = {}
    for arm in arms:
        al = ARMS[arm]
        t = SideTile(cfg, stats, None, width=a.width, every=a.every, write_site=a.write_site,
                     gate_per_layer=a.gate_per_layer, heads=a.heads, ffn_mult=a.ffn_mult, align=al,
                     soft_k=a.soft_k, soft_rank=a.soft_rank, tau=a.tau, alpha=a.alpha_b)
        rep = param_report(t)
        if al == "grow":       # starts fully tied
            rep["effective_B_at_start"] = rep["trainable_B"] - len(t.blocks) * t.blocks[0].mlp.tie_mask.shape[0] * a.width
        out[arm] = rep
    return out


def _shape_args(t):
    t.add_argument("--model", default=Q.QWEN3_06B)
    t.add_argument("--arm", choices=list(ARMS), default="imitate")
    t.add_argument("--width", type=int, default=256)
    t.add_argument("--every", type=int, default=1, help="one tile block per this many donor blocks")
    t.add_argument("--write-site", default="last", help="last | mid | donor block index (write at its input)")
    t.add_argument("--gate-per-layer", action="store_true", help="variant: a write at every aligned layer")
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--ffn-mult", type=int, default=3)
    t.add_argument("--tau", type=float, default=0.5)
    t.add_argument("--alpha-b", type=float, default=0.5,
                   help="ladder alpha held in phase B (default 0.5: ladders read the donor; 0 = free-running variant)")
    t.add_argument("--soft-k", type=int, default=8)
    t.add_argument("--soft-rank", type=int, default=16)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--corpus", required=True)
    _shape_args(t)
    t.add_argument("--steps-a", type=int, default=2000, help="phase A (imitation) steps; ignored by correction")
    t.add_argument("--steps", type=int, default=12000, help="phase B (corpus) steps")
    t.add_argument("--alpha-hold", type=float, default=0.25)
    t.add_argument("--anneal-to-zero", action="store_true",
                   help="old phase-A anneal 1 -> 0 (default 1 -> --alpha-b, so A ends in the mode B uses)")
    t.add_argument("--compute-matched", action="store_true",
                   help="correction only: add round(--steps-a x --a-cost-ratio) phase-B steps (compute-matched control)")
    t.add_argument("--a-cost-ratio", type=float, default=0.5,
                   help="cost of a phase-A step / a phase-B step (estimate; receipts log the measured ratio)")
    t.add_argument("--a-len", type=int, default=512)
    t.add_argument("--batch", type=int, default=6)
    t.add_argument("--accum", type=int, default=1,
                   help="micro-batches per step (gradient accumulation, exact; 1 = the unaccumulated path)")
    t.add_argument("--p-text", type=float, default=0.25)
    t.add_argument("--kl-weight", type=float, default=1.0)
    t.add_argument("--lr-a", type=float, default=1e-3)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--clip", type=float, default=1.0, help="global gradient-norm clip (0 = off; runs before 2026-11-25 had none)")
    t.add_argument("--match-every", type=int, default=100)
    t.add_argument("--match", choices=["greedy", "hungarian"], default="greedy")
    t.add_argument("--tau-on", type=float, default=0.5)
    t.add_argument("--tau-off", type=float, default=0.3)
    t.add_argument("--tau-on-act", type=float, default=0.5)
    t.add_argument("--tau-off-act", type=float, default=0.3)
    t.add_argument("--grow-eps", type=float, default=0.01)
    t.add_argument("--grow-frac", type=float, default=0.1)
    t.add_argument("--grow-max-frac", type=float, default=0.5, help="align_grow: at most this share freed per run")
    t.add_argument("--stat-rows", type=int, default=256)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--track-every", type=int, default=0,
                   help="every N phase-B steps log tile vs base NLL on a fixed 8 + 8 trained / held chunk sample (0 = off)")
    t.add_argument("--save-every", type=int, default=250)
    t.add_argument("--max-seconds", type=float, default=None)
    t.add_argument("--max-steps-this-launch", type=int, default=None, help="stop cleanly (exit 3) after this many")
    t.add_argument("--gate-gib", type=float, default=16.0, help="free memory required before loading")
    t.add_argument("--no-memory-gate", action="store_true")
    t.add_argument("--chunk-tag", default=None)
    t.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--run", required=True)
    e.add_argument("--questions", default=None)
    e.add_argument("--gate-gib", type=float, default=16.0)
    e.add_argument("--no-memory-gate", action="store_true")
    e.add_argument("--nll-batch", type=int, default=4, help="document chunks (512 tokens) per NLL forward")
    e.add_argument("--text-batch", type=int, default=16, help="ordinary-text rows (128 tokens) per NLL forward")
    e.add_argument("--qa-batch", type=int, default=8, help="questions per generation / choice batch")
    c = sub.add_parser("params")
    _shape_args(c)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    sys.exit(train(a) if a.cmd == "train" else evaluate(a) if a.cmd == "eval" else params_only(a))


if __name__ == "__main__":
    main()
