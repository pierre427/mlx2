"""Contributor module training against a frozen core (spec v0.4 section 6.1, step 4).

The contributor receives the published ABI (bus width, site, tap/field
positions, the codebook digest), the frozen core artifact and the *training
pool* of one family. It never sees held rows, core-training recipes or other
modules. Two module kinds:

  mlp           the contributor module: a small MLP from the bus-site tap view at
                the query position to a bus message. It learns the family's
                hidden law only through the frozen core's answer loss.
  oracle-table  a bus-capacity probe, not a contributor: a free per-z message
                table indexed by the true intermediate. It measures how much the
                frozen port can carry for this family in either arm, which the
                typed-field oracle of c7 cannot (review finding, 2026-11-21).

Confirmation families are sealed: --unseal-confirm is required, and should only
be passed after the decision record is written.

  python -m research.ens.contrib --core <core dir> --family perm.8 --seed 0 --out <dir>
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from research.ens import abi
from research.ens import common as C
from research.ens import families as F
from research.ens import vocab as V
from research.ens.model import BusCall, value_logits

RECIPES = {"mlp": "ens.a0.contrib.mlp.v1", "oracle-table": "ens.a0.probe.oracle_table.v1",
           "key-table": "ens.a0.contrib.key_table.v1", "key-choice": "ens.a0.contrib.key_choice.v1"}
_EPS = 1e-6


def unit_rms(m):
    """Messages leave a module at unit RMS, the published codebook's scale (v1, 2026-11-21).

    Pilot 2's free-scale MLP messages reached RMS 600-1100; the final-site bound then clipped every
    one, so only direction reached the core and the reader saw inputs far off the codebook's range.
    """
    return m * mx.rsqrt(mx.mean(m * m, axis=-1, keepdims=True) + _EPS)


class Module(nn.Module):
    kind = "mlp"

    def __init__(self, width: int, hidden: int = 256):
        super().__init__()
        self.l1 = nn.Linear(width, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, width)

    def __call__(self, b):
        return unit_rms(self.l3(nn.silu(self.l2(nn.silu(self.l1(b))))))


class OracleTable(nn.Module):
    kind = "oracle-table"

    def __init__(self, width: int, z_domain: int):
        super().__init__()
        self.table = nn.Embedding(z_domain, width)

    def __call__(self, z):
        return unit_rms(self.table(z))


class KeyChoice(nn.Module):
    """A 1-byte tile whose parameter *is* the stored form (H11 follow-up): per key, logits over the published
    codebook. Training reads softmax(logits / temp) @ M, annealed toward one-hot; serving reads M[argmax],
    one byte per key. Straight-through snapping can lock onto a wrong row; a soft choice can move off it."""
    kind = "key-choice"

    def __init__(self, width: int):
        super().__init__()
        self.logits = mx.zeros((V.N_KEYS, V.N_VALUES))
        self.temp = 1.0
        self.hard = False

    def __call__(self, x):
        M = _codebook_cached(int(self.width_hint))
        lg = self.logits[x]
        if self.hard:
            return M[mx.argmax(lg, axis=-1)]
        return mx.softmax(lg / self.temp, axis=-1) @ M


class KeyTable(nn.Module):
    """A deterministic tile (H11): a lookup keyed by the prompt's visible key token, no compute.

    The PLE / Engram shape under the bus contract: trigger = the key, value = a bus message. Unlike the
    oracle table it never sees z; it has to learn H(x) from the answer loss like the MLP does.
    """
    kind = "key-table"

    def __init__(self, width: int, snap: bool = False):
        super().__init__()
        self.table = nn.Embedding(V.N_KEYS, width)
        self.snap = snap      # train against the codebook-snapped table (straight-through), so the 1-byte
                              # stored form is the trained form (H11 follow-up)

    def __call__(self, x):
        m = unit_rms(self.table(x))
        if not self.snap:
            return m
        M = _codebook_cached(m.shape[-1])
        snapped = M[mx.argmax(m @ M.T, axis=-1)]
        return m + mx.stop_gradient(snapped - m)


@functools.lru_cache(maxsize=None)
def _codebook_cached(width: int):
    return C.codebook(width)


def build(kind: str, width: int, *, hidden: int, z_domain: int, snap: bool = False):
    if kind == "mlp":
        return Module(width, hidden)
    if kind == "key-table":
        return KeyTable(width, snap=snap)
    if kind == "key-choice":
        m = KeyChoice(width)
        m.width_hint = width
        return m
    return OracleTable(width, z_domain)


def message(mod, tokens, zs, q=None):
    """The module's message source: a tap callable (mlp), the key-indexed table (key-table, reads only the
    visible key token at the query position), or the z-indexed probe (oracle-table)."""
    if mod.kind == "mlp":
        return mod
    if mod.kind in ("key-table", "key-choice"):
        B = tokens.shape[0]
        qi = mx.full((B,), 2, dtype=mx.int32) if q is None else q
        if not bool(mx.all(tokens[:, 1] == V.Q).item()):
            raise ValueError("key-table needs bus/typed-layout rows (Q marker at position 1)")
        k = tokens[mx.arange(B), qi] - V.K0
        if not bool(mx.all((k >= 0) & (k < V.N_KEYS)).item()):
            raise ValueError("key-table: query position does not hold a key token")
        return mod(k)
    return mod(zs)


def core_site(core_dir) -> str:
    rec = json.loads((Path(core_dir) / "receipt.json").read_text())
    return rec.get("site", "middle")          # receipts before the site knob used the middle site


def check_core_families(core_dir) -> None:
    """The core's training families must be exactly today's. Only they shape the core (with random z);
    contributor families are checked per module through family_digest."""
    rec = json.loads((Path(core_dir) / "receipt.json").read_text())
    want = {f["fid"]: f["digest"] for f in rec["families"]["families"] if f["split"] == "train"}
    have = {f.fid: f.digest for f in F.split("train")}
    if want != have:
        raise SystemExit("training-family registry differs from the one the core was trained with")


def manifest(core_ckpt: str, cfg, site: str, fam: F.Family, weights_path, kind: str) -> dict:
    w = cfg.bus_width
    suffix = {"mlp": "", "key-table": ".table", "key-choice": ".choice", "oracle-table": ".probe"}[kind]
    key_input = kind in ("key-table", "key-choice")
    return {
        "schema": abi.SCHEMA,
        "address": f"ens://0/a0.local/{fam.fid}{suffix}@0.1.0",
        "core_artifact": core_ckpt,
        "neural_abi_major": 0,
        "execution_abi_major": 0,
        "law": "stateless.residual.v1",
        "implementation_digest": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ports": [
            {"direction": "consume", "site": "prompt.key" if key_input else (f"{site}.tap.query" if kind == "mlp" else "oracle.z"),
             "type": "latent.bus.v1", "axes": ["B", "D"], "width": w, "dtype": "float32",
             "encoding": "key.index.v0" if key_input else ("tap.linear.v0" if kind == "mlp" else "z.index.v0")},
            {"direction": "produce", "site": f"{site}.residual.field", "type": "latent.bus.v1", "axes": ["B", "D"],
             "width": w, "dtype": "float32", "encoding": f"codebook.sha256:{C.codebook_digest(w)}"},
        ],
        "state": {"kind": "stateless", "clock": "none", "rollback": "none"},
        "composition": {"writers": "exclusive.v1", "bound": "final_l2.v1", "tau_milli": int(round(cfg.tau * 1000))},
        "limits": {"max_batch": 256, "max_positions": F.SEQ_LEN, "max_workspace_bytes": 1 << 20},
        "artifacts": [abi.artifact_descriptor(weights_path)],
    }


def train(args) -> dict:
    out = C.fresh_out(args.out)
    core, core_ckpt = C.load_core(args.core)
    core.freeze()
    if core.cfg.loop_body:     # an H45 loop core needs passes=; this trainer would silently run it at one pass
        raise SystemExit("this trainer does not drive the loop port; refusing an H45 loop core")
    check_core_families(args.core)
    site = core_site(args.core)
    fam = F.BY_ID[args.family]
    if fam.split not in F.CONTRIBUTOR_SPLITS:
        raise SystemExit("contributors train on dev/confirm/gen_* families only")
    if fam.split in F.SEALED and not args.unseal_confirm:
        raise SystemExit(f"{fam.fid} is a sealed confirmation family; pass --unseal-confirm only after "
                         "the decision record is written")
    pool = F.family_rows(fam, "train")
    mx.random.seed(args.seed)
    if args.snap_codebook and args.kind != "key-table":
        raise SystemExit("--snap-codebook applies to key-table tiles")
    mod = build(args.kind, core.cfg.bus_width, hidden=args.hidden, z_domain=fam.z_domain, snap=args.snap_codebook)

    def loss_fn(m, tokens, q, f, a, ans, zs):
        logits = value_logits(core(tokens, BusCall(site, q, f, message(m, tokens, zs, q))), a)
        return nn.losses.cross_entropy(logits, ans, reduction="mean")

    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
    grad_fn = nn.value_and_grad(mod, loss_fn)
    rng = random.Random(20_000 + args.seed)
    t0, log = time.time(), []
    for it in range(1, args.steps + 1):
        rows = [pool[rng.randrange(len(pool))] for _ in range(args.batch)]
        tokens, q, f, a, ans, zs = C.batch(rows, "bus")
        loss, g = grad_fn(mod, tokens, q, f, a, ans, zs)
        opt.update(mod, g)
        mx.eval(mod.parameters(), opt.state)
        if mod.kind == "key-choice":          # anneal 1.0 -> 0.05 over training, then serve hard
            mod.temp = max(0.05, 1.0 * (0.05 ** (it / args.steps)))
        if it % args.log_every == 0 or it == args.steps:
            log.append({"step": it, "loss": round(float(loss.item()), 5), "elapsed_s": round(time.time() - t0, 1)})
    if mod.kind == "key-choice":
        mod.hard = True
    wpath = out / "module.safetensors"
    mx.save_safetensors(str(wpath), dict(tree_flatten(mod.parameters())))
    man = manifest(core_ckpt, core.cfg, site, fam, wpath, args.kind)
    if args.snap_codebook:      # the served function differs from an unsnapped tile: part of the package
        man["limits"]["serving_form"] = "codebook_index.v1"
    C.write_json(out / "manifest.json", man)
    receipt = {"schema": "ens.a0.contrib_receipt.v0",
               "recipe": RECIPES[args.kind] + (".snap" if args.snap_codebook else ""), "kind": args.kind,
               "hidden": args.hidden, "snap_codebook": args.snap_codebook, "core_checkpoint": core_ckpt, "site": site,
               "family": fam.record(), "family_digest": fam.digest, "seed": args.seed,
               "budget": {"updates": args.steps, "batch": args.batch, "lr": args.lr, "train_pool": len(pool),
                          "train_pool_distinct": len({r.signature for r in pool})},
               "package_digest": abi.package_digest(man), "contract_digest": abi.contract_digest(man),
               "log": log, "assistance": args.assistance, **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    return receipt


def load_module(d, width):
    rec = json.loads((Path(d) / "receipt.json").read_text())
    fam = F.BY_ID[rec["family"]["fid"]]
    kind = rec.get("kind", "mlp")
    if kind not in RECIPES or rec.get("recipe", "").removesuffix(".snap") != RECIPES[kind]:
        raise SystemExit(f"{d}: module recipe {rec.get('recipe')} is not served by this code (v1 normalizes "
                         "messages to unit RMS)")
    path = Path(d)
    man = abi.parse((path / "manifest.json").read_text())
    if (abi.package_digest(man) != rec["package_digest"]
            or man["artifacts"] != [abi.artifact_descriptor(path / "module.safetensors")]
            or man["core_artifact"] != rec["core_checkpoint"]):
        raise SystemExit(f"{d}: module artifact or package digest differs from the training receipt")
    m = build(kind, width, hidden=rec.get("hidden", 256), z_domain=fam.z_domain,
              snap=rec.get("snap_codebook", False))
    m.load_weights(str(Path(d) / "module.safetensors"))
    if kind == "key-choice":
        m.hard = True
    m.freeze()
    return m


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--core", required=True)
    p.add_argument("--family", required=True, choices=sorted(F.BY_ID))
    p.add_argument("--kind", default="mlp", choices=sorted(RECIPES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--unseal-confirm", action="store_true")
    p.add_argument("--snap-codebook", action="store_true",
                   help="key-table only: straight-through training against the 1-byte codebook-snapped table")
    p.add_argument("--assistance", default="none", help="record any documentation help given to the contributor")
    p.add_argument("--out", required=True)
    train(p.parse_args(argv))


if __name__ == "__main__":
    main()
