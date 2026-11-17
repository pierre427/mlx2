#!/usr/bin/env python3
"""Shared MLX machinery for the amoeba pipeline on Qwen3.8-27B (docs/research/amoeba-27b/DESIGN.md).

The 0.6B recipe (probe_amoeba_english_core_cpu.py, probe_amoeba_skill_boxes_cpu.py) ported to MLX
on the GPU, because a torch fp32 27B does not fit:

- AmoebaMLP wraps each decoder layer's SwiGLU MLP. With no core set it is the base MLP, bit for
  bit (``down(silu(gate(x)) * up(x))`` through the same quantized projections). With a core it uses
      used = mean + m * (act - mean)      (mean ablation; zero ablation when mean is None)
      m    = core_l + sum_b box_l * g_b(x),  g_b = sigmoid(x . W_b[l] + b_b[l]) (optional, per token)
  Position 0 keeps the exact activation in layers < sink_layers (training: every layer, as at 0.6B).
  Layers in dense_layers run whole at every position (the dense sink stand-in).
- Mask training is the 0.6B straight-through recipe: forward uses (Z > 0), gradients flow through
  sigmoid(Z), loss = KL(full || masked) + lam * mean kept fraction of the free units.
- Data: the 2026-11-13 amoeba corpus (Qwen3-0.6B ids) decoded and re-encoded with the 27B tokenizer.

Research tooling only; nothing here is imported by src/mlx2.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODEL_27B = Path.home() / "mlx-models/Qwen3.8-27B-BF16-q8-mlx2"     # Pierre 2026-11-14: BF16 base, q8 build
BF16_27B = Path.home() / "Models/Qwen3.8-27B-BF16"
CORPUS = Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba/corpus-2026-11-13/corpus.json"
ARTIFACTS = Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba"
DOMAINS = ["tinystories_train", "repo_markdown", "donor_generated"]


# ----------------------------------------------------------------------------- run plumbing

def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def sha12(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def fresh_dir(path) -> Path:
    out = Path(path)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty evidence dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_log(out: Path):
    f = open(out / "run.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


def footprint_gb(pid: int) -> float | None:
    out = subprocess.run(["/usr/bin/top", "-l", "1", "-pid", str(pid), "-stats", "mem"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return None
    m = re.match(r"([\d.]+)([KMGT])", out[-1].strip())
    if not m:
        return None
    return float(m.group(1)) * {"K": 1e-6, "M": 1e-3, "G": 1.0, "T": 1e3}[m.group(2)]


def swap_used_gb() -> float:
    out = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) / 1024 if m else 0.0


def pressure_level() -> int:
    out = subprocess.run(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True).stdout
    try:
        return int(out.strip())
    except ValueError:
        return 0


WATCH = {}


def footprint_watchdog(limit_gb: float, log, every_s: float = 15.0, min_avail_gb: float = 15.0, swap_growth_gb: float = 2.0):
    """Exit 3 if this process's physical footprint passes limit_gb, machine available memory falls
    under min_avail_gb, swap grows more than swap_growth_gb above its level now, or memory pressure
    goes critical (coordinator's waiver terms, 2026-11-14). Peak footprint is kept in WATCH."""
    import amoeba_memcheck
    swap0 = swap_used_gb()
    WATCH.update(swap_at_start_gb=round(swap0, 2), peak_footprint_gb=0.0, limit_gb=limit_gb,
                 min_avail_gb=min_avail_gb, swap_growth_gb=swap_growth_gb)

    def run():
        pid = os.getpid()
        while True:
            sw = swap_used_gb()
            if sw > swap0 + swap_growth_gb:
                log(f"watchdog: swap {sw:.1f} GB grew more than {swap_growth_gb} GB over {swap0:.1f} GB, aborting")
                os._exit(3)
            if pressure_level() >= 4:
                log("watchdog: memory pressure critical, aborting")
                os._exit(3)
            fp = footprint_gb(pid)
            if fp is not None:
                WATCH["peak_footprint_gb"] = max(WATCH["peak_footprint_gb"], round(fp, 1))
            if fp is not None and fp > limit_gb:
                log(f"footprint watchdog: {fp:.1f} GB > {limit_gb} GB, aborting")
                os._exit(3)
            av = amoeba_memcheck.snapshot_avail()
            if av < min_avail_gb:
                log(f"footprint watchdog: machine available {av:.1f} GB < {min_avail_gb} GB, aborting")
                os._exit(3)
            time.sleep(every_s)
    threading.Thread(target=run, daemon=True).start()


ALLOWED_FOREIGN = {760}   # Pierre's MLX2 desktop app; named in the coordinator's waiver (2026-11-14)


def big_by_footprint(limit_gb: float = 8.0, exclude=()):
    """Other processes whose physical footprint (top's mem) exceeds limit_gb: the standing memory
    rule (resident memory counts mapped files; coordinator 2026-11-14). Skips ALLOWED_FOREIGN,
    hf downloads and ``exclude``."""
    import amoeba_memcheck
    mine = {os.getpid(), os.getppid()}
    out = []
    for b in amoeba_memcheck.snapshot()["foreign_procs_over_8gb"]:
        pid = b["pid"]
        if pid in mine or pid in ALLOWED_FOREIGN or pid in set(exclude) or is_hf_download(pid):
            continue
        fp = footprint_gb(pid)
        if fp is not None and fp > limit_gb:
            out.append({**b, "footprint_gb": round(fp, 1)})
    return out


def is_hf_download(pid) -> bool:
    """hf / huggingface-cli download processes: footprint is mostly file cache, not a model.
    Exempt from the foreign-process rule under the coordinator's waiver (2026-11-14, as
    amoeba_memguard c9a57cac6); available-memory, swap-growth and footprint aborts still apply."""
    cmd = subprocess.run(["/bin/ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
    return bool(re.search(r"(^|/)(hf|huggingface-cli)\s+download\b", cmd))


def admit(need_gb: float, policy: str, log, wait_s: int = 0) -> dict:
    """amoeba_memcheck.require, plus (under a waiver) refuse while any other process over 8 GB runs
    apart from ALLOWED_FOREIGN: no second large model process, whatever available memory says."""
    import amoeba_memcheck
    if policy.startswith("auto:"):
        # strict when it passes (Pierre freed memory 2026-11-14), else the coordinator's waiver
        ok, rec = amoeba_memcheck.admit(need_gb, "strict")
        policy = "strict" if ok else policy[len("auto:"):]
        log(f"admit: auto policy chose {policy} (strict {'passed' if ok else 'refused: ' + str(rec.get('reasons'))})")
    if policy.startswith("waived:"):
        end = time.time() + wait_s
        while True:
            big = big_by_footprint()
            if not big:
                break
            log(f"admit: waiting, other large processes {[(b['pid'], b.get('rss_gb', b.get('footprint_gb')), b['command'][:60]) for b in big]}")
            if time.time() >= end:
                sys.exit("admit: another large process is running")
            time.sleep(60)
    rec = amoeba_memcheck.require(need_gb, policy, wait_s=wait_s, log=log)
    rec["waiver"] = policy if policy.startswith("waived:") else None
    rec["exempt_hf_downloads"] = [b["pid"] for b in rec.get("foreign_procs_over_8gb", []) if is_hf_download(b["pid"])]
    rec["waiver_terms"] = ("coordinator 2026-11-14: allowed foreign pid 760 only; abort on swap +2 GB, critical "
                           "pressure, available < 15 GB, own footprint > 80 GB (Pierre raised 60 to 80, 2026-11-14)") if policy.startswith("waived:") else None
    return rec


def setup_mlx(memory_gb: float, cache_gb: float):
    import mlx.core as mx
    mx.set_memory_limit(int(memory_gb * 2**30))
    mx.set_cache_limit(int(cache_gb * 2**30))


def peak_gb() -> float:
    import mlx.core as mx
    return round(mx.get_peak_memory() / 2**30, 2)


# ----------------------------------------------------------------------------- model

class Ctl:
    """Mutable state shared by every AmoebaMLP of one model (set per forward)."""

    def __init__(self, L: int, N: int):
        self.L, self.N = L, N
        self.reset()

    def reset(self):
        self.core = None            # [L, N] mask, or None = full model
        self.mean = None            # [L, N] or None (zero ablation)
        self.sink_layers = 0        # position 0 exact in layers < sink_layers
        self.dense_layers = ()      # layers that run whole at every position
        self.boxes = []             # [{"mask": [L,N], "gate": (W [L,d], b [L]) | None, "hard": bool}]
        self.stats = None           # callback(l, act, x) or None
        self.gate_log = None        # list to append per-layer mean gate values, or None


def _silu(x):
    import mlx.core as mx
    return x * mx.sigmoid(x)


def make_wrapper_class():
    import mlx.core as mx
    import mlx.nn as nn

    class AmoebaMLP(nn.Module):
        def __init__(self, base, layer: int, ctl: Ctl):
            super().__init__()
            self.gate_proj, self.up_proj, self.down_proj = base.gate_proj, base.up_proj, base.down_proj
            self._layer = layer
            self._ctl = ctl

        def __call__(self, x):
            c, l = self._ctl, self._layer
            act = nn.silu(self.gate_proj(x)) * self.up_proj(x)   # = activations.swiglu, the served MLP
            if c.stats is not None:
                c.stats(l, act, x)
            if c.core is None or l in c.dense_layers:
                return self.down_proj(act)
            m = c.core[l]
            for box in c.boxes:
                bm = box["mask"][l]
                if box.get("gate") is not None:
                    W, b = box["gate"]
                    g = mx.sigmoid((x.astype(mx.float32) @ W[l][:, None]) + b[l])      # B,T,1
                    if box.get("hard"):
                        g = (g > 0.5).astype(mx.float32)
                    if c.gate_log is not None:
                        c.gate_log.append((box.get("name"), l, g))
                    m = m + bm * g
                else:
                    m = m + bm
            m = m.astype(act.dtype)
            if c.mean is not None:
                mu = c.mean[l].astype(act.dtype)
                used = mu + m * (act - mu)
            else:
                used = m * act
            if l < c.sink_layers and x.shape[1] > 1:
                used = mx.concatenate([act[:, :1], used[:, 1:]], axis=1)
            elif l < c.sink_layers:
                pass  # a single-row forward is never position 0 in these probes (full sequences only)
            return self.down_proj(used)

    return AmoebaMLP


class Host:
    """The 27B loaded through its mlx2 adapter, with every MLP wrapped."""

    def __init__(self, path=MODEL_27B, log=print, policy=None, wrap=True):
        """policy: adapter execution policy (e.g. {"amoeba_tiles": manifest}); wrap=False leaves the
        live MLPs alone (tile arms are evaluated through the serving runtime, not the research wrapper)."""
        import mlx.core as mx
        from mlx2.adapters.registry import inspect_model
        t0 = time.time()
        self.path = Path(path)
        cfg = json.loads((self.path / "config.json").read_text())
        if (cfg.get("text_config") or cfg).get("num_hidden_layers") == 64:
            self.adapter = inspect_model(self.path).adapter_type(str(self.path), **({"execution_policy": policy} if policy else {}))
            self.model = self.adapter.model
            self.tok = self.adapter.tokenizer
            self.fingerprint = self.adapter.identity["fingerprint"]
        else:                        # reduced-layer smoke slice: the adapter's inspection wants 64 layers
            self._load_direct(cfg)
        self.checkpoint = f"{self.path.name}@{self.fingerprint.split(':')[-1][:8]}"
        self.lm = self.model.language_model
        self.layers = list(self.lm.model.layers)
        self.L = len(self.layers)
        self.N = int(self.lm.args.intermediate_size)
        self.d = int(self.lm.args.hidden_size)
        self.ctl = Ctl(self.L, self.N)
        if wrap:
            W = make_wrapper_class()
            for i, layer in enumerate(self.layers):
                layer.mlp = W(layer.mlp, i, self.ctl)
        self.model.freeze()
        self.model.eval()
        mx.eval(self.model.parameters())
        log(f"loaded {self.checkpoint} ({self.L} layers x {self.N} neurons, d={self.d}) in {time.time() - t0:.0f}s, "
            f"MLX active {mx.get_active_memory() / 2**30:.1f} GB")

    def _load_direct(self, cfg):
        """Build the 27B model class exactly as the adapter does (sanitize, quantize, strict load)."""
        import hashlib as _h
        import mlx.core as mx
        from transformers import AutoTokenizer
        from mlx2.adapters.qwen38_27b import _quantize_target
        from mlx2.runtime.models.qwen38_27b import Model, ModelArgs
        cfg = dict(cfg)
        cfg["text_config"] = dict(cfg["text_config"])
        self.model = Model(ModelArgs.from_dict(cfg))
        weights = {}
        for f in sorted(self.path.glob("*.safetensors")):
            weights.update(mx.load(str(f)))
        weights = self.model.sanitize(weights)
        _quantize_target(self.model, cfg.get("quantization"), weights)
        self.model.load_weights(list(weights.items()), strict=True)
        self.adapter = None
        self.tok = AutoTokenizer.from_pretrained(str(self.path))
        self.fingerprint = "slice:" + _h.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False) if hasattr(self.tok, "encode") else self.tok(text)["input_ids"]

    def head(self, h):
        if self.lm.args.tie_word_embeddings:
            return self.lm.model.embed_tokens.as_linear(h)
        return self.lm.lm_head(h)

    def hidden(self, ids):
        return self.lm.model(ids)

    def logits_at(self, ids, pos=None):
        """Logits [B, P, V] in float32 at positions pos (list, same per row; or [B, P] int array)."""
        import mlx.core as mx
        h = self.hidden(ids)
        if pos is not None:
            if isinstance(pos, mx.array) and pos.ndim == 2:
                h = mx.take_along_axis(h, pos[..., None], axis=1)
            else:
                h = h[:, mx.array(pos)]
        return self.head(h).astype(mx.float32)

    def train_hidden(self, ids, core, boxes=()):
        """Differentiable forward (training mode: GDN on the ops path) with core/box arrays that may
        be tracers. No checkpointing: measured 2026-11-14 on the 27B slice, mx.checkpoint per layer
        saved no peak memory and doubled the time, because the GDN ops recurrence keeps one state per
        token regardless. Memory is bounded instead by micro-batching (train_mask), one ~64-token
        sequence per forward/backward: about 48 layers x 64 tokens x 3.1 MB = 9.7 GB of states."""
        c = self.ctl
        saved = (c.core, c.boxes)
        c.core, c.boxes = core, list(boxes)
        try:
            return self.lm.model(ids)
        finally:
            c.core, c.boxes = saved

    def train_logits_at(self, ids, core, boxes=(), pos=None):
        import mlx.core as mx
        h = self.train_hidden(ids, core, boxes)
        if pos is not None:
            if isinstance(pos, mx.array) and pos.ndim == 2:
                h = mx.take_along_axis(h, pos[..., None], axis=1)
            else:
                h = h[:, mx.array(pos)]
        return self.head(h).astype(mx.float32)

    def train_mode(self, on: bool):
        """GDN takes the differentiable ops path in training mode (use_kernel = not training)."""
        if on:
            self.model.train()
        else:
            self.model.eval()


# ----------------------------------------------------------------------------- losses and scoring

def kl_rows(lc, lm):
    """KL(clean || masked) per row, float32 logits [..., V]."""
    import mlx.core as mx
    pc = lc - mx.logsumexp(lc, axis=-1, keepdims=True)
    pm = lm - mx.logsumexp(lm, axis=-1, keepdims=True)
    return (mx.exp(pc) * (pc - pm)).sum(-1)


def ce_rows(logits, tgt):
    import mlx.core as mx
    lse = mx.logsumexp(logits, axis=-1)
    picked = mx.take_along_axis(logits, tgt[..., None], axis=-1)[..., 0]
    return lse - picked


def text_ce(host: Host, seqs, cfg_fn=None, bs=8):
    """Mean CE (true next token) over positions 1..T-2 (as ec.score_text), and token count."""
    import mlx.core as mx
    tot, n = 0.0, 0
    for i in range(0, len(seqs), bs):
        ids = mx.array(seqs[i:i + bs])
        if cfg_fn:
            cfg_fn()
        lg = host.logits_at(ids, list(range(1, ids.shape[1] - 1)))
        host.ctl.reset() if cfg_fn else None
        ce = ce_rows(lg, ids[:, 2:]).sum()
        mx.eval(ce)
        tot += float(ce)
        n += int(ids[:, 2:].size)
        mx.clear_cache()
    return tot / n, n


def english_cost(host, split, clean_cache, cfg_fn, key_domains=DOMAINS, part="held"):
    r = {}
    for dn in key_domains:
        if dn not in clean_cache:
            clean_cache[dn] = text_ce(host, split[dn][part])[0]
        r[dn] = text_ce(host, split[dn][part], cfg_fn)[0] - clean_cache[dn]
    r["mean_d_ce"] = sum(r[d] for d in key_domains) / len(key_domains)
    return r


# ----------------------------------------------------------------------------- data

def load_corpus(host: Host, seq_len=64, n_mean=240, n_fit=900, n_held=60, seed=0, log=print):
    """Decode the corpus (Qwen3-0.6B ids) and re-encode with the host tokenizer; disjoint splits."""
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    src = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True))
    blob = json.loads(CORPUS.read_text())["seqs"]
    rng = random.Random(seed)
    split = {}
    for dn in DOMAINS:
        seqs = []
        for q in blob[dn]:
            ids = host.encode(src.decode(q))
            if len(ids) >= seq_len:
                seqs.append(ids[:seq_len])
        rng.shuffle(seqs)
        km, kf = n_mean // 3, n_fit // 3
        split[dn] = {"held": seqs[:n_held], "mean": seqs[n_held:n_held + km],
                     "fit": seqs[n_held + km:n_held + km + kf],
                     "spare": seqs[n_held + km + kf:]}
        log(f"corpus {dn}: {len(seqs)} seqs of {seq_len} (held {len(split[dn]['held'])}, mean {len(split[dn]['mean'])}, "
            f"fit {len(split[dn]['fit'])}, spare {len(split[dn]['spare'])})")
    return split


COUNTRIES = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Japan", "Tokyo"),
    ("China", "Beijing"), ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("Australia", "Canberra"),
    ("Brazil", "Brasília"), ("Argentina", "Buenos Aires"), ("Mexico", "Mexico City"), ("India", "New Delhi"),
    ("Kenya", "Nairobi"), ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Poland", "Warsaw"), ("Austria", "Vienna"),
    ("Hungary", "Budapest"), ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Finland", "Helsinki"),
    ("Denmark", "Copenhagen"), ("Ireland", "Dublin"), ("Belgium", "Brussels"), ("Netherlands", "Amsterdam"),
    ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
    ("Indonesia", "Jakarta"), ("Peru", "Lima"), ("Chile", "Santiago"), ("Colombia", "Bogotá"), ("Cuba", "Havana"),
    ("Ukraine", "Kyiv"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"), ("Serbia", "Belgrade"),
    ("Croatia", "Zagreb"), ("Czechia", "Prague"), ("Switzerland", "Bern"), ("Morocco", "Rabat"),
    ("Ethiopia", "Addis Ababa"), ("Nigeria", "Abuja"), ("Ghana", "Accra"), ("Pakistan", "Islamabad"),
    ("Afghanistan", "Kabul"), ("Nepal", "Kathmandu"), ("Mongolia", "Ulaanbaatar"), ("Philippines", "Manila"),
    ("Malaysia", "Kuala Lumpur"), ("Venezuela", "Caracas"), ("Uruguay", "Montevideo"), ("Israel", "Jerusalem"),
    ("Jordan", "Amman"), ("Syria", "Damascus"), ("Lebanon", "Beirut"), ("Iceland", "Reykjavik"),
    ("Estonia", "Tallinn"), ("Latvia", "Riga"), ("Lithuania", "Vilnius"), ("Slovakia", "Bratislava"),
    ("Slovenia", "Ljubljana"), ("Albania", "Tirana"), ("Belarus", "Minsk"), ("Moldova", "Chișinău"),
    ("Georgia", "Tbilisi"), ("Armenia", "Yerevan"), ("Azerbaijan", "Baku"), ("Kazakhstan", "Astana"),
    ("Uzbekistan", "Tashkent"), ("Kyrgyzstan", "Bishkek"), ("Tajikistan", "Dushanbe"),
    ("Turkmenistan", "Ashgabat"), ("Saudi Arabia", "Riyadh"), ("Yemen", "Sanaa"), ("Oman", "Muscat"),
    ("Qatar", "Doha"), ("Kuwait", "Kuwait City"), ("Bahrain", "Manama"), ("Bangladesh", "Dhaka"),
    ("Sri Lanka", "Colombo"), ("Myanmar", "Naypyidaw"), ("Cambodia", "Phnom Penh"), ("Laos", "Vientiane"),
    ("South Korea", "Seoul"), ("North Korea", "Pyongyang"), ("New Zealand", "Wellington"), ("Fiji", "Suva"),
    ("Algeria", "Algiers"), ("Tunisia", "Tunis"), ("Libya", "Tripoli"), ("Sudan", "Khartoum"),
    ("Senegal", "Dakar"), ("Mali", "Bamako"), ("Niger", "Niamey"), ("Chad", "N'Djamena"),
    ("Cameroon", "Yaoundé"), ("Uganda", "Kampala"), ("Tanzania", "Dodoma"), ("Rwanda", "Kigali"),
    ("Zambia", "Lusaka"), ("Zimbabwe", "Harare"), ("Angola", "Luanda"), ("Mozambique", "Maputo"),
    ("Namibia", "Windhoek"), ("Botswana", "Gaborone"), ("Madagascar", "Antananarivo"), ("Somalia", "Mogadishu"),
    ("Ecuador", "Quito"), ("Bolivia", "Sucre"), ("Paraguay", "Asunción"), ("Guatemala", "Guatemala City"),
    ("Honduras", "Tegucigalpa"), ("Nicaragua", "Managua"), ("Costa Rica", "San José"), ("Panama", "Panama City"),
    ("Jamaica", "Kingston"), ("Haiti", "Port-au-Prince"), ("Luxembourg", "Luxembourg"), ("Malta", "Valletta"),
    ("Cyprus", "Nicosia"), ("Montenegro", "Podgorica"), ("Bosnia and Herzegovina", "Sarajevo"),
    ("North Macedonia", "Skopje"), ("Bhutan", "Thimphu"), ("Maldives", "Malé"), ("Brunei", "Bandar Seri Begawan"),
    ("Papua New Guinea", "Port Moresby"), ("Eritrea", "Asmara"), ("Gabon", "Libreville"), ("Benin", "Porto-Novo"),
    ("Togo", "Lomé"), ("Liberia", "Monrovia"), ("Sierra Leone", "Freetown"), ("Guinea", "Conakry"),
    ("Burkina Faso", "Ouagadougou"), ("Malawi", "Lilongwe"), ("Lesotho", "Maseru"), ("Mauritania", "Nouakchott"),
]

FRENCH = [("dog", "chien"), ("cat", "chat"), ("house", "maison"), ("water", "eau"), ("book", "livre"),
          ("bread", "pain"), ("cheese", "fromage"), ("apple", "pomme"), ("tree", "arbre"), ("sun", "soleil"),
          ("moon", "lune"), ("star", "étoile"), ("sea", "mer"), ("river", "rivière"), ("mountain", "montagne"),
          ("city", "ville"), ("street", "rue"), ("car", "voiture"), ("boat", "bateau"),
          ("bird", "oiseau"), ("fish", "poisson"), ("horse", "cheval"), ("cow", "vache"), ("mouse", "souris"),
          ("flower", "fleur"), ("garden", "jardin"), ("door", "porte"), ("window", "fenêtre"),
          ("chair", "chaise"), ("bed", "lit"), ("kitchen", "cuisine"), ("friend", "ami"), ("mother", "mère"),
          ("father", "père"), ("brother", "frère"), ("sister", "sœur"), ("child", "enfant"), ("man", "homme"),
          ("woman", "femme"), ("king", "roi"), ("queen", "reine"), ("school", "école"), ("teacher", "professeur"),
          ("money", "argent"), ("day", "jour"), ("night", "nuit"), ("week", "semaine"),
          ("year", "année"), ("morning", "matin"), ("evening", "soir"), ("winter", "hiver"), ("summer", "été"),
          ("spring", "printemps"), ("snow", "neige"), ("rain", "pluie"), ("wind", "vent"), ("fire", "feu"),
          ("milk", "lait"), ("egg", "œuf"), ("meat", "viande"), ("wine", "vin"), ("beer", "bière"),
          ("red", "rouge"), ("blue", "bleu"), ("green", "vert"), ("black", "noir"), ("white", "blanc"),
          ("yellow", "jaune"), ("big", "grand"), ("small", "petit"), ("old", "vieux"),
          ("good", "bon"), ("bad", "mauvais"), ("happy", "heureux"), ("sad", "triste"), ("hot", "chaud"),
          ("cold", "froid"), ("hand", "main"), ("head", "tête"), ("eye", "œil"), ("heart", "cœur"),
          ("road", "route"), ("bridge", "pont"), ("church", "église"), ("market", "marché"), ("shop", "magasin"),
          ("letter", "lettre"), ("paper", "papier"), ("pen", "stylo"), ("song", "chanson"), ("game", "jeu"),
          ("love", "amour"), ("war", "guerre"), ("peace", "paix"), ("country", "pays"), ("island", "île"),
          ("knife", "couteau"), ("spoon", "cuillère"), ("forest", "forêt"), ("cloud", "nuage"), ("sky", "ciel"),
          ("butter", "beurre"), ("sugar", "sucre"), ("salt", "sel"), ("bottle", "bouteille"), ("glass", "verre")]
FR_SHOTS = "English: shoe\nFrench: chaussure\nEnglish: hat\nFrench: chapeau\n"
CAP_TEMPLATES = ["The capital of {} is", "{}'s capital city is", "The seat of government of {} is the city of"]
CAP_VAL_TEMPLATE = "In {}, the capital city is"
RECALL_FILLERS = ["What color is the sky? A: Blue.", "How many legs does a cat have? A: Four.",
                  "What do bees make? A: Honey.", "Which season is coldest? A: Winter."]


def tf_items(host: Host, rows):
    """Teacher-forced items: ids = prompt + answer; pos = positions predicting each answer token."""
    items = []
    for p, a, label in rows:
        pi, ai = host.encode(p), host.encode(a)
        items.append({"ids": pi + ai, "pos": list(range(len(pi) - 1, len(pi) + len(ai) - 1)),
                      "tgt": ai, "label": label, "prompt": p, "answer": a})
    return items


def build_tasks(host: Host, seed=0):
    """Capitals (fit / held / fresh countries, templates 0-1 for fit+held, template 2 = new phrasing),
    3-digit sums, English->French words, and multi-turn number recall. Returns tasks, splits."""
    rng = random.Random(seed)
    tasks, splits = {}, {}
    countries = list(COUNTRIES)
    random.Random(seed + 7).shuffle(countries)
    fresh = countries[:40]                          # never used to fit or choose anything
    pool = countries[40:]
    caps_rows = [(CAP_TEMPLATES[i % 2].format(c), " " + k, c) for i, (c, k) in enumerate(pool)]
    tasks["capitals"] = tf_items(host, caps_rows)
    tasks["capitals_fresh"] = tf_items(host, [(CAP_TEMPLATES[0].format(c), " " + k, c) for c, k in fresh])
    tasks["capitals_phrasing"] = tf_items(host, [(CAP_TEMPLATES[2].format(c), " " + k, c) for c, k in pool])
    pairs = set()
    while len(pairs) < 240:
        pairs.add((rng.randint(100, 999), rng.randint(100, 999)))
    tasks["arithmetic"] = tf_items(host, [(f"{a} + {b} =", f" {a + b}", f"{a}+{b}") for a, b in sorted(pairs)])
    fr = list(FRENCH)
    tasks["french"] = tf_items(host, [(FR_SHOTS + f"English: {e}\nFrench:", f" {f}", e) for e, f in fr])
    # completion splits (2026-11-14, Pierre: a box is complete when core + box matches the full model):
    # the same facts in two further phrasings; one chooses, the other reports
    tasks["capitals_val"] = tf_items(host, [(CAP_VAL_TEMPLATE.format(c), " " + k, c) for c, k in pool])
    tasks["french_val"] = tf_items(host, [(f"English: {e}\nFrench:", f" {f}", e) for e, f in fr])
    tasks["french_phrasing"] = tf_items(host, [(f"The French word for \"{e}\" is", f" \"{f}", e) for e, f in fr])
    nums = rng.sample(range(10, 99), 80)
    tasks["recall"] = tf_items(host, [(f"Remember the number {n}. Q: {RECALL_FILLERS[i % 4]} Q: What number did I ask "
                                       f"you to remember? A: The number is", f" {n}", str(n)) for i, n in enumerate(nums)])
    # Added 2026-11-14 for cluster labelling. Their own RNG, so every task above keeps its items.
    rng2 = random.Random(seed + 4242)
    mpairs = set()
    while len(mpairs) < 160:
        mpairs.add((rng2.randint(20, 999), rng2.randint(11, 99)))
    tasks["multiplication"] = tf_items(host, [(f"{a} * {b} =", f" {a * b}", f"{a}*{b}") for a, b in sorted(mpairs)])
    tasks["logic_entail"] = tf_items(host, _logic_rows(rng2, 160))
    for name, items in tasks.items():
        idx = list(range(len(items)))
        if name in ("capitals_fresh", "capitals_phrasing", "capitals_val", "french_val", "french_phrasing"):
            splits[name] = {"fit": [], "held": idx}
            continue
        random.Random(seed + len(name)).shuffle(idx)
        h = len(idx) // 2
        splits[name] = {"fit": sorted(idx[:h]), "held": sorted(idx[h:])}
    # capitals_phrasing reports only on held-out countries of the capitals split
    held_c = {tasks["capitals"][i]["label"] for i in splits["capitals"]["held"]}
    splits["capitals_phrasing"]["held"] = [i for i, it in enumerate(tasks["capitals_phrasing"]) if it["label"] in held_c]
    return tasks, splits


def _logic_rows(rng, n):
    """Reasoning Pedagogy *train* logic items (never its eval holdout): user prompt + "FINAL:" -> label."""
    import gzip
    p = Path.home() / "corpora/logic/reasoning-pedagogy-corpus-full/train/logic_sft.jsonl.gz"
    if not p.exists():
        return []
    rows = []
    with gzip.open(p, "rt") as f:
        for i, line in enumerate(f):
            if i >= 4000:
                break
            r = json.loads(line)
            user = next((m["content"] for m in r.get("prompt", []) if m.get("role") == "user"), None)
            if user and r.get("answer"):
                rows.append((user + "\nFINAL:", " " + r["answer"], r.get("family", "")))
    rng.shuffle(rows)
    return rows[:n]


def groups_of(items, idx, cap=16):
    by = {}
    for i in idx:
        by.setdefault(len(items[i]["tgt"]), []).append(i)
    out = []
    for g in by.values():
        out += [g[j:j + cap] for j in range(0, len(g), cap)]
    return out


def batch_of(items, grp, pad=0):
    import mlx.core as mx
    T = max(len(items[i]["ids"]) for i in grp)
    ids = mx.array([items[i]["ids"] + [pad] * (T - len(items[i]["ids"])) for i in grp])
    pos = mx.array([items[i]["pos"] for i in grp])
    return ids, pos


def task_eval(host: Host, items, idx, cfg_fn=None, clean=None, key=None):
    """Teacher-forced exact match of the whole answer, and mean KL vs clean (if clean cache given)."""
    import mlx.core as mx
    correct, kl_tot, n_pos = 0, 0.0, 0
    for grp in groups_of(items, idx):
        ids, pos = batch_of(items, grp)
        if cfg_fn:
            cfg_fn()
        lg = host.logits_at(ids, pos)
        host.ctl.reset()
        tgt = mx.array([items[i]["tgt"] for i in grp])
        ok = (lg.argmax(-1) == tgt).all(-1).sum()
        if clean is not None:
            kl = kl_rows(clean[(key, tuple(grp))], lg).sum()
            mx.eval(ok, kl)
            kl_tot += float(kl)
        else:
            mx.eval(ok)
        correct += int(ok)
        n_pos += int(pos.size)
        mx.clear_cache()
    return correct / max(1, len(idx)), (kl_tot / max(1, n_pos) if clean is not None else None)


def cache_clean(host: Host, items, idx, cache, key, cap=16):
    import mlx.core as mx
    for grp in groups_of(items, idx, cap=cap):
        k = (key, tuple(grp))
        if k in cache:
            continue
        ids, pos = batch_of(items, grp)
        host.ctl.reset()
        lg = host.logits_at(ids, pos)
        mx.eval(lg)
        cache[k] = lg


def mean_acts(host: Host, seqs, bs=8):
    """Mean MLP activation per neuron over positions >= 1 (float32 [L, N])."""
    import mlx.core as mx
    acc = [mx.zeros((host.N,), dtype=mx.float32) for _ in range(host.L)]
    cnt = [0]

    def stats(l, act, x):
        acc[l] = acc[l] + act[:, 1:].astype(mx.float32).reshape(-1, host.N).sum(0)
        if l == 0:
            cnt[0] += act.shape[0] * (act.shape[1] - 1)
    for i in range(0, len(seqs), bs):
        host.ctl.reset()
        host.ctl.stats = stats
        h = host.hidden(mx.array(seqs[i:i + bs]))
        mx.eval(h, *acc)
        host.ctl.reset()
        mx.clear_cache()
    return mx.stack(acc) / cnt[0]


# ----------------------------------------------------------------------------- mask training

def layerwise_value_and_grad(host, ids, trainables, apply, head_loss):
    """Loss and gradients of per-layer trainables, one layer's backward tape alive at a time.

    trainables: {name: array [L, ...]}; apply(l, {name: slice}) sets host.ctl for layer l from the
    slices (they may be tracers). head_loss(hidden_after_final_norm) -> scalar.
    Forward: layer by layer in training mode (GDN ops path), evaluated at every layer boundary, keeping
    only the residual stream. Backward: from the head down, each layer recomputed under mx.vjp.
    Peak memory is one layer's GDN tape (about 0.4 GB per 64-token sequence) instead of all 48.
    mx.checkpoint did not give this (measured 2026-11-14)."""
    import mlx.core as mx
    from mlx2.runtime.models.base import create_attention_mask, create_ssm_mask
    tm = host.lm.model
    keys = sorted(trainables)
    h = tm.embed_tokens(ids)
    fa, ssm = create_attention_mask(h, None), create_ssm_mask(h, None)
    hs = [h]
    mx.eval(h)
    for l, layer in enumerate(tm.layers):
        apply(l, {k: mx.stop_gradient(trainables[k][l]) for k in keys})
        h = layer(h, mask=(ssm if layer.is_linear else fa), cache=None)
        mx.eval(h)
        hs.append(h)
    host.ctl.reset()
    (loss,), (gh,) = mx.vjp(lambda x: head_loss(tm.norm(x)), [hs[-1]], [mx.array(1.0)])
    mx.eval(loss, gh)
    grads = {k: [None] * len(tm.layers) for k in keys}
    for l in reversed(range(len(tm.layers))):
        layer = tm.layers[l]

        def fl(x, *vals, l=l, layer=layer):
            apply(l, dict(zip(keys, vals)))
            return layer(x, mask=(ssm if layer.is_linear else fa), cache=None)
        _, cots = mx.vjp(fl, [hs[l], *[trainables[k][l] for k in keys]], [gh])
        host.ctl.reset()
        gh = cots[0]
        for i, k in enumerate(keys):
            grads[k][l] = cots[1 + i]
        mx.eval(gh, *cots[1:])
        hs[l + 1] = None
        mx.clear_cache()
    return loss, {k: mx.stack(v) for k, v in grads.items()}


def train_mask(Z, fixed, lam, steps, lr, grad_fn, batches_fn, log, tag, every=10, peak_cap_gb=None):
    """Carve a unit (Pierre's term, 2026-11-14: choosing neurons is carving, not training; no weight
    changes). Straight-through hard mask over the free units: m = fixed + free * ST(Z).

    grad_fn(m, micro) -> (mean loss of that micro-batch, d loss / d m) at the hard mask m.
    batches_fn(step) -> list of micro-batches (gradient accumulation). Chain rule to Z through the
    straight-through sigmoid: dZ = free * s(1 - s) * (mean_micro dm + lam / n_free)."""
    import mlx.core as mx
    import mlx.optimizers as optim
    opt = optim.Adam(learning_rate=lr)
    free = 1.0 - fixed
    nfree = max(1.0, float(free.sum()))
    params = {"Z": Z}
    t0 = time.time()
    for step in range(steps):
        Zc = params["Z"]
        m = fixed + free * (Zc > 0).astype(mx.float32)
        micros = batches_fn(step)
        gsum, msum = None, 0.0
        for i, micro in enumerate(micros):
            main, gm = grad_fn(m, micro)
            gsum = gm if gsum is None else gsum + gm
            mx.eval(gsum)
            msum += float(main) / len(micros)
            mx.clear_cache()
            if step == 0 and i == 0:
                log(f"  {tag}: first micro-batch peak {peak_gb()} GB")
                if peak_cap_gb is not None and peak_gb() > peak_cap_gb:
                    raise SystemExit(f"{tag}: MLX peak {peak_gb()} GB over the {peak_cap_gb} GB cap after one micro-batch")
        s_ = mx.sigmoid(Zc)
        gZ = free * s_ * (1 - s_) * (gsum / len(micros) + lam / nfree)
        params = opt.apply_gradients({"Z": gZ}, params)
        mx.eval(params, opt.state)
        mx.clear_cache()
        if step % every == 0 or step == steps - 1:
            kept = float((fixed + free * (params["Z"] > 0)).mean())
            log(f"  {tag} lam={lam} step {step} main={msum:.4f} kept={kept:.4f} "
                f"({(time.time() - t0) / (step + 1):.1f} s/step, peak {peak_gb()} GB)")
    return params["Z"]


def topk_mask(score, frac):
    import mlx.core as mx
    L, N = score.shape
    k = max(1, int(round(frac * L * N)))
    flat = score.reshape(-1)
    thr = mx.sort(flat)[-k]
    return (score >= thr).astype(mx.float32)


def random_same_counts(mask, seed):
    import mlx.core as mx
    rng = np.random.default_rng(seed)
    m = np.array(mask)
    R = np.zeros_like(m)
    for l in range(m.shape[0]):
        k = int(m[l].sum())
        R[l, rng.choice(m.shape[1], k, replace=False)] = 1.0
    return mx.array(R)


def save_npz(path, **arrays):
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, **{k: np.array(v) for k, v in arrays.items()})
    tmp.rename(path)


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.rename(path)
