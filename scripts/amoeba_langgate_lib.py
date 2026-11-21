#!/usr/bin/env python3
"""aMoEba language gate for the BAU plan on Flash-Next (CPU only; no model, no MLX): the pieces that
scripts/amoeba_fn_langgate.py runs inside its GPU window and that the CPU tests exercise.

Pierre 2026-11-20: gate off the ten non-English language boxes and hum.religion and keep every other tile as an
always-open core. This module holds

  script_counts      Unicode script runs of the letters of a text (Han, Kana, Hangul, Arabic, Devanagari, Cyrillic,
                     Latin, other) and a Vietnamese-letter cue;
  FastTextLID        a numpy reader of a fastText supervised .bin (the approved lid.176.bin), predicting exactly as
                     fastText's predict does (subword hashing, hierarchical or flat softmax). It is referenced by
                     path and never bundled: the model is CC BY-SA 3.0 (share-alike); the receipt records its path
                     and sha256. No fastText package is installed in the venv, and no new dependency is added;
  ReligionNB         a hashed word uni/bigram naive Bayes topic detector fitted on fit rows only (val rows set its
                     threshold; held rows are never read by it);
  GatePolicy         the deterministic gate: prefill decision from the user text (script first, then LID, a
                     "answer in <language>" request cue, the religion detector) and decode re-detection on the
                     trailing output window every N tokens with hold-open;
  gate_groups        the partition's gated tiles grouped by the gates that open them (OR rule) and the BAU core;
  memory accounting  bytes of gated routed-expert weights a request leaves closed (approximate and exact layout);
  decision rules     the pre-registered pass rules of the window's summary.

States: implemented; not qualified; not selected by any route; never observed-used in serving.
"""
from __future__ import annotations

import hashlib
import json
import math
import mmap
import re
import struct
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

LANG10 = ("cmn", "fra", "deu", "spa", "jpn", "rus", "arb", "hin", "swh", "vie")
RELIGION = "hum.religion"
GATES = LANG10 + (RELIGION,)                      # gate order of every stream [T, len(GATES)]
LID_LABEL = {"cmn": "zh", "jpn": "ja", "fra": "fr", "deu": "de", "spa": "es", "rus": "ru", "hin": "hi",
             "swh": "sw", "arb": "ar", "vie": "vi", "eng": "en"}
LABEL_LANG = {v: k for k, v in LID_LABEL.items()}
LATIN_LID = ("fra", "deu", "spa", "vie", "swh")   # Latin-script boxes: LID decides (vie also has a letter cue)
LANG_GROUPS = ("group:langs", "group:cjkv")

# ----------------------------------------------------------------------------- scripts

_RANGES = (                                       # (lo, hi, script); letters only (checked with unicodedata)
    (0x3040, 0x309F, "kana"), (0x30A0, 0x30FF, "kana"), (0x31F0, 0x31FF, "kana"), (0xFF66, 0xFF9F, "kana"),
    (0x3400, 0x4DBF, "han"), (0x4E00, 0x9FFF, "han"), (0xF900, 0xFAFF, "han"), (0x20000, 0x2FA1F, "han"),
    (0x1100, 0x11FF, "hangul"), (0x3130, 0x318F, "hangul"), (0xAC00, 0xD7AF, "hangul"),
    (0x0600, 0x06FF, "arabic"), (0x0750, 0x077F, "arabic"), (0x08A0, 0x08FF, "arabic"), (0xFB50, 0xFDFF, "arabic"),
    (0xFE70, 0xFEFF, "arabic"),
    (0x0900, 0x097F, "devanagari"), (0xA8E0, 0xA8FF, "devanagari"),
    (0x0400, 0x052F, "cyrillic"), (0x2DE0, 0x2DFF, "cyrillic"), (0xA640, 0xA69F, "cyrillic"),
    (0x0041, 0x005A, "latin"), (0x0061, 0x007A, "latin"), (0x00C0, 0x024F, "latin"), (0x1E00, 0x1EFF, "latin"),
)
SCRIPTS = ("latin", "han", "kana", "hangul", "arabic", "devanagari", "cyrillic", "greek", "other")
# Vietnamese-only Latin letters: o/u horn, d stroke, a breve and the precomposed tone letters of U+1EA0..U+1EF9
_VIET = set("ơưđăƠƯĐĂ") | {chr(c) for c in range(0x1EA0, 0x1EFA)}


@lru_cache(maxsize=65536)
def char_script(ch: str) -> str | None:
    """The script of one letter (None for a non-letter: digits, punctuation, CJK symbols, spaces). Combining marks
    count for the abugida and abjad scripts (Devanagari vowel signs, Arabic harakat), never for Latin."""
    cat = unicodedata.category(ch)
    if not cat.startswith(("L", "M")):
        return None
    c = ord(ch)
    if 0x2100 <= c <= 0x214F or 0x1D400 <= c <= 0x1D7FF:
        return None                                # letterlike and mathematical symbols (ℝ, ℒ, 𝐱): no script
    if 0xFF21 <= c <= 0xFF3A or 0xFF41 <= c <= 0xFF5A:
        return "latin"                             # fullwidth Latin
    if c == 0x3005:
        return "han"                               # the ideographic iteration mark 々
    if 0x0370 <= c <= 0x03FF or 0x1F00 <= c <= 0x1FFF:
        return "greek" if cat.startswith("L") else None   # maths symbols mostly; LID decides Greek prose
    for lo, hi, s in _RANGES:
        if lo <= c <= hi:
            return s if cat.startswith("L") or s in ("devanagari", "arabic") else None
    return "other" if cat.startswith("L") else None


def script_counts(text: str) -> dict:
    """Letter counts per script, plus "viet" (Vietnamese-only letters, a subset of latin) and "letters"."""
    out = dict.fromkeys(SCRIPTS, 0)
    viet = 0
    for ch in text:
        s = char_script(ch)
        if s is None:
            continue
        out[s] += 1
        if ch in _VIET:
            viet += 1
    out["viet"] = viet
    out["letters"] = sum(out[s] for s in SCRIPTS)
    return out


# ----------------------------------------------------------------------------- fastText (numpy reader)

FT_MAGIC = 793712314
_LOSS = {1: "hs", 2: "ns", 3: "softmax", 4: "ova"}
_WS = re.compile(r"[ \n\r\t\v\f\0]+")


def ft_hash(b: bytes) -> int:
    """fastText's Dictionary::hash: FNV-1a over the bytes, each byte sign-extended as int8 (bytes >= 0x80 xor in
    0xFFFFFFxx), 32-bit."""
    h = 2166136261
    for x in b:
        h ^= (x | 0xFFFFFF00) if x >= 0x80 else x
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def ft_subwords(word: bytes, minn: int, maxn: int) -> list[bytes]:
    """fastText's computeSubwords over a BOW/EOW-wrapped word: character n-grams (UTF-8 code points) of length
    minn..maxn, skipping the 1-grams at either end (the bare '<' or '>')."""
    out = []
    n_ = len(word)
    for i in range(n_):
        if (word[i] & 0xC0) == 0x80:
            continue
        j, n, ng = i, 1, bytearray()
        while j < n_ and n <= maxn:
            ng.append(word[j])
            j += 1
            while j < n_ and (word[j] & 0xC0) == 0x80:
                ng.append(word[j])
                j += 1
            if n >= minn and not (n == 1 and (i == 0 or j == n_)):
                out.append(bytes(ng))
            n += 1
    return out


class FastTextLID:
    """Predicts with a fastText supervised model (.bin, version 12, dense input and output matrices) as fastText's
    Python predict does: the text's whitespace tokens plus the end-of-line token, each in-vocabulary word as its
    id plus its subword buckets (an out-of-vocabulary word: its subword buckets only), the hidden vector is their
    mean input row; hs = the label's path probability in the Huffman tree built from the label counts (fastText's
    buildTree), softmax = a flat softmax. Probabilities use fastText's std_log (log(x + 1e-5)) so that they equal
    its printed values. Quantized models (.ftz) are refused (fail closed)."""

    def __init__(self, path):
        self.path = str(path)
        self._fh = open(path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        buf, o = self._mm, 0

        def take(fmt):
            nonlocal o
            v = struct.unpack_from(fmt, buf, o)
            o += struct.calcsize(fmt)
            return v
        magic, version = take("<ii")
        if magic != FT_MAGIC or version not in (11, 12):
            raise ValueError(f"{path}: not a fastText model (magic {magic}, version {version})")
        (self.dim, self.ws, self.epoch, self.min_count, self.neg, self.word_ngrams, loss, self.model, self.bucket,
         self.minn, self.maxn, self.lr_update) = take("<12i")
        take("<d")
        self.loss = _LOSS.get(loss, str(loss))
        size, self.nwords, self.nlabels = take("<3i")
        _ntokens, prune = take("<2q")
        self.word2id, self.labels, counts = {}, [], []
        for i in range(size):
            e = buf.find(b"\0", o)
            w = bytes(buf[o:e])
            o = e + 1
            cnt, typ = take("<qb")
            if typ == 0:
                self.word2id[w] = i
            else:
                self.labels.append(w.decode("utf-8"))
                counts.append(cnt)
        if prune > 0:
            raise ValueError(f"{path}: pruned dictionary (a quantized/pruned model) is not supported")
        (quant,) = take("<?")
        if quant:
            raise ValueError(f"{path}: quantized input matrix (.ftz) is not supported; use the .bin")
        m, n = take("<2q")
        self.input = np.frombuffer(buf, np.float32, m * n, o).reshape(m, n)
        o += 4 * m * n
        (qout,) = take("<?")
        if qout:
            raise ValueError(f"{path}: quantized output matrix is not supported")
        m2, n2 = take("<2q")
        self.output = np.frombuffer(buf, np.float32, m2 * n2, o).reshape(m2, n2)
        o += 4 * m2 * n2
        if n != self.dim or n2 != self.dim or m != self.nwords + self.bucket:
            raise ValueError(f"{path}: matrix shapes {m}x{n} / {m2}x{n2} do not match dim {self.dim}")
        self.labels = [x.removeprefix("__label__") for x in self.labels]
        self._eos = self.word2id.get(b"</s>")
        if self.loss == "hs":
            self._build_tree(np.asarray(counts, np.int64))
        elif self.loss != "softmax":
            raise ValueError(f"{path}: loss {self.loss} is not supported (hs or softmax)")
        self._ids = lru_cache(maxsize=200000)(self._word_ids)

    def close(self):
        self.input = self.output = None
        self._mm.close()
        self._fh.close()

    def _build_tree(self, counts):
        """fastText HierarchicalSoftmaxLoss::buildTree (labels are in descending count order) and each leaf's path:
        internal node indices (rows of the output matrix are node - osz) and whether each step goes right."""
        osz = len(counts)
        N = 2 * osz - 1
        cnt = np.full(N, 1e15)
        cnt[:osz] = counts
        par = np.full(N, -1, np.int64)
        binary = np.zeros(N, bool)
        leaf, node = osz - 1, osz
        for i in range(osz, N):
            mini = [0, 0]
            for j in range(2):
                if leaf >= 0 and cnt[leaf] < cnt[node]:
                    mini[j] = leaf
                    leaf -= 1
                else:
                    mini[j] = node
                    node += 1
            cnt[i] = cnt[mini[0]] + cnt[mini[1]]
            par[mini[0]] = par[mini[1]] = i
            binary[mini[1]] = True
        depth = 0
        paths = []
        for lf in range(osz):
            p, x = [], lf
            while par[x] != -1:
                p.append((par[x] - osz, binary[x]))
                x = par[x]
            paths.append(p)
            depth = max(depth, len(p))
        self._pnode = np.zeros((osz, depth), np.int64)
        self._pright = np.zeros((osz, depth), bool)
        self._pvalid = np.zeros((osz, depth), bool)
        for i, p in enumerate(paths):
            for j, (nd, r) in enumerate(p):
                self._pnode[i, j], self._pright[i, j], self._pvalid[i, j] = nd, r, True

    def _word_ids(self, w: bytes):
        wid = self.word2id.get(w)
        sub = [] if w == b"</s>" else [self.nwords + ft_hash(s) % self.bucket
                                       for s in ft_subwords(b"<" + w + b">", self.minn, self.maxn)]
        if self.maxn <= 0:
            sub = []
        return tuple(([wid] if wid is not None else []) + sub)

    def hidden(self, text: str):
        ids = []
        toks = [t for t in _WS.split(text) if t] + ["</s>"]
        for t in toks:
            ids.extend(self._ids(t.encode("utf-8")))
        if self.word_ngrams > 1:
            raise ValueError("word n-gram models are not supported")
        if not ids:
            return None
        return self.input[np.asarray(ids, np.int64)].astype(np.float32).mean(0)

    def probs(self, text: str) -> dict:
        """{label: probability} over every label (exact; fastText prints these up to its top-k cut)."""
        h = self.hidden(text.replace("\n", " "))
        if h is None:
            return {}
        if self.loss == "softmax":
            z = self.output @ h
            z = np.exp(z - z.max())
            p = z / z.sum()
            return {lb: float(v) for lb, v in zip(self.labels, p)}
        f = 1.0 / (1.0 + np.exp(-(self.output @ h).astype(np.float64)))
        fs = f[self._pnode]
        step = np.where(self._pright, np.log(fs + 1e-5), np.log(1.0 - fs + 1e-5))
        score = np.where(self._pvalid, step, 0.0).sum(1)
        return {lb: float(math.exp(s)) for lb, s in zip(self.labels, score)}

    def top(self, text: str, k: int = 3):
        p = self.probs(text)
        return sorted(p.items(), key=lambda x: -x[1])[:k]


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_lid(model_dir, verify=True):
    """(FastTextLID, receipt) from the approved directory (lid.176.bin + SOURCE.json); the sha256 must match the
    SOURCE.json record (fail closed). Referenced by path, never copied: CC BY-SA 3.0 share-alike."""
    d = Path(model_dir).expanduser()
    src = json.loads((d / "SOURCE.json").read_text())
    p = d / "lid.176.bin"
    want = src["files"]["lid.176.bin"]["sha256"]
    got = sha256_file(p) if verify else None
    if verify and got != want:
        raise SystemExit(f"{p}: sha256 {got} != SOURCE.json {want} (refusing)")
    rec = {"path": str(p), "sha256": got or want, "sha256_verified": bool(verify), "license": src.get("license"),
           "approved_by": src.get("approved_by"), "source_url": src["files"]["lid.176.bin"].get("url"),
           "bundled": False, "note": "referenced by path only; never copied into the repository or an export"}
    return FastTextLID(p), rec


# ----------------------------------------------------------------------------- religion detector

_WORD = re.compile(r"\w+", re.UNICODE)


def nb_features(text: str, buckets: int):
    """Hashed lowercase word unigrams and bigrams (blake2b, deterministic across processes)."""
    w = [x.lower() for x in _WORD.findall(text) if not x.isdigit()]
    feats = w + [a + " " + b for a, b in zip(w, w[1:])]
    out = np.empty(len(feats), np.int64)
    for i, f in enumerate(feats):
        out[i] = int.from_bytes(hashlib.blake2b(f.encode("utf-8"), digest_size=8).digest(), "little") % buckets
    return out


def word_window(text: str, words: int, start: int = 0) -> str:
    """`words` whitespace words of text from word `start` (the prompt-sized window the gate sees)."""
    return " ".join(text.split()[start:start + words])


class ReligionNB:
    """Multinomial naive Bayes, positive = hum.religion fit rows, negative = other boxes' / English fit rows.
    score(text) = the mean per-feature log ratio over the text's features seen at least min_count times in fit
    (unseen features contribute 0, so non-English text stays near 0), a length-invariant score; the threshold
    is set on val windows of the gate's prompt size for --rel-recall (no held row is ever read)."""

    def __init__(self, buckets=1 << 20, alpha=1.0, min_count=2):
        self.buckets, self.alpha, self.min_count = int(buckets), float(alpha), int(min_count)
        self.w = None
        self.threshold = None
        self.meta = {}

    def fit(self, pos, neg, max_words=400):
        """Each training document contributes its first max_words words (bounded fit time)."""
        cp = np.zeros(self.buckets, np.float64)
        cn = np.zeros(self.buckets, np.float64)
        for t in pos:
            cp += np.bincount(nb_features(word_window(t, max_words), self.buckets), minlength=self.buckets)
        for t in neg:
            cn += np.bincount(nb_features(word_window(t, max_words), self.buckets), minlength=self.buckets)
        lp = np.log((cp + self.alpha) / (cp.sum() + self.alpha * self.buckets))
        ln = np.log((cn + self.alpha) / (cn.sum() + self.alpha * self.buckets))
        w = (lp - ln).astype(np.float32)
        w[(cp + cn) < self.min_count] = 0.0
        self.w = w
        self.meta.update(fit_pos_docs=len(pos), fit_neg_docs=len(neg), buckets=self.buckets, alpha=self.alpha,
                         min_count=self.min_count, max_words=max_words)
        return self

    def score(self, text: str) -> float:
        f = nb_features(text, self.buckets)
        if f.size == 0:
            return float("-inf")
        return float(self.w[f].mean())

    def calibrate(self, pos_val, neg_val, recall=0.95, recall_decode=0.8):
        """threshold = the (1 - recall) quantile of positive val scores (ties open), used at prefill;
        threshold_decode = the same at recall_decode (stricter: a decode window is checked every few tokens, so a
        per-window false positive compounds over a request, and an on-topic request is caught at prefill)."""
        sp = np.array([self.score(t) for t in pos_val])
        sn = np.array([self.score(t) for t in neg_val])
        sp = np.where(np.isfinite(sp), sp, -1e30)     # an empty window is a miss, never dropped
        q = lambda r: float(np.quantile(sp, 1.0 - r, method="lower")) if sp.size else float("inf")  # noqa: E731
        self.threshold, self.threshold_decode = q(recall), max(q(recall), q(recall_decode))
        rate = lambda v, t: float((v >= t).mean()) if v.size else None                            # noqa: E731
        self.meta.update(val_pos=int(sp.size), val_neg=int(sn.size), target_recall=recall,
                         val_recall=rate(sp, self.threshold), val_fpr=rate(sn, self.threshold),
                         threshold=self.threshold, target_recall_decode=recall_decode,
                         threshold_decode=self.threshold_decode, val_recall_decode=rate(sp, self.threshold_decode),
                         val_fpr_decode=rate(sn, self.threshold_decode))
        return self.meta

    threshold_decode = None

    def opens(self, text: str, stage: str = "prefill") -> tuple[bool, float]:
        s = self.score(text)
        t = self.threshold_decode if stage == "decode" and self.threshold_decode is not None else self.threshold
        return bool(t is not None and s >= t), s


# ----------------------------------------------------------------------------- gate policy

_REQ_NAMES = {"fra": "french|français|francais", "deu": "german|deutsch", "spa": "spanish|español|espanol",
              "cmn": "chinese|mandarin|中文|汉语|普通话", "jpn": "japanese|日本語", "rus": "russian|русском|русский",
              "hin": "hindi|हिंदी|हिन्दी", "swh": "swahili|kiswahili", "arb": "arabic|العربية",
              "vie": "vietnamese|tiếng việt"}
_REQ = {L: re.compile(r"(?:\b(?:in|into|to|en|auf)\s+(?:the\s+)?(?:" + names + r")\b)", re.IGNORECASE)
        for L, names in _REQ_NAMES.items()}


@dataclass
class GateConfig:
    """Every threshold of the gate (recorded in the receipt). Thresholds and why:
    script_min_chars 4      a non-Latin script opens its language(s) from 4 letters (a short quote opens; one stray
                            character does not); opening costs memory only, never quality;
    han_only_kana_share     Han text whose Kana share of Han+Kana is below 0.05 is "Han-only" and opens cmn and jpn
                            (next-experiments 3: Han-only text cannot tell the two apart; jpn also repairs cmn);
    lid_open_p 0.5          a Latin-script language opens when fastText gives it >= 0.5 on a segment of at least
                            lid_min_letters (12) letters, or on the whole text;
    uncertain_rule          gated_mass (default): a Latin segment of at least uncertain_min_letters (40) letters is
                            uncertain when the LID mass on the gated Latin languages (fr, de, es, vi, sw) is at least
                            uncertain_mass (0.2) and none of them reached lid_open_p: LID doubts between English and
                            a gated language. top1: uncertain when the top-1 probability is below uncertain_p (0.4)
                            (fires on code and identifiers, which no gated box serves; tuned on val rows 2026-11-20);
    unknown_open_p 0.5      a segment of >= uncertain_min_letters letters confidently in a language outside English +
                            the ten (it, pt, ...) is treated like an uncertain one;
    on_uncertain open_all   fail open: an uncertain segment opens every language (quality-safe; memory-costly);
    viet_min 3              three Vietnamese-only letters open vie (a script-level cue beside LID);
    request_cue on          "in/into <language>" in the user text opens that language at prefill;
    religion couples hin    hum.religion opening also opens hin (repair: hin recovers 49% of religion's gap);
                            the detector's threshold is set for 0.95 recall on val windows at prefill and 0.8 at
                            decode (ReligionNB.calibrate);
    check_every 4 / window 64 tokens   decode re-detection on the trailing output window (tuned on val rows
                            2026-11-20: 16 -> 13.7% of the mid-answer switch's language rows missed, 8 -> 7.7%,
                            4 -> 5.9%; a check is about 1 ms of CPU, under 0.3 ms per token); script_every 1:
                            the script rules alone between checks (val: 5.9% -> 5.55%; check_every 2 gave 5.14%,
                            not taken: the latency is LID needing a dozen letters of the new language);
    hold_tokens -1          an opened gate stays open for the rest of the request (hold-open); > 0 closes a gate
                            after that many tokens without a re-detection."""
    script_min_chars: int = 4
    han_only_kana_share: float = 0.05
    lid_open_p: float = 0.5
    lid_min_letters: int = 12
    uncertain_p: float = 0.4
    uncertain_rule: str = "gated_mass"      # gated_mass | top1
    uncertain_mass: float = 0.2
    uncertain_min_letters: int = 40
    unknown_open_p: float = 0.5
    on_uncertain: str = "open_all"          # open_all | none
    viet_min: int = 3
    request_cue: bool = True
    religion: bool = True
    religion_couples: tuple = ("hin",)
    check_every: int = 4
    script_every: int = 1                   # the script rules alone (microseconds) run every script_every tokens
    window_tokens: int = 64
    hold_tokens: int = -1
    segment_chars: int = 300
    lid_letter_ratio: float = 0.5           # a segment with fewer letters than this share of its non-space chars
                                            # (code, formulas, tables) is skipped by LID, not called uncertain

    def as_dict(self):
        d = asdict(self)
        d["religion_couples"] = list(self.religion_couples)
        return d


def segments(text: str, max_chars: int):
    """Lines, long lines cut into max_chars pieces at a space where possible."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        while len(line) > max_chars:
            cut = line.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            out.append(line[:cut])
            line = line[cut:].strip()
        if line:
            out.append(line)
    return out


class GatePolicy:
    """detect(text) -> (open gates, reasons); deterministic for a given LID, religion detector and config."""

    def __init__(self, cfg: GateConfig, lid=None, religion=None):
        self.cfg, self.lid, self.rel = cfg, lid, religion

    def _lid_segment(self, seg, add, reasons, flags):
        c = self.cfg
        sc = script_counts(seg)
        nonspace = sum(1 for ch in seg if not ch.isspace())
        if sc["letters"] < c.lid_min_letters or sc["letters"] < c.lid_letter_ratio * max(1, nonspace):
            return
        if sc["greek"] >= 0.5 * sc["letters"]:
            # Greek prose (a segment of >= lid_min_letters letters, most of them Greek: not maths symbols in Latin
            # text) is a language outside English + the ten and no script rule covers it: fail open, like the
            # other unlisted scripts
            flags.append(f"unknown_script:greek={sc['greek']}")
            return
        if self.lid is None:
            return
        if sc["latin"] < 0.5 * sc["letters"] and sc["han"] < c.script_min_chars:
            return                                     # non-Latin segments: the script rules decide (Han: LID too)
        probs = self.lid.probs(seg)
        if not probs:
            return
        top = sorted(probs.items(), key=lambda x: -x[1])[:3]
        lab, p = top[0]
        hit = False
        for lb, q in top:
            L = LABEL_LANG.get(lb)
            if L in LANG10 and q >= c.lid_open_p:
                add(L, f"lid:{lb}:{q:.2f}")
                hit = True
                if L in ("cmn", "jpn") and sc["han"] >= c.script_min_chars:
                    add("jpn" if L == "cmn" else "cmn", f"lid_han:{lb}:{q:.2f}")   # Han: both, as the Han rule
        if sc["letters"] < c.uncertain_min_letters:
            return
        if c.uncertain_rule == "top1":
            if p < c.uncertain_p:
                flags.append(f"uncertain:{lab}:{p:.2f}")
                return
        else:
            mass = sum(probs.get(LID_LABEL[L], 0.0) for L in LATIN_LID)
            if not hit and mass >= c.uncertain_mass:
                flags.append(f"uncertain:gated_mass:{mass:.2f}")
                return
        if lab not in LABEL_LANG and p >= c.unknown_open_p:
            flags.append(f"unknown_language:{lab}:{p:.2f}")

    def detect(self, text: str, request_text: str | None = None, stage: str = "prefill", full: bool = True):
        c = self.cfg
        opened, reasons = set(), {}

        def add(g, why):
            opened.add(g)
            reasons.setdefault(g, []).append(why)
        sc = script_counts(text)
        hk = sc["han"] + sc["kana"]
        if sc["kana"] >= c.script_min_chars:
            add("jpn", f"script:kana:{sc['kana']}")
        elif sc["kana"] >= 1 and hk >= c.script_min_chars:
            # any kana beside Han is Japanese: without this a short prompt (1-3 kana) is neither "kana" nor
            # "Han-only" and opened nothing
            add("jpn", f"script:kana_han:{sc['kana']}+{sc['han']}")
        if sc["han"] >= c.script_min_chars and sc["kana"] < c.han_only_kana_share * hk:
            add("cmn", f"script:han_only:{sc['han']}")
            add("jpn", f"script:han_only:{sc['han']}")
        for s, L in (("arabic", "arb"), ("devanagari", "hin"), ("cyrillic", "rus")):
            if sc[s] >= c.script_min_chars:
                add(L, f"script:{s}:{sc[s]}")
        if sc["viet"] >= c.viet_min:
            add("vie", f"script:viet_letters:{sc['viet']}")
        flags = []
        if sc["hangul"] >= c.script_min_chars or sc["other"] >= c.script_min_chars:
            flags.append(f"unknown_script:hangul={sc['hangul']},other={sc['other']}")
        if not full:                                   # script-only check between full checks
            return opened, {"reasons": reasons, "flags": [], "script": sc}
        segs = segments(text, c.segment_chars)         # without an LID the segment pass still flags Greek prose
        whole = " ".join(segs)
        for seg in segs + ([whole] if len(segs) > 1 else []):
            self._lid_segment(seg, add, reasons, flags)
        if c.request_cue:
            for L, rx in _REQ.items():
                if rx.search(request_text if request_text is not None else text):
                    add(L, "request_cue")
        if flags and c.on_uncertain == "open_all":
            for L in LANG10:
                add(L, "fail_open:" + flags[0])
        rel_score = None
        if c.religion and self.rel is not None:
            ok, rel_score = self.rel.opens(text, stage)
            if ok:
                add(RELIGION, f"religion_nb:{rel_score:.3f}")
        if RELIGION in opened:
            for g in c.religion_couples:
                add(g, "coupled:hum.religion")
        return opened, {"reasons": reasons, "flags": flags, "script": sc,
                        **({"religion_score": rel_score} if rel_score is not None else {})}


def couple(gates: set, couples=("hin",)):
    """The oracle's coupling: religion open -> its coupled gates open (the policy does the same)."""
    out = set(gates)
    if RELIGION in out:
        out |= set(couples)
    return out


def gate_vec(gs):
    return np.array([g in gs for g in GATES], bool)


def simulate(tokens, prompt_len, decode, policy: GatePolicy, request_text=None):
    """The gate stream of one request, teacher forced over its tokens: [T, len(GATES)] bool, where row t is the open
    set of the forward whose input is token t. Prefill: detect(decode(tokens[:P])) opens rows 0..; decode: after
    every check_every generated tokens (at input position t, token t already sampled), detect on the trailing
    window decode(tokens[max(P, t + 1 - window) : t + 1]); newly opened gates are open from row t on. Hold-open:
    a gate stays open (hold_tokens > 0: it closes once hold_tokens rows passed without a re-detection; a prefill
    detection counts as a detection at row P - 1). Returns (stream, log)."""
    c = policy.cfg
    T, P = len(tokens), int(prompt_len)
    S = np.zeros((T, len(GATES)), bool)
    gi = {g: i for i, g in enumerate(GATES)}
    last = np.full(len(GATES), -10**9)
    g0, info = policy.detect(decode(tokens[:P]), request_text)
    cur = np.zeros(len(GATES), bool)
    for g in g0:
        cur[gi[g]] = True
        last[gi[g]] = P - 1
    log = {"prefill": {"open": sorted(g0), **info}, "events": []}
    S[:P] = cur
    for t in range(P, T):
        k = t - P
        full = k % c.check_every == c.check_every - 1
        if full or (c.script_every > 0 and k % c.script_every == c.script_every - 1):
            gs, inf = policy.detect(decode(tokens[max(P, t + 1 - c.window_tokens): t + 1]), stage="decode", full=full)
            new = [g for g in gs if not cur[gi[g]]]
            for g in gs:
                last[gi[g]] = t
                cur[gi[g]] = True
            if new:
                log["events"].append({"t": t, "open": sorted(new), "reasons": {g: inf["reasons"][g] for g in new}})
        if c.hold_tokens > 0:
            close = cur & (t - last > c.hold_tokens)
            if close.any():
                log["events"].append({"t": t, "close": [GATES[i] for i in np.flatnonzero(close)]})
                cur = cur & ~close
        S[t] = cur
    return S, log


def oracle_stream(labels, couples=("hin",)):
    """labels: per token the set of gates its text needs (a segment label). Row t (the forward whose input is token
    t, which predicts token t + 1) needs the label of token t + 1 (the last row: its own)."""
    T = len(labels)
    S = np.zeros((T, len(GATES)), bool)
    for t in range(T):
        S[t] = gate_vec(couple(labels[min(t + 1, T - 1)], couples) if couples else labels[min(t + 1, T - 1)])
    return S


def gate_metrics(auto, oracle, prompt_len, scored_from=2):
    """Per request: prefill exact / superset / missed against the oracle at the prompt rows, and over the scored
    rows (r = target - 1 for targets >= scored_from): needed (gate, row) pairs, missed opens, extra opens, rows
    served fully gated (nothing gated open)."""
    T = auto.shape[0]
    P = int(prompt_len)
    rows = np.arange(scored_from - 1, T - 1)
    need = oracle[rows]
    got = auto[rows]
    miss = need & ~got
    extra = got & ~need
    pa, po = auto[:P].any(0), oracle[:P].any(0)
    dec = rows >= P - 1
    return {"prefill_exact": bool(np.array_equal(pa, po)), "prefill_superset": bool((po <= pa).all()),
            "prefill_missed": [GATES[i] for i in np.flatnonzero(po & ~pa)],
            "prefill_extra": [GATES[i] for i in np.flatnonzero(pa & ~po)],
            "needed": int(need.sum()), "missed": int(miss.sum()), "extra": int(extra.sum()),
            "needed_decode": int(need[dec].sum()), "missed_decode": int(miss[dec].sum()),
            "missed_rows": [int(r) for r in rows[miss.any(1)]],
            "missed_by_gate": {GATES[i]: int(miss[:, i].sum()) for i in range(len(GATES)) if miss[:, i].any()},
            "needed_by_gate": {GATES[i]: int(need[:, i].sum()) for i in range(len(GATES)) if need[:, i].any()},
            "rows": int(len(rows)), "rows_fully_gated": int((~got.any(1)).sum()),
            "row_exact": int((got == need).all(1).sum())}


# ----------------------------------------------------------------------------- tiles

def tile_gates(key, mixed_pairs="language"):
    """The gates that open tile `key` under BAU (OR rule), or None for the always-open core. mixed_pairs: a pair of
    a language and a topic (not religion) opens with its language ("language", default) or is core ("core")."""
    if key.startswith("box:"):
        b = key[4:]
        return frozenset([b]) if b in GATES else None
    if key == "group:langs":
        return frozenset(LANG10)
    if key == "group:cjkv":
        return frozenset(("cmn", "jpn", "vie"))
    if key.startswith("pair:"):
        mem = key[5:].split("__")
        langs = [m for m in mem if m in LANG10]
        rel = [m for m in mem if m == RELIGION]
        other = [m for m in mem if m not in GATES]
        if not langs:
            return None                                  # topic pairs (religion + topic included): core
        if other:
            return frozenset(langs) if mixed_pairs == "language" else None
        return frozenset(langs + rel)                    # language-language and language-religion pairs
    return None


def gate_groups(keys, mixed_pairs="language", pairs="or"):
    """(core_keys, groups {frozenset(gates): [tile keys]}, kinds {tile: kind}). pairs="closed" keeps every gated
    pair tile closed whatever the gates (the "without pair tiles" variant): those go to groups[frozenset()]."""
    core, groups, kinds = [], {}, {}
    for k in keys:
        if k == "layers":
            continue
        g = tile_gates(k, mixed_pairs)
        if g is None:
            core.append(k)
            kinds[k] = "core"
            continue
        if k.startswith("pair:"):
            mem = k[5:].split("__")
            kinds[k] = "pair_mixed" if any(m not in GATES for m in mem) else (
                "pair_lang_religion" if RELIGION in mem else "pair_lang")
            if pairs == "closed":
                g = frozenset()
        else:
            kinds[k] = "lang_group" if k.startswith("group:") else ("box_religion" if k == f"box:{RELIGION}"
                                                                     else "box_lang")
        groups.setdefault(g, []).append(k)
    return core, groups, kinds


def group_streams(stream, groups):
    """Per tile group, its [T] open bits: the OR of its gates' columns (frozenset() = never open)."""
    gi = {g: i for i, g in enumerate(GATES)}
    out = {}
    for g in groups:
        out[g] = stream[:, [gi[x] for x in sorted(g)]].any(1) if g else np.zeros(stream.shape[0], bool)
    return out


# ----------------------------------------------------------------------------- memory

def unit_bytes(d=2560, bits=4, group_size=64):
    """Bytes of one routed-expert neuron unit (gate row + up row + down column, affine quantized: packed weights
    plus a scale and a bias of 2 bytes per group), and of its gate and up rows alone."""
    row = d * bits // 8 + (d // group_size) * 4
    return 3 * row, 2 * row


def gated_bytes(unit_idx, units_per_layer, closed_keys, d=2560, bits=4, group_size=64):
    """Bytes of the closed gated tiles: approximate (units x unit bytes) and exact-layout pageable (gate/up rows of
    every closed unit + down groups whose group_size neurons along F are all closed; down_proj is quantized along F
    so a down column shares scales with its group). unit_idx: {tile: flat unit indices [layer * W + u]}."""
    ub, gu = unit_bytes(d, bits, group_size)
    if not closed_keys:
        return {"units": 0, "approx_bytes": 0, "exact_bytes": 0, "full_down_groups": 0}
    idx = np.concatenate([np.asarray(unit_idx[k], np.int64) for k in closed_keys])
    n = int(idx.size)
    if units_per_layer % group_size:
        full = 0
    else:
        cells = np.bincount(idx // group_size)
        full = int((cells == group_size).sum())
    down_group = group_size * (d * bits // 8 + (d // group_size) * 4)   # == group_size units' down columns
    return {"units": n, "approx_bytes": n * ub, "exact_bytes": n * gu + full * down_group,
            "full_down_groups": full}


# ----------------------------------------------------------------------------- decision rules

RULES = {
    "target": 0.05,
    "known_served_miss": {"arb": 0.061, "cmn": 0.084, "jpn": 0.053, RELIGION: 0.098},
    "known_served_miss_source": "battery core_plus net dCE (fn-catboxes-47214c256/battery/battery.jsonl); the "
                                "battery's served composition closes every other box, BAU keeps them open",
    "missed_open_rate_max": 0.05,
    "missed_open_rate_scope": "pooled over language, religion and mixed sets, and separately on the decode rows "
                              "of each mixed set (the sets whose opens must come from decode re-detection)",
    "missed_open_rate_why": "a closed language box costs +0.1..+1.5 nats per token (battery own_without, median "
                            "about 0.5): 5% missed tokens x 0.5 nats = 0.025 nats, half the +0.05 bar",
    "mixed_gate_cost_max": 0.02,
    "sanity_open_vs_full_max": 1e-3,
    "pass": "INVALID unless |open - full| <= sanity on every scored set; then pass = (1) english, gsm8k, code and "
            "every topic set: auto - full <= target; (2) each language set and religion: auto - full <= "
            "max(target, known_served_miss); (3) each mixed set: auto - oracle <= mixed_gate_cost_max; (4) pooled "
            "missed-open rate (missed / needed (gate, row) pairs over language, religion and mixed sets) <= "
            "missed_open_rate_max, and the same on each mixed set's decode rows; (5) output-language correct count auto >= full. Point estimates decide (the "
            "battery's convention); paired SE and the group-bootstrap 95% CI are reported beside each, and "
            "ci_upper <= bar is reported as confident_pass. Variants (hin coupling, pair tiles, Han opening) are "
            "reported, never gating.",
    "label": "unqualified until Pierre accepts (AGENTS.md: qualification is a statement of confidence)",
}


def decide(rows, gate_pool, gen, kinds, rules=RULES, gate_sets=None):
    """rows: {(arm, set): {"mean", "ci95"}} paired dCE vs full (mixed: also ("auto-oracle", set)); kinds: {set:
    "english"|"gsm8k"|"code"|"topic"|"lang"|"religion"|"mixed"}; gate_pool: {"needed", "missed"}; gen: {"full":
    correct, "auto": correct} or None. Returns the verdict dict (pure; CPU-tested)."""
    T = rules["target"]
    checks, missing = [], []
    for s, kd in sorted(kinds.items()):
        op = rows.get(("open", s))
        if op is None:
            missing.append(f"open:{s}")
        elif not abs(op["mean"]) <= rules["sanity_open_vs_full_max"]:       # NaN fails the sanity gate too
            return {"verdict": "INVALID", "why": f"open arm differs from full on {s} by {op['mean']:+.4f}",
                    "checks": [], "label": rules["label"]}
    for s, kd in sorted(kinds.items()):
        if kd == "mixed":
            r = rows.get(("auto-oracle", s))
            bar, what = rules["mixed_gate_cost_max"], "auto - oracle"
        else:
            r = rows.get(("auto", s))
            bar = max(T, rules["known_served_miss"].get(s, T)) if kd in ("lang", "religion") else T
            what = "auto - full"
        if r is None:
            missing.append(f"auto:{s}")
            continue
        checks.append({"set": s, "kind": kd, "test": what, "mean": r["mean"], "ci95": r.get("ci95"), "bar": bar,
                       "pass": bool(r["mean"] <= bar),
                       "confident_pass": bool(r.get("ci95") and r["ci95"][1] <= bar)})
    rate = gate_pool["missed"] / gate_pool["needed"] if gate_pool.get("needed") else None
    if rate is None:
        missing.append("gate pool")
    else:
        checks.append({"set": "*", "kind": "gate", "test": "pooled missed-open rate", "mean": rate,
                       "bar": rules["missed_open_rate_max"], "pass": bool(rate <= rules["missed_open_rate_max"])})
    for s, kd in sorted(kinds.items()):
        if kd != "mixed":
            continue
        g = (gate_sets or {}).get(s)
        if not g or not g.get("needed_decode"):
            missing.append(f"gate decode:{s}")
            continue
        r_ = g["missed_decode"] / g["needed_decode"]
        checks.append({"set": s, "kind": "gate", "test": "decode missed-open rate", "mean": r_,
                       "bar": rules["missed_open_rate_max"], "pass": bool(r_ <= rules["missed_open_rate_max"])})
    if gen is None:
        missing.append("generations")
    else:
        checks.append({"set": "*", "kind": "generation", "test": "output-language correct auto >= full",
                       "mean": gen["auto"], "bar": gen["full"], "pass": bool(gen["auto"] >= gen["full"])})
    ok = all(c["pass"] for c in checks)
    sanity_missing = any(m.startswith("open:") for m in missing)
    # without the open == full sanity on every set nothing is decided (a failed check is then provisional)
    verdict = ("incomplete" if sanity_missing else "pass" if ok and not missing else "fail" if not ok
               else "incomplete")
    return {"verdict": verdict, "checks": checks, "missing": missing, "label": rules["label"],
            "failed": [f"{c['set']}:{c['test']}" for c in checks if not c["pass"]],
            **({"provisional": True} if sanity_missing else {})}
