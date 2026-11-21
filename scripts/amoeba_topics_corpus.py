#!/usr/bin/env python3
"""Build the amoeba topic probe corpus from local sources (DESIGN: docs/research/amoeba-topics/DESIGN.md).

Each item: {"id", "topics": [primary, also...], "text", "source", "path_key", "group", "split",
"license", "method", "sha256"}. Texts are ~400-1200 characters, model-independent; the capture
tokenises with the host tokenizer and keeps the first 64 tokens (position 0 excluded from statistics).

Split: fit/held by a hash of `group` (a source file, a bundle, a manuscript family), so a file never
feeds both splits. Quota per primary topic: --per-split (default 24) per split.

Outputs:
  <out>/corpus.jsonl          the texts (third-party text: keep under research-artifacts, never in git)
  <manifest>                  per-topic counts, sources, licences, methods, text sha256 (in git)
Refuses a non-empty <out>. Stdlib + pyarrow (parquet) only; no model is loaded.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import plistlib
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()
TAXO = ROOT / "docs/research/amoeba-topics/taxonomy.v1.json"
HF = HOME / ".cache/huggingface/hub"
MIN_CHARS, WIN_CHARS, MAX_CHARS = 400, 700, 1200
MAX_FILE = 2_000_000


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()


def read_text(p: Path) -> str | None:
    try:
        if p.stat().st_size > MAX_FILE or p.stat().st_size < MIN_CHARS:
            return None
        b = p.read_bytes()
    except OSError:
        return None
    if b"\x00" in b[:4096]:
        return None
    t = b.decode("utf-8", "replace")
    if t.count("\ufffd") > 5:
        return None
    return t


LICENSE_BOILER = re.compile(r"Permission is hereby granted|Licensed under the Apache License|GNU (Lesser )?General Public"
                            r"|THE SOFTWARE IS PROVIDED|Redistribution and use in source and binary", re.I)


def window(text: str, rng: random.Random, allow_boiler=False) -> str | None:
    """A window of whole lines, ~WIN_CHARS long, from a seeded start in the first 80% of the file."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return None
    for _ in range(4):
        start = rng.randrange(max(1, int(len(lines) * 0.8)))
        out, n = [], 0
        for ln in lines[start:]:
            out.append(ln)
            n += len(ln)
            if n >= WIN_CHARS:
                break
        w = "".join(out)[:MAX_CHARS]
        if len(w.strip()) < MIN_CHARS:
            continue
        if not allow_boiler and LICENSE_BOILER.search(w):
            continue
        if len(set(w.split())) < 12:            # repeated-string inputs mislead models (lab lesson)
            continue
        return w
    return None


# ----------------------------------------------------------------------------------- code by file type

EXT = {
    ".pyx": "prog.python.cython", ".c": "prog.c", ".cc": "prog.cpp", ".cpp": "prog.cpp", ".cxx": "prog.cpp",
    ".hpp": "prog.cpp", ".hh": "prog.cpp", ".m": "prog.objc", ".mm": "prog.objc", ".swift": "prog.swift",
    ".rs": "prog.rust", ".go": "prog.go", ".java": "prog.java", ".kt": "prog.kotlin", ".scala": "prog.scala",
    ".cs": "prog.csharp", ".fs": "prog.fsharp", ".js": "prog.javascript", ".mjs": "prog.javascript",
    ".ts": "prog.typescript", ".tsx": "prog.typescript", ".rb": "prog.ruby", ".pl": "prog.perl",
    ".pm": "prog.perl", ".pod": "prog.perl.pod", ".php": "prog.php", ".lua": "prog.lua", ".r": "prog.r",
    ".R": "prog.r", ".jl": "prog.julia", ".f90": "prog.fortran", ".f": "prog.fortran", ".f95": "prog.fortran",
    ".hs": "prog.haskell", ".ml": "prog.ocaml", ".lisp": "prog.lisp.common_lisp", ".el": "prog.lisp.emacs_lisp",
    ".scm": "prog.lisp.scheme", ".rkt": "prog.lisp.scheme", ".clj": "prog.lisp.clojure", ".erl": "prog.erlang",
    ".ex": "prog.elixir", ".exs": "prog.elixir", ".dart": "prog.dart", ".zig": "prog.zig", ".nim": "prog.nim",
    ".tcl": "prog.tcl", ".lean": "prog.lean", ".metal": "prog.gpu.metal", ".cu": "prog.gpu.cuda",
    ".cuh": "prog.gpu.cuda", ".glsl": "prog.gpu.glsl", ".vert": "prog.gpu.glsl", ".frag": "prog.gpu.glsl",
    ".hlsl": "prog.gpu.hlsl", ".wgsl": "prog.gpu.wgsl", ".fish": "prog.shell.fish", ".ps1": "prog.shell.powershell",
    ".psm1": "prog.shell.powershell", ".bat": "prog.shell.batch", ".cmd": "prog.shell.batch", ".sql": "prog.sql",
    ".html": "prog.web.html", ".htm": "prog.web.html", ".css": "prog.web.css", ".scss": "prog.web.scss",
    ".sass": "prog.web.scss", ".jinja": "prog.web.templates", ".j2": "prog.web.templates", ".erb": "prog.web.templates",
    ".hbs": "prog.web.templates", ".md": "prog.markup.markdown", ".rst": "prog.markup.rst", ".tex": "prog.markup.latex",
    ".adoc": "prog.markup.asciidoc", ".json": "prog.data.json", ".yaml": "prog.data.yaml", ".yml": "prog.data.yaml",
    ".toml": "prog.data.toml", ".xml": "prog.data.xml", ".plist": "prog.data.xml", ".csv": "prog.data.csv",
    ".tsv": "prog.data.csv", ".ini": "prog.data.ini", ".cfg": "prog.data.ini", ".conf": "prog.data.ini",
    ".proto": "prog.data.protobuf", ".cmake": "prog.build.cmake", ".mk": "prog.build.make", ".bzl": "prog.build.bazel",
    ".nix": "prog.build.nix", ".tf": "prog.build.hcl", ".hcl": "prog.build.hcl", ".sh": "prog.shell",
    ".bash": "prog.shell.bash", ".zsh": "prog.shell.zsh", ".s": "prog.assembly", ".S": "prog.assembly",
    ".asm": "prog.assembly", ".h": "prog.c", ".py": "prog.python", ".pyi": "prog.python.typing",
}
NAMES = {"Makefile": "prog.build.make", "GNUmakefile": "prog.build.make", "CMakeLists.txt": "prog.build.cmake",
         "Dockerfile": "prog.build.dockerfile", "BUILD": "prog.build.bazel", "BUILD.bazel": "prog.build.bazel",
         "WORKSPACE": "prog.build.bazel", "LICENSE": "swe.licensing", "LICENSE.txt": "swe.licensing",
         "LICENSE.md": "swe.licensing", "COPYING": "swe.licensing", "CHANGELOG.md": "swe.repo_management",
         "CHANGES.md": "swe.repo_management", "HISTORY.md": "swe.repo_management", "CONTRIBUTING.md": "swe.repo_management",
         "SECURITY.md": "swe.security", "pyproject.toml": "swe.packaging", "setup.cfg": "swe.packaging",
         "Cargo.toml": "swe.packaging", "go.mod": "swe.packaging", "package.json": "swe.packaging"}
ALSO = {"swe.licensing": [], "swe.repo_management": ["prog.markup.markdown"], "swe.security": ["prog.markup.markdown"],
        "swe.packaging": []}
PY_PKG = {
    "numpy": "prog.python.scientific", "pandas": "prog.python.scientific", "networkx": "cs.algorithms",
    "sklearn": "ml.foundations", "torch": "ml.frameworks.pytorch", "mlx": "ml.frameworks.mlx",
    "mlx_lm": "ml.frameworks.mlx", "jax": "ml.frameworks.jax", "transformers": "ml.frameworks.huggingface",
    "fastapi": "prog.python.web", "starlette": "prog.python.web", "django": "prog.python.web",
    "flask": "prog.python.web", "aiohttp": "prog.python.web", "uvicorn": "prog.python.web",
    "vllm": "ml.frameworks.vllm", "sglang": "ml.frameworks.vllm",
}
SYMPY = [("sympy/physics/quantum", "sci.physics.quantum"), ("sympy/physics/mechanics", "sci.physics.classical_mechanics"),
         ("sympy/physics/optics", "sci.physics.optics"), ("sympy/physics/units", "sci.physics.units"),
         ("sympy/physics/hep", "sci.physics.particle"), ("sympy/physics/continuum_mechanics", "eng.civil"),
         ("sympy/integrals", "math.analysis.calculus"), ("sympy/series", "math.analysis.calculus"),
         ("sympy/calculus", "math.analysis.calculus"), ("sympy/geometry", "math.geometry.euclidean"),
         ("sympy/solvers/ode", "math.analysis.ode"), ("sympy/matrices", "math.algebra.linear_algebra"),
         ("sympy/stats", "math.statistics"), ("sympy/polys", "math.algebra.ring_theory"),
         ("sympy/ntheory", "math.number_theory"), ("sympy/logic", "math.logic"),
         ("sympy/combinatorics", "math.algebra.group_theory"), ("sympy/categories", "math.algebra.category_theory"),
         ("scipy/stats", "math.statistics"), ("scipy/optimize", "math.optimization"),
         ("scipy/linalg", "math.algebra.linear_algebra"), ("scipy/signal", "eng.signal_processing"),
         ("scipy/integrate", "math.numerical_analysis"), ("scipy/interpolate", "math.numerical_analysis"),
         ("scipy/spatial", "math.geometry.metric_geometry"), ("scipy/special", "math.analysis.real_complex"),
         ("scipy/fft", "math.analysis.harmonic")]
SHEBANG = re.compile(rb"^#!\s*(?:/usr/bin/env\s+)?\S*?/?(sh|bash|zsh|ksh|dash|perl|python3?|ruby|fish|tclsh|lua)\b")
SHE_TOPIC = {"sh": "prog.shell.sh", "dash": "prog.shell.sh", "ksh": "prog.shell.sh", "bash": "prog.shell.bash",
             "zsh": "prog.shell.zsh", "perl": "prog.perl", "python": "prog.python", "python3": "prog.python",
             "ruby": "prog.ruby", "fish": "prog.shell.fish", "tclsh": "prog.tcl", "lua": "prog.lua"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist", ".tox", "target", "output",
             "Photos", "Library"}
THIN_CODE = {"prog.kotlin", "prog.scala", "prog.haskell", "prog.swift", "prog.objc", "prog.fortran", "prog.csharp", "prog.php", "prog.r", "prog.tcl", "prog.lisp.common_lisp",
             "prog.lisp.emacs_lisp", "prog.lisp.scheme", "swe.testing"}
STDLIB = re.compile(r"/lib/python3\.\d+/(?!site-packages)")


def py_topic(p: Path, text: str):
    s = str(p)
    if "@triton.jit" in text:
        return ["prog.gpu.triton", "prog.python"]
    for frag, t in SYMPY:
        if f"/{frag}/" in s:
            return [t, "prog.python.scientific"]
    name = p.name
    if p.suffix == ".pyi":
        return ["prog.python.typing"]
    if name.startswith("test_") or name.endswith("_test.py") or "/tests/" in s:
        return ["prog.python.testing", "swe.testing"]
    if name in ("setup.py",):
        return ["prog.python.packaging", "swe.packaging"]
    m = re.search(r"/site-packages/([A-Za-z0-9_]+)/", s) or re.search(r"/archive-v0/[^/]+/([A-Za-z0-9_]+)/", s)
    if m and m.group(1) in PY_PKG:
        t = PY_PKG[m.group(1)]
        return [t] + (["prog.python.ml_frameworks"] if t.startswith("ml.frameworks") else []) + ["prog.python"] * (t[:4] != "prog")
    if STDLIB.search(s):
        return ["prog.python.stdlib"]
    return ["prog.python"]


ASM_ARM = re.compile(r"\b(ldr|str|stp|ldp|adrp|x[0-9]{1,2}|w[0-9]{1,2})\b")
ASM_X86 = re.compile(r"%(r|e)[a-z]{2}\b|\bmovq?\b|\b(rax|rbx|rsp|eax)\b")


def zsh_completion(p: Path) -> bool:
    return p.name.startswith("_") and "zsh" in str(p.parent) and not p.suffix


def code_candidates(roots, rng, per_topic_files, log):
    """Walk roots once; keep up to per_topic_files candidate paths per topic (reservoir)."""
    seen = defaultdict(int)
    keep = defaultdict(list)
    for root, lic in roots:
        root = Path(root)
        if not root.exists():
            log(f"  (missing root {root})")
            continue
        t0 = time.time()
        n = 0
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in fns:
                p = Path(dp) / fn
                n += 1
                topic = NAMES.get(fn) or EXT.get(p.suffix)
                if p.suffix == ".r" and (".sdk/" in dp or ".platform/" in dp):
                    continue                                # Apple Rez resource files, not R
                if fn.endswith("_test.go"):
                    topic = "swe.testing"                   # Go test files: natural text about testing
                elif p.suffix == ".h" and "/Headers/" in dp + "/" and ".framework/" in dp:
                    topic = "prog.objc"                     # Apple framework headers (text check below)
                if topic is None and zsh_completion(p):
                    topic = "prog.shell.zsh"
                if topic is None and not p.suffix and ("bin" in dp.split("/")[-1] or dp.startswith("/etc")):
                    try:
                        with open(p, "rb") as f:
                            m = SHEBANG.match(f.read(80))
                    except OSError:
                        m = None
                    if m:
                        topic = SHE_TOPIC[m.group(1).decode()]
                if topic is None:
                    continue
                if p.suffix in (".py", ".pyi") or topic == "prog.python":
                    k = "py:" + py_topic(p, "")[0]          # one reservoir per path-derived Python topic
                else:
                    k = topic
                seen[k] += 1
                if len(keep[k]) < per_topic_files:
                    keep[k].append((p, lic))
                else:
                    j = rng.randrange(seen[k])
                    if j < per_topic_files:
                        keep[k][j] = (p, lic)
        log(f"  walked {root} ({n} files, {time.time() - t0:.0f}s)")
    return keep


def man_text(p: Path) -> str | None:
    try:
        r = subprocess.run(["/usr/bin/mandoc", "-T", "utf8", str(p)], capture_output=True, timeout=10)
        t = re.sub(r".\x08", "", r.stdout.decode("utf-8", "replace"))
        return t if len(t) > MIN_CHARS else None
    except Exception:
        return None


# ----------------------------------------------------------------------------------- inference-stack prose

KW = {
    "ml.inference.kv_cache": r"\bkv[ _-]?cache|KVCache|key[- ]value cache|kv_q8|paged kv",
    "ml.inference.prefix_cache": r"\bAPC(v2)?\b|prefix[- ]cach",
    "ml.inference.quantization": r"quantiz|quantis",
    "ml.inference.quantization.affine": r"\bq[48]\b|affine|group[_ ]size|\bbits=",
    "ml.inference.quantization.microscaling": r"mxfp[48]|nvfp4|microscal",
    "ml.inference.quantization.trellis": r"\bexl3\b|trellis|\bqtip\b|codebook",
    "ml.inference.quantization.gguf": r"\bgguf\b|Q4_K|Q8_0",
    "ml.inference.kernels.matmul": r"\bqmv\b|\bqmm\b|\bgemv\b|\bgemm\b|matmul",
    "ml.inference.kernels.attention_kernels": r"flash[- ]?attention|\bsdpa\b|paged attention|attention kernel",
    "ml.inference.kernels.metal_kernels": r"threadgroup|simdgroup|\.metal\b|metal kernel|simd_sum",
    "ml.inference.kernels.cuda_kernels": r"\bcuda\b|cublas|warp shuffle",
    "ml.inference.speculative.draft_model": r"draft model|drafter|dflash",
    "ml.inference.speculative.mtp": r"\bmtp\b|multi-token prediction",
    "ml.inference.speculative.tree": r"tree ?(spec|tiles?|verify)|\btree(7|8|9|15)\b|\bAST\b",
    "ml.inference.speculative.prompt_lookup": r"prompt[- ]lookup|n-?gram (draft|lookup)",
    "ml.inference.batching": r"continuous batch|batch width|scheduler|inflight|\blanes?\b",
    "ml.inference.serving.openai_api": r"/v1/chat/completions|openai[- ]compatible|chat completions|/v1/models",
    "ml.inference.serving.streaming": r"\bsse\b|server-sent|stream(ing)? (response|chunk|delta)",
    "ml.inference.sampling": r"temperature|top[-_ ]p\b|top[-_ ]k\b|sampler|logit bias",
    "ml.inference.long_context": r"\brope\b|\byarn\b|long[- ]context|\b(128|256)K\b|sparse attention",
    "ml.inference.memory": r"footprint|\bswap\b|unified memory|memory pressure|\bwired\b",
    "ml.inference.distributed": r"tensor[- ]parallel|pipeline[- ]parallel|data[- ]parallel|\bfabric\b|distributed (inference|serving|training|prefill)|multi-(node|GPU|host)",
    "ml.inference.conversion": r"safetensors|convert(er|ed|ing)?\b|sanitiz",
    "ml.inference.benchmarking": r"tok/s|tokens per second|\bttft\b|throughput",
    "ml.inference.prefill": r"prefill",
    "ml.inference.decode": r"\bdecode\b|decoding step",
    "ml.llm.tokenization": r"tokeni[sz]er|\bbpe\b|vocabular",
    "ml.llm.prompting": r"chat template|system prompt|prompt format",
    "ml.llm.evaluation": r"\bgsm8k\b|\bmmlu\b|\bhumaneval\b|eval(uation)? (set|harness)",
    "ml.llm.reasoning": r"reasoning|thinking|chain[- ]of[- ]thought",
    "ml.llm.tool_use": r"tool[- ]call|function call|tool parser",
    "ml.llm.agents": r"\bagents?\b|harness",
    "ml.llm.finetuning": r"\blora\b|fine[- ]?tun|\bsft\b",
    "ml.llm.alignment": r"\brlhf\b|\bdpo\b|preference optim",
    "ml.deep_learning.moe": r"mixture of experts|\bmoe\b|experts?\b|router",
    "ml.deep_learning.linear_attention": r"\bgdn\b|gated deltanet|\bmamba\b|linear attention|\bssm\b",
    "ml.deep_learning.attention": r"attention heads?|\bgqa\b|\bmla\b|multi-head attention",
    "ml.deep_learning.diffusion": r"diffusion|denois",
    "ml.deep_learning.vision": r"vision tower|image (encoder|tokens)|\bvit\b|\bvlm\b",
    "ml.deep_learning.speech": r"\basr\b|speech|\btts\b|audio",
    "ml.deep_learning.embeddings": r"embedding model|retrieval|rerank",
    "ml.interpretability.pruning": r"prun|ablat|\bmask(ed|s)? (neurons|units)|english core",
    "ml.interpretability.neurons": r"\bneurons?\b|virtual experts?|cluster C\d+",
    "ml.interpretability.probing": r"\bprobes?\b|\bcka\b|linear probe",
    "ml.interpretability.sae": r"sparse autoencoder|\bsae\b",
    "ml.hardware.apple_silicon": r"\bM[1-5] (Max|Ultra|Pro)\b|apple silicon|gpu cores",
    "ml.hardware.ane": r"\bane\b|neural engine|core ml",
    "ml.hardware.nvidia": r"\bh100\b|\ba100\b|nvidia|hopper|blackwell",
    "ml.frameworks.vllm": r"\bvllm\b|\bsglang\b",
    "ml.frameworks.llama_cpp": r"llama\.cpp|\bggml\b",
    "swe.git.branching": r"\brebase\b|cherry-pick|\bmerge\b.*\bbranch|worktree",
    "swe.git.commits": r"\bcommit(s|ted)?\b",
    "swe.testing": r"\bpytest\b|unit tests?|test suite",
    "swe.devops.deployment": r"launchd|launchctl|\bplist\b|release tree|promote",
    "swe.performance": r"profil(e|er|ing)|hot path|flame",
}
KW_RE = {k: re.compile(v, re.I) for k, v in KW.items()}


def keyword_label(text: str, min_hits=2, margin=2.0):
    hits = {k: len(r.findall(text)) for k, r in KW_RE.items()}
    # a specific quantisation leaf subsumes the generic quantisation hit count
    best = sorted(hits.items(), key=lambda kv: -kv[1])
    (k1, h1), (k2, h2) = best[0], best[1]
    if k2 == "ml.inference.quantization" and k1.startswith("ml.inference.quantization."):
        h2 = best[2][1]
    if k1 == "ml.inference.quantization" and k2.startswith("ml.inference.quantization.") and h2 >= min_hits:
        k1, h1, h2 = k2, h2, best[2][1]
    if h1 >= min_hits and h1 >= margin * max(h2, 0.5):
        return k1, h1
    return None, 0


def md_windows(text: str):
    """Paragraph-merged windows of markdown prose (no fenced code)."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)                 # HTML comments (licence headers)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 40 and not LICENSE_BOILER.search(p)
             and not re.search(r"Copyright \d{4}|SPDX-License|all rights reserved", p, re.I)]
    buf, n = [], 0
    for p in paras:
        buf.append(p)
        n += len(p)
        if n >= WIN_CHARS:
            w = "\n\n".join(buf)[:MAX_CHARS]
            yield w
            buf, n = [], 0


# ----------------------------------------------------------------------------------- languages (UI strings)

LPROJ = {"en": "eng", "en_GB": "eng", "en_AU": "eng", "en_CA": "eng", "en_IN": "eng", "English": "eng",
         "fr": "fra", "fr_CA": "fra", "French": "fra", "de": "deu", "German": "deu", "ja": "jpn", "Japanese": "jpn",
         "es": "spa", "es_419": "spa", "es_MX": "spa", "es_US": "spa", "Spanish": "spa", "it": "ita", "Italian": "ita",
         "nl": "nld", "Dutch": "nld", "pt": "por", "pt_BR": "por", "pt_PT": "por", "ru": "rus",
         "zh_CN": "cmn", "zh_TW": "cmn", "zh_HK": "cmn", "zh-Hans": "cmn", "zh-Hant": "cmn", "yue_CN": "yue",
         "ko": "kor", "ar": "arb", "he": "heb", "el": "ell", "uk": "ukr", "tr": "tur", "sv": "swe", "ro": "ron",
         "pl": "pol", "hu": "hun", "fi": "fin", "da": "dan", "cs": "ces", "ca": "cat", "vi": "vie", "sk": "slk",
         "id": "ind", "hr": "hrv", "th": "tha", "ms": "zsm", "no": "nob", "nb": "nob", "hi": "hin", "sl": "slv",
         "bg": "bul", "ta": "tam", "kk": "kaz"}
# One box id per language: a macrolanguage code from a source that has an individual-language box is folded into
# that box (Apple UI strings say zh/ar/ms/no; FLORES+ says cmn/arb/zsm/nob). Recorded in the manifest.
LANG_BOX_ALIASES = {"lang.zho": "lang.cmn", "lang.ara": "lang.arb", "lang.msa": "lang.zsm", "lang.nor": "lang.nob"}

# BISAC subject headings in Cosmopedia v2 prompts -> taxonomy nodes; longest prefix wins, unmapped headings skipped.
BISAC = {
    "HISTORY": "hum.history", "HISTORY / Ancient": "hum.history.ancient", "HISTORY / Medieval": "hum.history.medieval",
    "HISTORY / Modern": "hum.history.modern", "HISTORY / Military": "hum.history.military",
    "HISTORY / Wars & Conflicts": "hum.history.military",
    "PHILOSOPHY": "hum.philosophy", "PHILOSOPHY / Ethics & Moral Philosophy": "hum.philosophy.ethics",
    "PHILOSOPHY / Epistemology": "hum.philosophy.epistemology", "PHILOSOPHY / Metaphysics": "hum.philosophy.metaphysics",
    "PHILOSOPHY / Logic": "hum.philosophy.logic", "PHILOSOPHY / Political": "hum.philosophy.political",
    "RELIGION": "hum.religion", "POETRY": "hum.literature.poetry", "DRAMA": "hum.literature.drama",
    "LITERARY CRITICISM": "hum.literature.criticism", "LITERARY COLLECTIONS": "hum.literature",
    "LANGUAGE ARTS & DISCIPLINES / Linguistics": "hum.linguistics", "ART": "hum.art_history", "MUSIC": "hum.music",
    "PERFORMING ARTS / Film": "hum.film", "PERFORMING ARTS / Television": "hum.film",
    "BUSINESS & ECONOMICS": "soc.business", "BUSINESS & ECONOMICS / Accounting": "soc.economics.accounting",
    "BUSINESS & ECONOMICS / Marketing": "soc.business.marketing", "BUSINESS & ECONOMICS / Economics": "soc.economics",
    "BUSINESS & ECONOMICS / Finance": "soc.economics.finance", "BUSINESS & ECONOMICS / Corporate Finance": "soc.economics.finance",
    "BUSINESS & ECONOMICS / Investments & Securities": "soc.economics.finance",
    "BUSINESS & ECONOMICS / Personal Finance": "life.personal_finance", "PSYCHOLOGY": "soc.psychology",
    "SOCIAL SCIENCE / Anthropology": "soc.anthropology", "SOCIAL SCIENCE / Sociology": "soc.sociology",
    "SOCIAL SCIENCE / Human Geography": "soc.geography", "POLITICAL SCIENCE": "soc.political_science",
    "LAW": "soc.law", "LAW / Constitutional": "soc.law.constitutional", "LAW / Contracts": "soc.law.contracts",
    "LAW / Criminal Law": "soc.law.criminal", "TRAVEL": "life.travel", "COOKING": "life.cooking",
    "SPORTS & RECREATION": "life.sports", "HEALTH & FITNESS": "life.health_wellness", "HOUSE & HOME": "life.home_diy",
    "GAMES & ACTIVITIES": "life.games", "SCIENCE / Physics": "sci.physics",
    "SCIENCE / Physics / Electromagnetism": "sci.physics.electromagnetism", "SCIENCE / Chemistry": "sci.chemistry",
    "SCIENCE / Chemistry / Organic": "sci.chemistry.organic", "SCIENCE / Chemistry / Inorganic": "sci.chemistry.inorganic",
    "SCIENCE / Chemistry / Analytic": "sci.chemistry.analytical", "SCIENCE / Life Sciences": "sci.biology",
    "SCIENCE / Life Sciences / Biochemistry": "sci.chemistry.biochemistry", "SCIENCE / Life Sciences / Ecology": "sci.biology.ecology",
    "SCIENCE / Life Sciences / Botany": "sci.biology.botany", "SCIENCE / Life Sciences / Zoology": "sci.biology.zoology",
    "SCIENCE / Earth Sciences": "sci.earth", "SCIENCE / Earth Sciences / Oceanography": "sci.earth.oceanography",
    "SCIENCE / Space Science": "sci.astronomy", "NATURE / Animals": "sci.biology.zoology", "NATURE / Plants": "sci.biology.botany",
    "NATURE / Ecosystems & Habitats": "sci.biology.ecology", "MEDICAL": "sci.medicine", "MEDICAL / Nursing": "sci.medicine.nursing",
    "MEDICAL / Oncology": "sci.medicine.oncology", "TECHNOLOGY & ENGINEERING / Mechanical": "eng.mechanical",
    "TECHNOLOGY & ENGINEERING / Civil": "eng.civil", "TECHNOLOGY & ENGINEERING / Chemical & Biochemical": "eng.chemical",
    "TECHNOLOGY & ENGINEERING / Aeronautics & Astronautics": "eng.aerospace",
    "TECHNOLOGY & ENGINEERING / Biomedical": "eng.biomedical", "TECHNOLOGY & ENGINEERING / Industrial Engineering": "eng.industrial",
    "TECHNOLOGY & ENGINEERING / Electronics": "eng.electrical",
}
BISAC_RE = re.compile(r"\b([A-Z][A-Z&,' -]{2,40}(?: / [^\"\n.]{1,60}){1,4})")
COSMO_AUDIENCE = {"young_children": "form.audience.young_children", "children": "form.audience.children",
                  "middle_school_students": "form.audience.middle_school", "high_school_studnets": "form.audience.high_school",
                  "college_students": "form.audience.college", "college_studnets": "form.audience.college",
                  "researchers": "form.audience.researchers"}
COSMO_FORMAT = {"textbook": "form.textbook", "textbook_academic": "form.textbook", "textbook_narrative": "form.textbook",
                "wikihow": "form.instructions", "dialogue": "form.dialogue"}
COREV2_LIT = {"narrative": "hum.literature.fiction", "dialogue": "hum.literature.fiction", "poetry": "hum.literature.poetry",
              "drama": "hum.literature.drama"}


# Origin of a probe text, for the rule that held text should not be model-generated (coordinator, 2026-11-16).
#   natural  external human-written text (code, man pages, arXiv, Wikipedia, PG-19, FLORES+, standards, UI strings)
#   lab      this lab's own repos (mlx2 / mlx-uag docs and sources, git log, cti-radio): largely written with AI agents
#   template generated here by a program (arithmetic), not by a model
#   synthetic model-generated datasets (Cosmopedia v2: Mixtral; TinyStories: GPT-3.5/4)
ORIGIN_PREFIX = {"cosmopedia-v2": "synthetic", "tinystories": "synthetic", "gen": "template", "doc": "lab", "src": "lab",
                 "gitlog": "lab", "cti-radio": "lab"}
HELD_PREFERENCE = {"natural": 0, "lab": 1, "template": 2, "synthetic": 3}


# Redistribution of a probe's text, for published prompt sets (mapping-workflow §10): a row whose class is
# "no" or "unresolved" is referenced (source, revision, sha256), never shipped as text.
#   yes / attribution / share-alike / noncommercial: shippable under that licence's terms
#   own: this lab's work (Pierre decides); unresolved: per-package/per-file licence not resolved; no: may not ship
REDIST = [("Apple", "no"), ("private", "no"), ("CC0", "yes"), ("public domain", "yes"), ("MIT", "attribution"),
          ("BSD", "attribution"), ("Apache-2.0 (github", "attribution"), ("GPL", "share-alike"),
          ("Apache-2.0 (openai", "attribution"), ("Apache-2.0 (mlflow", "attribution"),
          ("CPython builds", "attribution"), ("ODC-BY", "attribution"), ("ODC-By", "attribution"),
          ("MITRE", "attribution"), ("Detection Rule License", "attribution"), ("CC BY-SA", "share-alike"),
          ("CDLA-Sharing", "share-alike"), ("SciFact", "noncommercial"), ("mlx2 (own)", "own"), ("mlx-uag (own)", "own"),
          ("generated (own)", "own")]


def redistribution(license_):
    return next((v for k, v in REDIST if license_.startswith(k)), "unresolved")


# Source revision per dataset prefix; path-based sources get the sha256 of the source file at build time.
SOURCE_REV = {"flores": "openlanguagedata/flores_plus devtest (downloads/flores_plus-1147e2dc)",
              "arxiv": "gfissore/arxiv-abstracts-2021@e4c5fbd4", "cosmopedia-v2": "HuggingFaceTB/smollm-corpus@3ba9d605",
              "tinystories": "roneneldan/TinyStories (HF cache snapshot)", "gsm8k": "openai/gsm8k@740312ad",
              "billsum": "FiscalNote/billsum@3d851044", "scifact": "BeIR/scifact@a75ae049",
              "800-53": "NIST SP 800-53 rev5 OSCAL catalog (downloads/sec-20261114, provenance.json)",
              "kev": "CISA KEV 2026.11.13 (downloads/sec-20261114)", "nvd": "NVD API pages (downloads/sec-20261114)",
              "attack": "MITRE ATT&CK enterprise (downloads/sec-20261114)", "attack-det": "MITRE ATT&CK enterprise (downloads/sec-20261114)",
              "cwe": "MITRE CWE cwec_latest (downloads/sec-20261114)", "sigma": "SigmaHQ (downloads/sec-20261114)",
              "gen": "scripts/amoeba_topics_corpus.py", "lproj": "macOS on this host", "loctable": "macOS on this host"}
_REV_CACHE = {}
CODE_REV = {}                       # downloaded repo root -> github:<repo>@<commit>
CTI_RADIO_REV = {}                  # "rev" -> the --cti-radio checkout's revision, set by the build
UNKNOWN_REV = "unknown (not recorded at build)"


def checkout_revision(root):
    """git:<commit> of a local checkout (+dirty when it has uncommitted changes), or None when git cannot say"""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root, capture_output=True, text=True)
        # the checkout's OWN repository: a plain directory inside some other repository has no revision
        if top.returncode or Path(top.stdout.strip()).resolve() != Path(root).resolve():
            return None
        rev = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=root, capture_output=True, text=True)
        if rev.returncode or not rev.stdout.strip():
            return None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout.strip()
        return "git:" + rev.stdout.strip() + ("+dirty" if dirty else "")
    except OSError:
        return None


def T_HASH(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def source_revision(source, corev2_rev=None, git_rev=None):
    pre = source.split(":")[0]
    if pre in SOURCE_REV:
        return SOURCE_REV[pre]
    if pre == "corev2":
        return f"core corpus v2 {corev2_rev or UNKNOWN_REV}"
    if pre == "gitlog":
        return f"mlx2 git log at {git_rev or UNKNOWN_REV}"
    if pre == "cti-radio":
        # the source names a bare file name inside the --cti-radio checkout: never resolve it as a path (it
        # would hash a same-named file of the working directory, e.g. mlx2's own README.md)
        return f"cti-radio checkout {CTI_RADIO_REV.get('rev') or UNKNOWN_REV}"
    if pre == "wikipedia":
        return f"enwiki revision {source.rsplit('#', 1)[-1]}"
    if pre == "wikiforms":
        return f"{source.split(':')[1]} revision {source.rsplit('#', 1)[-1]}"
    if pre == "nist":
        return "downloads/nist-20261116 (provenance.json sha256 per PDF)"
    path = source.split(":", 1)[1].split("#")[0] if ":" in source else ""
    for root, rev in CODE_REV.items():
        if path.startswith(root + "/"):
            return rev
    if path not in _REV_CACHE:
        try:
            _REV_CACHE[path] = "file-sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
        except OSError:
            _REV_CACHE[path] = "file missing at build"
    return _REV_CACHE[path]


def origin_of(source):
    return ORIGIN_PREFIX.get(source.split(":")[0], "natural")


def bisac_node(prompt):
    m = BISAC_RE.search(prompt or "")
    if not m:
        return None, None
    parts = [x.strip() for x in m.group(1).split(" / ")]
    for k in range(len(parts), 0, -1):
        node = BISAC.get(" / ".join(parts[:k]))
        if node:
            return node, " / ".join(parts)
    return None, " / ".join(parts)


STR_RE = re.compile(r'"((?:[^"\\]|\\.)*)"\s*=\s*"((?:[^"\\]|\\.)*)"\s*;')
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\u0e00-\u0e7f]")


def strings_values(p: Path):
    try:
        b = p.read_bytes()
    except OSError:
        return []
    if len(b) > 3_000_000:
        return []
    try:
        d = plistlib.loads(b)
        if isinstance(d, dict):
            return [v for v in d.values() if isinstance(v, str)]
    except Exception:
        pass
    for enc in ("utf-16", "utf-8"):
        try:
            t = b.decode(enc)
            break
        except UnicodeDecodeError:
            t = None
    if not t:
        return []
    return [m.group(2).replace('\\"', '"').replace("\\n", "\n") for m in STR_RE.finditer(t)]


def good_ui(v: str) -> bool:
    v = v.strip()
    if "%" in v and len(v) < 30:
        return False
    return (len(v) >= 8 and CJK.search(v)) or (len(v) >= 18 and " " in v)


def loctable_langs(p: Path):
    try:
        d = plistlib.loads(p.read_bytes())
    except Exception:
        return {}
    out = {}
    if isinstance(d, dict):
        for lang, kv in d.items():
            if isinstance(kv, dict):
                out[lang] = [v for v in kv.values() if isinstance(v, str)]
    return out


def ui_text(vals, rng):
    vals = [v.strip() for v in vals if isinstance(v, str) and good_ui(v)]
    vals = list(dict.fromkeys(vals))
    if len(vals) < 3:
        return None
    rng.shuffle(vals)
    out, n = [], 0
    for v in vals:
        out.append(v)
        n += len(v) + 1
        if n >= (240 if CJK.search(v) else WIN_CHARS):
            break
    t = "\n".join(out)
    return t if len(t) >= (120 if CJK.search(t) else MIN_CHARS) else None


# ----------------------------------------------------------------------------------- cybersecurity

CITE = re.compile(r"\(Citation:[^)]*\)|<code>.*?</code>", re.S)
MDLINK = re.compile(r"\[([^\]]*)\]\(https?://[^)]*\)")
OSCAL_PARAM = re.compile(r"\{\{\s*insert:\s*param,[^}]*\}\}")


def clean_attack(t):
    return MDLINK.sub(r"\1", CITE.sub("", t or "")).strip()
UNSAFE = re.compile(r"```|<script|\bpowershell\s+-|cmd\.exe\s+/c|\bcurl\s+-|\bwget\s+http|base64\s+-d|[A-Za-z0-9+/]{120,}={0,2}"
                    r"|\\x[0-9a-f]{2}\\x[0-9a-f]{2}|msfvenom|shellcode", re.I)


def safe_sec(text: str) -> bool:
    """Defensive, descriptive text only: no code fences, command lines, encoded blobs or payload words."""
    return not UNSAFE.search(text)


SEC_KW = {"sec.cti.tlp": r"\bTLP\b|traffic light protocol|TLP:(RED|AMBER|GREEN|CLEAR)",
          "sec.cti.stix_taxii": r"\bSTIX\b|\bTAXII\b",
          "sec.cti.indicators": r"\bIOCs?\b|indicators? of compromise",
          "sec.cti.threat_actors": r"threat actor|intrusion set|\bAPT\d*\b",
          "sec.cti.lifecycle": r"intelligence requirements?|collection plan|dissemination",
          "sec.grc.privacy": r"\bGDPR\b|personal data|privacy"}
SEC_KW_RE = {k: re.compile(v, re.I) for k, v in SEC_KW.items()}


def sec_keyword(text):
    hits = sorted(((len(r.findall(text)), k) for k, r in SEC_KW_RE.items()), reverse=True)
    (h1, k1), (h2, _) = hits[0], hits[1]
    return k1 if h1 >= 2 and h1 >= 2 * max(h2, 0.5) else None


NIST_FAM = {"ac": "sec.csf.protect.access_control", "ia": "sec.csf.protect.access_control",
            "at": "sec.csf.protect.awareness_training", "au": "sec.csf.detect.continuous_monitoring",
            "si": "sec.csf.detect.adverse_event_analysis", "cm": "sec.csf.protect.platform_security",
            "ma": "sec.csf.protect.platform_security", "mp": "sec.csf.protect.data_security",
            "sc": "sec.domain.network", "pe": "sec.csf.protect.infrastructure_resilience",
            "cp": "sec.csf.recover.recovery_plan", "ir": "sec.csf.respond.incident_management",
            "pl": "sec.csf.govern.policy", "pm": "sec.csf.govern.risk_strategy", "ps": "sec.csf.govern.roles",
            "pt": "sec.grc.privacy", "ra": "sec.csf.identify.risk_assessment", "ca": "sec.grc.audit",
            "sa": "sec.domain.appsec", "sr": "sec.csf.govern.supply_chain"}


def _oscal_text(parts):
    out = []
    for p in parts or []:
        if p.get("prose"):
            out.append(p["prose"])
        out += _oscal_text(p.get("parts"))
    return out


def sec_sources(d: Path):
    """Yield (topics, text, source, group, licence) from the approved security sources in d."""
    f = d / "known_exploited_vulnerabilities.json"
    if f.exists():
        for v in json.loads(f.read_text())["vulnerabilities"]:
            t = (f"{v['cveID']} {v.get('vendorProject', '')} {v.get('product', '')}: {v.get('vulnerabilityName', '')}. "
                 f"{v.get('shortDescription', '')} Required action: {v.get('requiredAction', '')}")
            yield ["sec.vuln.kev", "sec.vuln"], t, f"kev:{v['cveID']}", f"kev/{v.get('vendorProject', '')}", "public domain (CISA KEV)"
    for f in sorted(d.glob("nvd_page*.json")):
        for v in json.loads(f.read_text()).get("vulnerabilities", []):
            c = v["cve"]
            desc = next((x["value"] for x in c.get("descriptions", []) if x.get("lang") == "en"), "")
            cwes = sorted({x["value"] for w in c.get("weaknesses", []) for x in w.get("description", [])})
            t = f"{c['id']} ({', '.join(cwes)}): {desc}"
            yield ["sec.vuln.cve", "sec.vuln"], t, f"nvd:{c['id']}", f"nvd/{c['id']}", "public domain (NVD)"
    f = d / "enterprise-attack.json"
    if f.exists():
        for o in json.loads(f.read_text())["objects"]:
            if o.get("revoked") or o.get("x_mitre_deprecated"):
                continue
            desc = clean_attack(o.get("description", ""))
            name, ty = o.get("name", ""), o.get("type")
            top = {"attack-pattern": "sec.cti.attack_techniques", "intrusion-set": "sec.cti.threat_actors",
                   "course-of-action": "sec.csf.respond.mitigation", "malware": "sec.dfir.malware_analysis",
                   "tool": "sec.dfir.malware_analysis", "campaign": "sec.cti.threat_actors"}.get(ty)
            if top and len(desc) > 200:
                yield [top, "sec.cti"], f"{name}: {desc}", f"attack:{o['id']}", f"attack/{o['id']}", "MITRE ATT&CK Terms of Use"
            det = clean_attack(o.get("x_mitre_detection", ""))
            if ty == "attack-pattern" and len(det) > 200:
                yield ["sec.defensive.detection_engineering", "sec.defensive"], f"Detecting {name}: {det}", \
                    f"attack-det:{o['id']}", f"attack/{o['id']}", "MITRE ATT&CK Terms of Use"
    f = d / "cwec_latest.xml"
    if f.exists():
        import xml.etree.ElementTree as ET
        for w in ET.parse(f).getroot().iter():
            if not w.tag.endswith("}Weakness"):
                continue
            txt = " ".join(" ".join(e.itertext()) for e in w if e.tag.endswith(("}Description", "}Extended_Description")))
            txt = " ".join(txt.split())
            if len(txt) > 200:
                yield ["sec.vuln.cwe", "sec.vuln"], f"CWE-{w.get('ID')} {w.get('Name')}: {txt}", f"cwe:{w.get('ID')}", \
                    f"cwe/{w.get('ID')}", "MITRE CWE Terms of Use"
    f = d / "NIST_SP-800-53_rev5_catalog.json"
    if f.exists():
        for g in json.loads(f.read_text())["catalog"]["groups"]:
            top = NIST_FAM.get(g["id"], "sec.grc.nist_800_53")     # families without a CSF node: the 800-53 leaf
            for c in [c for c0 in g.get("controls", []) for c in [c0] + c0.get("controls", [])]:   # with enhancements
                txt = OSCAL_PARAM.sub("[organization-defined value]", " ".join(" ".join(_oscal_text(c.get("parts"))).split()))
                if top and len(txt) > 200:
                    yield list(dict.fromkeys([top, "sec.grc.nist_800_53", "sec.grc"])), f"{c['id'].upper()} {c['title']}: {txt}", \
                        f"800-53:{c['id']}", f"800-53/{c['id']}", "public domain (NIST SP 800-53 rev5, OSCAL)"
                # v5.2: the control's own parts as extra texts (same group, so the same split as the control):
                # guidance -> the 800-53 leaf; assessment objective and methods -> audit (privacy for PT)
                for names, leaf in ((("guidance",), "sec.grc.nist_800_53"),
                                    (("assessment-objective", "assessment-method"),
                                     "sec.grc.privacy" if g["id"] == "pt" else "sec.grc.audit")):
                    part = [q for q in c.get("parts") or [] if q.get("name") in names]
                    ptxt = OSCAL_PARAM.sub("[organization-defined value]", " ".join(" ".join(_oscal_text(part)).split()))
                    if len(ptxt) > 200:
                        yield list(dict.fromkeys([leaf, "sec.grc"])), f"{c['id'].upper()} {c['title']} ({names[0]}): {ptxt}", \
                            f"800-53:{c['id']}#{names[0]}", f"800-53/{c['id']}", "public domain (NIST SP 800-53 rev5, OSCAL)"
    sig = d / "sigma"
    if sig.exists():
        for p in sorted(sig.rglob("*.yml")):
            try:
                t = p.read_text(errors="replace")
            except OSError:
                continue
            keep, block = [], None          # top-level YAML blocks; the detection logic is never kept
            for l in t.splitlines():
                if re.match(r"^[A-Za-z_]+:", l):
                    block = l.split(":", 1)[0]
                if block in ("title", "description", "status", "level", "logsource", "falsepositives", "tags"):
                    keep.append(l)
            txt = "\n".join(keep)[:MAX_CHARS]
            if "description" in txt and len(txt) > 200:
                yield ["sec.defensive.detection_engineering", "sec.defensive", "prog.data.yaml"], txt, f"sigma:{p.name}", \
                    f"sigma/{p.parent.name}", "Detection Rule License 1.1 (SigmaHQ)"


# ----------------------------------------------------------------------------------- main

def parquet_rows(path, cols):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=cols).to_pylist()


def annotate(corpus_path):
    """Sidecar provenance.jsonl for an already-built corpus (its corpus.jsonl and hash stay unchanged)."""
    corpus_path = Path(corpus_path)
    out = corpus_path.parent / "provenance.jsonl"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    counts = Counter()
    with open(out, "w") as f:
        for line in open(corpus_path):
            r = json.loads(line)
            row = {"id": r["id"], "sha256": r["sha256"], "split": r["split"], "topic": r["topics"][0],
                   "origin": r.get("origin") or origin_of(r["source"]), "source": r["source"], "license": r["license"],
                   # the build's own record when the corpus row has one: recomputing here would read today's
                   # files, and the corev2 / git-log revisions are known only to the build
                   "source_revision": r.get("source_revision") or source_revision(r["source"]),
                   "redistribute": redistribution(r["license"])}
            counts[row["redistribute"]] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out}: {dict(counts)}")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--annotate":
        return annotate(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="research-artifacts dir for corpus.jsonl")
    ap.add_argument("--manifest", default=str(ROOT / "docs/research/amoeba-topics/probe-corpus-manifest.json"))
    ap.add_argument("--per-split", type=int, default=64, help="fit texts per topic")
    ap.add_argument("--held-per-split", type=int, default=32, help="held texts per topic")
    ap.add_argument("--lang-per-split", type=int, default=0, help="lang.* fit quota (0: same as --per-split)")
    ap.add_argument("--cosmopedia", default="", help="local cosmopedia-v2 parquet shard (BISAC subject, audience, format)")
    ap.add_argument("--cosmo-per-node", type=int, default=400, help="Cosmopedia candidates kept per node")
    ap.add_argument("--extra-roots", action="store_true", help="v5.2 code roots for thin branches (Tcl, R, Fortran, ObjC)")
    ap.add_argument("--code-downloads", default="", help="downloads/code-<date> (provenance.json: repo, licence, commit)")
    ap.add_argument("--wikipedia", default="", help="downloads/wikipedia-<date> (amoeba_wiki_fetch.py output)")
    ap.add_argument("--wiki-forms", default="", help="downloads/wikiforms-<date> (amoeba_wiki_forms_fetch.py output)")
    ap.add_argument("--nist", default="", help="downloads/nist-<date> (SP PDFs as text + provenance.json)")
    ap.add_argument("--form-natural", action="store_true", help="natural text for form.* nodes (PG-19, arXiv, tutorial)")
    ap.add_argument("--carry", default="", help="earlier probe corpus.jsonl: its texts are kept and selected first")
    ap.add_argument("--corev2", default="", help="core corpus v2 corpus.jsonl (PG-19 literature, Wikipedia leads)")
    ap.add_argument("--files-per-topic", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlx2", default=str(HOME / "Desktop/mlx2"))
    ap.add_argument("--mlx-uag", default=str(HOME / "Desktop/mlx-uag"))
    ap.add_argument("--math", default=str(HOME / "Desktop/math"))
    ap.add_argument("--arxiv-sample", default="", help="arxiv_sample.jsonl from amoeba_topics_arxiv_sample.py")
    ap.add_argument("--flores", default="", help="FLORES+ devtest dir of <iso3>_<Script>.jsonl files (optional)")
    ap.add_argument("--sec", default="", help="dir with the approved security sources (see sec_sources())")
    ap.add_argument("--cti-radio", default="", help="cti-radio checkout, e.g. $HOME/Desktop/cti-radio (read-only; private, withheld)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "build.log", "a")

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        logf.write(line + "\n")

    taxo = json.loads(TAXO.read_text())
    valid = {n["id"] for n in taxo["nodes"]}
    rng = random.Random(args.seed)
    cands = []          # dicts without split

    def add(topics, text, source, group, lic, method, path_key):
        topics = [LANG_BOX_ALIASES.get(t, t) for t in topics]
        for t in topics:
            if t not in valid:
                raise SystemExit(f"unknown topic {t} from {source}")
        if text and len(text.strip()) >= 100:
            cands.append({"topics": list(dict.fromkeys(topics)), "text": text, "source": source, "group": group,
                          "license": lic, "method": method, "path_key": path_key})

    # 0. carry an earlier build forward: same texts, groups and labels (macrolanguage codes folded), selected first,
    #    so disk drift in the code roots (pruned caches) cannot shrink a topic below the earlier build
    carried, carry_dropped = set(), Counter()
    if args.carry:
        dropped = Counter()
        for line in open(args.carry):
            r = json.loads(line)
            src = r["source"]
            if r["topics"][0] == "prog.r" and src.endswith(".r") and (".sdk/" in src or ".platform/" in src):
                dropped["Rez .r labelled prog.r"] += 1          # mislabels fixed in v5.2 are not carried
                continue
            if r["topics"][0] == "prog.objc" and src.endswith(".m") and not re.search(
                    r"#import\b|@(interface|implementation|protocol)\b", r["text"]):
                dropped["non-ObjC .m labelled prog.objc"] += 1
                continue
            add(r["topics"], r["text"], r["source"], r["group"], r["license"], r["method"], r["path_key"])
            carried.add(sha(r["text"]))
        carry_dropped.update(dropped)
        log(f"carried {len(carried)} texts from {args.carry}; dropped {dict(dropped)}")

    # 1. code and config by file type -------------------------------------------------------------
    uv = HOME / ".cache/uv/archive-v0"
    roots = [(HOME / "Desktop/mlx2/.venv/lib", "per-package (site-packages METADATA)"),
             (uv, "per-package (site-packages METADATA)"),
             ("/opt/homebrew/Cellar", "per-formula (Homebrew; see each formula)"),
             ("/opt/homebrew/share/zsh", "per-formula (Homebrew)"),
             ("/opt/homebrew/share/fish", "per-formula (Homebrew)"),
             (HOME / ".cargo/registry/src", "per-crate (crates.io; MIT/Apache-2.0 mostly)"),
             (HOME / "go/pkg/mod", "per-module (Go modules)"),
             ("/System/Library/Perl", "Perl (Artistic/GPL) as shipped by Apple"),
             ("/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include",
              "Apple SDK headers (APSL/BSD; internal research use)"),
             ("/Applications/Xcode.app/Contents/Developer/Toolchains", "Apple toolchain (Apache-2.0 Swift / APSL)"),
             ("/etc", "macOS system configuration (Apple)"), ("/usr/bin", "macOS system scripts (Apple)"),
             ("/usr/sbin", "macOS system scripts (Apple)"),
             (Path(args.math) / "lean", "Apache-2.0 (openai/math)")]
    if args.code_downloads:                 # v5.3: permissive GitHub repos (provenance.json per repo: licence, commit)
        cprov = json.loads((Path(args.code_downloads) / "provenance.json").read_text())["repos"]
        for d, meta in sorted(cprov.items()):
            roots.append((Path(args.code_downloads) / d,
                          f"{meta['license']} (github:{meta['url'].split('github.com/')[1]}@{meta['commit'][:12]})"))
            CODE_REV[str(Path(args.code_downloads) / d)] = f"github:{meta['url'].split('github.com/')[1]}@{meta['commit']}"
    if args.extra_roots:                    # v5.2 (2026-11-16): thin language branches, local licensed text only
        xp = "/Applications/Xcode.app/Contents/Developer/Platforms"
        roots += [(HOME / ".local/share/uv/python", "CPython builds: PSF; bundled Tcl/Tk library: Tcl/Tk licence (BSD-style)"),
                  (HOME / "Desktop/NKOS/buggy-repos/mlflow-current/mlflow/R", "Apache-2.0 (mlflow R package)"),
                  (HOME / "Desktop/mlx-uag/.venv-sdpa-base/lib", "per-package (site-packages METADATA)"),
                  (f"{xp}/MacOSX.platform/Developer/Library/Xcode/Templates", "Apple Xcode templates (internal research use)"),
                  (f"{xp}/iPhoneOS.platform/Developer/Library/Xcode/Templates", "Apple Xcode templates (internal research use)"),
                  (f"{xp}/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks",
                   "Apple SDK framework headers (internal research use)")]
    log("walking code roots")
    keep = code_candidates(roots, rng, args.files_per_topic, log)
    for k, files in sorted(keep.items()):
        thin = len(files) < 80 or k in THIN_CODE      # thin topics: up to 3 windows per file (the per-group cap)
        n_win = 3 if thin else 1
        for p, lic in [f for f in files for _ in range(n_win)]:
            text = read_text(p)
            if not text:
                continue
            topic = k if not k.startswith("py:") else None
            if k.startswith("py:"):
                topics = py_topic(p, text)
            elif k == "prog.assembly":
                a, x = len(ASM_ARM.findall(text)), len(ASM_X86.findall(text))
                topics = ["prog.assembly.arm64"] if a > 2 * x else ["prog.assembly.x86"] if x > 2 * a else ["prog.assembly"]
            elif k == "prog.c" and p.suffix == ".h" and ("c++" in str(p) or "/v1/" in str(p)):
                topics = ["prog.cpp"]
            elif k == "prog.objc" and p.suffix == ".h":
                if not re.search(r"@(interface|protocol|property)\b", text):
                    continue
                topics = ["prog.objc", "prog.c"]
            elif k == "prog.objc" and not re.search(r"#import\b|@(interface|implementation|protocol)\b", text):
                continue                                    # .m without Objective-C markers: MATLAB/Octave
            elif k == "swe.testing":
                topics = ["swe.testing", "prog.go"]
            else:
                topics = [topic] + ALSO.get(topic, [])
            if topics[0] == "prog.lean":
                topics = ["prog.lean", "cs.formal_methods"]
            w = window(text, rng, allow_boiler=(topics[0] == "swe.licensing"))
            parts = p.parts
            gkey = "/".join(parts[-3:])
            add(topics, w, f"file:{p}", gkey, lic, "path" if k.startswith("py:") else "extension", gkey)
    # man pages: git -> swe.git, section 1 -> unix tools (rendered text) and raw roff -> prog.markup.man
    log("man pages")
    for mdir in ["/usr/share/man/man1", "/Applications/Xcode.app/Contents/Developer/usr/share/man/man1",
                 "/opt/homebrew/share/man/man1"]:
        md = Path(mdir)
        if not md.exists():
            continue
        files = sorted(md.iterdir())
        rng.shuffle(files)
        for p in files[:400]:
            name = p.name
            if name.startswith("git"):
                topics = ["swe.git", "swe.os.unix_tools"]
                if any(x in name for x in ("rebase", "merge", "branch", "cherry", "worktree", "checkout", "switch")):
                    topics = ["swe.git.branching", "swe.git"]
                elif any(x in name for x in ("commit", "log", "blame", "show")):
                    topics = ["swe.git.commits", "swe.git"]
            else:
                topics = ["swe.os.unix_tools"]
            t = man_text(p)
            if t:
                add(topics, window(t, rng), f"man:{p}", f"man/{name}", "man page of the tool (BSD/GPL/Apple)",
                    "path", f"man/{name}")
            if len(cands) % 7 == 0:
                raw = read_text(p)
                if raw:
                    add(["prog.markup.man"], window(raw, rng), f"roff:{p}", f"roff/{name}", "man page source", "extension",
                        f"roff/{name}")
    # git commit messages (own work)
    log("git log")
    try:
        msgs = subprocess.run(["git", "-C", args.mlx2, "log", "-n", "6000", "--format=%B%x00"], capture_output=True,
                              text=True, timeout=60).stdout.split("\x00")
        buf, n, gi = [], 0, 0
        for m in msgs:
            m = m.strip()
            if not m or "Co-Authored-By" in m and len(m) < 80:
                continue
            m = "\n".join(l for l in m.splitlines() if not l.startswith("Co-Authored-By"))
            buf.append(m)
            n += len(m)
            if n >= WIN_CHARS:
                gi += 1
                add(["swe.git.commits", "swe.git"], "\n\n".join(buf)[:MAX_CHARS], f"gitlog:mlx2#{gi}", f"gitlog/{gi // 4}",
                    "mlx2 (own)", "path", f"gitlog/{gi}")
                buf, n = [], 0
    except Exception as e:
        log(f"  git log failed: {e}")

    # 2. inference stack and lab prose (keyword labels) ------------------------------------------
    log("inference-stack prose")
    docs = []
    for base, lic in [(Path(args.mlx2) / "docs", "mlx2 (own)"), (Path(args.mlx_uag), "mlx-uag (own)")]:
        for p in sorted(base.rglob("*.md")):
            if any(s in p.parts for s in SKIP_DIRS) or "cairn" in p.parts:
                continue
            docs.append((p, lic))
    rng.shuffle(docs)
    kw_n = Counter()
    for p, lic in docs[:6000]:
        t = read_text(p)
        if not t:
            continue
        for i, w in enumerate(md_windows(t)):
            lab, h = keyword_label(w)
            if lab and kw_n[lab] < 400:
                kw_n[lab] += 1
                also = ["swe.git"] if lab.startswith("swe.git.") else []
                add([lab] + also, w, f"doc:{p}#{i}", f"doc/{p.parent.name}/{p.name}", lic, "keyword", f"{p.name}#{i}")
    for base, lic in [(Path(args.mlx2) / "src/mlx2", "mlx2 (own)")]:
        for p in sorted(base.rglob("*.py")):
            t = read_text(p)
            if not t:
                continue
            w = window(t, rng)
            if not w:
                continue
            lab, h = keyword_label(w)
            topics = [lab, "prog.python", "ml.frameworks.mlx2"] if lab else ["ml.frameworks.mlx2", "prog.python"]
            add(topics, w, f"src:{p}", f"src/{p.parent.name}/{p.name}", lic, "keyword" if lab else "path", p.name)
    for p in sorted((Path(args.mlx2) / "src").rglob("*.metal")):
        t = read_text(p)
        if t:
            add(["ml.inference.kernels.metal_kernels", "prog.gpu.metal"], window(t, rng), f"src:{p}", f"metal/{p.name}",
                "mlx2 (own)", "path", p.name)
    # amoeba research notes -> interpretability
    for p in sorted((Path(args.mlx2) / "docs/research").glob("*amoeba*")):
        for q in ([p] if p.is_file() else sorted(p.rglob("*.md"))):
            t = read_text(q)
            if not t:
                continue
            for i, w in enumerate(md_windows(t)):
                if i > 6:
                    break
                lab, _ = keyword_label(w)
                if lab and lab.startswith("ml.interpretability"):
                    add([lab], w, f"doc:{q}#{i}", f"amoeba/{q.name}", "mlx2 (own)", "keyword", f"{q.name}#{i}")

    # 3. math manuscripts (openai/math, Apache-2.0), by catalogue subject -------------------------
    log("math manuscripts")
    SUBJ = {"Number theory": "math.number_theory", "Algebraic and complex geometry": "math.geometry.algebraic_geometry",
            "Real and complex analysis": "math.analysis.real_complex", "Convex and metric geometry": "math.geometry.metric_geometry",
            "Theoretical computer science": "math.theoretical_cs", "Dynamical systems and ergodic theory": "math.dynamical_systems",
            "Combinatorics": "math.combinatorics", "Algebra": "math.algebra", "Probability and statistical mechanics": "math.probability",
            "Mathematical logic": "math.logic", "Group theory": "math.algebra.group_theory",
            "Mathematical physics": "math.mathematical_physics", "Operator algebras": "math.algebra.operator_algebras",
            "Topology": "math.topology", "Functional analysis": "math.analysis.functional_analysis",
            "Differential geometry": "math.geometry.differential_geometry", "Partial differential equations": "math.analysis.pde"}
    ov = Path(args.math) / "overview.tex"
    if ov.exists():
        subj, fam_of = None, {}
        for line in ov.read_text().splitlines():
            m = re.match(r"\\cataloguesection\{([^}]*)\}", line)
            if m:
                subj = m.group(1)
            m = re.match(r"\\resultentry\{(\d+)\}", line)
            if m and subj:
                for d in re.findall(r"preprints/([^/]+)/", line):
                    fam_of[d] = (m.group(1), SUBJ.get(subj))
        for d, (fam, topic) in sorted(fam_of.items()):
            if not topic:
                continue
            texs = [q for q in sorted((Path(args.math) / "preprints" / d).rglob("*.tex"))
                    if not re.search(r"ref|bib|macro|preamble", q.name, re.I)]
            texs.sort(key=lambda q: -q.stat().st_size)
            for tex in texs[:2]:
                t = read_text(tex)
                if not t:
                    continue
                body = t.split("\\begin{document}", 1)[-1]
                w = window(body, rng)
                add([topic, "form.academic_register"], w, f"tex:{tex}", f"mathfam/{fam}", "Apache-2.0 (openai/math)",
                    "dataset_label", f"{d}/{tex.name}")

    # 4. datasets in the local HF cache ----------------------------------------------------------
    log("local datasets")
    sf = HF / "datasets--BeIR--scifact/snapshots/a75ae049398addde9b70f6b268875f5cbce99089/corpus.jsonl.gz"
    BIO = {"sci.biology.genetics": r"\bgene(s|tic|tics|ome)?\b|\bDNA\b|mutation|allele|genom",
           "sci.biology.molecular": r"protein|\bRNA\b|transcription|kinase|receptor|phosphorylat",
           "sci.biology.cell": r"\bcells?\b|mitochond|stem cell|apoptosis|cell cycle",
           "sci.biology.immunology": r"immun|\bT cells?\b|antibod|cytokine|antigen",
           "sci.biology.neuroscience": r"neuron|brain|cortex|cortical|synap|neural",
           "sci.biology.microbiology": r"bacteri|microb|viral|virus|pathogen",
           "sci.medicine.oncology": r"cancer|tumou?r|oncolog|carcinoma|metasta",
           "sci.medicine.cardiology": r"cardi|heart|coronary|myocard|vascular",
           "sci.medicine.epidemiology": r"cohort|prevalence|incidence|risk factor|population-based",
           "sci.medicine.clinical": r"\bpatients?\b|clinical trial|randomi[sz]ed|placebo",
           "sci.medicine.pharmacology": r"\bdrug|dose|pharmaco|inhibitor",
           "sci.medicine.infectious_disease": r"\bHIV\b|malaria|tubercul|infection|influenza"}
    BIO_RE = {k: re.compile(v, re.I) for k, v in BIO.items()}
    if sf.exists():
        for line in gzip.open(sf, "rt"):
            r = json.loads(line)
            t = (r.get("title", "") + "\n" + r.get("text", "")).strip()
            if len(t) < MIN_CHARS:
                continue
            hits = sorted(((len(rx.findall(t)), k) for k, rx in BIO_RE.items()), reverse=True)
            (h1, k1), (h2, _) = hits[0], hits[1]
            if h1 >= 3 and h1 >= 2 * max(h2, 0.5):
                add([k1, "form.academic_register"], t[:MAX_CHARS], f"scifact:{r['_id']}", f"scifact/{r['_id']}",
                    "SciFact corpus (CC BY-NC 2.0)", "keyword", r["_id"])
    gs = HF / "datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet"
    if gs.exists():
        for i, r in enumerate(parquet_rows(gs, ["question", "answer"])):
            add(["math.word_problems"], r["question"] + "\n" + r["answer"], f"gsm8k:test#{i}", f"gsm8k/{i}", "MIT (GSM8K)",
                "dataset_label", f"gsm8k#{i}")
    bs = HF / "datasets--FiscalNote--billsum/snapshots/3d8510441c06a3d9dfb32eb0d7f80151730bcc4f/data/train-00000-of-00001.parquet"
    if bs.exists():
        rows = parquet_rows(bs, ["text"])
        for i, r in enumerate(rows[:600]):
            w = window(r["text"], rng)
            add(["soc.law.us_legislation", "form.legal_register"], w, f"billsum:train#{i}", f"billsum/{i}", "CC0-1.0 (BillSum)",
                "dataset_label", f"billsum#{i}")
    tsl = sorted(HF.glob("datasets--roneneldan--TinyStories/snapshots/*/data/validation-*.parquet"))
    ts = tsl[0] if tsl else HF / "missing"
    if ts.exists():
        rows = parquet_rows(ts, ["text"])
        for i, r in enumerate(rows[:600]):
            add(["form.narrative.childrens_story", "lang.eng"], r["text"][:MAX_CHARS], f"tinystories:val#{i}",
                f"tinystories/{i}", "CDLA-Sharing-1.0 (TinyStories)", "dataset_label", f"ts#{i}")
            if r["text"].lstrip().lower().startswith("once upon a time") and i % 3 == 0:
                opening = " ".join(re.split(r"(?<=[.!?])\s+", r["text"].strip())[:3])
                add(["form.narrative.story_opening", "form.narrative.childrens_story"], opening,
                    f"tinystories:val#{i}o", f"tinystories/{i}", "CDLA-Sharing-1.0 (TinyStories)", "dataset_label", f"ts#{i}o")

    # 4b. arXiv abstracts (CC0 metadata; downloaded with Pierre's approval 2026-11-14) ----------------
    if args.arxiv_sample:
        for line in open(args.arxiv_sample):
            r = json.loads(line)
            topics = [r["topic"], "form.academic_register"]
            if args.form_natural and T_HASH(r["arxiv_id"]) % 5 == 0:     # natural researcher-register text
                topics = ["form.audience.researchers"] + topics
            add(topics, r["text"][:MAX_CHARS], f"arxiv:{r['arxiv_id']}",
                f"arxiv/{r['arxiv_id']}", "CC0-1.0 (arXiv metadata via gfissore/arxiv-abstracts-2021@e4c5fbd4)",
                "dataset_label", f"arxiv/{r['primary']}")
    # 4c. FLORES+ devtest: the same sentences in every language, so language is the only variable --------
    if args.flores:
        for f in sorted(Path(args.flores).glob("*.jsonl")):
            iso = f.stem.split("_")[0]
            if f"lang.{iso}" not in valid:
                continue
            sents = [json.loads(l).get("text", "") for l in open(f)]
            for i in range(0, len(sents) - 4, 5):         # 5 consecutive sentences per text; group by block
                add([f"lang.{iso}"], " ".join(sents[i:i + 5])[:MAX_CHARS], f"flores:{f.stem}#{i}", f"flores/{i}",
                    "CC BY-SA 4.0 (FLORES+)", "dataset_label", f"flores/{f.stem}#{i}")

    # 4e. Cosmopedia v2 (local shard, synthetic, ODC-BY): BISAC subject headings in the prompt, audience and format
    #     labels. Each row is one text with one primary label: whichever of its labels has the fewest texts so far.
    if args.cosmopedia:
        import pyarrow.parquet as pq
        log("Cosmopedia v2")
        kept = Counter()
        pf = pq.ParquetFile(args.cosmopedia)
        row = 0
        for b in pf.iter_batches(batch_size=20000, columns=["prompt", "text", "audience", "format"]):
            d = b.to_pydict()
            for prompt, text, aud, fmt in zip(d["prompt"], d["text"], d["audience"], d["format"]):
                row += 1
                subj, heading = bisac_node(prompt)
                aud_n, fmt_n = COSMO_AUDIENCE.get(aud), COSMO_FORMAT.get(fmt)
                open_ = [x for x in (subj, aud_n, fmt_n) if x and kept[x] < args.cosmo_per_node]
                if not open_:
                    continue
                choice = min(open_, key=lambda x: kept[x])          # the thinnest label gets the row
                w = window(text, rng)
                if not w:
                    continue
                kept[choice] += 1
                also = [x for x in (subj, aud_n, fmt_n) if x and x != choice]
                add([choice] + also, w, f"cosmopedia-v2:{heading or aud + '/' + fmt}", f"cosmo/{row}",
                    "ODC-BY (cosmopedia-v2, smollm-corpus @3ba9d605; synthetic, Mixtral-generated)",
                    "dataset_label", f"cosmo#{row}")
            if kept and min(kept.get(n, 0) for n in set(BISAC.values()) | set(COSMO_AUDIENCE.values())
                            | set(COSMO_FORMAT.values())) >= args.cosmo_per_node:
                break
        log(f"  {row} rows read, {sum(kept.values())} kept over {len(kept)} nodes")
    # 4f. core corpus v2: PG-19 literature windows (public domain) and Wikipedia leads (encyclopedic form) --------
    if args.corev2:
        for line in open(args.corev2):
            r = json.loads(line)
            src = r["source"]
            if src.startswith("pg19:") and r["register"] in COREV2_LIT:
                topics = [COREV2_LIT[r["register"]], "lang.eng"]
                if args.form_natural and T_HASH(src) % 2 == 0 and r["register"] in ("dialogue", "narrative"):
                    topics = ["form.dialogue" if r["register"] == "dialogue" else "form.narrative"] + topics   # by book
                add(topics, r["text"], f"corev2:{src}#{r['id']}", "pg19/" + src[5:], r["license"], "dataset_label", r["id"])
            elif args.form_natural and src.startswith("python-tutorial:"):
                add(["form.instructions", "prog.python", "lang.eng"], r["text"], f"corev2:{src}#{r['id']}", r["group"],
                    r["license"], "dataset_label", r["id"])
            elif src.startswith("wikipedia:"):
                add(["form.encyclopedic", "lang.eng"], r["text"], f"corev2:{src}#{r['id']}", r["group"], r["license"],
                    "dataset_label", r["id"])

    # 4g. v5.3 downloads (Pierre, in chat, 2026-11-16): Wikipedia leads per node, NIST SPs ------------
    if args.wikipedia:
        log("Wikipedia leads")
        wdir = Path(args.wikipedia)
        for f in sorted(wdir.glob("*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                text = r["text"][:MAX_CHARS]
                if len(text) < MIN_CHARS:
                    continue
                topics = [r["node"], "form.encyclopedic", "lang.eng"]
                if T_HASH(r["title"]) % 8 == 0:
                    topics = ["form.encyclopedic", r["node"], "lang.eng"]
                elif args.form_natural and T_HASH(r["title"]) % 8 == 1:     # general-adult reading level
                    topics = ["form.audience.college", r["node"], "form.encyclopedic", "lang.eng"]
                add(topics, text, f"wikipedia:{r['title']}#{r['revid']}", f"wiki/{r['title']}",
                    "CC BY-SA 4.0 (Wikipedia)", "dataset_label", f"enwiki/{r['pageid']}")
    if args.wiki_forms:
        log("Wikibooks / Simple English Wikipedia")
        for f in sorted(Path(args.wiki_forms).glob("*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                text = r["text"][:MAX_CHARS]
                if len(text) < MIN_CHARS:
                    continue
                book = r["title"].split("/")[0]                 # a Wikibooks book stays in one split
                add([r["node"], "lang.eng"], text, f"wikiforms:{r['site']}:{r['title']}#{r['revid']}",
                    f"wikiforms/{r['site']}/{book}", f"CC BY-SA 4.0 ({r['site']})", "dataset_label", f"{r['site']}/{r['pageid']}")
    if args.nist:
        log("NIST SPs")
        nprov = json.loads((Path(args.nist) / "provenance.json").read_text())["files"]
        for stem, meta in sorted(nprov.items()):
            t = (Path(args.nist) / f"{stem}.txt").read_text(errors="replace")
            paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", t)]
            paras = [p for p in paras if 300 <= len(p) <= 2400 and not re.search(r"\.{5,}|^\s*(Table|Figure) \d", p)]
            for i, ptxt in enumerate(paras):
                add([meta["topic"], meta["topic"].rsplit(".", 1)[0]], ptxt[:MAX_CHARS], f"nist:{stem}#{i}",
                    f"nist/{stem}/{i // 4}", "public domain (NIST SP; US government work)", "dataset_label", f"{stem}#{i}")

    # 4d. cybersecurity (defensive, descriptive text only) ------------------------------------------
    if args.sec:
        for topics, text, src, group, lic in sec_sources(Path(args.sec)):
            if safe_sec(text):
                add(topics, text[:MAX_CHARS], src, group, lic, "dataset_label", group)
    if args.cti_radio:
        CTI_RADIO_REV["rev"] = checkout_revision(args.cti_radio)
        for p in sorted(Path(args.cti_radio).rglob("*.md")):
            if any(x in p.parts for x in SKIP_DIRS):
                continue
            t = read_text(p)
            if not t:
                continue
            for i, w in enumerate(md_windows(t)):
                lab = sec_keyword(w)
                if lab and safe_sec(w):
                    add([lab], w, f"cti-radio:{p.name}#{i}", f"cti-radio/{p.name}", "private (Pierre's cti-radio project)",
                        "keyword", f"{p.name}#{i}")

    # 5. arithmetic (generated here) ---------------------------------------------------------------
    for i in range(400):
        r2 = random.Random(1000 + i)
        if i % 2 == 0:
            lines = [f"{a} + {b} = {a + b}" for a, b in ((r2.randint(10, 9999), r2.randint(10, 9999)) for _ in range(30))]
            topics = ["math.arithmetic.addition", "form.digits"]
        else:
            lines = [f"{a} x {b} = {a * b}" for a, b in ((r2.randint(2, 99), r2.randint(2, 999)) for _ in range(30))]
            topics = ["math.arithmetic.multiplication", "form.digits"]
        add(topics, "\n".join(lines), f"gen:arith#{i}", f"arith/{i}", "generated (own)", "template", f"arith#{i}")

    # 6. languages from UI strings ----------------------------------------------------------------
    log("UI strings (.lproj)")
    per_lang = defaultdict(int)
    bundles = []
    for base in ["/System/Library/CoreServices", "/System/Library/PrivateFrameworks", "/System/Library/Frameworks",
                 "/System/Applications", "/Applications/Safari.app"]:
        for dp, dns, fns in os.walk(base):
            if dp.endswith(".lproj"):
                bundles.append(Path(dp))
                dns[:] = []
            elif len(bundles) > 200000:
                break
    rng.shuffle(bundles)
    log(f"  {len(bundles)} .lproj dirs")
    loct_done = 0
    for d in bundles:
        lang = LPROJ.get(d.name[:-6])
        if lang and per_lang[lang] < 160:
            vals = []
            for f in d.iterdir():
                if f.suffix == ".strings":
                    vals += strings_values(f)
            t = ui_text(vals, rng)
            if t:
                per_lang[lang] += 1
                g = "lproj/" + "/".join(d.parent.parts[-3:])
                add([f"lang.{lang}"], t, f"lproj:{d}", g, "Apple macOS UI strings (internal research use)", "path", g)
        if loct_done < 300:
            for f in d.parent.glob("*.loctable") if d.name == "en.lproj" else []:
                pass
    for base in ["/System/Library/PrivateFrameworks", "/System/Library/CoreServices"]:
        for f in list(Path(base).rglob("*.loctable"))[:3000]:
            lt = loctable_langs(f)
            for code, vals in lt.items():
                lang = LPROJ.get(code)
                if not lang or per_lang[lang] >= 160:
                    continue
                t = ui_text(vals, rng)
                if t:
                    per_lang[lang] += 1
                    g = "loctable/" + "/".join(f.parts[-4:])
                    add([f"lang.{lang}"], t, f"loctable:{f}:{code}", g, "Apple macOS UI strings (internal research use)",
                        "path", g)

    # ---------------------------------------------------------------------------------- select + split
    log(f"{len(cands)} candidates; selecting")
    bysha = {}
    for c in cands:
        c["sha256"] = sha(c["text"])
        bysha.setdefault(c["sha256"], c)
    by_topic = defaultdict(list)
    for c in bysha.values():
        by_topic[c["topics"][0]].append(c)
    items = []
    for t, cs in sorted(by_topic.items()):
        cs.sort(key=lambda c: c["sha256"])
        random.Random(f"{args.seed}:{t}").shuffle(cs)
        for c in cs:
            c["split"] = "fit" if int(hashlib.sha256(c["group"].encode()).hexdigest()[:8], 16) % 2 == 0 else "held"
            c["origin"] = origin_of(c["source"])
        # The split stays the group-hash parity; only the choice within a split is ordered. fit: carried texts
        # first (an atlas fitted on the earlier fit rows stays inside this one). held: natural text first, model-
        # generated last, so a held verdict rests on human text wherever the topic has it.
        cs.sort(key=lambda c: (0, c["sha256"] not in carried) if c["split"] == "fit"
                else (HELD_PREFERENCE[c["origin"]], c["sha256"] not in carried))
        taken = {"fit": 0, "held": 0}
        per_group = Counter()
        for c in cs:
            sp = c["split"]
            quota = args.per_split if sp == "fit" else args.held_per_split
            if t.startswith("lang.") and args.lang_per_split:
                quota = args.lang_per_split if sp == "fit" else args.lang_per_split // 2
            if taken[sp] >= quota or per_group[c["group"]] >= 3:
                continue
            taken[sp] += 1
            per_group[c["group"]] += 1
            items.append(c)
    git_rev = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    corev2_rev = hashlib.sha256(Path(args.corev2).read_bytes()).hexdigest()[:16] if args.corev2 else None
    for i, c in enumerate(items):
        c["id"] = f"tp{i:05d}"
        c["source_revision"] = source_revision(c["source"], corev2_rev, git_rev)
        c["redistribute"] = redistribution(c["license"])
    with open(out / "corpus.jsonl", "w") as f:
        for c in items:
            f.write(json.dumps({k: c[k] for k in ("id", "topics", "split", "text", "source", "origin", "group", "license", "method",
                                                  "sha256", "path_key", "source_revision", "redistribute")}, ensure_ascii=False) + "\n")
    corpus_sha = hashlib.sha256((out / "corpus.jsonl").read_bytes()).hexdigest()
    man = {"schema": "mlx2.amoeba-topic-probes.v1", "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "builder": "scripts/amoeba_topics_corpus.py", "mlx2_source_revision": subprocess.run(
               ["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "taxonomy_version": taxo["version"], "corpus_path": str(out / "corpus.jsonl"), "corpus_sha256": corpus_sha,
           "items": len(items), "per_split_quota": {"fit": args.per_split, "held": args.held_per_split},
           "seed": args.seed, "lang_box_aliases": LANG_BOX_ALIASES,
           "carried_from": {"path": args.carry, "texts": len(carried),
                            "kept": sum(c["sha256"] in carried for c in items),
                            "dropped_mislabels": dict(carry_dropped)} if args.carry else None,
           "rules": {"split": "sha256(group) parity; a group (file/bundle/family) never spans splits",
                     "tested_min_per_split": 16, "window_chars": [MIN_CHARS, MAX_CHARS],
                     "keyword_labels": ">=2 hits and >=2x the runner-up leaf"},
           "downloads_requested": [
               {"name": "FLORES+ devtest", "source": "huggingface openlanguagedata/flores_plus", "license": "CC BY-SA 4.0",
                "size": "~30 MB", "for": "lang.* running text (parallel)", "status": "downloaded 2026-11-14 (approved)"},
               {"name": "arXiv abstracts 2021", "source": "huggingface gfissore/arxiv-abstracts-2021", "license": "CC0 (metadata)",
                "size": "~1 GB parquet, 25 MB sample kept", "for": "sci.*, math.*, cs.* subfields", "status": "downloaded 2026-11-14 (approved); sample kept"}],
           "no_new_downloads": "v5 (2026-11-16) adds only local sources: Cosmopedia v2 shard, core corpus v2",
           "topics": {}}
    agg = defaultdict(lambda: {"fit": 0, "held": 0, "sources": Counter(), "licenses": Counter(), "methods": Counter(),
                               "also": Counter(), "sha256": [], "origin": {"fit": Counter(), "held": Counter()}})
    for c in items:
        a = agg[c["topics"][0]]
        a[c["split"]] += 1
        a["sources"][c["source"].split(":")[0]] += 1
        a["origin"][c["split"]][c["origin"]] += 1
        a["licenses"][c["license"]] += 1
        a["methods"][c["method"]] += 1
        for t in c["topics"][1:]:
            a["also"][t] += 1
        a["sha256"].append(c["sha256"][:16])
    for t, a in sorted(agg.items()):
        man["topics"][t] = {"fit": a["fit"], "held": a["held"], "tested": a["fit"] >= 16 and a["held"] >= 16,
                            "sources": dict(a["sources"]), "licenses": dict(a["licenses"]), "methods": dict(a["methods"]),
                            "also": dict(a["also"]), "origin": {k: dict(v) for k, v in a["origin"].items()},
                            "held_synthetic": a["origin"]["held"]["synthetic"] * 2 > max(1, a["held"]),
                            "held_lab": a["origin"]["held"]["lab"] * 2 > max(1, a["held"]),
                            "text_sha256_16": sorted(a["sha256"])}
    man["summary"] = {"topics_with_items": len(agg), "tested": sum(v["tested"] for v in man["topics"].values()),
                      "by_branch": dict(Counter(t.split(".")[0] for t, v in man["topics"].items() if v["tested"])),
                      "held_synthetic_topics": sorted(t for t, v in man["topics"].items() if v["held_synthetic"]),
                      "held_lab_topics": sorted(t for t, v in man["topics"].items() if v["held_lab"])}
    man["redistribution"] = {"rule": "rows marked no or unresolved are referenced (source, source_revision, sha256) "
                                      "in a published prompt set, never shipped as text",
                             "counts": dict(Counter(c["redistribute"] for c in items)), "by_licence_prefix": REDIST}
    man["origins"] = {"rule": "held text should not be model-generated: held is filled natural > lab > template > "
                              "synthetic; the split itself stays sha256(group) parity, so no row moves between splits",
                      "flag": "held_synthetic / held_lab: more than half the topic's held texts have that origin; "
                              "box verdicts on such topics carry the flag",
                      "classes": {"natural": "external human-written text", "lab": "this lab's repos, largely "
                                  "AI-agent-written", "template": "program-generated (arithmetic)",
                                  "synthetic": "model-generated datasets (Cosmopedia v2, TinyStories)"},
                      "by_prefix": ORIGIN_PREFIX}
    Path(args.manifest).write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    log(f"wrote {len(items)} items, {man['summary']}")


if __name__ == "__main__":
    main()
