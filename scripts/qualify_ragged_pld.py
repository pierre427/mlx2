#!/usr/bin/env python3
"""Bounded ragged batched prompt-lookup (PLD) qualification driver.

Scope: DIRECT-MODEL. One process, one model load, generators driven
directly. It is not HTTP serving, not a serving qualification receipt and
selects no route or default. Timings are not recorded.

Arms (fresh generators and caches each; identical pinned token IDs, greedy
sampling, stop tokens and per-lane output caps):

``ordinary_b1``     every lane alone in a width-1 ``BatchGenerator``. This is
                    the declared primary reference, fixed before any run.
``ordinary_bN``     all lanes in one width-N ``BatchGenerator``.
``pld_per_lane``    ``PromptLookupBatchGenerator`` with ``batched_verify=False``.
``pld_batched``     the same with ``batched_verify=True`` (segmented ragged
                    verify of plain/rotating KV lanes).
``pld_removal``     batched PLD where lane ``--remove-lane`` is removed at a
                    closed boundary (between polls) after it emitted
                    ``--remove-after`` tokens; survivors continue at lower
                    width (B2 -> B1 when N = 2).

Every comparison names its reference. PLD arms are judged against
``ordinary_b1``; the comparison with ``ordinary_bN`` and the
``ordinary_b1`` vs ``ordinary_bN`` geometry comparison are reported
separately and never swapped in after the fact. Per lane the driver records
token IDs and hash, logprob-row storage-bit digests (one per delivered token,
up to ``--logprob-rows``), the final target cache ``state_digest`` (class,
state and ``meta_state``), the committed prompt boundary, and a greedy
ordinary B1 continuation from the final cache. A missing digest is
``unavailable``, never equal.

Evidence (v2): bit parity needs, on each arm independently, exactly
``min(--logprob-rows, delivered tokens)`` complete row digest records and
complete final and continuation digests (``paired_direct_ab.snapshot_refusal``);
equal dictionaries that are incomplete are ``incomparable``, never exact.
Receipts without the v2 lane evidence (every ``ragged-pld.v1`` record) are
incomparable for bit parity; their legacy hash lists are not upgraded.

Coverage (all required for ``pass``): unequal prompt lengths and output
caps, a zero-proposal lane, proposals and a rejected suffix (partial
acceptance), rollback followed by appended tokens, batched verify at width
>= 2, and a closed-boundary removal whose survivor then ran at lower width.
A real workload that never proposes or rolls back is ``coverage_refused``,
not ``pass``. Any PLD lane that differs from its reference is a
``counterexample``.

Real runs need ``--i-own-the-gpu`` and explicit token prompts
(``--prompt-ids`` JSON object with prompts/caps) or the default deterministic
constructor. The September 18 campaign's task constructor is retained in
``qualification/runs/known-limits-20260918/sanity_20x20.py``; this driver
does not reproduce it. September 26's warm-prefix parity receipt instead
retains prompt hashes without the original raw prompts.
``--tiny`` runs a deterministic random Muse-class model on CPU.

Identity (real runs, fail closed). Before any native import (standard
library only): a 40-hex HEAD, a known clean and tracked ``git status`` for
the required files (``IDENTITY_FILES`` plus the selected family's adapter,
config and model files), a sha256 for each, the ``--prompt-ids`` file
sha256, and a metadata/stat artifact manifest in the selected adapter
inspector's own fingerprint recipe (``FAMILIES``). North's recipe also
reads each shard's 8-byte length prefix and bounded safetensors header
(never payload bytes), folds the header sha256 list into its fingerprint
and validates the full pinned checkpoint schema from those headers, as
``north_mini_code.inspect_artifact`` does. Then, in order: the
actual MLX build identity (package, path, metallib sha256, GPU device)
before any mlx2 import; the dispatched adapter class, its source sha256
and every loaded mlx2 module path before the adapter is constructed; the
constructed adapter's identity against the manifest; and the full loaded
module closure (worktree paths, clean and tracked) before any arm. A
failure writes an unexecuted ``refused`` receipt (``arms_executed: []``).
After the arms every identity is taken again; any drift refuses the
receipt while keeping the real results. The manifest hashes metadata
files and shard (name, size, mtime_ns), plus North's raw header bytes: it
is not tensor-payload verification. ``--i-own-the-gpu`` is the caller's assertion; this driver
neither acquires nor checks the GPU lease. ``--tiny`` records the same
source identity without enforcing it (CPU diagnostic, nothing qualified).

  PYTHONPATH=src .venv/bin/python scripts/qualify_ragged_pld.py --tiny --out /tmp/pld.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import struct
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path
from stat import S_ISREG

# Full float32 matmuls unless the caller chose otherwise; must precede the
# first MLX operation (adapters pin the same value for real artifacts).
os.environ.setdefault("MLX_ENABLE_TF32", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SCHEMA = "mlx2.direct-model.ragged-pld.v2"
EVIDENCE = "mlx2.ragged-pld.lane-evidence.v2"
ARMS = ("ordinary_b1", "ordinary_bN", "pld_per_lane", "pld_batched", "pld_removal")
PRIMARY_REFERENCE = "ordinary_b1"
# Shared route closure hashed before any native import: this driver, its
# digest oracle, the ragged PLD scheduler/cache/sampler path, adapter
# dispatch and stop tokens. Other loaded mlx2 modules are bound by the
# loaded-module closure check (clean, tracked, worktree paths).
IDENTITY_FILES = (
    "scripts/qualify_ragged_pld.py",
    "scripts/paired_direct_ab.py",
    "src/mlx2/runtime/pld.py",
    "src/mlx2/runtime/generate.py",
    "src/mlx2/runtime/segmented_rotating_kv.py",
    "src/mlx2/runtime/models/cache.py",
    "src/mlx2/runtime/prompt_lookup.py",
    "src/mlx2/runtime/sample_utils.py",
    "src/mlx2/adapters/registry.py",
    "src/mlx2/contracts.py",
    "src/mlx2/serving.py",
)
# Metadata files each adapter inspector folds into its fingerprint, in its
# order: Muse omits generation_config.json; Qwen3.8 and Qwen3.6 include it.
MUSE_METADATA = ("config.json", "model.safetensors.index.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja")
QWEN_METADATA = MUSE_METADATA + ("generation_config.json",)
# Families whose adapter fingerprint this driver reproduces from metadata
# and shard stats (``headers``: plus bounded safetensors headers).
# ``files[0]`` is the adapter module.
FAMILIES = {
    "muse": {"adapter": "mlx2.adapters.muse_glimmer.MuseGlimmerAdapter",
             "metadata": MUSE_METADATA, "single_shard": True,
             "files": ("src/mlx2/adapters/muse_glimmer.py", "src/mlx2/adapters/muse_glimmer_config.py",
                       "src/mlx2/runtime/models/muse_glimmer.py")},
    "qwen38": {"adapter": "mlx2.adapters.qwen38_27b.Qwen3827BAdapter",
               "metadata": QWEN_METADATA, "single_shard": False,
               "files": ("src/mlx2/adapters/qwen38_27b.py", "src/mlx2/runtime/models/qwen38_27b.py",
                         "src/mlx2/runtime/models/qwen3_5.py")},
    "qwen36": {"adapter": "mlx2.adapters.qwen36_35b.Qwen3635BA3BAdapter",
               "metadata": QWEN_METADATA, "single_shard": False,
               "files": ("src/mlx2/adapters/qwen36_35b.py", "src/mlx2/adapters/qwen38_27b.py",
                         "src/mlx2/runtime/models/qwen36_35b.py", "src/mlx2/runtime/models/qwen3_5.py")},
    # North's ordinary route: adapter, its drafter-policy base class, model
    # body and MoE layers, the shard loader and the tokenizer repair it runs.
    "north": {"adapter": "mlx2.adapters.north_mini_code.NorthMiniCodeAdapter",
              "metadata": QWEN_METADATA, "single_shard": False, "headers": True,
              "files": ("src/mlx2/adapters/north_mini_code.py", "src/mlx2/adapters/external_draft_policy.py",
                        "src/mlx2/runtime/models/cohere2_moe.py", "src/mlx2/runtime/models/switch_layers.py",
                        "src/mlx2/runtime/ubc_evict.py", "src/mlx2/runtime/tokenizer_integrity.py")},
}
FINGERPRINT_SCOPE = ("sha256 over the present metadata files (name + bytes) in the adapter "
                     "inspector's order, then json [name, size, mtime_ns] per sorted shard; "
                     "filesystem metadata and stat only, no shard or tensor bytes read, so this "
                     "is not tensor-content verification")
NORTH_FINGERPRINT_SCOPE = (
    "sha256 over the present metadata files (name + bytes) in north_mini_code.inspect_artifact's "
    "order, then json [name, size, mtime_ns] per sorted shard, then json of the sha256 of each "
    "shard's raw safetensors header in sorted shard order; per shard only the 8-byte length prefix "
    "and the header (at most 64 MiB) are read, never payload bytes, so this is not tensor-payload "
    "content verification: a payload rewritten at the same size with a restored mtime is not seen")
NORTH_HEADER_SCOPE = ("sha256 of each raw safetensors header (bytes 8 .. 8 + length) in sorted shard "
                      "order; the header declares names, dtypes, shapes and offsets, not tensor values")
# Pinned from north_mini_code.inspect_artifact (sha256 7b6cecf3...): the
# topology, norm epsilons, layer order and architecture it requires, its
# header limit and the dtypes its schema allows.
NORTH_CONFIG = {
    "model_type": "cohere2_moe", "hidden_size": 2048, "head_dim": 128, "num_hidden_layers": 49,
    "intermediate_size": 768, "prefix_dense_intermediate_size": 3072, "num_attention_heads": 32,
    "num_key_value_heads": 4, "vocab_size": 262144, "num_experts": 128, "num_experts_per_tok": 8,
    "num_shared_experts": 0, "first_k_dense_replace": 1, "prefix_dense_sliding_window_pattern": 1,
    "expert_selection_fn": "sigmoid", "sliding_window": 4096, "rope_theta": 50000,
    "max_position_embeddings": 500000, "use_parallel_block": True, "use_qk_norm": False,
    "norm_topk_prob": False, "tie_word_embeddings": None, "rms_norm_eps": 1e-06, "layer_norm_eps": 1e-05,
}
NORTH_LAYER_TYPES = ["full_attention" if i % 4 == 0 else "sliding_attention" for i in range(49)]
NORTH_ARCHITECTURES = ["Cohere2MoeForCausalLM"]
NORTH_HEADER_LIMIT = 64 << 20
NORTH_DTYPE_BYTES = {"BF16": 2, "U32": 4}
NORTH_SPECULATIVE_MARKERS = ("mtp.", "eagle", "draft")
# Guarded metadata reads (the initial config.json of every family, every
# North metadata file): larger files refuse before any byte is read.
MAX_METADATA_BYTES = 256 << 20
ORACLE_MODULES = ("scripts.paired_direct_ab",)
# Imported after the adapter is constructed (import-order guard), before
# any arm, so the loaded-module closure covers the route the arms run.
ROUTE_MODULES = ("mlx2.serving", "mlx2.runtime.generate", "mlx2.runtime.pld", "mlx2.runtime.sample_utils")
MAX_WORKLOAD_BYTES = 64 << 20
GPU_OWNERSHIP = ("asserted by the caller with --i-own-the-gpu; not acquired or verified here "
                 "(the external admission wrapper remains mandatory)")
STAGE_PREFLIGHT = "source/artifact preflight (no native import)"
STAGE_BUILD = "MLX build identity (MLX imported; no mlx2 import, adapter or model)"
STAGE_DISPATCH = "dispatch and loaded mlx2 modules (no adapter or model constructed)"
STAGE_ADAPTER = "adapter artifact identity (model loaded; no arm run)"
STAGE_CLOSURE = "loaded module closure (model loaded; no arm run)"
STAGE_POST_RUN = "post-run identity (all arms executed; results kept, bound to nothing)"
# Failures of identity COLLECTION (not of decode) that become refusals.
COLLECTION_ERRORS = (AttributeError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError)
MAX_LANES = 4
MAX_PROMPT_TOKENS = 16384
MAX_OUTPUT_TOKENS = 512
REPEAT_UNIT = "def area(width, height):\n    return width * height\n\n"
DISTINCT_UNIT = "Quartz vexing jumbled fog; my wry pixie bank."


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# The one seam for native imports (``mlx.core`` and mlx2 modules), so tests
# can pin the gate order without importing MLX.
_import = importlib.import_module


def _is_sha256(value) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _is_commit(value) -> bool:
    return type(value) is str and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


class IdentityRefusal(Exception):
    """A native identity gate refused; no arm has run."""

    def __init__(self, stage, refusals, identity=None):
        super().__init__(f"{stage}: {'; '.join(refusals)}"[:500])
        self.stage, self.refusals, self.identity = stage, list(refusals), dict(identity or {})


# ---------------------------------------------------------------- identity

def _git(*args):
    """Stdout of one git command in this worktree, or None on any failure (never '')."""
    try:
        done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def source_identity(names):
    """Raw source identity of ``names`` (repo-relative); git failures stay None, never clean."""
    names = list(names)
    files = {}
    for name in names:
        path = ROOT / name
        try:
            files[name] = _sha(path.read_bytes()) if path.is_file() else None
        except OSError:
            files[name] = None
    head = _git("rev-parse", "--verify", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=all", "--", *names) if names else None
    tracked = _git("ls-files", "--error-unmatch", "--", *names) if names else None
    return {
        "commit": None if head is None else head.strip(),
        "status_known": status is not None,
        "dirty": None if status is None else bool(status.strip()),
        "status_porcelain": None if status is None else status[:2000],
        "tracked": tracked is not None,
        "files": files,
    }


def _bounded_detail(prefix, detail, suffix=""):
    """Keep fixed refusal text intact; spend the 200-character budget only on details."""
    return prefix + detail[:max(0, 200 - len(prefix) - len(suffix))] + suffix


def source_identity_refusals(identity, required):
    """Why a raw source identity cannot bind a native run (empty when it can)."""
    if not isinstance(identity, dict):
        return ["source identity missing"]
    out = []
    if not _is_commit(identity.get("commit")):
        out.append(_bounded_detail("source commit ", repr(identity.get("commit")),
                                   " is not a 40-hex revision (git failed or HEAD unknown)"))
    if identity.get("status_known") is not True or type(identity.get("dirty")) is not bool:
        out.append("worktree status of the required files is unknown (git status failed)")
    elif identity["dirty"]:
        out.append("required files are not clean at HEAD (modified, staged, deleted or untracked)")
    if identity.get("tracked") is not True:
        out.append("required files are not all tracked (git ls-files failed or a file is untracked)")
    files = identity.get("files")
    if not required or not isinstance(files, dict) or set(files) != set(required):
        out.append("hashed file set is not exactly the required set")
    else:
        missing = sorted(str(name) for name, value in files.items() if not _is_sha256(value))
        if missing:
            out.append("no sha256 for required files: " + ", ".join(missing)[:300])
    return out


def identity_changes(label, before, after):
    """Refusals when a raw source identity moved between two points of the run."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return [f"{label} identity before or after the run is missing"]
    out = [f"{label} {key} changed during the run" for key in ("commit", "status_known", "dirty", "tracked")
           if before.get(key) != after.get(key)]
    files_before, files_after = before.get("files") or {}, after.get("files") or {}
    out.extend(f"{label} file {name} changed during the run"
               for name in sorted(set(files_before) | set(files_after))
               if files_before.get(name) != files_after.get(name))
    return out


def artifact_family(config):
    """(family, None) for the adapters this driver can bind, else (None, why).

    Mirrors ``registry._RESOLVERS`` for the supported families; the
    dispatched class is checked again against ``FAMILIES`` after import.
    """
    if not isinstance(config, dict):
        return None, "config.json is not a JSON object"
    architectures = config.get("architectures", [])
    if "dflash_config" in config or (isinstance(architectures, list)
                                     and any("Draft" in str(name) for name in architectures)):
        return None, "a speculative drafter is not a ragged PLD target"
    model_type = config.get("model_type")
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        return None, "text_config is not a JSON object"
    topology = (text.get("num_hidden_layers"), text.get("hidden_size"))
    if model_type in ("muse_glimmer", "muse_glimmer_text"):
        return "muse", None
    if model_type == "qwen3_5" and topology == (64, 5120) and not text.get("num_experts", 0):
        return "qwen38", None
    if model_type == "qwen3_5_moe" and topology != (48, 3072):
        return "qwen36", None
    if model_type == "cohere2_moe":
        why = north_config_refusal(config)
        if why is None:
            return "north", None
        return None, f"cohere2_moe: {why}; the North bounded-header digests recipe binds only that artifact"
    return None, _bounded_detail("model_type ", f"{model_type!r} topology {topology!r}", " has no identity recipe here")


def north_config_refusal(config):
    """Why ``config`` is not the pinned North Mini Code 1.0 (as its inspector checks), or None."""
    if any(config.get(key) != value for key, value in NORTH_CONFIG.items()):
        return "artifact topology does not match North Mini Code 1.0"
    if config.get("layer_types") != NORTH_LAYER_TYPES:
        return "artifact layer order does not match North Mini Code 1.0"
    if config.get("architectures") != NORTH_ARCHITECTURES:
        return "artifact architecture does not match North Mini Code 1.0"
    return None


def _metadata_json(path, name):
    try:
        return json.loads((path / name).read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{name} unreadable: {type(error).__name__}: {error}"[:200]) from None


# ---------------------------------------------------------------- North headers

# Seams for every guarded read (metadata files and shard headers), so tests
# can prove which files are opened and that only the length prefix and the
# bounded header of a shard are ever requested.
_open_fd, _read_fd, _fstat_fd = os.open, os.read, os.fstat
_NOFOLLOW = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)


def _unique_object(pairs, label):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r} in {label}"[:200])
        value[key] = item
    return value


def _strict_json(raw, label, *, text=True):
    """Parse like North's inspector: UTF-8 text (metadata) or raw bytes (headers), duplicate keys refused."""
    try:
        return json.loads(raw.decode("utf-8") if text else raw,
                          object_pairs_hook=lambda pairs: _unique_object(pairs, label))
    except RecursionError:
        raise ValueError(_bounded_detail("", label, " nests too deeply to parse")) from None
    except ValueError as error:
        raise ValueError(_bounded_detail("is not valid JSON: ", f"{error}; {label}")) from None


def _quantized_shapes(name, shape, quant):
    """``{suffix: (dtype, shape)}`` of one quantized module (north_mini_code._quantized_shapes)."""
    override = quant.get(name, quant)
    if not isinstance(override, dict):
        raise ValueError(f"Invalid North quantization override: {name}")  # noqa: TRY004 - a refusal
    group_size, bits = override.get("group_size"), override.get("bits")
    # group_size <= 0 would divide by zero or give negative shapes there.
    if type(group_size) is not int or type(bits) is not int or bits not in {4, 8} or group_size <= 0:
        raise ValueError(f"Invalid North quantization parameters: {name}")
    if shape[-1] % group_size or (shape[-1] * bits) % 32:
        raise ValueError("North quantization dimensions are not integral")
    groups = [*shape[:-1], shape[-1] // group_size]
    return {"weight": ("U32", [*shape[:-1], shape[-1] * bits // 32]),
            "scales": ("BF16", groups), "biases": ("BF16", groups)}


def north_expected_headers(config):
    """``{tensor: (dtype, shape)}`` North's pinned config and quantization imply (_expected_weight_headers)."""
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"]) * int(config["head_dim"])
    kv_heads = int(config["num_key_value_heads"]) * int(config["head_dim"])
    experts, intermediate = int(config["num_experts"]), int(config["intermediate_size"])
    dense = int(config["prefix_dense_intermediate_size"])
    quant = config.get("quantization", config.get("quantization_config"))
    if not isinstance(quant, dict):
        raise ValueError("North artifact must declare quantization metadata")  # noqa: TRY004 - a refusal
    expected = {}

    def add(name, shape):
        for suffix, entry in _quantized_shapes(name, shape, quant).items():
            expected[f"{name}.{suffix}"] = entry

    add("model.embed_tokens", [int(config["vocab_size"]), hidden])
    for index in range(int(config["num_hidden_layers"])):
        prefix = f"model.layers.{index}."
        expected[prefix + "input_layernorm.weight"] = ("BF16", [hidden])
        for projection, shape in (("q_proj", [heads, hidden]), ("k_proj", [kv_heads, hidden]),
                                  ("v_proj", [kv_heads, hidden]), ("o_proj", [hidden, heads])):
            add(prefix + "self_attn." + projection, shape)
        if index < int(config["first_k_dense_replace"]):
            for projection, shape in (("gate_proj", [dense, hidden]), ("up_proj", [dense, hidden]),
                                      ("down_proj", [hidden, dense])):
                add(prefix + "mlp." + projection, shape)
        else:
            add(prefix + "mlp.gate", [experts, hidden])
            for projection, shape in (("gate_proj", [experts, intermediate, hidden]),
                                      ("up_proj", [experts, intermediate, hidden]),
                                      ("down_proj", [experts, hidden, intermediate])):
                add(prefix + "mlp.switch_mlp." + projection, shape)
    expected["model.norm.weight"] = ("BF16", [hidden])
    return expected


def _stat_key(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _read_exact(fd, count, short):
    chunks, remaining = [], count
    while remaining:
        chunk = _read_fd(fd, min(remaining, 1 << 20))
        if not chunk:
            raise ValueError(short[:200])
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_metadata(item, name):
    """Bytes of one metadata file, read through one descriptor; refusals come before any byte.

    The name must be a regular file itself (a symlink refuses) with one
    hard link and at most ``MAX_METADATA_BYTES``, so it cannot redirect a
    read into shard payload. The path before the open, the descriptor at
    open and after the read, and the path after the read must be the same
    file (device, inode, size, mtime_ns); the read stops at the stat size.
    """
    info = os.lstat(item)
    if not S_ISREG(info.st_mode):
        raise ValueError(f"metadata {name} is not a regular file (symlinks are refused)")
    if info.st_nlink != 1:
        raise ValueError(f"metadata {name} has {info.st_nlink} hard links")
    if info.st_size > MAX_METADATA_BYTES:
        raise ValueError(f"metadata {name} exceeds {MAX_METADATA_BYTES} bytes")
    fd = _open_fd(item, _NOFOLLOW)
    try:
        if _stat_key(_fstat_fd(fd)) != _stat_key(info):
            raise ValueError(f"metadata {name} changed between stat and open")
        data = _read_exact(fd, info.st_size, f"metadata {name} shrank while read")
        after = _fstat_fd(fd)
    finally:
        os.close(fd)
    if _stat_key(after) != _stat_key(info) or _stat_key(os.lstat(item)) != _stat_key(info):
        raise ValueError(f"metadata {name} changed or was replaced while read")
    return data


def _config_json(path):
    """The initial config.json (every family, before dispatch) through the guarded reader."""
    try:
        return json.loads(_read_metadata(path / "config.json", "config.json").decode("utf-8"))
    except RecursionError:
        raise ValueError("config.json unreadable: nests too deeply to parse") from None
    except (OSError, ValueError) as error:
        raise ValueError(f"config.json unreadable: {type(error).__name__}: {error}"[:200]) from None


def north_shard_header(item, label):
    """(stat, raw header) of one shard; reads its 8-byte length and the bounded header only.

    The path stat before the open, the descriptor at open and after the
    read, and the path stat after the read must be the same file (device,
    inode, size, mtime_ns), so a shard swapped, replaced at its path or
    rewritten with a new mtime while its header is read refuses.
    """
    before = os.stat(item)
    if not S_ISREG(before.st_mode):
        raise ValueError(_bounded_detail("weight shard ", repr(label), " is not a regular file"))
    fd = _open_fd(item, _NOFOLLOW)
    try:
        if _stat_key(_fstat_fd(fd)) != _stat_key(before):
            raise ValueError(_bounded_detail("weight shard ", repr(label), " changed between stat and open"))
        short = f"Truncated safetensors file: {label}"
        length = struct.unpack("<Q", _read_exact(fd, 8, short))[0]
        if not 0 < length <= min(NORTH_HEADER_LIMIT, before.st_size - 8):
            raise ValueError(f"Invalid safetensors header length: {label}"[:200])
        raw = _read_exact(fd, length, short)
        after = _fstat_fd(fd)
    finally:
        os.close(fd)
    if _stat_key(after) != _stat_key(before):
        raise ValueError(_bounded_detail("weight shard ", repr(label), " changed while its header was read"))
    if _stat_key(os.stat(item)) != _stat_key(before):
        raise ValueError(_bounded_detail("weight shard ", repr(label), " was replaced at its path while its header was read"))
    return before, raw


def _check_north_header(header, name, weight_map, observed, payload_size):
    """Validate one shard header into ``observed`` (north_mini_code._validate_weight_headers)."""
    ranges = []
    for tensor, record in header.items():
        if tensor == "__metadata__":
            continue
        if tensor in observed:
            raise ValueError(f"Duplicate North tensor across shards: {tensor}"[:200])
        if not isinstance(record, dict) or set(record) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Invalid North tensor metadata: {tensor}"[:200])
        dtype, shape, offsets = record["dtype"], record["shape"], record["data_offsets"]
        if (type(dtype) is not str or dtype not in NORTH_DTYPE_BYTES or not isinstance(shape, list)
                or not all(type(value) is int and value >= 0 for value in shape)):
            raise ValueError(f"Invalid North tensor dtype/shape: {tensor}"[:200])
        if not (isinstance(offsets, list) and len(offsets) == 2 and all(type(value) is int for value in offsets)
                and 0 <= offsets[0] <= offsets[1] <= payload_size):
            raise ValueError(f"Invalid North tensor offsets: {tensor}"[:200])
        if offsets[1] - offsets[0] != math.prod(shape) * NORTH_DTYPE_BYTES[dtype]:
            raise ValueError(f"Invalid North tensor byte size: {tensor}"[:200])
        if weight_map.get(tensor) != name:
            raise ValueError(f"North index/shard mismatch: {tensor}"[:200])
        observed[tensor] = (dtype, shape)
        ranges.append((offsets[0], offsets[1], tensor))
    ranges.sort()
    for previous, current in pairwise(ranges):
        if previous[1] > current[0]:
            raise ValueError(f"Overlapping North tensor payloads: {previous[2]}, {current[2]}"[:200])


def _schema_mismatch(observed, expected):
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    wrong = sorted(name for name in set(expected) & set(observed) if expected[name] != observed[name])
    return ("North checkpoint schema mismatch "
            + ", ".join(f"{label}={len(names)} {names[:3]}" for label, names in
                        (("missing", missing), ("extra", extra), ("wrong", wrong))))[:300]


def _north_collect(path):
    """One collection of North's identity: metadata bytes, index, shard stats, raw headers, schema."""
    raw = {}
    for name in FAMILIES["north"]["metadata"]:
        item = path / name
        if not os.path.lexists(item):
            continue
        raw[name] = _read_metadata(item, name)
    for name in ("config.json", "model.safetensors.index.json"):
        if name not in raw:
            raise ValueError(f"North requires {name}")
    # The parsed config and index are the bytes hashed below.
    config = _strict_json(raw["config.json"], "config.json")
    if not isinstance(config, dict):
        raise ValueError("config.json is not a JSON object")  # noqa: TRY004 - a refusal
    why = north_config_refusal(config)
    if why is not None:
        raise ValueError(why)
    index = _strict_json(raw["model.safetensors.index.json"], "model.safetensors.index.json")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("North artifact must have a nonempty weight index")
    if any(type(name) is not str for name in weight_map.values()):
        raise ValueError("weight index maps a tensor to a non-string shard")
    if any(marker in key.lower() for key in weight_map for marker in NORTH_SPECULATIVE_MARKERS):
        raise ValueError("embedded speculative tensors are not a North target artifact")
    if "model.embed_tokens.weight" not in weight_map or "lm_head.weight" in weight_map:
        raise ValueError("North artifact must use its tied embedding output head")
    names, resolved, inodes = sorted(set(weight_map.values())), {}, set()
    for name in names:
        item = (path / name).resolve()
        if (not name or Path(name).is_absolute() or ".." in Path(name).parts
                or not item.is_relative_to(path) or item.suffix != ".safetensors"):
            raise ValueError(_bounded_detail("weight shard ", repr(name), " is not a local .safetensors file in the artifact"))
        if not item.is_file():
            raise ValueError(f"missing weight shard {name!r}"[:200])
        info = item.stat()
        if item in resolved.values() or (info.st_dev, info.st_ino) in inodes:
            raise ValueError(_bounded_detail("weight shard ", repr(name), " is another index name for the same file"))
        resolved[name] = item
        inodes.add((info.st_dev, info.st_ino))
    expected = north_expected_headers(config)
    observed, records, headers, files = {}, [], [], []
    for name in names:
        info, header_raw = north_shard_header(resolved[name], name)
        header = _strict_json(header_raw, f"safetensors header {name}", text=False)
        if not isinstance(header, dict):
            raise ValueError(f"Invalid safetensors header object: {name}"[:200])  # noqa: TRY004 - a refusal
        _check_north_header(header, name, weight_map, observed, info.st_size - 8 - len(header_raw))
        records.append((name, info.st_size, info.st_mtime_ns))
        headers.append(_sha(header_raw))
        files.append([name, info.st_dev, info.st_ino])
    if set(weight_map) != set(observed):
        raise ValueError("North weight index does not match shard headers")
    if observed != expected:
        raise ValueError(_schema_mismatch(observed, expected))
    digest = hashlib.sha256()
    for name, data in raw.items():  # metadata order, absent files skipped
        digest.update(name.encode())
        digest.update(data)
    for record in records:
        digest.update(json.dumps(record).encode())
    digest.update(json.dumps(headers).encode())
    return {"fingerprint": digest.hexdigest(), "metadata_sha256": {name: _sha(data) for name, data in raw.items()},
            "shards": [list(record) for record in records], "header_sha256": headers,
            "shard_file_ids": files, "schema": {"tensors": len(observed), "shards": len(names)}}


def north_artifact_manifest(path):
    """North's identity, collected twice; any difference (a change mid-preflight) refuses.

    Each collection records metadata hashes, shard stats, shard (device,
    inode) and header hashes; files replaced between the collections with
    identical bytes and stats are seen through their inode.
    """
    try:
        first = _north_collect(path)
        second = _north_collect(path)
    except (KeyError, OverflowError, TypeError) as error:
        raise ValueError(f"North artifact cannot be bound: {type(error).__name__}: {error}"[:200]) from None
    if first != second:
        raise ValueError("North metadata, index, shard stats or headers changed during the preflight")
    return {**first, "fingerprint_scope": NORTH_FINGERPRINT_SCOPE, "header_scope": NORTH_HEADER_SCOPE}


def artifact_manifest(model_path):
    """Metadata/stat identity in the selected adapter inspector's recipe (stdlib).

    Reads config (guarded: a regular, single-link, bounded file, never a
    symlink), index and the recipe's metadata files; shards are only
    resolved and stat'ed, never opened (North: every metadata read is
    guarded and only each shard's length prefix and bounded header are
    read). Raises ``ValueError`` (or ``OSError``) when the artifact cannot
    be bound.
    """
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"artifact {str(path)!r} is not a directory")
    config = _config_json(path)
    family, why = artifact_family(config)
    if family is None:
        raise ValueError(why)
    recipe = FAMILIES[family]
    if recipe.get("headers"):
        return {"path": str(path), "family": family, "adapter": recipe["adapter"],
                "model_type": config.get("model_type"), **north_artifact_manifest(path)}
    if (path / "model.safetensors.index.json").exists():
        index = _metadata_json(path, "model.safetensors.index.json")
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("weight index has no nonempty weight_map")
        if any(type(name) is not str for name in weight_map.values()):
            raise ValueError("weight index maps a tensor to a non-string shard")
        names = sorted(set(weight_map.values()))
    elif recipe["single_shard"]:
        names = ["model.safetensors"]
    else:
        raise ValueError(f"{family} requires model.safetensors.index.json")
    digest = hashlib.sha256()
    metadata = {}
    for name in recipe["metadata"]:
        item = path / name
        if not item.exists():
            continue
        if not item.is_file():
            raise ValueError(f"metadata {name} is not a regular file")
        data = item.read_bytes()
        metadata[name] = _sha(data)
        digest.update(name.encode())
        digest.update(data)
    shards = []
    for name in names:
        item = (path / name).resolve()
        if (not name or Path(name).is_absolute() or ".." in Path(name).parts
                or not item.is_relative_to(path) or item.suffix != ".safetensors"):
            raise ValueError(_bounded_detail("weight shard ", repr(name), " is not a local .safetensors file in the artifact"))
        if not item.is_file():
            raise ValueError(f"missing weight shard {name!r}"[:200])
        stat = item.stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        shards.append(list(record))
        digest.update(json.dumps(record).encode())
    return {"path": str(path), "family": family, "adapter": recipe["adapter"],
            "model_type": config.get("model_type"), "fingerprint": digest.hexdigest(),
            "fingerprint_scope": FINGERPRINT_SCOPE, "metadata_sha256": metadata, "shards": shards}


def workload_identity(path):
    """(identity, raw bytes, refusals) of the pinned ``--prompt-ids`` file."""
    try:
        item = Path(path).expanduser().resolve()
        if not item.is_file() or item.stat().st_size > MAX_WORKLOAD_BYTES:
            return None, None, [f"--prompt-ids {str(item)!r} is not a file of at most {MAX_WORKLOAD_BYTES} bytes"]
        raw = item.read_bytes()
    except OSError as error:
        return None, None, [f"--prompt-ids unreadable: {type(error).__name__}: {error}"[:200]]
    return {"path": str(item), "bytes": len(raw), "sha256": _sha(raw)}, raw, []


def preflight(args):
    """Standard-library identity taken before any native import.

    Real runs refuse on any entry of ``refusals``; ``--tiny`` records the
    same source identity without enforcing it.
    """
    refusals, manifest, required = [], None, IDENTITY_FILES
    if not args.tiny:
        try:
            manifest = artifact_manifest(args.model)
        except (OSError, ValueError) as error:
            refusals.append(f"artifact: {error}"[:300])
        else:
            required += tuple(name for name in FAMILIES[manifest["family"]]["files"] if name not in required)
    source = source_identity(required)
    refusals.extend(source_identity_refusals(source, required))
    workload, raw = None, None
    if args.prompt_ids:
        workload, raw, problems = workload_identity(args.prompt_ids)
        refusals.extend(problems)
    return {"enforced": not args.tiny, "required_files": list(required), "source": source,
            "artifact": manifest, "workload": workload, "workload_raw": raw, "refusals": refusals}


def public_preflight(gate):
    return {key: value for key, value in gate.items() if key != "workload_raw"}


def build_identity(mx):
    """``paired_direct_ab.mlx_identity`` of the imported MLX (version, package, path, metallib)."""
    from scripts.paired_direct_ab import mlx_identity

    return mlx_identity(mx)


def build_identity_refusals(build):
    """Why an MLX build identity cannot bind a native GPU run (empty when it can)."""
    if not isinstance(build, dict):
        return ["MLX build identity missing"]
    out = [f"MLX build identity has no {key}" for key in ("version", "package", "device")
           if type(build.get(key)) is not str or not build[key].strip()]
    path = build.get("path")
    if type(path) is not str or not os.path.isabs(path) or not os.path.isdir(path):
        out.append(_bounded_detail("MLX core path ", repr(path), " is not an existing absolute directory"))
    if not _is_sha256(build.get("metallib_sha256")):
        out.append("MLX build identity has no metallib sha256")
    if type(build.get("device")) is str and "gpu" not in build["device"].lower():
        out.append(_bounded_detail("MLX default device ", repr(build["device"]), " is not the GPU"))
    return out


def collect_build_identity(mx):
    """(build identity or None, refusals); a collector failure is a refusal, never a crash."""
    try:
        build = build_identity(mx)
    except COLLECTION_ERRORS as error:
        return None, [f"MLX build identity could not be collected: {type(error).__name__}: {error}"[:300]]
    return build, build_identity_refusals(build)


def loaded_module_files(modules=None):
    """``{name: __file__}`` of every loaded mlx2 module and the digest oracle."""
    modules = sys.modules if modules is None else modules
    return {name: getattr(module, "__file__", None) for name, module in list(modules.items())
            if module is not None and (name == "mlx2" or name.startswith("mlx2.") or name in ORACLE_MODULES)}


def _worktree_file(name, file):
    """The resolved file when module ``name`` comes from this worktree, else None."""
    root = ROOT / "scripts" if name in ORACLE_MODULES else ROOT / "src" / "mlx2"
    if not isinstance(file, str) or not os.path.isabs(file):
        return None
    path = Path(file).resolve()
    return path if path.suffix == ".py" and path.is_file() and root in path.parents else None


def module_path_refusals(files):
    """Every loaded mlx2 module (and the oracle) must be a source file of this worktree."""
    out = [] if any(name == "mlx2" or name.startswith("mlx2.") for name in files) else ["no mlx2 module is loaded"]
    out.extend(_bounded_detail(f"not this worktree: {name} imported from ", repr(file))
               for name, file in sorted(files.items()) if _worktree_file(name, file) is None)
    return out


def module_closure(files):
    """Loaded modules with a git-bound source identity of their files."""
    paths = (_worktree_file(name, file) for name, file in files.items())
    names = sorted({str(path.relative_to(ROOT)) for path in paths if path is not None})
    return {"modules": dict(sorted(files.items())), "required": names,
            "source": source_identity(names) if names else None}


def module_closure_refusals(closure):
    if not isinstance(closure, dict) or not isinstance(closure.get("modules"), dict):
        return ["loaded module closure missing"]
    out = module_path_refusals(closure["modules"])
    out.extend(f"loaded modules: {p}" for p in source_identity_refusals(closure.get("source"), closure.get("required")))
    return out


def closure_drift_refusals(closure, source):
    """Loaded files that the preflight also hashed must still carry the preflight hash."""
    now = (((closure or {}).get("source") or {}).get("files")) or {}
    pre = ((source or {}).get("files")) or {}
    return [f"loaded {name} differs from its preflight hash" for name in sorted(set(now) & set(pre))
            if now[name] != pre[name]]


def adapter_source_refusals(adapter_name, adapter_file, manifest, source):
    """The dispatched class must be the family's adapter, from this worktree, at the preflight hash."""
    family = FAMILIES[manifest["family"]]
    out = []
    if adapter_name != family["adapter"]:
        out.append(_bounded_detail("dispatch selected ", str(adapter_name), f", the preflight recipe binds {family['adapter']}"))
    path = _worktree_file(family["adapter"].rsplit(".", 1)[0], adapter_file)
    if path is None or str(path.relative_to(ROOT)) != family["files"][0]:
        out.append(_bounded_detail(f"not this worktree's {family['files'][0]}: adapter module file ", repr(adapter_file)))
        return out
    try:
        now = _sha(path.read_bytes())
    except OSError:
        now = None
    if not _is_sha256(now) or now != ((source or {}).get("files") or {}).get(family["files"][0]):
        out.append("adapter source sha256 differs from the preflight hash")
    return out


def adapter_identity_snapshot(adapter):
    """Path, fingerprint, shard records (and North's header digests) of a constructed adapter."""
    identity = getattr(adapter, "identity", None)
    if not isinstance(identity, dict):
        return None
    try:
        files = [list(record) for record in identity.get("files")]
    except TypeError:
        files = None
    snapshot = {"path": identity.get("path"), "fingerprint": identity.get("fingerprint"), "files": files}
    if "header_sha256" in identity:
        headers = identity["header_sha256"]
        snapshot["header_sha256"] = list(headers) if isinstance(headers, (list, tuple)) else headers
    return snapshot


def adapter_identity_refusals(adapter, manifest):
    """The constructed adapter must report exactly the preflight artifact, with no drafter bound."""
    snapshot = adapter_identity_snapshot(adapter)
    if snapshot is None:
        return ["adapter has no identity mapping"]
    out = []
    if not _is_sha256(snapshot["fingerprint"]) or snapshot["fingerprint"] != manifest["fingerprint"]:
        out.append("adapter artifact fingerprint differs from the preflight manifest")
    if snapshot["path"] != manifest["path"]:
        out.append("adapter artifact path differs from the preflight manifest")
    if snapshot["files"] != manifest["shards"]:
        out.append("adapter shard records differ from the preflight manifest")
    if "header_sha256" in manifest and snapshot.get("header_sha256") != manifest["header_sha256"]:
        out.append("adapter safetensors header digests differ from the preflight manifest")
    identity = adapter.identity
    if (getattr(adapter, "draft_model", None) is not None
            or any(key in identity for key in ("draft_fingerprint", "target_fingerprint", "draft_revision"))):
        out.append("an external drafter is bound; this ordinary PLD diagnostic binds the target alone")
    return out


def post_run_identity(args, gate, driver):
    """(identity after the arms, refusals): nothing that bound the run may have moved."""
    required = gate["required_files"]
    after = {"source": source_identity(required)}
    refusals = [f"after the arms: {p}" for p in source_identity_refusals(after["source"], required)]
    refusals.extend(identity_changes("source", gate["source"], after["source"]))
    if gate["workload"] is not None:
        after["workload"], _raw, problems = workload_identity(args.prompt_ids)
        refusals.extend(f"after the arms: {p}" for p in problems)
        if (after["workload"] or {}).get("sha256") != gate["workload"]["sha256"]:
            refusals.append("the --prompt-ids file changed during the run")
    if args.tiny:
        return after, refusals
    try:
        after["artifact"] = artifact_manifest(args.model)
    except (OSError, ValueError) as error:
        after["artifact"] = None
        refusals.append(f"after the arms: artifact: {error}"[:300])
    else:
        refusals.extend(f"artifact {key} changed during the run"
                        for key in ("path", "family", "fingerprint", "metadata_sha256", "shards", "header_sha256",
                                    "shard_file_ids")
                        if after["artifact"].get(key) != (gate["artifact"] or {}).get(key))
    native = getattr(driver, "native_identity", None) or {}
    after["mlx"], problems = collect_build_identity(driver.mx)
    refusals.extend(f"after the arms: {p}" for p in problems)
    if after["mlx"] != native.get("mlx"):
        refusals.append("MLX build identity changed during the run")
    after["modules"] = module_closure(loaded_module_files())
    refusals.extend(f"after the arms: {p}" for p in module_closure_refusals(after["modules"]))
    before = ((native.get("modules_before_arms") or {}).get("source") or {}).get("files")
    now = (after["modules"].get("source") or {}).get("files") or {}
    if not before:
        refusals.append("no pre-arm module closure was recorded")
    else:
        refusals.extend(f"loaded module file {name} changed or unloaded during the run"
                        for name in sorted(before) if now.get(name) != before[name])
    after["adapter_identity"] = adapter_identity_snapshot(getattr(driver, "adapter", None))
    if after["adapter_identity"] is None or after["adapter_identity"] != native.get("adapter_identity"):
        refusals.append("adapter artifact identity changed during the run")
    return after, refusals


def checked_post_run_identity(args, gate, driver):
    """``post_run_identity``, with a collection failure refused instead of discarding the results."""
    try:
        return post_run_identity(args, gate, driver)
    except COLLECTION_ERRORS as error:
        return ({"error": f"{type(error).__name__}: {error}"[:300]},
                [f"post-run identity could not be collected: {type(error).__name__}: {error}"[:300]])


# ---------------------------------------------------------------- models

def tiny_model():
    import mlx.core as mx
    from mlx2.adapters.muse_glimmer_config import ModelArgs
    from mlx2.runtime.models.muse_glimmer import Model

    mx.random.seed(8)
    model = Model(ModelArgs(
        hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=48, sliding_window=8, max_position_embeddings=1024,
    ))
    model.eval()
    mx.eval(model.parameters())
    return model


def tiny_prompts(lanes=3):
    """Fixed tiny fixtures; each reaches compute width == lanes on CPU.

    Unequal lengths and caps, repeated-structure lanes and one all-distinct
    lane. The width-3 fixture (the default) uses cap 6 on its distinct lane: with
    cap 2 that lane finished before the cohort verified at width 3.
    """
    repeat = [3, 4, 5, 6, 7] * 4 + [3, 4]
    looping = [9, 10, 11, 12] * 5 + [9]
    distinct = [20, 21, 22, 23, 24, 25, 26]
    triple = [13, 14, 15] * 6 + [13]
    fixtures = {
        2: ([looping, distinct], [28, 12]),
        3: ([repeat, looping, distinct], [24, 28, 6]),
        4: ([repeat, looping, distinct, triple], [24, 28, 8, 20]),
    }
    return fixtures[lanes]


def constructed_prompts(tokenizer, lanes):
    """Deterministic token construction for real artifacts (not old prompts)."""
    def encode(text):
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))

    repeat = encode(REPEAT_UNIT * 6)
    distinct = encode(DISTINCT_UNIT)
    prompts = [repeat, repeat[: max(8, len(repeat) * 2 // 3)], distinct, repeat[: len(repeat) // 2]]
    caps = [64, 48, 3, 40]
    return prompts[:lanes], caps[:lanes]


def load_workload(path, raw=None):
    """(prompts, caps, receipt, source) from a pinned token file; fail closed.

    ``raw`` is the preflight's bytes, so the hashed file is the one used.
    """
    raw = Path(path).read_bytes() if raw is None else raw
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) - {"prompts", "max_tokens", "receipt"} \
            or not isinstance(data.get("prompts"), list) or not isinstance(data.get("max_tokens"), list):
        raise SystemExit("refused: token file needs prompts and max_tokens lists (optional receipt)")
    return data["prompts"], data["max_tokens"], data.get("receipt"), f"explicit token file sha256 {_sha(raw)}"


# ---------------------------------------------------------------- run one arm

class Driver:
    def __init__(self, args, gate=None):
        """Load the model; real runs gate every identity before the first arm.

        Order is part of the contract: MLX build identity before any mlx2
        import; dispatch, adapter source and loaded mlx2 paths before the
        adapter is constructed; then the adapter identity and the loaded
        module closure. Each failure raises ``IdentityRefusal``.
        """
        if not args.tiny and (not isinstance(gate, dict) or gate.get("refusals") or not gate.get("artifact")):
            raise IdentityRefusal(STAGE_PREFLIGHT, ["no clean preflight was passed to the native driver"])
        mx = _import("mlx.core")

        self.args, self.mx = args, mx
        self.stops = ()
        self.identity = {}
        self.native_identity = None
        self.workload_receipt = None
        self.lane_policies = None
        if args.tiny:
            mx.set_default_device(mx.cpu)
            self.model = tiny_model()
            self.prompts, self.caps = tiny_prompts(args.lanes)
            self.identity = {"model": "tiny-random-muse", "fingerprint": None}
            self.followup = [5, 6, 7]
            return
        manifest = gate["artifact"]
        build, refusals = collect_build_identity(mx)
        if refusals:
            raise IdentityRefusal(STAGE_BUILD, refusals, {"mlx": build})
        registry = _import("mlx2.adapters.registry")
        try:
            cls = registry.resolve_adapter(args.model, mtp=False, qualification_mode=True)
        except (ImportError, OSError, ValueError, TypeError, KeyError) as error:
            raise IdentityRefusal(STAGE_DISPATCH, [f"adapter dispatch failed: {type(error).__name__}: {error}"[:300]],
                                  {"mlx": build}) from None
        adapter_name = f"{cls.__module__}.{cls.__qualname__}"
        loaded = loaded_module_files()
        before_adapter = module_closure(loaded)
        refusals = (adapter_source_refusals(adapter_name, loaded.get(cls.__module__), manifest, gate["source"])
                    + module_closure_refusals(before_adapter)
                    + closure_drift_refusals(before_adapter, gate["source"]))
        if refusals:
            raise IdentityRefusal(STAGE_DISPATCH, refusals, {"mlx": build, "adapter": adapter_name,
                                                             "modules_before_adapter": before_adapter})
        # Constructed before any runtime module is imported (import-order guard).
        self.adapter = cls(args.model)
        adapter_identity = adapter_identity_snapshot(self.adapter)
        refusals = adapter_identity_refusals(self.adapter, manifest)
        if refusals:
            raise IdentityRefusal(STAGE_ADAPTER, refusals, {"mlx": build, "adapter": adapter_name,
                                                            "adapter_identity": adapter_identity})
        serving = _import("mlx2.serving")
        for name in ROUTE_MODULES:
            _import(name)
        before_arms = module_closure(loaded_module_files())
        refusals = module_closure_refusals(before_arms) + closure_drift_refusals(before_arms, gate["source"])
        if refusals:
            raise IdentityRefusal(STAGE_CLOSURE, refusals, {"mlx": build, "adapter": adapter_name,
                                                            "modules_before_arms": before_arms})
        self.native_identity = {"mlx": build, "adapter": adapter_name, "adapter_identity": adapter_identity,
                                "modules_before_adapter": before_adapter, "modules_before_arms": before_arms}
        self.model = self.adapter.model
        self.stops = () if args.ignore_eos else serving.generation_stop_token_ids(self.adapter)
        if args.prompt_ids:
            self.prompts, self.caps, self.workload_receipt, prompt_source = load_workload(
                args.prompt_ids, raw=gate["workload_raw"])
        else:
            self.prompts, self.caps = constructed_prompts(self.adapter.tokenizer, args.lanes)
            prompt_source = "deterministic constructor (not the 2026-09-18 campaign prompts)"
        self.followup = list(self.prompts[0][:3])
        self.identity = {
            "model": str(args.model),
            "adapter": adapter_name,
            "adapter_sha256": gate["source"]["files"][FAMILIES[manifest["family"]]["files"][0]],
            "fingerprint": adapter_identity["fingerprint"],
            "environment": dict(getattr(self.adapter, "environment", {}) or {}),
            "prompt_source": prompt_source,
        }

    def check_lane_policies(self):
        """Validate per-lane PLD policy overrides; fail closed."""
        policies = self.args.lane_policies
        if policies is None:
            self.lane_policies = None
            return
        from mlx2.runtime.pld import PromptLookupBatchGenerator

        if not isinstance(policies, list) or len(policies) != len(self.prompts):
            raise SystemExit("refused: --lane-policies needs one object per lane")
        validated = []
        for index, policy in enumerate(policies):
            if not isinstance(policy, dict):
                raise SystemExit(f"refused: lane {index} policy must be an object")
            if "batched_verify" in policy:
                raise SystemExit("refused: batched_verify is fixed per arm, not per lane")
            unknown = set(policy) - PromptLookupBatchGenerator.POLICY_KEYS
            if unknown:
                raise SystemExit(f"refused: lane {index} unknown policy keys {sorted(unknown)}")
            try:
                # The generator's own lane rule, before any arm runs: rounds
                # run at the arm's num_draft, so a lane may not change it.
                PromptLookupBatchGenerator.validate_lane_policy(
                    PromptLookupBatchGenerator.validate_policy(self.args.pld_policy), policy)
            except ValueError as error:
                raise SystemExit(f"refused: lane {index} policy: {error}") from None
            validated.append(dict(policy))
        self.lane_policies = validated

    def check_geometry(self):
        if not 2 <= len(self.prompts) <= MAX_LANES or len(self.caps) != len(self.prompts):
            raise SystemExit(f"refused: need 2..{MAX_LANES} lanes with one cap each")
        for prompt, cap in zip(self.prompts, self.caps):
            if not 2 <= len(prompt) <= MAX_PROMPT_TOKENS or not 1 <= cap <= MAX_OUTPUT_TOKENS:
                raise SystemExit("refused: prompt 2..16384 tokens and cap 1..512 per lane")
            if any(type(t) is not int or t < 0 for t in prompt):
                raise SystemExit("refused: prompts must be token id lists")
        if not 0 <= self.args.remove_lane < len(self.prompts):
            raise SystemExit("refused: --remove-lane out of range")
        self.check_lane_policies()

    def generator(self, arm, width):
        stops = [[t] for t in self.stops]
        if arm.startswith("ordinary"):
            from mlx2.runtime.generate import BatchGenerator

            return BatchGenerator(self.model, completion_batch_size=width, prefill_batch_size=1,
                                  prefill_step_size=self.args.prefill_step, stop_tokens=stops)
        from mlx2.runtime.pld import PromptLookupBatchGenerator

        policy = {**self.args.pld_policy, "batched_verify": arm != "pld_per_lane"}
        return PromptLookupBatchGenerator(self.model, completion_batch_size=width,
                                          prefill_step_size=self.args.prefill_step,
                                          stop_tokens=stops, prompt_lookup=policy)

    def run(self, arm, lanes, *, caches=None, all_tokens=None, caps=None, remove=None, prompts=None):
        """Drive one generator over ``lanes`` (indices); returns per-lane records."""
        from mlx2.runtime.sample_utils import LaneRNG
        from scripts.paired_direct_ab import state_digest

        mx, args = self.mx, self.args  # noqa: F841 -- body pinned to the baseline (tests/test_ragged_pld_north_identity_cpu.py)
        caps = caps or [self.caps[i] for i in lanes]
        prompts = prompts or [list(self.prompts[i]) for i in lanes]
        gen = self.generator(arm, len(lanes))
        records = {i: {"tokens": [], "logprob_rows": [], "widths": set(), "receipts": [],
                       "finish_reason": None, "final": None, "boundary": None} for i in lanes}
        failures, polls, deadline = [], 0, time.monotonic() + args.time_limit_s
        limit = sum(caps) * 4 + sum(len(p) for p in prompts) // max(1, args.prefill_step) + 64
        removed = None
        try:
            insert = {"max_tokens": caps}
            if caches is not None:
                insert.update(caches=caches, all_tokens=all_tokens)
            if arm.startswith("ordinary"):
                insert["lane_rngs"] = [LaneRNG(args.seed + i) for i in lanes]
            elif self.lane_policies is not None and caches is None:
                # Per-lane PLD overrides only alter PLD arms; the ordinary
                # references keep the same tokens, stops and caps.
                insert["prompt_lookup_configs"] = [dict(self.lane_policies[i]) for i in lanes]
            uids = gen.insert(prompts, **insert)
            by_uid = dict(zip(uids, lanes))
            live = set(uids)
            while live:
                polls += 1
                if polls > limit or time.monotonic() > deadline:
                    failures.append(f"bounded: stopped after {polls} polls")
                    break
                _, responses = gen.next()
                lost = gen.take_lane_failures()
                if lost:
                    failures.extend(str(f) for f in lost)
                    break
                for uid in list(live):
                    boundary = gen.pop_prompt_boundary(uid) if hasattr(gen, "pop_prompt_boundary") else None
                    if boundary is not None and records[by_uid[uid]]["boundary"] is None:
                        records[by_uid[uid]]["boundary"] = boundary
                for response in responses:
                    record = records[by_uid[response.uid]]
                    record["tokens"].append(int(response.token))
                    if not arm.startswith("ordinary"):
                        # PLD rounds report their compute width; ordinary
                        # responses carry only a dataclass default, so their
                        # width is not reported rather than assumed.
                        record["widths"].add(int(getattr(response, "execution_width", 1) or 1))
                    receipt = getattr(response, "speculative_receipt", None)
                    if receipt is not None and (not record["receipts"] or record["receipts"][-1] is not receipt):
                        record["receipts"].append(receipt)
                    row = getattr(response, "logprobs", None)
                    if len(record["logprob_rows"]) < args.logprob_rows:
                        # One slot per delivered token: a missing row is an
                        # unavailable digest in its place, never a shifted one.
                        record["logprob_rows"].append(state_digest(None if row is None else [row]))
                    if response.finish_reason:
                        record["finish_reason"] = response.finish_reason
                        record["final"] = response
                        live.discard(response.uid)
                if remove is not None and removed is None:
                    lane, after = remove
                    uid = next(u for u, i in by_uid.items() if i == lane)
                    if uid in live and len(records[lane]["tokens"]) >= after:
                        gen.remove([uid])  # between polls: a closed boundary
                        live.discard(uid)
                        removed = {"lane": lane, "after_tokens": len(records[lane]["tokens"])}
            stats = dict(getattr(gen, "scheduler_stats", {}) or {})
        finally:
            gen.close()
        out = {}
        for i, record in records.items():
            final = record["final"]
            boundary = record["boundary"]
            cache = getattr(final, "prompt_cache", None)
            out[i] = {
                "covered_tokens": covered_tokens(cache),
                "tokens": record["tokens"],
                "token_sha256": _sha(json.dumps(record["tokens"]).encode()),
                "finish_reason": record["finish_reason"],
                "execution_widths": (sorted(record["widths"]) if not arm.startswith("ordinary")
                                     else "not reported by the ordinary route"),
                **lane_row_evidence(record["logprob_rows"], record["tokens"], args.logprob_rows),
                "final_state": state_digest(getattr(final, "prompt_cache", None)),
                "boundary": None if boundary is None else {
                    "covered_tokens": boundary.get("covered_tokens") if isinstance(boundary, dict) else None,
                    "state": state_digest(boundary),
                },
                "rounds": _round_summary(record["receipts"]),
                "_final": final,
            }
        return out, stats, failures, removed

    def continuation(self, lane, record):
        """Greedy ordinary B1 continuation from the lane's final cache.

        The lane's ``final_state`` was digested by ``run`` before this reuses
        the cache. ``complete`` needs delivered tokens and a complete final
        digest; anything less is ``unavailable`` with the refusal reason.
        """
        from scripts.paired_direct_ab import snapshot_refusal

        final = record.pop("_final", None)
        if not self.args.continuation_tokens:
            return {"status": "unavailable", "reason": "disabled (--continuation-tokens 0)"}
        covered = record["covered_tokens"]
        if final is None or getattr(final, "prompt_cache", None) is None or covered is None:
            return {"status": "unavailable", "reason": "no final cache or inconsistent offsets"}
        # Each route covers its own prefix of prompt + output (ordinary decode
        # may already hold the last token); feed exactly the uncovered rest.
        full = list(self.prompts[lane]) + list(record["tokens"])
        if not 0 < covered <= len(full):
            return {"status": "unavailable", "reason": f"covered {covered} outside 1..{len(full)}"}
        try:
            out, _stats, failures, _ = self.run(
                "ordinary_b1", [lane], caches=[final.prompt_cache], all_tokens=[full[:covered]],
                prompts=[full[covered:] + list(self.followup)], caps=[self.args.continuation_tokens])
        except Exception as error:  # noqa: BLE001 - recorded as unavailable, never exact
            return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"[:200]}
        if failures:
            return {"status": "unavailable", "reason": "; ".join(failures)[:200]}
        result = out[lane]
        result.pop("_final", None)
        refusal = ("no continuation tokens" if not result["tokens"]
                   else snapshot_refusal(result["final_state"]))
        if refusal is not None:
            return {"status": "unavailable", "reason": f"continuation final state: {refusal}"[:200],
                    "tokens": result["tokens"], "final_state": result["final_state"]}
        return {"status": "complete", "tokens": result["tokens"], "final_state": result["final_state"]}


def covered_tokens(cache):
    """Tokens a final cache covers (one offset shared by every plane), or None."""
    if not cache:
        return None
    offsets = {int(entry.offset) for entry in cache if hasattr(entry, "offset")}
    return offsets.pop() if len(offsets) == 1 else None


def _round_summary(receipts):
    """Per-lane PLD rounds from receipts (one receipt object per round)."""
    rounds = [(int(r.get("round_proposed", 0)), int(r.get("round_accepted", 0))) for r in receipts]
    rejected = [i for i, (p, a) in enumerate(rounds) if p and a < p]
    return {
        "rounds": len(rounds),
        "proposed": sum(p for p, _ in rounds),
        "accepted": sum(a for _, a in rounds),
        "rejected_rounds": len(rejected),
        "partial_accept_rounds": sum(1 for p, a in rounds if 0 < a < p),
        "rollback_then_append": bool(rejected and rejected[0] < len(rounds) - 1),
    }


# ---------------------------------------------------------------- evidence

def lane_row_evidence(digests, tokens, requested):
    """Logprob-row evidence fields of one lane record (pure).

    ``digests`` are the emitted ``state_digest`` records, one per delivered
    token up to ``requested``; the hash and status lists are readable
    duplicates only and never qualify on their own.
    """
    return {
        "evidence": EVIDENCE,
        "logprob_rows_requested": requested,
        "logprob_rows_expected": min(requested, len(tokens)),
        "logprob_row_digests": list(digests),
        "logprob_rows": [d["sha256"] for d in digests],
        "logprob_rows_status": sorted({d["status"] for d in digests}) or ["unavailable"],
    }


def _is_count(value):
    return type(value) is int and value >= 0


def row_evidence_refusal(record, requested):
    """Why one lane's logprob rows cannot support bit parity, or None (pure).

    ``requested`` is the protocol bound (``--logprob-rows``); the expected
    count is recomputed from it and the lane's own delivered tokens, so a
    record cannot lower its own bar.
    """
    from scripts.paired_direct_ab import snapshot_refusal

    if not _is_count(requested):
        return "no protocol logprob row bound"
    if requested == 0:
        return "disabled (--logprob-rows 0)"
    if record.get("evidence") != EVIDENCE:
        return "legacy record without strict row evidence"
    tokens = record.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        return "no delivered tokens"
    expected = min(requested, len(tokens))
    declared = record.get("logprob_rows_requested")
    if not _is_count(declared) or declared != requested:
        return _bounded_detail("requested ", repr(declared), f" is not the protocol bound {requested}")
    declared = record.get("logprob_rows_expected")
    if not _is_count(declared) or declared != expected:
        return _bounded_detail("declares ", repr(declared), f" rows; protocol bound and delivered tokens give {expected}")
    digests = record.get("logprob_row_digests")
    if not isinstance(digests, list):
        return "no row digest records"
    if len(digests) != expected:
        return f"{len(digests)} row digests, expected {expected}"
    for k, digest in enumerate(digests):
        refusal = snapshot_refusal(digest)
        if refusal is not None:
            return f"row {k}: {refusal}"[:200]
    if "logprob_rows" in record and record["logprob_rows"] != [d["sha256"] for d in digests]:
        return "logprob_rows hashes contradict the row digests"
    if "logprob_rows_status" in record and record["logprob_rows_status"] != ["complete"]:
        return "logprob_rows_status contradicts the row digests"
    return None


def continuation_refusal(continuation):
    """Why a continuation record is not complete evidence, or None (pure)."""
    from scripts.paired_direct_ab import snapshot_refusal

    if not isinstance(continuation, dict):
        return "absent"
    if continuation.get("status") != "complete":
        return f"status {continuation.get('status')!r}: {continuation.get('reason')}"[:200]
    if continuation.get("reason") is not None:
        return "complete with a refusal reason"
    tokens = continuation.get("tokens")
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
        return "complete without delivered token ids"
    refusal = snapshot_refusal(continuation.get("final_state"))
    return None if refusal is None else f"final state: {refusal}"[:200]


def _both(check, arm, got, reference, want):
    """Each arm's refusal from ``check`` (named), joined; '' when both pass."""
    return "; ".join(f"{side}: {refusal}" for side, refusal in
                     ((arm, check(got)), (reference, check(want))) if refusal is not None)


# ---------------------------------------------------------------- compare

def compare(arm, reference, lanes, results, *, prefix_only=(), continuation=True, logprob_rows=None):
    """Token and storage-bit parity of ``arm`` against a named ``reference``.

    Token parity and bit parity are separate verdicts: equal tokens with
    different logprob or state bits is reported as bit divergence (different
    forward geometry), never as exact. Final states whose routes cover a
    different number of tokens are ``incomparable``, not equal and not a
    counterexample. Every digest is validated on each arm before equality
    (``snapshot_refusal``, ``row_evidence_refusal`` against the protocol
    bound ``logprob_rows``, ``continuation_refusal``); incomplete evidence is
    ``incomparable`` even when both arms record the same dictionary. A
    removed (``prefix_only``) lane needs a non-empty token prefix only.
    """
    from scripts.paired_direct_ab import snapshot_refusal

    token_diffs, bit_diffs, incomparable = [], [], []
    for i in lanes:
        got, want = results[arm][i], results[reference][i]
        a, b = got["tokens"], want["tokens"]
        if i in prefix_only:
            if not a:
                incomparable.append(f"lane {i}: removed-lane prefix is empty")
            elif a != b[: len(a)]:
                token_diffs.append(f"lane {i}: removed-lane prefix differs")
            continue
        if a != b:
            index = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            token_diffs.append(f"lane {i}: tokens differ at {index}")
            continue
        refused = _both(lambda r: row_evidence_refusal(r, logprob_rows), arm, got, reference, want)
        if refused:
            incomparable.append(f"lane {i}: logprob rows unavailable ({refused})"[:400])
        else:
            g_rows = [d["sha256"] for d in got["logprob_row_digests"]]
            w_rows = [d["sha256"] for d in want["logprob_row_digests"]]
            if g_rows != w_rows:
                first = next(k for k in range(len(g_rows)) if g_rows[k] != w_rows[k])
                bit_diffs.append(f"lane {i}: logprob row bits differ from row {first}")
        refused = _both(lambda r: snapshot_refusal(r.get("final_state")), arm, got, reference, want)
        g_cover, w_cover = got.get("covered_tokens"), want.get("covered_tokens")
        if refused:
            incomparable.append(f"lane {i}: final state unavailable ({refused})"[:400])
        elif not all(type(c) is int and c > 0 for c in (g_cover, w_cover)):
            incomparable.append(f"lane {i}: covered tokens unavailable ({g_cover!r} vs {w_cover!r})")
        elif g_cover != w_cover:
            incomparable.append(f"lane {i}: final caches cover {got['covered_tokens']} vs "
                                f"{want['covered_tokens']} tokens")
        elif got["final_state"]["sha256"] != want["final_state"]["sha256"]:
            bit_diffs.append(f"lane {i}: final state bits differ")
        if not continuation:
            continue
        g, w = got.get("continuation"), want.get("continuation")
        refused = _both(continuation_refusal, arm, g, reference, w)
        if refused:
            incomparable.append(f"lane {i}: continuation unavailable ({refused})"[:400])
        elif g["tokens"] != w["tokens"]:
            token_diffs.append(f"lane {i}: continuation tokens differ")
        elif g["final_state"]["sha256"] != w["final_state"]["sha256"]:
            bit_diffs.append(f"lane {i}: continuation state bits differ")
    return {"arm": arm, "reference": reference, "tokens_exact": not token_diffs,
            "bits_exact": not token_diffs and not bit_diffs and not incomparable,
            "token_differences": token_diffs, "bit_differences": bit_diffs,
            "incomparable": incomparable}


def coverage(driver, results, stats, removed):
    batched, removal = results["pld_batched"], results["pld_removal"]
    lanes = range(len(driver.prompts))
    survivor = [i for i in lanes if removed and i != removed["lane"]]
    items = {
        "unequal_prompt_lengths": len({len(p) for p in driver.prompts}) > 1,
        "unequal_output_caps": len(set(driver.caps)) > 1,
        "zero_proposal_lane": any(batched[i]["rounds"]["proposed"] == 0 for i in lanes),
        "proposals": stats["pld_batched"].get("pld_proposed", 0) > 0,
        "rollback_counter": stats["pld_batched"].get("pld_rollbacks", 0) > 0,
        "rejected_suffix": any(batched[i]["rounds"]["rejected_rounds"] > 0 for i in lanes),
        "partial_acceptance": any(batched[i]["rounds"]["partial_accept_rounds"] > 0 for i in lanes),
        "rollback_then_append": any(batched[i]["rounds"]["rollback_then_append"] for i in lanes),
        "batched_width_ge2": stats["pld_batched"].get("pld_batched_max_width", 0) >= 2,
        # Inserted requests are not a compute width: the batched verify must
        # actually have run at the requested cohort width.
        "requested_width_engaged": stats["pld_batched"].get("pld_batched_max_width", 0) == len(driver.prompts),
        "closed_boundary_removal": bool(removed),
        "survivor_ran_at_lower_width": any(
            len(removal[i]["execution_widths"]) > 1 and min(removal[i]["execution_widths"]) < len(driver.prompts)
            for i in survivor
        ),
    }
    return items


def run_all(args):
    gate = preflight(args)
    if gate["enforced"] and gate["refusals"]:
        return refused_receipt(args, STAGE_PREFLIGHT, gate["refusals"], {"preflight": public_preflight(gate)})
    try:
        driver = Driver(args, gate)
    except IdentityRefusal as refusal:
        return refused_receipt(args, refusal.stage, refusal.refusals,
                               {"preflight": public_preflight(gate), **refusal.identity})
    driver.check_geometry()
    lanes = list(range(len(driver.prompts)))
    results, stats, failures = {}, {}, {}
    removed = None
    for arm in ARMS:
        if arm == "ordinary_b1":
            merged, arm_stats, arm_failures = {}, [], []
            for i in lanes:
                out, st, fl, _ = driver.run(arm, [i])
                merged.update(out)
                arm_stats.append(st)
                arm_failures += fl
            results[arm], stats[arm], failures[arm] = merged, {"per_lane": arm_stats}, arm_failures
        else:
            remove = (args.remove_lane, args.remove_after) if arm == "pld_removal" else None
            out, st, fl, rem = driver.run(arm, lanes, remove=remove)
            results[arm], stats[arm], failures[arm] = out, st, fl
            if arm == "pld_removal":
                removed = rem
        # Finish this arm's continuations now and drop its final responses and
        # caches, so no arm's tensors are alive while the next one runs.
        # Only the lane the removal arm actually removed is skipped: its cache
        # is dead and never resumed. Every surviving lane is continued.
        dead = removed["lane"] if arm == "pld_removal" and removed else None
        for i in lanes:
            if i == dead:
                results[arm][i].pop("_final", None)
            else:
                results[arm][i]["continuation"] = driver.continuation(i, results[arm][i])
    rows = args.logprob_rows
    comparisons = [
        compare("ordinary_bN", PRIMARY_REFERENCE, lanes, results, logprob_rows=rows)
        | {"kind": "ordinary geometry (B1 vs BN)"},
        compare("pld_per_lane", PRIMARY_REFERENCE, lanes, results, logprob_rows=rows)
        | {"kind": "pld vs primary reference"},
        compare("pld_batched", PRIMARY_REFERENCE, lanes, results, logprob_rows=rows)
        | {"kind": "pld vs primary reference"},
        compare("pld_batched", "ordinary_bN", lanes, results, logprob_rows=rows)
        | {"kind": "pld vs batched ordinary (secondary)"},
        compare("pld_per_lane", "ordinary_bN", lanes, results, logprob_rows=rows)
        | {"kind": "pld vs batched ordinary (secondary)"},
    ]
    removal_cmp = compare("pld_removal", PRIMARY_REFERENCE, lanes, results, logprob_rows=rows,
                          prefix_only=(removed["lane"],) if removed else ())
    removal_cmp["kind"] = "membership survivor vs primary reference"
    comparisons.append(removal_cmp)
    cov = coverage(driver, results, stats, removed)
    primary = [c for c in comparisons if c["reference"] == PRIMARY_REFERENCE and c["arm"].startswith("pld")]
    pld_token_bad = [c for c in primary if not c["tokens_exact"]]
    pld_bits_bad = [c for c in primary if not c["bits_exact"]]
    engagement = [f for arm, fl in failures.items() for f in (f"{arm}: {x}" for x in fl)]
    for arm in ARMS:
        for i in lanes:
            rec = results[arm][i]
            if arm == "pld_removal" and removed and i == removed["lane"]:
                continue
            cap = driver.caps[i]
            if rec["finish_reason"] is None or (len(rec["tokens"]) < cap and rec["finish_reason"] != "stop"):
                engagement.append(f"{arm} lane {i}: early stop ({len(rec['tokens'])}/{cap}, {rec['finish_reason']})")
    if engagement:
        verdict = "refused"
    elif pld_token_bad:
        verdict = "counterexample"
    elif not all(cov.values()):
        verdict = "coverage_refused"
    elif pld_bits_bad:
        # Greedy tokens match the declared reference but logprob/state bits
        # do not (or cannot be compared): not exact, and not relabeled so.
        verdict = "token_exact_bits_diverge"
    else:
        verdict = "pass"
    after, identity_refusals = checked_post_run_identity(args, gate, driver)
    enforced = gate["enforced"]
    if enforced and identity_refusals:
        # Real results stay in the record; they are bound to no identity.
        verdict = "refused"
    native = driver.native_identity or {}

    return {
        "schema": SCHEMA,
        "scope": _scope(args),
        "executed": True,
        "arms_executed": list(ARMS),
        "refused_at": STAGE_POST_RUN if enforced and identity_refusals else None,
        "verdict": verdict,
        "qualification": "none: diagnostic only; no route, default or model qualification is asserted",
        "gpu_ownership": "not applicable (tiny CPU run)" if args.tiny else GPU_OWNERSHIP,
        "identity_gate": ("enforced before native import, before the adapter and after the arms" if enforced
                          else "recorded only: tiny CPU diagnostic, nothing qualified"),
        "identity_refusals": identity_refusals if enforced else [],
        "identity_advisory": [] if enforced else gate["refusals"] + identity_refusals,
        "primary_reference": PRIMARY_REFERENCE,
        "ordinary_geometry": ("bit_exact" if comparisons[0]["bits_exact"] else
                              "token_exact_bits_diverge" if comparisons[0]["tokens_exact"] else
                              "tokens_diverge"),
        "comparisons": comparisons,
        "coverage": cov,
        "refusals": engagement + ([f"identity: {p}" for p in identity_refusals] if enforced else []),
        "removed": removed,
        "widths": {
            "requested": len(driver.prompts),
            "observed_max": {arm: stats[arm].get("pld_batched_max_width", 0)
                             for arm in ("pld_batched", "pld_removal")},
            "note": ("observed = PLD batched-verify compute width; per-lane PLD and ordinary "
                     "routes do not report a compute width"),
        },
        "lane_policies": driver.lane_policies,
        "workload_receipt": driver.workload_receipt,
        "lanes": [{"prompt_tokens": len(p), "prompt_sha256": _sha(json.dumps(p).encode()), "max_tokens": c}
                  for p, c in zip(driver.prompts, driver.caps)],
        "results": {arm: {str(i): {k: v for k, v in r.items() if not k.startswith("_")}
                          for i, r in res.items()} for arm, res in results.items()},
        "scheduler_stats": stats,
        "protocol": protocol_section(args, list(driver.stops)),
        "identity": {"preflight": public_preflight(gate), "source": gate["source"],
                     "files": gate["source"]["files"], "after": after,
                     "mlx": native.get("mlx") or collect_build_identity(driver.mx)[0],
                     "modules_before_arms": native.get("modules_before_arms"),
                     "adapter_identity": native.get("adapter_identity"),
                     "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32"), **driver.identity},
    }


def _scope(args):
    return ("direct-model ragged PLD (one process); not HTTP serving, not serving "
            "qualification, no route or default selected; no timing"
            + ("; TINY random CPU model" if args.tiny else ""))


def protocol_section(args, stops):
    return {"arms": list(ARMS), "sampling": "greedy", "seed": args.seed,
            "prefill_step": args.prefill_step, "pld_policy": args.pld_policy,
            "lane_policies": args.lane_policies,
            "logprob_rows": args.logprob_rows, "continuation_tokens": args.continuation_tokens,
            "remove_lane": args.remove_lane, "remove_after": args.remove_after,
            "time_limit_s": args.time_limit_s, "lanes": args.lanes, "ignore_eos": args.ignore_eos,
            "stop_tokens": "not resolved (no adapter loaded)" if stops is None else stops}


def refused_receipt(args, stage, refusals, identity):
    """An unexecuted refusal: no arm ran, so no results, comparisons or coverage exist."""
    return {
        "schema": SCHEMA, "scope": _scope(args), "executed": False, "arms_executed": [],
        "refused_at": stage, "verdict": "refused", "refusals": list(refusals),
        "identity_refusals": list(refusals),
        "qualification": "none: refused before any arm ran; no parity, divergence or coverage evidence",
        "gpu_ownership": GPU_OWNERSHIP, "ordinary_geometry": "not_run", "comparisons": [],
        "coverage": None, "results": {}, "protocol": protocol_section(args, None), "identity": identity,
    }


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiny", action="store_true", help="deterministic random CPU model; never Metal")
    ap.add_argument("--model")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--prompt-ids", help='JSON {"prompts": [[ids]...], "max_tokens": [..]}')
    ap.add_argument("--lanes", type=int, default=None,
                    help="lanes: constructed real prompts (default 2) or tiny fixture (default 3); 2..4")
    ap.add_argument("--lane-policies", type=json.loads, default=None,
                    help="JSON list of per-lane PLD policy overrides (PLD arms only; no batched_verify)")
    ap.add_argument("--pld-policy", type=json.loads, default={}, help="PLD policy JSON (batched_verify set per arm)")
    ap.add_argument("--prefill-step", type=int, default=None)
    ap.add_argument("--logprob-rows", type=int, default=16)
    ap.add_argument("--continuation-tokens", type=int, default=4)
    ap.add_argument("--remove-lane", type=int, default=0)
    ap.add_argument("--remove-after", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--time-limit-s", type=float, default=900.0)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--out", required=True)
    return ap


def resolve_args(ap, argv=None):
    a = ap.parse_args(argv)
    if a.tiny:
        if a.i_own_the_gpu or a.model or a.prompt_ids:
            ap.error("--tiny runs a random CPU model; drop --i-own-the-gpu/--model/--prompt-ids")
        a.prefill_step = 8 if a.prefill_step is None else a.prefill_step
        a.lanes = 3 if a.lanes is None else a.lanes
    else:
        if not a.i_own_the_gpu:
            ap.error("refusing Metal execution without --i-own-the-gpu")
        if not a.model:
            ap.error("--model is required for a real run")
        a.prefill_step = 2048 if a.prefill_step is None else a.prefill_step
        a.lanes = 2 if a.lanes is None else a.lanes
    if "batched_verify" in a.pld_policy:
        ap.error("batched_verify is set per arm; drop it from --pld-policy")
    if not math.isfinite(a.time_limit_s) or a.time_limit_s <= 0:
        ap.error("--time-limit-s must be finite and positive")
    if not 1 <= a.prefill_step <= MAX_PROMPT_TOKENS:
        ap.error(f"--prefill-step 1..{MAX_PROMPT_TOKENS}")
    if not 0 <= a.logprob_rows <= MAX_OUTPUT_TOKENS or not 0 <= a.continuation_tokens <= MAX_OUTPUT_TOKENS:
        ap.error(f"--logprob-rows and --continuation-tokens 0..{MAX_OUTPUT_TOKENS} (0 = explicitly unavailable)")
    if not 2 <= a.lanes <= MAX_LANES or not 1 <= a.remove_after <= MAX_OUTPUT_TOKENS:
        ap.error("invalid bounds")
    return a


def main(argv=None):
    ap = build_parser()
    args = resolve_args(ap, argv)
    record = run_all(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=1, default=sorted) + "\n")
    print(json.dumps({k: record.get(k) for k in ("verdict", "executed", "refused_at", "ordinary_geometry",
                                                 "coverage", "refusals")}, indent=1))
    for c in record.get("comparisons") or []:
        print(f"{c['arm']} vs {c['reference']} ({c['kind']}): tokens_exact={c['tokens_exact']} "
              f"bits_exact={c['bits_exact']} {(c['token_differences'] + c['bit_differences'] + c['incomparable'])[:3]}")
    return 0 if record["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
