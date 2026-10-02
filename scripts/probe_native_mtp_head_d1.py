"""Bounded direct B1 probe: ordinary decode vs the native MTP head at depth 1.

Scope: DIAGNOSTIC, DIRECT-MODEL. One process loads the model once through its
adapter and drives the runtime ``BatchGenerator`` itself, one lane, greedy, in
ABBA order (ordinary, native, native, ordinary) with the same prompt token
ids, stop ids, ``LaneRNG`` seed, ``max_tokens`` and prefill step. Nothing here
is HTTP serving evidence, serving or native qualification, route selection, a
performance run, or the MTPLX paper's D1 acceptance metric.

Arms:

``ordinary``
    ``BatchGenerator`` without ``self_mtp``. It must make no self-MTP call
    and deliver no ``mtp_receipt``/``mtp_state``; its draft sidecar is
    legitimately absent and is never compared to the native one.
``native``
    ``BatchGenerator(self_mtp=adapter.execution_config(max_lanes=1))`` with
    ``num_draft`` 1 and no adaptive depth, ordinary handoff, admission
    callback, FLy relaxation, copy draft or acceptance logger. An artifact
    without the MTP capability or head refuses; nothing falls back.

Trace: the module attributes ``propose_batched_self_mtp``,
``advance_batched_self_mtp_zero`` and ``commit_batched_self_mtp`` of
``mlx2.runtime.hybrid_speculative`` are wrapped for the duration of one run
(the generator imports them at call time) and restored in ``finally``. Only
host metadata of outermost calls is kept (lane uids, draft depths, accepted
lengths, copy spans/decisions, relaxed accepts, output tokens and
``from_draft`` flags, emitted counts, terminal flags); a segmented commit's
nested per-row commit is counted, not recorded. No tensor math is redone.

Acceptance accounting (native runs) keeps three things apart:

- raw D1 head proposals: ``accepted_lengths`` over depth-1 cycles, before
  delivery;
- committed drafts: ``draft_accepted``/``draft_proposed`` as the runtime's
  lane stats record them after stop/max_tokens clipping
  (``consumed = min(emitted, accepted)``; ``draft_proposed`` adds the full
  depth), reconciled against the trace, the delivered ``from_draft`` flags
  and the token count;
- depth-0 cycles: with ``num_draft`` 1 the head depth is
  ``min(1, max_tokens - ntoks - 1)``, so the final position runs a depth-0
  budget cycle (zero fast path or a depth-0 proposal). These are reported,
  never pooled into D1. The prefill token is the anchor and is no cycle.

At depth 1 a delivered terminal prefix of at least one token always contains
the accepted draft, so stop clipping can only drop the bonus token; raw and
committed figures are still reported and reconciled separately. Any nonzero
copy span, copy decision, relaxed accept, retrieval counter, fallback
counter, lane width other than 1, depth off the fixed schedule, or a token
the trace cannot account for is refused: it would be something other than
head acceptance.

State: after each run the final target cache (``response.prompt_cache``) and,
on native runs, the draft sidecar (``response.mtp_state``) are digested by the
existing ``state_digest`` in ``scripts/paired_direct_ab.py`` (imported by
explicit path, unmodified) and must pass its ``snapshot_refusal``. Digests
bind cache class identity, ``state`` and ``meta_state``, so a cross-route
target difference is reported as a counterexample component without
deciding whether it is representation or value. RNG, scheduler and full
transaction state are outside the claim.

Logprob rows: each delivered ``response.logprobs`` array is digested, as the
original object and before any cast, by the same ``state_digest`` (dtype,
shape and raw storage bits) and must be complete. Only the selected token's
value is converted on the host (float32), as a diagnostic gap, never as a
digest input. NaN or +Inf anywhere in a row refuses; -Inf is legitimate on
masked or unselected positions; the selected value must be finite.

Identity: before any native import, the source identity (40-hex HEAD, a
known clean ``git status`` for the identity files, a sha256 for every
identity file) and the artifact manifest must be valid, or an unexecuted
``refused`` receipt is written. After importing MLX the build identity
(``paired_direct_ab.mlx_identity``) must be complete, then mlx2 must import
from this worktree, before the adapter or model is constructed. After the
runs the source identity is taken again and must be unchanged; both raw
identities are kept in the receipt.

Verdict: ``refused`` on any identity, contract, engagement, trace, logprob
or state refusal; otherwise ``counterexample`` when any exact component
(tokens, full logprob rows, target state, native sidecar across native
runs) differs; otherwise ``parity``. Exit 0/1/2. The acceptance block is a
diagnostic for this one prompt in every case.

Import, ``--help`` and ``--dry-run`` are standard-library only: no mlx,
mlx_lm or mlx2 import, no model load, no Metal. ``--dry-run`` reads bounded
artifact metadata (no weight bytes) and writes an unexecuted plan to
``--out``. ``--i-own-gpu`` runs the native body; the caller asserts it already
holds the GPU lease and locks, which this driver neither acquires nor checks.

  python scripts/probe_native_mtp_head_d1.py --dry-run --out /tmp/d1-plan.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mlx2.native-mtp-head-d1-probe.v1"
DEFAULT_MODEL = ("~/mlx-models/"
                 "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp")
LANES = 1
NUM_DRAFT = 1
ARM_ORDER = ("ordinary", "native", "native", "ordinary")
MAX_PROMPT_TOKENS = 1024
MAX_PROMPT_CHARS = 8192
MAX_TOKENS_RANGE = (8, 128)
DEFAULT_MAX_TOKENS = 64
PREFILL_STEP_RANGE = (64, 1024)
DEFAULT_PREFILL_STEP = 512
DEFAULT_SEED = 20261001
DEFAULT_PROMPT = ("Explain in one careful paragraph how a lighthouse keeper records the "
                  "weather every hour and why each measurement matters.")
IDLE_POLL_LIMIT = 4096
# A greedy argmax logprob is <= 0; allow float rounding of log-softmax.
LOGPROB_CEILING = 1e-4
ORACLE_PATH = ROOT / "scripts" / "paired_direct_ab.py"
ORACLE_VERSION = "mlx2.cache-state-digest.v1"
IDENTITY_FILES = (
    "scripts/probe_native_mtp_head_d1.py",
    "scripts/paired_direct_ab.py",
    "src/mlx2/runtime/hybrid_speculative.py",
    "src/mlx2/runtime/generate.py",
    "src/mlx2/runtime/sample_utils.py",
    "src/mlx2/adapters/registry.py",
    "src/mlx2/adapters/qwen36_35b.py",
    "src/mlx2/adapters/qwen38_27b.py",
    "src/mlx2/serving.py",
)
# The project's artifact binding (adapters/qwen36_35b.py inspect_artifact):
# these files' bytes, then each shard's (name, size, mtime_ns). No weight bytes.
METADATA_FILES = ("config.json", "model.safetensors.index.json", "tokenizer.json",
                  "tokenizer_config.json", "chat_template.jinja", "generation_config.json")
TRACED = ("propose_batched_self_mtp", "advance_batched_self_mtp_zero", "commit_batched_self_mtp")
# Scheduler counters that mean something other than the head proposed.
FALLBACK_COUNTERS = (
    "mtp_target_only_plain_fallbacks", "starved_mtp_plain_fallbacks",
    "mtp_ordinary_handoff_events", "mtp_draftless_lane_refusals",
    "self_mtp_copy_rounds", "self_mtp_copy_proposed_tokens",
    "self_mtp_copy_accepted_tokens", "fly_relaxed_accepts",
)
STATS_KEYS = ("cycles", "draft_cycles", "draft_proposed", "draft_accepted", "bonus_tokens",
              "plain_tokens", "plain_cycles", "retrieval_cycles", "retrieval_proposed",
              "retrieval_accepted", "total_emitted")
SCOPE = ("diagnostic direct-model B1 probe (adapter + generator in one process, lanes=1, "
         "num_draft=1, greedy, one prompt); not HTTP serving, not serving or native "
         "qualification, not route selection, not a controlled performance run")
ACCEPTANCE_SCOPE = (
    "one prompt at greedy on this artifact and build: raw = depth-1 head proposals the exact "
    "verify accepted before delivery; committed = runtime lane stats after stop/max_tokens "
    "clipping. Neither is the MTPLX paper D1 metric, an acceptance improvement, or a quality "
    "or speed claim")
NOT_CLAIMED = (
    "qualification, selection or observed-used state of any route",
    "MTPLX paper D1 acceptance or any comparison with published figures",
    "a loader fix, a head repair or an acceptance improvement",
    "RNG, scheduler or full transaction-state equality",
    "throughput, latency or memory performance",
)
STATE_SCOPE = ("final target cache on every run and native draft sidecar (response.mtp_state) "
               "on native runs, digested by paired_direct_ab.state_digest and required complete "
               "by its snapshot_refusal; ordinary sidecar legitimately absent and never "
               "compared; RNG, scheduler and full transaction state excluded")
RAW_ROW_ORACLE = {
    "function": "paired_direct_ab.state_digest on the original response.logprobs array, "
                "before any cast",
    "binds": "array dtype, shape and raw storage bits (mlx2.cache-state-digest.v1)",
    "required": "complete per paired_direct_ab.snapshot_refusal for every delivered token",
    "recorded": "one sha256 per row plus distinct dtype/shape layouts; no vectors",
    "chosen_value": "host float32 conversion of the selected entry; diagnostic gap only, "
                    "never a digest input",
    "refuses": "NaN or +Inf anywhere in a row; a non-finite or positive selected value",
    "allows": "-Inf on masked or unselected positions",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_ids(ids) -> str:
    return _sha(json.dumps([int(t) for t in ids], separators=(",", ":")).encode())


def _is_int(value) -> bool:
    return type(value) is int


def _is_sha256(value) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _jsonable(value, depth=0):
    """Bounded strict-JSON copy of host evidence (non-finite floats as strings)."""
    if depth > 16:
        return "<depth>"
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in list(value.items())[:512]}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in list(value)[:4096]]
    return f"<{type(value).__qualname__}>"


# ---------------------------------------------------------------- identity

def source_identity():
    """Raw source identity; git failures are recorded as None, never as clean."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    files = {}
    for name in IDENTITY_FILES:
        path = ROOT / name
        files[name] = _sha(path.read_bytes()) if path.is_file() else None
    status = git("status", "--porcelain", "--", *IDENTITY_FILES)
    return {
        "commit": git("rev-parse", "HEAD"),
        "status_known": status is not None,
        "dirty": None if status is None else bool(status),
        "status_porcelain": None if status is None else status[:2000],
        "files": files,
    }


def _is_commit(value) -> bool:
    return type(value) is str and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def source_identity_refusals(identity):
    """Why a raw source identity cannot bind a native run (empty when it can)."""
    if not isinstance(identity, dict):
        return ["source identity missing"]
    out = []
    if not _is_commit(identity.get("commit")):
        out.append(f"source commit {identity.get('commit')!r} is not a 40-hex revision "
                   "(git failed or HEAD unknown)")
    if identity.get("status_known") is not True or type(identity.get("dirty")) is not bool:
        out.append("worktree status of the identity files is unknown (git status failed)")
    elif identity["dirty"]:
        out.append("identity files are not clean at HEAD (modified or untracked)")
    files = identity.get("files")
    if not isinstance(files, dict) or set(files) != set(IDENTITY_FILES):
        out.append("identity file set is not exactly the required IDENTITY_FILES")
    else:
        missing = sorted(name for name, value in files.items() if not _is_sha256(value))
        if missing:
            out.append("no sha256 for identity files: " + ", ".join(missing))
    return out


def source_identity_changes(before, after):
    """Refusals when the source identity moved between preflight and the end of the run."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return ["source identity before or after the run is missing"]
    out = []
    for key in ("commit", "status_known", "dirty"):
        if before.get(key) != after.get(key):
            out.append(f"source {key} changed during the run")
    files_before, files_after = before.get("files") or {}, after.get("files") or {}
    for name in sorted(set(files_before) | set(files_after)):
        if files_before.get(name) != files_after.get(name):
            out.append(f"identity file {name} changed during the run")
    return out


def mlx_identity_refusals(build):
    """Why ``paired_direct_ab.mlx_identity`` output is not a complete build identity."""
    if not isinstance(build, dict):
        return ["MLX build identity missing"]
    out = []
    for key in ("version", "package", "device"):
        if type(build.get(key)) is not str or not build[key].strip():
            out.append(f"MLX build identity has no {key}")
    path = build.get("path")
    if type(path) is not str or not os.path.isabs(path) or not os.path.isdir(path):
        out.append(f"MLX core path {path!r} is not an existing absolute directory")
    if not _is_sha256(build.get("metallib_sha256")):
        out.append("MLX build identity has no metallib sha256")
    return out


def module_path_refusals(files):
    """Each imported module file must exist under this worktree's ``src``."""
    out = []
    src = ROOT / "src"
    for name, file in sorted(files.items()):
        path = Path(file).resolve() if isinstance(file, str) and os.path.isabs(file) else None
        if path is None or not path.is_file() or src not in path.parents:
            out.append(f"{name} imported from {file!r}, not a file under {src}")
    return out


def artifact_manifest(model_path):
    """Bounded artifact identity: metadata bytes and shard stats, never weights."""
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config", config) if isinstance(config, dict) else {}
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("artifact index has no weight_map")
    shards = sorted(set(weight_map.values()))
    digest = hashlib.sha256()
    metadata = {}
    for name in METADATA_FILES:
        item = path / name
        if item.is_file():
            data = item.read_bytes()
            metadata[name] = _sha(data)
            digest.update(name.encode())
            digest.update(data)
    records = []
    for name in shards:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("weight shard paths must stay within the artifact")
        stat = (path / name).stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(list(record))
        digest.update(json.dumps(record).encode())
    mtp_keys = [key for key in weight_map if key.startswith(("language_model.mtp.", "mtp."))]
    return {
        "path": str(path),
        "fingerprint": digest.hexdigest(),
        "fingerprint_recipe": ("sha256 over present metadata files (name + bytes) in "
                               "adapters/qwen36_35b.py order, then json [name, size, "
                               "mtime_ns] per sorted shard; no weight bytes read"),
        "metadata_sha256": metadata,
        "shards": records,
        "model_type": config.get("model_type") if isinstance(config, dict) else None,
        "mtp_num_hidden_layers": text.get("mtp_num_hidden_layers"),
        "indexed_tensors": len(weight_map),
        "mtp_indexed_tensors": len(mtp_keys),
    }


def manifest_refusals(manifest):
    out = []
    if not isinstance(manifest, dict):
        return ["artifact manifest unavailable"]
    if not _is_sha256(manifest.get("fingerprint")):
        out.append("artifact fingerprint is not a sha256")
    layers = manifest.get("mtp_num_hidden_layers")
    if not _is_int(layers) or layers < 1:
        out.append(f"config declares no MTP head (mtp_num_hidden_layers={layers!r})")
    if not _is_int(manifest.get("mtp_indexed_tensors")) or manifest["mtp_indexed_tensors"] <= 0:
        out.append("index has no MTP tensors")
    return out


_ORACLE = None


def load_oracle():
    """``scripts/paired_direct_ab.py`` by explicit path (its import is stdlib only)."""
    global _ORACLE
    if _ORACLE is None:
        spec = importlib.util.spec_from_file_location("mlx2_probe_state_oracle", ORACLE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.STATE_ORACLE != ORACLE_VERSION:
            raise RuntimeError(f"state oracle {module.STATE_ORACLE!r} is not {ORACLE_VERSION!r}")
        _ORACLE = module
    return _ORACLE


def oracle_identity():
    return {"path": str(ORACLE_PATH), "sha256": _sha(ORACLE_PATH.read_bytes()),
            "version": ORACLE_VERSION, "functions": ["state_digest", "snapshot_refusal"],
            "scope": STATE_SCOPE}


# ---------------------------------------------------------------- trace

def proposal_metadata(proposal, kind):
    """Host fields of one self-MTP cycle result; never touches arrays."""
    outputs = tuple(proposal.outputs)
    return {
        "kind": kind,
        "lane_uids": _jsonable(list(proposal.lane_uids)),
        "draft_depths": _jsonable(list(proposal.draft_depths)),
        "accepted_lengths": _jsonable(list(proposal.accepted_lengths)),
        "copy_spans": _jsonable(list(getattr(proposal, "copy_spans", ()) or ())),
        "copy_decisions": _jsonable(list(getattr(proposal, "copy_decisions", ()) or ())),
        "relaxed_accepts": _jsonable(list(getattr(proposal, "relaxed_accepts", ()) or ())),
        "draft_feature_rows": len(getattr(proposal, "draft_features", ()) or ()),
        "zero_fast_path": _jsonable(getattr(proposal, "zero_fast_path", None)),
        "true_batched": _jsonable(getattr(proposal, "true_batched", None)),
        "output_tokens": [[_jsonable(token.token) for token in row] for row in outputs],
        "output_from_draft": [[_jsonable(token.from_draft) for token in row] for row in outputs],
    }


class ProposalTrace:
    """Wrap the self-MTP proposal/commit entry points of ``module`` for one run.

    Outermost calls only; bounded to ``limit`` cycles; the originals are put
    back on exit whatever happened, and a patch that is not ours at exit is
    flagged.
    """

    def __init__(self, module, *, limit):
        self.module = module
        self.limit = int(limit)
        self.cycles, self.commits, self.errors = [], [], []
        self.nested_proposals = self.nested_commits = 0
        self.overflow = self.foreign_patch = False
        self.installed, self.restored = False, None
        self._originals, self._wrappers = {}, {}
        self._depth = {"propose": 0, "commit": 0}
        self._open = None

    def __enter__(self):
        originals = {name: getattr(self.module, name) for name in TRACED}
        self._originals = originals
        self._wrappers = {
            "propose_batched_self_mtp": self._proposal_wrapper(originals["propose_batched_self_mtp"], "propose"),
            "advance_batched_self_mtp_zero": self._proposal_wrapper(
                originals["advance_batched_self_mtp_zero"], "zero"),
            "commit_batched_self_mtp": self._commit_wrapper(originals["commit_batched_self_mtp"]),
        }
        try:
            for name, wrapper in self._wrappers.items():
                setattr(self.module, name, wrapper)
            self.installed = True
        except BaseException:
            self._restore()
            raise
        return self

    def __exit__(self, *exc):
        self._restore()
        return False

    def _restore(self):
        restored = True
        for name, original in self._originals.items():
            if name in self._wrappers and getattr(self.module, name, None) is not self._wrappers[name]:
                self.foreign_patch = True
            setattr(self.module, name, original)
            restored = restored and getattr(self.module, name) is original
        self.restored = restored
        self._open = None

    def _proposal_wrapper(self, original, kind):
        def traced(*args, **kwargs):
            outer = self._depth["propose"] == 0
            self._depth["propose"] += 1
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                if outer:
                    self.errors.append({"call": kind, "error": type(error).__name__})
                raise
            finally:
                self._depth["propose"] -= 1
            if outer:
                self._record_cycle(kind, result)
            else:
                self.nested_proposals += 1
            return result

        return traced

    def _record_cycle(self, kind, proposal):
        if len(self.cycles) >= self.limit:
            self.overflow = True
            self._open = None
            return
        try:
            record = proposal_metadata(proposal, kind)
        except Exception as error:  # noqa: BLE001 - recorded, refused later
            record = {"kind": kind, "malformed": f"{type(error).__name__}: {error}"[:200]}
        record["index"] = len(self.cycles)
        self.cycles.append(record)
        self._open = (proposal, record["index"])

    def _commit_wrapper(self, original):
        def traced(*args, **kwargs):
            outer = self._depth["commit"] == 0
            record = None
            if outer:
                proposal = args[1] if len(args) > 1 else kwargs.get("proposal")
                opened = self._open
                record = {
                    "cycle": opened[1] if opened is not None and opened[0] is proposal else None,
                    "emitted_counts": _jsonable(list(kwargs.get("emitted_counts", ()))),
                    "terminal": _jsonable(list(kwargs.get("terminal", ()))),
                    "completed": False,
                }
                if len(self.commits) < self.limit:
                    self.commits.append(record)
                else:
                    self.overflow = True
                self._open = None
            else:
                self.nested_commits += 1
            self._depth["commit"] += 1
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                if outer:
                    self.errors.append({"call": "commit", "error": type(error).__name__})
                raise
            finally:
                self._depth["commit"] -= 1
            if record is not None:
                record["completed"] = True
            return result

        return traced

    def summary(self):
        return {
            "installed": self.installed, "restored": self.restored,
            "foreign_patch": self.foreign_patch, "overflow": self.overflow, "limit": self.limit,
            "cycles": self.cycles, "commits": self.commits, "errors": self.errors,
            "nested_proposals": self.nested_proposals, "nested_commits": self.nested_commits,
        }


# ---------------------------------------------------------------- validators

def _one(values, name, where, refusals):
    if not isinstance(values, list) or len(values) != 1:
        refusals.append(f"{where}: {name} is not exactly one lane: {values!r}"[:200])
        return None
    return values[0]


def trace_hygiene_refusals(trace):
    out = []
    if not isinstance(trace, dict):
        return ["trace missing"]
    if trace.get("installed") is not True:
        out.append("trace was not installed")
    if trace.get("restored") is not True:
        out.append("traced entry points were not restored")
    if trace.get("foreign_patch"):
        out.append("a foreign patch replaced a traced entry point during the run")
    if trace.get("overflow"):
        out.append(f"trace exceeded its bound of {trace.get('limit')} cycles")
    for error in trace.get("errors", ()):
        if error != {"call": "zero", "error": "ZeroDepthFastUnavailable"}:
            out.append(f"traced call raised: {error}"[:200])
    return out


def validate_native_trace(trace, *, max_tokens, tokens, from_draft, stats, finish_reason):
    """(accounting, refusals) for one native run's host trace and delivery."""
    refusals = trace_hygiene_refusals(trace)
    cycles = list(trace.get("cycles", ())) if isinstance(trace, dict) else []
    commits = list(trace.get("commits", ())) if isinstance(trace, dict) else []
    by_cycle = {}
    for number, commit in enumerate(commits):
        index = commit.get("cycle") if isinstance(commit, dict) else None
        if not _is_int(index) or not 0 <= index < len(cycles):
            refusals.append(f"commit {number} does not belong to the open traced proposal")
        elif index in by_cycle:
            refusals.append(f"cycle {index} committed twice")
        elif commit.get("completed") is not True:
            refusals.append(f"commit {number} did not complete")
        else:
            by_cycle[index] = commit
    emitted = 1  # the prefill token: the lane's anchor (ntoks=1), no cycle
    expected_tokens, expected_flags = list(tokens[:1]), [False] if tokens else []
    d1 = d1_accepted = zero_fast = propose_zero = 0
    committed_proposed = committed_accepted = bonus = 0
    clipped_accepted = bonus_dropped = undelivered = 0
    terminal_at = None
    per_cycle = []
    for i, cycle in enumerate(cycles):
        where = f"cycle {i}"
        if not isinstance(cycle, dict) or "malformed" in cycle:
            refusals.append(f"{where}: malformed proposal metadata")
            continue
        if terminal_at is not None:
            refusals.append(f"{where}: cycle after the terminal cycle {terminal_at}")
        lanes = cycle.get("lane_uids")
        if not isinstance(lanes, list) or len(lanes) != LANES:
            prefix, suffix = f"{where}: lane width ", " is not exactly 1"
            refusals.append(prefix + repr(lanes)[:200 - len(prefix) - len(suffix)] + suffix)
            continue
        depth = _one(cycle.get("draft_depths"), "draft_depths", where, refusals)
        accepted = _one(cycle.get("accepted_lengths"), "accepted_lengths", where, refusals)
        tokens_out = _one(cycle.get("output_tokens"), "output_tokens", where, refusals)
        flags_out = _one(cycle.get("output_from_draft"), "output_from_draft", where, refusals)
        if not (_is_int(depth) and _is_int(accepted)):
            refusals.append(f"{where}: depth/accepted are not ints ({depth!r}, {accepted!r})")
            continue
        if not isinstance(tokens_out, list) or not isinstance(flags_out, list):
            continue
        planned = min(NUM_DRAFT, max(max_tokens - emitted - 1, 0))
        if depth != planned:
            refusals.append(f"{where}: draft depth {depth} off the fixed num_draft=1 schedule "
                            f"(expected {planned} at {emitted} emitted)")
        if not 0 <= accepted <= depth:
            refusals.append(f"{where}: accepted {accepted} outside 0..{depth}")
            continue
        if len(tokens_out) != accepted + 1 or len(flags_out) != accepted + 1:
            refusals.append(f"{where}: {len(tokens_out)} outputs for {accepted} accepted drafts")
            continue
        if flags_out != [True] * accepted + [False] or not all(_is_int(t) for t in tokens_out):
            refusals.append(f"{where}: output from_draft/tokens malformed")
        if cycle.get("copy_spans") not in ([], [0]):
            refusals.append(f"{where}: copy span {cycle.get('copy_spans')!r} (retrieval, not the head)")
        if cycle.get("copy_decisions") not in ([], ["off"]):
            refusals.append(f"{where}: copy decision {cycle.get('copy_decisions')!r}")
        if cycle.get("relaxed_accepts") not in ([], [0]):
            refusals.append(f"{where}: relaxed accepts {cycle.get('relaxed_accepts')!r}")
        if cycle.get("draft_feature_rows") != 0:
            refusals.append(f"{where}: confidence probe features present")
        zero = cycle.get("zero_fast_path") is True
        commit = by_cycle.get(i)
        if zero:
            if depth != 0:
                refusals.append(f"{where}: zero fast path at depth {depth}")
            if commit is not None:
                refusals.append(f"{where}: zero fast path was committed")
            count, terminal = 1, i == len(cycles) - 1
            zero_fast += 1
        else:
            if commit is None:
                refusals.append(f"{where}: proposal was never committed")
                continue
            count = _one(commit.get("emitted_counts"), "emitted_counts", where, refusals)
            terminal = _one(commit.get("terminal"), "terminal", where, refusals)
            if not _is_int(count) or type(terminal) is not bool:
                refusals.append(f"{where}: malformed commit vectors")
                continue
            if not 1 <= count <= accepted + 1:
                refusals.append(f"{where}: emitted {count} outside 1..{accepted + 1}")
                continue
            if not terminal and count != accepted + 1:
                refusals.append(f"{where}: nonterminal cycle delivered {count} of {accepted + 1}")
            consumed = min(count, accepted)
            committed_proposed += depth
            committed_accepted += consumed
            clipped_accepted += accepted - consumed
            if count > accepted:
                bonus += 1
            elif terminal:
                bonus_dropped += 1
            undelivered += accepted + 1 - count
            propose_zero += depth == 0
        if zero:
            bonus += 1
        if depth == 1:
            d1 += 1
            d1_accepted += accepted
        if terminal:
            terminal_at = i
        expected_tokens.extend(tokens_out[:count])
        expected_flags.extend(flags_out[:count])
        emitted += count
        per_cycle.append({"depth": depth, "accepted": accepted, "emitted": count,
                          "terminal": terminal, "zero_fast_path": zero})
    if cycles and terminal_at != len(cycles) - 1:
        refusals.append("the last traced cycle is not the terminal one")
    if emitted != len(tokens) or expected_tokens != list(tokens):
        refusals.append(f"delivered {len(tokens)} tokens but the trace accounts for {emitted}: "
                        "a token came from outside the traced head path")
    if expected_flags != [bool(flag) for flag in from_draft]:
        refusals.append("delivered from_draft flags disagree with the traced cycles")
    if len(tokens) > max_tokens:
        refusals.append(f"{len(tokens)} tokens exceed max_tokens {max_tokens}")
    if finish_reason not in ("stop", "length"):
        refusals.append(f"finish reason {finish_reason!r}")
    elif finish_reason == "length" and len(tokens) != max_tokens:
        refusals.append("length finish before max_tokens")
    if d1 == 0:
        refusals.append("zero native-head D1 engagement: no depth-1 proposal ran")
    if trace.get("nested_proposals") if isinstance(trace, dict) else 0:
        refusals.append("nested proposal calls (the segmented path calls the impl directly)")
    refusals.extend(stats_refusals(stats, len(tokens), len(cycles), committed_proposed,
                                   committed_accepted, bonus, from_draft))
    accounting = {
        "scope": ACCEPTANCE_SCOPE,
        "raw_head_d1": {"proposals": d1, "accepted": d1_accepted,
                        "rate": d1_accepted / d1 if d1 else None},
        "committed_draft": {"proposed": committed_proposed, "accepted": committed_accepted,
                            "rate": committed_accepted / committed_proposed
                            if committed_proposed else None,
                            "source": "runtime lane stats, reconciled with the trace"},
        "clipping": {"accepted_drafts_clipped": clipped_accepted,
                     "bonus_tokens_dropped": bonus_dropped,
                     "outputs_not_delivered": undelivered, "finish_reason": finish_reason,
                     "note": "at depth 1 a delivered prefix keeps the accepted draft; "
                             "only the bonus can be dropped"},
        "depth0": {"zero_fast_path_cycles": zero_fast, "depth0_proposal_cycles": propose_zero,
                   "reason": "num_draft=1 budget min(1, max_tokens - ntoks - 1) is 0 at the "
                             "final position; reported, never pooled into D1"},
        "anchor_tokens": 1 if tokens else 0,
        "cycles": len(cycles),
        "nested_commits": trace.get("nested_commits") if isinstance(trace, dict) else None,
        "per_cycle": per_cycle,
    }
    return accounting, refusals


def stats_refusals(stats, tokens, cycles, proposed, accepted, bonus, from_draft):
    """Reconcile the runtime's lane stats with the trace and the delivery."""
    if not isinstance(stats, dict):
        return ["native mtp_receipt carries no lane stats"]
    out = []
    for key in STATS_KEYS:
        if not _is_int(stats.get(key)):
            out.append(f"lane stats {key} missing or not an int")
    if out:
        return out
    for key in ("retrieval_cycles", "retrieval_proposed", "retrieval_accepted", "plain_cycles"):
        if stats[key] != 0:
            out.append(f"lane stats {key}={stats[key]}: not head-only decoding")
    if stats["plain_tokens"] != 1:
        out.append(f"lane stats plain_tokens={stats['plain_tokens']} (expected the one anchor)")
    expected = {"draft_proposed": proposed, "draft_accepted": accepted, "bonus_tokens": bonus,
                "total_emitted": tokens, "cycles": cycles, "draft_cycles": cycles}
    for key, value in expected.items():
        if stats[key] != value:
            out.append(f"lane stats {key}={stats[key]} but the trace gives {value}")
    if sum(bool(flag) for flag in from_draft) != stats["draft_accepted"]:
        out.append("delivered from_draft count disagrees with lane stats draft_accepted")
    return out


def config_refusals(config):
    if not isinstance(config, dict):
        return ["execution config is not a mapping"]
    out = []
    if config.get("num_draft") != NUM_DRAFT or not _is_int(config.get("num_draft")):
        out.append(f"self-MTP num_draft {config.get('num_draft')!r} is not {NUM_DRAFT}")
    if config.get("persistent", True) is not True:
        out.append("self-MTP config is not persistent")
    if config.get("rate_gate", False):
        out.append("rate gate enabled")
    for key in ("window_size", "speculation_router"):
        if config.get(key) is not None:
            out.append(f"self-MTP config sets {key}")
    if config.get("backend") is not None:
        out.append(f"execution backend {config.get('backend')!r} is not the native head")
    if config.get("segment_aware_cohort_size", LANES) != LANES:
        out.append("segmented cohort wider than one lane")
    return out


def generator_contract(gen, arm, config, environ=None):
    """(observed, refusals) for one freshly built generator's route switches."""
    environ = os.environ if environ is None else environ
    copy_policy = getattr(gen, "copy_draft", None)
    fly = getattr(gen, "fly_verification", None)
    observed = {
        "self_mtp": _jsonable(getattr(gen, "self_mtp", None)),
        "adaptive_mtp_depth": _jsonable(getattr(gen, "adaptive_mtp_depth", None)),
        "mtp_ordinary_handoff": getattr(gen, "mtp_ordinary_handoff", None) is not None,
        "mtp_admission": getattr(gen, "mtp_admission", None) is not None,
        "mtp_acceptance_logger": getattr(gen, "mtp_acceptance_logger", None) is not None,
        "fly_enabled": getattr(fly, "enabled", None),
        "copy_draft_enabled": getattr(copy_policy, "enabled", None),
        "host_accept_env": environ.get("MLX2_SELF_MTP_HOST_ACCEPT"),
    }
    out = []
    if arm == "ordinary":
        if observed["self_mtp"] is not None:
            out.append("ordinary generator has a self-MTP config")
        return observed, out
    if observed["self_mtp"] != _jsonable(config):
        out.append("native generator self_mtp differs from the adapter execution config")
    for key in ("mtp_ordinary_handoff", "mtp_admission", "mtp_acceptance_logger"):
        if observed[key]:
            out.append(f"native generator has {key}")
    if observed["adaptive_mtp_depth"] is not None:
        out.append("native generator has adaptive MTP depth")
    if observed["fly_enabled"] is not False:
        out.append("FLy verification not provably disabled")
    if observed["copy_draft_enabled"] is not False:
        out.append("copy draft not provably disabled")
    if observed["host_accept_env"] == "1":
        out.append("MLX2_SELF_MTP_HOST_ACCEPT=1 selects the grammar host-accept candidate")
    return observed, out


def native_receipt_refusals(receipt):
    if not isinstance(receipt, dict):
        return ["native run delivered no mtp_receipt"]
    out = []
    if receipt.get("num_draft") != NUM_DRAFT:
        out.append(f"mtp_receipt num_draft {receipt.get('num_draft')!r}")
    if receipt.get("observed_compute_widths") != [LANES]:
        out.append(f"observed compute widths {receipt.get('observed_compute_widths')!r}")
    if receipt.get("verification") != "exact":
        out.append(f"verification {receipt.get('verification')!r} is not exact")
    if receipt.get("relaxed_accepts") != 0:
        out.append("relaxed accepts in the mtp_receipt")
    if "copy_draft" in receipt:
        out.append("copy draft state in the mtp_receipt")
    if receipt.get("adaptive_depth") is not None:
        out.append("adaptive depth in the mtp_receipt")
    return out


def logprob_refusals(logprobs, token_count):
    if not isinstance(logprobs, dict):
        return ["logprob evidence missing"]
    out = []
    chosen, rows = logprobs.get("chosen"), logprobs.get("row_sha256")
    if not isinstance(chosen, list) or len(chosen) != token_count:
        out.append("chosen logprobs do not cover every delivered token")
    elif not all(type(v) is float and math.isfinite(v) and v <= LOGPROB_CEILING for v in chosen):
        out.append("a chosen-token logprob is non-finite, non-float or positive")
    if not isinstance(rows, list) or len(rows) != token_count or not all(map(_is_sha256, rows)):
        out.append("raw logprob row digests missing, incomplete or malformed")
    if logprobs.get("row_refusals") != []:
        out.extend(f"raw logprob {p}"[:200] for p in (logprobs.get("row_refusals") or ["row refusals missing"]))
    if logprobs.get("nan_rows") != 0:
        out.append(f"{logprobs.get('nan_rows')!r} logprob rows contain NaN")
    if logprobs.get("posinf_rows") != 0:
        out.append(f"{logprobs.get('posinf_rows')!r} logprob rows contain +Inf")
    return out


def state_refusals(record):
    """Snapshot requirements: target always; sidecar on native, absent on ordinary."""
    snapshot_refusal = load_oracle().snapshot_refusal
    out = []
    problem = snapshot_refusal(record.get("final_target_state"))
    if problem:
        out.append(f"final target state: {problem}")
    sidecar = record.get("final_native_sidecar")
    if record.get("arm") == "native":
        problem = snapshot_refusal(sidecar)
        if problem:
            out.append(f"native sidecar: {problem}")
    elif sidecar is not None:
        out.append("ordinary run delivered a draft sidecar")
    return out


def evaluate_run(record, *, max_tokens):
    """(accounting or None, refusals) for one run record."""
    out = [f"lane failure: {f}"[:200] for f in record.get("lane_failures", ())]
    if record.get("error"):
        out.append(f"run raised {record['error']}"[:200])
        return None, out
    tokens = record.get("tokens", [])
    out.extend(record.get("contract_refusals", ()))
    out.extend(logprob_refusals(record.get("logprobs"), len(tokens)))
    out.extend(state_refusals(record))
    counters = record.get("scheduler_counters", {})
    out.extend(f"scheduler counter {key}={counters[key]}" for key in FALLBACK_COUNTERS
               if counters.get(key))
    if not tokens:
        out.append("no token delivered")
    if record.get("arm") == "ordinary":
        trace = record.get("trace", {})
        out.extend(trace_hygiene_refusals(trace))
        if trace.get("cycles") or trace.get("commits") or trace.get("errors"):
            out.append("ordinary run made self-MTP calls")
        if record.get("mtp_receipt") is not None:
            out.append("ordinary run delivered an mtp_receipt")
        if any(record.get("from_draft", ())):
            out.append("ordinary run delivered draft tokens")
        if len(tokens) > max_tokens or record.get("finish_reason") not in ("stop", "length"):
            out.append("ordinary delivery out of bounds or unfinished")
        return None, out
    out.extend(native_receipt_refusals(record.get("mtp_receipt")))
    receipt = record.get("mtp_receipt")
    stats = receipt.get("stats") if isinstance(receipt, dict) else None
    accounting, trace_out = validate_native_trace(
        record.get("trace", {}), max_tokens=max_tokens, tokens=tokens,
        from_draft=record.get("from_draft", []), stats=stats,
        finish_reason=record.get("finish_reason"))
    return accounting, out + trace_out


def _digest_component(a, b):
    refusal = load_oracle().snapshot_refusal
    if refusal(a) or refusal(b):
        return "unavailable"
    return "equal" if a["sha256"] == b["sha256"] else "differs"


def _first_difference(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_pair(a, b, *, sidecar=False):
    tokens_a, tokens_b = a.get("tokens", []), b.get("tokens", [])
    rows_a = (a.get("logprobs") or {}).get("row_sha256", [])
    rows_b = (b.get("logprobs") or {}).get("row_sha256", [])
    chosen_a = (a.get("logprobs") or {}).get("chosen", [])
    chosen_b = (b.get("logprobs") or {}).get("chosen", [])
    diffs = [abs(x - y) for i, (x, y) in enumerate(zip(chosen_a, chosen_b))
             if i < len(tokens_a) and i < len(tokens_b) and tokens_a[i] == tokens_b[i]
             and type(x) is float and type(y) is float]
    result = {
        "tokens": "equal" if tokens_a == tokens_b else "differs",
        "first_token_difference": _first_difference(tokens_a, tokens_b),
        "logprob_rows": "equal" if rows_a == rows_b else "differs",
        "first_logprob_row_difference": _first_difference(rows_a, rows_b),
        "chosen_logprob_max_abs_diff_on_equal_tokens": max(diffs) if diffs else None,
        "target_state": _digest_component(a.get("final_target_state"), b.get("final_target_state")),
    }
    if sidecar:
        result["native_sidecar"] = _digest_component(a.get("final_native_sidecar"),
                                                     b.get("final_native_sidecar"))
    return result


EXACT_COMPONENTS = ("tokens", "logprob_rows", "target_state", "native_sidecar")


def compare_runs(runs):
    """Exact comparisons over the ABBA runs: repeats within an arm and across arms."""
    if [r.get("arm") for r in runs] != list(ARM_ORDER):
        return {}
    a1, b1, b2, a2 = runs
    return {
        "ordinary_repeat": compare_pair(a1, a2),
        "native_repeat": compare_pair(b1, b2, sidecar=True),
        "ordinary_vs_native_first": compare_pair(a1, b1),
        "ordinary_vs_native_second": compare_pair(a2, b2),
    }


def decide(runs, *, max_tokens, identity_refusals=()):
    """Evaluate every run; return the verdict sections of the receipt."""
    refusals = list(identity_refusals)
    acceptance = {}
    for index, record in enumerate(runs):
        accounting, problems = evaluate_run(record, max_tokens=max_tokens)
        label = f"run {index} {record.get('arm')}"
        refusals.extend(f"{label}: {p}" for p in problems)
        if accounting is not None:
            acceptance[label] = accounting
    if [r.get("arm") for r in runs] != list(ARM_ORDER):
        refusals.append(f"runs {[r.get('arm') for r in runs]} are not the full ABBA order")
    comparisons = compare_runs(runs)
    differing = sorted({f"{pair}.{key}" for pair, items in comparisons.items()
                        for key, value in items.items()
                        if key in EXACT_COMPONENTS and value == "differs"})
    native = [v for k, v in acceptance.items() if k.endswith("native")]
    if len(native) == 2:
        schedule = [[(c["depth"], c["accepted"], c["emitted"]) for c in n["per_cycle"]] for n in native]
        acceptance["native_repeat_trace_equal"] = schedule[0] == schedule[1]
    verdict = "refused" if refusals else ("counterexample" if differing else "parity")
    return {"verdict": verdict, "refusals": refusals, "differing_components": differing,
            "comparisons": comparisons, "acceptance": acceptance}


# ---------------------------------------------------------------- one run

def drive_run(arm, gen, module, ops, *, prompt, max_tokens, rng, trace_limit):
    """One fresh-generator B1 run; ``gen`` is closed and the trace restored on any exit."""
    record = {"arm": arm, "tokens": [], "from_draft": [], "lane_failures": [],
              "logprobs": {"chosen": [], "row_sha256": [], "row_refusals": [], "nan_rows": 0,
                           "posinf_rows": 0, "row_layouts": []}}
    snapshot_refusal = load_oracle().snapshot_refusal
    trace = ProposalTrace(module, limit=trace_limit)
    final = None
    try:
        with trace:
            kwargs = {"max_tokens": [max_tokens], "lane_rngs": [rng]}
            if arm == "native":
                kwargs["self_mtp_configs"] = [{"sampling_temp": 0.0}]
            started = ops.now()
            first = None
            (uid,) = gen.insert([list(prompt)], **kwargs)
            idle = polls = 0
            while final is None:
                polls += 1
                if polls > IDLE_POLL_LIMIT + 4 * max_tokens:
                    record["lane_failures"].append(f"unfinished after {polls - 1} polls")
                    break
                _, responses = gen.next()
                lost = gen.take_lane_failures()
                if lost:
                    record["lane_failures"].extend(str(f) for f in lost)
                    break
                idle = 0 if responses else idle + 1
                if idle > IDLE_POLL_LIMIT:
                    record["lane_failures"].append(f"no response in {IDLE_POLL_LIMIT} polls")
                    break
                for response in responses:
                    if response.uid != uid:
                        record["lane_failures"].append(f"foreign uid {response.uid!r}")
                        continue
                    if first is None:
                        ops.sync()
                        first = ops.now()
                    token = int(response.token)
                    record["tokens"].append(token)
                    record["from_draft"].append(bool(getattr(response, "from_draft", False)))
                    row, rows = ops.row(response.logprobs, token), record["logprobs"]
                    problem = snapshot_refusal(row.get("digest"))
                    rows["row_sha256"].append(None if problem else row["digest"]["sha256"])
                    if problem and len(rows["row_refusals"]) < 32:
                        rows["row_refusals"].append(f"row {len(record['tokens']) - 1}: {problem}")
                    rows["nan_rows"] += int(row.get("nan") is not False)
                    rows["posinf_rows"] += int(row.get("posinf") is not False)
                    rows["chosen"].append(_jsonable(row.get("chosen")))
                    layout = {"dtype": _jsonable(row.get("dtype")), "shape": _jsonable(row.get("shape"))}
                    if layout not in rows["row_layouts"] and len(rows["row_layouts"]) < 8:
                        rows["row_layouts"].append(layout)
                    if getattr(response, "finish_reason", None):
                        final = response
                        break
                if len(record["tokens"]) > max_tokens:
                    record["lane_failures"].append("more tokens than max_tokens")
                    break
            ops.sync()
            ended = ops.now()
        record["finish_reason"] = getattr(final, "finish_reason", None)
        record["elapsed_s"] = ended - started
        record["ttft_s"] = (first - started) if first is not None else None
        record["timing_note"] = "diagnostic wall time of one run; not a performance claim"
        record["memory"] = ops.memory()
        record["final_target_state"] = ops.digest(getattr(final, "prompt_cache", None))
        record["final_native_sidecar"] = (ops.digest(getattr(final, "mtp_state", None))
                                          if arm == "native" else
                                          None if getattr(final, "mtp_state", None) is None
                                          else {"status": "present-on-ordinary"})
        receipt = getattr(final, "mtp_receipt", None)
        record["mtp_receipt"] = None if receipt is None else _jsonable(receipt)
        record["rng_draws"] = _jsonable(getattr(final, "rng_draws", None))
        stats = getattr(gen, "scheduler_stats", None) or {}
        record["scheduler_counters"] = {
            str(k): v for k, v in list(stats.items())[:256] if _is_int(v) or type(v) is float}
        batch = getattr(gen, "_generation_batch", None)
        record["post_run_policies"] = {
            "adaptive_depth_policy": getattr(batch, "adaptive_depth_policy", None) is not None,
            "ordinary_handoff_policy": getattr(batch, "ordinary_handoff_policy", None) is not None,
        }
    except Exception as error:  # noqa: BLE001 - recorded and refused
        record["error"] = f"{type(error).__name__}: {error}"[:300]
    finally:
        try:
            gen.close()
            record["closed"] = True
        except Exception as error:  # noqa: BLE001
            record["closed"] = False
            record.setdefault("error", f"close raised {type(error).__name__}: {error}"[:300])
        record["trace"] = trace.summary()
    if any(record.get("post_run_policies", {}).values()):
        record.setdefault("contract_refusals", []).append(
            "adaptive depth or ordinary handoff policy present on the generation batch")
    if record.get("closed") is not True:
        record.setdefault("contract_refusals", []).append("generator did not close cleanly")
    return record


# ---------------------------------------------------------------- native body

class _NativeOps:
    """MLX host operations for ``drive_run`` (native body only)."""

    def __init__(self, mx, np, oracle):
        self.mx, self.np, self.oracle = mx, np, oracle

    def now(self):
        return time.perf_counter()

    def sync(self):
        self.mx.synchronize()

    def row(self, logprobs, token):
        # The original array, before any cast: dtype, shape and raw storage bits.
        digest = self.oracle.state_digest(logprobs)
        host = self.np.array(logprobs.astype(self.mx.float32)).reshape(-1)
        chosen = float(host[token]) if 0 <= token < host.size else float("nan")
        return {"digest": digest, "nan": bool(self.np.isnan(host).any()),
                "posinf": bool(self.np.isposinf(host).any()), "chosen": chosen,
                "dtype": str(logprobs.dtype), "shape": [int(n) for n in logprobs.shape]}

    def memory(self):
        mx = self.mx
        return {"peak_bytes": mx.get_peak_memory(), "active_bytes": mx.get_active_memory(),
                "cache_bytes": mx.get_cache_memory(),
                "note": "in-run samples before generator close; diagnostic only"}

    def digest(self, obj):
        return self.oracle.state_digest(obj)


def _render_prompt(tokenizer, text, prompt_format):
    if prompt_format == "chat":
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                            add_generation_prompt=True, tokenize=True,
                                            enable_thinking=False)
    else:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(text)
    ids = [int(t) for t in ids]
    if not ids or any(t < 0 for t in ids):
        raise ValueError("prompt rendered to no token ids or negative ids")
    return ids


def run_native(args, manifest, source_before):
    """The single bounded native execution (requires ``--i-own-gpu`` and a clean preflight).

    Order is part of the contract: MLX build identity, then worktree module
    paths, each refusing before the adapter or model is constructed.
    """
    sys.path.insert(0, str(ROOT / "src"))
    oracle = load_oracle()
    import gc

    import mlx.core as mx
    import numpy as np

    identity = {"source_before": source_before, "mlx": oracle.mlx_identity(mx),
                "artifact_manifest": manifest, "state_oracle": oracle_identity()}
    build_refusals = mlx_identity_refusals(identity["mlx"])
    if build_refusals:
        return refused_receipt(args, "MLX build identity (MLX imported; no mlx2 import, "
                               "adapter or model)", build_refusals, identity)

    import mlx2
    from mlx2 import contracts as contracts_module
    from mlx2 import serving as serving_module
    from mlx2.adapters import registry as registry_module
    from mlx2.runtime import generate as generate_module
    from mlx2.runtime import hybrid_speculative as hs
    from mlx2.runtime import sample_utils as sample_utils_module

    cls = registry_module.resolve_adapter(args.model, mtp=True)
    adapter_module = sys.modules[cls.__module__]
    modules = (mlx2, contracts_module, serving_module, registry_module, generate_module, hs,
               sample_utils_module, adapter_module)
    path_refusals = module_path_refusals({m.__name__: getattr(m, "__file__", None) for m in modules})
    if path_refusals:
        return refused_receipt(args, "worktree module paths (no adapter or model constructed)",
                               path_refusals, identity)

    Capability = contracts_module.Capability
    BatchGenerator = generate_module.BatchGenerator
    LaneRNG = sample_utils_module.LaneRNG
    refusals = []
    adapter = cls(args.model, require_mtp=True, execution_policy={"num_draft": NUM_DRAFT})
    model = adapter.model
    identity.update({
        "adapter": f"{cls.__module__}.{cls.__qualname__}",
        "adapter_sha256": _sha(Path(adapter_module.__file__).read_bytes()),
        "adapter_fingerprint": adapter.identity.get("fingerprint"),
        "adapter_environment": dict(getattr(adapter, "environment", {}) or {}),
    })
    if adapter.identity.get("fingerprint") != manifest.get("fingerprint"):
        refusals.append("adapter artifact fingerprint differs from the probe manifest")
    if Capability.MTP not in adapter.descriptor.capabilities or getattr(model, "mtp", None) is None:
        refusals.append("artifact has no loaded native MTP head; refusing rather than falling back")
    if getattr(adapter, "draft_model", None) is not None:
        refusals.append("an external drafter is bound; this probe is native-head only")
    config = dict(adapter.execution_config(max_lanes=LANES, prefill_step=args.prefill_step))
    refusals.extend(config_refusals(config))
    stops = list(serving_module.generation_stop_token_ids(adapter))
    prompt = _render_prompt(adapter.tokenizer, args.prompt, args.prompt_format)
    if len(prompt) > MAX_PROMPT_TOKENS:
        refusals.append(f"prompt is {len(prompt)} tokens (bound {MAX_PROMPT_TOKENS})")
    protocol = protocol_section(args)
    prompt_section = {"text_sha256": _sha(args.prompt.encode()), "format": args.prompt_format,
                      "tokens": len(prompt), "token_ids": prompt, "token_sha256": _sha_ids(prompt),
                      "stop_token_ids": stops}
    runs = []
    if not refusals:
        ops = _NativeOps(mx, np, oracle)
        stop_tokens = [[token] for token in stops]
        for arm in ARM_ORDER:
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
            mx.reset_peak_memory()
            gen = BatchGenerator(
                model, completion_batch_size=LANES, prefill_batch_size=LANES,
                prefill_step_size=args.prefill_step, stop_tokens=stop_tokens,
                **({"self_mtp": config, "adaptive_mtp_depth": None, "mtp_ordinary_handoff": None,
                    "mtp_admission": None, "fly_verification": None, "copy_draft": None,
                    "mtp_acceptance_log": None} if arm == "native" else {}))
            observed, contract = generator_contract(gen, arm, config)
            record = drive_run(arm, gen, hs, ops, prompt=prompt, max_tokens=args.max_tokens,
                               rng=LaneRNG(args.seed), trace_limit=2 * args.max_tokens + 8)
            record["contract"] = observed
            record["contract_refusals"] = contract + record.get("contract_refusals", [])
            runs.append(record)
            if record.get("error"):
                break
        after = artifact_manifest(args.model)
        if after["fingerprint"] != manifest["fingerprint"]:
            refusals.append("artifact manifest changed during the run")
    identity["source_after"] = source_identity()
    refusals.extend(f"after the run: {p}" for p in source_identity_refusals(identity["source_after"]))
    refusals.extend(source_identity_changes(source_before, identity["source_after"]))
    decision = decide(runs, max_tokens=args.max_tokens, identity_refusals=refusals)
    return {
        "schema": SCHEMA, "scope": SCOPE, "executed": True,
        "qualification": "none: diagnostic only; implemented/qualified/selected/observed-used "
                         "states are neither changed nor asserted",
        "gpu_ownership": "asserted by the caller with --i-own-gpu; not acquired or verified here",
        "not_claimed": list(NOT_CLAIMED),
        **decision,
        "protocol": protocol, "prompt": prompt_section, "self_mtp_config": _jsonable(config),
        "identity": identity, "runs": runs,
    }


# ---------------------------------------------------------------- CLI

def protocol_section(args):
    return {"lanes": LANES, "num_draft": NUM_DRAFT, "order": list(ARM_ORDER),
            "sampling": "greedy (ordinary argmax sampler; native sampling_temp 0.0)",
            "seed": args.seed, "max_tokens": args.max_tokens, "prefill_step": args.prefill_step,
            "max_prompt_tokens": MAX_PROMPT_TOKENS, "warmups": 0,
            "disabled": ["adaptive depth", "rate gate", "ordinary handoff", "admission callback",
                         "FLy relaxation", "copy draft / PLD", "acceptance logger",
                         "grammar host-accept candidate"],
            "state_scope": STATE_SCOPE, "raw_row_oracle": RAW_ROW_ORACLE}


def preflight(args):
    """Standard-library checks that must pass before any native import."""
    refusals = []
    try:
        manifest = artifact_manifest(args.model)
    except (OSError, ValueError) as error:
        manifest = None
        refusals.append(f"artifact manifest unreadable: {type(error).__name__}: {error}"[:300])
    else:
        refusals.extend(manifest_refusals(manifest))
    source = source_identity()
    refusals.extend(source_identity_refusals(source))
    return manifest, source, refusals


def refused_receipt(args, stage, refusals, identity):
    """An unexecuted refusal: no bounded run happened; never native evidence."""
    return {
        "schema": SCHEMA, "scope": SCOPE, "executed": False, "refused_at": stage,
        "verdict": "refused", "refusals": list(refusals), "differing_components": [],
        "qualification": "none: refused before any bounded run; not native evidence",
        "not_claimed": list(NOT_CLAIMED), "protocol": protocol_section(args),
        "identity": identity, "runs": [],
    }


def plan(args):
    """The unexecuted dry-run plan: identity and bounds only."""
    try:
        manifest, manifest_error = artifact_manifest(args.model), None
    except (OSError, ValueError) as error:
        manifest, manifest_error = None, f"{type(error).__name__}: {error}"[:300]
    return {
        "schema": SCHEMA, "scope": SCOPE, "executed": False, "plan_only": True,
        "verdict": "not_run",
        "qualification": "none: unexecuted plan; no model was loaded and nothing ran",
        "not_claimed": list(NOT_CLAIMED),
        "protocol": protocol_section(args),
        "prompt": {"text_sha256": _sha(args.prompt.encode()), "format": args.prompt_format,
                   "token_ids": "rendered by the adapter tokenizer at native run time"},
        "identity": {"source": source_identity(), "artifact_manifest": manifest,
                     "artifact_manifest_error": manifest_error, "state_oracle": oracle_identity()},
        "refusals_if_run": manifest_refusals(manifest) if manifest else ["artifact unreadable"],
        "native_requirements": [
            "--i-own-gpu from a caller already holding the fresh CPG GPU lease and both locks",
            "one adapter load; ABBA ordinary/native/native/ordinary at lanes=1, num_draft=1",
            "nonzero depth-1 head proposals reconciled with lane stats and delivery",
            "complete final target state on every run and native sidecar on native runs",
        ],
    }


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="write the unexecuted plan; standard library only")
    mode.add_argument("--i-own-gpu", action="store_true",
                      help="run the native body; the caller already holds the GPU lease and locks")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", required=True, help="new receipt path outside the repository")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--prompt-format", choices=("chat", "raw"), default="chat")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help=f"{MAX_TOKENS_RANGE[0]}..{MAX_TOKENS_RANGE[1]}")
    ap.add_argument("--prefill-step", type=int, default=DEFAULT_PREFILL_STEP,
                    help=f"{PREFILL_STEP_RANGE[0]}..{PREFILL_STEP_RANGE[1]}")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return ap


def resolve_args(ap, argv=None):
    a = ap.parse_args(argv)
    if not MAX_TOKENS_RANGE[0] <= a.max_tokens <= MAX_TOKENS_RANGE[1]:
        ap.error(f"--max-tokens must be {MAX_TOKENS_RANGE[0]}..{MAX_TOKENS_RANGE[1]}")
    if not PREFILL_STEP_RANGE[0] <= a.prefill_step <= PREFILL_STEP_RANGE[1]:
        ap.error(f"--prefill-step must be {PREFILL_STEP_RANGE[0]}..{PREFILL_STEP_RANGE[1]}")
    if not 0 <= a.seed < 2 ** 32:
        ap.error("--seed must be a 32-bit unsigned int")
    if not a.prompt.strip() or len(a.prompt) > MAX_PROMPT_CHARS:
        ap.error(f"--prompt must be non-empty and at most {MAX_PROMPT_CHARS} characters")
    out = Path(a.out).expanduser().resolve()
    if out == ROOT or ROOT in out.parents:
        ap.error("--out must be outside the repository (no source-tree writes)")
    if out.exists():
        ap.error("--out already exists; refusing to overwrite evidence")
    a.out = out
    return a


def write_receipt(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as handle:
        handle.write(json.dumps(receipt, indent=1, allow_nan=False) + "\n")


def main(argv=None):
    ap = build_parser()
    args = resolve_args(ap, argv)
    if args.dry_run:
        receipt = plan(args)
        write_receipt(args.out, receipt)
        print(json.dumps({k: receipt[k] for k in ("executed", "verdict", "refusals_if_run")}))
        return 0
    manifest, source, refusals = preflight(args)
    if refusals:
        receipt = refused_receipt(args, "source/artifact preflight (no native import)", refusals,
                                  {"source_before": source, "artifact_manifest": manifest,
                                   "state_oracle": oracle_identity()})
    else:
        receipt = run_native(args, manifest, source)
    write_receipt(args.out, _jsonable(receipt))
    print(json.dumps({k: receipt[k] for k in ("verdict", "refusals", "differing_components")},
                     indent=1, default=str))
    return {"parity": 0, "counterexample": 1}.get(receipt["verdict"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
