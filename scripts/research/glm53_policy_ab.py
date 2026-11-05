"""GLM-5.3 sparse-DSA cross-policy A/B and evaluator (design 9.10, R9, R10).

One process, one loaded checkpoint, direct model (no server, so no APCv2 state
to clear between cells). Every arm is a full execution-policy JSON; arms may
differ only in the six sparse-DSA keys (``glm53_indexer_pools``,
``glm53_indexer_select``, ``glm53_sparse_attention``, ``glm53_sparse_prefill``,
``glm53_tree_rows_attention``, ``glm53_tree_frontier``), which the model reads
as attributes, so the weights are loaded once and the arms switch in process.
The first arm is the reference.

Per context cell (default 1K, 8K, 16K) and arm (order rotated per repetition):
a realistic varied haystack (scripts/realistic_haystack.py) with one buried
code and a question, prefilled in fixed chunks, then greedy decode through
``glm53_ordinary_profile.run_pass`` (reused, not copied). Recorded: prefill and
decode tok/s, prefill and decode MLX peak bytes, the model's engagement
counters (``model.calls`` delta) with an engagement verdict, output ids, the
bounded top-k logprob window of every step, and the functional oracle (the
code appears in the answer). Each arm is then compared with the reference:
exact, or the first divergent position classified by ``classify_divergence``
(AGENTS.md near-tie contract: same top two, exact-tie and multi-way clauses,
0.5-nat ceiling, both continuations pass the functional oracle).

Window capture note: B1 direct greedy decode synchronizes on every token's
argmax already, so the bounded window read at each step adds no scheduling
change; there is no batching here for it to perturb.

``--kind-split-steps N`` adds the per-layer-kind decode attribution
(``glm53_ordinary_profile.kind_split``) per arm and cell, and the R10 verdict
between the smallest and largest context (MLA share of the step-time delta).

Not a qualification or a controlled performance run; tok/s here are A/B
signals only. Exit status 1 when any arm is not engaged or any comparison is
``failed`` / ``unclassified``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = str(Path.home() / "mlx-models" / "GLM-5.3-Flash-REAP50-Q3_K_M")
CHECKPOINT = (
    "GLM-5.3-Flash-REAP50-Q3_K_M.gguf@"
    "5a865918aea87a362ff8ae419581a0447929bfd68534a44d1125b394c272762d"
)
NEAR_TIE_CEILING = 0.5  # nats, AGENTS.md
SPARSE_KEYS = (
    "glm53_indexer_pools", "glm53_indexer_select", "glm53_sparse_attention",
    "glm53_sparse_prefill", "glm53_tree_rows_attention", "glm53_tree_frontier",
)
SPARSE_DEFAULTS = {
    "glm53_indexer_pools": "recompute", "glm53_indexer_select": "argsort",
    "glm53_sparse_attention": "mask_reference", "glm53_sparse_prefill": "dense_reference",
    "glm53_tree_rows_attention": "per_row", "glm53_tree_frontier": "clone",
}
PREFILL_THRESHOLDS = {"gathered_v1_8k": 8192, "gathered_v1_16k": 16384, "gathered_v1_all": 2049}
TOPK = 2048
NEEDLE_CODE = "QX-7194-LM"
NEEDLE_FACT = f"The archive access code for the Lisbon office is {NEEDLE_CODE}."
NEEDLE_QUESTION = "\n\nWhat is the archive access code for the Lisbon office? Reply with the code only."
DASHES = "‐‑‒–—―−﹘﹣－"


# ---------------------------------------------------------------------------
# Functional oracle and near-tie classification (pure Python; CPU-tested).

def normalize_dashes(text):
    """AGENTS.md: Unicode dash punctuation -> ASCII hyphen; nothing else."""
    return text.translate({ord(c): "-" for c in DASHES})


def needle_found(text, code=NEEDLE_CODE):
    return code in normalize_dashes(text)


def first_divergence(a, b):
    """Index of the first differing token, or None when the sequences are equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _window_logprob(window, token):
    for t, lp in window:
        if t == token:
            return lp
    return None


def _within(window, token, ceiling):
    """True/False when the window decides ``lp(token) >= top - ceiling``; None if it cannot.

    A token missing from a window has logprob <= the window's last entry; the
    window proves it is further than ``ceiling`` below the top only when that
    last entry already is.
    """
    top = window[0][1]
    lp = _window_logprob(window, token)
    if lp is not None:
        return lp >= top - ceiling
    if window[-1][1] < top - ceiling:
        return False
    return None


def classify_divergence(a_ids, b_ids, a_windows, b_windows, a_oracle, b_oracle,
                        *, ceiling=NEAR_TIE_CEILING):
    """Classify a cross-policy token comparison per AGENTS.md.

    ``*_windows[i]`` is the bounded top-k ``[[token, logprob], ...]`` (sorted
    by logprob, descending) of the distribution that emitted ``*_ids[i]``;
    ``*_oracle`` is the functional-oracle verdict of each arm's continuation.
    Verdicts: ``exact``; ``near_tie_equivalent`` (a pass, with the clause that
    admitted it); ``failed`` (a confident-token divergence, or a near tie whose
    continuation fails the oracle); ``unclassified`` (evidence incomplete).
    Only the first divergence is judged; later positions are consequences.
    """
    index = first_divergence(a_ids, b_ids)
    if index is None:
        return {"verdict": "exact", "first_divergence": None}
    record = {"first_divergence": index, "shared_prefix_tokens": index}
    if index >= min(len(a_ids), len(b_ids)) or index >= min(len(a_windows), len(b_windows)):
        return {**record, "verdict": "unclassified", "reason": "length divergence or no window at the divergence"}
    wa, wb = a_windows[index], b_windows[index]
    sel_a, sel_b = a_ids[index], b_ids[index]
    record.update(selected=[sel_a, sel_b], window_a=wa, window_b=wb, oracle=[bool(a_oracle), bool(b_oracle)])
    if len(wa) < 2 or len(wb) < 2:
        return {**record, "verdict": "unclassified", "reason": "window shorter than two entries"}
    values = [lp for w in (wa, wb) for _, lp in w]
    if not all(math.isfinite(v) for v in values):
        return {**record, "verdict": "failed", "reason": "nonfinite logprob at the divergence"}
    if wa[0][0] != sel_a or wb[0][0] != sel_b:
        return {**record, "verdict": "unclassified", "reason": "selected token is not its arm's window top"}
    margin_a, margin_b = wa[0][1] - wa[1][1], wb[0][1] - wb[1][1]
    record.update(margins=[margin_a, margin_b])
    clause = None
    if {wa[0][0], wa[1][0]} == {wb[0][0], wb[1][0]} and margin_a <= ceiling and margin_b <= ceiling:
        clause = "same_top_two"
    else:
        # Exact-tie clause: one arm's top two tie exactly (a third token may
        # fill the other arm's second slot); the other arm selected one of the
        # tied tokens, its margin is within the ceiling, and the tied arm's own
        # selection lies within the ceiling of the other arm's top (read from
        # the window; missing and not provably far -> no clause).
        for (w_tied, margin_tied, sel_tied, w_other, other_sel, margin_other) in (
                (wa, margin_a, sel_a, wb, sel_b, margin_b), (wb, margin_b, sel_b, wa, sel_a, margin_a)):
            if margin_tied == 0.0:
                tied = {t for t, lp in w_tied if lp == w_tied[0][1]}
                if (other_sel in tied and margin_other <= ceiling
                        and _within(w_other, sel_tied, ceiling) is True):
                    clause = "exact_tie"
                    break
    if clause is None:
        a_in_b = _within(wb, sel_a, ceiling)
        b_in_a = _within(wa, sel_b, ceiling)
        record.update(multiway=[a_in_b, b_in_a])
        if a_in_b is True and b_in_a is True:
            clause = "multi_way"
        elif a_in_b is False or b_in_a is False:
            return {**record, "verdict": "failed", "reason": "confident-token divergence (beyond the 0.5-nat ceiling)"}
        else:
            return {**record, "verdict": "unclassified", "reason": "a selected token is missing from the other window"}
    if not (a_oracle and b_oracle):
        return {**record, "verdict": "failed", "clause": clause, "reason": "a continuation fails the functional oracle"}
    return {**record, "verdict": "near_tie_equivalent", "clause": clause}


def comparison_passes(result):
    return result["verdict"] in ("exact", "near_tie_equivalent")


# ---------------------------------------------------------------------------
# Engagement: which model.calls counters a policy must move.

def expected_counters(policy, *, context_tokens, prompt_tokens=None, decode_steps=1, tree=False):
    """Counters (``Glm53GGUFModel.calls``) that must be non-zero after a run.

    ``context_tokens`` is the largest committed length the run reaches and
    ``prompt_tokens`` the prompt-phase length (default: the same); the sparse
    stages engage only past TOPK = 2048 keys, and a gathered prefill threshold
    only once a prompt query position reaches it (pos + 1 >= threshold).
    ``tree`` adds the layered tree counters (only meaningful when the run
    drives verify rows).
    """
    prompt_tokens = context_tokens if prompt_tokens is None else prompt_tokens
    get = lambda key: policy.get(key, SPARSE_DEFAULTS[key])
    want = []
    if context_tokens <= TOPK:
        return want
    want.append("pooled_indexer")
    pools, select = get("glm53_indexer_pools"), get("glm53_indexer_select")
    if pools in ("incremental_v1", "incremental_v1_audit") and decode_steps >= 4:
        want.append("indexer_pool_extend")
    if pools in ("recompute", "recompute_fixed_order"):
        want.append("indexer_pool_recompute_full")
    if pools == "incremental_v1_audit":
        want.append("indexer_pool_audit")
    want.append({"argsort": "indexer_select_argsort", "topk_v1": "indexer_select_topk",
                 "topk_v1_kernel": "indexer_select_kernel"}[select])
    attention = get("glm53_sparse_attention")
    if decode_steps:
        want.append({"mask_reference": "sparse_attention_mask", "gathered_v1": "sparse_attention_gathered",
                     "fused_decode_v1": "sparse_attention_fused"}[attention])
    threshold = PREFILL_THRESHOLDS.get(get("glm53_sparse_prefill"))
    if threshold is not None and prompt_tokens >= threshold and prompt_tokens > TOPK:
        want.append("sparse_prefill_gathered_blocks")
    if tree and get("glm53_tree_rows_attention") == "batched_v1":
        # gathered_v1 rows share one SDPA; fused rows are one launch each (4.3).
        want.append("tree_rows_attention_batched" if attention == "gathered_v1"
                    else "sparse_attention_fused_launches")
    if tree and get("glm53_tree_frontier") == "tails_v1":
        want.append("tree_frontier_materialized")
    return want


def engagement(policy, before, after, **kwargs):
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    want = expected_counters(policy, **kwargs)
    missing = [k for k in want if delta.get(k, 0) <= 0]
    # The kernel select may decline (first-use gate failed): that is the ops
    # law, not the kernel; it must be visible, never counted as engagement.
    declined = delta.get("indexer_select_kernel_declined", 0)
    return {"expected": want, "missing": missing, "engaged": not missing,
            "kernel_declined": declined, "delta": {k: v for k, v in delta.items() if v}}


# ---------------------------------------------------------------------------
# R9: per-arm float64 oracles for the pooling-law change (design 6.2).

EPS32 = 2.0**-23


def pool_bounds(raw, ape):
    """R9 (a): both pooling laws against a float64 oracle from the fp32-rounded logits.

    ``raw`` (rows, 256) committed rows (complete pools used), ``ape`` (4, 128).
    Returns ``{"pools", "a": {reference, fixed_order, between_laws}, "pooled"}``;
    each ``a`` value is max |error| / bound (8 eps32 M per law, 16 eps32 M
    between them, M = sum_j |p_j||k_j|); above 1 violates the bound.
    """
    import mlx.core as mx
    import numpy as np

    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from mlx2.runtime.models import glm53_gguf_indexer as ix

    raw32 = np.asarray(raw, np.float32)
    raw32 = raw32[: raw32.shape[0] // 4 * 4]
    ape32 = np.asarray(ape, np.float32)
    keys = raw32[:, :128].reshape(-1, 4, 128).astype(np.float64)
    logits = (raw32[:, 128:].reshape(-1, 4, 128) + ape32[None]).astype(np.float32).astype(np.float64)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    oracle = (p * keys).sum(axis=1)
    m = (np.abs(p) * np.abs(keys)).sum(axis=1)
    raw_mx, ape_mx = mx.array(raw32), mx.array(ape32)
    pooled = {
        "reference": np.array(ix.pool_reference(raw_mx[None], ape_mx)[0]),
        "fixed_order": np.array(ix.pool_fixed_order(raw_mx, ape_mx)),
    }
    tiny = np.finfo(np.float64).tiny
    a = {law: float(np.max(np.abs(v.astype(np.float64) - oracle) / np.maximum(8 * EPS32 * m, tiny)))
         for law, v in pooled.items()}
    a["between_laws"] = float(np.max(np.abs(pooled["reference"].astype(np.float64) - pooled["fixed_order"])
                                     / np.maximum(16 * EPS32 * m, tiny)))
    return {"pools": int(oracle.shape[0]), "a": a, "pooled": pooled}


def pool_law_audit(raw, ape, q, score_weights, positions, count):
    """R9 checks on real (or synthetic) rows, both pooling laws.

    ``raw`` (4P, 256) committed rows, ``ape`` (4, 128), ``q`` (L, 32, 128),
    ``score_weights`` (L, 32) pre-scaled, ``positions`` (L,). Returns:
    (a) per law, max over elements of |pooled - f64| / (8 eps32 M) and the
        inter-law |ref - fixed| / (16 eps32 M), M = sum_j |p_j||k_j| from the
        fp32-rounded logits; (b) per law, max |score_fp32 - score_f64(own
        pooled)| / (8 eps32 sum|terms|); (c) top-``count`` membership flips of
        the visible pools between the laws (``pooling_law_flip``) with both
        arms' scores. Ratios above 1 violate the bound (R9 falsified).
    """
    import mlx.core as mx
    import numpy as np

    record = pool_bounds(raw, ape)  # also puts src/ on the path
    from mlx2.runtime.models import glm53_gguf_indexer as ix

    pooled = record.pop("pooled")
    tiny = np.finfo(np.float64).tiny
    record["b"] = {}
    q32 = np.asarray(q, np.float32)
    w32 = np.asarray(score_weights, np.float32)
    pos = np.asarray(positions, np.int64)
    scores, chosen = {}, {}
    for law, value in pooled.items():
        s32 = np.array(ix.pool_scores(mx.array(q32)[None], mx.array(w32)[None], mx.array(value)[None])[0])
        k64 = value.astype(np.float64)
        dots = np.einsum("lhd,pd->lhp", q32.astype(np.float64), k64) * 128**-0.5
        exact = (np.maximum(dots, 0) * w32.astype(np.float64)[..., None]).sum(axis=1)
        terms = np.einsum("lhd,pd->lhp", np.abs(q32).astype(np.float64), np.abs(k64)) * 128**-0.5
        bound = 8 * EPS32 * (np.abs(w32).astype(np.float64)[..., None] * terms).sum(axis=1)
        record["b"][law] = float(np.max(np.abs(s32.astype(np.float64) - exact) / np.maximum(bound, tiny)))
        visible = (np.arange(s32.shape[1]) * 4 + 3)[None, :] <= pos[:, None]
        masked = np.where(visible & np.isfinite(s32), s32, -np.inf)
        scores[law] = masked
        chosen[law] = [set(np.argsort(-row, kind="stable")[: min(count, row.shape[0])].tolist()) for row in masked]
        record.setdefault("nonfinite_scores", {})[law] = int(np.count_nonzero(~np.isfinite(s32)))
    flips = []
    for row, (ref, fix) in enumerate(zip(chosen["reference"], chosen["fixed_order"])):
        out_, in_ = sorted(ref - fix), sorted(fix - ref)
        if out_ or in_:
            flips.append({
                "query": row, "position": int(pos[row]), "out": out_[:8], "in": in_[:8],
                "scores_reference": {int(i): float(scores["reference"][row, i]) for i in (out_ + in_)[:16]},
                "scores_fixed_order": {int(i): float(scores["fixed_order"][row, i]) for i in (out_ + in_)[:16]},
            })
    record["c"] = {"pooling_law_flip_queries": len(flips), "flips": flips[:16]}
    # A nonfinite ratio, pooled key or score is a violation, never a pass.
    record["violations"] = [k for k, v in {**{f"a.{k}": v for k, v in record["a"].items()},
                                             **{f"b.{k}": v for k, v in record["b"].items()}}.items()
                            if not math.isfinite(v) or v > 1.0]
    record["violations"] += [f"nonfinite_scores.{law}" for law, n in record["nonfinite_scores"].items() if n]
    record["violations"] += [f"nonfinite_pooled.{law}" for law, v in pooled.items() if not np.isfinite(v).all()]
    return record


# ---------------------------------------------------------------------------
# Shared run plumbing (also imported by scripts/gpu_check_glm53_sparse_dsa.py).

def atomic_write(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def claim_out_dir(out):
    """Create ``out``; refuse a nonempty one so retries stay distinguishable."""
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f"refusing to reuse a nonempty evidence directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


EVALUATORS = (
    "scripts/gpu_check_glm53_sparse_dsa.py", "scripts/research/glm53_policy_ab.py",
    "scripts/research/glm53_state_oracle.py", "scripts/research/glm53_ordinary_profile.py",
    "scripts/realistic_haystack.py",
)


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False).stdout.strip()


def source_record(source_revision, *, allow_mismatch=False, allow_dirty=False):
    """The mlx2 commit a run executes from; refuse a mismatch or a dirty src/."""
    head = git("rev-parse", "HEAD")
    dirty = [line for line in git("status", "--porcelain", "--", "src").splitlines() if line.strip()]
    matches = bool(source_revision) and head.startswith(source_revision)
    if not matches and not allow_mismatch:
        raise SystemExit(f"--source-revision {source_revision!r} is not HEAD {head}")
    if dirty and not allow_dirty:
        raise SystemExit("src/ has uncommitted changes; the receipt could not name the code it ran")
    import hashlib

    evaluators = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in EVALUATORS
                  if (ROOT / rel).is_file()}
    # A dirty src/ file, when allowed, is bound by content: the receipt names
    # what ran, not only that something differed from HEAD.
    dirty_sha256 = {}
    # NUL-separated porcelain: paths arrive unescaped, a rename or copy record
    # is "XY new\0old", untracked files are listed one by one.
    raw = subprocess.run(["git", "status", "--porcelain", "-z", "--untracked-files=all", "--", "src"],
                         cwd=ROOT, capture_output=True, check=True).stdout.decode()
    entries = raw.split("\0")
    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        if not entry:
            continue
        code, rel = entry[:2], entry[3:]
        if "R" in code or "C" in code:
            i += 1  # the old name follows; what ran is the new one
        if "D" in code:
            continue  # deleted in the worktree: nothing ran from it
        if not (ROOT / rel).is_file():
            raise SystemExit(f"cannot hash dirty source entry {entry!r}")
        dirty_sha256[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    return {"source_revision": source_revision, "head": head, "head_matches": matches,
            "src_dirty": dirty, "src_dirty_sha256": dirty_sha256, "scripts_dirty": [line for line in git(
                "status", "--porcelain", "--", "scripts").splitlines() if line.strip()],
            # The evaluator code that produced the verdict, even when uncommitted.
            "evaluator_sha256": evaluators}


def host_record(mx):
    def vm():
        text = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True, check=False).stdout
        return {line.split(":")[0]: line.split(":")[1].strip().rstrip(".") for line in text.splitlines()
                if line.startswith(("Swapouts", "Swapins", "Pageins"))}
    swap = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"], capture_output=True, text=True,
                          check=False).stdout.strip()
    return {"host": platform.node(), "machine": platform.machine(), "mlx": mx.__version__,
            "metal": bool(mx.metal.is_available()), "device": str(mx.default_device()),
            "device_info": {k: v for k, v in mx.device_info().items()}, "vm": vm(), "swapusage": swap,
            "python": sys.version.split()[0]}


def load_adapter(policy, model_path=MODEL):
    """Load the checkpoint once with ``policy``; refuse any other checkpoint."""
    import mlx.core as mx

    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    info = mx.device_info()
    wired = info.get("max_recommended_working_set_size")
    if wired:
        mx.set_wired_limit(wired)  # as the server's generator does (generate.py)
    started = time.perf_counter()
    adapter = Glm53GGUFAdapter(model_path, execution_policy=policy)
    seconds = time.perf_counter() - started
    checkpoint = adapter.diagnostics()["checkpoint"]
    if checkpoint != CHECKPOINT:
        adapter.close()
        raise SystemExit(f"checkpoint {checkpoint} is not {CHECKPOINT}")
    return adapter, seconds


def apply_sparse_policy(adapter, policy):
    """Switch the loaded model to ``policy``'s sparse keys (validated by the adapter's rules).

    Every other key must equal the loaded policy: weight-level laws are bound at
    load and cannot change in process. Direct model only; the adapter's
    APCv2 profile/identity is not rebound (no server, no APCv2 here).
    """
    from mlx2.adapters.glm53_gguf import SPARSE_KEYS as ADAPTER_KEYS
    from mlx2.adapters.glm53_gguf import _check_sparse_combination

    loaded = adapter.kernel_policy
    other = {k: v for k, v in policy.items() if k not in ADAPTER_KEYS}
    stray = {k for k in other if k not in loaded or loaded[k] != other[k]}
    # External (AST) keys are not part of kernel_policy; arms here carry none.
    if stray:
        raise SystemExit(f"arm changes non-sparse keys {sorted(stray)}; load a separate process for those")
    merged = {**loaded, **{k: policy.get(k, v[0]) for k, v in ADAPTER_KEYS.items()}}
    _check_sparse_combination(merged)
    for key in ADAPTER_KEYS:
        setattr(adapter.model, key.removeprefix("glm53_"), merged[key])
    adapter.kernel_policy = merged
    return merged


def needle_prompt(adapter, tokens, *, fraction=1 / 3, minimum=None):
    """Chat-rendered ids: varied haystack + buried code + question, as close to
    ``tokens`` as the haystack granularity allows and never above it.

    Bisects the haystack length. ``minimum`` (default ``tokens - 8``) is the
    shortest acceptable length; a structural gate that needs an exact boundary
    asserts its own condition on the returned length.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import realistic_haystack

    def render(chars):
        text = realistic_haystack.bury(realistic_haystack.haystack(chars, exclude=(NEEDLE_CODE,)),
                                       NEEDLE_FACT, fraction)
        return adapter.prompt_tokens({"messages": [{"role": "user", "content": text + NEEDLE_QUESTION}],
                                      "enable_thinking": False})

    minimum = tokens - 8 if minimum is None else minimum
    lo, hi, best = 0, max(64, tokens * 6), None
    for _ in range(24):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        ids = render(mid)
        if len(ids) <= tokens:
            lo = mid
            if best is None or len(ids) > len(best):
                best = ids
            if len(ids) == tokens:
                break
        else:
            hi = mid
    if best is None or len(best) < minimum:
        raise SystemExit(f"cannot render a needle prompt of {minimum}..{tokens} tokens "
                         f"(best {None if best is None else len(best)})")
    return best


def decode_text(adapter, ids):
    eos = set(getattr(adapter.tokenizer, "eos_token_ids", ()) or ())
    cut = next((i for i, t in enumerate(ids) if t in eos), len(ids))
    return adapter.tokenizer.decode(ids[:cut]), cut


def served(ids, eos_at):
    """The ids serving would emit: through the first stop token, inclusive.

    Forced continuations past a stop token are timing filler and are never
    compared."""
    return list(ids[: eos_at + 1])


def window_at(model, prompt, chunk, emitted, index, mx, k):
    """Bounded top-``k`` window of the distribution that emitted ``emitted[index]``.

    Replays the same chunked prompt phase and one-token decode steps as
    ``glm53_ordinary_profile.run_pass`` (deterministic B1), so only the first
    divergent position is captured (AGENTS.md). Returns (window, replayed argmax).
    """
    from glm53_ordinary_profile import top_window

    cache, logits = model.make_cache(), None
    for begin in range(0, len(prompt), chunk):
        piece = mx.array([prompt[begin : begin + chunk]])
        with model._prompt_scope(piece.shape[1]):
            logits = model(piece, cache, logits_to_keep=1)
        mx.eval(logits, [c.state for c in cache])
    for token in emitted[:index]:
        logits = model(mx.array([[token]]), cache, logits_to_keep=1)
    row = logits[0, -1]
    return top_window(row, mx, k), int(mx.argmax(row).item())


def compare_arms(adapter, ref, cand, prompt, chunk, k, mx, replay):
    """Classify ``cand`` against ``ref`` on their served ids; windows only at the divergence.

    ``replay(cell) -> context manager`` switches the model to that cell's
    policy for the window replay.
    """
    a, b = served(ref["output_ids"], ref["eos_at"]), served(cand["output_ids"], cand["eos_at"])
    index = first_divergence(a, b)
    if index is None:
        return classify_divergence(a, b, [], [], ref["needle_found"], cand["needle_found"])
    windows, consistent = {}, True
    for name, cell, ids in (("a", ref, a), ("b", cand, b)):
        if index >= len(ids):
            windows[name] = []
            continue
        with replay(cell):
            window, argmax = window_at(adapter.model, prompt, chunk, ids, index, mx, k)
        consistent &= argmax == ids[index]
        windows[name] = [None] * index + [window]
    result = classify_divergence(a, b, windows["a"], windows["b"], ref["needle_found"], cand["needle_found"])
    if not consistent:
        result = {**result, "verdict": "unclassified", "reason": "the window replay did not reproduce the token"}
    return {**result, "replay_consistent": consistent}


# ---------------------------------------------------------------------------
# The A/B driver.

def run_arm(adapter, ids, args, mx, profile):
    """One arm on one prompt: chunked prefill + greedy decode via profile.run_pass."""
    model = adapter.model
    before = dict(model.calls)
    peaks = {}

    def on_phase(phase):
        if phase == "decode" and "prefill" not in peaks:
            peaks["prefill"] = mx.get_peak_memory()
            mx.reset_peak_memory()

    mx.clear_cache()
    mx.reset_peak_memory()
    # IDs only: no per-step window readback in the timed pass.
    chunks, steps, out = profile.run_pass(model, ids, args.chunk, args.decode, mx, on_phase=on_phase,
                                          window=0)
    peaks["decode"] = mx.get_peak_memory()
    text, eos_at = decode_text(adapter, out)
    median = sorted(steps)[len(steps) // 2] if steps else None
    return {
        "prompt_tokens": len(ids),
        "prefill_seconds": sum(chunks),
        "prefill_tokens_per_second": len(ids) / sum(chunks),
        "decode_median_seconds": median,
        "decode_tokens_per_second": 1.0 / median if median else None,
        "prefill_peak_bytes": peaks.get("prefill"),
        "decode_peak_bytes": peaks["decode"],
        "output_ids": out,
        "eos_at": eos_at,
        "text": text,
        "needle_found": needle_found(text),
        "prefill_last_logits_sha256": profile.run_pass.prefill_logits_sha256,
        # Every emitted logits row was finite (full rows, reduced lazily).
        "finite": profile.run_pass.finite,
        "calls_before": before,
        "calls_after": dict(model.calls),
    }


def decode_rate(cells, arm, context):
    rates = [c["decode_tokens_per_second"] for c in cells
             if c["arm"] == arm and c["context"] == context and c["decode_tokens_per_second"]]
    return sorted(rates)[len(rates) // 2] if rates else None


def r1_r2(cells, arms, contexts):
    """R1: decode flat in T past TOPK (largest vs smallest context > 2048, 0.97);
    R2: gathered not slower than the mask path at 4K (0.97). Signals, not stats."""
    selecting = sorted(c for c in contexts if c > TOPK)
    out = {"R1": {}, "R2": None}
    if len(selecting) >= 2:
        lo, hi = selecting[0], selecting[-1]
        for arm in arms:
            a, b = decode_rate(cells, arm, lo), decode_rate(cells, arm, hi)
            out["R1"][arm] = {"contexts": [lo, hi], "tok_s": [a, b],
                              "falsified": None if not (a and b) else b < 0.97 * a}
    by_policy = {}
    for c in cells:
        by_policy.setdefault(c["policy"]["glm53_sparse_attention"], set()).add(c["arm"])
    if 4096 in contexts and "mask_reference" in by_policy and "gathered_v1" in by_policy:
        pairs = []
        for g in sorted(by_policy["gathered_v1"]):
            for m in sorted(by_policy["mask_reference"]):
                cg = next(c for c in cells if c["arm"] == g)
                cm = next(c for c in cells if c["arm"] == m)
                keys = ("glm53_indexer_pools", "glm53_indexer_select", "glm53_sparse_prefill")
                if all(cg["policy"][k] == cm["policy"][k] for k in keys):
                    a, b = decode_rate(cells, m, 4096), decode_rate(cells, g, 4096)
                    pairs.append({"mask_arm": m, "gathered_arm": g, "tok_s": [a, b],
                                  "falsified": None if not (a and b) else b < 0.97 * a})
        out["R2"] = pairs
    return out


def r10(cells, arm):
    """MLA share of the decode-step growth between the smallest and largest context."""
    rows = [c for c in cells if c["arm"] == arm and c.get("kind_split")]
    if len(rows) < 2:
        return None
    lo, hi = min(rows, key=lambda c: c["context"]), max(rows, key=lambda c: c["context"])
    total = hi["kind_split"]["step_ms"] - lo["kind_split"]["step_ms"]
    mla = hi["kind_split"]["ms_per_token"]["mla"] - lo["kind_split"]["ms_per_token"]["mla"]
    share = mla / total if total > 0 else None
    # No growth to attribute: inconclusive, not a pass.
    return {"contexts": [lo["context"], hi["context"]], "step_delta_ms": total, "mla_delta_ms": mla,
            "mla_share": share, "falsified": None if share is None else share < 0.9}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="new or empty evidence directory")
    parser.add_argument("--source-revision", required=True, help="the mlx2 commit this run executes from")
    parser.add_argument("--arms", type=Path, nargs="+", required=True, help="policy JSONs; the first is the reference")
    parser.add_argument("--contexts", default="1024,4096,8192,16384",
                        help="prompt lengths; 4096 serves R2, the largest vs the smallest > 2048 serves R1")
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--decode", type=int, default=48)
    parser.add_argument("--window", type=int, default=16,
                        help="bounded top-k logprob window, captured only at a first divergence (by replay)")
    parser.add_argument("--reps", type=int, default=1, help="repetitions; arm order rotates per repetition")
    parser.add_argument("--kind-split-steps", type=int, default=0)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--i-own-the-gpu", action="store_true", help="required: run only under the GPU lease")
    parser.add_argument("--allow-revision-mismatch", action="store_true")
    parser.add_argument("--allow-dirty-src", action="store_true")
    args = parser.parse_args(argv)
    if not args.i_own_the_gpu:
        raise SystemExit("GPU run: pass --i-own-the-gpu under the GPU lease")
    if len(args.arms) < 2:
        raise SystemExit("an A/B needs a reference arm and at least one candidate")
    if args.reps < 1 or args.decode < 2 or args.window < 2 or not args.contexts.strip():
        raise SystemExit("--reps >= 1, --decode >= 2, --window >= 2 and at least one context are required")
    out = claim_out_dir(args.out)
    source = source_record(args.source_revision, allow_mismatch=args.allow_revision_mismatch,
                           allow_dirty=args.allow_dirty_src)
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import glm53_ordinary_profile as profile
    import mlx.core as mx

    arms = [(path.stem, json.loads(path.read_text())) for path in args.arms]
    contexts = [int(c) for c in args.contexts.split(",")]
    record = {"schema": "mlx2.glm53-policy-ab.v1", "checkpoint": CHECKPOINT, **source,
              "host": host_record(mx), "arms": {name: policy for name, policy in arms},
              "reference_arm": arms[0][0], "contexts": contexts, "chunk": args.chunk, "decode": args.decode,
              "window": args.window, "reps": args.reps, "status": "running", "cells": [], "comparisons": [],
              "apc": "not applicable: direct model, no server/APCv2 between cells",
              "tree_policies": "not exercised: ordinary decode never builds verify rows; the tree keys of an arm"
                               " are validated and recorded only (gpu_check ... tree / memory test them)",
              "thermal": "not gated: correctness/A-B signal, not publishable statistics"}
    atomic_write(out / "ab.json", record)
    adapter = None
    try:
        adapter, record["load_seconds"] = load_adapter(arms[0][1], args.model)
        for name, policy in arms:  # validate every arm before any GPU time is spent
            apply_sparse_policy(adapter, policy)
        # Warm kernels/compiles per arm on a short prompt (not recorded).
        warm = needle_prompt(adapter, 600)
        for name, policy in arms:
            apply_sparse_policy(adapter, policy)
            profile.run_pass(adapter.model, warm, args.chunk, 2, mx)
        prompts = {}
        for context in contexts:
            ids = prompts[context] = needle_prompt(adapter, context)
            for rep in range(args.reps):
                order = arms[rep % len(arms):] + arms[: rep % len(arms)]
                for name, policy in order:
                    merged = apply_sparse_policy(adapter, policy)
                    cell = {"context": context, "rep": rep, "arm": name, "policy": merged,
                            **run_arm(adapter, ids, args, mx, profile)}
                    cell["engagement"] = engagement(merged, cell.pop("calls_before"), cell.pop("calls_after"),
                                                    context_tokens=len(ids) + args.decode,
                                                    prompt_tokens=len(ids), decode_steps=args.decode)
                    if args.kind_split_steps:
                        cell["kind_split"] = profile.kind_split(adapter.model, ids, args.chunk,
                                                                args.kind_split_steps, mx)
                    record["cells"].append(cell)
                    atomic_write(out / "ab.json", record)
                    print(f"{context:6d} rep{rep} {name:24s} prefill {cell['prefill_tokens_per_second']:7.1f} "
                          f"decode {cell['decode_tokens_per_second'] or 0:6.2f} tok/s needle {cell['needle_found']} "
                          f"engaged {cell['engagement']['engaged']} {cell['engagement']['missing']}", flush=True)
        from contextlib import contextmanager

        @contextmanager
        def replay(cell):
            apply_sparse_policy(adapter, cell["policy"])
            yield

        for cell in record["cells"]:
            if cell["arm"] == arms[0][0]:
                continue
            ref = next(c for c in record["cells"] if c["arm"] == arms[0][0]
                       and c["context"] == cell["context"] and c["rep"] == cell["rep"])
            result = compare_arms(adapter, ref, cell, prompts[cell["context"]], args.chunk, args.window, mx, replay)
            record["comparisons"].append({"context": cell["context"], "rep": cell["rep"], "reference": ref["arm"],
                                          "candidate": cell["arm"], **result})
            atomic_write(out / "ab.json", record)
        record["r10"] = {name: r10(record["cells"], name) for name, _ in arms} if args.kind_split_steps else None
        record["falsifiers"] = r1_r2(record["cells"], [name for name, _ in arms], contexts)
        failed = [c for c in record["comparisons"] if not comparison_passes(c)]
        unengaged = [(c["context"], c["arm"]) for c in record["cells"] if not c["engagement"]["engaged"]]
        reference_misses = [(c["context"], c["rep"]) for c in record["cells"]
                            if c["arm"] == arms[0][0] and not c["needle_found"]]
        nonfinite = [(c["context"], c["rep"], c["arm"]) for c in record["cells"] if not c["finite"]]
        expected_cells = len(contexts) * args.reps * len(arms)
        record["summary"] = {"comparisons_failed_or_unclassified": len(failed), "not_engaged": unengaged,
                             "reference_oracle_failures": reference_misses, "nonfinite_arms": nonfinite,
                             "cells": len(record["cells"]), "expected_cells": expected_cells}
        record["status"] = ("passed" if not failed and not unengaged and not reference_misses and not nonfinite
                            and len(record["cells"]) == expected_cells and record["comparisons"] else "failed")
    except BaseException as error:
        record.update(status="error", error=repr(error))
        raise
    finally:
        record["host_after"] = host_record(mx)
        atomic_write(out / "ab.json", record)
        if adapter is not None:
            adapter.close()
    print(json.dumps({"status": record["status"], **record.get("summary", {})}), flush=True)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
