# SPDX-License-Identifier: MIT
"""Shared helpers for the DS4.1 approximate-operation GPU quality/speed scripts.

Nothing here runs on import. The scripts that use it load the production
Source checkpoint on the GPU and must only be run by the current GPU owner.
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path.home() / 'mlx-models/DeepSeek-V4.1-Flash-1017-Source'
Q1_ROOT = Path.home() / 'mlx-models/DeepSeek-V4.1-Flash-1017-Source-Expert-Q1-g64'
HOTSET = ROOT / 'docs/audits/2026-10-25-ds41-lfu-q1/hot60-map.json'
HOT_BYTES = 60_000_000_000
AUTO_MAXIMUM_BYTES = 78_000_000_000
# Realistic, varied English/technical prose from tracked documents. Never a
# repeated-string filler: each prompt is a distinct contiguous slice.
CORPUS_GLOBS = ('docs/research/*.md', 'docs/*.md', 'docs/audits/**/*.md')


def add_arguments(parser):
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--q1-root', type=Path, default=Q1_ROOT)
    parser.add_argument('--hotset', type=Path, default=HOTSET)
    parser.add_argument('--expert-budget-bytes', type=int, default=AUTO_MAXIMUM_BYTES)
    parser.add_argument('--owner-receipt', type=Path, required=True,
                        help='GPU owner receipt (e.g. gpu.lock/owner.json) held by the caller; recorded verbatim')
    parser.add_argument('--out', type=Path, required=True, help='new or empty evidence directory')
    parser.add_argument('--continuation', type=int, default=128)
    parser.add_argument('--reps', type=int, default=2)


def prepare_out(path):
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise SystemExit('refusing to overwrite nonempty evidence directory: ' + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=str)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_revision():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True)
    return dict(commit=head, dirty=bool(dirty.strip()), dirty_files=dirty.split('\n')[:50] if dirty.strip() else [])


def checkpoint(args):
    """``<artifact name>@<fingerprint>`` exactly as .local-source-lfu/run.py binds it."""
    from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact
    from mlx2.desktop_runner import digest
    artifact = inspect_artifact(args.model)
    bank = json.loads((args.q1_root / 'manifest.json').read_text())
    binding = {'source': artifact['identity']['fingerprint'], 'q1': bank['fingerprint'],
               'hotmap': hashlib.sha256(args.hotset.read_bytes()).hexdigest()}
    return dict(checkpoint=args.model.name + f'-Hot{HOT_BYTES // 1_000_000_000}GB-Q1g64@' + digest(binding),
                original_artifact=args.model.name + '@' + artifact['identity']['fingerprint'], binding=binding)


def expert_policy(args):
    return {'mode': 'hot_original_q1', 'budget_bytes': args.expert_budget_bytes, 'hot_bytes': HOT_BYTES,
            'q1_root': str(args.q1_root.resolve()), 'hotset_path': str(args.hotset.resolve())}


def load(args, extra_policy):
    import mlx.core as mx

    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
    from mlx2.runtime.streamed_load import WeightStreamRequest
    from mlx2.serving import preload_admission_limit_bytes
    mx.set_default_device(mx.gpu)
    policy = dict(DeepSeekV41SourceAdapter.default_route_execution_policy['ordinary'])
    policy['deepseek_source_experts'] = expert_policy(args)
    policy.update(extra_policy)
    adapter = DeepSeekV41SourceAdapter(str(args.model), execution_policy=policy,
        weight_streaming=WeightStreamRequest(mode='moe_experts', budget_bytes=args.expert_budget_bytes,
            admission_limit_bytes=preload_admission_limit_bytes(), read_workers=2, bypass_page_cache=True))
    return adapter, policy


def corpus_tokens(adapter, needed):
    """Deterministic natural-text token stream from tracked documents."""
    files = sorted({p for pattern in CORPUS_GLOBS for p in ROOT.glob(pattern) if p.is_file()})
    tokens, used = [getattr(adapter.tokenizer, 'bos_token_id', None) or 0], []
    for path in files:
        text = path.read_text(errors='replace')
        if len(text) < 2000:
            continue
        ids = list(adapter.tokenizer.encode('\n\n' + text, add_special_tokens=False))
        tokens.extend(ids)
        used.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         tokens=len(ids)))
        if len(tokens) >= needed:
            break
    if len(tokens) < needed:
        raise SystemExit('tracked corpus too short for the requested context')
    return tokens[:needed], used


def _distinct_ratio(tokens):
    return len(set(tokens)) / max(1, len(tokens))


def assert_varied(tokens):
    """Refuse a degenerate repeated-string prompt (see lessons: never filler)."""
    window = tokens[-4096:]
    if _distinct_ratio(window) < 0.08:
        raise SystemExit('prompt slice is not varied natural text')


def logprobs(logits):
    import mlx.core as mx
    x = logits.astype(mx.float32).reshape(-1, logits.shape[-1])
    out = x - mx.logsumexp(x, axis=-1, keepdims=True)
    mx.eval(out)
    import numpy as np
    return np.asarray(out)


def compare(reference, candidate, targets):
    """KL(reference || candidate), top-1 agreement, and PPL of both on targets."""
    import numpy as np
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    p = np.exp(reference)
    kl = (p * (reference - candidate)).sum(-1)
    top_ref, top_cand = reference.argmax(-1), candidate.argmax(-1)
    idx = np.arange(len(targets))
    nll_ref = -reference[idx, targets]
    nll_cand = -candidate[idx, targets]
    return dict(rows=len(targets), mean_kl=float(kl.mean()), max_kl=float(kl.max()),
                top1_agree=int((top_ref == top_cand).sum()),
                ppl_reference=float(np.exp(nll_ref.mean())), ppl_candidate=float(np.exp(nll_cand.mean())),
                ppl_relative_change=float(np.exp(nll_cand.mean() - nll_ref.mean()) - 1.0),
                kl_thirds=[float(part.mean()) for part in np.array_split(kl, 3)])


def stream_counters(adapter):
    stats = dict(getattr(adapter.weight_stream, 'stats', {}) or {})
    return {k: int(stats.get(k, 0)) for k in ('bytes_read', 'cache_misses', 'cache_hits')}


def delta(before, after):
    return {k: after[k] - before[k] for k in before}


def close_cache(cache):
    for row in cache or ():
        close = getattr(row, 'close', None)
        if close is not None:
            close()


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start
