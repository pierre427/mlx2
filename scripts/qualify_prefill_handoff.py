#!/usr/bin/env python3
"""Qualification harness for approximate cross-host prefill handoff.

The steps are deliberately separate so a producer and receiver can share
only the small, resumable evidence directories needed by the next step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELL_TARGETS = (
    ("general", 1536),
    ("general", 6144),
    ("general", 12288),
    ("needle", 2048),
    ("needle", 8192),
    ("needle", 24576),
)
PLAN_CHUNK = 512
TOP1_FLOOR_SLACK = 0.01
# Wide enough to hold both arms' log-probabilities at a multi-way near tie.
SERVED_TOP_LOGPROBS = 5
BAR = {
    "doc": "docs/design/PREFILL-HANDOFF-APPROX.md",
    "committed_before_run": ["1a1de8e4f", "397b02d6c", "3363fd924"],
    "post_run_amendments": [
        "G2 top-1: arm >= floor - 0.01 (was absolute >= 0.98); Pierre 2026-10-18",
        (
            "G4: AGENTS.md exact-tie clause (tied arm margin exactly 0, other arm "
            "selected a tied token); Pierre 2026-10-18"
        ),
        (
            "G4: AGENTS.md multi-way clause (each arm's selected token within 0.5 nat "
            "of the other arm's top logprob); served top_logprobs 5; Pierre 2026-10-18"
        ),
        (
            "G5: a cell that falls back only on producer memory admission lowers the "
            "qualified ceiling instead of failing the record; Pierre 2026-10-18"
        ),
    ],
}
MAX_NEAR_TIE_MARGIN = 0.5
_DASH_TABLE = {
    code: "-"
    for code in range(sys.maxunicode + 1)
    if unicodedata.category(chr(code)) == "Pd"
}
_DASH_TABLE.update({0x2212: "-", 0x00AD: "-"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(value)
    temporary.replace(path)


def git_commit(override: str | None = None) -> str:
    if override:
        return override
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def prepare_evidence_dir(directory: Path, step: str, *, resume: bool) -> None:
    """Create an evidence directory, refusing prior contents unless resuming."""
    directory = directory.resolve()
    entries = list(directory.iterdir()) if directory.exists() else []
    if entries and not resume:
        raise RuntimeError(f"refusing nonempty evidence directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / ".prefill-handoff-step"
    if resume and entries:
        if not marker.is_file() or marker.read_text().strip() != step:
            raise RuntimeError(
                f"cannot resume {step!r} in evidence directory for another step"
            )
    elif not entries:
        temporary = marker.with_name(f"{marker.name}.tmp.{os.getpid()}")
        temporary.write_text(step + "\n")
        temporary.replace(marker)


def _handoff():
    from mlx2.runtime import prefill_handoff

    return prefill_handoff


def _model_identity(model_path: Path, commit: str | None = None) -> dict[str, str]:
    from mlx2.adapters.registry import inspect_model

    model_path = model_path.expanduser().resolve()
    fingerprint = inspect_model(model_path).artifact_fingerprint
    return {
        "model_checkpoint": f"{model_path.name}@{fingerprint[:8]}",
        "artifact_name": model_path.name,
        "artifact_fingerprint": fingerprint,
        "mlx2_commit": git_commit(commit),
    }


def tokenizer_digest(model_path: Path) -> str:
    """Return the adapter-owned tokenizer binding advertised by the server."""
    from mlx2.adapters.registry import inspect_model

    resolution = inspect_model(model_path)
    identity = resolution.artifact.get("identity", resolution.artifact)
    if not isinstance(identity, dict):
        raise TypeError("model inspection returned no artifact identity")
    digest = identity.get("tokenizer_digest", resolution.artifact_fingerprint)
    if not isinstance(digest, str) or not digest:
        raise ValueError("model inspection returned no tokenizer digest")
    return digest


def _load_tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    model_path = model_path.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    template = model_path / "chat_template.jinja"
    if template.is_file():
        tokenizer.chat_template = template.read_text()
    return tokenizer


def _render_shell(tokenizer, content: str, marker: str) -> tuple[list[int], list[int]]:
    messages = [{"role": "user", "content": content}]
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if marker not in rendered:
        raise RuntimeError("chat template lost the context filler marker")
    prefix, suffix = rendered.split(marker, 1)
    encode = lambda text: [
        int(v) for v in tokenizer.encode(text, add_special_tokens=False)
    ]
    return encode(prefix), encode(suffix)


def _corpus_tokens(tokenizer) -> list[int]:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import realistic_haystack

        corpus = realistic_haystack.artifact_corpus(ROOT)
    finally:
        sys.path.pop(0)
    return [int(v) for v in tokenizer.encode(corpus, add_special_tokens=False)]


def _build_cell_tokens(
    tokenizer, corpus: Sequence[int], kind: str, target: int, index: int
) -> tuple[list[int], str | None, list[dict[str, str]]]:
    marker = "<|MLX2_CONTEXT_FILLER|>"
    needle = None
    if kind == "needle":
        code = (
            hashlib.sha256(f"prefill-handoff-{target}-{index}".encode())
            .hexdigest()[:10]
            .upper()
        )
        needle = f"NEEDLE-PH-{code}"
        content = (
            "Read the varied technical record below. Preserve exact identifiers.\n"
            f"{marker}\n"
            "What exact deployment code appears in the operational note? "
            "Begin with that code."
        )
        note = (
            "\nOperational note: the exact deployment code is "
            f"{needle}. Keep every character unchanged.\n"
        )
    else:
        content = (
            "Read the varied technical record below.\n"
            f"{marker}\n"
            "Summarize the main engineering constraints and give one concrete risk."
        )
        note = ""
    prefix, suffix = _render_shell(tokenizer, content, marker)
    note_tokens = [int(v) for v in tokenizer.encode(note, add_special_tokens=False)]
    required = target - len(prefix) - len(suffix) - len(note_tokens)
    if required < 1:
        raise RuntimeError(f"prompt shell exceeds target {target}")
    # Fixed, different offsets avoid making the six prompts prefixes of one another.
    offset = (index * 7919) % max(1, len(corpus) - target + 1)
    decode = lambda values: tokenizer.decode(
        values, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    def build(filler_count: int) -> tuple[list[int], list[dict[str, str]]]:
        filler = list(corpus[offset : offset + filler_count])
        if len(filler) != filler_count:
            raise RuntimeError(f"realistic haystack is too short for {target} tokens")
        cut = filler_count // 2 if kind == "needle" else filler_count
        user_content = content.replace(
            marker,
            decode(filler[:cut]) + note + decode(filler[cut:]),
        )
        messages = [{"role": "user", "content": user_content}]
        kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "enable_thinking": False,
        }
        try:
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking")
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(rendered, str):
            values = tokenizer.encode(rendered, add_special_tokens=False)
        elif hasattr(rendered, "keys") and "input_ids" in rendered:
            # transformers v5 returns a BatchEncoding for tokenize=True.
            values = rendered["input_ids"]
        else:
            values = rendered
        if hasattr(values, "tolist"):
            values = values.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        return [int(value) for value in values], messages

    filler_count = required
    tried: set[int] = set()
    for _ in range(32):
        if filler_count in tried or not 1 <= filler_count <= len(corpus) - offset:
            break
        tried.add(filler_count)
        tokens, messages = build(filler_count)
        if len(tokens) == target:
            return tokens, needle, messages
        filler_count += target - len(tokens)
    for filler_count in range(max(1, required - 64), required + 65):
        if filler_count in tried or filler_count > len(corpus) - offset:
            continue
        tokens, messages = build(filler_count)
        if len(tokens) == target:
            return tokens, needle, messages
    raise RuntimeError(f"could not construct an exact {target}-token chat cell")


def _server_plan(server: str, token_file: Path, model_id: str, messages) -> dict:
    request = urllib.request.Request(
        server.rstrip("/") + "/v1/admin/prefill-handoff/plan",
        data=json.dumps({"model": model_id, "messages": messages}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_admin_token(token_file)}",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def run_cells(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    prepare_evidence_dir(out, "cells", resume=args.resume)
    identity = _model_identity(args.model, args.commit)
    identity["tokenizer_digest"] = tokenizer_digest(args.model)
    tokenizer = _load_tokenizer(args.model)
    corpus = _corpus_tokens(tokenizer)
    cells = []
    for index, (kind, target) in enumerate(CELL_TARGETS, 1):
        cell_id = f"{kind}-{target:05d}"
        path = out / "cells" / f"{cell_id}.json"
        if args.resume and path.is_file():
            cell = json.loads(path.read_text())
        else:
            tokens, needle, messages = _build_cell_tokens(
                tokenizer, corpus, kind, target, index
            )
            plan = _handoff().make_plan(len(tokens), PLAN_CHUNK)
            plan_source = "harness_default"
            if args.plan_server:
                # The decode node owns tokens and plan (committed bar): take
                # both from a live server's /plan for these exact messages.
                served = _server_plan(
                    args.plan_server, args.plan_admin_token_file, args.model_id, messages
                )
                tokens, plan = [int(v) for v in served["token_ids"]], served["plan"]
                plan_source = f"server_plan:{args.plan_server}"
                if served.get("artifact_fingerprint") != identity["artifact_fingerprint"]:
                    raise RuntimeError("plan server serves a different artifact")
            cell = {
                **identity,
                "cell_id": cell_id,
                "kind": kind,
                "token_ids": tokens,
                "prompt_tokens": len(tokens),
                "needle": needle,
                "messages": messages,
                "plan": plan,
                "plan_source": plan_source,
            }
            atomic_json(path, cell)
        cells.append(cell)
    atomic_json(out / "cells.json", cells)


def _load_cells(directory: Path) -> list[dict[str, Any]]:
    path = directory / "cells.json" if directory.is_dir() else directory
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not value:
        raise ValueError("cells.json must be a nonempty JSON list")
    return value


def _common_identity(cells: Sequence[dict[str, Any]]) -> dict[str, str]:
    keys = (
        "model_checkpoint",
        "artifact_name",
        "artifact_fingerprint",
        "tokenizer_digest",
        "mlx2_commit",
    )
    identity = {key: cells[0][key] for key in keys}
    if any(
        any(cell.get(key) != value for key, value in identity.items()) for cell in cells
    ):
        raise ValueError("cell identities are inconsistent")
    return identity


def _admin_token(path: Path) -> str:
    token = path.expanduser().read_text().strip()
    if not token:
        raise ValueError("admin token file is empty")
    return token


def _server_handoff_metadata(
    server: str, token: str, fingerprint: str
) -> dict[str, Any]:
    request = urllib.request.Request(
        server.rstrip("/") + "/v1/models",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=60.0) as response:
        value = json.loads(response.read())
    rows = value.get("data", []) if isinstance(value, dict) else []
    matches = [
        row["prefill_handoff"]
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("prefill_handoff"), dict)
        and row["prefill_handoff"].get("artifact_fingerprint") == fingerprint
        and row["prefill_handoff"].get("export") is True
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one export-enabled server model for {fingerprint}, got {len(matches)}"
        )
    return matches[0]


def run_export(args: argparse.Namespace) -> None:
    cells = _load_cells(args.cells)
    identity = _common_identity(cells)
    if args.commit and identity["mlx2_commit"] != args.commit:
        raise ValueError("--commit differs from cells.json")
    out = args.out.resolve()
    prepare_evidence_dir(out, "export", resume=args.resume)
    token = _admin_token(args.admin_token_file)
    server_metadata = _server_handoff_metadata(
        args.server, token, identity["artifact_fingerprint"]
    )
    if server_metadata.get("tokenizer_digest") != identity["tokenizer_digest"]:
        raise RuntimeError("producer tokenizer digest differs from cells.json")
    url = args.server.rstrip("/") + "/v1/admin/prefill-handoff/export"
    for cell in cells:
        blob_path = out / f"{cell['cell_id']}.handoff"
        json_path = out / f"{cell['cell_id']}.export.json"
        if args.resume and blob_path.is_file() and json_path.is_file():
            continue
        plan = cell["plan"]
        payload = {
            "token_ids": cell["token_ids"],
            "plan": plan,
            "plan_digest": _handoff().plan_digest(plan),
            "expected": {
                "artifact_fingerprint": identity["artifact_fingerprint"],
                "tokenizer_digest": identity["tokenizer_digest"],
                "layout_fingerprint": server_metadata["layout_fingerprint"],
            },
        }
        request = urllib.request.Request(
            url,
            data=_canonical(payload),
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                blob = response.read()
                headers = dict(response.headers.items())
                status = response.status
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(
                f"export failed for {cell['cell_id']}: HTTP {error.code}: {detail}"
            ) from error
        atomic_bytes(blob_path, blob)
        atomic_json(
            json_path,
            {
                **identity,
                "cell_id": cell["cell_id"],
                "http_status": status,
                "response_headers": headers,
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            },
        )
    atomic_json(
        out / "export.json",
        {
            **identity,
            "server": args.server,
            "producer_host": platform.node(),
            "producer": server_metadata,
            "cells": [
                json.loads((out / f"{c['cell_id']}.export.json").read_text())
                for c in cells
            ],
        },
    )


def swapouts() -> int:
    try:
        output = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return -1
    for line in output.splitlines():
        if "Swapouts" in line:
            try:
                return int(line.split()[-1].rstrip("."))
            except ValueError:
                return -1
    return -1


def kl_divergence(ref_logits: Sequence[float], other_logits: Sequence[float]) -> float:
    """KL(softmax(ref) || softmax(other)), using stable float64 arithmetic."""
    ref = np.asarray(ref_logits, dtype=np.float64)
    other = np.asarray(other_logits, dtype=np.float64)
    if ref.shape != other.shape or ref.ndim != 1:
        raise ValueError("KL logits must be same-shaped vectors")
    ref_lp = ref - (np.max(ref) + math.log(float(np.exp(ref - np.max(ref)).sum())))
    other_lp = other - (
        np.max(other) + math.log(float(np.exp(other - np.max(other)).sum()))
    )
    probability = np.exp(ref_lp)
    return float(np.sum(probability * (ref_lp - other_lp)))


def has_repetition_loop(
    tokens: Sequence[int], *, window: int = 32, repeats: int = 3
) -> bool:
    """Whether any exact token window occurs at least ``repeats`` times."""
    if window <= 0 or repeats <= 1:
        raise ValueError("loop window must be positive and repeats at least two")
    counts: dict[tuple[int, ...], int] = {}
    for index in range(len(tokens) - window + 1):
        key = tuple(int(v) for v in tokens[index : index + window])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= repeats:
            return True
    return False


def normalize_dashes(text: str) -> str:
    return text.translate(_DASH_TABLE)


def needle_match(content: str, needle: str) -> bool:
    return normalize_dashes(needle) in normalize_dashes(content)


def near_tie_passes(
    divergence: dict[str, Any] | None, *, functional_pass: bool
) -> bool:
    """Apply the committed G4 near-tie rule to a first-divergence receipt."""
    if (
        not divergence
        or not functional_pass
        or divergence.get("shared_prefix") is not True
    ):
        return False
    # The multi-way clause stands on its own: shared prefix, oracle, and each
    # arm's pick within the ceiling of the other arm's top logprob.
    if multi_way_near_tie(divergence):
        return True
    ref_top2 = divergence.get("ref_top2")
    arm_top2 = divergence.get("arm_top2")
    if not isinstance(ref_top2, list) or not isinstance(arm_top2, list):
        return False
    if len(ref_top2) != 2 or len(arm_top2) != 2:
        return False
    margins = (divergence.get("ref_margin"), divergence.get("arm_margin"))
    if not all(
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= MAX_NEAR_TIE_MARGIN
        for value in margins
    ):
        return False
    if set(ref_top2) == set(arm_top2):
        return True
    # AGENTS.md exact-tie clause (Pierre, 2026-10-18, post-run amendment):
    # an arm whose margin is exactly zero has two tied tokens and a third can
    # fill the other arm's second slot; the other arm must have selected one
    # of the tied tokens.
    for tied_top2, tied_margin, other_selected in (
        (ref_top2, margins[0], divergence.get("arm_selected")),
        (arm_top2, margins[1], divergence.get("ref_selected")),
    ):
        if float(tied_margin) == 0.0 and other_selected in tied_top2:
            return True
    return False


def _window_gap(window: Any, token: Any) -> float | None:
    """Top logprob minus ``token``'s logprob in a window; None when unknown."""
    if not isinstance(window, list) or not window:
        return None
    rows = [row for row in window if isinstance(row, list) and len(row) == 2]
    if not rows:
        return None
    top = float(rows[0][1])
    for token_id, value in rows:
        if token_id == token:
            return top - float(value)
    # Absent: it is at least as far down as the window's last entry.
    floor_gap = top - float(rows[-1][1])
    return math.inf if floor_gap > MAX_NEAR_TIE_MARGIN else None


def multi_way_near_tie(divergence: dict[str, Any]) -> bool:
    """AGENTS.md multi-way clause (Pierre, 2026-10-18, post-run amendment):
    each arm's selected token within 0.5 nat of the other arm's top logprob,
    read from that position's evidence window; unknown stays unclassified."""
    ref_gap = _window_gap(divergence.get("ref_window"), divergence.get("arm_selected"))
    arm_gap = _window_gap(divergence.get("arm_window"), divergence.get("ref_selected"))
    return (
        ref_gap is not None
        and arm_gap is not None
        and 0.0 <= ref_gap <= MAX_NEAR_TIE_MARGIN
        and 0.0 <= arm_gap <= MAX_NEAR_TIE_MARGIN
    )


def evaluate_g2(cell: dict[str, Any]) -> dict[str, Any]:
    floor = cell["floor"]
    arm = cell["arm_teacher_forced"]
    mean_limit = max(4.0 * float(floor["kl_mean"]), 2e-3)
    max_limit = max(4.0 * float(floor["kl_max"]), 0.05)
    checks = {
        "mean_kl": float(arm["kl_mean"]) <= mean_limit,
        "max_step_kl": float(arm["kl_max"]) <= max_limit,
        # Post-run amendment (Pierre, 2026-10-18): floor-relative, because the
        # same-host floor arm scored 0.965-0.977 against the original absolute
        # 0.98 bar.
        "top1_agreement": float(arm["top1_agreement"])
        >= float(floor["top1_agreement"]) - TOP1_FLOOR_SLACK,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "arm_mean_kl": float(arm["kl_mean"]),
        "mean_kl_limit": mean_limit,
        "arm_max_step_kl": float(arm["kl_max"]),
        "max_step_kl_limit": max_limit,
        "top1_agreement": float(arm["top1_agreement"]),
        "top1_limit": float(floor["top1_agreement"]) - TOP1_FLOOR_SLACK,
    }


def _cache_close(cache: Any) -> None:
    if cache is None:
        return
    for entry in cache:
        close = getattr(entry, "close", None)
        if callable(close):
            close()


def _top2(logits, mx) -> tuple[list[int], list[float]]:
    del mx
    host = np.asarray(logits).astype(np.float64, copy=False)
    indexes = np.argpartition(host, -2)[-2:]
    values = host[indexes]
    order = np.argsort(-values)
    return [int(indexes[i]) for i in order], [float(values[i]) for i in order]


def _decode_ref(model, cache, tokens, plan, eos, max_new, tokenizer, mx):
    _handoff().prefill_prefix(model, cache, tokens, plan)
    logits = model(mx.array([[int(tokens[-1])]]), cache=cache)[0, -1].astype(mx.float32)
    mx.eval(logits)
    generated, distributions, top2, selected_logprobs = [], [], [], []
    stop_reason = "length"
    for step in range(max_new):
        host_logits = np.asarray(logits).astype(np.float32, copy=True)
        ids, values = _top2(logits, mx)
        token = ids[0]
        distributions.append(host_logits)
        top2.append({"tokens": ids, "logits": values})
        maximum = float(np.max(host_logits))
        log_normalizer = maximum + math.log(
            float(np.exp(host_logits.astype(np.float64) - maximum).sum())
        )
        selected_logprobs.append(float(host_logits[token]) - log_normalizer)
        generated.append(token)
        if token in eos:
            stop_reason = "eos"
            break
        if step + 1 < max_new:
            logits = model(mx.array([[token]]), cache=cache)[0, -1].astype(mx.float32)
            mx.eval(logits)
    return {
        "tokens": generated,
        "text": tokenizer.decode(generated),
        "stop_reason": stop_reason,
        "top2": top2,
        "selected_token_logprobs": selected_logprobs,
    }, distributions


def _teacher_forced_after_prefix(
    model, cache, prompt, plan, ref_tokens, ref_logits, mx
):
    if plan["prefix_tokens"] != len(prompt) - 1:
        raise ValueError("handoff plan does not reserve exactly one final token")
    logits = model(mx.array([[int(prompt[-1])]]), cache=cache)[0, -1].astype(mx.float32)
    mx.eval(logits)
    kls, agreements, selected_logprobs = [], [], []
    for index, token in enumerate(ref_tokens):
        other = np.asarray(logits).astype(np.float32, copy=False)
        kls.append(kl_divergence(ref_logits[index], other))
        agreements.append(int(np.argmax(other)) == int(token))
        maximum = float(np.max(other))
        log_normalizer = maximum + math.log(
            float(np.exp(other.astype(np.float64) - maximum).sum())
        )
        selected_logprobs.append(float(other[int(token)]) - log_normalizer)
        if index + 1 < len(ref_tokens):
            logits = model(mx.array([[int(token)]]), cache=cache)[0, -1].astype(
                mx.float32
            )
            mx.eval(logits)
    return {
        "kl": kls,
        "kl_mean": float(np.mean(kls)) if kls else 0.0,
        "kl_max": float(np.max(kls)) if kls else 0.0,
        "top1_agreement": float(np.mean(agreements)) if agreements else 1.0,
        "reference_token_logprobs": selected_logprobs,
    }


def _teacher_forced(model, cache, prompt, plan, ref_tokens, ref_logits, mx):
    """Teacher-force after locally prefilling the prompt prefix."""
    _handoff().prefill_prefix(model, cache, prompt, plan)
    return _teacher_forced_after_prefix(
        model, cache, prompt, plan, ref_tokens, ref_logits, mx
    )


def _teacher_forced_imported(
    model, cache, prompt, plan, ref_tokens, ref_logits, mx
):
    """Teacher-force an imported cache that already contains ``prompt[:-1]``."""
    return _teacher_forced_after_prefix(
        model, cache, prompt, plan, ref_tokens, ref_logits, mx
    )


def _decode_arm_free(model, cache, prompt, plan, eos, max_new, tokenizer, mx):
    # The imported cache already covers prompt[:-1]; the plan is retained in
    # the signature to make accidental local prefill at this seam conspicuous.
    if plan["prefix_tokens"] != len(prompt) - 1:
        raise ValueError("handoff plan does not reserve exactly one final token")
    logits = model(mx.array([[int(prompt[-1])]]), cache=cache)[0, -1].astype(mx.float32)
    mx.eval(logits)
    generated, top2 = [], []
    stop_reason = "length"
    for step in range(max_new):
        ids, values = _top2(logits, mx)
        token = ids[0]
        generated.append(token)
        top2.append({"tokens": ids, "logits": values})
        if token in eos:
            stop_reason = "eos"
            break
        if step + 1 < max_new:
            logits = model(mx.array([[token]]), cache=cache)[0, -1].astype(mx.float32)
            mx.eval(logits)
    return {
        "tokens": generated,
        "text": tokenizer.decode(generated),
        "stop_reason": stop_reason,
        "top2": top2,
    }


def _first_divergence(
    ref: dict[str, Any], arm: dict[str, Any]
) -> dict[str, Any] | None:
    limit = min(len(ref["tokens"]), len(arm["tokens"]))
    index = next(
        (i for i in range(limit) if ref["tokens"][i] != arm["tokens"][i]), None
    )
    if index is None:
        if len(ref["tokens"]) == len(arm["tokens"]):
            return None
        return {"index": limit, "shared_prefix": True, "reason": "length_mismatch"}
    ref_row, arm_row = ref["top2"][index], arm["top2"][index]
    return {
        "index": index,
        "shared_prefix": ref["tokens"][:index] == arm["tokens"][:index],
        "ref_selected": ref["tokens"][index],
        "arm_selected": arm["tokens"][index],
        "ref_top2": ref_row["tokens"],
        "arm_top2": arm_row["tokens"],
        "ref_margin": ref_row["logits"][0] - ref_row["logits"][1],
        "arm_margin": arm_row["logits"][0] - arm_row["logits"][1],
    }


def _host_device_class() -> str:
    from mlx2.serving import _host_device_class as device_class

    return device_class()


def _runtime_side(identity: dict[str, Any], digest: str) -> dict[str, Any]:
    return {
        "runtime_digest": digest,
        "device_class": _host_device_class(),
        "mlx": importlib.metadata.version("mlx"),
        "host": platform.node(),
    }


def run_compare(args: argparse.Namespace) -> None:
    cells = _load_cells(args.cells)
    identity = _model_identity(args.model, args.commit)
    identity["tokenizer_digest"] = tokenizer_digest(args.model)
    expected_cells = _common_identity(cells)
    for key in (
        "model_checkpoint",
        "artifact_fingerprint",
        "tokenizer_digest",
        "mlx2_commit",
    ):
        if identity[key] != expected_cells[key]:
            raise ValueError(f"receiver {key} differs from cells.json")
    out = args.out.resolve()
    prepare_evidence_dir(out, "compare", resume=args.resume)

    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter

    # The adapter pins its profile's import-time environment, so it must be
    # constructed before any mlx2.runtime model module is imported (the
    # adapter's import-order guard refuses otherwise).
    adapter = resolve_adapter(str(args.model), mtp=False, qualification_mode=True)(
        str(args.model)
    )
    from mlx2.runtime.attention_topology import apcv2_cache_layout_fingerprint
    from mlx2.runtime.models.cache import make_prompt_cache
    from mlx2.serving import runtime_identity

    model, tokenizer = (
        adapter.model,
        getattr(adapter.tokenizer, "_tokenizer", adapter.tokenizer),
    )
    layout = str(apcv2_cache_layout_fingerprint(adapter))
    receiver_runtime = runtime_identity()
    receiver_digest = _handoff().runtime_digest(receiver_runtime)
    receiver = _runtime_side(receiver_runtime, receiver_digest)
    eos = {int(v) for v in (getattr(adapter.tokenizer, "eos_token_ids", None) or [])}
    raw_eos = getattr(tokenizer, "eos_token_id", None)
    if raw_eos is not None:
        eos.update([int(raw_eos)] if isinstance(raw_eos, int) else map(int, raw_eos))

    export_summary = args.handoffs / "export.json"
    producer_host = "unknown"
    if export_summary.is_file():
        producer_host = json.loads(export_summary.read_text()).get(
            "producer_host", "unknown"
        )
    if producer_host == "unknown":
        producer_host = urllib.parse.urlparse(str(args.handoffs)).hostname or "unknown"
    results = []
    for cell in cells:
        cell_path = out / "cells" / f"{cell['cell_id']}.json"
        if args.resume and cell_path.is_file():
            results.append(json.loads(cell_path.read_text()))
            continue
        before = swapouts()
        prompt, plan = cell["token_ids"], cell["plan"]
        cache = make_prompt_cache(model)
        ref, ref_logits = _decode_ref(
            model, cache, prompt, plan, eos, args.max_new, tokenizer, mx
        )
        _cache_close(cache)
        del cache
        mx.clear_cache()

        floor_plan = _handoff().make_plan(
            len(prompt), 2048 if int(plan["chunk"]) == 512 else 512
        )
        cache = make_prompt_cache(model)
        floor = _teacher_forced(
            model, cache, prompt, floor_plan, ref["tokens"], ref_logits, mx
        )
        _cache_close(cache)
        del cache
        mx.clear_cache()

        expected = {
            "artifact_fingerprint": identity["artifact_fingerprint"],
            "tokenizer_digest": identity["tokenizer_digest"],
            "layout_fingerprint": layout,
            "token_digest": _handoff().token_digest(prompt[:-1]),
            "plan_digest": _handoff().plan_digest(plan),
        }
        blob = (args.handoffs / f"{cell['cell_id']}.handoff").read_bytes()
        cache, manifest = _handoff().deserialize_handoff(blob, expected=expected)
        arm_teacher = _teacher_forced_imported(
            model, cache, prompt, plan, ref["tokens"], ref_logits, mx
        )
        _cache_close(cache)
        del cache
        mx.clear_cache()

        cache, manifest2 = _handoff().deserialize_handoff(blob, expected=expected)
        if manifest2 != manifest:
            raise RuntimeError("fresh handoff deserialization changed its manifest")
        arm_free = _decode_arm_free(
            model, cache, prompt, plan, eos, args.max_new, tokenizer, mx
        )
        _cache_close(cache)
        del cache, ref_logits
        mx.clear_cache()

        divergence = _first_divergence(ref, arm_free)
        result = {
            **identity,
            "cell_id": cell["cell_id"],
            "kind": cell["kind"],
            "prompt_tokens": cell["prompt_tokens"],
            "needle": cell.get("needle"),
            "plan": plan,
            "plan_digest": expected["plan_digest"],
            "producer": {**manifest["producer"], "host": producer_host},
            "receiver": receiver,
            "manifest": manifest,
            "ref": ref,
            "floor": floor,
            "arm_teacher_forced": arm_teacher,
            "arm_free": arm_free,
            "greedy_equal": ref["tokens"] == arm_free["tokens"],
            "first_divergence": divergence,
            "needle_ref": (
                needle_match(ref["text"], cell["needle"])
                if cell.get("needle")
                else None
            ),
            "needle_arm": (
                needle_match(arm_free["text"], cell["needle"])
                if cell.get("needle")
                else None
            ),
            "arm_repetition_loop": has_repetition_loop(arm_free["tokens"]),
            "swapouts_before": before,
            "swapouts_after": swapouts(),
        }
        result["swapouts_delta"] = (
            result["swapouts_after"] - before
            if before >= 0 and result["swapouts_after"] >= 0
            else None
        )
        atomic_json(cell_path, result)
        results.append(result)
    del model, adapter
    mx.clear_cache()
    atomic_json(
        out / "compare.json",
        {
            **identity,
            "receiver": receiver,
            "max_new": args.max_new,
            "cells": results,
        },
    )


def _http_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 3600.0,
) -> dict[str, Any]:
    request_headers = {"Accept": "application/json", **(headers or {})}
    data = None
    method = "GET"
    if payload is not None:
        data = _canonical(payload)
        method = "POST"
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, method=method, headers=request_headers
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object from {url}")
    return value


def _receiver_capability(
    server: str, token: str, model_id: str, *, timeout: float
) -> dict[str, Any]:
    value = _http_json(
        server.rstrip("/") + "/v1/models",
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    matches = [
        row["prefill_handoff"]
        for row in value.get("data", [])
        if isinstance(row, dict)
        and row.get("id") == model_id
        and isinstance(row.get("prefill_handoff"), dict)
        and row["prefill_handoff"].get("import") is True
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one import-enabled receiver capability for {model_id}, "
            f"got {len(matches)}"
        )
    capability = matches[0]
    for field in ("runtime_digest", "execution_digest"):
        if not isinstance(capability.get(field), str) or not capability[field]:
            raise RuntimeError(f"receiver capability has no {field}")
    return capability


def _response_evidence(value: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    choices = value.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    probabilities = (
        (choice.get("logprobs") or {}).get("content", [])
        if isinstance(choice, dict)
        else []
    )
    mlx2 = value.get("mlx2") if isinstance(value.get("mlx2"), dict) else {}
    return {
        "response": value,
        "content": content if isinstance(content, str) else str(content),
        "token_logprobs": probabilities if isinstance(probabilities, list) else [],
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "mlx2": mlx2,
        "mlx2_fabric": value.get("mlx2_fabric"),
        "timing": {
            "elapsed_ms": elapsed_ms,
            "server": value.get("timing") or mlx2.get("timing"),
        },
    }


def _served_request(
    front: str,
    payload: dict[str, Any],
    *,
    opted_in: bool,
    timeout: float,
) -> dict[str, Any]:
    headers = {}
    if opted_in:
        headers["X-Mlx2-Allow-Approximate-Prefill"] = _handoff().OPERATION
    started = time.perf_counter()
    value = _http_json(
        front.rstrip("/") + "/v1/chat/completions",
        payload=payload,
        headers=headers,
        timeout=timeout,
    )
    return _response_evidence(value, (time.perf_counter() - started) * 1000.0)


def _logprob_token_id(row: dict[str, Any]) -> int | None:
    value = row.get("id")
    return int(value) if type(value) is int else None


def _row_window(row: dict[str, Any]) -> list[list[float]]:
    """The position's evidence window as [[token_id, logprob], ...], best first."""
    candidates = [row, *(row.get("top_logprobs") or [])]
    by_id: dict[int, float] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        token_id = _logprob_token_id(candidate)
        logprob = candidate.get("logprob")
        if token_id is None or not isinstance(logprob, (int, float)):
            continue
        value = float(logprob)
        if math.isfinite(value):
            by_id[token_id] = max(value, by_id.get(token_id, -math.inf))
    return [[token, value] for token, value in sorted(by_id.items(), key=lambda i: (-i[1], i[0]))]


def _row_top_two(row: dict[str, Any]) -> tuple[list[int], list[float]] | None:
    candidates = [row, *(row.get("top_logprobs") or [])]
    by_id: dict[int, float] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        token_id = _logprob_token_id(candidate)
        logprob = candidate.get("logprob")
        if token_id is None or not isinstance(logprob, (int, float)):
            continue
        value = float(logprob)
        if not math.isfinite(value):
            continue
        by_id[token_id] = max(value, by_id.get(token_id, -math.inf))
    ranked = sorted(by_id.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) < 2:
        return None
    return [ranked[0][0], ranked[1][0]], [ranked[0][1], ranked[1][1]]


def served_first_divergence(
    ref_rows: Sequence[dict[str, Any]], arm_rows: Sequence[dict[str, Any]]
) -> dict[str, Any] | None:
    """Localize served divergence by emitted-token position, not text offsets."""
    ref_ids = [_logprob_token_id(row) for row in ref_rows]
    arm_ids = [_logprob_token_id(row) for row in arm_rows]
    limit = min(len(ref_ids), len(arm_ids))
    index = next((i for i in range(limit) if ref_ids[i] != arm_ids[i]), None)
    if index is None:
        if len(ref_ids) == len(arm_ids):
            return None
        return {"index": limit, "shared_prefix": True, "reason": "length_mismatch"}
    ref_top = _row_top_two(ref_rows[index])
    arm_top = _row_top_two(arm_rows[index])
    if ref_top is None or arm_top is None:
        return {
            "index": index,
            "shared_prefix": ref_ids[:index] == arm_ids[:index],
            "reason": "missing_top_two",
        }
    return {
        "index": index,
        "shared_prefix": ref_ids[:index] == arm_ids[:index],
        "ref_selected": ref_ids[index],
        "arm_selected": arm_ids[index],
        "ref_top2": ref_top[0],
        "arm_top2": arm_top[0],
        "ref_margin": ref_top[1][0] - ref_top[1][1],
        "arm_margin": arm_top[1][0] - arm_top[1][1],
        "ref_window": _row_window(ref_rows[index]),
        "arm_window": _row_window(arm_rows[index]),
    }


def _server_handoff_receipt(evidence: dict[str, Any]) -> dict[str, Any] | None:
    mlx2 = evidence.get("mlx2")
    if not isinstance(mlx2, dict):
        return None
    receipt = mlx2.get("prefill_handoff")
    return receipt if isinstance(receipt, dict) else None


def evaluate_served_cell(
    cell: dict[str, Any],
    opted: dict[str, Any],
    ordinary: dict[str, Any],
    *,
    producer_nid: str,
    receiver_nid: str,
    record_id: str,
    plan_evidence: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    needle = cell.get("needle")
    if cell["kind"] == "needle":
        ref_retrieved = needle_match(ordinary["content"], needle)
        arm_retrieved = needle_match(opted["content"], needle)
        # Committed bar: every needle the ordinary arm retrieves, the handoff
        # arm retrieves too.
        g3_pass = (not ref_retrieved) or arm_retrieved
    else:
        ref_retrieved = arm_retrieved = None
        g3_pass = True
    g3 = {
        "pass": g3_pass,
        "ordinary_retrieved": ref_retrieved,
        "opted_in_retrieved": arm_retrieved,
    }

    ref_rows = ordinary.get("token_logprobs", [])
    arm_rows = opted.get("token_logprobs", [])
    ref_ids = [_logprob_token_id(row) for row in ref_rows]
    arm_ids = [_logprob_token_id(row) for row in arm_rows]
    exact = ref_ids == arm_ids
    divergence = served_first_divergence(ref_rows, arm_rows)
    if cell["kind"] == "needle":
        functional = bool(arm_retrieved)
    else:
        functional = opted.get("finish_reason") in {"stop", "length", "eos"}
        functional = functional and not has_repetition_loop(
            [value for value in arm_ids if value is not None]
        )
    near_tie = (not exact) and near_tie_passes(
        divergence, functional_pass=functional
    )
    g4 = {
        "pass": exact or near_tie,
        "classification": "exact" if exact else (
            "near_tie" if near_tie else "unclassified"
        ),
        "functional_oracle": functional,
        "first_divergence": divergence,
    }

    route = opted.get("mlx2_fabric")
    ordinary_route = ordinary.get("mlx2_fabric")
    server_receipt = _server_handoff_receipt(opted)
    ordinary_server_receipt = _server_handoff_receipt(ordinary)
    plan_response = (
        plan_evidence.get("response") if isinstance(plan_evidence, dict) else None
    )
    expected_token_digest = _handoff().token_digest(cell["token_ids"][:-1])
    expected_plan_digest = _handoff().plan_digest(cell["plan"])
    checks = {
        "token_ids_match": isinstance(plan_evidence, dict)
        and plan_evidence.get("token_ids_match") is True,
        "token_digest_bound": isinstance(plan_response, dict)
        and plan_response.get("token_digest") == expected_token_digest,
        "plan_digest_bound": isinstance(plan_response, dict)
        and plan_response.get("plan_digest") == expected_plan_digest,
        "route_kind": isinstance(route, dict)
        and route.get("kind") == "approximate_prefill_offload",
        "approximate": isinstance(route, dict) and route.get("approximate") is True,
        "observed_used": isinstance(route, dict)
        and (route.get("states") or {}).get("observed_used") is True,
        "producer": isinstance(route, dict)
        and (route.get("prefill_handoff") or {}).get("producer") == producer_nid,
        "served_by": isinstance(route, dict)
        and (route.get("served_by") or {}).get("nid") == receiver_nid,
        "server_observed_used": isinstance(server_receipt, dict)
        and server_receipt.get("observed_used") is True,
        "server_fidelity": isinstance(server_receipt, dict)
        and server_receipt.get("fidelity") == "approximate",
        "route_record_id": isinstance(route, dict)
        and (route.get("prefill_handoff") or {}).get("record_id") == record_id,
        "route_record_state": isinstance(route, dict)
        and (route.get("prefill_handoff") or {}).get("record_state") == "provisional",
        "server_record_id": isinstance(server_receipt, dict)
        and server_receipt.get("record_id") == record_id,
        "server_record_state": isinstance(server_receipt, dict)
        and server_receipt.get("record_state") == "provisional",
        "provisional_unqualified": isinstance(route, dict)
        and (route.get("states") or {}).get("qualified") is False,
        "ordinary_not_approximate": isinstance(ordinary_route, dict)
        and ordinary_route.get("approximate") is not True,
        "ordinary_no_server_handoff": ordinary_server_receipt is None,
    }
    return {
        "G3": g3,
        "G4": g4,
        "G5": {"pass": all(checks.values()), "checks": checks},
    }


def run_served(args: argparse.Namespace) -> None:
    cells = _load_cells(args.cells)
    identity = _common_identity(cells)
    out = args.out.resolve()
    prepare_evidence_dir(out, "served", resume=args.resume)
    token = _admin_token(args.admin_token_file)
    capability = _receiver_capability(
        args.server, token, args.model_id, timeout=args.timeout
    )
    # Persist this before issuing any served request so an interrupted run still
    # preserves the exact execution identity used by the provisional record.
    atomic_json(out / "receiver-capability.json", capability)
    results = []
    for cell in cells:
        path = out / "cells" / f"{cell['cell_id']}.json"
        if args.resume and path.is_file():
            results.append(json.loads(path.read_text()))
            continue
        messages = cell.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"cell {cell['cell_id']} has no chat messages")
        body = {
            "model": args.model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 256,
            "logprobs": True,
            "top_logprobs": SERVED_TOP_LOGPROBS,
        }
        plan = _http_json(
            args.server.rstrip("/") + "/v1/admin/prefill-handoff/plan",
            payload={**body, "producer_runtime_digest": args.producer_runtime_digest},
            headers={"Authorization": f"Bearer {token}"},
            timeout=args.timeout,
        )
        plan_evidence = {
            "response": plan,
            "token_ids_match": plan.get("token_ids") == cell.get("token_ids"),
        }
        opted = _served_request(
            args.front, body, opted_in=True, timeout=args.timeout
        )
        ordinary = _served_request(
            args.front, body, opted_in=False, timeout=args.timeout
        )
        result = {
            "cell_id": cell["cell_id"],
            "kind": cell["kind"],
            "prompt_tokens": cell["prompt_tokens"],
            "needle": cell.get("needle"),
            "messages": messages,
            "plan": plan_evidence,
            "opted_in": opted,
            "ordinary": ordinary,
        }
        result["gates"] = evaluate_served_cell(
            cell,
            opted,
            ordinary,
            producer_nid=args.producer_nid,
            receiver_nid=args.receiver_nid,
            record_id=args.record_id,
            plan_evidence=plan_evidence,
        )
        atomic_json(path, result)
        results.append(result)
    atomic_json(
        out / "served.json",
        {
            **identity,
            "front": args.front,
            "server": args.server,
            "model_id": args.model_id,
            "producer_nid": args.producer_nid,
            "receiver_nid": args.receiver_nid,
            "record_id": args.record_id,
            "receiver_capability": capability,
            "cells": results,
        },
    )


def _read_pass_evidence(path: Path) -> tuple[bool, dict[str, Any]]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"evidence must be a JSON object: {path}")
    return value.get("pass") is True, value


def _cell_gate_results(cell: dict[str, Any]) -> dict[str, dict[str, Any]]:
    g2 = evaluate_g2(cell)
    retrieval = (
        not bool(cell.get("needle_ref")) or bool(cell.get("needle_arm"))
        if cell["kind"] == "needle"
        else True
    )
    g3 = {
        "pass": retrieval,
        "ref_retrieved": cell.get("needle_ref"),
        "arm_retrieved": cell.get("needle_arm"),
    }
    if cell.get("greedy_equal"):
        g4 = {"pass": True, "classification": "exact"}
    else:
        if cell["kind"] == "needle":
            functional = bool(cell.get("needle_arm"))
        else:
            functional = cell["arm_free"].get("stop_reason") in {
                "eos",
                "length",
            } and not cell.get("arm_repetition_loop", False)
        passed = near_tie_passes(
            cell.get("first_divergence"), functional_pass=functional
        )
        g4 = {
            "pass": passed,
            "classification": "near_tie" if passed else "unclassified",
            "functional_oracle": functional,
            "first_divergence": cell.get("first_divergence"),
        }
    return {"G2": g2, "G3": g3, "G4": g4}


def _capability_from_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("receiver capability must be a JSON object")
    capability = value.get("receiver_capability", value)
    if not isinstance(capability, dict):
        raise TypeError("receiver_capability must be a JSON object")
    for field in ("runtime_digest", "execution_digest"):
        if not isinstance(capability.get(field), str) or not capability[field]:
            raise ValueError(f"receiver capability has no {field}")
    return capability


def _load_served(path: Path) -> dict[str, Any]:
    source = path / "served.json" if path.is_dir() else path
    value = json.loads(source.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("cells"), list):
        raise TypeError("served evidence must contain a cells list")
    return value


_PRODUCER_MEMORY_REFUSAL = "does not fit ordinary execution headroom"


def producer_memory_fallback(served_cell: dict[str, Any] | None) -> bool:
    """The opted-in request fell back to ordinary serving only because the
    producer refused the export on memory admission (receipted)."""
    route = ((served_cell or {}).get("opted_in") or {}).get("mlx2_fabric") or {}
    if route.get("fallback") != "ordinary_local" or route.get("approximate") is True:
        return False
    handoff = [
        candidate
        for candidate in route.get("candidates") or []
        if candidate.get("option") == "approximate_prefill_offload"
    ]
    return bool(handoff) and all(
        candidate.get("verdict") == "fallback"
        and _PRODUCER_MEMORY_REFUSAL in str(candidate.get("reason", ""))
        for candidate in handoff
    )


def build_verdict(
    comparison: dict[str, Any],
    *,
    g1_pass: bool,
    g1_details: dict[str, Any],
    commit: str,
    g1_path: str,
    receiver_capability: dict[str, Any],
    receiver_capability_path: str,
    served: dict[str, Any] | None = None,
    served_path: str | None = None,
    provisional: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if provisional and served is not None:
        raise ValueError("a provisional verdict does not consume served evidence")
    if not provisional and served is None:
        raise ValueError("the final verdict requires served evidence")
    served_cells = {
        cell.get("cell_id"): cell
        for cell in (served or {}).get("cells", [])
        if isinstance(cell, dict)
    }
    cells = []
    for source in comparison["cells"]:
        in_process = _cell_gate_results(source)
        served_cell = served_cells.get(source["cell_id"])
        if not provisional and served_cell is None:
            raise ValueError(f"served evidence is missing {source['cell_id']}")
        served_gates = served_cell.get("gates", {}) if served_cell else {}
        g2 = in_process["G2"]
        g3 = {
            "pass": in_process["G3"]["pass"]
            and (provisional or (served_gates.get("G3") or {}).get("pass") is True),
            "in_process": in_process["G3"],
            "served": None if provisional else served_gates.get("G3"),
        }
        g4 = {
            "pass": in_process["G4"]["pass"]
            and (provisional or (served_gates.get("G4") or {}).get("pass") is True),
            "in_process": in_process["G4"],
            "served": None if provisional else served_gates.get("G4"),
        }
        g5 = (
            {"pass": None, "status": "pending", "served": None}
            if provisional
            else {
                "pass": (
                    (served_gates.get("G5") or {}).get("pass") is True
                    and isinstance(served_cell.get("plan"), dict)
                    and served_cell["plan"].get("token_ids_match") is True
                    and isinstance(served_cell["plan"].get("response"), dict)
                    and served_cell["plan"]["response"].get("token_digest")
                    == (source.get("manifest") or {}).get("token_digest")
                    and served_cell["plan"]["response"].get("plan_digest")
                    == source.get("plan_digest")
                ),
                "served": served_gates.get("G5"),
                "compare_binding": {
                    "token_ids_match": served_cell["plan"].get("token_ids_match")
                    if isinstance(served_cell.get("plan"), dict)
                    else False,
                    "token_digest": (
                        served_cell["plan"].get("response") or {}
                    ).get("token_digest")
                    == (source.get("manifest") or {}).get("token_digest")
                    if isinstance(served_cell.get("plan"), dict)
                    and isinstance(served_cell["plan"].get("response"), dict)
                    else False,
                    "plan_digest": (
                        served_cell["plan"].get("response") or {}
                    ).get("plan_digest") == source.get("plan_digest")
                    if isinstance(served_cell.get("plan"), dict)
                    and isinstance(served_cell["plan"].get("response"), dict)
                    else False,
                },
            }
        )
        gates = {"G2": g2, "G3": g3, "G4": g4, "G5": g5}
        cell_pass = all(gates[name]["pass"] for name in ("G2", "G3", "G4"))
        capacity_limited = False
        if not provisional:
            capacity_limited = (
                cell_pass
                and not g5["pass"]
                and all(g5["compare_binding"].values())
                and producer_memory_fallback(served_cell)
            )
            cell_pass = cell_pass and g5["pass"]
        cells.append(
            {
                "cell_id": source["cell_id"],
                "kind": source["kind"],
                "prompt_tokens": source["prompt_tokens"],
                "plan": source["plan"],
                "gates": gates,
                "pass": cell_pass,
                "capacity_limited": capacity_limited,
            }
        )
    # Post-run amendment (Pierre, 2026-10-18): a G5 cell whose opted-in request
    # fell back only because the producer refused the export on memory
    # admission lowers the qualified ceiling instead of failing the record.
    # Walking cells by prompt size, the ceiling is the last passing cell before
    # the first capacity-limited one; any other failure fails the record.
    ceiling, hard_failures, out_of_scope = 0, [], []
    for cell in sorted(cells, key=lambda item: item["prompt_tokens"]):
        if out_of_scope:
            out_of_scope.append(cell["cell_id"])
        elif cell["pass"]:
            ceiling = cell["prompt_tokens"]
        elif cell["capacity_limited"]:
            out_of_scope.append(cell["cell_id"])
        else:
            hard_failures.append(cell["cell_id"])
    hard_failures += [
        cell["cell_id"]
        for cell in cells
        if cell["cell_id"] in out_of_scope and not cell["pass"] and not cell["capacity_limited"]
    ]
    for cell in cells:
        cell["scope"] = "above_ceiling" if cell["cell_id"] in out_of_scope else "in_scope"
    in_scope = [cell for cell in cells if cell["scope"] == "in_scope"]
    passing = [cell["prompt_tokens"] for cell in in_scope if cell["pass"]]
    aggregate = {
        name: {
            "pass": bool(in_scope) and all(cell["gates"][name]["pass"] for cell in in_scope),
            "details": {cell["cell_id"]: cell["gates"][name] for cell in cells},
        }
        for name in ("G2", "G3", "G4", "G5")
    }
    if provisional:
        aggregate["G5"] = {
            "pass": None,
            "status": "pending",
            "details": {cell["cell_id"]: cell["gates"]["G5"] for cell in cells},
        }
    gates = {
        "G1": {"pass": g1_pass, "details": g1_details},
        **aggregate,
    }
    if provisional:
        qualified = all(gates[name]["pass"] for name in ("G1", "G2", "G3", "G4"))
    else:
        qualified = (
            all(gate["pass"] for gate in gates.values())
            and not hard_failures
            and ceiling > 0
        )
    handoff = _handoff()
    producer = dict(comparison["cells"][0]["producer"])
    receiver = dict(comparison["receiver"])
    if receiver.get("runtime_digest") != receiver_capability.get("runtime_digest"):
        raise ValueError("receiver capability runtime digest differs from compare evidence")
    receiver["execution_digest"] = receiver_capability["execution_digest"]
    for side in (producer, receiver):
        side.setdefault("host", side.get("nid", "unknown"))
    binding = {
        "operation": handoff.OPERATION,
        "operation_revision": handoff.OPERATION_REVISION,
        "artifact_fingerprint": comparison["artifact_fingerprint"],
        "producer_runtime_digest": producer.get("runtime_digest"),
        "receiver_runtime_digest": receiver.get("runtime_digest"),
        "receiver_execution_digest": receiver.get("execution_digest"),
        "mlx2_commit": commit,
    }
    record_id = hashlib.sha256(_canonical(binding)).hexdigest()[:24]
    record = {
        "schema": handoff.RECORD_SCHEMA,
        "record_id": record_id,
        "operation": handoff.OPERATION,
        "operation_revision": handoff.OPERATION_REVISION,
        "verdict": "qualified" if qualified else "failed",
        "artifact_fingerprint": comparison["artifact_fingerprint"],
        "artifact_name": comparison["artifact_name"],
        "model_checkpoint": comparison["model_checkpoint"],
        "producer": producer,
        "receiver": receiver,
        "plan_chunks": sorted({int(cell["plan"]["chunk"]) for cell in cells}),
        "max_prompt_tokens": ceiling if not provisional else max(passing, default=0),
        "capacity_ceiling": None if provisional else {
            "max_prompt_tokens": ceiling,
            "above_ceiling": out_of_scope,
            "hard_failures": sorted(set(hard_failures)),
            "rule": "producer memory-admission fallbacks lower the ceiling (Pierre 2026-10-18)",
        },
        "gates": gates,
        "cells": cells,
        "mlx2_commit": commit,
        "provisional": provisional,
        "bar": BAR,
        "evidence": [
            str(g1_path),
            "compare.json",
            str(receiver_capability_path),
            *([str(served_path)] if served_path else []),
        ],
        "created_at": datetime.now(UTC).isoformat(),
    }
    return record, cells


def run_verdict(args: argparse.Namespace) -> None:
    out = args.record_dir.resolve()
    prepare_evidence_dir(out, "verdict", resume=args.resume)
    path = args.compare / "compare.json" if args.compare.is_dir() else args.compare
    comparison = json.loads(path.read_text())
    if comparison.get("max_new") != 256:
        raise ValueError("the committed qualification bar requires max_new=256")
    source_cells = _load_cells(args.cells)
    indexed = {cell["cell_id"]: cell for cell in source_cells}
    for cell in comparison.get("cells", []):
        source = indexed.get(cell.get("cell_id"))
        if source is None or source.get("prompt_tokens") != cell.get("prompt_tokens"):
            raise ValueError("compare evidence does not match cells.json")
    commit = git_commit(args.commit)
    if comparison.get("mlx2_commit") != commit:
        raise ValueError("verdict commit differs from compare evidence")
    g1_pass, g1 = _read_pass_evidence(args.g1_evidence)
    provisional = getattr(args, "provisional", False)
    receiver_capability = _capability_from_file(args.receiver_capability)
    served = None
    if provisional:
        if args.served is not None:
            raise ValueError("--provisional does not accept --served")
    elif args.served is None:
        raise ValueError("the final verdict requires --served")
    else:
        served = _load_served(args.served)
        served_capability = served.get("receiver_capability")
        if served_capability != receiver_capability:
            raise ValueError("served receiver capability differs from supplied capability")
        if served.get("artifact_fingerprint") != comparison.get("artifact_fingerprint"):
            raise ValueError("served evidence artifact differs from compare evidence")
    record, cells = build_verdict(
        comparison,
        g1_pass=g1_pass,
        g1_details=g1,
        commit=commit,
        g1_path=str(args.g1_evidence),
        receiver_capability=receiver_capability,
        receiver_capability_path=str(args.receiver_capability),
        served=served,
        served_path=str(args.served) if args.served else None,
        provisional=provisional,
    )
    if served is not None and served.get("record_id") != record["record_id"]:
        raise ValueError("served evidence used a different provisional record id")
    record_path = out / f"{record['record_id']}.json"
    atomic_json(record_path, record)
    lines = [
        "# Approximate prefill-handoff qualification",
        "",
        f"- Checkpoint: `{record['model_checkpoint']}`",
        f"- Full fingerprint: `{record['artifact_fingerprint']}`",
        f"- mlx2 commit: `{record['mlx2_commit']}`",
        f"- Verdict: **{record['verdict']}**",
        f"- Maximum passing prompt: {record['max_prompt_tokens']} tokens",
        "",
        "| Gate | Pass |",
        "|---|---|",
    ]
    lines += [
        f"| {name} | {str(gate['pass']).lower()} |"
        for name, gate in record["gates"].items()
    ]
    lines += ["", "| Cell | Tokens | Pass |", "|---|---:|---|"]
    lines += [
        f"| {cell['cell_id']} | {cell['prompt_tokens']} | {str(cell['pass']).lower()} |"
        for cell in cells
    ]
    temporary = (out / "summary.md").with_name(f"summary.md.tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(out / "summary.md")


def _load_cell(path: Path) -> dict[str, Any]:
    candidate = path
    if not candidate.is_file() and candidate.suffix != ".json":
        candidate = candidate.with_suffix(".json")
    value = json.loads(candidate.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
        raise TypeError("confirmation cell must contain chat messages")
    return value


def run_confirm(args: argparse.Namespace) -> None:
    cell = _load_cell(args.cell)
    body = {
        "model": args.model_id,
        "messages": cell["messages"],
        "temperature": 0,
        "max_tokens": 256,
        "logprobs": True,
        "top_logprobs": SERVED_TOP_LOGPROBS,
    }
    evidence = _served_request(
        args.front, body, opted_in=True, timeout=args.timeout
    )
    route = evidence.get("mlx2_fabric")
    checks = {
        "qualified": isinstance(route, dict)
        and (route.get("states") or {}).get("qualified") is True,
        "record_state": isinstance(route, dict)
        and (route.get("prefill_handoff") or {}).get("record_state") == "final",
    }
    atomic_json(
        args.out.resolve(),
        {
            "cell_id": cell.get("cell_id"),
            "pass": all(checks.values()),
            "checks": checks,
            "opted_in": evidence,
        },
    )


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="step", required=True)
    cells = sub.add_parser("cells")
    cells.add_argument("--model", type=Path, required=True)
    cells.add_argument("--out", type=Path, required=True)
    cells.add_argument("--commit")
    cells.add_argument("--resume", action="store_true")
    cells.add_argument("--plan-server", help="take token ids and plan from this server's /plan")
    cells.add_argument("--plan-admin-token-file", type=Path)
    cells.add_argument("--model-id", help="served model id for --plan-server")
    cells.set_defaults(run=run_cells)

    export = sub.add_parser("export")
    export.add_argument("--cells", type=Path, required=True)
    export.add_argument("--server", default="http://127.0.0.1:8285")
    export.add_argument("--admin-token-file", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--timeout", type=float, default=3600.0)
    export.add_argument("--commit")
    export.add_argument("--resume", action="store_true")
    export.set_defaults(run=run_export)

    compare = sub.add_parser("compare")
    compare.add_argument("--model", type=Path, required=True)
    compare.add_argument("--cells", type=Path, required=True)
    compare.add_argument("--handoffs", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--max-new", type=int, default=256)
    compare.add_argument("--commit")
    compare.add_argument("--resume", action="store_true")
    compare.set_defaults(run=run_compare)

    served = sub.add_parser("served")
    served.add_argument("--cells", type=Path, required=True)
    served.add_argument("--front", required=True)
    served.add_argument("--server", required=True)
    served.add_argument("--model-id", required=True)
    served.add_argument("--admin-token-file", type=Path, required=True)
    served.add_argument("--out", type=Path, required=True)
    served.add_argument("--producer-nid", required=True)
    served.add_argument("--receiver-nid", required=True)
    served.add_argument("--producer-runtime-digest", required=True)
    served.add_argument("--record-id", required=True)
    served.add_argument("--timeout", type=float, default=3600.0)
    served.add_argument("--resume", action="store_true")
    served.set_defaults(run=run_served)

    verdict = sub.add_parser("verdict")
    verdict.add_argument("--compare", type=Path, required=True)
    verdict.add_argument("--cells", type=Path, required=True)
    verdict.add_argument("--record-dir", type=Path, required=True)
    verdict.add_argument("--g1-evidence", type=Path, required=True)
    verdict.add_argument("--receiver-capability", type=Path, required=True)
    verdict.add_argument("--served", type=Path)
    verdict.add_argument(
        "--provisional",
        action="store_true",
        help="evaluate G1-G4 only and mark G5 pending (for the G5 run itself)",
    )
    verdict.add_argument("--commit")
    verdict.add_argument("--resume", action="store_true")
    verdict.set_defaults(run=run_verdict)

    confirm = sub.add_parser("confirm")
    confirm.add_argument("--front", required=True)
    confirm.add_argument("--model-id", required=True)
    confirm.add_argument("--cell", type=Path, required=True)
    confirm.add_argument("--out", type=Path, required=True)
    confirm.add_argument("--timeout", type=float, default=3600.0)
    confirm.set_defaults(run=run_confirm)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
