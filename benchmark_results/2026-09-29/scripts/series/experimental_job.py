#!/usr/bin/env python3
"""Series round ``experimental``: short correctness-and-health checks of the
default-off and unqualified mechanisms that apply to one model.

This is not a performance round (except the fused-kernel group's quick decode
comparison).  Every feature runs on a real server with the feature enabled and
must show, through its receipt or a status counter, that it actually engaged:
a feature that was enabled but never executed is recorded ``engaged: false``
and is not ok.  Features are grouped into as few server starts as their
startup rules allow (``docs/SERVING.md`` default-off sections and the
``ServingEngine`` startup refusals):

* ``ordinary``: ``--ordinary`` with cache capsules, block persistence, APCv2
  junction snapshots, rolling prefill checkpoints, SRPT prefill scheduling,
  preempt-and-replay (fault-injected, qualification mode) and host memory
  signals.
* ``speculative``: the model's MTP or external-draft route with FLy relaxed
  verification and (native MTP only) self-MTP copy drafts.
* ``pld``: the prompt-lookup route with ``rotating_replay`` (sliding-window
  models only; a model without rotating KV has nothing to replay).
* ``approx``: live Spomin compaction (standard-attention adapters) or int8
  NAX prefill (adapters declaring a scope).  Both are approximate and
  qualification-mode only.
* ``lora``: concurrent multi-LoRA, only when a local LoRA exists for the
  model's family.
* ``bitexact``: ``--verify-bitexact``, only when the installed mlx has
  ``mx.metal.set_qmv_bitexact``.
* ``fused``: an unfused base arm, then one arm per fused/compiled kernel that
  applies to the family (greedy output must equal the base arm; 1- and
  4-stream decode tok/s next to the base).

Every server start also runs the round's base checks: a short chat with a
known answer, a repeated-prefix request that must hit APCv2, and a streaming
request.  A startup refusal that names a feature (e.g. junction checkpoints
on a KV-only cache) marks that feature not applicable, with the refusal text,
and the group restarts without it.

Status: ``pass`` if every applicable feature and kernel is ok, ``partial``
otherwise (``error`` only when the harness itself broke).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extra_common as xc  # noqa: E402

ROUND = "experimental"
BLOCK_BYTES = 4 << 20
ROLLING_INTERVAL = 512
SPOMIN_SEGMENT = 256
MUSE_LORA = Path(str(Path.home()) + "/Desktop/mlx-uag/muse_cyber_lora/adapters")
# Families whose attention keeps sliding-window RotatingKVCache planes
# (docs/SERVING.md: Muse Glimmer and North Mini Code "sliding-window rotating
# planes"); only there does prompt lookup's rotating replay have a ring to arm.
ROTATING_FAMILIES = {"muse", "north", "laguna"}
# Spomin's adapter-owned surgery is standard attention (Muse, North, Laguna);
# hybrid recurrent state is declined by design and Nemotron/Xing/media ship no
# backend.
SPOMIN_FAMILIES = {"muse", "north", "laguna"}
# Checkpointed-hybrid caches (GDN/Mamba + attention): junction snapshots only
# exist there.  A KV-only cache is refused at startup, which is also handled.
HYBRID_FAMILIES = {"qwen36", "qwen38", "flash-next", "nemotron"}

FEATURES = (
    ("cache_capsules", "ordinary"),
    ("block_persistence", "ordinary"),
    ("apc_junction_snapshots", "ordinary"),
    ("apc_rolling_checkpoints", "ordinary"),
    ("srpt_prefill_scheduling", "ordinary"),
    ("memory_preemption", "ordinary"),
    ("host_memory_signals", "ordinary"),
    ("fly_verification", "speculative"),
    ("self_mtp_copy_draft", "speculative"),
    ("pld_rotating_replay", "pld"),
    ("spomin_live_compaction", "approx"),
    ("int8_prefill", "approx"),
    ("multi_lora", "lora"),
    ("verify_bitexact", "bitexact"),
)

# Startup refusal text -> feature it names (ServingEngine / bind_for_serving).
REFUSALS = (
    (r"junction checkpoints cannot capture", "apc_junction_snapshots"),
    (r"rolling checkpoints cannot capture", "apc_rolling_checkpoints"),
    (r"cache capsules", "cache_capsules"),
    (r"block persistence", "block_persistence"),
    (r"prefill_scheduling requires", "srpt_prefill_scheduling"),
    (r"memory preemption is incompatible", "memory_preemption"),
    (r"host memory signal", "host_memory_signals"),
    (r"FLy verification requires", "fly_verification"),
    (r"self_mtp_copy_draft requires", "self_mtp_copy_draft"),
    (r"rotating_replay|prompt_lookup policy", "pld_rotating_replay"),
    (r"Spomin", "spomin_live_compaction"),
    (r"int8 (NAX )?prefill|Int8Prefill", "int8_prefill"),
    (r"multi-LoRA|max_loras", "multi_lora"),
    (r"bitexact|set_qmv_bitexact", "verify_bitexact"),
)

# --- fused / compiled kernels ------------------------------------------------
# Every mlx2 text adapter's configure_environment() deletes inherited
# MLX_QWEN*/MLX_LM_*/MLX_GDN_*/MLXUAG_* variables and pins its own profile
# (tests/test_qwen38_27b_port.py asserts it), so these kernels cannot be set
# from the server environment alone.  The arms pass the variables in the
# environment *and* through kernel_shim/sitecustomize.py, which re-applies
# them after the adapter profile; the probe records which ones the adapter
# would otherwise have overwritten.  ``counter`` names the status path whose
# delta proves the kernel ran; None means the runtime exports no counter and
# engagement is only the effective environment.
FLASH_NEXT_UNFUSED = {
    "MLX_QWEN4_FUSED_GDN_DECODE": "0", "MLX_QWEN4_FUSED_GDN_VERIFY": "0",
    "MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK": "0", "MLX_QWEN4_FUSED_GDN_DYNAMIC_ACCEPT": "0",
    "MLX_QWEN4_MOE_FUSED_GATE_UP": "0", "MLX_QWEN4_FUSED_EXPERT_KERNEL": "stock",
    "MLX_QWEN4_MOE_ROUTER_KERNEL": "0", "MLX_QWEN4_EAGER_DISPATCH": "0",
}
QWEN36_UNFUSED = {
    "MLX_QWEN36_FUSED_GDN_DECODE": "0", "MLX_QWEN4_MOE_FUSED_GATE_UP": "0",
    "MLX_QWEN4_MOE_ROUTER_KERNEL": "0", "MLX_QWEN4_FUSED_EXPERT_KERNEL": "stock", "MLX_GDN_CORE": "0",
}
KERNELS = {
    "qwen36": (QWEN36_UNFUSED, (
        ("qwen36_fused_gdn_decode", {"MLX_QWEN36_FUSED_GDN_DECODE": "1"}, None),
        ("moe_fused_gate_up", {"MLX_QWEN4_MOE_FUSED_GATE_UP": "1"}, None),
        ("moe_router_kernel", {"MLX_QWEN4_MOE_ROUTER_KERNEL": "1"}, None),
        ("moe_fused_expert_kernel", {"MLX_QWEN4_MOE_FUSED_GATE_UP": "1", "MLX_QWEN4_FUSED_EXPERT_KERNEL": "auto"}, None),
        ("gdn_core", {"MLX_GDN_CORE": "1"}, None),
    )),
    "qwen38": ({"MLX_GDN_CORE": "0"}, (
        ("gdn_core", {"MLX_GDN_CORE": "1"}, None),
    )),
    "flash-next": (FLASH_NEXT_UNFUSED, (
        ("fused_gdn_decode", {"MLX_QWEN4_FUSED_GDN_DECODE": "1"}, ("execution", "fused_gdn", "fused_calls")),
        ("fused_gdn_verify_replay", {"MLX_QWEN4_FUSED_GDN_VERIFY": "1", "MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK": "1"},
         ("execution", "fused_gdn", "verify_calls")),
        ("fused_gdn_dynamic_accept", {"MLX_QWEN4_FUSED_GDN_VERIFY": "1", "MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK": "1",
                                      "MLX_QWEN4_FUSED_GDN_DYNAMIC_ACCEPT": "1"},
         ("execution", "fused_gdn", "replay_dynamic_rollback_calls")),
        ("moe_fused_gate_up_expert", {"MLX_QWEN4_MOE_FUSED_GATE_UP": "1", "MLX_QWEN4_FUSED_EXPERT_KERNEL": "auto"},
         ("execution", "moe", "dispatches")),
        ("moe_router_kernel", {"MLX_QWEN4_MOE_ROUTER_KERNEL": "1"}, ("execution", "moe", "router_calls")),
        ("eager_dispatch", {"MLX_QWEN4_EAGER_DISPATCH": "1"}, ("execution", "round_levers", "eager_async_evals")),
        ("serving_profile", {"MLX_QWEN4_FUSED_GDN_DECODE": "1", "MLX_QWEN4_FUSED_GDN_VERIFY": "1",
                             "MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK": "1", "MLX_QWEN4_MOE_FUSED_GATE_UP": "1",
                             "MLX_QWEN4_FUSED_EXPERT_KERNEL": "auto"}, ("execution", "fused_gdn", "fused_calls")),
    )),
    "laguna": ({"MLX_LAGUNA_FUSED_DOWN": "stock", "MLX_LAGUNA_FUSED_ROUTER": "stock"}, (
        ("laguna_fused_down", {"MLX_LAGUNA_FUSED_DOWN": "on"}, ("execution", "fused_moe", "down_calls")),
        ("laguna_fused_router", {"MLX_LAGUNA_FUSED_ROUTER": "on"}, ("execution", "fused_moe", "router_calls")),
    )),
    "xing": ({"MLX2_XING_MHC_KERNEL": "0", "MLX2_XING_COMPILE_MHC": "0"}, (
        ("xing_mhc_compiled", {"MLX2_XING_COMPILE_MHC": "1"}, ("execution", "mhc", "compiled_calls")),
        ("xing_mhc_metal_kernel", {"MLX2_XING_COMPILE_MHC": "1", "MLX2_XING_MHC_KERNEL": "1"},
         ("execution", "mhc", "kernel_calls")),
    )),
}
NO_KERNELS = {
    "muse": "no fused/compiled kernel toggle in the Muse Glimmer runtime",
    "north": "no fused/compiled kernel toggle in the North Mini Code runtime",
    "nemotron": "fused chunked-SSD prefill kernels are unconditional (no toggle to A/B)",
    "gemma4": "mlx-vlm runtime; mlx2 exposes no kernel toggle",
    "gemma3n": "mlx-vlm runtime; mlx2 exposes no kernel toggle",
    "minicpmo": "mlx-vlm runtime; mlx2 exposes no kernel toggle",
}
KERNEL_NOTE = ("MLX_LM_COMPILED_DECODE is pinned to 0 by every adapter but nothing in the mlx2 runtime reads it, "
               "so there is no compiled-decode path to test")


def sliding_window(model):
    """The model's sliding-attention window in tokens (0 when it has none)."""
    try:
        config = json.loads((Path(model.path) / "config.json").read_text())
    except (OSError, ValueError):
        return 0
    text = config.get("text_config") or config
    window = text.get("sliding_window") or config.get("sliding_window") or 0
    return int(window) if isinstance(window, int) else 0


def spomin_geometry(model, context):
    """Capacity and prompt size under which live surgery can actually apply.

    Surgery removes the oldest unprotected ``segment_tokens`` chunks down to
    0.65 x capacity, and a sliding-window layer refuses the edit while its
    ring still holds a removed token ("sliding window still holds tokens
    selected for removal").  The removed span ends at about
    ``segment + (L - 0.65 N)`` while the ring holds the last ``W`` tokens, so
    the edit is legal only when ``0.65 N >= W + 2 * segment`` -- independent
    of the prompt length L.  The first round ran North (W = 4096) at
    capacity 4096 with a 5.6K prompt: removal ended at token 4096, the ring
    started at 1463, and every request was declined ``capability_refused``.
    """
    window = sliding_window(model)
    need = (window + 3 * SPOMIN_SEGMENT) / 0.65
    capacity = max(4096, -(-int(need) // 1024) * 1024)
    prompt = min(int(0.8 * capacity), context - 512)
    feasible = prompt > int(0.7 * capacity) + SPOMIN_SEGMENT
    return {"window": window, "capacity_tokens": capacity, "segment_tokens": SPOMIN_SEGMENT,
            "protect_prefix_segments": 1, "prompt_tokens": prompt, "feasible": feasible}


def kernel_env_request(model):
    base, kernels = KERNELS.get(xc.cc.family(model), ({}, ()))
    request = {"base": dict(base)} if kernels else {}
    for name, env, _ in kernels:
        request[name] = {**base, **env}
    return request


# --- applicability -----------------------------------------------------------

def speculative_route(model):
    for route in model.routes:
        if route.name.startswith("mtp") or route.name == "dflash2":
            return route
    return None


def pld_route(model):
    return next((route for route in model.routes if route.name == "prompt-lookup"), None)


def applicability(model, facts, bitexact_api):
    fam = xc.cc.family(model)
    ordinary = next((r for r in model.routes if r.name == "ordinary"), None)
    spec = speculative_route(model)
    pld = pld_route(model)
    out = {}

    def put(name, applies, reason=""):
        out[name] = (bool(applies), reason)

    for name in ("block_persistence", "apc_rolling_checkpoints", "srpt_prefill_scheduling",
                 "memory_preemption", "host_memory_signals"):
        put(name, ordinary is not None, "" if ordinary else "no ordinary route")
    put("cache_capsules", ordinary is not None and fam != "flash-next",
        "" if ordinary is not None and fam != "flash-next" else
        "Flash-Next hybrid cache has no plain KVCache plane eligible for capsules"
        if fam == "flash-next" else "no ordinary route")
    put("apc_junction_snapshots", ordinary is not None and fam in HYBRID_FAMILIES,
        "" if fam in HYBRID_FAMILIES else "KV-only cache: trims to any prefix, junctions never needed (startup refuses)")
    put("fly_verification", spec is not None, "" if spec else "no MTP or external-draft route")
    native = spec is not None and spec.name.startswith("mtp")
    put("self_mtp_copy_draft", native, "" if native else "no native self-MTP route")
    if pld is None:
        put("pld_rotating_replay", False, "no prompt-lookup route in the series matrix")
    else:
        put("pld_rotating_replay", fam in ROTATING_FAMILIES,
            "" if fam in ROTATING_FAMILIES else "no sliding-window RotatingKVCache layers: nothing to replay")
    put("spomin_live_compaction", fam in SPOMIN_FAMILIES and facts.get("spomin_backend"),
        "" if fam in SPOMIN_FAMILIES else "no standard-attention Spomin backend (hybrid state is declined by design)")
    scopes = facts.get("int8_prefill_scopes") or []
    put("int8_prefill", "mlp" in scopes, "" if scopes else "adapter declares no int8 prefill scope")
    if fam == "muse" and (MUSE_LORA / "adapters.safetensors").is_file():
        put("multi_lora", True, "")
    else:
        put("multi_lora", False, "no local LoRA adapter for this base (searched ~/mlx-models, mlx-uag, tests)")
    if not bitexact_api:
        put("verify_bitexact", False, "installed mlx has no mx.metal.set_qmv_bitexact (startup fails closed)")
    else:
        put("verify_bitexact", facts.get("quantized"), "" if facts.get("quantized") else "unquantized artifact")
    return out


# --- server groups -----------------------------------------------------------

def group_spec(model, group, features, stage_dir: Path, *, policy_dir: Path):
    """Route, policy, flags for one group; returns None when nothing applies."""
    if not features:
        return None
    policy, flags = {}, []
    if group in ("ordinary", "approx", "lora", "bitexact"):
        route = xc.route_by_name(model, "ordinary")
        policy.update(xc.route_policy(route))
    elif group == "speculative":
        route = speculative_route(model)
        policy.update(xc.route_policy(route))
    elif group == "pld":
        route = pld_route(model)
        policy.update(xc.route_policy(route))
    else:
        raise ValueError(group)
    cache = stage_dir / f"cache-{group}"
    if "cache_capsules" in features:
        flags.append("--cache-capsules")
    if "block_persistence" in features:
        flags += ["--persistent-block-bytes", str(BLOCK_BYTES)]
    if "apc_junction_snapshots" in features:
        policy["apc_junction_checkpoints"] = True
    if "apc_rolling_checkpoints" in features:
        policy["apc_rolling_checkpoints"] = {"interval_tokens": ROLLING_INTERVAL}
    if "srpt_prefill_scheduling" in features:
        policy["prefill_scheduling"] = {"order": "srpt", "max_bypass": 3, "one_slice_contention": True}
    if "memory_preemption" in features:
        policy["memory_preemption"] = {"enabled": True, "stall_seconds": 60, "on_pressure": True}
    if "host_memory_signals" in features:
        policy["host_memory_signals"] = {"enabled": True, "fall_after_seconds": 5.0}
    if "fly_verification" in features:
        policy["fly_verification"] = {"enabled": True, "entropy_threshold": 2.0, "window": 2, "min_prob": 0.01}
    if "self_mtp_copy_draft" in features:
        policy["self_mtp_copy_draft"] = {"enabled": True}
    if "pld_rotating_replay" in features:
        policy["prompt_lookup"] = {**policy.get("prompt_lookup", {}), "rotating_replay": True}
    spomin = None
    if "spomin_live_compaction" in features:
        spomin = spomin_geometry(model, command_value(xc.cc.server_args(model, route, "smoke"), "--max-context", int))
        flags += ["--spomin-live-surgery", json.dumps({
            "enabled": True, "capacity_tokens": spomin["capacity_tokens"],
            "segment_tokens": spomin["segment_tokens"],
            "protect_prefix_segments": spomin["protect_prefix_segments"]})]
    if "int8_prefill" in features:
        flags += ["--int8-prefill", "mlp"]
    if "multi_lora" in features:
        flags += ["--lora-dir", str(stage_dir / "lora"), "--max-loras", "2", "--max-lora-rank", "16"]
    if "verify_bitexact" in features:
        flags.append("--verify-bitexact")
    policy_file = None
    if policy:
        policy_dir.mkdir(parents=True, exist_ok=True)
        policy_file = policy_dir / f"{group}.json"
        policy_file.write_text(json.dumps(policy, indent=1, sort_keys=True) + "\n")
    command = xc.server_command(model, route, "smoke", cache, policy_file=policy_file, extra=flags)
    return {"group": group, "route": route.name, "policy": policy, "flags": flags, "command": command,
            "features": list(features), "spomin": spomin}


def kernel_arms(model, stage_dir: Path):
    fam = xc.cc.family(model)
    base, kernels = KERNELS.get(fam, ({}, ()))
    if not kernels:
        return []
    route = xc.route_by_name(model, model.default_route)
    arms = [("base", dict(base), None)] + [(name, {**base, **env}, counter) for name, env, counter in kernels]
    out = []
    for name, env, counter in arms:
        command = xc.server_command(model, route, "smoke", stage_dir / f"cache-fused-{name}",
                                    keep_route_policy=True)
        out.append({"arm": name, "route": route.name, "env": env, "counter": counter, "command": command})
    return out


# --- checks ------------------------------------------------------------------

def nonce():
    return uuid.uuid4().hex[:10]


def base_checks(http):
    rows = {}
    reply = http.chat(xc.PROMPT_SUM, max_tokens=32)
    rows["short_chat"] = {"status": reply["status"], "text": xc.text_of(reply)[:200],
                          "ok": reply["status"] == 200 and bool(xc.oracle(xc.PROMPT_SUM, xc.text_of(reply)))}
    prefix = xc.filler(40, f"prefix-{nonce()}") + "\nReply with the word READY."
    first, second = http.chat(prefix, max_tokens=8), http.chat(prefix, max_tokens=8)
    cached = xc.cached_tokens(second)
    rows["repeated_prefix"] = {"status": [first["status"], second["status"]], "cached_tokens": cached,
                               "ok": first["status"] == second["status"] == 200 and cached > 0}
    stream = http.stream(xc.PROMPT_CAPITAL, max_tokens=32)
    rows["streaming"] = {"status": stream.get("status"), "done": stream.get("done"), "text": (stream.get("text") or "")[:200],
                         "error": stream.get("error"),
                         "ok": stream.get("status") == 200 and bool(stream.get("done"))
                         and bool(xc.oracle(xc.PROMPT_CAPITAL, stream.get("text")))}
    return {"ok": all(row["ok"] for row in rows.values()), **rows}


ADMISSION_REFUSAL = "host memory admission"


def refused_by_admission(replies):
    """How many replies memory admission turned away (HTTP 429 after its deadline)."""
    def text(reply):
        return json.dumps(reply.get("body")) + str(reply.get("error") or "")
    return sum(ADMISSION_REFUSAL in text(reply) for reply in replies)


def memory_evidence(before, after, replies=()):
    """Why a probe may not have run as designed: admission and APCv2 pressure.

    On a 36 GB host a 27B model leaves one lane's worth of headroom, so a
    probe can be refused or serialized by memory admission, and optional
    checkpoints can be budgeted away or pressure-spilled.  Without these
    counters "not engaged" cannot be told apart from "not exercisable here".
    """
    count = lambda key: xc.delta(before, after, "counts", key)  # noqa: E731
    return {
        "admission_deferred": count("memory_admission_deferred"),
        "admission_timeouts": count("memory_admission_timeouts"),
        "admission_deferred_behind_grants": count("memory_admission_deferred_behind_grants"),
        "pressure_evictions": count("memory_pressure_evictions"),
        "apc_pressure_spills": xc.delta(before, after, "apcv2", "idle_disk", "pressure_spills"),
        "apc_resident_bytes": xc.dig(after, "apcv2", "idle_disk", "resident_bytes", default=None),
        "headroom_bytes": [xc.dig(before, "headroom_bytes", default=None), xc.dig(after, "headroom_bytes", default=None)],
        "metal_active_bytes": xc.dig(after, "metal_active_bytes", default=None),
        "footprint_bytes": xc.dig(after, "process_physical_footprint_bytes", default=None),
        "refused_by_admission": refused_by_admission(replies),
    }


def admission_note(memory):
    refused = memory.get("refused_by_admission") or 0
    if refused:
        return (f"host memory admission refused {refused} probe request(s): the host could not seat "
                "the probe as designed (a host limit, not a feature verdict)")
    return ""


CAPSULE_SERVING_COUNTS = (
    "cache_capsule_prepared", "cache_capsule_prepare_fallbacks", "cache_capsule_incompatible_fallbacks",
    "cache_capsule_deadline_fallbacks", "cache_capsule_width_fallbacks", "apcv2_fanout_boundaries",
    "apcv2_fanout_boundary_misses", "apcv2_fanout_store_failures",
    "cache_capsule_boundary_mismatch_fallbacks",
)
# Words of each greedy fanout sample compared across runs.  A capsule built
# from the wrong prompt length still produces fluent text (North passed round
# one that way); only a comparison catches it.
CAPSULE_PREFIX_WORDS = 5


def choice_summary(choice):
    message = choice.get("message") or {}
    content = message.get("content") or ""
    return {"finish_reason": choice.get("finish_reason"), "content_chars": len(content),
            "reasoning_chars": len(message.get("reasoning_content") or message.get("reasoning") or ""),
            "text": content[:80]}


def check_cache_capsules(http):
    prompt = xc.filler(40, f"capsule-{nonce()}") + "\nSummarize the archive in one sentence."
    warm = http.chat(prompt, max_tokens=24)
    before = http.settled()
    fan = http.chat(prompt, n=3, temperature=0, max_tokens=24)
    after = http.settled()
    counts = {key: xc.delta(before, after, "cache_capsules", key)
              for key in ("requests", "primary_successes", "fallbacks", "timeouts", "errors", "stale",
                          "capacity_rejections", "reclaim_rebases")}
    serving = {key: xc.delta(before, after, "counts", key) for key in CAPSULE_SERVING_COUNTS}
    choices = (fan.get("body") or {}).get("choices") or []
    engaged = (counts["requests"] or 0) > 0 and ((counts["primary_successes"] or 0) + (counts["fallbacks"] or 0)) > 0
    reference = capsule_head(xc.text_of(warm))
    heads = [capsule_head((c.get("message") or {}).get("content")) for c in choices]
    # The gate is the same-width control run after the group (``run_capsule_control``):
    # a width-1 reference cannot tell capsule state from batch-width numerics.
    ok = (warm["status"] == fan["status"] == 200 and len(choices) == 3 and all(heads)
          and not counts["errors"] and not counts["stale"])
    memory = memory_evidence(before, after, [warm, fan])
    return {"engaged": engaged, "ok": ok, "notes": admission_note(memory), "evidence": {
        "capsule_deltas": counts, "serving_deltas": serving, "statuses": [warm["status"], fan["status"]],
        "sample_capsule_receipts": [sample.get("cache_capsule") for sample in
                                    ((fan.get("body") or {}).get("mlx2") or {}).get("samples", [])],
        "choices": [choice_summary(c) for c in choices], "single_sample_text": xc.text_of(warm)[:80],
        "prefix_matches_single_sample": [h == reference for h in heads],
        "control_prompt": prompt, "fanout_heads": heads,
        "apc_capsule_capacity": xc.dig(after, "apcv2", "cache_capsules", default=None),
        "fanout_groups_delta": xc.delta(before, after, "counts", "apcv2_fanout_groups"), "memory": memory}}


def capsule_head(text):
    return (text or "").split()[:CAPSULE_PREFIX_WORDS]


def run_capsule_control(model, command, stage_dir, results):
    """Same-width control for the capsule check.

    The capsule run's siblings decode at width 2-3 while a single-sample
    reference decodes at width 1, and North's first tokens are near-ties, so
    a width-1 mismatch cannot tell bad capsule state from batch-width
    numerics.  Restart the group's server with the identical command minus
    ``--cache-capsules`` (there is no per-request opt-out) and a fresh cache
    directory, replay the identical warm request and greedy n=3 fanout, and
    require every sample's first words to equal the capsule run's.  Only a
    divergence the control does not share implicates the capsule.
    """
    row = results["cache_capsules"]
    evidence = row.get("evidence") or {}
    prompt = evidence.get("control_prompt")
    if not prompt or not row.get("ok"):
        return
    if not command or "--cache-capsules" not in command:
        return
    command = [arg for arg in command if arg != "--cache-capsules"]
    command[command.index("--cache-dir") + 1] = str(stage_dir / "cache-capsule-control")
    server = xc.Server("capsule-control", command, xc.stage_env(model), stage_dir / "server-capsule-control.log")
    print("CAPSULE CONTROL: same group without --cache-capsules", flush=True)
    control = {"command_differs_by": "--cache-capsules removed, fresh --cache-dir"}
    try:
        startup = server.start()
        if not startup["ready"]:
            control["startup"] = {k: startup.get(k) for k in ("ready", "reason", "log_tail")}
            heads = None
        else:
            http = xc.HTTP(model_id=Path(model.path).name, timeout=900)
            warm = http.chat(prompt, max_tokens=24)
            fan = http.chat(prompt, n=3, temperature=0, max_tokens=24)
            choices = (fan.get("body") or {}).get("choices") or []
            heads = [capsule_head((c.get("message") or {}).get("content")) for c in choices]
            control.update(statuses=[warm["status"], fan["status"]], single_sample_text=xc.text_of(warm)[:80],
                           choices=[choice_summary(c) for c in choices], fanout_heads=heads,
                           cache_capsules_status=xc.dig(http.settled(), "cache_capsules", default=None))
    finally:
        server.stop()
    capsule_heads = evidence.get("fanout_heads") or []
    matches = ([a == b for a, b in zip(capsule_heads, heads, strict=False)]
               if heads and len(heads) == len(capsule_heads) else [])
    control["matches_capsule_run"] = matches
    evidence["no_capsule_control"] = control
    ok = bool(matches) and all(matches)
    notes = [row.get("notes") or ""]
    if not heads:
        notes.append("no-capsule control did not run; capsule output unverified")
    elif not ok:
        notes.append("capsule fanout diverged from the same-width no-capsule fanout")
    elif not all(evidence.get("prefix_matches_single_sample") or []):
        notes.append("capsule and no-capsule fanouts agree but differ from the width-1 answer "
                     "(batch-width numerics, not capsule state)")
    row.update(ok=ok, notes="; ".join(n for n in notes if n))


def check_block_persistence(http):
    sid = f"series-blocks-{nonce()}"
    prompt = xc.filler(60, sid) + "\nReply with the word BLOCKS."
    before = http.settled()
    seeded = http.chat(prompt, max_tokens=8, session_id=sid)
    parked = http.post(f"/v1/apc/sessions/{sid}/park", {"ttl_seconds": 300})
    state, deadline = {}, time.monotonic() + 60
    while time.monotonic() < deadline:
        state = http.get(f"/v1/apc/sessions/{sid}")
        if (state.get("body") or {}).get("state") == "disk":
            break
        time.sleep(0.2)
    resumed = http.post(f"/v1/apc/sessions/{sid}/resume", {})
    seeded_tokens = xc.dig(seeded, "body", "usage", "prompt_tokens")
    required_cached = ((seeded_tokens * 9) // 10
                       if isinstance(seeded_tokens, int) and seeded_tokens >= 100 else None)
    restore_state = {}
    deadline = time.monotonic() + 60
    if (state.get("body") or {}).get("state") == "disk" and resumed["status"] == 202:
        while time.monotonic() < deadline:
            restore_state = http.get(f"/v1/apc/sessions/{sid}")
            if restore_state["status"] != 200:
                break
            restored_body = restore_state.get("body") or {}
            if (restored_body.get("state") == "resident"
                    and required_cached is not None
                    and restored_body.get("covered_tokens", 0) >= required_cached):
                break
            time.sleep(0.2)
    restore_ready = ((restore_state.get("body") or {}).get("state") == "resident"
                     and required_cached is not None
                     and (restore_state.get("body") or {}).get("covered_tokens", 0) >= required_cached)
    restored = http.chat(prompt, max_tokens=8, session_id=sid) if restore_ready else {"status": None}
    after = http.settled()
    idle = lambda key: xc.delta(before, after, "apcv2", "idle_disk", key)  # noqa: E731
    block_bytes = xc.dig(after, "apcv2", "idle_disk", "block_bytes")
    evidence = {"block_bytes": block_bytes, "parks": idle("parks"), "restores": idle("restores"),
                "resumes": idle("resumes"), "restore_failures": idle("restore_failures"),
                "restore_digest_failures": idle("restore_digest_failures"), "bytes_written": idle("bytes_written"),
                "state": (state.get("body") or {}).get("state"),
                "restore_state": (restore_state.get("body") or {}).get("state"),
                "restore_covered_tokens": (restore_state.get("body") or {}).get("covered_tokens"),
                "seeded_prompt_tokens": seeded_tokens, "required_cached_tokens": required_cached,
                "cached_tokens": xc.cached_tokens(restored)}
    engaged = (block_bytes == BLOCK_BYTES and (evidence["parks"] or 0) > 0
               and (evidence["restores"] or 0) > 0)
    ok = ((state.get("body") or {}).get("state") == "disk" and restore_ready
          and seeded["status"] == parked["status"] == restored["status"] == 200
          and resumed["status"] == 202
          and not evidence["restore_failures"] and not evidence["restore_digest_failures"]
          and evidence["cached_tokens"] >= required_cached
          and xc.text_of(restored) == xc.text_of(seeded))
    http.request("DELETE", f"/v1/apc/sessions/{sid}")
    return {"engaged": engaged, "ok": ok, "evidence": evidence}


def check_junction(http):
    system = xc.filler(100, f"junction-{nonce()}") + "\nYou answer questions about the archive."
    replies = []
    before = http.settled()
    for index, question in enumerate(("How many entries are there?", "Name the first entry.",
                                      "Which shipment was logged last?", "Reply with OK.")):
        replies.append(http.chat("", messages=[{"role": "system", "content": system},
                                               {"role": "user", "content": f"Turn {index}: {question}"}], max_tokens=12))
    after = http.settled()
    published = xc.delta(before, after, "counts", "apc_junction_checkpoints_published") or 0
    hits = xc.delta(before, after, "apcv2", "lifetime", "junction_hits") or 0
    memory = memory_evidence(before, after, replies)
    return {"engaged": min(published, hits) > 0, "ok": all(r["status"] == 200 for r in replies),
            "notes": admission_note(memory),
            "evidence": {"published": published, "junction_hits": hits,
                         "planned": xc.delta(before, after, "counts", "apc_junction_checkpoints_planned"),
                         "degraded": xc.delta(before, after, "counts", "apc_junction_checkpoints_degraded"),
                         "publish_failed": xc.delta(before, after, "counts",
                                                    "apc_junction_checkpoints_skipped_publish_failed"),
                         "interior_hits": xc.delta(before, after, "apcv2", "lifetime", "interior_hits"),
                         "cached_tokens": [xc.cached_tokens(r) for r in replies],
                         "statuses": [r["status"] for r in replies], "memory": memory}}


ROLLING_UNITS = 140        # ~2.6K tokens: five 512-token strides, one lane on a 36 GB host


def check_rolling(http):
    """Abandon a prefill after it has passed rolling boundaries; the retry
    must resume from the published rolling checkpoint.

    The first round used ~9K- and ~13K-token prompts.  On 27B models at 16K
    context on the 36 GB host memory admission refused them outright (the
    probe itself answered 429 and its TTFT silently fell back to 2 s), and
    where one was admitted its checkpoint budget (headroom minus the lane's
    own requirement) left no room for a rolling slot.  A successful prefill
    retires its rolling point (the committed boundary supersedes it) and a
    same-prompt peer needs a second lane, so the cancel-and-retry path is the
    one that shows the checkpoint being used.
    """
    before = http.settled()
    probe = http.chat(xc.filler(ROLLING_UNITS, f"rolling-probe-{nonce()}") + "\nReply OK.", max_tokens=4)
    ttft = xc.receipt_of(probe).get("ttft_seconds")
    evidence = {"probe_status": probe["status"], "ttft_probe_s": ttft}
    if probe["status"] != 200 or not ttft:
        after = http.settled()
        memory = memory_evidence(before, after, [probe])
        evidence["memory"] = memory
        return {"engaged": False, "ok": False, "evidence": evidence,
                "notes": admission_note(memory) or "rolling probe request failed; no prefill timing to abandon against"}
    abandoned = xc.filler(ROLLING_UNITS, f"rolling-cancel-{nonce()}") + "\nReply OK."
    # Past the first strides, well before the prompt boundary.
    after_s = max(0.3, 0.35 * float(ttft))
    http.abandon(abandoned, after_s)
    time.sleep(2)
    retry = http.chat(abandoned, max_tokens=4)
    after = http.settled()
    published = xc.delta(before, after, "counts", "apc_rolling_checkpoints_published") or 0
    hits = xc.delta(before, after, "apcv2", "lifetime", "rolling_hits") or 0
    cancel = xc.delta(before, after, "counts", "apc_rolling_checkpoints_cancel_published") or 0
    memory = memory_evidence(before, after, [probe, retry])
    engaged = max(min(published, hits), cancel) > 0
    ok = retry["status"] == 200
    evidence.update({
        "abandon_after_s": after_s, "published": published, "rolling_hits": hits, "cancel_published": cancel,
        "planned": xc.delta(before, after, "counts", "apc_rolling_checkpoints_planned"),
        "degraded": xc.delta(before, after, "counts", "apc_rolling_checkpoints_degraded"),
        "retired": xc.delta(before, after, "counts", "apc_rolling_checkpoints_retired"),
        "retry_status": retry["status"], "retry_cached_tokens": xc.cached_tokens(retry), "memory": memory})
    return {"engaged": engaged, "ok": ok, "notes": admission_note(memory), "evidence": evidence}


SRPT_LONG_UNITS = 140      # ~2.6K tokens: several prefill slices, one lane on a 36 GB host
SRPT_LONGS, SRPT_SHORTS = 2, 4


def check_srpt(http, inflight):
    """Long prefills first, then shorts queued behind them.

    The request count stays below the server's ``--max-inflight``: the first
    round sent 3 + 6 = 9 streams to a server capped at 8, and the ninth was
    refused "maximum inflight requests reached" (North).  Longs are sized to
    span several prefill slices without needing more than one lane of
    headroom; 27B models on the 36 GB host refused ~5K-token longs at memory
    admission, so nothing ever reached the prefill queue SRPT orders.
    """
    total = SRPT_LONGS + SRPT_SHORTS
    if total >= inflight:
        raise ValueError(f"SRPT probe sends {total} streams; server max-inflight is {inflight}")
    before = http.settled()
    longs = [xc.filler(SRPT_LONG_UNITS, f"srpt-long-{i}-{nonce()}") + "\nReply with the word LONG."
             for i in range(SRPT_LONGS)]
    shorts = [f"Short question {i}. {xc.PROMPT_SUM}" for i in range(SRPT_SHORTS)]
    calls = [(lambda p=p, d=0.05 * i: (time.sleep(d), http.stream(p, max_tokens=8))[1]) for i, p in enumerate(longs)]
    calls += [(lambda p=p, d=0.3 + 0.05 * i: (time.sleep(d), http.stream(p, max_tokens=16))[1])
              for i, p in enumerate(shorts)]
    results = xc.parallel(calls)
    after = http.settled()
    bypasses = xc.delta(before, after, "scheduler", "prefill_scheduling_bypasses") or 0
    forced = xc.delta(before, after, "scheduler", "prefill_scheduling_bypass_forced") or 0
    clamps = xc.delta(before, after, "scheduler", "prefill_scheduling_one_slice_clamps") or 0
    shorts_ok = [bool(xc.oracle(xc.PROMPT_SUM, r.get("text"))) for r in results[SRPT_LONGS:]]
    ok = all(r.get("status") == 200 and r.get("done") for r in results) and all(shorts_ok)
    memory = memory_evidence(before, after, results)
    # A one-slice clamp is the policy's contention rule acting on a live long
    # prefill; a bypass is SRPT reordering the queue.  Either is the policy
    # running.
    return {"engaged": bypasses + forced + clamps > 0, "ok": ok, "notes": admission_note(memory), "evidence": {
        "bypasses": bypasses, "bypass_forced": forced, "one_slice_clamps": clamps,
        "streams": total, "max_inflight": inflight,
        "shorts_correct": shorts_ok, "errors": [r.get("error") for r in results if r.get("error")],
        "memory": memory}}


def check_preemption(http):
    before = http.settled()
    prompt = xc.filler(120, f"preempt-{nonce()}") + "\nIn one sentence, restate the archive's rule."
    peer_prompt = xc.filler(120, f"preempt-peer-{nonce()}") + "\nIn one sentence, say what the archive is."
    fault = {"kind": "memory_preempt", "after_tokens": 0}
    warm = http.chat(prompt, max_tokens=32)
    reference = http.chat(prompt, max_tokens=32)
    solo = http.chat(prompt, max_tokens=32, mlx_fault=fault)
    peer, victim = xc.parallel([lambda: http.chat(peer_prompt, max_tokens=32),
                                lambda: http.chat(prompt + " Be brief.", max_tokens=32, mlx_fault=fault)])
    after = http.settled()
    record = lambda r: xc.receipt_of(r).get("preemption") or {}  # noqa: E731
    fired = lambda r: bool(record(r).get("replays")) and record(r).get("fault_unfired") is None  # noqa: E731
    preemptions = xc.delta(before, after, "counts", "memory_preemptions") or 0
    replays = xc.delta(before, after, "counts", "preempted_replays") or 0
    named = {"warm": warm, "reference": reference, "solo": solo, "peer": peer, "victim": victim}
    failed = {name: {"status": r["status"], "error": json.dumps(r.get("body"))[:300] if r["status"] != 200 else None,
                     "empty_text": not xc.text_of(r)}
              for name, r in named.items() if r["status"] != 200 or not xc.text_of(r)}
    ok = (not failed and (not fired(solo) or xc.text_of(solo) == xc.text_of(reference)) and not fired(peer))
    memory = memory_evidence(before, after, list(named.values()))
    return {"engaged": min(preemptions, replays) > 0 and (fired(victim) or fired(solo)), "ok": ok,
            "notes": admission_note(memory), "evidence": {
        "preemptions": preemptions, "replays": replays, "solo_fired": fired(solo), "victim_fired": fired(victim),
        "solo_equals_reference": xc.text_of(solo) == xc.text_of(reference), "failed_requests": failed,
        "ttft_s": {name: xc.receipt_of(r).get("ttft_seconds") for name, r in named.items()},
        "solo_fault_unfired": record(solo).get("fault_unfired"), "victim_fault_unfired": record(victim).get("fault_unfired"),
        "memory": memory}}


def check_host_signals(http):
    status = http.settled()
    available = status.get("host_memory_available_bytes")
    level = status.get("host_memory_pressure_level")
    engaged = type(available) is int and available > 0
    return {"engaged": engaged, "ok": engaged and level is not None,
            "evidence": {"host_memory_available_bytes": available, "host_memory_pressure_level": level}}


def check_fly(http):
    before = http.settled()
    # Vendor penalty defaults add logits processors, which intentionally
    # disable FLy. Ask explicitly for neutral penalties to test FLy itself.
    neutral = {"repetition_penalty": 1, "presence_penalty": 0,
               "frequency_penalty": 0}
    greedy = http.chat(xc.PROMPT_OPEN, max_tokens=128, **neutral)
    sampled = http.chat(xc.PROMPT_OPEN, max_tokens=128, temperature=0.7,
                        seed=5, **neutral)
    after = http.settled()
    mechanisms = []
    for reply in (greedy, sampled):
        receipt = xc.receipt_of(reply)
        mechanisms.append(receipt.get("mtp") or receipt.get("speculation") or {})
    verification = [m.get("verification") for m in mechanisms]
    relaxed = (xc.delta(before, after, "scheduler", "fly_relaxed_accepts") or 0) + sum(
        int(m.get("relaxed_accepts") or 0) for m in mechanisms)
    engaged = (verification == ["fly", "exact"]
               and not mechanisms[0].get("fly_disabled")
               and bool(mechanisms[1].get("fly_disabled")))
    ok = greedy["status"] == sampled["status"] == 200 and bool(xc.text_of(greedy))
    return {"engaged": engaged, "ok": ok, "evidence": {"verification": verification, "relaxed_accepts": relaxed,
            "fly_disabled": [m.get("fly_disabled") for m in mechanisms]}}


PASSAGE = ("The lighthouse keeper climbed the spiral stairs at dusk, trimmed the wick, polished the great lens, "
           "and wrote in the log that the sea was calm, the wind was from the west, and two fishing boats "
           "had passed the point before nightfall. ")


def check_copy_draft(http):
    before = http.settled()
    reply = http.chat("Repeat the following text exactly, word for word, three times:\n\n" + PASSAGE, max_tokens=220)
    after = http.settled()
    rounds = xc.delta(before, after, "scheduler", "self_mtp_copy_rounds") or 0
    copy = (xc.receipt_of(reply).get("mtp") or {}).get("copy_draft")
    ok = reply["status"] == 200 and PASSAGE[:60].lower() in xc.text_of(reply).lower()
    return {"engaged": rounds > 0, "ok": ok, "evidence": {"self_mtp_copy_rounds": rounds, "receipt_copy_draft": copy}}


def check_rotating_replay(http):
    before = http.settled()
    prompt = xc.filler(400, f"pld-{nonce()}") + "\n\n" + PASSAGE * 3 + "\nRepeat the last paragraph above exactly."
    reply = http.chat(prompt, max_tokens=160)
    after = http.settled()
    rounds = xc.delta(before, after, "scheduler", "pld_rotating_replay_rounds") or 0
    return {"engaged": rounds > 0, "ok": reply["status"] == 200 and bool(xc.text_of(reply)), "evidence": {
        "rotating_replay_rounds": rounds,
        "replayed_tokens": xc.delta(before, after, "scheduler", "pld_rotating_replay_replayed_tokens"),
        "refusals": xc.delta(before, after, "scheduler", "pld_rotating_replay_refusals"),
        "pld_proposed": xc.delta(before, after, "scheduler", "pld_proposed")}}


def check_spomin(http, geometry):
    if not geometry["feasible"]:
        return {"applicable": False, "engaged": False, "ok": False, "evidence": {"geometry": geometry},
                "notes": "server context cannot hold a prompt above the compaction trigger for this sliding window"}
    before = http.settled()
    needle = f"NEEDLE-{nonce()[:6].upper()}"
    tail = f"\nThe unique fact is: the launch code is {needle}.\nWhat is the launch code? Reply with {needle} first."
    per_unit = max(1.0, http.count(xc.filler(100, "calibrate")) / 100.0)
    units = max(1, int(geometry["prompt_tokens"] / per_unit))
    prompt = xc.filler(units, f"spomin-{nonce()}") + tail
    reply = http.chat(prompt, max_tokens=24)
    after = http.settled()
    receipt = xc.receipt_of(reply).get("spomin_live_surgery") or {}
    applied = xc.delta(before, after, "spomin_live_surgery", "counts", "applied") or 0
    return {"engaged": applied > 0 or receipt.get("status") == "applied",
            "ok": reply["status"] == 200 and needle in xc.text_of(reply),
            "evidence": {"applied_delta": applied, "receipt_status": receipt.get("status"),
                         "receipt_reason": receipt.get("reason"), "receipt_detail": receipt.get("detail"),
                         "source_tokens": receipt.get("source_tokens"), "target_tokens": receipt.get("target_tokens"),
                         "geometry": geometry, "prompt_tokens": xc.receipt_of(reply).get("prompt_tokens")}}


def check_int8(http):
    before = http.settled()
    prompt = xc.filler(90, f"int8-{nonce()}") + "\n" + xc.PROMPT_SUM
    reply = http.chat(prompt, max_tokens=24)
    after = http.settled()
    engaged_calls = xc.delta(before, after, "int8_prefill", "counts", "engaged_calls") or 0
    return {"engaged": engaged_calls > 0, "ok": reply["status"] == 200 and bool(xc.oracle(xc.PROMPT_SUM, xc.text_of(reply))),
            "evidence": {"engaged_calls": engaged_calls, "active": xc.dig(after, "int8_prefill", "active", default=None),
                         "text": xc.text_of(reply)[:100]}}


def check_multi_lora(http, stage_dir):
    load = http.post("/v1/load_lora_adapter", {"lora_name": "muse-cyber", "lora_path": "muse-cyber"})
    if load["status"] != 200:
        message = json.dumps(load.get("body"))[:600]
        if "is not a Linear/QuantizedLinear module" in message or "shape" in message:
            return {"applicable": False, "engaged": False, "ok": False,
                    "notes": "local Muse LoRA does not fit this artifact's module tree: " + message}
        return {"engaged": False, "ok": False, "evidence": {"load": load}}
    before = http.settled()
    replies = xc.parallel([lambda: http.chat(xc.PROMPT_SUM, max_tokens=24),
                           lambda: http.chat(xc.PROMPT_SUM, max_tokens=24, model="muse-cyber"),
                           lambda: http.chat(xc.PROMPT_CAPITAL, max_tokens=24),
                           lambda: http.chat(xc.PROMPT_CAPITAL, max_tokens=24, model="muse-cyber")])
    after = http.settled()
    lora_receipts = [(xc.receipt_of(r).get("lora") or {}).get("name") for r in replies]
    counts = {key: xc.delta(before, after, "multi_lora", "counts", key)
              for key in ("forwards", "mixed_forwards", "single_adapter_forwards", "delta_applications")}
    unload = http.post("/v1/unload_lora_adapter", {"lora_name": "muse-cyber"})
    base_ok = [bool(xc.oracle(p, xc.text_of(r))) for p, r in zip((xc.PROMPT_SUM, xc.PROMPT_CAPITAL), replies[0::2], strict=True)]
    return {"engaged": (counts["delta_applications"] or 0) > 0 and "muse-cyber" in lora_receipts,
            "ok": all(r["status"] == 200 and xc.text_of(r) for r in replies) and all(base_ok) and unload["status"] == 200,
            "evidence": {"counts": counts, "lora_receipts": lora_receipts, "base_answers_correct": base_ok}}


def check_bitexact(http):
    before = http.settled()
    replies = [http.chat(xc.PROMPT_OPEN, max_tokens=48, verify_bitexact=True) for _ in range(2)]
    after = http.settled()
    dispatches = xc.delta(before, after, "verify_bitexact", "dispatches") or 0
    claims = [xc.receipt_of(r).get("verify_bitexact") for r in replies]
    return {"engaged": dispatches > 0 and all(claims), "ok": all(r["status"] == 200 for r in replies)
            and xc.text_of(replies[0]) == xc.text_of(replies[1]),
            "evidence": {"dispatches": dispatches, "receipt_claims": claims}}


CHECKS = {
    "cache_capsules": check_cache_capsules, "block_persistence": check_block_persistence,
    "apc_junction_snapshots": check_junction, "apc_rolling_checkpoints": check_rolling,
    "srpt_prefill_scheduling": check_srpt, "memory_preemption": check_preemption,
    "host_memory_signals": check_host_signals, "fly_verification": check_fly,
    "self_mtp_copy_draft": check_copy_draft, "pld_rotating_replay": check_rotating_replay,
    "spomin_live_compaction": check_spomin, "int8_prefill": check_int8, "verify_bitexact": check_bitexact,
}
CHECK_TEXT = {
    "cache_capsules": ("greedy n=3 fanout on a warm prompt; cache_capsules.requests and primary_successes|fallbacks move, "
                       "no errors/stale; then the same server command without --cache-capsules replays it and every "
                       "sample's first words must match (width-1 answer recorded, not gated)"),
    "block_persistence": "session park to disk, resume, re-ask; idle_disk.block_bytes set, parks+restores move, cached hit, same text",
    "apc_junction_snapshots": "4 chats sharing a long system prompt, diverging user turns; junction published and hit",
    "apc_rolling_checkpoints": ("~2.6K-token prefill abandoned at 0.6x its TTFT, then retried; rolling published+hit "
                                "or cancel-published (planned/degraded and admission counters recorded)"),
    "srpt_prefill_scheduling": (f"{SRPT_LONGS} long (~2.6K tok) + {SRPT_SHORTS} short concurrent streams, below max-inflight; "
                                "bypasses|bypass_forced|one_slice_clamps move, short answers correct"),
    "memory_preemption": "warm, reference, solo prefill fault, concurrent peer+faulted victim; preemptions and replays move",
    "host_memory_signals": "/v1/status host_memory_available_bytes > 0 and a pressure level",
    "fly_verification": "greedy and sampled 128-token chats; receipt verification == fly",
    "self_mtp_copy_draft": "verbatim-repeat request; scheduler self_mtp_copy_rounds moves",
    "pld_rotating_replay": "long prompt that wraps the sliding window, repeat request; pld_rotating_replay_rounds moves",
    "spomin_live_compaction": ("prompt at 0.8x a capacity sized so removed segments have left the sliding window "
                               "(0.65 N >= W + 2 segments); surgery applied, needle still answered"),
    "int8_prefill": "~1.5K-token prefill (rows >= 512); int8_prefill.counts.engaged_calls moves, answer correct",
    "multi_lora": "load local Muse LoRA, 2 base + 2 adapter concurrent chats; delta_applications moves, receipts name it",
    "verify_bitexact": "two verify_bitexact requests; dispatches move, receipts claim it, outputs equal",
}


def command_value(command, flag, cast=str):
    return cast(command[command.index(flag) + 1])


def matched_refusal(log_tail, features):
    for pattern, feature in REFUSALS:
        if feature in features and re.search(pattern, log_tail, re.IGNORECASE):
            line = next((ln for ln in log_tail.splitlines()[::-1] if re.search(pattern, ln, re.IGNORECASE)), "")
            return feature, line.strip()[:400]
    return None, None


def run_group(model, group, features, stage_dir, results, policy_dir):
    remaining = list(features)
    attempt = 0
    while remaining:
        attempt += 1
        spec = group_spec(model, group, remaining, stage_dir, policy_dir=policy_dir)
        if group == "lora":
            stage_lora(stage_dir)
        log = stage_dir / f"server-{group}-{attempt}.log"
        server = xc.Server(group, spec["command"], xc.stage_env(model), log)
        print(f"GROUP {group} attempt {attempt}: {remaining}", flush=True)
        try:
            startup = server.start()
            if not startup["ready"]:
                feature, line = matched_refusal(startup.get("log_tail", ""), remaining)
                if feature is not None and attempt <= len(features):
                    results[feature].update(applicable=False, engaged=False, ok=False,
                                            notes=f"startup refused: {line}")
                    remaining.remove(feature)
                    continue
                for name in remaining:
                    results[name].update(engaged=False, ok=False,
                                         notes=f"server failed to start: {startup.get('reason')}")
                results["_groups"][group] = {"startup": {k: startup.get(k) for k in ("ready", "reason", "log_tail")}}
                return
            http = xc.HTTP(model_id=Path(model.path).name, timeout=900)
            record = {"route": spec["route"], "policy": spec["policy"], "flags": spec["flags"],
                      "command": spec["command"],
                      "load_seconds": startup.get("load_seconds")}
            record["base_checks"] = base = base_checks(http)
            for name in remaining:
                started = time.monotonic()
                try:
                    if name == "multi_lora":
                        outcome = check_multi_lora(http, stage_dir)
                    elif name == "spomin_live_compaction":
                        outcome = check_spomin(http, spec["spomin"])
                    elif name == "srpt_prefill_scheduling":
                        outcome = check_srpt(http, command_value(spec["command"], "--max-inflight", int))
                    else:
                        outcome = CHECKS[name](http)
                except Exception as error:  # noqa: BLE001 - one feature fails, the group continues
                    outcome = {"engaged": False, "ok": False, "notes": f"check raised {type(error).__name__}: {error}"}
                healthy = xc.health_state()["ready"] and server.alive()
                outcome["ok"] = bool(outcome.get("ok")) and bool(outcome.get("engaged")) and base["ok"] and healthy
                notes = [outcome.get("notes") or ""]
                if not outcome.get("engaged") and outcome.get("applicable", True):
                    notes.append("enabled but not observed engaged")
                if not base["ok"]:
                    notes.append("group base checks failed")
                if not healthy:
                    notes.append("server unhealthy after the check")
                results[name].update(
                    applicable=outcome.get("applicable", True), engaged=bool(outcome.get("engaged")),
                    ok=outcome["ok"], notes="; ".join(n for n in notes if n),
                    evidence=outcome.get("evidence"), seconds=round(time.monotonic() - started, 1),
                    group=group, route=spec["route"])
                print(f"  {name}: engaged={results[name]['engaged']} ok={results[name]['ok']} {results[name]['notes']}",
                      flush=True)
                if not healthy:
                    break
            record["healthy_after"] = xc.health_state()["ready"] and server.alive()
            results["_groups"][group] = record
            return
        finally:
            server.stop()


def stage_lora(stage_dir):
    """The local Muse LoRA with dropout zeroed (a training-only field the serving
    loader refuses); the weights are symlinked, not copied or edited."""
    target = stage_dir / "lora" / "muse-cyber"
    target.mkdir(parents=True, exist_ok=True)
    config = json.loads((MUSE_LORA / "adapter_config.json").read_text())
    config.setdefault("lora_parameters", {})["dropout"] = 0.0
    params = config["lora_parameters"]
    config["lora_parameters"] = {k: params[k] for k in ("rank", "scale", "dropout", "keys") if k in params}
    (target / "adapter_config.json").write_text(json.dumps(config))
    link = target / "adapters.safetensors"
    if not link.exists():
        link.symlink_to(MUSE_LORA / "adapters.safetensors")


def run_kernels(model, stage_dir, facts):
    fam = xc.cc.family(model)
    arms = kernel_arms(model, stage_dir)
    rows = []
    if not arms:
        return rows, {"reason": NO_KERNELS.get(fam, "no kernel toggles for this family")}
    import thermal_ladder as tl
    survival = facts.get("kernel_env_survival") or {}
    measured = {}
    for arm in arms:
        env = {**arm["env"], "MLX2_SERIES_KERNEL_ENV": json.dumps(arm["env"])}
        server = xc.Server(f"fused-{arm['arm']}", arm["command"], xc.stage_env(model, extra_env=env, shim=True),
                           stage_dir / f"server-fused-{arm['arm']}.log")
        print(f"FUSED arm {arm['arm']} env={arm['env']}", flush=True)
        try:
            startup = server.start()
            if not startup["ready"]:
                measured[arm["arm"]] = {"startup_error": startup.get("reason"), "log_tail": startup.get("log_tail", "")[-800:]}
                continue
            http = xc.HTTP(model_id=Path(model.path).name, timeout=900)
            before = http.settled()
            greedy = []
            for prompt in (xc.PROMPT_SUM, xc.PROMPT_CAPITAL, xc.PROMPT_OPEN):
                reply = http.chat(prompt, max_tokens=64)
                greedy.append({"prompt": prompt, "status": reply["status"], "text": xc.text_of(reply),
                               "correct": xc.oracle(prompt, xc.text_of(reply))})
            client = tl.Stream(xc.BASE_URL, 900, Path(model.path).name)
            speed = {}
            try:
                one = client.request("Write a long story about a lighthouse keeper and the sea.", 128)
                four = client.batch([f"Write a long story about lighthouse keeper number {i}." for i in range(4)], 128)
                speed = {"tok_s_1": one.get("decode_tokens_per_second"),
                         "tok_s_4_aggregate": four.get("aggregate_completion_tokens_per_second"),
                         "tok_s_4_per_stream": tl.median([r.get("decode_tokens_per_second") for r in four["rows"]])}
            except Exception as error:  # noqa: BLE001
                speed = {"error": f"{type(error).__name__}: {error}"[:300]}
            after = http.settled()
            effective = xc.dig(after, "settings", "environment", default={}) or {}
            counter = None
            if arm["counter"]:
                a, b = xc.dig(after, *arm["counter"], default=None), xc.dig(before, *arm["counter"], default=None)
                if isinstance(a, dict):  # e.g. moe dispatches by mode
                    a, b = sum(a.values()), sum((b or {}).values())
                counter = (a - (b or 0)) if isinstance(a, (int, float)) else None
            measured[arm["arm"]] = {
                "greedy": greedy, "speed": speed, "counter_delta": counter,
                "env_effective": {k: effective.get(k) for k in arm["env"]},
                "healthy": xc.health_state()["ready"] and server.alive(), "load_seconds": startup.get("load_seconds")}
        finally:
            server.stop()
    base = measured.get("base") or {}
    for arm in arms[1:]:
        m = measured.get(arm["arm"]) or {}
        applies = True
        notes = []
        info = survival.get(arm["arm"]) or {}
        if info and not info.get("survives"):
            notes.append("adapter configure_environment() overwrites "
                         + ",".join(k for k, v in info.get("requested", {}).items()
                                    if (info.get("effective") or {}).get(k) != v)
                         + "; applied via kernel_shim")
        if "startup_error" in m or "startup_error" in base:
            rows.append({"kernel": arm["arm"], "applicable": applies, "engaged": False, "output_equal": None,
                         "tok_s_fused": None, "tok_s_base": None, "ok": False,
                         "notes": "; ".join(notes + [f"startup failed: {m.get('startup_error') or base.get('startup_error')}"])})
            continue
        env_ok = all(str(m["env_effective"].get(k)) == str(v) for k, v in arm["env"].items())
        if arm["counter"]:
            engaged = bool(env_ok and (m.get("counter_delta") or 0) > 0)
            notes.append(f"counter {'.'.join(arm['counter'])} delta={m.get('counter_delta')}")
        else:
            engaged = None if env_ok else False
            notes.append("no runtime counter exported; engagement = effective environment only"
                         if env_ok else "requested environment not effective")
        equal = [a["text"] == b["text"] for a, b in zip(m["greedy"], base.get("greedy") or [], strict=False)]
        output_equal = bool(equal) and all(equal)
        correct = all(g["status"] == 200 and g["correct"] is not False for g in m["greedy"])
        ok = bool(m.get("healthy")) and correct and output_equal and engaged is not False
        if not output_equal:
            notes.append(f"greedy differs from base on prompts {[i for i, e in enumerate(equal) if not e]}")
        rows.append({"kernel": arm["arm"], "applicable": applies, "engaged": engaged, "output_equal": output_equal,
                     "tok_s_fused": {k: m["speed"].get(k) for k in ("tok_s_1", "tok_s_4_aggregate", "tok_s_4_per_stream")},
                     "tok_s_base": {k: (base.get("speed") or {}).get(k) for k in ("tok_s_1", "tok_s_4_aggregate",
                                                                                "tok_s_4_per_stream")},
                     "ok": ok, "notes": "; ".join(notes), "env": arm["env"]})
    base_ok = "startup_error" not in base and all(g["status"] == 200 and g["correct"] is not False
                                                   for g in base.get("greedy") or [{"status": None, "correct": False}])
    return rows, {"base_arm_ok": base_ok, "base_env": arms[0]["env"], "arms": measured, "note": KERNEL_NOTE}


# --- driver ------------------------------------------------------------------

def plan(model, facts, bitexact_api, stage_dir, policy_dir):
    table = applicability(model, facts, bitexact_api)
    groups = {}
    for name, group in FEATURES:
        if table[name][0]:
            groups.setdefault(group, []).append(name)
    specs = [group_spec(model, g, feats, stage_dir, policy_dir=policy_dir) for g, feats in groups.items()]
    return table, [s for s in specs if s], kernel_arms(model, stage_dir)


def run(model, stage_dir):
    probe = xc.probe([model], {model.name: kernel_env_request(model)})
    facts = probe["models"][model.name]
    table, specs, _ = plan(model, facts, probe["bitexact_api"], stage_dir, stage_dir / "policies")
    results = {"_groups": {}}
    for name, _group in FEATURES:
        applies, reason = table[name]
        results[name] = {"name": name, "applicable": applies, "engaged": False if applies else None,
                         "ok": None if not applies else False, "notes": reason}
    if cc_requires_vlm(model):
        runtime = xc.cc.verify_mlx_vlm_runtime(env=xc.stage_env(model), model=model)
        results["_groups"]["mlx_vlm_runtime"] = runtime
    for spec in specs:
        run_group(model, spec["group"], spec["features"], stage_dir, results, stage_dir / "policies")
        if "cache_capsules" in spec["features"]:
            run_capsule_control(model, (results["_groups"].get(spec["group"]) or {}).get("command"),
                                stage_dir, results)
        xc.save_json(stage_dir / "partial.json", results)
    kernels, kernel_detail = run_kernels(model, stage_dir, facts)
    features = [{key: results[name].get(key) for key in ("name", "applicable", "engaged", "ok", "notes")}
                for name, _ in FEATURES]
    applicable = [f for f in features if f["applicable"]] + [k for k in kernels if k["applicable"]]
    status = "pass" if all(f["ok"] for f in applicable) else "partial"
    summary = {
        "features": features, "kernels": kernels,
        "applicable": len(applicable), "ok": sum(bool(f["ok"]) for f in applicable),
        "failed": [f.get("name") or f.get("kernel") for f in applicable if not f["ok"]],
        "detail": {name: {k: v for k, v in results[name].items() if k not in ("name",)} for name, _ in FEATURES},
        "groups": results["_groups"], "kernel_detail": kernel_detail,
        "adapter": facts.get("adapter"), "mlx": probe.get("mlx"),
    }
    return status, summary


def cc_requires_vlm(model):
    return xc.cc.requires_mlx_vlm(model)


def dry_run(model):
    xc.cpu_only()
    from mlx2.server import build_parser
    probe = xc.probe([model], {model.name: kernel_env_request(model)})
    facts = probe["models"][model.name]
    with tempfile.TemporaryDirectory(prefix="series-experimental-dry-") as scratch:
        stage_dir = Path(scratch) / xc.stage_name(ROUND, model, model.default_route)
        table, specs, arms = plan(model, facts, probe["bitexact_api"], stage_dir, stage_dir / "policies")
        parser = build_parser()
        print(f"EXPERIMENTAL {model.name} adapter={facts['adapter']} family={xc.cc.family(model)}")
        for name, _group in FEATURES:
            applies, reason = table[name]
            print(f"  FEATURE {name}: {'applicable' if applies else 'n/a'}{'' if applies else ' (' + reason + ')'}")
        for spec in specs:
            if spec["group"] == "lora":
                stage_lora(stage_dir)
            xc.validate_command(spec["command"], parser)
            print(f"  SERVER[{spec['group']}] route={spec['route']} {shlex.join(spec['command'])}")
            if spec["policy"]:
                print(f"    POLICY {json.dumps(spec['policy'], sort_keys=True)}")
            print("    CHECK base: short chat (95), repeated prefix (APCv2 hit), streaming (Ottawa)")
            for name in spec["features"]:
                print(f"    CHECK {name}: {CHECK_TEXT[name]}")
        if arms:
            survival = facts.get("kernel_env_survival") or {}
            for arm in arms:
                xc.validate_command(arm["command"], parser)
                info = survival.get(arm["arm"]) or {}
                clobbered = [k for k, v in (info.get("requested") or {}).items()
                             if (info.get("effective") or {}).get(k) != v]
                print(f"  SERVER[fused:{arm['arm']}] route={arm['route']} env={json.dumps(arm['env'], sort_keys=True)} "
                      f"shim=kernel_shim/sitecustomize.py overwritten_by_adapter={clobbered or 'none'}")
                print(f"    {shlex.join(arm['command'])}")
                if arm["arm"] != "base":
                    print(f"    CHECK greedy x3 == base, answers correct, healthy, 1/4-stream decode tok/s; "
                          f"engaged via {'.'.join(arm['counter']) if arm['counter'] else 'effective env only'}")
        else:
            print(f"  FUSED: none ({NO_KERNELS.get(xc.cc.family(model), 'no kernel toggles')})")
    print(f"  RESULT -> {xc.JOBS / model.name / (ROUND + '.json')}")
    return {"model": model.name, "features": {n: table[n][0] for n, _ in FEATURES},
            "kernels": [a["arm"] for a in arms[1:]], "bitexact_api": probe["bitexact_api"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="series model name (campaign_config.MODELS)")
    parser.add_argument("--dry-run", action="store_true", help="print server commands and checks; start nothing")
    args = parser.parse_args(argv)
    model = xc.model_by_name(args.model)
    if args.dry_run:
        dry_run(model)
        print(f"DRY-RUN PASS experimental {model.name}")
        return 0
    stage = xc.stage_name(ROUND, model, model.default_route)
    stage_dir = xc.RUN / "results" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    xc.ensure_runtime_files()
    started = time.time()
    try:
        with xc.Ownership():
            status, summary = run(model, stage_dir)
    except Exception as error:  # noqa: BLE001 - the series records, never crashes
        import traceback
        status, summary = "error", {"error": f"{type(error).__name__}: {error}",
                                    "traceback": traceback.format_exc()[-3000:]}
    xc.write_job(ROUND, model, stage, status, started, summary)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
