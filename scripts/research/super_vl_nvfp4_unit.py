#!/usr/bin/env python3
"""Nemotron 3.5 Super VL NVFP4 export: one-artifact smoke and LiLiCorr A/B.

``owned OUT`` takes a desktop CPG lease and the paired GPU locks and runs
``unit OUT`` under them.  The unit starts one loopback server per arm, in its
own process group, and per arm sends at temperature 0 with the generated-token
trace:

* ``code`` then ``code_warm``: the same text prompt twice (APCv2 cold/warm);
* ``image`` then ``image_warm`` on media-capable arms (the red circle / blue
  square fixture, oracle words), cold then restored from APCv2;
* the GSM8K acceptance harness (``scripts/spec_acceptance_gsm8k.py``,
  base-url mode) for ``--gsm8k`` questions.

Arms: ``ordinary``; ``native_mtp`` (adapter default route); ``lilicorr``
(external LiLiCorr BF16, adapter policy defaults); ``lilicorr_kernels`` (the
same plus the four short-exact Mamba kernel keys native MTP and prompt lookup
already default to).  Every server group is stopped in ``finally``.  A
nonempty evidence directory is refused.  Smoke and attribution evidence
without thermal holds: never qualification, not publishable statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/model_matrix"))
sys.path.insert(0, str(ROOT / "src"))
import media_plain_parity as mpp  # noqa: E402
import pld_parity as pp  # noqa: E402
import run_matrix as rm  # noqa: E402

MODELS = Path.home() / "mlx-models"
TARGET = MODELS / "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
LILICORR = MODELS / "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-LiLiCorr-mlx-bf16"
LILICORR_NVFP4 = MODELS / "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-LiLiCorr-mlx-nvfp4"
KERNELS = dict(pp.ROW_EXACT_KERNELS_POLICY)
ARMS = {
    "ordinary": {"flags": ["--ordinary"], "media": True},
    # The adapter route without the best-route default (since 30aa296fd the
    # no-flag start pairs the LiLiCorr BF16 drafter: arm ``default``).
    "native_mtp": {"flags": ["--no-auto-route"], "media": False},
    "default": {"flags": [], "media": False},
    "default_b": {"flags": [], "media": False},  # a closing bracket of the same arm
    "default_guarded": {"flags": [], "media": False,
                        "env": {"MLX2_SUPER_TREE_COST_HANDOFF": "guarded"}},
    # The LiLiCorr chain after the tree8 default (operator opt-out).
    "lilicorr_chain_optout": {"flags": ["--external-draft"], "media": False,
                              "policy": {"draft_model": str(LILICORR),
                                         "batch_size_route": "chain"}},
    "lilicorr": {"flags": ["--external-draft"], "policy": {"draft_model": str(LILICORR)},
                 "media": False},
    "lilicorr_kernels": {"flags": ["--external-draft"],
                         "policy": {"draft_model": str(LILICORR), **KERNELS}, "media": False},
    "lilicorr_nvfp4_kernels": {"flags": ["--external-draft"],
                               "policy": {"draft_model": str(LILICORR_NVFP4), **KERNELS},
                               "media": False},
    # Fused Mamba step kernels on the ordinary route (lever 5 A/B).
    "ordinary_mamba_short": {"flags": ["--ordinary"], "media": False,
                             "policy": {"mamba_conv": "fused_short",
                                        "mamba_ssm": "fused_short_gate"}},
    "ordinary_mamba_b1": {"flags": ["--ordinary"], "media": False,
                          "policy": {"mamba_conv": "fused_b1", "mamba_norm": "fused_b1"}},
    # Single-pass fused exact verify kernel (MLX2_FP_ROW_EXACT_KERNEL=fused).
    "native_mtp_fused": {"flags": [], "media": False,
                         "env": {"MLX2_FP_ROW_EXACT_KERNEL": "fused"}},
    "lilicorr_kernels_fused": {"flags": ["--external-draft"],
                               "policy": {"draft_model": str(LILICORR)}, "media": False,
                               "env": {"MLX2_FP_ROW_EXACT_KERNEL": "fused"}},
    # One-token routed-decode kernels + compiled scaled epilogue.
    "ordinary_routed": {"flags": ["--ordinary"], "media": False, "env": {"MLX2_NEMOTRON_ROUTED_DECODE": "on", "MLX2_SCALED_EPILOGUE": "compiled"}},
    "native_mtp_routed": {"flags": [], "media": False, "env": {"MLX2_NEMOTRON_ROUTED_DECODE": "on", "MLX2_SCALED_EPILOGUE": "compiled"}},
    "lilicorr_routed": {"flags": ["--external-draft"], "policy": {"draft_model": str(LILICORR)},
                        "media": False, "env": {"MLX2_NEMOTRON_ROUTED_DECODE": "on", "MLX2_SCALED_EPILOGUE": "compiled"}},
    # Exact parent-indexed tree verify (opt-in; runtime/nemotron_tree_executor):
    # B1 trees of up to 16 rows from LiLiCorr's lattice; chain arm = lilicorr.
    "lilicorr_tree": {"flags": ["--external-draft"], "media": False,
                      "policy": {"draft_model": str(LILICORR),
                                 "batch_size_route": "tree15_b1_chain_b2plus_v1"}},
    "lilicorr_tree8": {"flags": ["--external-draft"], "media": False,
                       "policy": {"draft_model": str(LILICORR),
                                  "batch_size_route": "tree15_b1_chain_b2plus_v1",
                                  "ast_tile_rows": 8}},
    "lilicorr_tree6": {"flags": ["--external-draft"], "media": False,
                        "policy": {"draft_model": str(LILICORR),
                                   "batch_size_route": "tree15_b1_chain_b2plus_v1",
                                   "ast_tile_rows": 6}},
    "lilicorr_tree10": {"flags": ["--external-draft"], "media": False,
                        "policy": {"draft_model": str(LILICORR),
                                   "batch_size_route": "tree15_b1_chain_b2plus_v1",
                                   "ast_tile_rows": 10}},
    "lilicorr_tree12": {"flags": ["--external-draft"], "media": False,
                        "policy": {"draft_model": str(LILICORR),
                                   "batch_size_route": "tree15_b1_chain_b2plus_v1",
                                   "ast_tile_rows": 12}},
    # Fair chain control for tree6: five drafts, six verify rows.
    "lilicorr_chain5": {"flags": ["--external-draft"], "media": False,
                        "policy": {"draft_model": str(LILICORR), "num_draft": 5}},
    # 9..16-row trees with the opt-in chunked fused exact FP8/NVFP4 kernel.
    "lilicorr_tree10c": {"flags": ["--external-draft"], "media": False,
                         "env": {"MLX2_FP_ROW_EXACT_KERNEL": "fused_chunked"},
                         "policy": {"draft_model": str(LILICORR),
                                    "batch_size_route": "tree15_b1_chain_b2plus_v1",
                                    "ast_tile_rows": 10}},
    # 9..16-row trees with the opt-in chunked fused exact FP8/NVFP4 kernel.
    "lilicorr_tree12c": {"flags": ["--external-draft"], "media": False,
                         "env": {"MLX2_FP_ROW_EXACT_KERNEL": "fused_chunked"},
                         "policy": {"draft_model": str(LILICORR),
                                    "batch_size_route": "tree15_b1_chain_b2plus_v1",
                                    "ast_tile_rows": 12}},
    # 9..16-row trees with the opt-in chunked fused exact FP8/NVFP4 kernel.
    "lilicorr_tree16c": {"flags": ["--external-draft"], "media": False,
                         "env": {"MLX2_FP_ROW_EXACT_KERNEL": "fused_chunked"},
                         "policy": {"draft_model": str(LILICORR),
                                    "batch_size_route": "tree15_b1_chain_b2plus_v1",
                                    "ast_tile_rows": 16}},
    # Round-phase timing (diagnostic; MLX2_EXTERNAL_ROUND_TIMING[_SYNC]) on
    # the default chain (num_draft 5) and on tree6: where a served round goes.
    "lilicorr_timing": {"flags": ["--external-draft"], "media": False,
                        "env": {"MLX2_EXTERNAL_ROUND_TIMING": "1"},
                        "policy": {"draft_model": str(LILICORR)}},
    "lilicorr_timing_sync": {"flags": ["--external-draft"], "media": False,
                             "env": {"MLX2_EXTERNAL_ROUND_TIMING": "1",
                                     "MLX2_EXTERNAL_ROUND_TIMING_SYNC": "1"},
                             "policy": {"draft_model": str(LILICORR)}},
    "lilicorr_tree6_timing_sync": {"flags": ["--external-draft"], "media": False,
                                   "env": {"MLX2_EXTERNAL_ROUND_TIMING": "1",
                                           "MLX2_EXTERNAL_ROUND_TIMING_SYNC": "1"},
                                   "policy": {"draft_model": str(LILICORR),
                                              "batch_size_route": "tree15_b1_chain_b2plus_v1",
                                              "ast_tile_rows": 6}},
    # Drafter-side round cost: early verify build behind the drafter
    # (MLX2_EXTERNAL_VERIFY_PREBUILD) and the Q8 drafter profile.
    "lilicorr_prebuild": {"flags": ["--external-draft"], "media": False,
                          "env": {"MLX2_EXTERNAL_VERIFY_PREBUILD": "1"},
                          "policy": {"draft_model": str(LILICORR)}},
    "lilicorr_prebuild_timing": {"flags": ["--external-draft"], "media": False,
                                 "env": {"MLX2_EXTERNAL_VERIFY_PREBUILD": "1",
                                         "MLX2_EXTERNAL_ROUND_TIMING": "1"},
                                 "policy": {"draft_model": str(LILICORR)}},
    # The prebuild with the drafter on its own stream (default) or shared.
    "lilicorr_prebuild_stream": {"flags": ["--external-draft"], "media": False,
                                 "env": {"MLX2_EXTERNAL_VERIFY_PREBUILD": "1",
                                         "MLX2_EXTERNAL_DRAFT_STREAM": "draft"},
                                 "policy": {"draft_model": str(LILICORR)}},
    "lilicorr_prebuild_stream_timing": {"flags": ["--external-draft"], "media": False,
                                        "env": {"MLX2_EXTERNAL_VERIFY_PREBUILD": "1",
                                                "MLX2_EXTERNAL_DRAFT_STREAM": "draft",
                                                "MLX2_EXTERNAL_ROUND_TIMING": "1"},
                                        "policy": {"draft_model": str(LILICORR)}},
    # Exact verify attention blocks with strip projections (opt-in).
    "default_attn_rows": {"flags": [], "media": False,
                          "env": {"MLX2_NEMOTRON_EXACT_ATTENTION_ROWS": "on"}},
    "default_prebuild": {"flags": [], "media": False,
                         "env": {"MLX2_EXTERNAL_VERIFY_PREBUILD": "1"}},
    # Quantized drafter with the multi-row drafter kernel (MLX2_DRAFTER_ROWS_QMV),
    # on the default tree8 route; BF16 arms bracket.
    "lilicorr_bf16": {"flags": ["--external-draft"], "media": False,
                      "policy": {"draft_model": str(LILICORR)}},
    "lilicorr_bf16_b": {"flags": ["--external-draft"], "media": False,
                        "policy": {"draft_model": str(LILICORR)}},
    "lilicorr_draftq4_rowsqmv": {"flags": ["--external-draft"], "media": False,
                                 "env": {"MLX2_DRAFTER_ROWS_QMV": "on"},
                                 "policy": {"draft_model": str(LILICORR),
                                            "draft_weight_quantization": "affine_q4_g64"}},
    "lilicorr_nvfp4_rowsqmv": {"flags": ["--external-draft"], "media": False,
                               "env": {"MLX2_DRAFTER_ROWS_QMV": "on"},
                               "policy": {"draft_model": str(LILICORR_NVFP4)}},
    "lilicorr_draftq8": {"flags": ["--external-draft"], "media": False,
                         "policy": {"draft_model": str(LILICORR),
                                    "draft_weight_quantization": "affine_q8_g64"}},
    "lilicorr_prebuild_draftq8": {"flags": ["--external-draft"], "media": False,
                                  "env": {"MLX2_EXTERNAL_VERIFY_PREBUILD": "1"},
                                  "policy": {"draft_model": str(LILICORR),
                                             "draft_weight_quantization": "affine_q8_g64"}},
    # After the NVFP4 4-row cap and FP32 router cache (defaults): the opt-in
    # fused general-layout kernel, and wider chains / trees retested.
    "lilicorr_general": {"flags": ["--external-draft"], "media": False,
                         "env": {"MLX2_FP_ROW_EXACT_GENERAL": "fused"},
                         "policy": {"draft_model": str(LILICORR)}},
    "lilicorr_chain7_general": {"flags": ["--external-draft"], "media": False,
                                "env": {"MLX2_FP_ROW_EXACT_GENERAL": "fused"},
                                "policy": {"draft_model": str(LILICORR), "num_draft": 7}},
    "lilicorr_tree8_general": {"flags": ["--external-draft"], "media": False,
                               "env": {"MLX2_FP_ROW_EXACT_GENERAL": "fused"},
                               "policy": {"draft_model": str(LILICORR),
                                          "batch_size_route": "tree15_b1_chain_b2plus_v1",
                                          "ast_tile_rows": 8}},
    "lilicorr_draftq4": {"flags": ["--external-draft"], "media": False,
                         "policy": {"draft_model": str(LILICORR),
                                    "draft_weight_quantization": "affine_q4_g64"}},
    "lilicorr_chain7": {"flags": ["--external-draft"], "media": False,
                        "policy": {"draft_model": str(LILICORR), "num_draft": 7}},
    "lilicorr_tree8b": {"flags": ["--external-draft"], "media": False,
                        "policy": {"draft_model": str(LILICORR),
                                   "batch_size_route": "tree15_b1_chain_b2plus_v1",
                                   "ast_tile_rows": 8}},
    "lilicorr_tree_serial": {"flags": ["--external-draft"], "media": False,
                             "policy": {"draft_model": str(LILICORR),
                                        "batch_size_route": "tree15_b1_chain_b2plus_v1",
                                        "ast_target_executor": "serial"}},
    # Controls for the NVFP4-export defaults (FP8 dense on): exported BF16.
    "ordinary_bf16dense": {"flags": ["--ordinary"], "media": False,
                           "env": {"MLX2_NVFP4_DENSE": "bf16"}},
    "native_mtp_bf16dense": {"flags": [], "media": False,
                             "env": {"MLX2_NVFP4_DENSE": "bf16"}},
    "lilicorr_kernels_bf16dense": {"flags": ["--external-draft"],
                                   "policy": {"draft_model": str(LILICORR), **KERNELS},
                                   "media": False, "env": {"MLX2_NVFP4_DENSE": "bf16"}},
}


def atomic(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def source() -> dict:
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", "src"]).returncode != 0
    return {"commit": commit, "src_dirty": dirty}


def checkpoint() -> tuple[str, str]:
    from mlx2.adapters.registry import inspect_model

    resolution = inspect_model(TARGET)
    return f"{TARGET.name}@sha256:{resolution.artifact_fingerprint}", resolution.adapter_type.__name__


def run_arm(name, spec, out: Path, args, model_id, ident) -> dict:
    out.mkdir(parents=True)
    flags = list(spec["flags"])
    if "policy" in spec:
        (out / "policy.json").write_text(json.dumps(spec["policy"], indent=1) + "\n")
        flags += ["--execution-policy", str(out / "policy.json")]
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(TARGET),
               "--host", "127.0.0.1", "--port", str(args.port), "--fabric", "off",
               "--max-context", "32768", "--cache-bytes", str(2 << 30), *flags]
    row = {"arm": name, "command": command, "requests": [], "started_at": time.time()}
    base = f"http://127.0.0.1:{args.port}"
    log = (out / "server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                              start_new_session=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT / "src"),
                                   **spec.get("env", {})})
    row["env"] = spec.get("env", {})
    try:
        row["load_seconds"] = rm.wait_health(base, server, 1800, lambda: None)
        code_prompt = [{"role": "user", "content": rm.CODE_PROMPT}]
        for kind in ("code", "code_warm"):
            body = pp.request_body(model_id, code_prompt, args.tokens, thinking=False)
            status, response, wall = pp.post(base, body, 900)
            summary = pp.summarize(kind, response, wall, "def ") if status == 200 else {
                "kind": kind, "http_status": status, "error": response}
            summary["cached_tokens"] = ((response.get("usage") or {}).get("prompt_tokens_details")
                                        or {}).get("cached_tokens") if status == 200 else None
            row["requests"].append(summary)
            print(json.dumps({"arm": name, "kind": kind, "status": status,
                              "tok_s": summary.get("decode_tokens_per_second"),
                              "cached": summary.get("cached_tokens")}), flush=True)
        if spec["media"]:
            for kind in ("image", "image_warm"):
                body = {"model": model_id, "messages": mpp.image_messages(), "temperature": 0,
                        "max_tokens": 96, "stream": False, "return_token_trace": True,
                        "enable_thinking": False}
                status, response, wall = pp.post(base, body, 900)
                summary = mpp.summarize(kind, status, response, wall)
                row["requests"].append(summary)
                print(json.dumps({"arm": name, "kind": kind, "status": status,
                                  "oracle": summary.get("oracle"),
                                  "cached": summary.get("cached_tokens")}), flush=True)
        if args.gsm8k:
            gsm = subprocess.run(
                [sys.executable, str(ROOT / "scripts/spec_acceptance_gsm8k.py"),
                 "--base-url", base, "--checkpoint", ident, "--output", str(out / "gsm8k"),
                 "--num-questions", str(args.gsm8k)],
                cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                capture_output=True, text=True)
            (out / "gsm8k.log").write_text(gsm.stdout + gsm.stderr)
            row["gsm8k_returncode"] = gsm.returncode
            receipt = out / "gsm8k" / "receipt.json"
            if receipt.is_file():
                row["gsm8k"] = json.loads(receipt.read_text())["arms"][0]["summary"]
            print(json.dumps({"arm": name, "gsm8k": row.get("gsm8k")}), flush=True)
        row["status"] = rm.http_json(base + "/v1/status", None, timeout=30)[1]
        row["passed"] = True
    except BaseException as error:  # noqa: BLE001 - recorded, then the next arm runs
        row["passed"] = False
        row["error"] = repr(error)
    finally:
        row["server_stop"] = rm.stop_server(server)
        log.close()
        row["finished_at"] = time.time()
        atomic(out / "arm.json", row)
    return row


def unit(out: Path, args) -> int:
    ident, adapter = checkpoint()
    report = {"schema": "mlx2.super-vl-nvfp4-unit.v1", "checkpoint": ident, "adapter": adapter,
              "source": source(), "host": "m5", "qualified": False,
              "evidence_kind": "smoke + attribution A/B, no thermal hold", "arms": []}
    model_id = TARGET.name
    for name in args.arms:
        row = run_arm(name, ARMS[name], out / name, args, model_id, ident)
        report["arms"].append({k: v for k, v in row.items() if k != "status"})
        atomic(out / "report.json", report)
    return 0 if all(arm["passed"] for arm in report["arms"]) else 1


def owned(out: Path, args) -> int:
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    ident, _ = checkpoint()
    job_id = "super-vl-nvfp4-" + out.name
    with DesktopCPGLease(out, job_id, checkpoint=ident, source_revision=source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out),
                   "--arms", *args.arms, "--gsm8k", str(args.gsm8k),
                   "--tokens", str(args.tokens), "--port", str(args.port)]
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                                 env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=120)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--gsm8k", type=int, default=30)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--port", type=int, default=8797)
    args = parser.parse_args()
    if args.mode == "owned":
        return owned(args.out, args)
    return unit(args.out, args)


if __name__ == "__main__":
    sys.exit(main())
