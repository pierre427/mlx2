#!/usr/bin/env python3
"""Media plain decode on the default prompt-lookup server, one artifact.

For one checkpoint whose adapter declares media plain decode on the
prompt-lookup route (``multimodal_plain_decode_routes``), this starts
``python -m mlx2.server`` twice, each in its own process group: first with
``--ordinary`` (the reference), then with no route flag (the capability
default: latched prompt lookup).  On each it waits for ``GET /health == 200``
and sends, at temperature 0 with the generated-token trace:

1. a copy-heavy text request (``text``);
2. an image request about a real image, cold (``image``);
3. the same image request again, its media prefix restored from APCv2
   (``image_warm``);
4. a text request and an image request at the same time
   (``concurrent_text`` / ``concurrent_image``).

It stops the server group in ``finally`` and refuses to overwrite a nonempty
evidence directory.  The default server must report ``prompt_lookup`` for
text and a ``media_decode`` receipt saying plain decode for every image
request; image token ids are compared with the ordinary server's.  The
caller owns the GPU lock (``scripts/run_with_gpu_locks.py``).  Smoke evidence,
never qualification.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pld_parity as pp  # noqa: E402
import run_matrix as rm  # noqa: E402

ROOT = rm.ROOT
SCHEMA = "mlx2.media-plain-parity-cell.v1"
IMAGE = ROOT / "qualification/runs/sept26-model-smokes/special/agnes-shapes.png"
IMAGE_QUESTION = (
    "Describe this image in one or two sentences: which shapes are there, "
    "what colour is each, and which is on the left?"
)
# The image shows a red circle on the left and a blue square on the right.
IMAGE_ORACLE = ("red", "circle", "blue", "square")


def image_messages():
    data = base64.b64encode(IMAGE.read_bytes()).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
                {"type": "text", "text": IMAGE_QUESTION},
            ],
        }
    ]


def summarize(kind, code, response, wall):
    if code != 200 or not isinstance(response, dict):
        return {"kind": kind, "http_status": code, "error": response}
    choice = (response.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    block = response.get("mlx2") or {}
    trace = block.get("token_trace") or {}
    speculation = block.get("speculation") if isinstance(block.get("speculation"), dict) else {}
    oracle = (
        all(word in content.lower() for word in IMAGE_ORACLE)
        if "image" in kind
        else "def unpaid_totals" in content
    )
    return {
        "kind": kind,
        "http_status": code,
        "wall_seconds": round(wall, 3),
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": (response.get("usage") or {}).get("completion_tokens"),
        "cached_tokens": ((response.get("usage") or {}).get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ),
        "token_ids": trace.get("token_ids"),
        "route": block.get("route"),
        "route_selection_source": block.get("route_selection_source"),
        "qualification": block.get("qualification"),
        "media_decode": block.get("media_decode"),
        "speculation": {
            key: speculation.get(key)
            for key in (
                "execution", "speculation", "decode", "plain_decode_reason",
                "proposed", "accepted", "verify_path_rounds", "plain_only_rounds",
                "media_prefill_inputs", "verify_forward", "target_width",
            )
            if key in speculation
        },
        "ordinary_compute_widths": block.get("ordinary_compute_widths"),
        "content": content,
        "oracle": oracle,
    }


def run_server(args, target, route, out: Path) -> dict:
    receipt = {
        "schema": SCHEMA,
        "checkpoint": target["checkpoint"],
        "model_path": target["path"],
        "route_flags": ["--ordinary"] if route == "ordinary" else [],
        "arm": route,
        "host": args.host_label,
        "source_revision": args.revision,
        "source_tracked_dirty": args.dirty,
        "evidence_kind": "smoke (implemented/observed-used only; never qualification)",
        "image": str(IMAGE.relative_to(ROOT)),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out.mkdir(parents=True, exist_ok=True)
    need = int(int(target.get("weight_bytes") or 0) * args.memory_factor) + (4 << 30)
    deadline = time.monotonic() + 180
    counters = rm.vm_counters()
    while rm.available_bytes(counters) < need and time.monotonic() < deadline:
        time.sleep(5)
        counters = rm.vm_counters()
    receipt["host_before"] = {
        "available_bytes": rm.available_bytes(counters),
        "swapouts": counters.get("Swapouts"),
        "swap_used_bytes": rm.swap_used_bytes(),
        "foreign_model_processes": rm.model_processes(),
    }
    if receipt["host_before"]["foreign_model_processes"]:
        receipt.update(status="skipped", reason="foreign model process present")
        rm.atomic_json(out / "cell.json", receipt)
        return receipt
    if receipt["host_before"]["available_bytes"] < need:
        receipt.update(status="skipped", reason="insufficient available memory")
        rm.atomic_json(out / "cell.json", receipt)
        return receipt
    swapouts_start = counters.get("Swapouts", 0)

    def swap_guard():
        now = rm.vm_counters().get("Swapouts", 0)
        if now - swapouts_start > args.swap_pages_limit:
            raise rm.SwapRise(f"Swapouts rose by {now - swapouts_start} pages")

    command = [
        sys.executable, "-u", "-m", "mlx2.server", "--model", target["path"],
        "--host", "127.0.0.1", "--port", str(args.port), "--fabric", "off",
        "--max-context", str(args.max_context), "--cache-bytes", str(args.cache_gib << 30),
        *receipt["route_flags"],
    ]
    receipt["server_command"] = command
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "MLX_ENABLE_TF32": "0", "PYTHONUNBUFFERED": "1"}
    base = f"http://127.0.0.1:{args.port}"
    process = sampler = None
    results = []
    try:
        with socket.socket() as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                raise RuntimeError(f"port {args.port} already in use")
        log = (out / "server.log").open("w")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        sampler = rm.Sampler(process.pid)
        sampler.start()
        receipt["load_seconds"] = round(rm.wait_health(base, process, args.load_timeout, swap_guard), 2)
        _code, status, _ = rm.http_json(base + "/v1/status", timeout=30)
        if isinstance(status, dict):
            settings = status.get("settings") or {}
            receipt["startup_status"] = {
                "artifact": status.get("artifact"),
                "route": settings.get("route"),
                "route_selection_source": status.get("route_selection_source"),
                "qualification": status.get("qualification"),
                "media_route": status.get("media_route"),
                "selected_capabilities": status.get("selected_capabilities"),
            }
        _code, models, _ = rm.http_json(base + "/v1/models", timeout=30)
        model_id = ((models.get("data") or [{}])[0].get("id")) if isinstance(models, dict) else None
        model_id = model_id or Path(target["path"]).name
        code, _response, _ = pp.post(
            base, pp.request_body(model_id, pp.WARMUP, 16, thinking=False), args.request_timeout
        )
        receipt["warmup_http_status"] = code

        def ask(kind, messages, max_tokens):
            swap_guard()
            body = pp.request_body(model_id, messages, max_tokens, thinking=False)
            code, response, wall = pp.post(base, body, args.request_timeout)
            return summarize(kind, code, response, wall)

        results.append(ask("text", pp.COPY_PROMPT, args.text_tokens))
        results.append(ask("image", image_messages(), args.image_tokens))
        results.append(ask("image_warm", image_messages(), args.image_tokens))
        concurrent = {}

        def worker(kind, messages, max_tokens):
            concurrent[kind] = ask(kind, messages, max_tokens)

        threads = [
            threading.Thread(target=worker, args=("concurrent_text", pp.COPY_PROMPT, args.text_tokens)),
            threading.Thread(target=worker, args=("concurrent_image", image_messages(), args.image_tokens)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        results.extend(concurrent[kind] for kind in ("concurrent_text", "concurrent_image"))
        if process.poll() is not None:
            raise RuntimeError(f"server died rc={process.returncode}")
        receipt["status"] = "completed"
    except rm.SwapRise as error:
        receipt.update(status="aborted_swap", error=str(error))
    except Exception as error:  # noqa: BLE001 - every exit writes evidence
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if sampler is not None:
            sampler.stop_event.set()
            receipt["peak_rss_bytes"] = sampler.peak_rss
            receipt["peak_phys_footprint_bytes"] = sampler.peak_footprint or None
        receipt["server_stop"] = rm.stop_server(process)
        counters = rm.vm_counters()
        receipt["host_after"] = {
            "available_bytes": rm.available_bytes(counters),
            "swapouts_delta_pages": counters.get("Swapouts", 0) - swapouts_start,
            "swap_used_bytes": rm.swap_used_bytes(),
        }
        receipt["requests"] = results
        log_path = out / "server.log"
        if log_path.is_file():
            receipt["server_log_tail"] = log_path.read_text(errors="replace").splitlines()[-40:]
        receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rm.atomic_json(out / "cell.json", receipt)
    return receipt


def verdicts(ordinary, default):
    """Gates of the decision (2026-10-21) on one checkpoint's two servers."""
    a = {r["kind"]: r for r in ordinary.get("requests", ()) if r.get("token_ids") is not None}
    b = {r["kind"]: r for r in default.get("requests", ()) if r.get("token_ids") is not None}
    checks = {}

    def plain_media(row):
        media = row.get("media_decode") or {}
        return (
            row.get("route") == "prompt_lookup"
            and media.get("speculation") == "disabled_for_media"
            and media.get("decode") == "ordinary"
            and (row.get("speculation") or {}).get("proposed") == 0
        )

    text = b.get("text") or {}
    checks["default_route_is_prompt_lookup"] = (
        (default.get("startup_status") or {}).get("route") == "prompt_lookup"
    )
    checks["text_reports_prompt_lookup"] = (
        text.get("route") == "prompt_lookup" and text.get("media_decode") is None
    )
    checks["text_speculated"] = (text.get("speculation") or {}).get("execution") == "prompt_lookup_verify"
    for kind in ("image", "image_warm", "concurrent_image"):
        row = b.get(kind) or {}
        checks[f"{kind}_plain_decode_receipt"] = plain_media(row)
        checks[f"{kind}_oracle"] = bool(row.get("oracle"))
    checks["image_media_prefill"] = ((b.get("image") or {}).get("media_decode") or {}).get(
        "media_prefill"
    )
    checks["image_warm_media_prefill"] = ((b.get("image_warm") or {}).get("media_decode") or {}).get(
        "media_prefill"
    )
    for kind in ("text", "image", "image_warm", "concurrent_text", "concurrent_image"):
        if kind in a and kind in b:
            index = pp.first_divergence(a[kind]["token_ids"], b[kind]["token_ids"])
            checks[f"{kind}_tokens_equal_ordinary"] = index is None
            if index is not None:
                checks[f"{kind}_first_divergence"] = index
    # The default server's media lane is a B1 plain lane even beside text.
    if "image" in a and "concurrent_image" in b:
        checks["concurrent_image_equals_ordinary_single"] = (
            pp.first_divergence(a["image"]["token_ids"], b["concurrent_image"]["token_ids"]) is None
        )
    if "concurrent_text" in b:
        checks["concurrent_text_speculated"] = (
            (b["concurrent_text"].get("speculation") or {}).get("execution") == "prompt_lookup_verify"
        )
    return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--host-label", default="m5")
    parser.add_argument("--port", type=int, default=8743)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--cache-gib", type=int, default=2)
    parser.add_argument("--text-tokens", type=int, default=512)
    parser.add_argument("--image-tokens", type=int, default=96)
    parser.add_argument("--load-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=1200)
    parser.add_argument("--memory-factor", type=float, default=1.08)
    parser.add_argument("--swap-pages-limit", type=int, default=4096)
    args = parser.parse_args(argv)
    inventory = json.loads(args.inventory.read_text())
    matches = [t for t in inventory["targets"]
               if args.target in {t.get("checkpoint"), t["path"], Path(t["path"]).name, t.get("name")}]
    if len(matches) != 1:
        parser.error(f"target {args.target!r} matched {len(matches)} inventory rows")
    target = matches[0]
    if (ROOT / ".git").exists():
        args.revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        args.dirty = bool(subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    else:
        args.revision, args.dirty = (ROOT / "SYNCED_HEAD").read_text().strip(), None
    unit = args.out_root / re.sub(r"[^A-Za-z0-9._-]+", "_", target["name"])
    if unit.exists() and any(unit.iterdir()):
        print(f"refusing to overwrite nonempty evidence directory: {unit}", file=sys.stderr)
        return 2
    cells = {}
    for arm in ("ordinary", "default"):
        cells[arm] = run_server(args, target, arm, unit / arm)
        print(json.dumps({"arm": arm, "status": cells[arm].get("status"),
                          "error": cells[arm].get("error")}), flush=True)
        if cells[arm].get("status") != "completed":
            break
    summary = {
        "schema": "mlx2.media-plain-parity-summary.v1",
        "checkpoint": target["checkpoint"],
        "source_revision": args.revision,
        "source_tracked_dirty": args.dirty,
        "host": args.host_label,
        "arms": {arm: cell.get("status") for arm, cell in cells.items()},
        "checks": (
            verdicts(cells["ordinary"], cells["default"])
            if all(cells.get(arm, {}).get("status") == "completed" for arm in ("ordinary", "default"))
            else None
        ),
    }
    rm.atomic_json(unit / "summary.json", summary)
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
