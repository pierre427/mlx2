"""One bounded exact-artifact reference/server/APCv2 repair smoke, not qualification."""

from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL_NAME = "GLM-5.3-Flash-REAP50-Q3_K_M.gguf"
MODEL_SHA = "5a865918aea87a362ff8ae419581a0447929bfd68534a44d1125b394c272762d"
REQUEST = {
    "messages": [
        {"role": "user", "content": "What is 2 + 2? Reply with just the number."}
    ],
    "temperature": 0,
    "max_tokens": 12,
    "enable_thinking": False,
    "logprobs": True,
    "top_logprobs": 2,
}


def request_for(args):
    if args.scenario == "math":
        return REQUEST
    if args.scenario == "long":
        return {
            **REQUEST,
            "messages": [
                {
                    "role": "user",
                    "content": "Background:\n"
                    + "alpha beta gamma delta\n" * 420
                    + "\nThe secret code is K9R7. Reply with only that code.",
                }
            ],
        }
    return {
        **REQUEST,
        "messages": [
            {
                "role": "user",
                "content": "Repeat this sequence exactly, without adding anything: red blue green red blue green red blue green red blue green",
            }
        ],
        "max_tokens": 20,
    }


def atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temp.replace(path)


def identity():
    root = Path(__file__).resolve().parents[2]
    paths = (
        list(root.glob("src/mlx2/**/*glm53*.py"))
        + list(root.glob("src/mlx2/runtime/gguf/*.py"))
        + [
            root / "src/mlx2/runtime/gguf/k_metal.py",
            root / "src/mlx2/runtime/gguf/packed_metal.py",
            root / "src/mlx2/runtime/gguf/reader.py",
            root / "src/mlx2/runtime/recurrent_state_codec.py",
            root / "src/mlx2/runtime/drafters/dflash_base.py",
            root / "src/mlx2/runtime/drafters/dflash2_config.py",
            root / "src/mlx2/runtime/drafters/dflash2.py",
            root / "src/mlx2/runtime/drafters/native_mtp.py",
            root / "src/mlx2/adapters/dflash2.py",
            Path(__file__).resolve(),
        ]
    )
    return {
        "checkpoint": MODEL_NAME + "@" + MODEL_SHA,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_files": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "qualified": False,
        "scope": "bounded repair smoke; no formal qualification or controlled performance claim",
    }


def health():
    return {
        name: subprocess.run(
            command, capture_output=True, text=True, check=False
        ).stdout
        for name, command in {
            "thermal": ["pmset", "-g", "therm"],
            "swap": ["sysctl", "vm.swapusage"],
            "processes": ["ps", "-axo", "pid=,comm="],
        }.items()
    }


def reference(args):
    import mlx.core as mx

    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    request = request_for(args)
    record = {**identity(), "status": "running", "before": health(), "request": request}
    atomic(args.out / "reference.json", record)
    adapter = None
    try:
        print("loading ordinary-reference model", flush=True)
        adapter = Glm53GGUFAdapter(
            args.model,
            execution_policy={
                "glm53_math": "reference",
                "glm53_ast_target": args.target_implementation,
            },
        )
        tokens = adapter.prompt_tokens(request)
        cache = adapter.model.make_cache()
        for start in range(0, len(tokens), 32):
            logits = adapter.model(
                mx.array([tokens[start : start + 32]]), cache, logits_to_keep=1
            )
            mx.eval(logits, [c.state for c in cache])
            if start and start % 320 == 0:
                print(f"reference prefill {start + 32}/{len(tokens)}", flush=True)
        output = []
        for _ in range(request["max_tokens"]):
            token = int(mx.argmax(logits[0, -1]).item())
            if token in adapter.tokenizer.eos_token_ids:
                break
            output.append(token)
            logits = adapter.model(mx.array([[token]]), cache, logits_to_keep=1)
        record.update(
            status="completed",
            prompt_tokens=tokens,
            output_tokens=output,
            text=adapter.tokenizer.decode(output),
            diagnostics=adapter.diagnostics(),
            finite=bool(mx.all(mx.isfinite(logits)).item()),
            after=health(),
        )
        print(
            json.dumps({"reference_text": record["text"], "tokens": output}), flush=True
        )
        if args.oracle:
            from glm53_state_oracle import run

            result = run(
                adapter,
                args.out / "state-oracle.json",
                MODEL_NAME + "@" + MODEL_SHA,
                record["source_revision"],
            )
            record["state_oracle"] = result["status"]
        if not output or not record["finite"]:
            raise AssertionError("ordinary reference has no finite continuation")
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        atomic(args.out / "reference.json", record)
        if adapter is not None:
            adapter.close()


def driver(args):
    if args.out.exists():
        raise ValueError("refusing nonempty/reused evidence destination")
    args.out.mkdir(parents=True)
    record = {**identity(), "status": "running", "before": health()}
    atomic(args.out / "unit.json", record)
    server = None
    old_handlers = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(signum, interrupted)
    try:
        with (args.out / "reference.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--mode",
                    "reference",
                    "--scenario",
                    args.scenario,
                    *(["--oracle"] if args.oracle else []),
                    "--target-implementation",
                    args.target_implementation,
                    "--model",
                    args.model,
                    "--out",
                    str(args.out),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        reference_record = json.loads((args.out / "reference.json").read_text())
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = [
            sys.executable,
            "-m",
            "mlx2.server",
            "--model",
            args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--fabric",
            "off",
            "--ordinary",
            "--max-context",
            "4096" if args.scenario == "long" else "512",
            "--prefill-step",
            "32",
            "--max-lanes",
            "1",
            "--max-inflight",
            "1",
            "--cache-bytes",
            str(512 << 20),
            "--cache-dir",
            str(args.out / "cache"),
            "--no-thinking-auto-calibration",
            "--lane-matmul",
            "off",
            "--int8-prefill",
            "off",
            "--no-usage-stats",
        ]
        policy_path = args.out / "execution-policy.json"
        atomic(
            policy_path,
            {
                "glm53_ast_target": args.target_implementation,
                "glm53_draft_activation": args.draft_activation,
            },
        )
        command += ["--execution-policy", str(policy_path)]
        if args.route != "ordinary":
            command.remove("--ordinary")
            draft = {
                "pld": "pld",
                "mtp": "mtp",
                "modal": str(Path.home() / "mlx-models" / "GLM-5.3-Flash-DFlash"),
                "inco": str(Path.home() / "mlx-models" / "GLM-5.3-Flash-DFlash2"),
            }[args.route]
            command += ["--ast", "--ast-drafter", draft]
            if not args.ast_defaults:
                command += ["--ast-tile-rows", "4", "--ast-topology", "chain"]
        log = (args.out / "server.log").open("w")
        try:
            server = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            record["server"] = {"pid": server.pid, "command": command}
            atomic(args.out / "unit.json", record)
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited {server.returncode}")
                try:
                    with urllib.request.urlopen(
                        url + "/v1/models", timeout=2
                    ) as response:
                        models = json.load(response)
                    if models["data"][0].get("loaded"):
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("server model loading")
                    time.sleep(1)
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("server readiness")
                    time.sleep(1)
            model = models["data"][0]["id"]
            record["models"] = models
            atomic(args.out / "unit.json", record)
            cells = []
            for arm in ("cold", "warm"):
                before = health()
                body = {**request_for(args), "model": model}
                req = urllib.request.Request(
                    url + "/v1/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=900) as response:
                        result = json.load(response)
                except urllib.error.HTTPError as error:
                    (args.out / (arm + "-error.txt")).write_bytes(error.read())
                    raise
                cell = {
                    "arm": arm,
                    "request": body,
                    "response": result,
                    "before": before,
                    "after": health(),
                }
                atomic(args.out / (arm + ".json"), cell)
                cells.append(cell)
            texts = [
                c["response"]["choices"][0]["message"].get("content", "") for c in cells
            ]
            logprobs = [c["response"]["choices"][0].get("logprobs") for c in cells]
            cold_tokens = [x["token"] for x in (logprobs[0] or {}).get("content", [])]
            warm_tokens = [x["token"] for x in (logprobs[1] or {}).get("content", [])]
            cached = (
                cells[1]["response"]
                .get("usage", {})
                .get("prompt_tokens_details", {})
                .get("cached_tokens", 0)
            )
            with urllib.request.urlopen(url + "/v1/status", timeout=15) as response:
                atomic(args.out / "status.json", json.load(response))
            record.update(
                status="completed",
                texts=texts,
                exact_text_parity=texts[0] == texts[1],
                exact_token_parity=cold_tokens == warm_tokens,
                token_evidence_available=bool(cold_tokens),
                warm_cached_tokens=cached,
                reference_text=reference_record["text"],
                reference_text_parity=reference_record["text"] == texts[0],
                after=health(),
            )
            if texts[0] != texts[1] or cold_tokens != warm_tokens:
                raise AssertionError("width-one APCv2 replay drift")
            if cached <= 0:
                raise AssertionError("warm replay has no observed APCv2 reuse")
            print(
                json.dumps(
                    {
                        k: record[k]
                        for k in (
                            "texts",
                            "warm_cached_tokens",
                            "reference_text_parity",
                        )
                    }
                ),
                flush=True,
            )
        finally:
            if server is not None and server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=30)
            log.close()
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["server_stopped"] = server is None or server.poll() is not None
        atomic(args.out / "unit.json", record)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("unit", "reference"), default="unit")
    parser.add_argument(
        "--route",
        choices=("ordinary", "pld", "mtp", "modal", "inco"),
        default="ordinary",
    )
    parser.add_argument("--scenario", choices=("math", "copy", "long"), default="math")
    parser.add_argument(
        "--target-implementation", choices=("serial", "layered"), default="serial"
    )
    parser.add_argument(
        "--draft-activation", choices=("target", "bfloat16"), default="target"
    )
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--ast-defaults", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (reference if args.mode == "reference" else driver)(args)
