"""Bounded source-bound D3 diagnostic under an already-owned GPU window.

This probe produces diagnostic evidence, not qualification or a performance
result.  It validates the live CPG lease and both ancestor-owned host locks
before importing MLX.  A parent ``run_qualification_owned.py`` process keeps
that lease live while this process runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

EXPECTED_ARTIFACT_FILES = {
    "LICENSE": (
        11_357,
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    ),
    "MODEL_MANIFEST.json": (
        11_428,
        "3bae8989a6feb36a66bf4d8c44112dc63199d0dc69f564dc895fe368b1e08e59",
    ),
    "NOTICE": (106, "0e7694dee1927d83b3637be7a8f989b941ebd884dd40767b995c7adee484789b"),
    "chat_template.jinja": (
        7_755,
        "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    ),
    "config.json": (
        2_572,
        "5e529d9aea2f8d387750acf5b02cc32a47d7e87dc65d6d008fba98e5d15acc4d",
    ),
    "decision_config.json": (
        5_504,
        "30b6095b945d5a93697868afdf9be9123ea892085fbaff3078112c48da20dc6b",
    ),
    "merges.txt": (
        3_353_259,
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    ),
    "model-00001-of-00001.safetensors": (
        1_188_677_968,
        "91bb8618450a7f628f1432bbced76bc19ee7346c93d191a228fdd33c25a57a9d",
    ),
    "model.safetensors.index.json": (
        25_448,
        "7d3951baba643a2c4fe2263d8602ef37b1c1f804b58949a5469449da6a588b03",
    ),
    "readout.safetensors": (
        1_044_560,
        "de33d5dc795bbd77168e40bfadc340cca9b9f163b88744c10e1a788f4e166d3d",
    ),
    "tokenizer.json": (
        12_807_982,
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    "tokenizer_config.json": (
        16_709,
        "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    ),
    "vocab.json": (
        6_722_759,
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    ),
}
SOURCE_SUPPORT_PATHS = (
    "scripts/probe_d3_artifact.py",
    "scripts/qualification_gpu_ownership.py",
    "scripts/run_with_gpu_locks.py",
    "scripts/run_qualification_owned.py",
    "scripts/thermal_probe.swift",
)
MODEL_PROCESS = re.compile(
    r"mlx2\.(?:server|decisions\.server)(?:\s|$)|mlx_lm\.server|"
    r"mlx_vlm\.server|rapid-mlx|llama-server",
    re.IGNORECASE,
)
GPU_DIAGNOSTIC_PROCESS = re.compile(
    r"(?:probe|qualif(?:y|ication)|benchmark|bench|smoke)[-_].*"
    r"(?:mlx|metal|gpu|model|inference|d3)|"
    r"(?:mlx|metal|gpu|model|inference|d3).*"
    r"(?:probe|qualif(?:y|ication)|benchmark|bench|smoke)[-_]",
    re.IGNORECASE,
)
GPU_SERVICE = re.compile(
    r"(?:^|[.\s])(?:mlx2?|llama|rapid-mlx)(?:[.\s-]|$)|storyforge.*mlx",
    re.I,
)


def _digest(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_artifact_files(root: Path) -> dict:
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    if actual_names != set(EXPECTED_ARTIFACT_FILES):
        raise ValueError(
            "pinned D3 artifact inventory changed: "
            f"missing={sorted(set(EXPECTED_ARTIFACT_FILES) - actual_names)}, "
            f"extra={sorted(actual_names - set(EXPECTED_ARTIFACT_FILES))}"
        )
    evidence = {}
    for name, (size, digest) in sorted(EXPECTED_ARTIFACT_FILES.items()):
        path = root / name
        actual = {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        if actual != {"bytes": size, "sha256": digest}:
            raise ValueError(f"pinned D3 artifact file changed: {name}: {actual}")
        evidence[name] = actual
    return evidence


def _source_hashes(root: Path) -> dict[str, str]:
    # Hash the complete local Python runtime rather than maintaining a fragile
    # hand-written import closure.  This includes the direct D3 dependencies
    # and their transitive imports (metrics, base/qwen-next math, precise ops,
    # GDN state, invariant prefill and ragged observation helpers).
    source = root / "src/mlx2"
    paths = [*SOURCE_SUPPORT_PATHS]
    paths.extend(str(path.relative_to(root)) for path in sorted(source.rglob("*.py")))
    return {name: _file_sha256(root / name) for name in paths}


def _thermal(root: Path) -> dict:
    result = subprocess.run(
        ["/usr/bin/swift", str(root / "scripts/thermal_probe.swift")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict) or type(value.get("thermal_state")) is not int:
        raise RuntimeError("thermal probe returned a malformed record")
    return value


def _contamination_inventory() -> dict:
    """Inventory known model servers and GPU diagnostics outside our ancestry."""
    from qualification_gpu_ownership import _ancestor_pids
    from run_with_gpu_locks import foreign_model_processes

    ancestors = _ancestor_pids(os.getpid())
    rows = subprocess.check_output(["ps", "-axo", "pid=,ppid=,command="], text=True)
    extended = []
    for row in rows.splitlines():
        fields = row.strip().split(None, 2)
        if len(fields) != 3 or int(fields[0]) in ancestors:
            continue
        if MODEL_PROCESS.search(fields[2]) or GPU_DIAGNOSTIC_PROCESS.search(fields[2]):
            extended.append(row.strip()[:500])
    services = subprocess.check_output(["launchctl", "list"], text=True)
    loaded_services = [
        row.strip()[:500] for row in services.splitlines() if GPU_SERVICE.search(row)
    ]
    active_services = [
        row for row in loaded_services if row.split(None, 1)[0].isdigit()
    ]
    canonical = foreign_model_processes()
    return {
        "canonical_foreign_model_processes": canonical,
        "extended_gpu_processes": extended,
        "loaded_gpu_services": loaded_services,
        "active_gpu_services": active_services,
        "scope": (
            "run_with_gpu_locks.foreign_model_processes plus non-ancestor MLX/Metal/"
            "GPU/model/inference probe, qualification, benchmark and smoke processes; "
            "loaded launchd services named for MLX, mlx2, llama, rapid-mlx or "
            "StoryForge MLX"
        ),
        "contaminated": bool(canonical or extended or active_services),
    }


def _environment(root: Path) -> dict:
    return {
        "thermal": _thermal(root),
        "swap": subprocess.check_output(
            ["/usr/sbin/sysctl", "vm.swapusage"], text=True
        ).strip(),
        "contamination": _contamination_inventory(),
    }


def _watch_thermal(
    root: Path, stop: threading.Event, samples: list, errors: list
) -> None:
    """Sample thermals during execution; retain probe errors for the main thread."""
    while not stop.is_set():
        try:
            samples.append({"monotonic_seconds": time.monotonic(), **_thermal(root)})
        except BaseException as error:  # noqa: BLE001 - watcher failure is evidence
            errors.append(f"{type(error).__name__}: {error}")
            return
        stop.wait(2.0)


def _ownership(args) -> dict:
    from qualification_gpu_ownership import require_qualification_lease
    from run_qualification_owned import observe

    paired = require_qualification_lease(
        task_id=args.cpg_task,
        cpg_owner_lock=args.cpg_owner_lock,
        generation=args.cpg_generation,
    )
    owner, live = observe(args.cpg_owner_lock, args.cpg_task, args.mcp_url)
    if paired["cpg_owner"] != owner:
        raise RuntimeError("live and paired-lock CPG owner receipts differ")
    return {"validated": paired, "live": live}


def _ownership_identity(value) -> dict:
    owner = value["validated"]["cpg_owner"]
    paired = value["validated"]["paired_gpu_owners"]
    return {
        "agent_id": owner["agent_id"],
        "cpg_session": owner["cpg_session"],
        "cpg_task": owner["cpg_task"],
        "cpg_generation": owner["cpg_generation"],
        "cpg_pid": owner["pid"],
        "paired_gpu_owners": paired,
    }


def _publish_private(path: Path, value) -> None:
    """Publish once with a private exclusive temp and an atomic no-clobber link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_unqualified(route) -> None:
    if route["qualified"] or route["qualification"] != "unqualified":
        raise AssertionError("diagnostic must not promote D3 qualification")


def _alarm_timeout(_signum, _frame):
    raise TimeoutError("bounded D3 diagnostic deadline reached")


def _track_execution(engine, report) -> None:
    """Mark observed use at the adapter's exact pre-forward boundary."""
    begin = engine._begin_execution

    def tracked():
        report["gpu_executed"] = True
        report["observed_used"] = True
        return begin()

    engine._begin_execution = tracked


def _questions():
    return {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "returns": "Damaged deliveries",
                "billing": "Payment problems",
                "shipping": "Late or missing deliveries",
            },
        },
        "receipt": {
            "type": "noul",
            "instructions": "Does the customer have a receipt?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["Routine", "Soon", "Today"],
        },
    }


def _request(engine, *, state, questions):
    from mlx2.decisions.schema import normalize_request

    return normalize_request(
        {"model": engine.model_name, "state": state, "questions": questions},
        default_model=engine.model_name,
        reserved_tokens=engine.reserved_tokens,
    )


def _answer_digest(response) -> str:
    return _digest(response["answers"])


def _timed_predict(engine, request, *, label):
    started = time.monotonic()
    response = engine.predict(request)
    return response, {
        "label": label,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "input_tokens": response["usage"]["input_tokens"],
        "answer_sha256": _answer_digest(response),
        "observed_used": response["mlx2"]["observed_used"],
        "qualified": response["mlx2"]["qualified"],
    }


def _state_near_tokens(engine, *, question, target: int) -> tuple[dict[str, str], int]:
    from mlx2.decisions.candidates.d3 import prepare_question, render_prompt

    prepared, _ = prepare_question(question)

    def count(repetitions: int) -> int:
        state = {"record": "parcel damaged receipt present " * repetitions}
        return len(
            render_prompt(
                engine.tokenizer,
                state,
                prepared,
                engine.codes,
                max_length=engine.artifact["max_context"],
            )
        )

    low, high = 0, 1
    while count(high) <= target:
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if count(middle) <= target:
            low = middle
        else:
            high = middle
    state = {"record": "parcel damaged receipt present " * low}
    return state, count(low)


def _run_once(engine, *, context_tokens: int):
    from mlx2.decisions.schema import DecisionInputTooLong

    state = {
        "event": "The parcel arrived damaged.",
        "customer": {"receipt": True, "requested": "replacement"},
    }
    questions = _questions()
    request = _request(engine, state=state, questions=questions)
    first, first_timing = _timed_predict(engine, request, label="short-first")
    second, second_timing = _timed_predict(engine, request, label="short-repeat")
    if first["answers"] != second["answers"]:
        raise AssertionError("repeated short D3 answers changed in one loaded engine")
    isolated, isolated_timing = _timed_predict(
        engine,
        _request(engine, state=state, questions={"receipt": questions["receipt"]}),
        label="isolated-question",
    )
    if isolated["answers"]["receipt"] != first["answers"]["receipt"]:
        raise AssertionError(
            "D3 independent question changed inside a multi-question request"
        )

    before = engine.status()["counters"]
    overlong = _request(
        engine,
        state={"record": "too long " * 40000},
        questions={"receipt": questions["receipt"]},
    )
    try:
        engine.predict(overlong)
    except DecisionInputTooLong as error:
        refusal = {"type": type(error).__name__, "message": str(error)}
    else:
        raise AssertionError("D3 accepted an over-context request")
    after = engine.status()["counters"]
    if (
        after["executions"] != before["executions"]
        or after["refusals"] != before["refusals"] + 1
    ):
        raise AssertionError("pre-execution D3 refusal counters are inconsistent")
    recovered, recovery_timing = _timed_predict(
        engine, request, label="short-after-refusal"
    )
    if recovered["answers"] != first["answers"]:
        raise AssertionError("D3 answer changed after an over-context refusal")

    long_state, actual_tokens = _state_near_tokens(
        engine, question=questions["receipt"], target=context_tokens
    )
    long_response, long_timing = _timed_predict(
        engine,
        _request(engine, state=long_state, questions={"receipt": questions["receipt"]}),
        label="bounded-context",
    )
    if long_response["usage"]["input_tokens"] != actual_tokens:
        raise AssertionError(
            "measured D3 context length changed between planning and execution"
        )
    return {
        "answers": first["answers"],
        "answer_sha256": _answer_digest(first),
        "checks": {
            "same_loaded_engine_repeat_exact": True,
            "isolated_question_matches_multi_question": True,
            "over_context_refused_before_execution": True,
            "post_refusal_recovery_exact": True,
            "bounded_context_executed": True,
        },
        "refusal": refusal,
        "timings": [
            first_timing,
            second_timing,
            isolated_timing,
            recovery_timing,
            long_timing,
        ],
        "status": engine.status(),
    }


def _execute(args, report) -> None:
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("D3 artifact diagnostic requires an owned MLX GPU window")
    mx.set_default_device(mx.gpu)
    mx.reset_peak_memory()
    from mlx2.decisions.candidates.d3 import SOURCE_REVISION, D3Engine

    engine = D3Engine(args.artifact)
    try:
        _track_execution(engine, report)
        first = _run_once(engine, context_tokens=args.context_tokens)
        route = engine.route_receipt(observed_used=True)
        _assert_unqualified(route)
    finally:
        engine.close()
    mx.clear_cache()

    engine = D3Engine(args.artifact)
    try:
        _track_execution(engine, report)
        repeated, reload_timing = _timed_predict(
            engine,
            _request(
                engine,
                state={
                    "event": "The parcel arrived damaged.",
                    "customer": {"receipt": True, "requested": "replacement"},
                },
                questions=_questions(),
            ),
            label="fresh-engine-reload",
        )
        if _answer_digest(repeated) != first["answer_sha256"]:
            raise AssertionError("D3 answer changed across close and fresh reload")
    finally:
        engine.close()
    report.update(
        source_revision=SOURCE_REVISION,
        device=mx.device_info(),
        route=route,
        first_engine=first,
        reload={
            "answer_sha256": _answer_digest(repeated),
            "matches_first_engine": True,
            "timing": reload_timing,
        },
        peak_mlx_bytes=int(mx.get_peak_memory()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--cpg-task", required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--cpg-generation", type=int, required=True)
    parser.add_argument("--mcp-url", default="http://127.0.0.1:8766/mcp")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output path must be fresh")
    if not 512 <= args.context_tokens <= 8192:
        parser.error("context-tokens must be between 512 and 8192")
    if not 1 <= args.timeout_seconds <= 900:
        parser.error("timeout-seconds must be between 1 and 900")
    if args.cpg_generation < 1:
        parser.error("cpg-generation must be positive")

    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    report = {
        "schema": "mlx2.d3-artifact-diagnostic.v2",
        "artifact": str(args.artifact.resolve()),
        "gpu_executed": False,
        "diagnostic_only": True,
        "implemented": True,
        "qualified": False,
        "selected": True,
        "observed_used": False,
        "formal_qualification": False,
        "default_promoted": False,
        "context_target_tokens": args.context_tokens,
        "passed": False,
        "limits": [
            "One Apple-Silicon diagnostic run; no controlled performance claim.",
            "No full preflight, formal qualification, quality evaluation, batching, "
            "APCv2, or serving concurrency evidence.",
            "The bounded context cell is deterministic stability evidence, not "
            "maximum-context validation.",
        ],
    }
    failure = None
    ownership_before = source_before = artifact_before = environment_before = None
    precondition_errors = []

    def initial(name, function):
        nonlocal failure
        try:
            value = function()
            report[name] = value
            return value
        except BaseException as error:  # noqa: BLE001 - publish preflight evidence
            message = f"{name}: {type(error).__name__}: {error}"
            precondition_errors.append(message)
            if failure is None:
                failure = error
            return None

    ownership_before = initial("ownership_before", lambda: _ownership(args))
    source_before = initial("source_sha256", lambda: _source_hashes(root))
    artifact_before = initial(
        "artifact_files", lambda: _verify_artifact_files(args.artifact)
    )
    environment_before = initial("environment_before", lambda: _environment(root))
    if environment_before is not None:
        if environment_before["thermal"]["thermal_state"] != 0:
            error = RuntimeError("D3 diagnostic must start at nominal thermal state")
            precondition_errors.append(f"environment_before: {error}")
            failure = failure or error
        if environment_before["contamination"]["contaminated"]:
            error = RuntimeError("GPU contamination present before D3 diagnostic")
            precondition_errors.append(f"environment_before: {error}")
            failure = failure or error
    if precondition_errors:
        report["precondition_errors"] = precondition_errors
        report["error"] = precondition_errors[0].split(": ", 1)[1]

    thermal_samples = []
    thermal_errors = []
    if environment_before is not None:
        thermal_samples.append(
            {
                "phase": "pre",
                "monotonic_seconds": time.monotonic(),
                **environment_before["thermal"],
            }
        )
    if failure is None:
        stop_thermal = threading.Event()
        watcher = threading.Thread(
            target=_watch_thermal,
            args=(root, stop_thermal, thermal_samples, thermal_errors),
            name="d3-thermal-watcher",
            daemon=True,
        )
        watcher.start()
        previous_alarm = signal.signal(signal.SIGALRM, _alarm_timeout)
        signal.alarm(args.timeout_seconds)
        try:
            _execute(args, report)
        except BaseException as error:  # noqa: BLE001 - publish bounded failure evidence
            failure = error
            report["error"] = f"{type(error).__name__}: {error}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_alarm)
            stop_thermal.set()
            watcher.join(timeout=35)
            if watcher.is_alive():
                thermal_errors.append("thermal watcher did not stop within 35 seconds")
    report["thermal_monitor"] = {
        "interval_seconds": 2.0,
        "samples": thermal_samples,
        "errors": thermal_errors,
    }
    if thermal_errors or any(
        sample.get("thermal_state") != 0 for sample in thermal_samples
    ):
        error = RuntimeError(
            "thermal watcher failed or observed a non-nominal excursion"
        )
        if failure is None:
            failure = error
            report["error"] = f"{type(error).__name__}: {error}"

    postcondition_errors = []

    def post(name, function):
        nonlocal failure
        try:
            value = function()
            report[name] = value
            return value
        except BaseException as error:  # noqa: BLE001 - collect every postcondition
            message = f"{name}: {type(error).__name__}: {error}"
            postcondition_errors.append(message)
            if failure is None:
                failure = error
                report["error"] = f"{type(error).__name__}: {error}"
            return None

    source_after = post("source_sha256_end", lambda: _source_hashes(root))
    artifact_after = post(
        "artifact_files_end", lambda: _verify_artifact_files(args.artifact)
    )
    environment_after = post("environment_after", lambda: _environment(root))
    # Ownership is deliberately last so earlier postcondition failures cannot
    # suppress validation of the live CPG lease and paired lock receipts.
    ownership_after = post("ownership_after", lambda: _ownership(args))

    def validate(name, condition, message):
        nonlocal failure
        if condition is None:
            return
        report[name] = condition
        if not condition:
            error = RuntimeError(message)
            postcondition_errors.append(f"{name}: {type(error).__name__}: {error}")
            if failure is None:
                failure = error
                report["error"] = f"{type(error).__name__}: {error}"

    validate(
        "source_unchanged",
        None
        if source_before is None or source_after is None
        else source_after == source_before,
        "D3 diagnostic source changed during execution",
    )
    validate(
        "artifact_identity_stable_through_run",
        None
        if artifact_before is None or artifact_after is None
        else artifact_after == artifact_before,
        "D3 artifact changed during execution",
    )
    validate(
        "ownership_unchanged",
        None
        if ownership_before is None or ownership_after is None
        else _ownership_identity(ownership_after)
        == _ownership_identity(ownership_before),
        "D3 ownership changed during execution",
    )
    if environment_after is not None:
        thermal_samples.append(
            {
                "phase": "post",
                "monotonic_seconds": time.monotonic(),
                **environment_after["thermal"],
            }
        )
        if environment_after["thermal"]["thermal_state"] != 0:
            error = RuntimeError("D3 diagnostic ended at non-nominal thermal state")
            postcondition_errors.append(
                f"environment_after: {type(error).__name__}: {error}"
            )
            if failure is None:
                failure = error
                report["error"] = f"{type(error).__name__}: {error}"
        if environment_after["contamination"]["contaminated"]:
            error = RuntimeError("GPU contamination present after D3 diagnostic")
            postcondition_errors.append(
                f"environment_after: {type(error).__name__}: {error}"
            )
            if failure is None:
                failure = error
                report["error"] = f"{type(error).__name__}: {error}"
    if postcondition_errors:
        report["postcondition_errors"] = postcondition_errors
    report["elapsed_seconds"] = round(time.monotonic() - started, 6)
    report["passed"] = failure is None
    _publish_private(args.output, report)
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
