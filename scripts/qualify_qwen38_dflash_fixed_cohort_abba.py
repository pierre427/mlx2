"""Deterministic fixed-cohort 20x20 DFlash qualification campaigns.

This is a performance qualification driver, not a selector.  It binds the
mlx2, model, TensorFold and policy identities before starting Metal work.  Run
it only as the child of ``scripts/run_with_gpu_locks.py``: both owner receipts
must exist and agree.

The original varlen gate and its controlled reverse use A/B-B/A and B/A-A/B
orders respectively.  They require complete phase-local B4 timing for draft,
search, target, transaction and fence work.  The separate PLD-to-DFlash gate
uses chain verification, a half-greedy/half-seeded-sampling workload, and
request-local receipts proving accepted committed verification from both PLD
and the external DFlash law.  Passing any campaign records evidence; it does
not select a default or by itself qualify the whole DFlash route.

Each round contains exactly five client-declared four-request cohorts.  All
twenty worker threads share an immutable, precomputed open-loop admission
schedule, so the four arms receive byte-identical request bodies at identical
nominal offsets.  There are no HTTP retries.  A 429, cohort timeout, receipt
drift, greedy output drift, same-policy sampled drift, missing mechanism
receipt, or incomplete cohort fails the campaign.  A sampled output may differ
across proposal policies because exact residual verification preserves the
target law, not the coupling of one seed through two different proposal laws.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_ARTIFACT = "adf65acbac3e8e7e42b14d87893f6dec3f68ca38c2d7946d1b96155f9a5cab2e"
EXPECTED_MODEL_CONFIG = (
    "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85"
)
EXPECTED_TENSORFOLD_SOURCE = "1a5f38e12afbb560d8fc61c88ccb5900f7d5d170"
ARM_ORDERS = {
    "abba": ("control", "candidate", "candidate", "control"),
    "baab": ("candidate", "control", "control", "candidate"),
}
ORDER = ARM_ORDERS["abba"]
PLD_COMPOSITION = {
    "prompt_lookup": True,
    "ngram_min": 3,
    "ngram_max": 6,
    "lookback": 4096,
    "native_mtp": False,
    "mtp_max_history": 4096,
}
GPU_OWNERS = (
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    Path("/tmp/gpu.lock/owner.json"),
)
COHORT_SIZE = 4
COHORTS_PER_ROUND = 5
WIDTH = COHORT_SIZE * COHORTS_PER_ROUND
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
CITY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "city",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "population": {"type": "integer"},
                "coastal": {"type": "boolean"},
            },
            "required": ["city", "population", "coastal"],
            "additionalProperties": False,
        },
    },
}
CAPITALS = (
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
    ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("Spain", "Madrid"),
    ("Germany", "Berlin"), ("Kenya", "Nairobi"), ("Peru", "Lima"),
    ("Norway", "Oslo"), ("Greece", "Athens"), ("Portugal", "Lisbon"),
    ("Austria", "Vienna"), ("Ireland", "Dublin"), ("Cuba", "Havana"),
    ("Poland", "Warsaw"), ("Sweden", "Stockholm"),
    ("Finland", "Helsinki"), ("Hungary", "Budapest"),
    ("Chile", "Santiago"),
)
WORDS = (("cat", "chat"), ("dog", "chien"), ("house", "maison"),
         ("water", "eau"), ("book", "livre"))
FAILURE_COUNTERS = (
    "batch_cohort_timeouts",
    "batch_cohort_jobs_timed_out",
    "batch_cohort_jobs_failed_closed",
    "batch_cohort_attachment_failures",
    "batch_cohort_scheduler_failures",
    "batch_cohort_staged_cancellations",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=path, text=True).strip()


def require_gpu_owners() -> dict:
    owners = []
    for path in GPU_OWNERS:
        if not path.is_file():
            raise RuntimeError(
                f"missing GPU owner receipt {path}; use run_with_gpu_locks.py"
            )
        owner = json.loads(path.read_text())
        if not isinstance(owner, dict) or not owner.get("lease_id"):
            raise RuntimeError(f"invalid GPU owner receipt: {path}")
        owners.append(owner)
    if owners[0] != owners[1]:
        raise RuntimeError("GPU owner receipts disagree")
    return owners[0]


def runtime_source_sha256() -> str:
    root = ROOT / "src/mlx2"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_identity(expected_source: str) -> dict:
    revision = git(ROOT, "rev-parse", "HEAD")
    if revision != expected_source:
        raise RuntimeError(
            f"mlx2 source {revision} != expected {expected_source}"
        )
    production_diff = git(
        ROOT, "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts"
    )
    if production_diff:
        raise RuntimeError(f"mlx2 src/scripts differ from HEAD:\n{production_diff}")
    return {
        "root": str(ROOT),
        "revision": revision,
        "production_diff": production_diff,
        "runtime_source_sha256": runtime_source_sha256(),
    }


def tensorfold_identity(path: Path) -> dict:
    from mlx2.adapters.qwen38_tensorfold_source import qualification_source_identity

    identity = qualification_source_identity(path)
    if identity["revision"] != EXPECTED_TENSORFOLD_SOURCE:
        raise RuntimeError("TensorFold helper and campaign revision pins disagree")
    return identity


def model_identity(path: Path) -> dict:
    path = path.resolve()
    config = path / "config.json"
    if not config.is_file():
        raise RuntimeError(f"model config missing: {config}")
    config_sha256 = sha256(config)
    if config_sha256 != EXPECTED_MODEL_CONFIG:
        raise RuntimeError(
            f"model config {config_sha256} != pinned {EXPECTED_MODEL_CONFIG}"
        )
    from mlx2.adapters.qwen38_27b import inspect_artifact

    artifact = inspect_artifact(path)["identity"]
    if artifact["fingerprint"] != EXPECTED_MODEL_ARTIFACT:
        raise RuntimeError(
            f"model artifact {artifact['fingerprint']} != pinned "
            f"{EXPECTED_MODEL_ARTIFACT}"
        )
    return {
        "root": str(path),
        "config_sha256": config_sha256,
        "artifact_fingerprint": artifact["fingerprint"],
        "files": artifact["files"],
    }


def validate_policy_manifest(
    path: Path,
    *,
    expected_source: str,
    model: dict,
    policies: dict[str, Path],
    role_prefix: str,
    tensorfold: dict | None = None,
) -> dict:
    """Bind qualification policies to the metadata-only inspection manifest.

    The served adapter fingerprint is intentionally not predicted here.  It is
    re-derived after model construction from the target/draft binding and the
    full serving numerical namespace, including hardware-selected lane laws.
    The host manifest instead pins every input to that derivation.
    """

    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != (
        "mlx2.qwen38-dflash-closeout-policy-derivation.v1"
    ):
        raise RuntimeError("invalid DFlash policy derivation manifest")
    source = value.get("source") or {}
    if source.get("commit") != expected_source or Path(
        source.get("root", "")
    ).resolve() != ROOT.resolve():
        raise RuntimeError("policy manifest source binding differs from this checkout")
    expected_modules = {
        "mlx2.adapters.dflash2": ROOT / "src/mlx2/adapters/dflash2.py",
        "mlx2.adapters.qwen38_27b": ROOT / "src/mlx2/adapters/qwen38_27b.py",
        "mlx2.adapters.qwen38_tensorfold_source": (
            ROOT / "src/mlx2/adapters/qwen38_tensorfold_source.py"
        ),
    }
    modules = source.get("modules") or {}
    if set(modules) != set(expected_modules):
        raise RuntimeError("policy manifest inspector module set differs")
    for name, expected_path in expected_modules.items():
        record = modules.get(name) or {}
        if Path(record.get("path", "")).resolve() != expected_path.resolve():
            raise RuntimeError(f"policy manifest {name} path differs")
        if record.get("sha256") != sha256(expected_path):
            raise RuntimeError(f"policy manifest {name} digest differs")
    inspection = value.get("real_adapter_inspection") or {}
    if inspection.get("mlx_import_blocked") is not True or inspection.get(
        "mlx_modules_after"
    ) != []:
        raise RuntimeError("policy manifest did not preserve host-only inspection")
    target = value.get("target") or {}
    if target.get("artifact_fingerprint") != model["artifact_fingerprint"]:
        raise RuntimeError("policy manifest target artifact differs from campaign")
    if not is_sha256(target.get("payload_revision")):
        raise RuntimeError("policy manifest target payload revision is missing")
    draft = value.get("draft") or {}
    if not is_sha256(draft.get("payload_revision")):
        raise RuntimeError("policy manifest draft payload revision is missing")
    tensorfold_binding = None
    if role_prefix == "abba":
        if tensorfold is None or value.get("tensorfold") != tensorfold:
            raise RuntimeError(
                "policy manifest TensorFold source binding differs from campaign"
            )
        tensorfold_binding = tensorfold
    elif role_prefix != "pld":
        raise ValueError(f"unsupported policy manifest role prefix: {role_prefix}")

    records = {
        record.get("role"): record
        for record in (value.get("policies") or [])
        if isinstance(record, dict)
    }
    bound = {}
    for arm, policy_path in policies.items():
        role = f"{role_prefix}-{arm}"
        record = records.get(role)
        if not isinstance(record, dict):
            raise TypeError(f"policy manifest is missing {role}")
        resolved = policy_path.resolve()
        if Path(record.get("generated_path", "")).resolve() != resolved:
            raise RuntimeError(f"policy manifest {role} path differs from campaign")
        digest = sha256(resolved)
        if record.get("generated_sha256") != digest:
            raise RuntimeError(f"policy manifest {role} digest differs from campaign")
        policy = read_policy(resolved)
        adapter = record.get("adapter_inspection") or {}
        if policy.get("target_revision") != target.get("payload_revision"):
            raise RuntimeError(f"policy manifest {role} target payload pin differs")
        if policy.get("draft_revision") != draft.get("payload_revision"):
            raise RuntimeError(f"policy manifest {role} draft payload pin differs")
        if adapter.get("target_revision") != policy.get("target_revision"):
            raise RuntimeError(f"policy manifest {role} adapter target differs")
        if adapter.get("draft_revision") != policy.get("draft_revision"):
            raise RuntimeError(f"policy manifest {role} adapter draft differs")
        if not is_sha256(adapter.get("runtime_draft_fingerprint")):
            raise RuntimeError(f"policy manifest {role} runtime draft binding is missing")
        bound[arm] = {
            "role": role,
            "path": str(resolved),
            "sha256": digest,
            "target_revision": adapter["target_revision"],
            "draft_revision": adapter["draft_revision"],
            "runtime_draft_fingerprint": adapter["runtime_draft_fingerprint"],
        }
    if (
        bound["control"]["runtime_draft_fingerprint"]
        != bound["candidate"]["runtime_draft_fingerprint"]
    ):
        raise RuntimeError("qualification arms bind different runtime DFlash payloads")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "source": source.get("commit"),
        "target_artifact_fingerprint": target["artifact_fingerprint"],
        "target_payload_revision": target.get("payload_revision"),
        "draft_payload_revision": draft["payload_revision"],
        "tensorfold": tensorfold_binding,
        "policies": bound,
    }


def read_policy(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"policy must be a JSON object: {path}")
    return value


def validate_policies(
    control: dict,
    candidate: dict,
    *,
    qualification_kind: str = "varlen",
) -> None:
    left = dict(control)
    right = dict(candidate)
    if qualification_kind == "varlen":
        control_varlen = left.pop("varlen_dense_mlp", None)
        candidate_varlen = right.pop("varlen_dense_mlp", None)
    elif qualification_kind == "pld-dflash":
        control_composition = left.pop("proposal_composition", None)
        candidate_composition = right.pop("proposal_composition", None)
        if control_composition is not None:
            raise RuntimeError("PLD/DFlash control must not select proposal composition")
        if candidate_composition != PLD_COMPOSITION:
            raise RuntimeError(
                "PLD/DFlash candidate does not select the pinned prompt-lookup policy"
            )
        control_varlen = left.get("varlen_dense_mlp")
        candidate_varlen = right.get("varlen_dense_mlp")
    else:
        raise ValueError(f"unknown qualification kind: {qualification_kind}")
    if left != right:
        raise RuntimeError(
            f"control/candidate policies differ outside {qualification_kind}"
        )
    if qualification_kind == "varlen":
        if control_varlen != {
            "enabled": True,
            "minimum_padding_fraction": 1.0,
            "minimum_padding_rows": 1,
        }:
            raise RuntimeError("control is not the pinned 100%-padding crossover")
        if candidate_varlen != {
            "enabled": True,
            "minimum_padding_fraction": 0.25,
            "minimum_padding_rows": 1,
        }:
            raise RuntimeError("candidate is not the corrected p25 dense-MLP varlen")
    elif control_varlen != candidate_varlen:
        raise RuntimeError("PLD/DFlash arms must use the same varlen policy")
    required = {
        "external_varlen_prefill": True,
        "pairwise_selection": "batched",
        "num_draft": 7,
    }
    if qualification_kind == "varlen":
        required["batch_size_route"] = "tree15_b1_b4_chain_b5plus_v1"
        required["tree_node_budget_by_lanes"] = {
            "1": 15,
            "2": 7,
            "3": 4,
            "4": 3,
        }
    elif left.get("batch_size_route") is not None:
        raise RuntimeError("PLD/DFlash composition requires the chain route")
    for key, expected in required.items():
        if left.get(key) != expected:
            raise RuntimeError(
                f"shared policy {key}={left.get(key)!r}, expected {expected!r}"
            )


def body(prompt: str, *, max_tokens: int = 96, **extra: object) -> dict:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "reasoning_effort": "none",
        "skip_writing_prefix_cache": True,
        **extra,
    }


def workload(round_index: int) -> list[tuple[str, dict]]:
    """Twenty deterministic mixed-domain requests derived from sanity_20x20."""
    a, b = 17 + round_index * 3, 25 + round_index * 7
    country, capital = CAPITALS[round_index % len(CAPITALS)]
    country2, capital2 = CAPITALS[(round_index + 7) % len(CAPITALS)]
    english, _french = WORDS[round_index % len(WORDS)]
    code = f"ORCHID-{5000 + round_index * 37}"
    ledger = "".join(
        f"Ledger line {i}: account {(i * 13 + round_index) % 500} moved "
        f"{(i * 17 + round_index) % 900} credits.\n"
        for i in range(90)
    )
    needle = (
        f"Run {round_index}.\n{ledger[:len(ledger) // 2]}"
        f"Important: the launch password is {code}.\n"
        f"{ledger[len(ledger) // 2:]}What is the launch password? "
        "Reply with the password only."
    )
    start = 3 + round_index
    return [
        ("arithmetic", body(f"What is {a} + {b}? Reply with the number only.")),
        ("arithmetic_mul", body(
            f"What is {round_index + 3} times 12? Reply with the number only."
        )),
        ("capital", body(f"What is the capital of {country}? One word.")),
        ("capital_sentence", body(
            f"Write four sentences about the capital city of {country2}, naming the city.",
            max_tokens=160,
        )),
        ("sequence", body(
            "Continue the sequence with the next number only: "
            f"{start}, {start + 4}, {start + 8}, {start + 12},"
        )),
        ("translate", body(
            f"Translate the English word '{english}' to French, then use the French "
            "word in two short French sentences.", max_tokens=120,
        )),
        ("code", body(
            f"Write a Python function named add_{round_index} that returns the sum "
            "of its two arguments. Include a docstring and two example calls in comments.",
            max_tokens=200,
        )),
        ("list", body(
            "List three primary colors and describe each one in a sentence.",
            max_tokens=160,
        )),
        ("json_object", body(
            f'Return a JSON object with keys "country" and "capital" for {country}.',
            response_format={"type": "json_object"},
        )),
        ("json_schema", body(
            f"Describe the city {capital2} as JSON.", response_format=CITY_SCHEMA,
        )),
        ("stop_string", body(
            "Count from 1 to 20, one number per line.", stop=["10"]
        )),
        ("subtraction", body(
            f"What is {a} minus {round_index}? Reply with the number only."
        )),
        ("tool_call", body(
            f"What is the weather in {capital} right now? Use the tool.",
            tools=TOOLS, max_tokens=128,
        )),
        ("system_multi_turn", {
            **body("placeholder", max_tokens=48),
            "messages": [
                {"role": "system", "content": "You are terse. Use as few words as possible."},
                {"role": "user", "content": f"Remember the number {a}."},
                {"role": "assistant", "content": "Noted."},
                {"role": "user", "content": "What number did I ask you to remember?"},
            ],
        }),
        ("needle", body(needle, max_tokens=32)),
        ("lighthouse", body(
            f"Write six sentences about a lighthouse keeper named number {round_index}.",
            max_tokens=200,
        )),
        ("number_facts", body(
            f"Give three facts about the number {a}, one sentence each.", max_tokens=160,
        )),
        ("logprobs", body(
            f"Name one fruit that is {('red', 'yellow', 'green', 'orange')[round_index % 4]}. One word.",
            logprobs=True, top_logprobs=2, max_tokens=16,
        )),
        ("boat", body(
            f"Suggest one name for a boat, round {round_index}. Name only.", max_tokens=24,
        )),
        ("explain", body(
            f"Explain in five sentences what compiler optimization number {round_index + 1} "
            "of a typical -O2 pipeline might do.", max_tokens=200,
        )),
    ]


def build_manifest(
    rounds: int,
    group_spacing_ms: float,
    *,
    sampling_profile: str = "greedy",
) -> dict:
    if sampling_profile not in {"greedy", "mixed-seeded"}:
        raise ValueError(f"unknown sampling profile: {sampling_profile}")
    rows = []
    all_rounds = [("warmup", -1), *(('timed', index) for index in range(rounds))]
    for ordinal, (phase, round_index) in enumerate(all_rounds):
        task_round = max(round_index, 0)
        requests = workload(task_round)
        if len(requests) != WIDTH:
            raise AssertionError("workload width changed")
        round_rows = []
        for request_index, (task, request_body) in enumerate(requests):
            group = request_index // COHORT_SIZE
            cohort_id = f"fixed20-{phase}-r{round_index:02d}-g{group:02d}"
            request_body = copy.deepcopy(request_body)
            if sampling_profile == "mixed-seeded":
                request_body["seed"] = 610_000 + task_round * WIDTH + request_index
                # Half the corpus remains the ordinary greedy reference; the
                # other half exercises exact stochastic proposal-law parity.
                if request_index % 2:
                    request_body["temperature"] = 0.7
            request_body["batch_cohort"] = {"id": cohort_id, "size": COHORT_SIZE}
            row = {
                "phase": phase,
                "round": round_index,
                "round_ordinal": ordinal,
                "request": request_index,
                "group": group,
                "task": task,
                "cohort_id": cohort_id,
                "planned_offset_ns": int(group * group_spacing_ms * 1_000_000),
                "body": request_body,
                "body_sha256": digest_bytes(canonical(request_body)),
            }
            round_rows.append(row)
            rows.append(row)
        groups = {}
        for row in round_rows:
            groups.setdefault(row["cohort_id"], []).append(row)
        if len(groups) != COHORTS_PER_ROUND or any(
            len(group_rows) != COHORT_SIZE for group_rows in groups.values()
        ):
            raise AssertionError("manifest does not contain five B4 cohorts")
    public_rows = [
        {key: value for key, value in row.items() if key != "body"} for row in rows
    ]
    body_hashes = [row["body_sha256"] for row in rows]
    return {
        "schema": "mlx2.fixed-cohort-20x20-workload.v1",
        "rounds": rounds,
        "warmup_rounds": 1,
        "width": WIDTH,
        "cohort_size": COHORT_SIZE,
        "cohorts_per_round": COHORTS_PER_ROUND,
        "group_spacing_ms": group_spacing_ms,
        "sampling_profile": sampling_profile,
        "timed_seeded_non_greedy_requests": (
            rounds * (WIDTH // 2) if sampling_profile == "mixed-seeded" else 0
        ),
        "rows": rows,
        "public_rows": public_rows,
        "body_hashes_sha256": digest_bytes(canonical(body_hashes)),
    }


def get_json(url: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def wait_ready(base: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with rc={process.returncode}")
        try:
            status = get_json(base + "/v1/status")
            if status.get("healthy") and status.get("ready", True):
                return status
        except Exception as error:  # noqa: BLE001 - startup polling
            last_error = repr(error)
        time.sleep(2)
    raise TimeoutError(f"server did not become ready; last_error={last_error}")


def status_counts(status: dict) -> dict:
    return {key: int(value) for key, value in (status.get("counts") or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)}


def counter_delta(before: dict, after: dict) -> dict:
    left, right = status_counts(before), status_counts(after)
    return {
        key: right.get(key, 0) - left.get(key, 0)
        for key in sorted(set(left) | set(right))
        if right.get(key, 0) != left.get(key, 0)
    }


NON_MONOTONIC_SCHEDULER_GAUGES = frozenset(
    {
        "external_adaptive_round_depth",
        "external_adaptive_verify_width",
        "external_tree_node_budget_last",
        "reservation_bytes",
    }
)


def numeric_delta(
    before: dict,
    after: dict,
    *,
    non_monotonic_gauges: frozenset[str] = frozenset(),
) -> dict:
    """Subtract numeric snapshots while rejecting true counter rollback.

    A declared gauge may legitimately decrease between snapshots.  All other
    numeric fields are treated as counters and remain fail-closed.
    """

    left = {
        key: int(value)
        for key, value in before.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    right = {
        key: int(value)
        for key, value in after.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    delta = {
        key: right.get(key, 0) - left.get(key, 0)
        for key in sorted(set(left) | set(right))
        if right.get(key, 0) != left.get(key, 0)
    }
    negative = {
        key: value
        for key, value in delta.items()
        if value < 0 and key not in non_monotonic_gauges
    }
    if negative:
        raise RuntimeError(f"counter rollback in status snapshot: {negative}")
    return delta


PHASE_NAMES = (
    "tree_recovery_capture",
    "tree_draft_setup",
    "tree_draft_build",
    "tree_draft_wait",
    "tree_search",
    "tree_target_launch",
    "tree_target_wait",
    "tree_target_law",
    "transaction_commit",
    "emit",
    "tree_draft_prelaunch",
    "tree_round",
)


def phase_observation(before: dict, after: dict) -> dict:
    """Return one B4 round's phase-local scheduler-counter deltas."""

    delta = numeric_delta(
        before.get("scheduler") or {},
        after.get("scheduler") or {},
        non_monotonic_gauges=NON_MONOTONIC_SCHEDULER_GAUGES,
    )
    rounds = int(delta.get("external_phase_rounds", 0))
    phase_ns = {
        name: int(delta.get(f"external_phase_{name}_ns", 0))
        for name in PHASE_NAMES
    }
    missing = [name for name, value in phase_ns.items() if value <= 0]
    if rounds <= 0 or missing:
        raise RuntimeError(
            "B4 phase instrumentation did not engage: "
            f"rounds={rounds}, missing={missing}"
        )
    return {
        "physical_tree_rounds": rounds,
        "phase_ns": phase_ns,
        "phase_ns_per_tree_round": {
            name: value / rounds for name, value in phase_ns.items()
        },
    }


def combine_phase_observations(rows: list[dict]) -> dict:
    rounds = sum(int(row["physical_tree_rounds"]) for row in rows)
    phase_ns = {
        name: sum(int(row["phase_ns"][name]) for row in rows)
        for name in PHASE_NAMES
    }
    return {
        "physical_tree_rounds": rounds,
        "phase_ns": phase_ns,
        "phase_ns_per_tree_round": {
            name: value / rounds if rounds else None
            for name, value in phase_ns.items()
        },
    }


def output_identity(response: dict) -> dict:
    choices = []
    for choice in response.get("choices") or []:
        message = choice.get("message") or {}
        # Tool-call IDs are transport identifiers, not model output semantics.
        # Compare the emitted type/name/arguments and discard only those IDs.
        tool_calls = [
            {
                "type": call.get("type"),
                "function": call.get("function"),
            }
            for call in (message.get("tool_calls") or [])
        ]
        choices.append({
            "index": choice.get("index"),
            "finish_reason": choice.get("finish_reason"),
            "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content") or "",
            "tool_calls": tool_calls,
        })
    return {"choices": choices}


def proposal_composition_observation(rows: list[dict], *, required: bool) -> dict:
    """Prove PLD and external DFlash both reached committed verification.

    The receipt counters are lane-local and cumulative for one request.  A
    campaign therefore sums only the final receipt from each response; it does
    not infer mechanism use from policy selection or scheduler-wide counters.
    """

    totals = {
        source: {"verified_rounds": 0, "proposed_tokens": 0, "accepted_tokens": 0}
        for source in ("prompt_lookup", "external")
    }
    selected = 0
    observed_used = 0
    for row in rows:
        envelope = row.get("receipt") or {}
        if not isinstance(envelope, dict):
            raise TypeError("invalid mlx2 response receipt envelope")
        speculation = envelope.get("speculation")
        if speculation is None:
            continue
        if not isinstance(speculation, dict):
            raise TypeError("invalid mlx2 speculation receipt envelope")
        receipt = speculation.get("proposal_composition")
        if not receipt:
            continue
        if not isinstance(receipt, dict):
            raise TypeError("invalid proposal composition receipt")
        if receipt.get("selected") is not True:
            continue
        selected += 1
        if receipt.get("qualified") is not False:
            raise RuntimeError(
                "proposal composition receipt must retain qualified=false during the gate"
            )
        if receipt.get("observed_used") is True:
            observed_used += 1
        committed = receipt.get("committed_verification") or {}
        if not isinstance(committed, dict):
            raise TypeError("invalid proposal composition committed_verification")
        for source, total in totals.items():
            counters = committed.get(source) or {}
            for name in total:
                value = counters.get(name, 0)
                if type(value) is not int or value < 0:
                    raise RuntimeError(
                        f"invalid {source} proposal composition counter {name}={value!r}"
                    )
                total[name] += value
    engaged = {
        source: all(totals[source][name] > 0 for name in totals[source])
        for source in totals
    }
    if required:
        if selected != len(rows):
            raise RuntimeError(
                f"proposal composition receipt selected on {selected}/{len(rows)} responses"
            )
        if observed_used <= 0 or not all(engaged.values()):
            raise RuntimeError(
                "PLD-to-DFlash composition did not prove accepted committed verification "
                f"from both sources: observed={observed_used}, totals={totals}"
            )
    elif selected:
        raise RuntimeError("control unexpectedly selected proposal composition")
    return {
        "receipt_source": "mlx2.speculation.proposal_composition",
        "selected_responses": selected,
        "observed_used_responses": observed_used,
        "committed_verification": totals,
        "engaged": engaged,
    }


def post(base: str, row: dict, epoch_ns: int, timeout: float) -> dict:
    target_ns = epoch_ns + int(row["planned_offset_ns"])
    while True:
        remaining = target_ns - time.monotonic_ns()
        if remaining <= 0:
            break
        time.sleep(min(remaining / 1_000_000_000, 0.002))
    released_ns = time.monotonic_ns()
    payload = canonical(row["body"])
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Tenant-ID": "fixed-cohort-20x20",
        },
    )
    started_ns = time.monotonic_ns()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            code = response.status
            value = json.load(response)
    except urllib.error.HTTPError as error:
        code = error.code
        value = {"error": error.read().decode(errors="replace")[:1000]}
    except Exception as error:  # noqa: BLE001 - preserve per-request failure
        code = -1
        value = {"error": f"{type(error).__name__}: {error}"}
    finished_ns = time.monotonic_ns()
    receipt = value.get("mlx2") or {}
    output = output_identity(value)
    return {
        "phase": row["phase"],
        "round": row["round"],
        "request": row["request"],
        "group": row["group"],
        "task": row["task"],
        "cohort_id": row["cohort_id"],
        "body_sha256": row["body_sha256"],
        "planned_offset_ns": row["planned_offset_ns"],
        "actual_release_offset_ns": released_ns - epoch_ns,
        "release_drift_ns": released_ns - target_ns,
        "wall_seconds": (finished_ns - started_ns) / 1_000_000_000,
        "http_status": code,
        "usage": value.get("usage") or {},
        "receipt": receipt,
        "receipt_cohort": (receipt.get("request_controls") or {}).get(
            "batch_cohort"
        ),
        "output": output,
        "output_sha256": digest_bytes(canonical(output)),
        "error": value.get("error"),
    }


def validate_server_identity(
    status: dict,
    model: Path,
    policy: dict,
    arm: str,
    *,
    expected_runtime_source: str,
    expected_artifact: str | None,
    qualification_kind: str,
) -> dict:
    runtime = status.get("runtime") or {}
    settings = status.get("settings") or {}
    execution = settings.get("execution_policy") or {}
    if status.get("model") != model.name:
        raise RuntimeError(
            f"{arm}: served model {status.get('model')!r} != {model.name!r}"
        )
    served_artifact = status.get("artifact")
    if not is_sha256(served_artifact):
        raise RuntimeError(f"{arm}: served model artifact is not a SHA-256 identity")
    if expected_artifact is not None and served_artifact != expected_artifact:
        raise RuntimeError(
            f"{arm}: served model artifact {served_artifact!r} "
            f"!= pinned {expected_artifact!r}"
        )
    if runtime.get("source_sha256") != expected_runtime_source:
        raise RuntimeError(f"{arm}: runtime source hash differs from pin")
    if settings.get("route") != "external_draft" or settings.get("speculation") != "external_draft":
        raise RuntimeError(f"{arm}: external-draft route not selected")
    expected_varlen = policy["varlen_dense_mlp"]
    selected_varlen = (settings.get("prefill_execution") or {}).get("varlen", {}).get("policy")
    if expected_varlen is True:
        expected_selected = {
            "enabled": True,
            "minimum_padding_fraction": 0.0,
            "minimum_padding_rows": 1,
        }
    else:
        expected_selected = expected_varlen
    if selected_varlen != expected_selected:
        raise RuntimeError(
            f"{arm}: selected varlen policy {selected_varlen!r} != {expected_selected!r}"
        )
    if execution.get("num_draft") != 7 or execution.get("pairwise_selection") != "batched":
        raise RuntimeError(f"{arm}: DFlash K7/batched policy not selected")
    if (execution.get("external_varlen_prefill") or {}).get("enabled") is not True:
        raise RuntimeError(f"{arm}: external varlen prefill not selected")
    return {
        "model": status.get("model"),
        "profile": status.get("profile"),
        "artifact": served_artifact,
        "runtime": runtime,
        "route": settings.get("route"),
        "speculation": settings.get("speculation"),
        "execution_policy": execution,
        "prefill_execution": settings.get("prefill_execution"),
    }


def run_round(base: str, rows: list[dict], lead_ms: float, timeout: float) -> tuple[list[dict], float]:
    epoch_ns = time.monotonic_ns() + int(lead_ms * 1_000_000)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WIDTH) as pool:
        futures = [pool.submit(post, base, row, epoch_ns, timeout) for row in rows]
        results = [future.result() for future in futures]
    return results, time.perf_counter() - started


def validate_round(rows: list[dict]) -> None:
    if len(rows) != WIDTH:
        raise RuntimeError(f"round returned {len(rows)} rows, expected {WIDTH}")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["cohort_id"], []).append(row)
        if row["http_status"] != 200:
            raise RuntimeError(
                f"{row['cohort_id']} request {row['request']} HTTP {row['http_status']}: "
                f"{row['error']}"
            )
        expected = {"id": row["cohort_id"], "size": COHORT_SIZE}
        if row["receipt_cohort"] != expected:
            raise RuntimeError(
                f"{row['cohort_id']} receipt cohort {row['receipt_cohort']!r} != {expected!r}"
            )
    if len(groups) != COHORTS_PER_ROUND:
        raise RuntimeError(f"observed {len(groups)} cohorts, expected {COHORTS_PER_ROUND}")
    for cohort_id, group in groups.items():
        if len(group) != COHORT_SIZE or sorted(row["request"] for row in group) != list(
            range(group[0]["group"] * COHORT_SIZE, (group[0]["group"] + 1) * COHORT_SIZE)
        ):
            raise RuntimeError(f"cohort {cohort_id} did not preserve its four members")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def run_arm(
    args: argparse.Namespace,
    *,
    arm: str,
    occurrence: int,
    source: dict,
    policy_path: Path,
    policy: dict,
    manifest: dict,
) -> dict:
    port = args.base_port + occurrence - 1
    base = f"http://127.0.0.1:{port}"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    stem = f"{occurrence:02d}-{arm}"
    log_path = args.output_dir / f"{stem}.server.log"
    command = [
        str(args.python), "-m", "mlx2.server",
        "--model", str(args.model),
        "--host", "127.0.0.1", "--port", str(port),
        "--max-context", "8192", "--max-lanes", "4", "--max-inflight", "40",
        "--host-prompt-cache-entries", "0", "--host-prompt-cache-tokens", "0",
        "--qualification-mode", "--external-draft", "--execution-policy", str(policy_path),
    ]
    environment = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        MLX2_EXTERNAL_ROUND_TIMING="1",
    )
    if args.qualification_kind == "varlen":
        environment["MLX2_TENSORFOLD_SOURCE"] = str(args.tensorfold_source)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            ready = wait_ready(base, process, args.startup_timeout)
            identity = validate_server_identity(
                ready,
                args.model,
                policy,
                arm,
                expected_runtime_source=source["runtime_source_sha256"],
                expected_artifact=args.expected_arm_artifacts[arm],
                qualification_kind=args.qualification_kind,
            )
            before = get_json(base + "/v1/status")
            results = []
            round_stats = []
            for ordinal in range(args.rounds + 1):
                phase = "warmup" if ordinal == 0 else "timed"
                round_index = -1 if ordinal == 0 else ordinal - 1
                round_rows = [
                    row for row in manifest["rows"]
                    if row["phase"] == phase and row["round"] == round_index
                ]
                round_before = get_json(base + "/v1/status")
                observed, wall = run_round(
                    base, round_rows, args.admission_lead_ms, args.request_timeout
                )
                validate_round(observed)
                round_after = get_json(base + "/v1/status")
                phases = (
                    phase_observation(round_before, round_after)
                    if args.qualification_kind == "varlen"
                    else None
                )
                results.extend(observed)
                tokens = sum(
                    int(row["usage"].get("completion_tokens") or 0) for row in observed
                )
                round_stats.append({
                    "phase": phase,
                    "round": round_index,
                    "wall_seconds": wall,
                    "completion_tokens": tokens,
                    "end_to_end_tokens_per_second": tokens / wall if wall else None,
                    "max_release_drift_ms": max(
                        row["release_drift_ns"] for row in observed
                    ) / 1_000_000,
                    "b4_phase_observation": phases,
                })
                print(json.dumps({
                    "event": "round", "arm": arm, "occurrence": occurrence,
                    **round_stats[-1],
                }, sort_keys=True), flush=True)
            settle_deadline = time.monotonic() + 30
            while True:
                after = get_json(base + "/v1/status")
                leases = int(((after.get("apcv2") or {}).get("cow") or {}).get("active_leases") or 0)
                if int(after.get("inflight") or 0) == 0 and leases == 0:
                    break
                if time.monotonic() >= settle_deadline:
                    raise RuntimeError(f"{arm}: requests did not retire within 30 seconds")
                time.sleep(0.25)
            deltas = counter_delta(before, after)
            expected_releases = (args.rounds + 1) * COHORTS_PER_ROUND
            expected_jobs = expected_releases * COHORT_SIZE
            if deltas.get("batch_cohort_releases", 0) != expected_releases:
                raise RuntimeError(
                    f"{arm}: batch_cohort_releases={deltas.get('batch_cohort_releases', 0)}, "
                    f"expected {expected_releases}"
                )
            for name in ("batch_cohort_jobs_staged", "batch_cohort_jobs_released"):
                if deltas.get(name, 0) != expected_jobs:
                    raise RuntimeError(
                        f"{arm}: {name}={deltas.get(name, 0)}, expected {expected_jobs}"
                    )
            failures = {name: deltas.get(name, 0) for name in FAILURE_COUNTERS if deltas.get(name, 0)}
            if failures:
                raise RuntimeError(f"{arm}: cohort failure counters changed: {failures}")
            scheduler = after.get("scheduler") or {}
            if args.qualification_kind == "varlen":
                if int(scheduler.get("external_tensorfold_cohort_rounds", 0)) <= 0:
                    raise RuntimeError(f"{arm}: TensorFold cohort route did not engage")
                if int(scheduler.get("external_tensorfold_cohort_max_width", 0)) != COHORT_SIZE:
                    raise RuntimeError(f"{arm}: TensorFold did not reach physical B4")
            else:
                counts = status_counts(after)
                if int(counts.get("peak_observed_width", 0)) != COHORT_SIZE:
                    raise RuntimeError(f"{arm}: scheduler did not reach physical B4")
                for name in ("target_max_width", "draft_max_width"):
                    if int(scheduler.get(name, 0)) != COHORT_SIZE:
                        raise RuntimeError(f"{arm}: {name} did not reach physical B4")
            if int(scheduler.get("external_batched_prefill_max_cohort_width", 0)) != COHORT_SIZE:
                raise RuntimeError(f"{arm}: external varlen prefill did not reach B4")
            varlen = (after.get("execution") or {}).get("varlen_dense_mlp") or {}
            if (
                policy.get("varlen_dense_mlp") is True
                and varlen.get("observed_used") is not True
            ):
                raise RuntimeError(f"{arm}: dense-MLP varlen did not engage")
            timed = [row for row in results if row["phase"] == "timed"]
            timed_rounds = [row for row in round_stats if row["phase"] == "timed"]
            timed_phases = (
                combine_phase_observations(
                    [row["b4_phase_observation"] for row in timed_rounds]
                )
                if args.qualification_kind == "varlen"
                else None
            )
            return {
                "arm": arm,
                "occurrence": occurrence,
                "port": port,
                "server_command": command,
                "server_log": str(log_path),
                "server_identity": identity,
                "policy": str(policy_path),
                "policy_sha256": sha256(policy_path),
                "results": results,
                "rounds": round_stats,
                "timed_b4_phases": timed_phases,
                "counter_delta": deltas,
                "scheduler_delta": numeric_delta(
                    before.get("scheduler") or {},
                    after.get("scheduler") or {},
                    non_monotonic_gauges=NON_MONOTONIC_SCHEDULER_GAUGES,
                ),
                "scheduler": scheduler,
                "varlen_dense_mlp": varlen,
                "timed_completion_tokens": sum(
                    int(row["usage"].get("completion_tokens") or 0) for row in timed
                ),
                "timed_wall_seconds": sum(row["wall_seconds"] for row in timed_rounds),
                "median_round_end_to_end_tokens_per_second": statistics.median(
                    row["end_to_end_tokens_per_second"] for row in timed_rounds
                ),
            }
        finally:
            stop_server(process)


def write_arm_result(path: Path, result: dict) -> dict:
    """Persist complete request evidence and return its campaign index record."""

    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {
        key: value for key, value in result.items() if key != "results"
    } | {"report": str(path), "report_sha256": sha256(path)}


def validate_persisted_arm_mechanism(
    result: dict,
    path: Path,
    *,
    qualification_kind: str,
    arm: str,
) -> dict:
    """Validate request-local mechanism evidence after the raw arm is durable."""

    if not path.is_file() or json.loads(path.read_text()) != result:
        raise RuntimeError("arm evidence must be persisted before mechanism validation")
    timed = [row for row in result["results"] if row["phase"] == "timed"]
    try:
        composition = proposal_composition_observation(
            timed,
            required=(qualification_kind == "pld-dflash" and arm == "candidate"),
        )
    except BaseException as error:
        result["mechanism_validation"] = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        write_arm_result(path, result)
        raise
    result["proposal_composition"] = composition
    result["mechanism_validation"] = {"status": "passed"}
    write_arm_result(path, result)
    return result


def comparison_key(row: dict) -> tuple[str, int, int]:
    return row["phase"], int(row["round"]), int(row["request"])


def validate_cross_arm(
    arms: list[dict], manifest: dict, *, order: tuple[str, ...] = ORDER
) -> dict:
    manifest_rows = {
        (row["phase"], int(row["round"]), int(row["request"])): row
        for row in manifest["rows"]
    }
    expected_body_hashes = {
        key: row["body_sha256"] for key, row in manifest_rows.items()
    }
    observed_by_policy = {"control": [], "candidate": []}
    for arm in arms:
        if arm["arm"] not in observed_by_policy:
            raise RuntimeError(f"unknown campaign arm {arm['arm']!r}")
        observed = {comparison_key(row): row for row in arm["results"]}
        if set(observed) != set(expected_body_hashes):
            missing = sorted(set(expected_body_hashes) - set(observed))[:8]
            extra = sorted(set(observed) - set(expected_body_hashes))[:8]
            raise RuntimeError(
                f"arm {arm['occurrence']} workload keys differ: missing={missing}, extra={extra}"
            )
        for key, body_sha256 in expected_body_hashes.items():
            row = observed[key]
            if row["body_sha256"] != body_sha256:
                raise RuntimeError(f"arm {arm['occurrence']} body drift at {key}")
        observed_by_policy[arm["arm"]].append(observed)
    if any(len(values) != 2 for values in observed_by_policy.values()):
        raise RuntimeError("cross-arm validation requires two occurrences per policy")

    sampled_divergences = []
    sampled_requests = 0
    greedy_requests = 0
    for key, manifest_row in manifest_rows.items():
        body = manifest_row.get("body") or {}
        mixed_seeded = manifest.get("sampling_profile") == "mixed-seeded"
        temperature = float(
            body.get("temperature", 0.7 if mixed_seeded and key[2] % 2 else 0)
            or 0
        )
        outputs = {}
        for policy, occurrences in observed_by_policy.items():
            hashes = {occurrence[key]["output_sha256"] for occurrence in occurrences}
            if len(hashes) != 1:
                kind = "sampled" if temperature > 0 else "greedy"
                raise RuntimeError(
                    f"{kind} output drift within {policy} at {key}: {sorted(hashes)}"
                )
            outputs[policy] = next(iter(hashes))
        if temperature <= 0:
            greedy_requests += 1
            if outputs["control"] != outputs["candidate"]:
                raise RuntimeError(
                    f"greedy output drift at {key}: "
                    f"{outputs['candidate']} != {outputs['control']}"
                )
            continue
        sampled_requests += 1
        seed = body.get(
            "seed", 610_000 + max(key[1], 0) * WIDTH + key[2]
            if mixed_seeded else None,
        )
        if type(seed) is not int:
            raise RuntimeError(f"sampled request has no integer seed at {key}")
        if outputs["control"] != outputs["candidate"]:
            sampled_divergences.append({
                "phase": key[0],
                "round": key[1],
                "request": key[2],
                "seed": seed,
                "temperature": temperature,
                "control_output_sha256": outputs["control"],
                "candidate_output_sha256": outputs["candidate"],
            })
    controls = [arm for arm in arms if arm["arm"] == "control"]
    candidates = [arm for arm in arms if arm["arm"] == "candidate"]
    artifacts = {
        name: {
            arm["server_identity"]["artifact"]
            for arm in arms
            if arm["arm"] == name
        }
        for name in ("control", "candidate")
    }
    if any(len(values) != 1 for values in artifacts.values()):
        raise RuntimeError(f"served artifact identity drifted within an arm: {artifacts}")
    if artifacts["control"] == artifacts["candidate"]:
        raise RuntimeError(
            "control and candidate policies produced the same served artifact identity"
        )
    control_wall = sum(arm["timed_wall_seconds"] for arm in controls)
    candidate_wall = sum(arm["timed_wall_seconds"] for arm in candidates)
    return {
        "arms": len(arms),
        "order": list(order),
        "request_bodies_identical": True,
        "greedy_outputs_identical_across_policies": True,
        "greedy_requests": greedy_requests,
        "sampled_outputs_repeatable_within_policy": True,
        "sampled_requests": sampled_requests,
        "sampled_cross_policy_divergence_count": len(sampled_divergences),
        "sampled_cross_policy_divergences": sampled_divergences,
        "canonical_outputs_identical": not sampled_divergences,
        "cohort_receipts_exact": True,
        "served_artifacts": {
            name: next(iter(values)) for name, values in artifacts.items()
        },
        "served_artifacts_stable_within_arm": True,
        "served_artifacts_distinct_across_policies": True,
        "control_total_round_wall_seconds": control_wall,
        "candidate_total_round_wall_seconds": candidate_wall,
        "control_over_candidate_wall_ratio": (
            control_wall / candidate_wall if candidate_wall else None
        ),
        "candidate_wall_reduction_fraction": (
            1.0 - candidate_wall / control_wall if control_wall else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument(
        "--model", type=Path,
        default=Path.home() / "mlx-models" / "Qwen3.8-27B-MLX-4bit",
    )
    parser.add_argument(
        "--tensorfold-source", type=Path,
        default=(
            Path.home()
            / ".codex/worktrees/tensorfold-upstream-parity-20260928"
        ),
    )
    parser.add_argument("--control-policy", type=Path, required=True)
    parser.add_argument("--candidate-policy", type=Path, required=True)
    parser.add_argument(
        "--policy-manifest",
        type=Path,
        help=(
            "metadata-only derivation manifest; required for both gates and "
            "defaults to policy-derivation-manifest.json beside both policies"
        ),
    )
    parser.add_argument(
        "--qualification-kind",
        choices=("varlen", "pld-dflash"),
        default="varlen",
    )
    parser.add_argument(
        "--arm-order",
        choices=tuple(ARM_ORDERS),
        default="abba",
        help="abba is the original order; baab is the controlled reverse",
    )
    parser.add_argument(
        "--sampling-profile",
        choices=("greedy", "mixed-seeded"),
        default="greedy",
    )
    parser.add_argument(
        "--expected-control-artifact",
        default=None,
    )
    parser.add_argument(
        "--expected-candidate-artifact",
        default=None,
    )
    parser.add_argument(
        "--python", type=Path, default=Path(sys.executable),
        help="server interpreter; defaults to the interpreter running this driver",
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--group-spacing-ms", type=float, default=5.0)
    parser.add_argument("--admission-lead-ms", type=float, default=250.0)
    parser.add_argument("--base-port", type=int, default=18941)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.rounds != 20:
        raise SystemExit("this named gate requires exactly 20 timed rounds")
    if args.qualification_kind == "pld-dflash" and args.sampling_profile != "mixed-seeded":
        raise SystemExit(
            "the PLD-to-DFlash qualification requires mixed-seeded sampling"
        )
    if not args.python.is_file():
        raise SystemExit(f"Python executable missing: {args.python}")
    if args.output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {args.output_dir}")

    explicit_artifacts = {
        "control": args.expected_control_artifact,
        "candidate": args.expected_candidate_artifact,
    }
    expected_arm_artifacts = dict(explicit_artifacts)
    for name, value in expected_arm_artifacts.items():
        if value is not None and not is_sha256(value):
            raise SystemExit(f"invalid expected {name} artifact fingerprint")

    gpu_owner = require_gpu_owners()
    source = source_identity(args.expected_source)
    tensorfold = (
        tensorfold_identity(args.tensorfold_source)
        if args.qualification_kind == "varlen"
        else None
    )
    model = model_identity(args.model)
    source_policy_paths = {
        "control": args.control_policy.resolve(),
        "candidate": args.candidate_policy.resolve(),
    }
    control = read_policy(source_policy_paths["control"])
    candidate = read_policy(source_policy_paths["candidate"])
    validate_policies(
        control,
        candidate,
        qualification_kind=args.qualification_kind,
    )
    manifest = build_manifest(
        args.rounds,
        args.group_spacing_ms,
        sampling_profile=args.sampling_profile,
    )
    order = ARM_ORDERS[args.arm_order]
    args.expected_arm_artifacts = expected_arm_artifacts
    policy_binding = None
    policy_manifest = args.policy_manifest
    if policy_manifest is None:
        parents = {path.parent for path in source_policy_paths.values()}
        if len(parents) != 1:
            raise SystemExit(
                "qualification policies need an explicit common derivation manifest"
            )
        policy_manifest = next(iter(parents)) / "policy-derivation-manifest.json"
    if not policy_manifest.is_file():
        raise SystemExit(f"policy derivation manifest missing: {policy_manifest}")
    policy_binding = validate_policy_manifest(
        policy_manifest.resolve(),
        expected_source=args.expected_source,
        model=model,
        policies=source_policy_paths,
        role_prefix=("abba" if args.qualification_kind == "varlen" else "pld"),
        tensorfold=tensorfold,
    )

    args.output_dir.mkdir(parents=True)
    policy_paths = {}
    for name, value in (("control", control), ("candidate", candidate)):
        destination = args.output_dir / f"policy-{name}.json"
        destination.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
        policy_paths[name] = destination
    public_manifest = {
        key: value for key, value in manifest.items() if key != "rows"
    }
    public_manifest["rows"] = manifest["public_rows"]
    public_manifest.pop("public_rows", None)
    manifest_path = args.output_dir / "workload-manifest.json"
    manifest_path.write_bytes(json.dumps(public_manifest, indent=2, sort_keys=True).encode() + b"\n")

    campaign = {
        "schema": "mlx2.fixed-cohort-dflash-20x20-counterbalanced.v2",
        "started_at": utc_now(),
        "status": "running",
        "source": source,
        "runtime_source_sha256": source["runtime_source_sha256"],
        "model": model,
        "tensorfold": tensorfold,
        "gpu_owner": gpu_owner,
        "qualification_kind": args.qualification_kind,
        "arm_order": args.arm_order,
        "order": list(order),
        "sampling_profile": args.sampling_profile,
        "expected_arm_artifacts": args.expected_arm_artifacts,
        "policy_binding": policy_binding,
        "workload_manifest": str(manifest_path),
        "workload_manifest_sha256": sha256(manifest_path),
        "body_hashes_sha256": manifest["body_hashes_sha256"],
        "policies": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in policy_paths.items()
        },
        "arms": [],
    }
    campaign_path = args.output_dir / "campaign.json"
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    try:
        for occurrence, arm in enumerate(order, 1):
            result = run_arm(
                args,
                arm=arm,
                occurrence=occurrence,
                source=source,
                policy_path=policy_paths[arm],
                policy=control if arm == "control" else candidate,
                manifest=manifest,
            )
            arm_path = args.output_dir / f"{occurrence:02d}-{arm}.json"
            result["mechanism_validation"] = {"status": "pending"}
            campaign["arms"].append(write_arm_result(arm_path, result))
            campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
            try:
                validate_persisted_arm_mechanism(
                    result,
                    arm_path,
                    qualification_kind=args.qualification_kind,
                    arm=arm,
                )
            finally:
                campaign["arms"][-1] = write_arm_result(arm_path, result)
                campaign_path.write_text(
                    json.dumps(campaign, indent=2, sort_keys=True) + "\n"
                )
        reports = [
            json.loads(Path(arm["report"]).read_text()) for arm in campaign["arms"]
        ]
        campaign["summary"] = validate_cross_arm(reports, manifest, order=order)
        campaign["status"] = "passed"
        campaign["finished_at"] = utc_now()
        campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
        print(json.dumps(campaign["summary"], indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        campaign["status"] = "failed"
        campaign["finished_at"] = utc_now()
        campaign["error"] = f"{type(error).__name__}: {error}"
        campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
