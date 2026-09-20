#!/usr/bin/env python3
"""A/B the default-off ``memory_preemption`` policy over HTTP.

Real host pressure cannot be summoned on demand, so the arms use the
qualification-mode fault ``{"kind": "memory_preempt", "after_tokens": N}``:
the worker parks that lane exactly as a stall or CRITICAL pressure level
would, and replays it through the ordinary admission path.

The point of the run is correctness: a replayed greedy request must produce
the same text as the same request that was never preempted, on the same
server, and a concurrent peer must be unaffected apart from waiting while the
replay attaches.  Timings are reported as context, not as a target.

Usage (server already running in qualification mode with the policy on):
  python scripts/bench_suspend_replay.py --url http://127.0.0.1:8298 \
      --output run.json --prompt-repeats 400 --max-tokens 64 --fault-after 8
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

SCHEMA = "mlx2.memory-preemption-bench.v1"

PARAGRAPH = (
    "The scheduler parks a lane when the host cannot grow it, and replays the "
    "request from the tokens the client already received. "
)


def post(url: str, path: str, body: dict, timeout: float) -> dict:
    request = Request(
        url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def get(url: str, path: str, timeout: float) -> dict:
    with urlopen(url.rstrip("/") + path, timeout=timeout) as response:
        return json.loads(response.read())


def was_preempted(arm) -> bool:
    """Did this arm actually get preempted and replayed?

    Not ``preemption is not None``.  Since the fault-observability fix a
    *declined* fault also populates ``preemption`` -- ``replays: 0`` plus
    ``fault_unfired`` naming the reason -- so presence of the field is
    satisfied by the evidence that nothing happened.  Making a thing
    observable and making it asserted are different steps, and the first can
    quietly undo the second.  Assert the consequence instead.
    """
    record = arm.get("preemption") or {}
    return bool(record.get("replays")) and record.get("fault_unfired") is None


def chat(url, prompt, *, max_tokens, fault=None, timeout=1800.0):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    if fault is not None:
        body["mlx_fault"] = fault
    started = time.monotonic()
    response = post(url, "/v1/chat/completions", body, timeout)
    receipt = response.get("mlx2", {})
    return {
        "text": response["choices"][0]["message"].get("content") or "",
        "finish_reason": response["choices"][0].get("finish_reason"),
        "wall_s": time.monotonic() - started,
        "completion_tokens": receipt.get("completion_tokens"),
        "prompt_tokens": receipt.get("prompt_tokens"),
        "cached_tokens": receipt.get("cached_tokens"),
        "ttft_seconds": receipt.get("ttft_seconds"),
        "preemption": receipt.get("preemption"),
        # Interpretability against the admission changes: do not assume this
        # run's batch composition matches an earlier one.
        "ordinary_compute_width": receipt.get("ordinary_compute_width"),
        "execution_width": (receipt.get("speculation") or {}).get("target_width"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8298")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-repeats", type=int, default=400)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--fault-after", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    prompt = (
        PARAGRAPH * args.prompt_repeats
        + "\nIn one paragraph, restate the rule above."
    )
    peer_prompt = (
        PARAGRAPH * args.prompt_repeats
        + "\nIn one paragraph, say why replaying is safe."
    )
    results = {"schema": SCHEMA, "url": args.url, "arms": {}}
    results["status_before"] = get(args.url, "/v1/status", args.timeout)

    # Warm the prefix cache first so every arm below starts from the same
    # committed prompt boundary (a replay must not be the only warm run).
    results["arms"]["warm"] = chat(
        args.url, prompt, max_tokens=args.max_tokens, timeout=args.timeout
    )
    results["arms"]["reference"] = chat(
        args.url, prompt, max_tokens=args.max_tokens, timeout=args.timeout
    )
    # ``decode_replay`` is expected to be DECLINED: replay rebuilds
    # decode-produced state by prefilling, which is not bit-equal, so the
    # feature refuses it.  The arm proves the refusal is reported.
    for name, after in (("decode_replay", args.fault_after), ("prefill_replay", 0)):
        results["arms"][name] = chat(
            args.url, prompt, max_tokens=args.max_tokens,
            fault={"kind": "memory_preempt", "after_tokens": after},
            timeout=args.timeout,
        )
    results["arms"]["peer_reference"] = chat(
        args.url, peer_prompt, max_tokens=args.max_tokens, timeout=args.timeout
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        peer = pool.submit(
            # No fault=: this arm is never preempted.  It exists to put a
            # second lane in the batch, not to be checked for replay
            # exactness -- see the note in the checks block.
            chat, args.url, peer_prompt, max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        victim = pool.submit(
            chat, args.url, prompt, max_tokens=args.max_tokens,
            # Prefill phase: the only phase that can replay exactly, so the
            # concurrent arm must exercise preemption there.
            fault={"kind": "memory_preempt", "after_tokens": 0},
            timeout=args.timeout,
        )
        results["arms"]["concurrent_peer"] = peer.result()
        results["arms"]["concurrent_victim"] = victim.result()

    results["status_after"] = get(args.url, "/v1/status", args.timeout)
    counts = results["status_after"].get("counts", {})
    reference = results["arms"]["reference"]["text"]
    results["checks"] = {
        # Not a text comparison: that request is never preempted, so
        # comparing its text would pass vacuously.  Assert the refusal.
        "decode_replay_declined": (
            (results["arms"]["decode_replay"].get("preemption") or {}).get(
                "fault_unfired"
            )
            == "decode_state_not_reconstructible"
        ),
        "prefill_replay_text_matches": results["arms"]["prefill_replay"]["text"]
        == reference,
        # Deliberately NOT asserted: token identity for the two concurrent
        # arms against a reference taken at a different batch position.
        #
        # Served output is not reproducible across batch composition. Measured
        # on this stack, same binary and prompts: 8 of 8 prompts reproduced
        # token-for-token at width 1, 1 of 8 at width 16, and two lanes
        # running the *same* prompt in the *same* batch reported different
        # logprob margins for the same emitted token (0.25 vs 0.625 nats).
        # A cross-position comparison therefore fails for reasons that have
        # nothing to do with preemption -- which the GPU run showed directly:
        # ``concurrent_peer`` is never faulted (see the submission below, it
        # passes no ``fault=``) and it diverged anyway. A request that was
        # never preempted cannot exhibit a replay defect.
        #
        # Replay exactness is asserted where it is both achievable and
        # meaningful: the ``decode_replay`` and ``prefill_replay`` arms above
        # are faulted and compared against a reference taken at the same
        # width in the same process. The concurrent arms instead assert the
        # properties that are true at width, below.
        "concurrent_victim_preempted": was_preempted(
            results["arms"]["concurrent_victim"]
        ),
        "concurrent_peer_not_preempted": not was_preempted(
            results["arms"]["concurrent_peer"]
        ),
        "concurrent_victim_completed": bool(
            results["arms"]["concurrent_victim"]["text"]
        ),
        "concurrent_peer_completed": bool(
            results["arms"]["concurrent_peer"]["text"]
        ),
        "preemptions": counts.get("memory_preemptions"),
        "replays": counts.get("preempted_replays"),
        # Trigger provenance: with item 9's monitor live, a pressure
        # preemption inside a fault-scripted run is a different event and
        # would otherwise be invisible inside the total.
        "fault_preemptions": counts.get("memory_preemptions_fault"),
        "stall_preemptions": counts.get("memory_preemptions_stall", 0),
        "pressure_preemptions": counts.get("memory_preemptions_pressure", 0),
        # A fault that fired nothing makes every exactness claim resting on
        # it vacuous; it is a failure, not a silent pass.
        "faults_declined": counts.get("memory_preemption_fault_declined", 0),
        "faults_unfired": counts.get("memory_preemption_fault_unfired", 0),
        "every_prefill_fault_fired": all(
            was_preempted(results["arms"][name])
            for name in ("prefill_replay", "concurrent_victim")
        ),
        "fault_unfired_reasons": {
            name: (results["arms"][name].get("preemption") or {}).get("fault_unfired")
            for name in ("decode_replay", "prefill_replay", "concurrent_victim")
            if (results["arms"][name].get("preemption") or {}).get("fault_unfired")
        },
        "lane_evidence": {
            name: {
                "ordinary_compute_width": arm.get("ordinary_compute_width"),
                "execution_width": arm.get("execution_width"),
            }
            for name, arm in results["arms"].items()
        },
        "store_failures": counts.get("apcv2_store_failures", 0),
        "reference_has_no_preemption": results["arms"]["reference"]["preemption"]
        is None,
    }
    results["checks"]["accept"] = bool(
        results["checks"]["decode_replay_declined"]
        and results["checks"]["prefill_replay_text_matches"]
        and results["checks"]["concurrent_victim_preempted"]
        and results["checks"]["concurrent_peer_not_preempted"]
        and results["checks"]["concurrent_victim_completed"]
        and results["checks"]["concurrent_peer_completed"]
        and results["checks"]["every_prefill_fault_fired"]
        # Exactly one unfired fault: the decode arm, refused by design.
        and results["checks"]["faults_unfired"] == 1
        and results["checks"]["faults_declined"] == 1
        # Scripted arms: every preemption must come from the injection, not
        # from real pressure or a stall that happened to coincide.
        and results["checks"]["fault_preemptions"] == results["checks"]["preemptions"]
        # Two, not three: the decode arm is refused by design.
        and results["checks"]["preemptions"] == 2
        # Every preemption must be followed by exactly one replay: the
        # property preemption actually promises at width.
        and results["checks"]["replays"] == results["checks"]["preemptions"]
        and results["checks"]["replays"] == 3
        and results["checks"]["reference_has_no_preemption"]
        and not results["checks"]["store_failures"]
    )
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print(json.dumps(results["checks"], indent=2, sort_keys=True))
    return 0 if results["checks"]["accept"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
