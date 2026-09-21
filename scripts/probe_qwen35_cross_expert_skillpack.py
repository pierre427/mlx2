#!/usr/bin/env python3
"""A/B/C probe for a 64K-token crossed-expert system skillpack.

This is an exploratory long-context transfer test, not model qualification.
The held-out questions use numbers absent from both generated packs.  A
length-matched irrelevant pack separates expertise priming from context length.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path


EXPERT_MODULES = (
    "Model the system before calculating. Name state variables, assumptions, "
    "units, conservation laws, symmetries, and the observable. Use dimensional "
    "analysis to reject impossible expressions and limiting cases to test signs.",
    "Translate prose into mathematics. Distinguish definitions, identities, "
    "approximations, estimators, and hypotheses. Preserve exact symbolic forms "
    "until the last useful step, then report units and sensible precision.",
    "For noisy physical data, write a likelihood and an observation model. "
    "Combine independent Gaussian information by adding precisions. For Poisson "
    "counts use exposure explicitly and propagate uncertainty from the count.",
    "For Brownian motion remember that dimensions matter: each Cartesian axis "
    "contributes 2 D delta_t to mean squared displacement. State whether the "
    "measurement is one-, two-, or three-dimensional before estimating D.",
    "For small independent uncertainties, use first-order propagation: variance "
    "is the sum of squared partial derivatives times input variances, with "
    "covariance terms when inputs are correlated. Check whether a derivative "
    "vanishes at a symmetry point.",
    "Use stable numerical computation. Replace log(sum(exp(x_i))) with max(x) "
    "+ log(sum(exp(x_i-max(x)))). Avoid subtracting nearly equal numbers and "
    "track conditioning, truncation error, and floating-point scale separately.",
    "Match numerical methods to structure. Simpson quadrature fits smooth data "
    "on paired equal subintervals and integrates cubics exactly. Regression "
    "through the origin is justified only when the physical intercept is known.",
    "Physics scaling laws often answer faster than substitution: identify powers "
    "of length, time, mass, and temperature; take ratios; cancel shared constants; "
    "then verify the direction of change against intuition.",
    "Act as a skeptical statistical-computation expert. Produce an independent "
    "sanity check, identify the dominant error source, and separate epistemic "
    "assumptions from sampling variation. A plausible number without a check is "
    "not a completed solution.",
    "Cross disciplines deliberately: use physical constraints to regularize the "
    "statistical model, mathematical invariants to reduce computation, and "
    "numerical experiments only after deriving a result that can falsify them.",
)

CONTROL_MODULES = (
    "Archive vignette {i}: the fictional North Alcove catalogue assigns cobalt "
    "tabs to vellum folders, amber tabs to linen folders, and records shelf names "
    "in a ceremonial script. This material is descriptive, not instructional.",
    "Garden vignette {i}: a path passes painted gates, ornamental stones, wooden "
    "benches, and invented flowers. Curators debate names, pigments, and seasonal "
    "display order without measurements, equations, or analytic procedures.",
    "Typography vignette {i}: printers compare fictional ligatures, page borders, "
    "paper textures, and binding colors. The catalogue repeats provenance prose "
    "and makes no claim about science, mathematics, statistics, or computation.",
    "Travel vignette {i}: an imaginary ferry visits Bell Quay, Lantern Reach, "
    "Morrow Isle, and Quiet Harbor. The narrative concerns customs and scenery; "
    "all names and events are deliberately unrelated to technical problem solving.",
)

TASKS = (
    ("P1", 9.81, 0.06, ("regression", "origin", "pendulum")),
    ("P2", 11.28, 0.03, ("bayes", "precision", "posterior")),
    ("P3", 2.0, 0.02, ("diffusion", "brownian", "mean squared")),
    ("P4", 4.0, 0.05, ("delta", "derivative", "propagation")),
    ("P5", 12.5, 0.05, ("poisson", "background", "rate")),
    ("P6", 0.25, 0.01, ("scaling", "ratio", "power")),
    ("P7", -999.6867, 0.01, ("logsumexp", "log-sum-exp", "stable")),
    ("P8", 4.0, 0.01, ("simpson", "quadrature", "cubic")),
)

QUESTIONS = """Solve these held-out problems. Return exactly one line per problem as
P<number>=<numeric answer>|<method in a few words>. Do not add other prose.

P1. A pendulum obeys T^2=(4*pi^2/g)L with a known zero intercept. Measurements
(L metres, T seconds) are (0.25,1.003), (0.49,1.404), (0.81,1.806). Fit the
through-origin slope of T^2 on L and report g in m/s^2.
P2. A Gaussian prior for a parameter has mean 10 and standard deviation 2.
Four independent observations have sample mean 12 and known observation standard
deviation 3. Report the posterior mean.
P3. In two-dimensional Brownian motion, the measured mean squared displacement is
3.2 mm^2 over 0.4 s. Report D in mm^2/s.
P4. A projectile range model is R=v^2 sin(2 theta)/g. At v=20 m/s, theta=45
degrees, g=10 m/s^2, independent uncertainties are sigma_v=1 m/s and
sigma_theta=0.05 radians. First-order propagation: report sigma_R in metres.
P5. A detector records 145 counts in 10 seconds with a known background rate of
2 counts/s. Report the maximum-likelihood signal rate in counts/s.
P6. Luminosity scales as R^2 T^4. If radius doubles and temperature halves,
report new luminosity divided by old luminosity.
P7. Stably evaluate log(exp(-1000)+exp(-1001)).
P8. Apply Simpson's rule with nodes x=0,1,2 to integrate x^3 from 0 to 2.
"""


def _fit_pack(tokenizer, modules, target_tokens, title):
    chunks = [title]
    index = 0
    while len(tokenizer.encode("\n\n".join(chunks), add_special_tokens=False)) < target_tokens + 512:
        template = modules[index % len(modules)]
        chunks.append(f"Section {index + 1}. " + template.format(i=index + 1))
        index += 1
    ids = list(tokenizer.encode("\n\n".join(chunks), add_special_tokens=False))
    text = tokenizer.decode(ids[:target_tokens])
    return text, len(tokenizer.encode(text, add_special_tokens=False))


def _greedy_chunked(adapter, prompt_ids, max_tokens, prefill_step):
    import mlx.core as mx

    cache = adapter.model.make_cache()
    started = time.perf_counter()
    logits = None
    for start in range(0, len(prompt_ids), prefill_step):
        logits = adapter.model(
            mx.array([prompt_ids[start : start + prefill_step]]), cache=cache
        )
        mx.eval(logits)
    prefill_seconds = time.perf_counter() - started
    output = []
    eos = set(adapter.tokenizer.eos_token_ids)
    for _ in range(max_tokens):
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        if token in eos:
            break
        output.append(token)
        logits = adapter.model(mx.array([[token]]), cache=cache)
    text = adapter.tokenizer.decode(output)
    del cache
    mx.clear_cache()
    return text, prefill_seconds


def _score(text):
    rows = {}
    for match in re.finditer(
        r"(?im)^\s*(P[1-8])\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s*\|\s*(.*)$",
        text,
    ):
        rows[match.group(1).upper()] = (float(match.group(2)), match.group(3).casefold())
    details = []
    for task_id, expected, tolerance, keywords in TASKS:
        parsed = rows.get(task_id)
        numeric = parsed is not None and math.isfinite(parsed[0]) and abs(parsed[0] - expected) <= tolerance
        method = parsed is not None and any(word in parsed[1] for word in keywords)
        details.append(
            {
                "task": task_id,
                "expected": expected,
                "parsed": None if parsed is None else parsed[0],
                "numeric_correct": numeric,
                "method_recognized": method,
            }
        )
    return {
        "numeric_accuracy": sum(row["numeric_correct"] for row in details) / len(TASKS),
        "method_accuracy": sum(row["method_recognized"] for row in details) / len(TASKS),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skillpack-output", type=Path, required=True)
    parser.add_argument("--pack-tokens", type=int, default=65536)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--prefill-step", type=int, default=2048)
    args = parser.parse_args()

    from mlx2.adapters.qwen35_9b import Qwen359BAdapter

    adapter = Qwen359BAdapter(str(args.model))
    expert, expert_tokens = _fit_pack(
        adapter.tokenizer,
        EXPERT_MODULES,
        args.pack_tokens,
        "CROSSED EXPERT SKILLPACK: physics, mathematics, statistics, and numerical computation.",
    )
    control, control_tokens = _fit_pack(
        adapter.tokenizer,
        CONTROL_MODULES,
        args.pack_tokens,
        "LENGTH-MATCHED CONTROL ARCHIVE: unrelated fictional descriptive catalogue.",
    )
    args.skillpack_output.parent.mkdir(parents=True, exist_ok=True)
    args.skillpack_output.write_text(expert)

    conditions = (("baseline", ""), ("irrelevant_control_64k", control), ("cross_expert_64k", expert))
    observations = []
    for name, system in conditions:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": QUESTIONS})
        prompt_ids = list(
            adapter.prompt_tokens(
                {
                    "messages": messages,
                    "enable_thinking": False,
                    "reasoning_effort": "none",
                }
            )
        )
        text, prefill_seconds = _greedy_chunked(
            adapter, prompt_ids, args.max_tokens, args.prefill_step
        )
        observation = {
            "condition": name,
            "prompt_tokens": len(prompt_ids),
            "prefill_seconds": prefill_seconds,
            "output": text,
            **_score(text),
        }
        observations.append(observation)
        print(json.dumps({key: value for key, value in observation.items() if key != "output"}), flush=True)

    result = {
        "schema": "mlx2.qwen35-cross-expert-skillpack-ab.v1",
        "status": "exploratory-not-qualified",
        "pack_target_tokens": args.pack_tokens,
        "expert_pack_tokens": expert_tokens,
        "control_pack_tokens": control_tokens,
        "design": "single greedy run per condition; shared held-out task battery; length-matched irrelevant control",
        "limitations": [
            "one model and one greedy sample per condition",
            "eight compact synthetic tasks do not establish general intelligence",
            "pack is procedurally expanded from a small expert-method curriculum",
        ],
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "scores": [{"condition": row["condition"], "numeric_accuracy": row["numeric_accuracy"], "method_accuracy": row["method_accuracy"]} for row in observations]}, indent=2))


if __name__ == "__main__":
    main()
