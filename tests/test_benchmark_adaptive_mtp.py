import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import json
import pytest

from scripts import benchmark_adaptive_mtp as benchmark


def test_first_token_difference_reports_the_first_divergence():
    assert benchmark.first_token_difference(["a", "b"], ["a", "c"]) == {
        "index": 1,
        "reference": "b",
        "candidate": "c",
    }
    assert benchmark.first_token_difference(["a"], ["a", "b"]) == {
        "index": 1,
        "reference": None,
        "candidate": "b",
    }
    assert benchmark.first_token_difference(["a"], ["a"]) is None


def test_correctness_requires_adaptive_fixed_identity_but_allows_near_tie_ordinary():
    def request(text, token_ids, *, margin=0.0, adaptive=False):
        tokens = [
            {
                "id": token_id,
                "token": str(token_id),
                "top_logprobs": [
                    {"id": token_id, "logprob": -0.75},
                    {"id": token_id + 100, "logprob": -0.75 - margin},
                ],
            }
            for token_id in token_ids
        ]
        receipt = {}
        if adaptive:
            receipt = {
                "mtp": {
                    "num_draft": 2,
                    "adaptive_depth": {
                        "trace": [{"width": 1, "selected_depth": 2}]
                    },
                }
            }
        return {
            "output": text,
            "output_sha256": text,
            "tokens": tokens,
            "receipt": receipt,
        }

    arms = {
        "ordinary": {
            "sequential": {"requests": [request("ordinary", [1, 2], margin=0.125)]}
        },
        "fixed": {"sequential": {"requests": [request("mtp", [1, 3])]}},
        "adaptive": {
            "sequential": {"requests": [request("mtp", [1, 3], adaptive=True)]}
        },
    }
    result = benchmark.correctness(
        arms, fixed_depth=2, ordinary_margin_threshold=0.5
    )
    assert result["passed"]
    assert result["adaptive_vs_fixed"]["exact_match_fraction"] == 1.0
    comparison = result["versus_ordinary"]["fixed"]["comparisons"][0]
    assert comparison["first_differing_token"]["index"] == 1
    assert comparison["first_differing_token"][
        "reference_top2_margin_nats"
    ] == 0.125
    assert result["versus_ordinary"]["fixed"]["exact_match_fraction"] == 0.0
    assert result["unsafe_ordinary_divergences"] == []


def test_correctness_rejects_adaptive_drift_and_large_margin_ordinary_divergence():
    def request(token_id, *, margin=0.0, depth=2):
        return {
            "output": str(token_id),
            "output_sha256": str(token_id),
            "tokens": [
                {
                    "id": token_id,
                    "token": str(token_id),
                    "top_logprobs": [
                        {"id": token_id, "logprob": -0.1},
                        {"id": token_id + 10, "logprob": -0.1 - margin},
                    ],
                }
            ],
            "receipt": {
                "mtp": {
                    "num_draft": depth,
                    "adaptive_depth": {
                        "trace": [{"width": 1, "selected_depth": depth}]
                    },
                }
            },
        }

    arms = {
        "ordinary": {"sequential": {"requests": [request(1, margin=0.75)]}},
        "fixed": {"sequential": {"requests": [request(2)]}},
        "adaptive": {"sequential": {"requests": [request(3, depth=1)]}},
    }
    result = benchmark.correctness(
        arms, fixed_depth=2, ordinary_margin_threshold=0.5
    )
    assert not result["passed"]
    assert result["adaptive_vs_fixed"]["exact_match_fraction"] == 0.0
    assert len(result["unsafe_ordinary_divergences"]) == 2


def test_handoff_b1_correctness_requires_fixed_depth_identity():
    def request(token_id, *, depth=None):
        receipt = {} if depth is None else {"mtp": {"num_draft": depth}}
        return {
            "output": str(token_id),
            "output_sha256": str(token_id),
            "tokens": [{
                "id": token_id,
                "token": str(token_id),
                "top_logprobs": [
                    {"id": token_id, "logprob": -0.1},
                    {"id": token_id + 10, "logprob": -0.2},
                ],
            }],
            "receipt": receipt,
        }

    arms = {
        "ordinary": {"sequential": {"requests": [request(1)]}},
        "fixed": {"sequential": {"requests": [request(1, depth=2)]}},
        "handoff": {"sequential": {"requests": [request(1, depth=2)]}},
    }
    result = benchmark.correctness(
        arms,
        candidate_arm="handoff",
        fixed_depth=2,
        ordinary_margin_threshold=0.5,
    )
    assert result["passed"]
    assert result["handoff_vs_fixed"]["exact_match_fraction"] == 1.0
    arms["handoff"]["sequential"]["requests"][0]["receipt"]["mtp"][
        "num_draft"
    ] = 1
    assert not benchmark.correctness(
        arms,
        candidate_arm="handoff",
        fixed_depth=2,
        ordinary_margin_threshold=0.5,
    )["passed"]


def test_dry_run_prints_all_fresh_server_arms_without_starting_metal(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(benchmark.ROOT / "scripts/benchmark_adaptive_mtp.py"),
            "--model",
            "qwen38",
            "--output",
            str(tmp_path / "report.json"),
            "--dry-run",
        ],
        cwd=benchmark.ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.count("-m mlx2.server") == 3
    assert "[ordinary]" in completed.stdout
    assert "--ordinary" in completed.stdout
    assert "[fixed]" in completed.stdout
    assert "[adaptive]" in completed.stdout
    assert "--adaptive-mtp-depth" in completed.stdout
    assert "feature_smoke.py" in completed.stdout
    assert not (tmp_path / "report.json").exists()


def test_handoff_dry_run_replaces_adaptive_with_fixed_depth_shipping_arm(
    tmp_path,
):
    completed = subprocess.run(
        [
            sys.executable,
            str(benchmark.ROOT / "scripts/benchmark_adaptive_mtp.py"),
            "--model",
            "qwen38",
            "--mtp-ordinary-handoff-max-width",
            "4",
            "--output",
            str(tmp_path / "report.json"),
            "--dry-run",
        ],
        cwd=benchmark.ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.count("-m mlx2.server") == 3
    assert "[handoff]" in completed.stdout
    assert "[adaptive]" not in completed.stdout
    commands = benchmark.commands_for_arm(
        benchmark.parse_args([
            "--model", "qwen38",
            "--mtp-ordinary-handoff-max-width", "4",
            "--output", str(tmp_path / "report.json"),
        ]),
        "handoff",
        tmp_path,
        benchmark.PRESETS["qwen38"],
    )
    assert commands["policy"]["adaptive_mtp_depth"] is False
    assert commands["policy"]["mtp_ordinary_handoff"] == {
        "enabled": True,
        "max_mtp_width": 4,
    }
    assert "--adaptive-mtp-depth" not in commands["server"]


def test_stream_request_rows_match_the_evidence_schema():
    class StreamHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            events = [
                {
                    "choices": [{
                        "delta": {"content": "A"},
                        "logprobs": {"content": [{
                            "id": 7,
                            "token": "A",
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"id": 7, "token": "A", "logprob": -0.1},
                                {"id": 8, "token": "B", "logprob": -0.2},
                            ],
                        }]},
                    }],
                },
                {
                    "usage": {"completion_tokens": 1},
                    "mlx2": {"ordinary_compute_width": 1},
                },
            ]
            body = b"".join(
                f"data: {json.dumps(event)}\n\n".encode() for event in events
            ) + b"data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), StreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        row = benchmark._stream_request(
            f"http://127.0.0.1:{server.server_port}", "prompt", 5.0, 4
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert "token_ids" not in row
    assert row["tokens"] == [{
        "id": 7,
        "token": "A",
        "logprob": -0.1,
        "top_logprobs": [
            {"id": 7, "token": "A", "logprob": -0.1},
            {"id": 8, "token": "B", "logprob": -0.2},
        ],
    }]
    assert benchmark._token_identities(row) == [7]
    candidate = json.loads(json.dumps(row))
    candidate["receipt"] = {
        "mtp": {
            "mtp_ordinary_handoff": {
                "engaged": True,
                "committed_tokens_before_handoff": 0,
            }
        }
    }
    assert benchmark._handoff_comparison(
        [row, json.loads(json.dumps(row))],
        candidate,
        ordinary_margin_threshold=0.5,
        observed_widths=[16],
    )["passed"]


def _qualification_report(*, decreases=0, recoveries=0, d2_rate=104.8):
    status = {
        "settings": {
            "adaptive_mtp_depth": {"enabled": True, "min_samples_per_depth": 3},
            "mtp_ordinary_handoff": {"enabled": False},
        },
        "scheduler": {
            "adaptive_mtp_depth_decreases_concurrent": decreases,
            "adaptive_mtp_depth_recoveries_alone": recoveries,
            "adaptive_mtp_cost_model": {
                "buckets": {
                    "5-8": {
                        "rounds": 206,
                        "probe_fraction": 0.03,
                        "samples": {"0": 4, "1": 3, "2": 199},
                        "goodput_tokens_per_second": {
                            "0": 73.3,
                            "1": 80.7,
                            "2": d2_rate,
                        },
                    }
                }
            },
        },
    }
    return {
        "qualified_depth": 2,
        "correctness": {
            "passed": True,
            "ordinary_margin_failure_threshold_nats": 0.5,
        },
        "arms": {
            "fixed": {
                "sequential": {"median_decode_tokens_per_second": 42.3},
                "concurrent": [
                    {"width": 8, "aggregate_tokens_per_second": 86.3},
                    {"width": 16, "aggregate_tokens_per_second": 90.2},
                ],
            },
            "adaptive": {
                "sequential": {"median_decode_tokens_per_second": 42.2},
                "concurrent": [
                    {"width": 8, "aggregate_tokens_per_second": 86.1},
                    {"width": 16, "aggregate_tokens_per_second": 84.9},
                ],
                "final_status": status,
                "feature_smoke": {"returncode": 0},
            },
        },
    }


def _handoff_only_report():
    report = _qualification_report()
    prompt = "handoff-prompt"

    def request(*, handoff=False):
        receipt = {"ordinary_compute_width": 1}
        if handoff:
            receipt = {"mtp": {
                "observed_compute_widths": [16],
                "mtp_ordinary_handoff": {
                    "engaged": True,
                    "committed_tokens_before_handoff": 0,
                },
            }}
        return {
            "prompt_sha256": prompt,
            "output": "12",
            "output_sha256": "12",
            "tokens": [
                {
                    "id": token_id,
                    "token": str(token_id),
                    "top_logprobs": [
                        {"id": token_id, "logprob": -0.1},
                        {"id": token_id + 100, "logprob": -0.225},
                    ],
                }
                for token_id in (1, 2)
            ],
            "receipt": receipt,
        }

    handoff = report["arms"].pop("adaptive")
    report["arms"]["handoff"] = handoff
    report["qualification_arm"] = "handoff"
    status = handoff["final_status"]
    status["settings"]["adaptive_mtp_depth"] = {"enabled": False}
    status["settings"]["mtp_ordinary_handoff"] = {
        "enabled": True,
        "max_mtp_width": 4,
    }
    status["scheduler"].update(
        mtp_ordinary_handoff_events=1,
        mtp_ordinary_handoff_lanes=16,
    )
    status["scheduler"]["adaptive_mtp_cost_model"] = {"buckets": {}}
    for row in handoff["concurrent"]:
        row["requests"] = []
    handoff["concurrent"][1]["requests"] = [request(handoff=True)]
    # The control arm the differential screen needs: the same width, the same
    # references, no handoff.  Without it the screen cannot judge and fails
    # closed, so a fixture that omits it is not testing the gate.
    for row in report["arms"]["fixed"]["concurrent"]:
        row["requests"] = []
    report["arms"]["fixed"]["concurrent"][1]["requests"] = [request()]
    report["arms"]["ordinary"] = {
        "handoff_reference": {
            "passes": [[request()], [request()]],
        },
    }
    return report


def test_qualification_accepts_measured_optimal_depth_without_a_decrease():
    evidence = benchmark.adaptive_qualification_evidence(_qualification_report())
    assert evidence["passed"]
    assert evidence["eligible_concurrent_buckets"] == 1
    assert evidence["recovery"] == {
        "decreases": 0,
        "recoveries": 0,
        "required": False,
        "passed": True,
    }
    assert all(row["passed"] for row in evidence["throughput"])


def test_qualification_requires_recovery_only_after_a_decrease_and_all_depths():
    report = _qualification_report(decreases=1, recoveries=0)
    assert not benchmark.adaptive_qualification_evidence(report)["passed"]
    report["arms"]["adaptive"]["final_status"]["scheduler"][
        "adaptive_mtp_depth_recoveries_alone"
    ] = 1
    samples = report["arms"]["adaptive"]["final_status"]["scheduler"][
        "adaptive_mtp_cost_model"
    ]["buckets"]["5-8"]["samples"]
    samples.pop("1")
    evidence = benchmark.adaptive_qualification_evidence(report)
    assert not evidence["passed"]
    assert evidence["buckets"][0]["missing_depths"] == [1]


def test_handoff_only_qualification_does_not_require_adaptive_buckets():
    evidence = benchmark.adaptive_qualification_evidence(_handoff_only_report())
    assert evidence["passed"]
    assert evidence["eligible_concurrent_buckets"] == 0
    assert evidence["features"]["adaptive_mtp_depth"] == {
        "selected": False,
        "passed": True,
    }
    assert evidence["features"]["mtp_ordinary_handoff"]["selected"] is True
    assert evidence["features"]["mtp_ordinary_handoff"]["passed"] is True
    assert evidence["missing_feature_evidence"] == []


def test_enabled_benchmark_feature_without_evidence_fails_by_name():
    report = _handoff_only_report()
    status = report["arms"]["handoff"]["final_status"]
    status["scheduler"]["mtp_ordinary_handoff_events"] = 0
    status["scheduler"]["mtp_ordinary_handoff_lanes"] = 0
    for row in report["arms"]["handoff"]["concurrent"]:
        row["requests"] = []
    evidence = benchmark.adaptive_qualification_evidence(report)
    assert not evidence["passed"]
    assert evidence["missing_feature_evidence"] == ["mtp_ordinary_handoff"]

    status["settings"]["adaptive_mtp_depth"] = {"enabled": True}
    evidence = benchmark.adaptive_qualification_evidence(report)
    assert not evidence["passed"]
    assert evidence["missing_feature_evidence"] == [
        "adaptive_mtp_depth",
        "mtp_ordinary_handoff",
    ]


def test_handoff_gate_uses_stable_width_one_margin_bound_when_batched():
    def request(
        token_ids,
        *,
        margins=(0.125, 0.125, 0.125),
        handoff=False,
        observed_widths=(16,),
    ):
        receipt = {
            "ordinary_compute_width": observed_widths[0]
        } if observed_widths else {}
        if handoff:
            receipt = {
                "mtp": {
                    "route": "ordinary_after_mtp_handoff",
                    "observed_compute_widths": list(observed_widths),
                    "mtp_ordinary_handoff": {
                        "engaged": True,
                        "committed_tokens_before_handoff": 2,
                    },
                }
            }
        return {
            "prompt_sha256": "same-prompt",
            "output": "".join(map(str, token_ids)),
            "output_sha256": "".join(map(str, token_ids)),
            "tokens": [
                {
                    "id": token_id,
                    "token": str(token_id),
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"id": token_id, "logprob": -0.1},
                        {"id": token_id + 100, "logprob": -0.1 - margin},
                    ],
                }
                for token_id, margin in zip(token_ids, margins)
            ],
            "receipt": receipt,
        }

    report = _qualification_report()
    adaptive = report["arms"]["adaptive"]
    adaptive["final_status"]["settings"]["mtp_ordinary_handoff"] = {
        "enabled": True,
        "max_mtp_width": 8,
    }
    adaptive["final_status"]["scheduler"].update(
        mtp_ordinary_handoff_events=1,
        mtp_ordinary_handoff_lanes=16,
    )
    for row in adaptive["concurrent"]:
        row["requests"] = []
    adaptive["concurrent"][1]["requests"] = [request([1, 2, 3], handoff=True)]
    # The differential screen needs the arm the handoff replaces, at the same
    # width and against the same references, or it fails closed.
    for row in report["arms"]["fixed"]["concurrent"]:
        row["requests"] = []
    report["arms"]["fixed"]["concurrent"][1]["requests"] = [request([1, 2, 3])]
    report["arms"]["ordinary"] = {
        "concurrent": [
            {"width": 8, "requests": []},
            {"width": 16, "requests": [request([1, 2, 3])]},
        ],
        "handoff_reference": {
            "passes": [
                [request([1, 2, 3], observed_widths=(1,))],
                [request([1, 2, 3], observed_widths=(1,))],
            ]
        },
    }
    evidence = benchmark.adaptive_qualification_evidence(report)
    assert evidence["passed"]
    assert evidence["handoff"]["exact_match_fraction"] == 1.0
    comparison = evidence["handoff"]["comparisons"][0]
    assert comparison["correctness_rule"] == "differential_high_margin_screen"
    assert comparison["observed_compute_widths"] == [16]
    assert comparison["width"] == 16
    assert comparison["reference_sample_count"] == 2
    assert comparison["reference_widths"] == [[1], [1]]

    # At batch width, a measured bf16 near-tie is safe on either side of the
    # handoff boundary because even ordinary-vs-ordinary runs are not bit-stable.
    adaptive["concurrent"][1]["requests"] = [request([1, 2, 4], handoff=True)]
    evidence = benchmark.adaptive_qualification_evidence(report)
    assert evidence["passed"]
    assert evidence["handoff"]["exact_match_fraction"] == 0.0
    assert evidence["handoff"]["unsafe_divergences"] == []

    # The margin bound is still the conservative maximum across stable width-one
    # references, and a bound above the ceiling still CLASSIFIES the row as a
    # high-margin divergence.  What changed is that one such row no longer
    # decides the gate on its own: a lone divergence cannot be distinguished
    # from the control, which diverges at batch width too.
    report["arms"]["ordinary"]["handoff_reference"]["passes"] = [
        [request([1, 2, 3], margins=(0.125, 0.125, 0.25), observed_widths=(1,))],
        [request([1, 2, 3], margins=(0.125, 0.125, 0.625), observed_widths=(1,))],
    ]
    adaptive["concurrent"][1]["requests"] = [request([1, 2, 4], handoff=True)]
    evidence = benchmark.adaptive_qualification_evidence(report)
    comparison = evidence["handoff"]["comparisons"][0]
    assert comparison["reference_top2_margin_samples_nats"] == pytest.approx(
        [0.25, 0.625]
    )
    assert comparison["reference_top2_margin_bound_nats"] == 0.625
    assert comparison["reference_top2_margin_spread_nats"] == 0.375
    assert comparison["high_margin_divergence"]
    excess = evidence["handoff"]["high_margin_excess"]
    assert excess["candidate_high_margin_divergences"] == 1
    assert excess["p_value"] > excess["alpha"]
    assert evidence["passed"]

    # Width one remains cross-run reproducible and therefore identity-strict.
    strict = benchmark._handoff_comparison(
        [
            request([1, 2, 3], observed_widths=(1,)),
            request([1, 2, 3], observed_widths=(1,)),
        ],
        request([1, 2, 4], handoff=True, observed_widths=(1,)),
        ordinary_margin_threshold=0.5,
        observed_widths=[1],
    )
    assert strict["correctness_rule"] == "strict_token_identity"
    assert not strict["passed"]

    # Workload width is not evidence that this lane actually decoded at that
    # width. A missing receipt width must not silently select either oracle.
    adaptive["concurrent"][1]["requests"] = [
        request([1, 2, 3], handoff=True, observed_widths=())
    ]
    evidence = benchmark.adaptive_qualification_evidence(report)
    comparison = evidence["handoff"]["comparisons"][0]
    assert not evidence["passed"]
    assert comparison["observed_compute_widths"] == []
    assert comparison["width"] is None
    assert comparison["correctness_rule"] == "unknown_width_fail_closed"


def test_handoff_gate_fails_closed_without_two_stable_width_one_references():
    def request(token_ids, *, width=1, top_ids=None, handoff=False):
        top_ids = top_ids or [token_ids[-1], token_ids[-1] + 100]
        receipt = {"ordinary_compute_width": width}
        if handoff:
            receipt = {"mtp": {
                "observed_compute_widths": [16],
                "mtp_ordinary_handoff": {
                    "engaged": True,
                    "committed_tokens_before_handoff": 0,
                },
            }}
        return {
            "prompt_sha256": "same-prompt",
            "output": "".join(map(str, token_ids)),
            "output_sha256": "".join(map(str, token_ids)),
            "tokens": [
                {
                    "id": token_id,
                    "token": str(token_id),
                    "top_logprobs": [
                        {"id": top_ids[0], "logprob": -0.1},
                        {"id": top_ids[1], "logprob": -0.2},
                    ],
                }
                for token_id in token_ids
            ],
            "receipt": receipt,
        }

    candidate = request([1, 3], handoff=True)
    missing = benchmark._handoff_comparison(
        [request([1, 2])], candidate,
        ordinary_margin_threshold=0.5, observed_widths=[16],
    )
    assert not missing["passed"]
    assert missing["correctness_rule"] == "unstable_width_one_reference_fail_closed"
    assert missing["reference_stability_reason"] == "insufficient_samples"

    mismatched = benchmark._handoff_comparison(
        [request([1, 2]), request([1, 2], top_ids=[2, 999])], candidate,
        ordinary_margin_threshold=0.5, observed_widths=[16],
    )
    assert not mismatched["passed"]
    assert mismatched["reference_stability_reason"] == "top_two_identity_mismatch"


def test_qualifier_binds_adaptive_benchmark_to_exact_candidate(tmp_path):
    from scripts.qualify_serving import validate_adaptive_benchmark

    report = _qualification_report()
    report.update(
        schema=benchmark.SCHEMA,
        passed=True,
        benchmark_harness={
            "name": "scripts/benchmark_adaptive_mtp.py",
            "sha256": benchmark.hashlib.sha256(
                benchmark.Path(benchmark.__file__).read_bytes()
            ).hexdigest(),
        },
        adaptive_qualification=benchmark.adaptive_qualification_evidence(report),
    )
    adaptive_status = report["arms"]["adaptive"]["final_status"]
    adaptive_status.update(runtime={"source_sha256": "abc"}, artifact="weights")
    path = tmp_path / "adaptive.json"
    path.write_text(json.dumps(report))
    loaded, evidence = validate_adaptive_benchmark(path, adaptive_status)
    assert loaded["passed"]
    assert len(evidence["sha256"]) == 64
    mismatched = {**adaptive_status, "artifact": "other"}
    with pytest.raises(ValueError, match="artifact does not match"):
        validate_adaptive_benchmark(path, mismatched)


def test_qualifier_binds_handoff_only_benchmark_to_handoff_arm(tmp_path):
    from scripts.qualify_serving import validate_adaptive_benchmark

    report = _handoff_only_report()
    report.update(
        schema=benchmark.SCHEMA,
        passed=True,
        benchmark_harness={
            "name": "scripts/benchmark_adaptive_mtp.py",
            "sha256": benchmark.hashlib.sha256(
                benchmark.Path(benchmark.__file__).read_bytes()
            ).hexdigest(),
        },
        adaptive_qualification=benchmark.adaptive_qualification_evidence(report),
    )
    handoff_status = report["arms"]["handoff"]["final_status"]
    handoff_status.update(runtime={"source_sha256": "abc"}, artifact="weights")
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(report))
    loaded, evidence = validate_adaptive_benchmark(path, handoff_status)
    assert loaded["qualification_arm"] == "handoff"
    assert evidence["adaptive_qualification"]["features"][
        "adaptive_mtp_depth"
    ]["selected"] is False


def test_qualifier_recomputes_the_local_benchmark_harness_hash(
    tmp_path, monkeypatch
):
    from scripts import qualify_serving

    report = _qualification_report()
    report.update(
        schema=benchmark.SCHEMA,
        passed=True,
        benchmark_harness={
            "name": "scripts/benchmark_adaptive_mtp.py",
            "sha256": "0" * 64,
        },
        adaptive_qualification=benchmark.adaptive_qualification_evidence(report),
    )
    path = tmp_path / "adaptive.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(
        qualify_serving, "APPROVED_ADAPTIVE_BENCHMARK_SHA256", "0" * 64
    )
    with pytest.raises(ValueError, match="local adaptive benchmark harness"):
        qualify_serving.validate_adaptive_benchmark(path, {})


def test_qualifier_script_loads_sibling_benchmark_from_documented_command(
    tmp_path
):
    report = {
        "schema": benchmark.SCHEMA,
        "benchmark_harness": {
            "name": "scripts/benchmark_adaptive_mtp.py",
            "sha256": benchmark.hashlib.sha256(
                benchmark.Path(benchmark.__file__).read_bytes()
            ).hexdigest(),
        },
        "adaptive_qualification": {
            "throughput_tolerance": 1.0,
            "max_probe_fraction": 0.075,
            "min_bucket_rounds": 64,
        },
    }
    benchmark_path = tmp_path / "adaptive.json"
    benchmark_path.write_text(json.dumps(report))

    class StatusHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"healthy": True, "inflight": 0}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), StatusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/qualify_serving.py",
                "--url",
                f"http://127.0.0.1:{server.server_port}",
                "--output",
                str(tmp_path / "qualification.json"),
                "--adaptive-benchmark",
                str(benchmark_path),
            ],
            cwd=benchmark.ROOT,
            text=True,
            capture_output=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert completed.returncode != 0
    assert "weaker qualification limits" in completed.stderr
    assert "ModuleNotFoundError" not in completed.stderr


def test_arm_refuses_a_route_it_is_not_labelled_with():
    """An arm must not measure whatever route the adapter default picked.

    Qwen3.6 defaults to ordinary, so a native-MTP arm started without an
    explicit route flag silently measured ordinary decode.
    """
    import pytest

    _assert_arm_route = benchmark._assert_arm_route

    _assert_arm_route("ordinary", {"settings": {"route": "ordinary"}})
    _assert_arm_route("fixed", {"settings": {"route": "native_mtp"}})
    _assert_arm_route("adaptive", {"settings": {"route": "native_mtp"}})
    for arm, route in (("fixed", "ordinary"), ("adaptive", "ordinary"), ("ordinary", "native_mtp")):
        with pytest.raises(RuntimeError, match="expected"):
            _assert_arm_route(arm, {"settings": {"route": route}})
    with pytest.raises(RuntimeError):
        _assert_arm_route("fixed", {"settings": {}})


def test_native_mtp_arms_name_their_route_explicitly():
    from pathlib import Path
    from types import SimpleNamespace

    bench = benchmark

    args = SimpleNamespace(
        python="python", model_path="/tmp/model", port=8296, widths=[8, 16],
        max_tokens=256, server_arg=[], mtp_ordinary_handoff_max_width=None,
        depth=2, model="qwen36", check_timeout=1800.0, request_timeout=1800.0, cache_gib=None, max_context=None,
        startup_timeout=900.0,
    )
    preset = bench.PRESETS["qwen36"]
    for arm in ("fixed", "adaptive"):
        commands = bench.commands_for_arm(args, arm, Path("/tmp/run"), preset)
        server = commands["server"]
        assert "--native-mtp" in server, (arm, server)
        assert commands["policy"]["mtp_ordinary_handoff"] is False
    ordinary = bench.commands_for_arm(args, "ordinary", Path("/tmp/run"), preset)["server"]
    assert "--ordinary" in ordinary and "--native-mtp" not in ordinary


def _differential_report(*, candidate_high, control_high, n=8):
    """A handoff report whose batched arms diverge a controlled number of times.

    Every divergence is identical in kind -- same prompt, same reference pair,
    same 0.75 nat reference margin.  Only how many rows diverge differs between
    the candidate and its control, which is exactly what the screen is supposed
    to be sensitive to.
    """

    def request(prompt, token_ids, *, handoff=False, width=16):
        receipt = {"ordinary_compute_width": width}
        if handoff:
            receipt = {
                "mtp": {
                    "observed_compute_widths": [width],
                    "mtp_ordinary_handoff": {
                        "engaged": True,
                        "committed_tokens_before_handoff": 1,
                    },
                }
            }
        return {
            "prompt_sha256": prompt,
            "output": "".join(map(str, token_ids)),
            "output_sha256": "".join(map(str, token_ids)),
            "tokens": [
                {
                    "id": token_id,
                    "token": str(token_id),
                    "top_logprobs": [
                        {"id": token_id, "logprob": -0.1},
                        {"id": token_id + 100, "logprob": -0.85},
                    ],
                }
                for token_id in token_ids
            ],
            "receipt": receipt,
        }

    prompts = [f"prompt-{index}" for index in range(n)]
    report = _qualification_report()
    handoff = report["arms"].pop("adaptive")
    report["arms"]["handoff"] = handoff
    report["qualification_arm"] = "handoff"
    status = handoff["final_status"]
    status["settings"]["adaptive_mtp_depth"] = {"enabled": False}
    status["settings"]["mtp_ordinary_handoff"] = {
        "enabled": True,
        "max_mtp_width": 4,
    }
    status["scheduler"].update(
        mtp_ordinary_handoff_events=1, mtp_ordinary_handoff_lanes=16
    )
    status["scheduler"]["adaptive_mtp_cost_model"] = {"buckets": {}}

    def rows(high, *, handoff_arm):
        return [
            request(
                prompt,
                [1, 3] if index < high else [1, 2],
                handoff=handoff_arm,
            )
            for index, prompt in enumerate(prompts)
        ]

    for row in handoff["concurrent"]:
        row["requests"] = []
    handoff["concurrent"][1]["requests"] = rows(
        candidate_high, handoff_arm=True
    )
    for row in report["arms"]["fixed"]["concurrent"]:
        row["requests"] = []
    report["arms"]["fixed"]["concurrent"][1]["requests"] = rows(
        control_high, handoff_arm=False
    )
    reference = [request(prompt, [1, 2], width=1) for prompt in prompts]
    report["arms"]["ordinary"] = {
        "handoff_reference": {"passes": [reference, list(reference)]}
    }
    return report


def test_handoff_gate_fails_on_a_significant_excess_over_its_control():
    # The gate must be able to fail, or passing it means nothing.  Every
    # candidate row diverges at a confident reference position while the
    # control diverges on none.
    evidence = benchmark.adaptive_qualification_evidence(
        _differential_report(candidate_high=8, control_high=0)
    )
    excess = evidence["handoff"]["high_margin_excess"]
    assert excess["candidate_high_margin_divergences"] == 8
    assert excess["control_high_margin_divergences"] == 0
    assert excess["p_value"] < excess["alpha"]
    assert not excess["passed"]
    assert not evidence["passed"]
    assert evidence["missing_feature_evidence"] == ["mtp_ordinary_handoff"]


def test_handoff_gate_passes_when_its_control_diverges_just_as_often():
    # The same candidate divergences, now matched by the arm the handoff
    # replaces.  This is the measured reality at batch width, and it is not
    # evidence against the handoff.
    evidence = benchmark.adaptive_qualification_evidence(
        _differential_report(candidate_high=8, control_high=8)
    )
    excess = evidence["handoff"]["high_margin_excess"]
    assert excess["candidate_high_margin_divergences"] == 8
    assert excess["control_high_margin_divergences"] == 8
    assert excess["p_value"] == 1.0
    assert evidence["passed"]
    assert evidence["handoff"]["exact_match_fraction"] == 0.0
    assert evidence["handoff"]["control_exact_match_fraction"] == 0.0


def test_handoff_gate_fails_closed_without_a_control_arm():
    report = _differential_report(candidate_high=0, control_high=0)
    for row in report["arms"]["fixed"]["concurrent"]:
        row["requests"] = []
    evidence = benchmark.adaptive_qualification_evidence(report)
    excess = evidence["handoff"]["high_margin_excess"]
    assert excess["control_comparisons"] == 0
    assert excess["reason"] == "no_control_comparisons_fail_closed"
    assert not excess["passed"]
    assert not evidence["passed"]


def test_handoff_screen_records_the_excess_it_could_not_detect():
    # An underpowered screen must say so in the report rather than let a pass
    # imply a sensitivity it does not have.
    evidence = benchmark.adaptive_qualification_evidence(
        _differential_report(candidate_high=3, control_high=0)
    )
    excess = evidence["handoff"]["high_margin_excess"]
    assert excess["p_value"] > excess["alpha"]
    assert excess["passed"]
    assert excess["minimum_detectable_excess"] == 4
    assert excess["candidate_high_margin_divergences"] < (
        excess["minimum_detectable_excess"]
    )


def test_high_margin_screen_counts_prompts_not_lanes():
    # A batch runs the same prompt on several lanes.  Those rows share a
    # reference and therefore the near-tie that classifies them, so counting
    # them separately inflates the sample and the confidence.  Measured on
    # Flash-Next: 3/24 rows was 2/8 prompts, one of them shared with the
    # control.
    rows = [
        {"prompt_sha256": "a", "high_margin_divergence": True},
        {"prompt_sha256": "a", "high_margin_divergence": True},
        {"prompt_sha256": "a", "high_margin_divergence": True},
        {"prompt_sha256": "b", "high_margin_divergence": False},
        {"prompt_sha256": "c", "high_margin_divergence": False},
    ]
    control = [
        {"prompt_sha256": "a", "high_margin_divergence": False},
        {"prompt_sha256": "b", "high_margin_divergence": False},
        {"prompt_sha256": "c", "high_margin_divergence": False},
    ]
    excess = benchmark._high_margin_excess(rows, control, alpha=0.05)
    assert excess["candidate_high_margin_divergences"] == 1
    assert excess["candidate_comparisons"] == 3
    assert excess["candidate_high_margin_rows"] == 3
    assert excess["candidate_rows"] == 5
    assert excess["passed"]


def test_high_margin_screen_still_fires_on_distinct_prompts():
    # Deduplicating must not disarm the screen: eight DISTINCT prompts that all
    # diverge are eight observations, and still fail.
    rows = [
        {"prompt_sha256": f"p{i}", "high_margin_divergence": True}
        for i in range(8)
    ]
    control = [
        {"prompt_sha256": f"p{i}", "high_margin_divergence": False}
        for i in range(8)
    ]
    excess = benchmark._high_margin_excess(rows, control, alpha=0.05)
    assert excess["candidate_high_margin_divergences"] == 8
    assert excess["candidate_comparisons"] == 8
    assert excess["p_value"] < excess["alpha"]
    assert not excess["passed"]
