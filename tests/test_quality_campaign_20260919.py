import base64
import importlib.util
import io
import json
import os
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

RUN = Path(__file__).parents[1] / "qualification/runs/quality-campaign-20260919"
ROOT = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(f"quality_campaign_{name}", RUN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/v1/messages/count_tokens":
            text = body["messages"][0]["content"]
            payload = {"input_tokens": len(text.split()) + 5}
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps(payload).encode()); return
        if self.path == "/v1/chat/completions" and body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n')
            self.wfile.write(b"data: [DONE]\n\n"); return
        payload = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 1},
            "mlx2": {"cached_tokens": 0},
        }
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps(payload).encode())


def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    return server, thread


class StartupHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _send(self, status, payload):
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        elapsed = time.monotonic() - self.server.started
        loading = elapsed < self.server.loading_seconds
        if self.path == "/health":
            if self.server.mode == "error":
                self._send(
                    503,
                    {
                        "status": "unavailable",
                        "error": "RuntimeError: adapter load failed",
                    },
                )
            elif self.server.mode == "draining":
                self._send(503, {"status": "draining"})
            elif loading:
                self._send(503, {"status": "unavailable", "error": None})
            else:
                self._send(200, {"status": "ok", "error": None})
            return
        if self.path == "/v1/status":
            self._send(
                200,
                {
                    "state": "loading" if loading else "ready",
                    "healthy": not loading,
                    "error": None,
                    "quiesce": {"state": "serving"},
                },
            )
            return
        self._send(404, {"error": "not found"})


def startup_server(*, mode="loading", loading_seconds=0.0):
    server = ThreadingHTTPServer(("127.0.0.1", 0), StartupHandler)
    server.mode = mode
    server.loading_seconds = loading_seconds
    server.started = time.monotonic()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_feature_smoke_real_http_and_sse_check_logic():
    feature = load("feature_smoke")
    server, thread = fake_server()
    try:
        http = feature.HTTP(f"http://127.0.0.1:{server.server_port}", 2)
        plain = feature._chat(http, "hello")
        streamed = feature._chat(http, "hello", stream=True)
        assert feature._ok_text(plain).passed
        assert plain["request"] == {
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {
                "model": "campaign",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0,
                "max_tokens": 96,
                "enable_thinking": False,
            },
        }
        assert feature._stream_text(streamed) == "OK"
        assert "[DONE]" in streamed["events"]
    finally:
        server.shutdown(); server.server_close(); thread.join()


@pytest.mark.parametrize(
    ("context_cap", "expected_budget"),
    [(16_384, 16_128), (131_072, 32_000)],
)
def test_messages_large_output_smoke_respects_served_context(
    context_cap, expected_budget
):
    feature = load("feature_smoke")

    class FakeHTTP:
        def __init__(self):
            self.budget = None

        def get(self, path):
            assert path == "/v1/status"
            return {"status": 200, "body": {"max_context": context_cap}}

        def post(self, path, body, *, stream=False):
            assert path == "/v1/messages" and not stream
            self.budget = body["max_tokens"]
            return {"status": 200 if self.budget <= context_cap - 256 else 400}

    class OnlyLargeBudgetCheck:
        def __init__(self):
            self.http = FakeHTTP()
            self.args = SimpleNamespace(capabilities=set())
            self.result = None

        def check(self, name, function=None, **_kwargs):
            if name in {"messages_max_tokens_32000", "messages_max_tokens_within_context"}:
                self.result = function()

    matrix = OnlyLargeBudgetCheck()
    feature.messages_checks(matrix, applies=True)
    assert matrix.result is not None and matrix.result.passed
    assert matrix.http.budget == expected_budget


def test_constrained_tool_smoke_uses_a_finite_exact_argument_language():
    feature = load("feature_smoke")
    from mlx2.runtime.tool_parsers.qwen3_coder import constrained_tool_grammar
    from mlx2.structured_output import compile_constraint

    grammar = constrained_tool_grammar(
        feature.CONSTRAINED_WEATHER_TOOL,
        "required",
        parallel_tool_calls=False,
    )
    expected = (
        "<tool_call>\n<function=weather>\n<parameter=city>\nToronto\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    assert compile_constraint(grammar=grammar).fullmatch(expected) is not None
    assert compile_constraint(grammar=grammar).fullmatch(
        expected.replace("Toronto", "Toronto Toronto")
    ) is None


def test_feature_matrix_pass_fail_skip_and_timeout(tmp_path):
    feature = load("feature_smoke")
    args = SimpleNamespace(raw_dir=tmp_path, check_timeout=0.02)
    matrix = feature.Matrix(args, None)
    matrix.check("pass", lambda: feature.Outcome(True, "good", {"ok": True}))
    matrix.check("fail", lambda: feature.Outcome(False, "bad", {"ok": False}))
    matrix.check("skip", applies=False, reason="not declared")
    matrix.check("timeout", lambda: time.sleep(0.2))
    assert [row["status"] for row in matrix.rows] == ["PASS", "FAIL", "SKIP", "FAIL"]
    assert "timeout" in matrix.rows[-1]["reason"]
    assert json.loads((tmp_path / "000-pass.json").read_text()) == {"ok": True}


def test_muse_skips_thinking_budget_checks_without_a_close_marker():
    feature = load("feature_smoke")
    muse = next(model for model in load("campaign_config").MODELS if model.name == "muse")
    assert "reasoning" in muse.capabilities
    assert "thinking-deferral" not in muse.capabilities
    assert not feature.supports_budgeted_thinking(muse.capabilities)
    assert feature.supports_budgeted_thinking(
        {"reasoning", "thinking-deferral"}
    )


def _load_sanity(monkeypatch, status, *, think_override=None):
    payload = json.dumps(status).encode()
    monkeypatch.setattr(sys, "argv", ["sanity_20x20.py", "http://fixture", "out.json"])
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )
    if think_override is None:
        monkeypatch.delenv("SANITY_THINK", raising=False)
    else:
        monkeypatch.setenv("SANITY_THINK", think_override)
    return load("sanity_20x20")


def test_sanity_uses_the_thinking_default_models_declared_allowance(monkeypatch):
    sanity = _load_sanity(
        monkeypatch,
        {"thinking_default": True, "thinking_allowance_tokens": 4096},
    )
    assert sanity.THINK is True
    assert sanity.THINKING_BUDGET_TOKENS == 4096
    assert sanity.user("needle", max_tokens=32)["max_tokens"] == 4128
    requests = {name: body for name, body, _grade in sanity.tasks(0)}
    assert requests["system_multi_turn"]["max_tokens"] == 4144


def test_sanity_does_not_change_non_thinking_default_requests(monkeypatch):
    sanity = _load_sanity(
        monkeypatch,
        {"thinking_default": False, "thinking_allowance_tokens": 4096},
    )
    assert sanity.THINK is False
    assert sanity.user("answer", max_tokens=32) == {
        "messages": [{"role": "user", "content": "answer"}],
        "temperature": 0,
        "max_tokens": 32,
        "enable_thinking": False,
        "reasoning_effort": "none",
    }


@pytest.mark.parametrize("declared", [None, 1024])
def test_sanity_retains_its_allowance_floor(monkeypatch, declared):
    sanity = _load_sanity(
        monkeypatch,
        {"thinking_default": True, "thinking_allowance_tokens": declared},
    )
    assert sanity.THINKING_BUDGET_TOKENS == 2048
    assert sanity.user("answer", max_tokens=32)["max_tokens"] == 2080


def test_speculative_fixed_prompt_oracles_allow_format_not_wrong_answers():
    feature = load("feature_smoke")
    assert feature.fixed_prompt_correct(feature.FIXED_PROMPTS[0], "CAMPAIGN_READY.")
    assert feature.fixed_prompt_correct(feature.FIXED_PROMPTS[1], "95")
    assert feature.fixed_prompt_correct(feature.FIXED_PROMPTS[2], "Ottawa.")
    assert feature.fixed_prompt_correct(feature.FIXED_PROMPTS[3], "Eau")
    assert not feature.fixed_prompt_correct(feature.FIXED_PROMPTS[1], "96")


def test_multimodal_smoke_image_meets_minimum_dimensions():
    feature = load("feature_smoke")
    encoded = feature._png_data_url().partition(",")[2]
    png = base64.b64decode(encoded)
    width, height = struct.unpack(">II", png[16:24])
    assert (width, height) == (8, 8)


def test_feature_contract_checks_accept_fail_closed_single_call_and_real_suspend():
    feature = load("feature_smoke")
    rejected = {
        "status": 502,
        "body": {
            "error": {
                "message": (
                    "parallel_tool_calls:false permits at most one tool call"
                )
            }
        },
    }
    assert feature._single_tool_enforced(rejected).passed
    rejected["body"]["error"]["message"] = (
        "model emitted parallel calls while parallel_tool_calls was false"
    )
    assert feature._single_tool_enforced(rejected).passed

    def chat(cached):
        return {
            "status": 200,
            "body": {
                "choices": [{"message": {"content": "ADMIN_OK"}}],
                "mlx2": {"cached_tokens": cached},
            },
        }

    class AdminHTTP:
        def __init__(self):
            self.chat_replies = iter((chat(0), chat(128), chat(128)))
            self.resume_body = None

        def post(self, path, body, **_kwargs):
            if path == "/v1/chat/completions":
                return next(self.chat_replies)
            if path == "/v1/admin/quiesce":
                return {"status": 202, "body": {"state": "draining"}}
            if path == "/v1/admin/resume":
                self.resume_body = body
                return {"status": 202, "body": {"state": "serving"}}
            raise AssertionError(path)

        def get(self, path):
            if path == "/v1/admin/state":
                return {"status": 200, "body": {"state": "suspended"}}
            if path.startswith("/v1/apc/sessions/"):
                return {"status": 200, "body": {"state": "resident"}}
            raise AssertionError(path)

    http = AdminHTTP()
    outcome = feature._admin_suspend_resume(http, timeout=1)
    assert outcome.passed
    assert http.resume_body == {
        "prefetch_sessions": [
            {
                "tenant": "default",
                "session_id": "quality-campaign-admin-suspend-190919",
            }
        ]
    }


def test_feature_fly_check_uses_receipt_and_neutral_penalties():
    feature = load("feature_smoke")

    class FlyHTTP:
        request_body = None

        def get(self, _path):
            return {"status": 200, "body": {"counts": {}}}

        def post(self, path, body, **_kwargs):
            assert path == "/v1/chat/completions"
            self.request_body = body
            return {
                "status": 200,
                "body": {
                    "mlx2": {
                        "mtp": {"verification": "fly", "fly_disabled": False}
                    }
                },
            }

    http = FlyHTTP()
    assert feature._fly_verification(http).passed
    assert http.request_body["repetition_penalty"] == 1.0
    assert http.request_body["presence_penalty"] == 0.0
    assert http.request_body["frequency_penalty"] == 0.0


def test_ladder_math_and_peak_memory_are_deterministic():
    ladder = load("ladder")
    metrics = ladder.calculate_metrics(
        prompt_tokens=1000, completion_tokens=5, wall_seconds=3.0,
        ttft_seconds=1.0, token_times=(1.0, 1.5, 2.0, 2.5, 3.0),
        receipt={"prefill_seconds": 0.5},
    )
    assert metrics["prefill_tokens_per_second"] == 2000
    assert metrics["decode_tokens_per_second"] == 2
    path, value = ladder.extract_peak_memory({"memory": {"peak_allocated_bytes": 200}, "other_bytes": 9})
    assert (path, value) == ("memory.peak_allocated_bytes", 200)
    assert ladder.percentile([1, 3], 0.5) == 2


def test_ladder_prompt_calibration_uses_fake_server_tokenizer():
    ladder = load("ladder")
    server, thread = fake_server()
    try:
        client = ladder.Client(f"http://127.0.0.1:{server.server_port}", 2)
        text, actual = ladder.calibrate_prompt(client, 128, "NEEDLE_X", "nonce")
        assert "NEEDLE_X" in text
        assert 100 <= actual <= 128
        assert client.count(text) == actual
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_campaign_matrix_has_nine_models_and_expected_stage_counts():
    config = load("campaign_config")
    assert len(config.MODELS) == 9
    assert len(config.stage_names("smoke")) == 18
    assert len(config.stage_names("sanity")) == 18
    assert len(config.stage_names("ladder")) == 9
    xing = next(model for model in config.MODELS if model.name == "xing")
    assert [route.name for route in xing.routes] == ["ordinary", "mtp1", "prompt-lookup"]
    north = next(model for model in config.MODELS if model.name == "north")
    args = config.server_args(north, north.routes[0], "ladder")
    assert args[args.index("--max-context") + 1] == "500000"
    muse = next(model for model in config.MODELS if model.name == "muse")
    assert "thinking-deferral" not in muse.capabilities
    assert "thinking-deferral" in north.capabilities


def test_opt_in_interior_checkpoint_policy_matches_selected_route(monkeypatch):
    config, _feature, runner = load_runner(monkeypatch)
    for model in config.MODELS:
        for route in model.routes:
            policy = json.loads(config.policy_path(route.opt_in_policy).read_text())
            assert ("apc_interior_checkpoints" in policy) is route.apc_interior
            command = runner.feature_command(
                model,
                route,
                config.RUN / "results" / "fixture",
                "opt-in",
            )
            declared = [
                command[index + 1]
                for index, value in enumerate(command)
                if value == "--capability"
            ]
            assert ("apc-interior" in declared) is route.apc_interior


def test_campaign_persistence_paths_match_and_use_engine_startup_validation(
    tmp_path, monkeypatch
):
    config = load("campaign_config")
    feature = load("feature_smoke")
    monkeypatch.setitem(sys.modules, "campaign_config", config)
    monkeypatch.setitem(sys.modules, "feature_smoke", feature)
    runner = load("run_campaign")
    model = config.MODELS[0]
    route = model.routes[0]
    command = runner.common_server_command(
        model, route, "smoke", tmp_path, persistence=True
    )
    server_args = command[4:]
    cache_dir = server_args[server_args.index("--cache-dir") + 1]
    persist_dir = server_args[server_args.index("--apc-persist-dir") + 1]

    assert cache_dir == persist_dir
    parsed = config.validate_server_arguments(server_args)
    assert parsed.apc_persist_on_shutdown is True

    # Persistence disabled permits an ordinary cache directory without an APC
    # persistence directory or shutdown flush.
    disabled = config.server_args(model, route, "smoke") + [
        "--cache-dir",
        str(tmp_path / "cache-only"),
    ]
    parsed_disabled = config.validate_server_arguments(disabled)
    assert parsed_disabled.apc_persist_dir is None
    assert parsed_disabled.apc_persist_on_shutdown is False

    base = runner.common_server_command(model, route, "smoke", tmp_path)
    parsed_base = config.validate_server_arguments(base[4:])
    assert parsed_base.cache_dir == str(tmp_path / "cache")
    assert parsed_base.apc_persist_dir is None
    assert parsed_base.apc_persist_on_shutdown is False

    smoke = runner.stage_specs("smoke")[0]
    by_name = {cycle["name"]: cycle["server"] for cycle in smoke["cycles"]}
    assert "--apc-persist-on-shutdown" not in by_name["base"]
    assert "--apc-persist-on-shutdown" not in by_name["opt-in"]
    assert "--apc-persist-on-shutdown" in by_name["persist-seed"]
    assert "--apc-persist-on-shutdown" in by_name["persist-rescan"]

    mismatched = list(server_args)
    mismatched[mismatched.index("--apc-persist-dir") + 1] = str(
        tmp_path / "different"
    )
    with pytest.raises(ValueError, match="must name the same directory"):
        config.validate_server_arguments(mismatched)


def test_campaign_preflight_always_rebinds_to_served_root(tmp_path, monkeypatch):
    _config, _feature, runner = load_runner(monkeypatch)
    served = tmp_path / "served"
    served.mkdir()
    receipt = tmp_path / "preflight.json"
    receipt.write_text('{"source":"stale"}\n')
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        receipt.write_text('{"source":"served"}\n')
        return SimpleNamespace(returncode=0, stdout="fresh receipt", stderr="")

    campaign = SimpleNamespace(state={}, save=lambda: None)
    monkeypatch.setattr(runner, "ROOT", served)
    monkeypatch.setattr(runner, "PREFLIGHT", receipt)
    monkeypatch.setattr(runner.subprocess, "run", run)

    runner.generate_preflight_receipt(campaign)

    assert observed["cwd"] == served
    assert observed["env"]["PYTHONPATH"] == "src"
    assert observed["command"][-2:] == ["--output", str(receipt)]
    assert json.loads(receipt.read_text()) == {"source": "served"}
    assert campaign.state["preflight"] == {
        "returncode": 0,
        "root": str(served),
        "replaced_existing": True,
        "tail": "fresh receipt",
    }


def test_campaign_preflight_failure_stops_before_stages(tmp_path, monkeypatch):
    _config, _feature, runner = load_runner(monkeypatch)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PREFLIGHT", tmp_path / "preflight.json")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=7, stdout="", stderr="identity failure"
        ),
    )
    campaign = SimpleNamespace(state={}, save=lambda: None)

    with pytest.raises(RuntimeError, match="identity failure"):
        runner.generate_preflight_receipt(campaign)
    assert campaign.state["preflight"]["returncode"] == 7


def load_runner(monkeypatch):
    config = load("campaign_config")
    feature = load("feature_smoke")
    monkeypatch.setitem(sys.modules, "campaign_config", config)
    monkeypatch.setitem(sys.modules, "feature_smoke", feature)
    return config, feature, load("run_campaign")


def test_multimodal_pythonpath_is_stage_scoped(monkeypatch):
    config, _feature, runner = load_runner(monkeypatch)
    gemma = next(model for model in config.MODELS if model.name == "gemma3n")
    qwen = next(model for model in config.MODELS if model.name == "qwen36")

    multimodal = runner.stage_environment(gemma)["PYTHONPATH"].split(os.pathsep)
    text = runner.stage_environment(qwen)["PYTHONPATH"].split(os.pathsep)
    assert multimodal == [str(config.MLX_VLM_ROOT), "src"]
    assert str(config.MLX_VLM_ROOT) not in text
    assert text == ["src"]


def test_mlx_vlm_runtime_validation_checks_revision_clean_detached_and_origin(
    tmp_path, monkeypatch
):
    config = load("campaign_config")
    checkout = tmp_path / "mlx-vlm"
    package = checkout / "mlx_vlm"
    package.mkdir(parents=True)
    origin = package / "__init__.py"
    origin.write_text("")
    monkeypatch.setattr(config, "MLX_VLM_ROOT", checkout.resolve())
    state = {
        "dirty": "",
        "revision": config.MLX_VLM_REVISION,
        "symbolic_returncode": 1,
        "origin": str(origin),
    }

    def command_output(command, **_kwargs):
        command = [str(item) for item in command]
        if command[0] == "git" and "--show-toplevel" in command:
            return SimpleNamespace(returncode=0, stdout=str(checkout), stderr="")
        if command[0] == "git" and command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=state["revision"], stderr="")
        if command[0] == "git" and "status" in command:
            return SimpleNamespace(returncode=0, stdout=state["dirty"], stderr="")
        if command[0] == "git" and "symbolic-ref" in command:
            return SimpleNamespace(
                returncode=state["symbolic_returncode"], stdout="refs/heads/main", stderr=""
            )
        return SimpleNamespace(returncode=0, stdout=state["origin"], stderr="")

    monkeypatch.setattr(config, "_command_output", command_output)
    receipt = config.verify_mlx_vlm_runtime(env={})
    assert receipt["revision"] == config.MLX_VLM_REVISION
    assert receipt["clean"] is receipt["detached"] is True
    assert receipt["import_origin"] == str(origin.resolve())

    state["dirty"] = "?? local-edit.py"
    with pytest.raises(RuntimeError, match="checkout is dirty"):
        config.verify_mlx_vlm_runtime(env={})
    state["dirty"] = ""

    state["revision"] = "0" * 40
    with pytest.raises(RuntimeError, match="revision"):
        config.verify_mlx_vlm_runtime(env={})
    state["revision"] = config.MLX_VLM_REVISION

    state["symbolic_returncode"] = 0
    with pytest.raises(RuntimeError, match="not detached"):
        config.verify_mlx_vlm_runtime(env={})
    state["symbolic_returncode"] = 1

    outside = tmp_path / "other" / "mlx_vlm" / "__init__.py"
    outside.parent.mkdir(parents=True)
    outside.write_text("")
    state["origin"] = str(outside)
    with pytest.raises(RuntimeError, match="expected under"):
        config.verify_mlx_vlm_runtime(env={})


def test_multimodal_stage_fails_closed_when_runtime_validation_fails(monkeypatch):
    _config, _feature, runner = load_runner(monkeypatch)
    stage = next(
        stage
        for stage in runner.stage_specs("smoke")
        if stage["model"].name == "gemma3n"
    )
    campaign = runner.Campaign.__new__(runner.Campaign)
    campaign.state = {"stages": {}, "history": []}
    campaign.save = lambda: None
    monkeypatch.setattr(
        runner,
        "verify_mlx_vlm_runtime",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("wrong revision")),
    )
    campaign.run_stage(stage)
    entry = campaign.state["stages"][stage["name"]]
    assert entry["state"] == "failed"
    assert "wrong revision" in entry["reason"]


def test_feature_and_ladder_checks_abort_on_engine_health_error(tmp_path):
    feature = load("feature_smoke")
    feature_health = iter((None, "adapter load failed"))
    matrix = feature.Matrix(
        SimpleNamespace(raw_dir=tmp_path / "raw", check_timeout=30),
        SimpleNamespace(health_error=lambda: next(feature_health, "adapter load failed")),
    )
    started = time.monotonic()
    matrix.check("hung", lambda: time.sleep(30))
    assert time.monotonic() - started < 2
    assert "adapter load failed" in matrix.rows[0]["reason"]

    ladder = load("ladder")
    client = ladder.Client("http://127.0.0.1:1", 30)
    ladder_health = iter((None, "worker exited"))
    client.health_error = lambda: next(ladder_health, "worker exited")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="worker exited"):
        client._guarded(lambda: time.sleep(30))
    assert time.monotonic() - started < 2


def test_feature_smoke_saves_structured_failure_receipt(tmp_path):
    feature = load("feature_smoke")
    reply = {
        "status": 502,
        "body": {
            "error": {"message": "structured output failed closed"},
            "mlx2": {
                "structured_output_failure": {
                    "schema": "mlx2.structured-output-dead-end.v1"
                }
            },
        },
        "request": {"method": "POST", "path": "/v1/chat/completions"},
    }
    matrix = feature.Matrix(
        SimpleNamespace(raw_dir=tmp_path / "raw", check_timeout=1),
        None,
    )
    matrix.check("json_object", lambda: feature._structured_json(reply))
    assert matrix.rows[0]["status"] == "FAIL"
    saved = json.loads(Path(matrix.rows[0]["raw"]).read_text())
    assert saved == reply


def test_campaign_step_aborts_when_server_process_exits(tmp_path, monkeypatch):
    _config, _feature, runner = load_runner(monkeypatch)
    campaign = runner.Campaign.__new__(runner.Campaign)
    campaign.save = lambda: None
    entry = {"steps": {}}
    server = SimpleNamespace(poll=lambda: 17)
    started = time.monotonic()
    passed = campaign.run_step(
        entry,
        "hung",
        [sys.executable, "-c", "import time; time.sleep(30)"],
        30,
        tmp_path / "hung.log",
        server=server,
    )
    assert not passed
    assert time.monotonic() - started < 2
    assert entry["steps"]["hung"]["server_failure"] is True
    assert "server exited 17" in entry["steps"]["hung"]["reason"]


def test_campaign_health_preserves_explicit_engine_error(monkeypatch):
    _config, _feature, runner = load_runner(monkeypatch)
    response = urllib.error.HTTPError(
        runner.BASE_URL + "/health",
        503,
        "unavailable",
        {},
        io.BytesIO(b'{"status":"unavailable","error":"adapter load failed"}'),
    )

    def fail(*_args, **_kwargs):
        raise response

    monkeypatch.setattr(runner.urllib.request, "urlopen", fail)
    state = runner.Campaign.health_state()
    assert state["ready"] is False
    assert state["error"] == "adapter load failed"


def test_campaign_startup_waits_through_loading_503_then_becomes_ready(monkeypatch):
    _config, _feature, runner = load_runner(monkeypatch)
    server, thread = startup_server(loading_seconds=2.0)
    monkeypatch.setattr(runner, "BASE_URL", f"http://127.0.0.1:{server.server_port}")
    campaign = runner.Campaign.__new__(runner.Campaign)
    process = SimpleNamespace(poll=lambda: None)
    started = time.monotonic()
    try:
        result = campaign.wait_for_server(
            process, timeout=5, poll_seconds=0.05
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result["ready"] is True
    assert result["health"]["status"] == "ok"
    assert result["health"]["error"] is None
    assert time.monotonic() - started >= 2.0


def test_campaign_startup_fails_immediately_on_engine_error(monkeypatch):
    _config, _feature, runner = load_runner(monkeypatch)
    server, thread = startup_server(mode="error")
    monkeypatch.setattr(runner, "BASE_URL", f"http://127.0.0.1:{server.server_port}")
    campaign = runner.Campaign.__new__(runner.Campaign)
    process = SimpleNamespace(poll=lambda: None)
    started = time.monotonic()
    try:
        result = campaign.wait_for_server(process, timeout=30, poll_seconds=0.05)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result["ready"] is False
    assert "adapter load failed" in result["reason"]
    assert time.monotonic() - started < 2


def test_campaign_health_does_not_treat_draining_as_engine_error(monkeypatch):
    _config, _feature, runner = load_runner(monkeypatch)
    server, thread = startup_server(mode="draining")
    monkeypatch.setattr(runner, "BASE_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        health = runner.Campaign.health_state()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert health["ready"] is False
    assert health["status"] == "draining"
    assert health["error"] is None


def test_campaign_startup_fails_before_health_poll_when_process_exits(monkeypatch):
    _config, _feature, runner = load_runner(monkeypatch)
    campaign = runner.Campaign.__new__(runner.Campaign)
    monkeypatch.setattr(
        campaign,
        "health_state",
        lambda: (_ for _ in ()).throw(AssertionError("health must not be polled")),
    )
    result = campaign.wait_for_server(
        SimpleNamespace(poll=lambda: 23), timeout=30, poll_seconds=0.05
    )
    assert result["ready"] is False
    assert result["reason"] == "server exited 23"


def test_scripts_and_campaign_have_no_ruff_f821():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--select",
            "F821",
            "scripts",
            str(RUN.relative_to(ROOT)),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_scripts_and_campaign_have_no_ruff_f_violations():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--select",
            "F",
            "scripts",
            str(RUN.relative_to(ROOT)),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
