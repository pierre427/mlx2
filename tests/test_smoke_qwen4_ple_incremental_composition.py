import json
import os

from scripts import smoke_qwen4_ple_incremental_composition as smoke

REVISION = "a" * 64


def _token_receipt(action):
    return {
        "schema": smoke.TOKENIZER_SCHEMA,
        "implemented": True,
        "qualified": False,
        "selected": True,
        "observed_used": action == "incremental_hit",
        "serving_qualified": False,
        "action": action,
        "exact": True,
        "tokenizer_revision": REVISION,
        "lifecycle_epoch": 1,
        "refusal": None,
    }


def _status(*, serial_calls=0, serial_rows=0, lookups=0, rows=0, unique=0, bytes_read=0):
    return {
        "execution": {
            "ple_tables": [
                {
                    "lookups": lookups,
                    "rows": rows,
                    "unique_rows": unique,
                    "bytes_read": bytes_read,
                    "read_policy": {
                        "schema": smoke.READ_POLICY_SCHEMA,
                        "configured": "adaptive",
                        "selected": True,
                        "qualified": False,
                        "effective": "serial",
                        "phase": "load",
                        "failure": None,
                        "warming": {
                            "enabled": False,
                            "state": "disabled",
                            "refresh_pending": False,
                            "error": None,
                        },
                        "calibration": {
                            "phase": "load",
                            "rows_per_arm": 128,
                            "serial_ns": 100,
                            "pooled_ns": 200,
                            "pooled_workers": 8,
                            "margin_percent": 20,
                            "selected": "serial",
                        },
                        "counts": {
                            "load_calibrations": 1,
                            "warmed_calibrations": 0,
                            "warm_signals": 0,
                            "warm_refreshes": 0,
                            "warm_refresh_failures": 0,
                            "foreground_serial_calls": serial_calls,
                            "foreground_serial_rows": serial_rows,
                            "foreground_pooled_calls": 0,
                            "foreground_pooled_rows": 0,
                        },
                        "last_receipt": (
                            {
                                "event": "calibration",
                                "phase": "load",
                                "selected": "serial",
                            }
                            if not serial_calls
                            else {
                                "event": "foreground_read",
                                "configured": "adaptive",
                                "selected_arm": "serial",
                                "actual_arm": "serial",
                                "workers": 1,
                                "rows": serial_rows,
                                "phase": "load",
                                "warm_refresh_pending": False,
                            }
                        ),
                    },
                }
            ]
        },
        "incremental_tokenizer_cache": {
            "schema": smoke.TOKENIZER_SCHEMA,
            "implemented": True,
            "qualified": False,
            "selected": True,
            "observed_used": bool(serial_calls),
            "serving_qualified": False,
            "default": "off",
            "tokenizer_revision": REVISION,
            "lifecycle_epoch": 1,
            "refusal": None,
            "cold_validations": int(bool(serial_calls)),
            "incremental_hits": int(bool(serial_calls)),
        },
    }


def test_dry_run_prints_exact_default_off_composition(capsys):
    code = smoke.main(["--model", "/artifact/qwen", "--dry-run"])
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["server_environment"] == {
        "PYTHONPATH": str(smoke.ROOT / "src"),
    }
    assert plan["execution_policy"] == {
        "ple_read_policy": "adaptive",
        "ple_adaptive_warm": False,
    }
    command = plan["server_command"]
    assert command[command.index("--execution-policy") + 1] == str(smoke.POLICY_PATH)
    assert command[command.index("--max-lanes") + 1] == "1"
    assert command[command.index("--max-inflight") + 1] == "1"
    assert command[command.index("--incremental-tokenizer-cache-entries") + 1] == "4"
    assert plan["safety"]["launches_server"] is False
    assert plan["safety"]["whole_table_warming"] is False
    assert plan["will_send_requests"] is False


def test_request_pair_suppresses_apc_but_grows_chat():
    cold, grown = smoke.request_bodies("model", 4096)
    for body in (cold, grown):
        assert body["skip_writing_prefix_cache"] is True
        assert body["max_tokens"] == 1
        assert body["temperature"] == 0
    assert grown["messages"][: len(cold["messages"])] == cold["messages"]
    assert len(grown["messages"]) == len(cold["messages"]) + 2


def test_flash_next_policy_owns_adaptive_read_and_warm_controls(monkeypatch, tmp_path):
    from mlx2.adapters.flash_next import configure_environment
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_READ_POLICY", "serial")
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_ADAPTIVE_WARM", "1")
    default = configure_environment(tmp_path, FlashNextPolicy())
    assert "MLX_QWEN4_PLE_NVME_READ_POLICY" not in default
    assert "MLX_QWEN4_PLE_NVME_ADAPTIVE_WARM" not in default
    assert "MLX_QWEN4_PLE_NVME_READ_POLICY" not in os.environ
    selected_policy = FlashNextPolicy.from_mapping(
        {"ple_read_policy": "adaptive", "ple_adaptive_warm": False}
    )
    assert selected_policy.as_dict()["ple_read_policy"] == "adaptive"
    assert selected_policy.as_dict()["ple_adaptive_warm"] is False
    selected = configure_environment(tmp_path, selected_policy)
    assert selected["MLX_QWEN4_PLE_NVME_READ_POLICY"] == "adaptive"
    assert selected["MLX_QWEN4_PLE_NVME_ADAPTIVE_WARM"] == "0"


def test_flash_next_policy_rejects_invalid_ple_controls():
    import pytest

    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    with pytest.raises(ValueError, match="ple_read_policy"):
        FlashNextPolicy.from_mapping({"ple_read_policy": "maybe"})
    with pytest.raises(ValueError, match="ple_adaptive_warm"):
        FlashNextPolicy.from_mapping({"ple_adaptive_warm": 0})


def test_evaluate_accepts_exact_status_deltas_and_receipts():
    before = _status()
    after = _status(
        serial_calls=6,
        serial_rows=9000,
        lookups=6,
        rows=9100,
        unique=9000,
        bytes_read=900000,
    )
    cold = {"mlx2": {"prompt_tokenization": _token_receipt("ordinary_full_validation")}}
    grown = {"mlx2": {"prompt_tokenization": _token_receipt("incremental_hit")}}
    assert smoke.evaluate(before, cold, grown, after) == []
    assert smoke._status_advanced(before, after) is True


def test_evaluate_rejects_warming_and_cross_arm_activity():
    before = _status()
    after = _status(
        serial_calls=2,
        serial_rows=100,
        lookups=2,
        rows=100,
        unique=80,
        bytes_read=8000,
    )
    policy = after["execution"]["ple_tables"][0]["read_policy"]
    policy["warming"]["enabled"] = True
    policy["counts"]["foreground_pooled_calls"] = 1
    policy["counts"]["foreground_pooled_rows"] = 5
    cold = {"mlx2": {"prompt_tokenization": _token_receipt("ordinary_full_validation")}}
    grown = {"mlx2": {"prompt_tokenization": _token_receipt("incremental_hit")}}
    failures = smoke.evaluate(before, cold, grown, after)
    assert any("warming.enabled" in failure for failure in failures)
    assert any("unselected foreground_pooled_calls" in failure for failure in failures)


def test_live_mode_refuses_without_explicit_request_route_ownership(capsys):
    code = smoke.main(
        [
            "--model",
            "/artifact/qwen",
            "--live-url",
            "http://127.0.0.1:8398",
        ]
    )
    assert code == 2
    assert "--i-own-request-route" in capsys.readouterr().err
