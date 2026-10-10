"""Fail-closed CLI and summary tests for the DLoop A/B producer."""

from types import ModuleType, SimpleNamespace

import pytest

from scripts import dloop_ab


def test_cli_rejects_single_token_before_frozen_preflight_or_model_load(
    tmp_path, monkeypatch
):
    def unexpected_frozen_check(_revision):
        pytest.fail("invalid token count reached frozen preflight")

    monkeypatch.setattr(dloop_ab, "_verify_frozen_git", unexpected_frozen_check)
    with pytest.raises(SystemExit) as error:
        dloop_ab.main(
            [
                "--model",
                "test/model",
                "--source-revision",
                "a" * 40,
                "--runtime-source-sha256",
                "b" * 64,
                "--runtime-native-sha256",
                "c" * 64,
                "--artifact-identity",
                "d" * 64,
                "--arm",
                "fixed=1",
                "--max-tokens",
                "1",
                "--out",
                str(tmp_path / "result.json"),
            ]
        )

    assert error.value.code == 2


def test_cli_requires_explicit_ordinary_reference_before_preflight(
    tmp_path, monkeypatch
):
    def unexpected_frozen_check(_revision):
        pytest.fail("missing ordinary reference reached frozen preflight")

    monkeypatch.setattr(dloop_ab, "_verify_frozen_git", unexpected_frozen_check)
    with pytest.raises(SystemExit) as error:
        dloop_ab.main(
            [
                "--model",
                "test/model",
                "--source-revision",
                "a" * 40,
                "--runtime-source-sha256",
                "b" * 64,
                "--runtime-native-sha256",
                "c" * 64,
                "--artifact-identity",
                "d" * 64,
                "--arm",
                "fixed1=1",
                "--out",
                str(tmp_path / "result.json"),
            ]
        )

    assert error.value.code == 2


@pytest.mark.parametrize(
    "rows",
    [
        [{"decode_tokens": 0, "decode_seconds": 0.01}],
        [
            {"decode_tokens": 3, "decode_seconds": 0.03},
            {"decode_tokens": 0, "decode_seconds": 0.01},
        ],
    ],
)
def test_summary_rejects_early_eos_rows_without_decode_tokens(rows):
    results = [{"arm": "fixed", "pair": 0, "rows": rows}]

    with pytest.raises(ValueError, match="insufficient decode-token count"):
        dloop_ab.summarize(results, ["fixed"], "fixed")


@pytest.mark.parametrize("count", [None, True, -1])
def test_summary_rejects_missing_malformed_or_negative_decode_counters(count):
    rows = [{"decode_tokens": count, "decode_seconds": 0.01}]

    with pytest.raises(ValueError, match="insufficient decode-token count"):
        dloop_ab.summarize(
            [{"arm": "fixed", "pair": 0, "rows": rows}], ["fixed"], "fixed"
        )


def test_summary_calculates_rate_from_positive_decode_tokens():
    rows = [
        {
            "decode_tokens": 2,
            "decode_seconds": 0.04,
            "draft_loop": None,
            "prompt": 0,
            "tokens": [1, 2, 3],
        },
    ]
    result = dloop_ab.summarize(
        [{"arm": "fixed", "pair": 0, "rows": rows}], ["fixed"], "fixed"
    )

    assert result["fixed"]["ms_per_token_median"] == pytest.approx(20.0)


def test_terminal_response_token_is_separate_from_all_tokens_and_continuations():
    from types import SimpleNamespace

    response = SimpleNamespace(
        prompt_cache=["target-cache"], all_tokens=[10, 20, 30], token=40, mtp_state=None
    )
    state = dloop_ab._terminal_state(response)
    assert state["all_tokens"] == [10, 20, 30]
    assert state["terminal_response_token"] == 40
    assert dloop_ab._continuation_request(state) == ([40], [10, 20, 30])
    assert dloop_ab._cold_history(state) == [10, 20, 30, 40]


def test_state_continuation_rejects_missing_terminal_token():
    with pytest.raises(TypeError, match="response.token"):
        dloop_ab._continuation_request({"all_tokens": [10, 20]})


def _install_fake_generator(monkeypatch):
    captured = {}

    class FakeGenerator:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured["constructor"] = kwargs

        def insert(self, prompts, **kwargs):
            captured["prompts"] = prompts
            captured["insert"] = kwargs
            return [91]

        def next(self):
            return None, [SimpleNamespace(uid=91, token=123, finish_reason="length")]

        def close(self):
            captured["closed"] = True

    mlx_module = ModuleType("mlx")
    mlx_module.__path__ = []
    core_module = ModuleType("mlx.core")
    core_module.synchronize = lambda: None
    mlx_module.core = core_module
    monkeypatch.setitem(__import__("sys").modules, "mlx", mlx_module)
    monkeypatch.setitem(__import__("sys").modules, "mlx.core", core_module)

    runtime_module = ModuleType("mlx2.runtime")
    runtime_module.__path__ = []
    generate_module = ModuleType("mlx2.runtime.generate")
    generate_module.BatchGenerator = FakeGenerator
    sample_module = ModuleType("mlx2.runtime.sample_utils")
    sample_module.LaneRNG = lambda seed: ("lane-rng", seed)
    monkeypatch.setitem(__import__("sys").modules, "mlx2.runtime", runtime_module)
    monkeypatch.setitem(
        __import__("sys").modules, "mlx2.runtime.generate", generate_module
    )
    monkeypatch.setitem(
        __import__("sys").modules, "mlx2.runtime.sample_utils", sample_module
    )
    return captured


@pytest.mark.parametrize("config", [None, {"num_draft": 1}])
def test_saved_cache_continuation_actual_insert_feeds_terminal_once(
    monkeypatch, config
):
    captured = _install_fake_generator(monkeypatch)
    target_cache = ["target"]
    mtp_state = (["draft"], "seed")
    state = {
        "all_tokens": [10, 20, 30],
        "terminal_response_token": 40,
        "target_cache": target_cache,
        "mtp_state": mtp_state,
    }

    assert dloop_ab._state_continuation("model", state, config, 4) == [123]
    assert captured["prompts"] == [[40]]
    assert captured["insert"]["all_tokens"] == [[10, 20, 30]]
    assert captured["insert"]["all_tokens"][0][-1] == 30
    assert captured["insert"]["caches"] == [["target"]]
    assert captured["insert"]["caches"][0] is not target_cache
    if config is None:
        assert "mtp_states" not in captured["insert"]
    else:
        assert captured["insert"]["mtp_states"] == [mtp_state]
        assert captured["insert"]["mtp_states"][0] is not mtp_state
    assert captured["closed"] is True


def test_cold_ordinary_continuation_actual_insert_recomputes_terminal_once(
    monkeypatch,
):
    captured = _install_fake_generator(monkeypatch)
    state = {"all_tokens": [10, 20, 30], "terminal_response_token": 40}

    assert dloop_ab._cold_continuation("model", dloop_ab._cold_history(state), 4) == [
        123
    ]
    assert captured["prompts"] == [[10, 20, 30, 40]]
    assert "self_mtp" not in captured["constructor"]
    assert captured["closed"] is True


def test_ordinary_reference_is_distinct_from_fixed1_self_mtp():
    assert dloop_ab.parse_arm("ordinary=ordinary") == (
        "ordinary",
        {"reference": "ordinary_decode"},
    )
    dloop_ab._validate_arms(
        [
            ("ordinary", {"reference": "ordinary_decode"}),
            ("fixed1", {"num_draft": 1}),
        ]
    )
    with pytest.raises(ValueError, match="ordinary reference"):
        dloop_ab._validate_arms(
            [("ordinary", {"reference": "ordinary_decode", "num_draft": 1})]
        )
