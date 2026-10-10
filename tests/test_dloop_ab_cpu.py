"""Fail-closed CLI and summary tests for the DLoop A/B producer."""

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
