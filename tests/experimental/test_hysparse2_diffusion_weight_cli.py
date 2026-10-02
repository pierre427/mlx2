"""Reject invalid auxiliary objective weights before model/training allocation."""

import pytest

from mlx2.experimental.hysparse2 import train


@pytest.mark.parametrize("weight", ["-1", "nan", "inf", "-inf"])
def test_invalid_diffusion_weight_never_enters_training(tmp_path, monkeypatch, weight):
    def poison(*args):
        pytest.fail("invalid diffusion weight reached model training")

    monkeypatch.setattr(train, "train", poison)
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as error:
        train.main(
            [
                "--smoke",
                "--device",
                "cpu",
                "--output",
                str(output),
                "--diffusion-weight=" + weight,
            ]
        )
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("weight", ["0", "0.2", "1"])
def test_valid_diffusion_weight_forwarded_unchanged(tmp_path, monkeypatch, weight):
    calls = []
    monkeypatch.setattr(
        train, "train", lambda args, config: calls.append(args.diffusion_weight) or 0
    )
    assert (
        train.main(
            [
                "--smoke",
                "--device",
                "cpu",
                "--output",
                str(tmp_path / "unused"),
                "--diffusion-weight",
                weight,
            ]
        )
        == 0
    )
    assert calls == [float(weight)]
