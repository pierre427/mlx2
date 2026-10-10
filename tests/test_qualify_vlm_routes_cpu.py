from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import qualify_vlm_routes as dispatcher
from scripts.qualify_vlm_routes import FAMILIES, artifact_sha256, campaign_spec


def artifact(tmp_path, family):
    path = tmp_path / family
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": family}))
    (path / "weights.safetensors").write_bytes(b"fixture")
    return path


def make_spec(tmp_path, family="gemma3n"):
    return campaign_spec(
        family,
        artifact=artifact(tmp_path, family),
        reference_revision=FAMILIES[family]["revision"],
        reference_source_sha256="a" * 64,
        producer_sha256="c" * 64,
        runtime={"python": "3.12", "source": "b" * 64},
        settings={"route": "ordinary", "cache": "apcv2", "max_lanes": 1},
    )


def test_campaign_spec_binds_family_source_artifact_runtime_settings_and_features(
    tmp_path,
):
    spec = make_spec(tmp_path)
    assert spec["family"] == "gemma3n"
    assert spec["reference_revision"] == FAMILIES["gemma3n"]["revision"]
    assert spec["reference_source_sha256"] == "a" * 64
    assert spec["producer_sha256"] == "c" * 64
    assert spec["artifact_sha256"] == artifact_sha256(Path(spec["artifact"]))
    assert spec["modalities"] == ["image", "video", "audio"]
    assert set(spec["required_features"]) == {
        "ordinary_reference",
        "continuous_batch",
        "encoder_batching",
        "image",
        "video",
        "audio",
        "prefix_reuse",
        "replay",
        "receipts",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_revision", "0" * 40),
        ("reference_source_sha256", "bad"),
        ("producer_sha256", "bad"),
    ],
)
def test_campaign_spec_refuses_unbound_source_identity(tmp_path, field, value):
    path = artifact(tmp_path, "gemma3n")
    args = {
        "family": "gemma3n",
        "artifact": path,
        "reference_revision": FAMILIES["gemma3n"]["revision"],
        "reference_source_sha256": "a" * 64,
        "producer_sha256": "c" * 64,
        "runtime": {"python": "3.12"},
        "settings": {"route": "ordinary"},
    }
    args[field] = value
    with pytest.raises(ValueError):
        campaign_spec(**args)


def test_missing_artifact_and_wrong_family_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        campaign_spec(
            "gemma3n",
            artifact=tmp_path / "absent",
            reference_revision=FAMILIES["gemma3n"]["revision"],
            reference_source_sha256="a" * 64,
            producer_sha256="c" * 64,
            runtime={"python": "3.12"},
            settings={"route": "ordinary"},
        )
    wrong = artifact(tmp_path, "gemma4")
    with pytest.raises(ValueError, match="model_type"):
        campaign_spec(
            "gemma3n",
            artifact=wrong,
            reference_revision=FAMILIES["gemma3n"]["revision"],
            reference_source_sha256="a" * 64,
            producer_sha256="c" * 64,
            runtime={"python": "3.12"},
            settings={"route": "ordinary"},
        )


def test_family_modality_contracts_are_explicit():
    assert FAMILIES["gemma3n"]["modalities"] == ("image", "video", "audio")
    assert FAMILIES["gemma4"]["modalities"] == ("image", "video")
    assert FAMILIES["minicpmo"]["modalities"] == ("image", "audio")
    assert FAMILIES["smolvlm"]["producer"] == "qualify_media_serving.py"


def test_legacy_producer_runs_in_fresh_exact_source_environment(tmp_path, monkeypatch):
    checkout = tmp_path / "mlx-vlm-8a5e"
    (checkout / "mlx_vlm").mkdir(parents=True)
    monkeypatch.setattr(dispatcher, "LEGACY_SOURCE_ROOT", checkout)
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    producer = repo / "scripts" / "producer.py"
    producer.parent.mkdir()
    producer.write_text("# test producer")
    artifact_path = tmp_path / "artifact"
    artifact_path.mkdir()
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs["env"]
        captured["cwd"] = kwargs["cwd"]
        return SimpleNamespace(returncode=0, stdout='{"passed": true}', stderr="")

    monkeypatch.setattr(dispatcher.subprocess, "run", fake_run)
    report, code = dispatcher.run_legacy_sourcebound_producer(
        repo, producer, artifact_path
    )
    assert report == {"passed": True}
    assert code == 0
    assert captured["env"]["PYTHONPATH"].split(":") == [
        str(checkout.resolve()),
        str((repo / "src").resolve()),
    ]
    assert captured["args"][-1] == str(artifact_path.resolve())
