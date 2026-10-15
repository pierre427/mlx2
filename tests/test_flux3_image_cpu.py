"""Artifact admission and request failure contracts without MLX or weights."""

import hashlib
import io
import json

import pytest
from PIL import Image

from mlx2.adapters import flux3_image as module


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    entries = []
    for name, raw in (
        ("config.json", b'{"model":"tiny-test-artifact"}'),
        ("model.safetensors", b"tiny"),
    ):
        (tmp_path / name).write_bytes(raw)
        entries.append(
            {
                "path": name,
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob": None,
            }
        )
    manifest = {
        "repo": module.FLUX3_REPO,
        "revision": module.FLUX3_REVISION,
        "files": entries,
    }
    monkeypatch.setattr(module, "snapshot_manifest", lambda: manifest)
    (tmp_path / ".hf-download-manifest.json").write_text(json.dumps(manifest))
    (tmp_path / ".hf-download-complete.json").write_text(
        json.dumps(
            {
                "repo": module.FLUX3_REPO,
                "revision": module.FLUX3_REVISION,
                "files": len(entries),
                "total_bytes": sum(e["size"] for e in entries),
            }
        )
    )
    return tmp_path


def test_hash_and_revision_admission(artifact):
    assert module.inspect_flux3_image(artifact).execution_qualified is False
    (artifact / "model.safetensors").write_bytes(b"evil")
    with pytest.raises(ValueError, match="hash differs"):
        module.inspect_flux3_image(artifact)


def test_receipts_and_reference_bytes(artifact):
    class Backend:
        def generate(self, prompt, references, **kwargs):
            assert prompt == "a teapot" and len(references) == 1
            return Image.new("RGB", (kwargs["width"], kwargs["height"]), "blue")

    adapter = module.Flux3ImageKleinAdapter(
        artifact, backend_factory=lambda root: Backend()
    )
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, format="PNG")
    reference = artifact / "reference.png"
    reference.write_bytes(buffer.getvalue())
    result = adapter.edit_image("a teapot", [reference], width=64, height=64)
    assert result.reference_sha256 == (hashlib.sha256(buffer.getvalue()).hexdigest(),)
    assert adapter.last_route_receipt["success"] is True
    assert adapter.last_route_receipt["observed_used"] is True
    assert adapter.last_route_receipt["qualified"] is False
    assert adapter.status()["apcv2"] == "not_applicable"
    assert adapter.status()["observed_used"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": True},
        {"height": 65},
        {"steps": 0},
        {"seed": -1},
        {"width": 4096, "height": 4096},
    ],
)
def test_requests_fail_before_loading(artifact, kwargs):
    def forbidden(root):
        pytest.fail("invalid request loaded the backend")

    adapter = module.Flux3ImageKleinAdapter(artifact, backend_factory=forbidden)
    with pytest.raises(ValueError):
        adapter.generate_image("a teapot", **kwargs)
    assert adapter.last_route_receipt["success"] is False
    assert adapter.last_route_receipt["observed_used"] is False


def test_changed_artifact_fails_before_loading(artifact):
    adapter = module.Flux3ImageKleinAdapter(artifact)
    (artifact / "config.json").write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed since inspection"):
        adapter.generate_image("a teapot")


def test_too_many_references(artifact):
    adapter = module.Flux3ImageKleinAdapter(artifact)
    with pytest.raises(ValueError, match="one to ten"):
        adapter.edit_image("a teapot", ["absent"] * 11)


def test_unsupported_capability_has_failure_receipt(artifact):
    adapter = module.Flux3ImageKleinAdapter(artifact)
    with pytest.raises(ValueError, match="unsupported FLUX 3 capabilities: guidance"):
        adapter.generate_image("a teapot", guidance=3.5)
    assert adapter.last_route_receipt["success"] is False
    assert adapter.last_route_receipt["observed_used"] is False


def test_unbound_processor_input_is_rejected(artifact):
    adapter = module.Flux3ImageKleinAdapter(artifact)
    folder = artifact / "text_encoder"
    folder.mkdir()
    (folder / "chat_template.jinja").write_text("unbound")
    with pytest.raises(ValueError, match="unbound"):
        adapter.generate_image("a teapot")
