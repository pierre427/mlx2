"""Host-only checks for external-DFlash target-head pruning."""

from __future__ import annotations

from mlx2.adapters.qwen38_27b import _load_target_weights
from mlx2.runtime import ubc_evict


def test_external_draft_prunes_embedded_mtp_before_sanitize(monkeypatch):
    observed = {}

    def fake_load(files, **options):
        observed["files"] = list(files)
        observed["options"] = options
        weights = {
            "language_model.model.embed_tokens.weight": "target",
            "language_model.mtp.fc.weight": "embedded-head",
            "model.language_model.mtp.norm.weight": "embedded-head",
            "mtp.layers.0.mlp.up_proj.weight": "embedded-head",
        }
        if options.get("prune_lazy"):
            keep_lazy = options["keep_lazy"]
            weights = {
                name: value
                for name, value in weights.items()
                if not keep_lazy(name)
            }
        return weights

    monkeypatch.setattr(ubc_evict, "load_shards_evicting", fake_load)
    result = _load_target_weights(
        ["one.safetensors"], external_draft=True, sanitize=dict
    )

    assert result == {"language_model.model.embed_tokens.weight": "target"}
    assert observed["files"] == ["one.safetensors"]
    assert observed["options"]["prune_lazy"] is True


def test_ordinary_route_preserves_embedded_mtp(monkeypatch):
    def fake_load(_files, **options):
        assert options == {}
        return {
            "language_model.model.embed_tokens.weight": "target",
            "language_model.mtp.fc.weight": "embedded-head",
        }

    monkeypatch.setattr(ubc_evict, "load_shards_evicting", fake_load)
    result = _load_target_weights([], external_draft=False, sanitize=dict)

    assert result["language_model.mtp.fc.weight"] == "embedded-head"
