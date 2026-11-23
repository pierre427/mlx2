"""Hy4-flash operator defaults of 2026-11-21 (CPU, tiny random checkpoint).

Pierre: "wire in the calculator as default on for this model" and "save all
of the other helpful defaults you found".  Three defaults, each with an
opt-out and a receipt that says ``defaulted``:

* ``amoeba_det_tiles``: the packaged calculator manifest for the target
  fingerprint ``Hy4-flash-MLX-4.6bit-mlx2@f2a8fa8d`` only
  (``DET_TILE_DEFAULT_MANIFESTS``); ``false`` opts out;
* ``sliding_attention_blocks`` 512 on Hy4-flash; ``0`` opts out;
* ``prefill_step_cap`` 2048 on Hy4-flash; ``0`` opts out.

The tiny artifact's fingerprint is mocked to the target's, and the
calculator default points at a tiny-geometry manifest built for that
fingerprint, so the default path (admission, install, receipts, APCv2
namespace) is the served one.
"""

import json
from pathlib import Path

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_det_tiles_cpu import (  # noqa: E402
    FIT,
    _char_artifact,
    _manifest,
    tiny,  # noqa: F401  (fixture)
)

from mlx2.adapters import hy_v4_flash as A  # noqa: E402
from mlx2.runtime import amoeba_det_tiles as D  # noqa: E402

TARGET = "f2a8fa8d2b7f22150ee5bd988c5acc1814d8494aa82ee01aff40a7efc13aad93"
PACKAGED = Path(A.__file__).resolve().parent / "assets/det_tiles/hy4-flash-mlx-4.6bit-mlx2-f2a8fa8d2b7f"
BAND = "+swa-band512-v1"


@pytest.fixture(scope="module")
def artifact(tiny, tmp_path_factory):  # noqa: F811
    config, W, _, _ = tiny
    return _char_artifact(tmp_path_factory.mktemp("hy4-defaults") / "Hy4-tiny", config, W)


@pytest.fixture(scope="module")
def tiny_target_root(artifact, tmp_path_factory):
    """An asset root whose default directory holds a tiny-geometry manifest for TARGET."""
    root = tmp_path_factory.mktemp("hy4-default-assets")
    _manifest(root / A.DET_TILE_DEFAULT_MANIFESTS[TARGET], artifact, bias=20.0, fingerprint=TARGET)
    return root


@pytest.fixture
def as_target(monkeypatch):
    """The tiny artifact inspects as the target checkpoint."""
    real = A.inspect_artifact

    def inspect(path):
        out = real(path)
        out["identity"] = dict(out["identity"], fingerprint=TARGET)
        return out

    monkeypatch.setattr(A, "inspect_artifact", inspect)


@pytest.fixture
def tiny_default(monkeypatch, as_target, tiny_target_root):
    monkeypatch.setattr(A, "DET_TILE_ASSET_ROOT", tiny_target_root)


def _adapter(artifact, policy=None):
    return A.HYV4FlashAdapter(str(artifact), execution_policy=policy)


def _diag(adapter):
    return adapter.diagnostics()


# ----------------------------------------------------------------- packaged

def test_packaged_manifest_is_the_fitted_tile_and_admits_the_served_numerics():
    """The shipped files are the fitted tile (digests checked by load_manifest), keyed on
    the target fingerprint, and admit the served default numerics (iq4_nl only pack)."""
    path = A.default_det_tile_manifest(TARGET)
    assert Path(path) == PACKAGED / "manifest.json"
    manifest = D.load_manifest(path)
    assert manifest["_digest"] == "166c16deea739231b82f4e44907d6d5aa828eff90bb39ccf7d907ec95bac4e8d"
    assert manifest["files"] == {
        "tiles.safetensors": "fdb74390b3a2a8a4fa75b10a5b8c68053e52d455298ff2f858cba51c72fff78b"}
    assert manifest["source"]["fingerprint"] == TARGET
    (tile,) = manifest["tiles"]
    assert (tile["read_layer"], tile["gate"]["threshold"], tile["parser"], tile["write_mode"]) == \
        (12, 0.7, D.PARSER_V2, "force")
    assert manifest["capture"] == A.DET_TILE_CAPTURE
    # Vision is not a fit-numerics key: the served default loads the tower.
    assert "vision" not in A.DET_TILE_NUMERICS_KEYS
    served = {"engram_table_format": "iq4_nl", "engram_precision": "transformers",
              "latent_kv": "native", "moe_top_k": None}
    stub = {"identity": {"fingerprint": TARGET}, "config": {"text_config": {"num_hidden_layers": 42}}}
    assert A.HYV4FlashAdapter._admit_det_tiles(path, stub, served)["_digest"] == manifest["_digest"]
    with pytest.raises(D.DetTileRefused, match="engram_table_format"):
        A.HYV4FlashAdapter._admit_det_tiles(path, stub, {**served, "engram_table_format": "bf16"})
    # The shipped manifest predates the fusion keys: it reads them as their defaults.
    assert not set(A.DET_TILE_NUMERICS_DEFAULTS) & set(manifest["fit_execution_policy"])
    fusions_off = {**served, **A.DET_TILE_NUMERICS_DEFAULTS}
    assert A.HYV4FlashAdapter._admit_det_tiles(path, stub, fusions_off)["_digest"] == manifest["_digest"]
    for key, value in (("hc_seam_fused", True), ("engram_gate_rounding", "torch"),
                       ("engram_fused_proj", True), ("indexer_fused_scores", True)):
        with pytest.raises(D.DetTileRefused, match=key):
            A.HYV4FlashAdapter._admit_det_tiles(path, stub, {**fusions_off, key: value})


def test_packaged_files_are_package_data_and_licensed():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text()
    assert '"assets/det_tiles/*/manifest.json"' in pyproject
    assert '"assets/det_tiles/*/*.safetensors"' in pyproject
    scope = json.loads((root / "provenance/LICENSE-calculator-scope.json").read_text())
    for name in ("manifest.json", "tiles.safetensors"):
        rel = str((PACKAGED / name).relative_to(root / "src").as_posix())
        assert f"src/{rel}" in scope["whole_files"]


def test_unknown_fingerprint_has_no_packaged_default():
    assert A.default_det_tile_manifest("0" * 64) is None


# ----------------------------------------------------------------- defaults

def test_no_policy_applies_every_default(artifact, tiny_default):
    adapter = _adapter(artifact)
    try:
        d = _diag(adapter)
        det = d["amoeba_det_tiles"]
        assert det["selected"] is True and det["defaulted"] is True
        assert det["source"] == "checkpoint_default" and det["default_skipped"] is None
        assert det["route"] == "ordinary"
        band = d["sliding_attention_blocks"]
        assert band["defaulted"] is True and band["rows"] == 512 and band["selected"] is True
        cap = d["prefill_step_cap"]
        assert cap["defaulted"] is True and cap["cap"] == 2048
        assert adapter.prefill_step_autoscale_cap() == 2048
        assert adapter.sliding_attention_blocks == 512
        assert adapter.model.sliding_attention_blocks == 512
        # APCv2: the band's own namespace, and the tile set joins the numerics contract.
        assert adapter.layout.endswith(BAND + "+sparse-prefill-v1") and adapter.descriptor.cache_layout == adapter.layout
        contract = adapter.execution_numerics_contract()
        assert contract["amoeba_det_tiles"]["manifest_sha256"] == adapter.amoeba_det_tiles.identity[
            "manifest_sha256"]
    finally:
        adapter.close()


def test_default_tile_equals_the_explicit_manifest(artifact, tiny_default, tiny_target_root):
    """The default goes through the explicit path: same tile identity, same namespace."""
    default = _adapter(artifact)
    explicit_path = A.default_det_tile_manifest(TARGET)
    explicit = _adapter(artifact, {"amoeba_det_tiles": explicit_path})
    try:
        assert explicit._det_tiles_receipt()["defaulted"] is False
        assert explicit._det_tiles_receipt()["source"] == "policy"
        assert default.amoeba_det_tiles.identity == explicit.amoeba_det_tiles.identity
        assert default.layout == explicit.layout
        assert default.execution_numerics_contract() == explicit.execution_numerics_contract()
    finally:
        default.close()
        explicit.close()


def test_default_keeps_the_dfly2_route(artifact, tiny_default):
    adapter = _adapter(artifact, {"draft_model": "embedded_dfly"})
    try:
        det = adapter._det_tiles_receipt()
        assert det["defaulted"] is True and det["route"] == "external_draft"
        assert adapter.identity["target_fingerprint"] == TARGET
        # The route identity (a server's reported ``artifact``) is not the checkpoint key.
        assert adapter.identity["fingerprint"] != TARGET
        assert BAND in adapter.layout and adapter.layout.index(BAND) < adapter.layout.index(
            A.HYV4FlashAdapter.EXTERNAL_ROUTE_TAG)
    finally:
        adapter.close()


def test_per_request_toggle_still_disables_the_default(artifact, tiny_default):
    adapter = _adapter(artifact)
    try:
        tiles = adapter.amoeba_det_tiles
        assert D.request_toggles(tiles, {"det_tiles": {"calculator": False}}) == {"calculator": False}
        on = adapter.request_logits_processors({"prompt": "2+2="}, prompt_length=4)
        off = adapter.request_logits_processors({"prompt": "2+2=", "det_tiles": {"calculator": False}},
                                                prompt_length=4)
        assert on and not off
    finally:
        adapter.close()


# ----------------------------------------------------------------- opt-outs

def test_each_opt_out(artifact, tiny_default):
    adapter = _adapter(artifact, {"amoeba_det_tiles": False, "sliding_attention_blocks": 0,
                                  "prefill_step_cap": 0})
    try:
        d = _diag(adapter)
        det = d["amoeba_det_tiles"]
        assert adapter.amoeba_det_tiles is None and det["selected"] is False
        assert det["defaulted"] is False and det["source"] == "policy_opt_out"
        assert det["default_manifest"] == A.default_det_tile_manifest(TARGET)
        band = d["sliding_attention_blocks"]
        assert band["rows"] == 0 and band["defaulted"] is False and band["source"] == "policy"
        assert not adapter.layout.endswith(BAND + "+sparse-prefill-v1") and BAND not in adapter.layout
        cap = d["prefill_step_cap"]
        assert cap["cap"] is None and cap["defaulted"] is False and cap["source"] == "policy_opt_out"
        assert adapter.prefill_step_autoscale_cap() is None
        assert adapter.execution_numerics_contract() is None
    finally:
        adapter.close()


def test_explicit_values_are_not_defaulted(artifact, tiny_default):
    adapter = _adapter(artifact, {"sliding_attention_blocks": 1024, "prefill_step_cap": 4096})
    try:
        d = _diag(adapter)
        assert d["sliding_attention_blocks"]["rows"] == 1024
        assert d["sliding_attention_blocks"]["defaulted"] is False
        assert adapter.layout.endswith("+swa-band1024-v1+sparse-prefill-v1")
        assert d["prefill_step_cap"] == {**d["prefill_step_cap"], "cap": 4096, "defaulted": False}
        assert d["amoeba_det_tiles"]["defaulted"] is True  # untouched key keeps its default
    finally:
        adapter.close()


def test_opt_out_values_parse_and_bad_values_refuse():
    model, _, _ = A._parse_policy({"amoeba_det_tiles": False, "prefill_step_cap": 0,
                                   "sliding_attention_blocks": 0})
    assert model == {"amoeba_det_tiles": False, "prefill_step_cap": 0, "sliding_attention_blocks": 0}
    for bad in ({"amoeba_det_tiles": True}, {"amoeba_det_tiles": ""}, {"prefill_step_cap": -1},
                {"prefill_step_cap": 2.0}):
        with pytest.raises(ValueError):
            A._parse_policy(bad)


# ---------------------------------------------------------- fingerprints / skips

def test_other_fingerprint_gets_no_calculator_default(artifact, monkeypatch, tiny_target_root):
    monkeypatch.setattr(A, "DET_TILE_ASSET_ROOT", tiny_target_root)
    adapter = _adapter(artifact)  # the tiny artifact's own fingerprint
    try:
        assert adapter.identity["fingerprint"] != TARGET
        det = adapter._det_tiles_receipt()
        assert adapter.amoeba_det_tiles is None and det["selected"] is False
        assert det["defaulted"] is False and det["source"] == "none" and det["default_manifest"] is None
        # The adapter-wide defaults still apply.
        assert adapter.sliding_attention_blocks == 512 and adapter.prefill_step_autoscale_cap() == 2048
    finally:
        adapter.close()


def test_default_skips_with_a_receipt_when_fit_numerics_differ(artifact, tiny_default):
    """A served policy under other trunk numerics loads without the default (never refuses)."""
    adapter = _adapter(artifact, {"engram_precision": "vllm"})
    try:
        det = adapter._det_tiles_receipt()
        assert adapter.amoeba_det_tiles is None and det["selected"] is False
        assert det["defaulted"] is False and "engram_precision" in det["default_skipped"]
        assert adapter.execution_numerics_contract() is None
    finally:
        adapter.close()
    # The same mismatch on an explicit manifest still fails closed.
    with pytest.raises(D.DetTileRefused, match="engram_precision"):
        _adapter(artifact, {"engram_precision": "vllm",
                            "amoeba_det_tiles": A.default_det_tile_manifest(TARGET)})


def test_default_skips_with_a_receipt_for_the_real_manifest_on_other_geometry(artifact, as_target):
    """The packaged 42-layer manifest on a 6-layer model: skipped, not a crash."""
    adapter = _adapter(artifact)
    try:
        det = adapter._det_tiles_receipt()
        assert det["selected"] is False and det["defaulted"] is False
        assert "42 layers" in det["default_skipped"]
    finally:
        adapter.close()


def test_default_skips_with_a_receipt_on_tensorfold_rows(artifact, tiny_default, monkeypatch):
    from mlx2.runtime import tensorfold_rows

    class Rows:  # the tiny checkpoint is unquantized: no projection the row law admits
        def receipt(self):
            return {"selected": True}

    monkeypatch.setattr(tensorfold_rows, "install", lambda *a, **k: Rows())
    adapter = _adapter(artifact, {"draft_model": "embedded_dfly", "ast_tensorfold_rows": True,
                                  "ast_tensorfold_rows_backend": "reference"})
    try:
        det = adapter._det_tiles_receipt()
        assert det["selected"] is False and "ast_tensorfold_rows" in det["default_skipped"]
    finally:
        adapter.close()


def test_band_default_skips_when_the_window_is_wider(monkeypatch, artifact):
    real = A.inspect_artifact

    def inspect(path):
        out = real(path)
        text = dict(out["config"]["text_config"], sliding_window=1024)
        out["config"] = dict(out["config"], text_config=text)
        return out

    monkeypatch.setattr(A, "inspect_artifact", inspect)
    stub = A.HYV4FlashAdapter.__new__(A.HYV4FlashAdapter)
    policy = {}
    stub._resolve_policy_defaults(policy, A.inspect_artifact(str(artifact)))
    assert "sliding_attention_blocks" not in policy
    assert stub.policy_defaults["sliding_attention_blocks"]["source"] == "default_skipped"
    assert "1024" in stub.policy_defaults["sliding_attention_blocks"]["skipped"]
    assert policy["prefill_step_cap"] == 2048


def test_band_and_cap_defaults_are_route_neutral_keys():
    assert {"sliding_attention_blocks", "prefill_step_cap", "amoeba_det_tiles"} <= \
        A.HYV4FlashAdapter.route_neutral_policy_keys
    assert FIT["engram_table_format"] == "bf16"  # the tiny manifest's fit numerics


@pytest.mark.parametrize("policy,key", [({"hc_seam_fused": True}, "hc_seam_fused"),
                                        ({"engram_gate_rounding": "torch"}, "engram_gate_rounding"),
                                        ({"engram_fused_proj": True}, "engram_fused_proj"),
                                        ({"indexer_fused_scores": True}, "indexer_fused_scores")])
def test_default_skips_under_a_numerics_changing_fusion(artifact, tiny_default, policy, key):
    """A calculator fitted on the unfused path: the default steps aside, an explicit manifest refuses."""
    adapter = _adapter(artifact, policy)
    try:
        det = adapter._det_tiles_receipt()
        assert adapter.amoeba_det_tiles is None and det["defaulted"] is False
        assert key in det["default_skipped"]
    finally:
        adapter.close()
    with pytest.raises(D.DetTileRefused, match=key):
        _adapter(artifact, {**policy, "amoeba_det_tiles": A.default_det_tile_manifest(TARGET)})


def test_default_still_loads_with_the_bitwise_projection_fusion(artifact, tiny_default):
    adapter = _adapter(artifact, {"mla_fused_input_proj": True})
    try:
        assert adapter._det_tiles_receipt()["selected"] is True
    finally:
        adapter.close()


def test_default_tile_set_is_reported_as_defaulted(artifact, tiny_default):
    """The server's default-route skip (serving.det_tile_default_skips_route) reads this flag."""
    adapter = _adapter(artifact)
    try:
        assert adapter.amoeba_det_tiles is not None and adapter.amoeba_det_tiles_defaulted is True
    finally:
        adapter.close()
