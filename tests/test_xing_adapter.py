import json
import subprocess
import sys

import pytest

from mlx2.adapters import xing
from mlx2.adapters.registry import inspect_model, resolve_adapter
from mlx2.adapters.xing_memory import XingCacheBudget
from mlx2.contracts import Capability

CONFIG = {
    **xing._TOPOLOGY,
    "architectures": ["Xing4_0ForCausalLM"],
    "num_nextn_predict_layers": 1,
    "max_position_embeddings": 262144,
    "eos_token_id": 2,
    "mlx2_conversion": {
        "layout": xing.CONVERSION_LAYOUT, "bits": 6, "mtp": True,
        "mtp_embedding_shared": True, "mtp_head_shared": True,
        "shard_sha256": {"model-00001-of-00001.safetensors": "0" * 64},
    },
}
SHARD = b'{"__metadata__":{"format":"mlx"}}'
SHARD = len(SHARD).to_bytes(8, "little") + SHARD
MTP_KEYS = [
    "mtp.layers.0.enorm.weight",
    "mtp.layers.0.hnorm.weight",
    "mtp.layers.0.eh_proj.weight",
    "mtp.layers.0.shared_head.norm.weight",
    "mtp.layers.0.self_attn.kv_a_proj_with_mqa.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.switch_mlp.down_proj.weight",
]


def _artifact(tmp_path, *, config=None, extra_keys=(), mtp=True, drop=()):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps(config or CONFIG))
    keys = ["model.embed_tokens.weight", "lm_head.weight", *extra_keys]
    if mtp:
        keys += MTP_KEYS
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model-00001-of-00001.safetensors" for key in keys}})
    )
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(SHARD)
    for name in ("tokenizer.json", "xing4_0_tokenizer_parity.json", "chat_template.jinja"):
        if name not in drop:
            (tmp_path / name).write_text("{}")
    return tmp_path


def test_registry_inspection_does_not_import_mlx(tmp_path):
    path = _artifact(tmp_path)
    code = (
        "import sys; from mlx2.adapters.registry import inspect_model;"
        f"r = inspect_model({str(path)!r});"
        "assert r.descriptor.family == 'xing4.0-29b-a4b';"
        "assert 'mlx' not in sys.modules and 'mlx.core' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_descriptor_tracks_embedded_mtp(tmp_path):
    with_mtp = inspect_model(_artifact(tmp_path / "a"))
    assert with_mtp.adapter_type is xing.XingAdapter
    assert {Capability.MTP, Capability.SEGMENTED_MTP} <= with_mtp.descriptor.capabilities
    assert with_mtp.descriptor.cache_layout == xing.CACHE_LAYOUT
    ordinary = inspect_model(_artifact(tmp_path / "b", mtp=False))
    assert Capability.MTP not in ordinary.descriptor.capabilities
    for capability in (Capability.APC_V2, Capability.PROMPT_LOOKUP, Capability.GRAMMAR,
                       Capability.CONTINUOUS_BATCH, Capability.TOOLS, Capability.REASONING):
        assert capability in ordinary.descriptor.capabilities
    with pytest.raises(ValueError, match="no implemented native MTP"):
        resolve_adapter(tmp_path / "b", mtp=True)
    assert resolve_adapter(tmp_path / "a", mtp=True) is xing.XingAdapter


@pytest.mark.parametrize("key,value", [("hc_mult", 1), ("n_group", 8), ("hidden_size", 4096),
                                       ("tie_word_embeddings", True), ("scoring_func", "softmax")])
def test_wrong_topology_fails_closed(tmp_path, key, value):
    with pytest.raises(ValueError, match="topology"):
        xing.inspect_artifact(_artifact(tmp_path, config={**CONFIG, key: value}))


def test_unconverted_or_incomplete_artifacts_fail_closed(tmp_path):
    raw = {key: value for key, value in CONFIG.items() if key != "mlx2_conversion"}
    with pytest.raises(ValueError, match="convert_xing4_0"):
        xing.inspect_artifact(_artifact(tmp_path / "raw", config=raw))
    with pytest.raises(ValueError, match="unconverted"):
        xing.inspect_artifact(_artifact(tmp_path / "hf", extra_keys=["model.layers.40.enorm.weight"]))
    with pytest.raises(ValueError, match="incomplete"):
        xing.inspect_artifact(_artifact(tmp_path / "partial", mtp=False, extra_keys=MTP_KEYS[:2]))
    with pytest.raises(ValueError, match="disagree"):
        xing.inspect_artifact(_artifact(tmp_path / "count", config={**CONFIG, "num_nextn_predict_layers": 0}))
    with pytest.raises(ValueError, match="tokenizer.json"):
        xing.inspect_artifact(_artifact(tmp_path / "tok", drop=("tokenizer.json",)))
    with pytest.raises(ValueError, match="parity"):
        xing.inspect_artifact(_artifact(tmp_path / "stamp", drop=("xing4_0_tokenizer_parity.json",)))


def test_traversing_shard_fails_closed(tmp_path):
    path = _artifact(tmp_path / "a")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lm_head.weight": "../outside.safetensors"}})
    )
    (tmp_path / "outside.safetensors").write_bytes(SHARD)
    with pytest.raises(ValueError, match="local safetensors"):
        xing.inspect_artifact(path)


def test_identity_changes_with_tokenizer(tmp_path):
    path = _artifact(tmp_path)
    first = xing.inspect_artifact(path)["identity"]["fingerprint"]
    (path / "tokenizer.json").write_text('{"changed": true}')
    assert xing.inspect_artifact(path)["identity"]["fingerprint"] != first


@pytest.mark.parametrize(
    "request_fields,expected",
    [
        ({}, ("high", True)),
        ({"enable_thinking": False}, ("none", False)),
        ({"reasoning_effort": "none"}, ("none", False)),
        ({"reasoning_effort": "low"}, ("low", True)),
        # Constrained output keeps thinking: grammar defers past </think>.
        ({"response_format": {"type": "json_object"}}, ("high", True)),
        ({"grammar": "root ::= \"a\""}, ("high", True)),
    ],
)
def test_reasoning_policy_matches_vendor_template_default(request_fields, expected):
    assert xing.reasoning_policy(request_fields) == expected


def test_reasoning_policy_rejects_unknown_effort():
    with pytest.raises(ValueError):
        xing.reasoning_policy({"reasoning_effort": "extreme"})


def test_thinking_close_marker_is_the_atomic_token():
    assert xing.XingAdapter.thinking_close_token_ids() == (10,)


def test_sampling_defaults_are_vendor_profiles():
    sampling = xing.XingAdapter.sampling_defaults
    general = sampling.profiles["general"].values()
    assert general == {"temperature": 1.0, "top_p": 0.95, "repetition_penalty": 1.05}
    assert sampling.profiles["coding"].values()["temperature"] == 0.8
    assert sampling.profiles["agent"] is sampling.profiles["coding"]


@pytest.mark.parametrize(
    "policy,match",
    [
        ("x", "JSON object"),
        ({"num_draft": 4}, "num_draft"),
        ({"num_draft": True}, "num_draft"),
        ({"qsa": 1}, "supports only"),
        ({"tokenizer_reference_fallback": "yes"}, "boolean"),
    ],
)
def test_execution_policy_is_validated_before_artifact_read(tmp_path, policy, match):
    with pytest.raises(ValueError, match=match):
        xing.XingAdapter(str(tmp_path / "missing"), execution_policy=policy)


def test_cache_budget_is_mla_geometry():
    budget = XingCacheBudget.from_config(CONFIG, mtp=False)
    with_mtp = XingCacheBudget.from_config(CONFIG, mtp=True)
    assert (budget.latent_dim, budget.rope_dim, budget.attention_layers) == (512, 64, 40)
    assert with_mtp.mtp_layers == 1
    small, large = budget.project(1024), budget.project(8192)
    assert 0 < small < large
    # 576 fp32 values per token per layer, 40 layers: ~92 KB/token.
    assert 88_000 < (large - small) / (8192 - 1024) < 96_000
    assert with_mtp.project(8192) > large
    with pytest.raises(ValueError):
        XingCacheBudget.from_config({**CONFIG, "num_nextn_predict_layers": 0}, mtp=True)


def test_int8_prefill_scopes_are_declared():
    assert set(xing.XingAdapter.int8_prefill_supported()) == {"mlp", "all"}


def _conversion(**changes):
    return {**CONFIG, "mlx2_conversion": {**CONFIG["mlx2_conversion"], **changes}}


def test_mtp_sharing_must_be_recorded_and_match_tensors(tmp_path):
    head = "mtp.layers.0.shared_head.head.weight"
    # Distinct head recorded and present: accepted.
    xing.inspect_artifact(_artifact(tmp_path / "own", config=_conversion(mtp_head_shared=False), extra_keys=[head]))
    # Distinct head recorded but missing: a defect, not an implicit share.
    with pytest.raises(ValueError, match="mtp_head_shared"):
        xing.inspect_artifact(_artifact(tmp_path / "lost", config=_conversion(mtp_head_shared=False)))
    # Shared recorded but a head is present: contradictory.
    with pytest.raises(ValueError, match="mtp_head_shared"):
        xing.inspect_artifact(_artifact(tmp_path / "both", extra_keys=[head]))
    # No recorded decision at all.
    meta = {k: v for k, v in CONFIG["mlx2_conversion"].items() if k != "mtp_embedding_shared"}
    with pytest.raises(ValueError, match="mtp_embedding_shared"):
        xing.inspect_artifact(_artifact(tmp_path / "none", config={**CONFIG, "mlx2_conversion": meta}))


def test_fingerprint_binds_recorded_shard_digests_and_headers(tmp_path):
    base = xing.inspect_artifact(_artifact(tmp_path / "a"))["identity"]["fingerprint"]
    other = xing.inspect_artifact(
        _artifact(tmp_path / "b", config=_conversion(shard_sha256={"model-00001-of-00001.safetensors": "1" * 64}))
    )["identity"]["fingerprint"]
    assert other != base
    with pytest.raises(ValueError, match="every shard"):
        xing.inspect_artifact(_artifact(tmp_path / "c", config=_conversion(shard_sha256={})))
    path = _artifact(tmp_path / "d")
    header = b'{"__metadata__":{"format":"mlx2"}}'
    (path / "model-00001-of-00001.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
    changed = xing.inspect_artifact(path)["identity"]["fingerprint"]
    assert changed != xing.inspect_artifact(_artifact(tmp_path / "e"))["identity"]["fingerprint"]


def test_full_shard_verification_detects_content_changes(tmp_path):
    import hashlib

    good = hashlib.sha256(SHARD).hexdigest()
    path = _artifact(tmp_path, config=_conversion(shard_sha256={"model-00001-of-00001.safetensors": good}))
    xing.verify_shard_digests(path)
    (path / "model-00001-of-00001.safetensors").write_bytes(SHARD + b"x")
    with pytest.raises(ValueError, match="sha256"):
        xing.verify_shard_digests(path)
