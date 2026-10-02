"""Differential APCv2 key checker (idea: arXiv 2609.38706, KV provenance).

The paper's checker varies every dimension that can change the K/V produced
for the same token ids and flags any that leaves the cache key unchanged. Here
each row of ``BOUND`` builds the serving key twice, through the serving
helpers (artifact fingerprint, ``persistent_runtime_revision``,
``apc_request_semantic``, ``apc_semantic_namespace``), once at the baseline
and once with one dimension varied, and asserts the keys differ. The rows in
``NOT_BOUND`` are deliberate exemptions, each with its reason. They are
asserted to leave the key unchanged, so binding one of them later has to
change this table on purpose. The persistent tier is checked separately:
``_identity_matches`` must refuse an entry written under another law.

Committed falsifier (recon-20261001 lane 8): a dimension that changes KV bits
but not the key is a bug. Before the execution-numerics wrapper, every
``env:*``, ``sp_qmm`` and ``verify_bitexact`` row failed.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from mlx2.adapters.flash_next import artifact_identity
from mlx2.multimodal import MediaValue, media_fingerprint
from mlx2.runtime.apc_numerics import CANDIDATE_ENV, execution_numerics_identity
from mlx2.runtime.apc_v2 import APCv2
from mlx2.runtime.int8_prefill import Int8PrefillPolicy
from mlx2.runtime.multi_lora import lora_apc_scope
from mlx2.runtime.prefill_plan import execution_identity
from mlx2.serving import (
    apc_request_semantic,
    apc_semantic_namespace,
    persistent_runtime_revision,
)

SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
RUNTIME = {
    "source_sha256": "src-a",
    "mlx_native_sha256": "native-a",
    "python": "3.12.0",
    "macos": "26.0",
    "mlx": "0.32.2.dev20260919+39400a0d4",
    "transformers": "5.0.0",
    "dependencies": {"numpy": "2.3.0", "tokenizers": "0.22.0"},
}


def _artifact(root, **files):
    """A Flash-Next-shaped artifact directory; ``files`` override contents."""
    root.mkdir(parents=True, exist_ok=True)
    contents = {
        "config.json": json.dumps({"model_type": "qwen4_exp", "quantization": {"bits": 4, "group_size": 64}}),
        "model.safetensors.index.json": json.dumps({"weight_map": {"a": SHARDS[0], "b": SHARDS[1]}}),
        "tokenizer.json": '{"model": {"vocab": {"a": 0}}}',
        "tokenizer_config.json": '{"eos_token": "a"}',
        "chat_template.jinja": "{{ messages }}",
        SHARDS[0]: b"\x01" * 64,
        SHARDS[1]: b"\x02" * 64,
        "ple_rows.bin": b"\x00" * 16,
        **files,
    }
    for name, value in contents.items():
        path = root / name
        path.write_bytes(value if isinstance(value, bytes) else value.encode())
        os.utime(path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    return root


def _media(sha="aa", audio=None):
    value = MediaValue("image", "image/png", sha, 3, None, {})
    policy = {"family": "gemma3n", **({"audio_frontend": audio} if audio else {})}
    return media_fingerprint([value], policy=policy)


BASE = {
    "artifact": {},
    "mtime_ns": None,
    "runtime": {},
    "layout": "qwen4-exp-layer-segments-v1",
    "tenant": None,
    "media": None,
    "lora": None,
    "hyper": None,
    "env": {},
    "sp_qmm": False,
    "verify_bitexact": False,
    "prefill": None,
    "lane": None,
    "int8": None,
}


def build_key(tmp_path, **varied):
    cfg = {**BASE, **varied}
    root = _artifact(tmp_path / "artifact", **cfg["artifact"])
    if cfg["mtime_ns"] is not None:
        os.utime(root / SHARDS[0], ns=(cfg["mtime_ns"], cfg["mtime_ns"]))
    fingerprint = artifact_identity(root)["fingerprint"]
    revision = persistent_runtime_revision({**RUNTIME, **cfg["runtime"]})
    scope = lora_apc_scope(cfg["media"], cfg["lora"])
    if cfg["hyper"] is not None:
        scope = (scope, "hyper-directory", cfg["hyper"])  # request_apc_scope
    semantic = apc_semantic_namespace(
        apc_request_semantic(cfg["tenant"], scope),
        execution_numerics=execution_numerics_identity(
            cfg["env"], sp_qmm=cfg["sp_qmm"], verify_bitexact=cfg["verify_bitexact"]
        ),
        prefill_execution=cfg["prefill"],
        lane_matmul_receipt=cfg["lane"],
        int8_prefill_policy=cfg["int8"],
    )
    return APCv2.key(
        fingerprint,
        revision=revision,
        adapter=fingerprint,
        tokenizer_fingerprint=fingerprint,
        cache_layout_fingerprint=cfg["layout"],
        semantic_fingerprint=semantic,
    )


LANE = {"law_id": "lane-v1:metal:min_rows=4", "covered": {"affine-q4-g64": 7}}
SCAN = {"chunk_size": 64, "segment_max_rows": 2048, "layers": 48}

# dimension -> (variation, where the K/V or recurrent state changes)
BOUND = {
    "weights: quantization (config.json)": (
        {"artifact": {"config.json": json.dumps({"model_type": "qwen4_exp", "quantization": {"bits": 8, "group_size": 64}})}},
        "dequantized weights",
    ),
    "weights: shard size": ({"artifact": {SHARDS[0]: b"\x01" * 65}}, "weights"),
    "weights: shard mtime": ({"mtime_ns": 1_800_000_000_000_000_000}, "weights (replaced file)"),
    "weights: index map": (
        {"artifact": {"model.safetensors.index.json": json.dumps({"weight_map": {"a": SHARDS[1], "b": SHARDS[0]}})}},
        "weights",
    ),
    "tokenizer": ({"artifact": {"tokenizer.json": '{"model": {"vocab": {"b": 0}}}'}}, "token ids"),
    "chat template": ({"artifact": {"chat_template.jinja": "{{ messages | tojson }}"}}, "token ids"),
    "runtime: mlx2 source": ({"runtime": {"source_sha256": "src-b"}}, "every kernel"),
    "runtime: mlx native binary": ({"runtime": {"mlx_native_sha256": "native-b"}}, "every kernel"),
    "runtime: mlx version": ({"runtime": {"mlx": "0.32.3"}}, "every kernel"),
    "runtime: transformers": ({"runtime": {"transformers": "5.1.0"}}, "processors/audio front ends"),
    "runtime: dependency": ({"runtime": {"dependencies": {"numpy": "2.4.0", "tokenizers": "0.22.0"}}}, "host prep"),
    "cache layout": ({"layout": "qwen4-exp-layer-segments-v2"}, "plane layout"),
    "tenant": ({"tenant": "tenant-b"}, "isolation (not bits)"),
    "media bytes": ({"media": _media("bb")}, "vision/audio embeddings"),
    "audio front end": (
        {"media": _media(audio={"class": "x.Whisper", "transformers": "5", "config_sha256": "c2"})},
        "audio features",
    ),
    "lora adapter": ({"lora": "lora-sha-1"}, "adapted projections"),
    "hyper-directory semantic": ({"hyper": "hd-1"}, "semantic sidecar"),
    "int8 prefill": ({"int8": Int8PrefillPolicy(enabled=True)}, "approximate prefill MLP"),
    "int8 prefill threshold": ({"int8": Int8PrefillPolicy(enabled=True, row_threshold=4096)}, "which rows int8"),
    "lane matmul law": ({"lane": LANE}, "row-invariant projections"),
    "prefill execution (GDN scan)": ({"prefill": execution_identity(scan=SCAN)}, "scan chunking"),
    "prefill execution (TensorFold)": (
        {"prefill": execution_identity(projection={"kernel": "tf", "installed": 3, "names_sha256": "n", "tile": 64})},
        "tiled projections",
    ),
    **{
        f"env: {name}": (
            {"env": {name: "1" if kind == "flag" else "65536"}},
            "candidate or reduced-precision kernel law",
        )
        for name, kind in CANDIDATE_ENV.items()
    },
    # Sorted-MoE pad policy: adaptive/always run gather_qmv or the padded
    # gather_qmm_rhs where the default floor runs the other, for the same
    # prefill chunk (review item 1; the floor row count stays NOT_BOUND).
    "env: MLX2_MOE_RHS_PAD_POLICY=adaptive": (
        {"env": {"MLX2_MOE_RHS_PAD_POLICY": "adaptive"}}, "sorted-MoE expert kernel choice",
    ),
    "env: MLX2_MOE_RHS_PAD_POLICY=always": (
        {"env": {"MLX2_MOE_RHS_PAD_POLICY": "always"}}, "sorted-MoE expert kernel choice",
    ),
    # oMLX #4070 batched one-token sparse QSA (Codex port review item 1):
    # gather/indexed attend each row's selected K/V where the default runs a
    # dense SDPA over the padded width, so served decode bits differ.
    "env: MLX_QWEN4_QSA_BATCH_DECODE_SPARSE=gather": (
        {"env": {"MLX_QWEN4_QSA_BATCH_DECODE_SPARSE": "gather"}}, "decode attention law",
    ),
    "env: MLX_QWEN4_QSA_BATCH_DECODE_SPARSE=indexed": (
        {"env": {"MLX_QWEN4_QSA_BATCH_DECODE_SPARSE": "indexed"}}, "decode attention law",
    ),
    "env: MLX_QWEN4_QSA_BATCH_DECODE_SPARSE threshold": (
        {"env": {
            "MLX_QWEN4_QSA_BATCH_DECODE_SPARSE": "gather",
            "MLX_QWEN4_QSA_BATCH_DECODE_SPARSE_MIN_CONTEXT": "16384",
        }},
        "which contexts take the sparse arm",
    ),
    "sp_qmm": ({"sp_qmm": True}, "M=2..16 matmuls (prefill tails), not bitwise identical"),
    "verify_bitexact": ({"verify_bitexact": True}, "every M<=max_m matmul (prefill tails)"),
}


@pytest.mark.parametrize("dimension", sorted(BOUND))
def test_dimension_changes_the_key(tmp_path, dimension):
    variation, _where = BOUND[dimension]
    base = build_key(tmp_path / "base")
    varied = build_key(tmp_path / "varied", **variation)
    assert varied != base, f"{dimension} changes KV state but not the APCv2 key"


def test_baseline_is_stable(tmp_path):
    assert build_key(tmp_path / "a") == build_key(tmp_path / "b")


# Reported, not bound. Asserted unchanged so a later decision is explicit.
NOT_BOUND = {
    # (Also not bound, not a key input at all: prefill_step / adaptive prefill
    # slices. APCv2 reuse itself restores at a block boundary and prefills the
    # suffix with other chunk boundaries, so state is promised only up to
    # chunking.)
    # Qualified default-on exact-family lever whose kernel choice already
    # follows the (dynamic) chunk length.
    "env: MLX2_FUSED_SDPA_MIN_L": {"env": {"MLX2_FUSED_SDPA_MIN_L": "0"}},
    # Qualified bit-identical Flash-Next levers (per-call Metal bit gates).
    "env: MLX_QWEN4_MOE_ROUTED_DECODE": {"env": {"MLX_QWEN4_MOE_ROUTED_DECODE": "off"}},
    "env: MLX_QWEN4_HC_DECODE": {"env": {"MLX_QWEN4_HC_DECODE": "0"}},
    "env: MLX_QWEN4_ATTN_FUSED_ROWS": {"env": {"MLX_QWEN4_ATTN_FUSED_ROWS": "0"}},
    "env: MLX_QWEN4_MOE_TOPK_FOLD": {"env": {"MLX_QWEN4_MOE_TOPK_FOLD": "off"}},
    # MoE rows padded to MLX's streaming floor (default 3 rows/expert): the
    # same kernel switch a prompt already gets when it crosses MLX's own
    # floor, and chunk geometry already moves it (tolerance class, like
    # prefill slicing).
    "env: MLX2_MOE_RHS_PAD_MIN_ROWS": {"env": {"MLX2_MOE_RHS_PAD_MIN_ROWS": "0"}},
    # A zero floor turns padding off whatever the pad policy says.
    "env: MLX2_MOE_RHS_PAD_POLICY under a zero floor": {
        "env": {"MLX2_MOE_RHS_PAD_POLICY": "adaptive", "MLX2_MOE_RHS_PAD_MIN_ROWS": "0"}
    },
    # Fused routed weighted sum: replays MLX's col_reduce_small order, bit-exact
    # against the eager tail (qualification/runs/recon-20261001/l4-moe-wsum).
    "env: MLX_QWEN4_MOE_WEIGHTED_SUM": {"env": {"MLX_QWEN4_MOE_WEIGHTED_SUM": "1"}},
    # Decode/verify-side row windows: generated-token KV already depends on
    # batch width and MTP acceptance at run time (exact-family arithmetic).
    "env: MLX_QWEN4_MOE_WINDOW": {"env": {"MLX_QWEN4_MOE_WINDOW": "batch_decode,verify"}},
    # Weight bytes rewritten in place with identical size and mtime_ns: the
    # artifact fingerprint stats shards instead of hashing ~100 GB.
    "weights: same size and mtime, other bytes": {"artifact": {SHARDS[0]: b"\x07" * 64}},
}


@pytest.mark.parametrize("dimension", sorted(NOT_BOUND))
def test_reported_dimension_is_not_bound(tmp_path, dimension):
    base = build_key(tmp_path / "base")
    assert build_key(tmp_path / "varied", **NOT_BOUND[dimension]) == base


def test_default_environment_keeps_existing_namespaces():
    """All-default switches add no wrapper, so persisted default blocks keep
    their identity; explicit defaults count as default."""
    assert execution_numerics_identity({}) is None
    defaults = {name: ("0" if kind == "flag" else "0") for name, kind in CANDIDATE_ENV.items()}
    assert execution_numerics_identity(defaults) is None
    assert execution_numerics_identity({"MLX_GDN_CORE": "off", "MLX_ENABLE_TF32": ""}) is None
    assert apc_semantic_namespace("text-token-v1") == "text-token-v1"


def test_qsa_batch_decode_sparse_modes_get_distinct_namespaces(tmp_path):
    """off/gather/indexed and two thresholds: five laws, five namespaces;
    an explicit off (with or without a threshold) is the default namespace."""
    variants = [
        {},
        {"MLX_QWEN4_QSA_BATCH_DECODE_SPARSE": "gather"},
        {"MLX_QWEN4_QSA_BATCH_DECODE_SPARSE": "indexed"},
        {"MLX_QWEN4_QSA_BATCH_DECODE_SPARSE": "gather",
         "MLX_QWEN4_QSA_BATCH_DECODE_SPARSE_MIN_CONTEXT": "65536"},
        {"MLX_QWEN4_QSA_BATCH_DECODE_SPARSE": "indexed",
         "MLX_QWEN4_QSA_BATCH_DECODE_SPARSE_MIN_CONTEXT": "65536"},
    ]
    keys = [build_key(tmp_path / str(i), env=env) for i, env in enumerate(variants)]
    assert len(set(keys)) == len(keys)
    assert execution_numerics_identity({"MLX_QWEN4_QSA_BATCH_DECODE_SPARSE": "off"}) is None
    assert execution_numerics_identity({
        "MLX_QWEN4_QSA_BATCH_DECODE_SPARSE": "off",
        "MLX_QWEN4_QSA_BATCH_DECODE_SPARSE_MIN_CONTEXT": "1024",
    }) is None
    # The policy's default threshold written explicitly is the same law.
    assert execution_numerics_identity({"MLX_QWEN4_QSA_BATCH_DECODE_SPARSE": "gather"}) == (
        execution_numerics_identity({
            "MLX_QWEN4_QSA_BATCH_DECODE_SPARSE": "gather",
            "MLX_QWEN4_QSA_BATCH_DECODE_SPARSE_MIN_CONTEXT": "32768",
        })
    )


def test_flash_next_policy_sparse_mode_reaches_the_identity():
    """The serving environment the policy pins is what the identity reads."""
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    assert execution_numerics_identity(FlashNextPolicy().environment()) is None
    gather = FlashNextPolicy(qsa_batch_decode_sparse="gather").environment()
    indexed = FlashNextPolicy(
        qsa_batch_decode_sparse="indexed", qsa_batch_decode_sparse_min_context=65536
    ).environment()
    assert execution_numerics_identity(gather) is not None
    assert execution_numerics_identity(gather) != execution_numerics_identity(indexed)


def test_execution_numerics_identity_is_order_and_value_canonical():
    a = execution_numerics_identity({"MLX_GDN_CORE": "true", "MLX_ENABLE_TF32": "1"})
    b = execution_numerics_identity({"MLX_ENABLE_TF32": "on", "MLX_GDN_CORE": "1"})
    assert a == b == {"version": 1, "MLX_ENABLE_TF32": "1", "MLX_GDN_CORE": "1"}
    assert execution_numerics_identity({"MLX2_QSDPA_SCORES_BUDGET_BYTES": "1024"}) != (
        execution_numerics_identity({"MLX2_QSDPA_SCORES_BUDGET_BYTES": "2048"})
    )


def test_serving_composes_both_keys_through_the_helpers():
    """The persistent template and every request key use one composition."""
    import inspect

    from mlx2 import serving

    source = inspect.getsource(serving)
    assert source.count("semantic_fingerprint=apc_semantic_namespace(") == 2
    assert source.count("apc_request_semantic(scope, media_fingerprint)") == 1
    for wrapper in ("apc_semantic_fingerprint(", "apc_lane_fingerprint(", "apc_prefill_fingerprint("):
        assert source.count(wrapper) == 1, wrapper  # only inside apc_semantic_namespace
    assert "execution_numerics_identity(" in source


# --------------------------------------------------------------------------
# Persistent tier: a restart under another law must not adopt the blocks
# --------------------------------------------------------------------------
def _server(directory, template):
    return APCv2(
        max_size=16,
        max_bytes=1 << 20,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(directory),
        persist_dir=str(directory),
        persist_identity=template,
        persist_semantic_namespace="shared",
    )


@pytest.mark.parametrize(
    "env",
    [
        {"MLX_GDN_CORE": "1"},
        {"MLX_ENABLE_TF32": "1"},
        {"MLX_QWEN4_MOE_ROUTER_KERNEL": "1"},
        {"MLX2_MOE_RHS_PAD_POLICY": "adaptive"},
    ],
)
def test_persistent_identity_refuses_another_execution_law(tmp_path, env):
    base = build_key(tmp_path / "k")
    varied = build_key(tmp_path / "k2", env=env)
    default_server = _server(tmp_path / "default", base)
    varied_server = _server(tmp_path / "varied", varied)
    try:
        assert default_server._identity_matches(base)
        assert not default_server._identity_matches(varied)
        assert varied_server._identity_matches(varied)
        assert not varied_server._identity_matches(base)
        # Request keys with a media scope keep matching their own law only.
        scoped = replace(
            varied,
            semantic_fingerprint=apc_semantic_namespace(
                apc_request_semantic(None, _media("cc")),
                execution_numerics=execution_numerics_identity(env),
            ),
        )
        assert varied_server._identity_matches(scoped)
        assert not default_server._identity_matches(scoped)
    finally:
        default_server.close()
        varied_server.close()
