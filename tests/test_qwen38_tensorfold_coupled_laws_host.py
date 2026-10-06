from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from mlx_blocker import mlx_module_names, unimportable_mlx

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_qwen38_tensorfold_coupled_laws.py"


def _load():
    spec = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with unimportable_mlx("Qwen3.8 TensorFold coupled-law oracle import"):
        spec.loader.exec_module(module)
    return module


HEADER = r"""
template <typename U> inline U mlx_sigmoid_fast(U x) {
  U e = static_cast<U>(metal::exp(metal::abs(x)));
  U y = U(1) / (U(1) + e);
  return (x < 0) ? y : (U(1) - y);
}
template <typename U> inline U mlx_sigmoid_precise(U x) {
  U e = static_cast<U>(metal::precise::exp(metal::abs(x)));
  U y = U(1) / (U(1) + e);
  return (x < 0) ? y : (U(1) - y);
}
template <typename U> inline U mlx_log1p_fast(U x) {
  float xf = float(x), xp1 = 1.0f + xf;
  float out = xp1 == 1.0f ? xf : xf * (metal::log(xp1) / (xp1 - 1.0f));
  return U(out);
}
template <typename U> inline U mlx_softplus_fast(U x) {
  U inf = metal::numeric_limits<U>::infinity();
  U hi = metal::max(x, U(0)), lo = metal::min(x, U(0));
  return (lo == -inf || hi == inf)
      ? hi : (hi + mlx_log1p_fast(U(metal::exp(lo - hi))));
}
"""

PRE = r"""
const bfloat conv = bfloat(acc);
const bfloat sig = mlx_sigmoid_fast(conv);
vals[j] = float(bfloat(conv * sig));
const float norm_eps = 1e-6f / float(DK);
const float inv = metal::precise::rsqrt(ss / float(DK) + norm_eps);
const float scale = isq ? float(bfloat(1.0f / float(DK)))
                        : float(bfloat(metal::precise::rsqrt(float(DK))));
const bfloat out = bfloat(scale * float(bfloat(vals[j] * inv)));
const bfloat s = bfloat(float(Ain[w * NV + hv]) + float(DT[hv]));
const bfloat sp = mlx_softplus_fast<bfloat>(s);
G[w * NV + hv] = metal::precise::exp(
    -metal::precise::exp(float(ALOG[hv])) * float(sp));
BETA[w * NV + hv] = mlx_sigmoid_precise<float>(float(Bin[w * NV + hv]));
"""

POST = r"""
const float inv = metal::precise::rsqrt(ss / float(DV) + eps[0]);
const int d = int(lane) * PER + j;
bfloat normalized = bfloat(yv[j] * inv);
normalized = bfloat(float(NW[d]) * float(normalized));
const float zf = float(Z[d]);
const float gate = zf * mlx_sigmoid_fast<float>(zf);
OUT[d] = bfloat(float(normalized) * gate);
"""

DENSE = r"""
constexpr int E = 4;
constexpr int TPG = K > 4096 ? 1024 : 32 * ((K + 127) / 128);
for (int r = 0; r < K; r += TPG * E) ss += hv[r];
float total = simd_sum(red[thread_index_in_simdgroup]);
const float inv = metal::precise::rsqrt(total / float(K) + eps[0]);
const bfloat normalized = bfloat(hv[i] * inv);
const bfloat x = bfloat(Wt[int(t) * E + i] * normalized);
const bfloat gate = bfloat(float(GATE[e]));
const bfloat mlp_sigmoid_exp = bfloat(metal::exp(metal::abs(gate)));
const bfloat mlp_sigmoid_low = bfloat(bfloat(1.0f) / bfloat(bfloat(1.0f) + mlp_sigmoid_exp));
const bfloat mlp_sigmoid = gate < bfloat(0.0f)
    ? mlp_sigmoid_low : bfloat(bfloat(1.0f) - mlp_sigmoid_low);
const bfloat activated = bfloat(gate * mlp_sigmoid);
const bfloat up = bfloat(float(UP[e]));
const bfloat h = bfloat(activated * up);
"""

TREE = r'''
_TREE_SOURCE = r"""
state[i] = state[i] * g_;
kv_mem += state[i] * k_[s_idx];
kv_mem = simd_sum(kv_mem);
auto delta = (v_[dv_idx] - kv_mem) * beta_;
state[i] = state[i] + k_[s_idx] * delta;
out += state[i] * q_[s_idx];
out = simd_sum(out);
y[(node * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(out);
"""
_REPLAY_SOURCE = r"""
state[i] = state[i] * g_;
kv_mem += state[i] * k_[s_idx];
kv_mem = simd_sum(kv_mem);
auto delta = (v_[dv_idx] - kv_mem) * beta_;
state[i] = state[i] + k_[s_idx] * delta;
o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
"""
'''

STREAM = TREE.replace("_TREE_SOURCE", "_TREE").replace("_REPLAY_SOURCE", "_REPLAY")


def _tree(tmp_path: Path, *, mutation: tuple[str, str, str] | None = None) -> Path:
    root = tmp_path / "tensorfold"
    kernels = root / "src/tensorfold/kernels/qwen/dense/v1"
    kernels.mkdir(parents=True)
    sources = {
        "lane_glue.py": HEADER + DENSE + PRE + POST,
        "lane_tree.py": TREE,
        "stream_gdn.py": PRE + STREAM,
        "row_glue.py": DENSE + POST + "\ndef _tree_source():\n    return lane_tree._TREE_SOURCE + 'o_state states[0][i]'\n",
    }
    if mutation:
        file, old, new = mutation
        assert old in sources[file]
        sources[file] = sources[file].replace(old, new)
    for name, source in sources.items():
        (kernels / name).write_text(source)
    return root


def test_oracle_is_host_only_and_good_fixture_covers_every_law(tmp_path):
    before = mlx_module_names()
    module = _load()
    result = module.validate_tensorfold(_tree(tmp_path))
    assert result["passed"] is True
    assert all(check["passed"] for check in result["checks"])
    assert mlx_module_names() == before
    assert {check["law"] for check in result["checks"]} >= {
        "stable_fast_sigmoid",
        "precise_sigmoid",
        "compensated_log1p",
        "stable_softplus",
        "conv_silu",
        "qk_rsqrt",
        "qk_storage",
        "softplus",
        "decay_exp",
        "beta_sigmoid",
        "output_norm_gate",
        "decoder_rms_norm",
        "dense_swiglu",
        "recurrent_tree_order",
        "recurrent_replay_order",
        "row_tree_derivation",
        "decoder_rms_reduction_geometry",
        "old_decoder_rms_reduction_removed",
    }


@pytest.mark.parametrize(
    ("mutation", "law"),
    [
        (("lane_glue.py", "metal::precise::rsqrt(ss / float(DK)", "metal::rsqrt(ss / float(DK)"), "qk_rsqrt"),
        (("stream_gdn.py", "mlx_sigmoid_fast(conv)", "bfloat(1.0f / (1.0f + metal::exp(-conv)))"), "conv_silu"),
        (("lane_glue.py", "mlx_softplus_fast<bfloat>(s)", "bfloat(metal::log(1.0f + metal::exp(s)))"), "softplus"),
        (("stream_gdn.py", "metal::precise::exp(\n    -metal::precise::exp", "metal::exp(\n    -metal::exp"), "decay_exp"),
        (("row_glue.py", "bfloat normalized = bfloat(yv[j] * inv);", "float normalized = yv[j] * inv;"), "output_norm_gate"),
        (("lane_glue.py", "zf * mlx_sigmoid_fast<float>(zf)", "1.0f / (1.0f + metal::exp(-zf))"), "output_norm_gate"),
        (("row_glue.py", "metal::precise::rsqrt(total / float(K)", "metal::rsqrt(total / float(K)"), "decoder_rms_norm"),
        (("lane_glue.py", "const bfloat activated = bfloat(gate * mlp_sigmoid);", "const float activated = gate * mlp_sigmoid;"), "dense_swiglu"),
        (("lane_tree.py", "state[i] = state[i] + k_[s_idx] * delta;\nout +=", "out +=\nstate[i] = state[i] + k_[s_idx] * delta;"), "recurrent_tree_order"),
        (("stream_gdn.py", "state[i] = state[i] + k_[s_idx] * delta;\no_state", "o_state\nstate[i] = state[i] + k_[s_idx] * delta;"), "recurrent_replay_order"),
        (("lane_glue.py", "constexpr int E = 4;", "constexpr int E = 16;"), "decoder_rms_reduction_geometry"),
    ],
)
def test_oracle_rejects_each_coupled_difference(tmp_path, mutation, law):
    module = _load()
    result = module.validate_tensorfold(_tree(tmp_path, mutation=mutation))
    assert result["passed"] is False
    assert any(check["law"] == law and not check["passed"] for check in result["checks"])


def test_independent_discriminators_make_old_laws_observable():
    module = _load()
    result = module.numerical_discriminators()
    assert result["qk_epsilon"]["distinguishable"] is True
    assert result["softplus"] == {
        "input": -20.0,
        "naive_float32": 0.0,
        "compensated_float32": pytest.approx(2.06115369216775e-09),
        "distinguishable": True,
    }
    assert result["stable_sigmoid"]["stable"] == 1.0
    assert result["output_rms_rounding"]["distinguishable"] is True
    assert result["decoder_rms_geometry"] == {
        "hidden_width": 5120,
        "ordinary": {
            "kernel": "rms_looped",
            "reads_per_iteration": 4,
            "threads": 1024,
            "simdgroups": 32,
            "second_stage": "simd_sum over 32 shared slots",
        },
        "old_tensorfold": {
            "reads_per_thread": 16,
            "threads": 320,
            "simdgroups": 10,
            "second_stage": "sequential sum over 10 shared slots",
        },
        "distinguishable": True,
    }


def test_reference_binding_uses_clean_exact_files_when_available():
    root = Path("~/Desktop/mlx-uag/mlx-lm-unified")
    if not root.exists():
        pytest.skip("source-bound mlx-lm checkout unavailable")
    module = _load()
    result = module.validate_reference(root)
    assert result["revision"] == module.EXPECTED_MLX_LM_REVISION
    assert result["errors"] == []


def test_mlx_execution_reference_binding_uses_clean_exact_files_when_available():
    root = Path("~/.codex/worktrees/mlx-rms-reference-39400a0d4")
    if not root.exists():
        pytest.skip("source-bound MLX checkout unavailable")
    module = _load()
    result = module.validate_mlx_reference(root)
    assert result["revision"] == module.EXPECTED_MLX_REVISION
    assert result["errors"] == []


def test_complete_receipt_binds_model_and_execution_references(tmp_path):
    mlx_lm = Path("~/Desktop/mlx-uag/mlx-lm-unified")
    mlx = Path("~/.codex/worktrees/mlx-rms-reference-39400a0d4")
    if not mlx_lm.exists() or not mlx.exists():
        pytest.skip("source-bound model and MLX checkouts unavailable")
    module = _load()
    receipt = module.build_receipt(_tree(tmp_path), mlx_lm, mlx)
    assert receipt["schema"] == "mlx2.qwen38_tensorfold_coupled_law_oracle.v2"
    assert receipt["reference"]["errors"] == []
    assert receipt["mlx_reference"]["errors"] == []
    assert receipt["passed"] is True


def test_known_f10_candidate_fails_the_new_coupled_laws_when_available():
    root = Path(
        "~/.codex/worktrees/"
        "tensorfold-dflash-epsilon-20261005"
    )
    if not root.exists():
        pytest.skip("f10 TensorFold candidate unavailable")
    module = _load()
    result = module.validate_tensorfold(root)
    failed = {check["law"] for check in result["checks"] if not check["passed"]}
    assert result["revision"] == "f10a58e3706f21457790676004e88581c01b6fb6"
    assert {
        "stable_fast_sigmoid",
        "compensated_log1p",
        "stable_softplus",
        "conv_silu",
        "qk_rsqrt",
        "softplus",
        "decay_exp",
        "output_norm_gate",
    } <= failed


def test_storage_corrected_candidate_still_fails_reduction_geometry_when_available():
    root = Path(
        "~/.codex/worktrees/"
        "tensorfold-dflash-execution-audit-20261006"
    )
    if not root.exists():
        pytest.skip("corrected TensorFold candidate unavailable")
    module = _load()
    result = module.validate_tensorfold(root)
    assert result["revision"] == "b8eec4665b67ecb08f6076dd14059641c18be078"
    assert result["files"] == {
        "src/tensorfold/kernels/qwen/dense/v1/lane_glue.py": (
            "43448604ba9e5674791e2a36c4b1dc5a5f2afe54bdd32ef5f67d16f381bc39ec"
        ),
        "src/tensorfold/kernels/qwen/dense/v1/lane_tree.py": (
            "59f88ea97fd61af306ace5da89c4af86558910c502e976d5d65aa804414f5a75"
        ),
        "src/tensorfold/kernels/qwen/dense/v1/row_glue.py": (
            "3201e7bc050cf06cb6c9ef0d792a8d926a23cd58bd3ccf2b06b5dfaf39c204d7"
        ),
        "src/tensorfold/kernels/qwen/dense/v1/stream_gdn.py": (
            "53531129f56eacedc83d5778189bae358be46c2832967fc6d92e8f3f4e86a892"
        ),
    }
    failed = {check["law"] for check in result["checks"] if not check["passed"]}
    assert result["passed"] is False
    assert failed == {
        "decoder_rms_reduction_geometry",
        "old_decoder_rms_reduction_removed",
    }
    assert len(result["checks"]) == 37


def test_dense_corrected_candidate_passes_every_execution_law_when_available():
    root = Path(
        "~/.codex/worktrees/"
        "tensorfold-dflash-dense-laws-20261006"
    )
    if not root.exists():
        pytest.skip("dense-corrected TensorFold candidate unavailable")
    module = _load()
    result = module.validate_tensorfold(root)
    assert result["revision"] == "72d30712186a0c3fbeb6208445f0047ef8679fa0"
    assert result["passed"] is True
    assert len(result["checks"]) == 37
