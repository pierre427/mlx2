"""Header-based detection of post-0.32.5 MLX changes (host only)."""

from pathlib import Path

from mlx2.runtime import mlx_upstream

KERNELS = Path("include/mlx/backend/metal/kernels")

OLD_SDPA = "template <typename T, int D, int V = D, int TK = 4>\n[[kernel]] void sdpa_vector_2pass_1(\n"
NEW_SDPA = (
    "// Larger TK spills registers for head dim 512 (up to 3.6x slower on M5 Max).\n"
    "template <typename T, int D, int V = D, int TK = (D + V > 512 ? 2 : 4)>\n"
    "[[kernel]] void sdpa_vector_2pass_1(\n"
)


def _root(tmp_path, sdpa=None, qnax=None):
    directory = tmp_path / KERNELS
    directory.mkdir(parents=True)
    if sdpa is not None:
        (directory / "sdpa_vector.h").write_text(sdpa)
    if qnax is not None:
        (directory / "quantized_nax.h").write_text(qnax)
    return tmp_path


def test_tk2_rule_detected_only_on_4941(tmp_path):
    assert mlx_upstream.sdpa_vector_d512_tk2(_root(tmp_path / "a", sdpa=NEW_SDPA)) is True
    assert mlx_upstream.sdpa_vector_d512_tk2(_root(tmp_path / "b", sdpa=OLD_SDPA)) is False


def test_int8_qmm_detected_only_on_4802(tmp_path):
    new = _root(tmp_path / "a", qnax="template <...> [[kernel]] void qmm_t_int_nax(\n")
    old = _root(tmp_path / "b", qnax="template <...> [[kernel]] void qmm_t_nax(\n")
    assert mlx_upstream.int8_qmm_available(new) is True
    assert mlx_upstream.int8_qmm_available(old) is False


def test_missing_headers_are_unknown(tmp_path):
    assert mlx_upstream.sdpa_vector_d512_tk2(tmp_path) is None
    assert mlx_upstream.int8_qmm_available(_root(tmp_path / "a")) is None
    status = mlx_upstream.status(tmp_path)
    assert status["sdpa_vector_d512_tk2_4941"] is None and status["int8_qmm_4802"] is None


def test_upgrade_pin_is_not_applied_yet():
    # The pin belongs to the upgrade commit; until then the process numerics
    # (and every adapter receipt) stay as they are on MLX 0.32.5.
    from mlx2 import process_env

    assert mlx_upstream.UPGRADE_PROCESS_NUMERICS == {"MLX_ENABLE_INT8_QMM": "0"}
    assert "MLX_ENABLE_INT8_QMM" not in process_env.PROCESS_NUMERICS


def test_installed_build_is_consistent():
    # Whatever MLX this venv holds, the cached and uncached answers agree.
    assert mlx_upstream.installed_sdpa_vector_d512_tk2() is bool(
        mlx_upstream.sdpa_vector_d512_tk2())
