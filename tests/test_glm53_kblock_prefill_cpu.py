"""K-block NAX/segmented prefill admission and GLM prompt scoping; no GPU dispatch."""

from types import SimpleNamespace

import pytest

from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter
from mlx2.runtime.gguf.nax_metal import NAX_FORMATS, nax_admits
from mlx2.runtime.gguf.prefill_scope import dense_prefill_active
from mlx2.runtime.gguf.reader import GGUFError
from mlx2.runtime.gguf.segmented_prefill import _validate, segmented_prefill_active
from mlx2.runtime.gguf.weights import _nax_candidate
from mlx2.runtime.models.glm53_gguf import Glm53GGUFModel

Q4_0, Q3_K, Q4_K, Q5_K, Q6_K = 2, 11, 12, 13, 14


def test_k_blocks_join_the_nax_formats_behind_tile_geometry():
    assert {Q4_0, Q3_K, Q4_K, Q5_K, Q6_K, 8, 10, 16} <= set(NAX_FORMATS)
    # GLM-5.3 REAP50 projections, numpy [out, in].
    assert nax_admits(Q3_K, 8192, 4096)  # KDA q/k
    assert nax_admits(Q4_K, 4096, 16384)  # MLA attn_output
    assert nax_admits(Q3_K, 32, 4096)  # indexer proj
    assert nax_admits(Q3_K, 512, 256)  # one attn_k_b group
    assert not nax_admits(Q4_0, 8192, 128)  # ssm_f_b/g_b stay reference
    assert not nax_admits(Q3_K, 144, 4096)
    assert not nax_admits(0, 8192, 4096)  # native F32 never takes the tile


def test_legacy_formats_keep_their_fail_closed_kernel_checks():
    # Q8_0/Q2_K/IQ2_XXS were always handed to the kernel, which raises on
    # bad geometry; only K-blocks are filtered at load-time geometry.
    assert _nax_candidate(8, 144, 4096)
    assert not _nax_candidate(Q4_0, 8192, 128)
    assert _nax_candidate(Q6_K, 4096, 4096)


def _packed(shape, ggml_type):
    return SimpleNamespace(shape=shape, ggml_type=ggml_type)


def test_segmented_moe_accepts_glm_mixed_k_block_experts():
    experts, hidden, k = 144, 2048, 4096
    gate = _packed((experts, hidden, k), Q3_K)
    down = _packed((experts, k, hidden), Q4_K)
    x = SimpleNamespace(ndim=2, shape=(512, k))
    ids = SimpleNamespace(ndim=2, shape=(512, 8))
    assert _validate(x, ids, ids, gate, gate, down, 512 << 20) == (experts, hidden, k)
    with pytest.raises(GGUFError):
        _validate(x, ids, ids, gate, _packed((experts, hidden, k), Q4_K), down, 512 << 20)


@pytest.mark.parametrize(
    "dense,expert", [("nax_fp16", "segmented_nax_fp16"), ("reference", "reference")]
)
def test_glm_prompt_scope_binds_exact_width_only(dense, expert):
    model = SimpleNamespace(
        weights=SimpleNamespace(dense_prefill_policy=dense, expert_prefill_policy=expert)
    )
    tiled = dense != "reference"
    with Glm53GGUFModel._prompt_scope(model, 512):
        assert dense_prefill_active(512) is tiled
        assert segmented_prefill_active(512) is tiled
        assert not dense_prefill_active(4)
        assert not segmented_prefill_active(4)
    assert not dense_prefill_active(512)
    assert not segmented_prefill_active(512)


def _parse(policy, tmp_path):
    adapter = object.__new__(Glm53GGUFAdapter)
    adapter.profile = "p"
    adapter.external_policy = {}
    # Policy parsing precedes artifact inspection; the missing artifact then
    # fails, after the profile and kernel policy have been fixed.
    with pytest.raises(Exception) as caught:
        Glm53GGUFAdapter._load(adapter, tmp_path / "missing", policy)
    return adapter, caught.value


def test_glm_prefill_policies_are_explicit_and_change_identity(tmp_path):
    adapter, error = _parse(
        {"glm53_dense_prefill": "nax_fp16", "glm53_expert_prefill": "segmented_nax_fp16"},
        tmp_path,
    )
    assert "unsupported GLM-5.3 packed execution policy" not in str(error)
    assert adapter.kernel_policy["glm53_dense_prefill"] == "nax_fp16"
    # The sparse DSA default suffix (SPARSE_DEFAULTS) follows these keys.
    assert ":dense_prefill=nax_fp16:expert_prefill=segmented_nax_fp16:" in adapter.profile + ":"
    reference, _ = _parse({}, tmp_path)
    assert "dense_prefill" not in reference.profile and "expert_prefill" not in reference.profile
    assert reference.kernel_policy["glm53_expert_prefill"] == "reference"
    rows, _ = _parse({"glm53_row_kernel": "kblock_rows", "glm53_shared_fold": "kblock",
                      "glm53_kda_group": "kblock"}, tmp_path)
    assert ":row_kernel=kblock_rows:shared_fold=kblock:kda_group=kblock:" in rows.profile + ":"
    assert "shared_fold" not in reference.profile
    for bad in ({"glm53_dense_prefill": "segmented_nax_fp16"},
                {"glm53_expert_prefill": "nax_fp16"},
                {"glm53_row_kernel": "nax_fp16"},
                {"glm53_shared_fold": "on"},
                {"glm53_kda_group": "on"},
                {"glm53_mla_group": "on"}):
        _, error = _parse(bad, tmp_path)
        assert "unsupported GLM-5.3 packed execution policy" in str(error)


def test_kblock_rows_admit_whole_chunks_of_k_blocks_only():
    from mlx2.runtime.gguf.kblock_metal import kblock_admits

    assert kblock_admits(Q3_K, 4096) and kblock_admits(Q6_K, 4096)
    assert kblock_admits(Q4_0, 128)  # GLM ssm_f_b/g_b
    assert not kblock_admits(Q3_K, 128)
    assert not kblock_admits(8, 4096)  # Q8_0 keeps its own kernels
    assert not kblock_admits(0, 4096)


def test_kblock_metal_sources_compile_on_the_cpu_compiler(tmp_path):
    import shutil
    import subprocess

    from mlx2.runtime.gguf import kblock_metal

    if not shutil.which("xcrun") or subprocess.run(
        ["xcrun", "--find", "metal"], capture_output=True, check=False
    ).returncode:
        pytest.skip("CPU Metal compiler unavailable")
    constants = ("constant uint AHEAD=1,K=4096,M=2048,N=4096,TOPK=8,EXPERTS=144,ROWS=4,"
                 "ROW_BYTES=1760,FORMAT=11,GROUPS=64,VEC=3,EXACT=1,TOK=8,TOKENS=512;\n")
    attrs = ("uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],"
             "uint3 thread_position_in_grid [[thread_position_in_grid]],"
             "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],"
             "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]")
    bindings = {
        "dense": "device const float* x [[buffer(0)]],device const uchar* w [[buffer(1)]],device float* y [[buffer(2)]],",
        "gate_up": "device const float* x [[buffer(0)]],device const uint* ids [[buffer(1)]],device const uchar* gate [[buffer(2)]],device const uchar* up [[buffer(3)]],device float* mid [[buffer(4)]],",
        "down": "device const float* mid [[buffer(0)]],device const uint* ids [[buffer(1)]],device const uchar* w [[buffer(2)]],device float* out [[buffer(3)]],",
        "combine": "device const float* expert_out [[buffer(0)]],device const float* scores [[buffer(1)]],device float* y [[buffer(2)]],",
        "gate_up_shared": "device const float* x [[buffer(0)]],device const uint* ids [[buffer(1)]],device const uchar* gate [[buffer(2)]],device const uchar* up [[buffer(3)]],device const uchar* gate_sh [[buffer(4)]],device const uchar* up_sh [[buffer(5)]],device float* mid [[buffer(6)]],",
        "down_shared": "device const float* mid [[buffer(0)]],device const uint* ids [[buffer(1)]],device const uchar* w [[buffer(2)]],device const uchar* w_sh [[buffer(3)]],device float* out [[buffer(4)]],",
        "combine_shared": "device const float* expert_out [[buffer(0)]],device const float* scores [[buffer(1)]],device float* y [[buffer(2)]],",
    }
    sources = {
        "dense": kblock_metal.DENSE.replace("WBASE", "size_t(token%GROUPS)*N"),
        "gate_up": kblock_metal.GATE_UP,
        "down": kblock_metal.DOWN,
        "combine": kblock_metal.COMBINE,
        "gate_up_shared": kblock_metal.GATE_UP_SHARED,
        "down_shared": kblock_metal.DOWN_SHARED,
        "combine_shared": kblock_metal.COMBINE_SHARED,
    }
    for kind, body in sources.items():
        path = tmp_path / (kind + ".metal")
        path.write_text("#include <metal_stdlib>\nusing namespace metal;\n" + kblock_metal.KBLOCK_HEADER
                        + constants + f"kernel void k_{kind}(" + bindings[kind] + attrs + ") {\n" + body + "\n}\n")
        result = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", str(path), "-o",
                                 str(tmp_path / (kind + ".air"))], capture_output=True, text=True,
                                check=False)
        assert result.returncode == 0, result.stderr


def test_kblock_multi_source_selects_each_tensor_by_row_range(tmp_path):
    import shutil
    import subprocess

    from mlx2.runtime.gguf import kblock_metal

    source = kblock_metal._multi_source(3)
    assert source.count("if(row0<R") == 3 and "x2[" in source and "w2+" in source
    if not shutil.which("xcrun") or subprocess.run(
        ["xcrun", "--find", "metal"], capture_output=True, check=False
    ).returncode:
        pytest.skip("CPU Metal compiler unavailable")
    constants = ("constant uint AHEAD=1,ROWS=4,VEC=1,EXACT=1,E0=0,E1=0,E2=1,R0=0,R1=8192,R2=8320,R3=8384,RTOTAL=8384,"
                 "N0=8192,N1=128,N2=64,K0=4096,K1=4096,K2=128,F0=11,F1=13,F2=2,"
                 "RB0=1760,RB1=2816,RB2=72;\n")
    bindings = "".join(f"device const float* x{t} [[buffer({t})]]," for t in range(3)) + "".join(
        f"device const uchar* w{t} [[buffer({3 + t})]]," for t in range(3)) + "device float* y [[buffer(6)]],"
    attrs = ("uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],"
             "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],"
             "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]")
    path = tmp_path / "multi.metal"
    path.write_text("#include <metal_stdlib>\nusing namespace metal;\n" + kblock_metal.KBLOCK_HEADER
                    + constants + "kernel void k_multi(" + bindings + attrs + ") {\n" + source + "\n}\n")
    result = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", str(path), "-o",
                             str(tmp_path / "multi.air")], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_q4_0_keeps_the_exact_law_other_k_blocks_reorder():
    from mlx2.runtime.gguf.kblock_metal import exact_law

    assert exact_law(Q4_0)
    assert not any(exact_law(t) for t in (Q3_K, Q4_K, Q5_K, Q6_K))


def test_kblock_nax_prefill_source_compiles_for_metal4(tmp_path):
    import shutil
    import subprocess

    from mlx2.runtime.gguf import kblock_prefill_metal as kp

    if not shutil.which("xcrun") or subprocess.run(
        ["xcrun", "--find", "metal"], capture_output=True, check=False
    ).returncode:
        pytest.skip("CPU Metal compiler unavailable")
    constants = "constant int TOKENS=512,K=4096,N=8192,ROW_BYTES=1760,FORMAT=11,MT=4,DIAG=0;\n"
    bindings = ("device const float* x [[buffer(0)]],device const uchar* w [[buffer(1)]],"
                "device float* y [[buffer(2)]],")
    attrs = ("uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],"
             "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],"
             "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],"
             "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]")
    path = tmp_path / "prefill.metal"
    path.write_text("#include <metal_stdlib>\nusing namespace metal;\n" + kp.HEADER + constants
                    + "kernel void k_prefill(" + bindings + attrs + ") {\n" + kp.SOURCE + "\n}\n")
    result = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-c", str(path), "-o",
                             str(tmp_path / "prefill.air")], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-3000:]


def test_segmented_plan_with_32_row_tiles_covers_every_assignment_once():
    import numpy as np

    from mlx2.runtime.gguf.segmented_prefill import plan_capacity, reference_plan

    rng = np.random.default_rng(5)
    ids = np.array([rng.choice(144, 8, replace=False) for _ in range(512)], np.uint32)
    plan = reference_plan(ids, 144, 32)
    assert len(plan["tiles"]) <= plan_capacity(ids.size, 144, 32) < plan_capacity(ids.size, 144)
    covered = sorted(int(r) + i for r, _, n, _ in plan["tiles"] for i in range(n))
    assert covered == list(range(ids.size))
    assert all(1 <= n <= 32 for _, _, n, _ in plan["tiles"])


def test_kblock_nax_prefill_v2_source_compiles_for_metal4(tmp_path):
    import shutil
    import subprocess

    from mlx2.runtime.gguf import kblock_prefill_metal as kp

    if not shutil.which("xcrun") or subprocess.run(
        ["xcrun", "--find", "metal"], capture_output=True, check=False
    ).returncode:
        pytest.skip("CPU Metal compiler unavailable")
    constants = "constant int TOKENS=512,K=4096,N=8192,ROW_BYTES=1760,FORMAT=11,PREFETCH=1;\n"
    bindings = ("device const half* x [[buffer(0)]],device const uchar* w [[buffer(1)]],"
                "device float* y [[buffer(2)]],")
    attrs = ("uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],"
             "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]")
    path = tmp_path / "prefill_v2.metal"
    path.write_text("#include <metal_stdlib>\nusing namespace metal;\n" + kp.HEADER + constants
                    + "kernel void k_prefill_v2(" + bindings + attrs + ") {\n" + kp.SOURCE_V2 + "\n}\n")
    result = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-c", str(path), "-o",
                             str(tmp_path / "prefill_v2.air")], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-3000:]


@pytest.mark.parametrize("kind", ["gate_up", "down"])
def test_kblock_segmented_v2_sources_compile_for_metal4(kind, tmp_path):
    import shutil
    import subprocess

    from mlx2.runtime.gguf import kblock_prefill_metal as kp

    if not shutil.which("xcrun") or subprocess.run(
        ["xcrun", "--find", "metal"], capture_output=True, check=False
    ).returncode:
        pytest.skip("CPU Metal compiler unavailable")
    constants = "constant int TOPK=8,K=4096,N=2048,ROW_BYTES=1760,FORMAT=11,DIAG=0,TMT=32;\n"
    bindings = {
        "gate_up": ("device const half* x [[buffer(0)]],device const uint* order [[buffer(1)]],"
                    "device const uint* tiles [[buffer(2)]],device const uint* tile_count [[buffer(3)]],"
                    "device const uchar* gate [[buffer(4)]],device const uchar* up [[buffer(5)]],"
                    "device float* y [[buffer(6)]],"),
        "down": ("device const half* x [[buffer(0)]],device const uint* tiles [[buffer(1)]],"
                 "device const uint* tile_count [[buffer(2)]],device const uchar* w [[buffer(3)]],"
                 "device float* y [[buffer(4)]],"),
    }[kind]
    source = {"gate_up": kp.SEG_GATE_UP_V2, "down": kp.SEG_DOWN_V2}[kind]
    attrs = ("uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],"
             "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]")
    path = tmp_path / (kind + ".metal")
    path.write_text("#include <metal_stdlib>\nusing namespace metal;\n" + kp.HEADER + constants
                    + "kernel void k_" + kind + "(" + bindings + attrs + ") {\n" + source + "\n}\n")
    result = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-c", str(path), "-o",
                             str(tmp_path / (kind + ".air"))], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-3000:]


def test_kblock_segmented_gate_up_v3_compiles_for_metal4(tmp_path):
    import shutil
    import subprocess

    from mlx2.runtime.gguf import kblock_prefill_metal as kp

    if not shutil.which("xcrun") or subprocess.run(
        ["xcrun", "--find", "metal"], capture_output=True, check=False
    ).returncode:
        pytest.skip("CPU Metal compiler unavailable")
    constants = "constant int K=4096,N=2048,ROW_BYTES=1760,FORMAT=11,TMT=64;\n"
    bindings = ("device const half* xs [[buffer(0)]],device const uint* tiles [[buffer(1)]],"
                "device const uint* tile_count [[buffer(2)]],device const uchar* gate [[buffer(3)]],"
                "device const uchar* up [[buffer(4)]],device float* y [[buffer(5)]],")
    attrs = ("uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],"
             "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]")
    # TMT is a template parameter as MLX instantiates it: dependent names need
    # their 'template' keywords.
    path = tmp_path / "gate_up_v3.metal"
    path.write_text("#include <metal_stdlib>\nusing namespace metal;\n" + kp.HEADER
                    + constants.replace(",TMT=64", "")
                    + "template <int TMT, int TNT, int DIAG, int PREFETCH> kernel void k(" + bindings + attrs + ") {\n" + kp.SEG_GATE_UP_V3
                    + "\n}\ntemplate [[host_name(\"k64\")]] kernel void k<64, 64, 0, 1>(" + bindings + attrs + ");\n"
                    + "template [[host_name(\"k32\")]] kernel void k<32, 32, 1, 0>(" + bindings + attrs + ");\n")
    result = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-c", str(path), "-o",
                             str(tmp_path / "v3.air")], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-3000:]


@pytest.mark.parametrize("kind", ["gate_up", "down"])
def test_kblock_segmented_v4_sources_compile_for_metal4(kind, tmp_path):
    import shutil
    import subprocess

    from mlx2.runtime.gguf import kblock_prefill_metal as kp

    if not shutil.which("xcrun") or subprocess.run(
        ["xcrun", "--find", "metal"], capture_output=True, check=False
    ).returncode:
        pytest.skip("CPU Metal compiler unavailable")
    bindings = {
        "gate_up": ("device const half* xs [[buffer(0)]],device const uint* tiles [[buffer(1)]],"
                    "device const uint* tile_count [[buffer(2)]],device const uchar* gate [[buffer(3)]],"
                    "device const uchar* up [[buffer(4)]],device float* y [[buffer(5)]],"),
        "down": ("device const half* x [[buffer(0)]],device const uint* tiles [[buffer(1)]],"
                 "device const uint* tile_count [[buffer(2)]],device const uchar* w [[buffer(3)]],"
                 "device float* y [[buffer(4)]],"),
    }[kind]
    source = {"gate_up": kp.SEG_GATE_UP_V4, "down": kp.SEG_DOWN_V4}[kind]
    attrs = ("uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],"
             "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]")
    path = tmp_path / (kind + ".metal")
    path.write_text("#include <metal_stdlib>\nusing namespace metal;\n" + kp.HEADER
                    + "template <int K, int N, int ROW_BYTES, int FORMAT> kernel void k(" + bindings + attrs
                    + ") {\n" + source + "\n}\ntemplate [[host_name(\"k\")]] kernel void k<4096, 2048, 1760, 11>("
                    + bindings + attrs + ");\n")
    result = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-c", str(path), "-o",
                             str(tmp_path / (kind + ".air"))], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-3000:]


def test_q3k_negated_integer_weights_fit_int8_exactly():
    # sc in [-32, 31], q in [-4, 3]: sc*q spans [-124, 128]; its negation fits int8.
    products = {sc * q for sc in range(-32, 32) for q in range(-4, 4)}
    assert min(products) == -124 and max(products) == 128
    assert all(-128 <= -p <= 127 for p in products)


def test_w8a8_gate_up_compiles_for_metal4(tmp_path):
    import shutil
    import subprocess

    from mlx2.runtime.gguf import kblock_prefill_metal as kp

    if not shutil.which("xcrun") or subprocess.run(
        ["xcrun", "--find", "metal"], capture_output=True, check=False
    ).returncode:
        pytest.skip("CPU Metal compiler unavailable")
    bindings = ("device const int8_t* xq [[buffer(0)]],device const float* xs [[buffer(1)]],"
                "device const uint* tiles [[buffer(2)]],device const uint* tile_count [[buffer(3)]],"
                "device const uchar* gate [[buffer(4)]],device const uchar* up [[buffer(5)]],"
                "device float* y [[buffer(6)]],")
    attrs = ("uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],"
             "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]")
    path = tmp_path / "w8a8.metal"
    path.write_text("#include <metal_stdlib>\nusing namespace metal;\n" + kp.W8A8_HEADER
                    + "template <int K, int N, int ROW_BYTES> kernel void k(" + bindings + attrs
                    + ") {\n" + kp.SEG_GATE_UP_W8A8 + "\n}\ntemplate [[host_name(\"k\")]] kernel void k<4096, 2048, 1760>("
                    + bindings + attrs + ");\n")
    result = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-c", str(path), "-o",
                             str(tmp_path / "w8a8.air")], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-3000:]


def test_kblock_register_prefill_compiles_for_metal4(tmp_path):
    import shutil
    import subprocess

    from mlx2.runtime.gguf import kblock_prefill_metal as kp

    if not shutil.which("xcrun") or subprocess.run(
        ["xcrun", "--find", "metal"], capture_output=True, check=False
    ).returncode:
        pytest.skip("CPU Metal compiler unavailable")
    bindings = ("device const half* x [[buffer(0)]],device const uchar* w [[buffer(1)]],"
                "device float* y [[buffer(2)]],")
    attrs = ("uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],"
             "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],"
             "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]")
    path = tmp_path / "kreg.metal"
    for fmt, rb in ((11, 1760), (12, 2304), (13, 2816), (14, 3360)):
        path.write_text("#include <metal_stdlib>\nusing namespace metal;\n" + kp.KREG_HEADER
                        + "template <int TOKENS, int K, int N, int ROW_BYTES, int FORMAT, int TMR> kernel void k("
                        + bindings + attrs + ") {\n" + kp.KREG_SOURCE + "\n}\n"
                        + f"template [[host_name(\"k\")]] kernel void k<512, 4096, 8192, {rb}, {fmt}, 2>("
                        + bindings + attrs + ");\n")
        result = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-c", str(path), "-o",
                                 str(tmp_path / "kreg.air")], capture_output=True, text=True, check=False)
        assert result.returncode == 0, (fmt, result.stderr[-3000:])


def test_mla_group_policy_extends_the_profile(tmp_path):
    adapter, error = _parse({"glm53_row_kernel": "kblock_rows", "glm53_mla_group": "kblock"}, tmp_path)
    assert "unsupported GLM-5.3 packed execution policy" not in str(error)
    assert ":row_kernel=kblock_rows:mla_group=kblock:" in adapter.profile + ":"


def test_glm_prompt_decoded_policy_is_explicit_and_changes_identity(tmp_path):
    adapter, error = _parse({"glm53_prompt_decoded": "decoded_f32_v1"}, tmp_path)
    assert "unsupported GLM-5.3 packed execution policy" not in str(error)
    assert adapter.kernel_policy["glm53_prompt_decoded"] == "decoded_f32_v1"
    assert ":prompt_decoded=decoded_f32_v1:" in adapter.profile + ":"
    reference, _ = _parse({}, tmp_path)
    assert reference.kernel_policy["glm53_prompt_decoded"] == "reference"
    assert "prompt_decoded" not in reference.profile
    _, bad = _parse({"glm53_prompt_decoded": "f16"}, tmp_path)
    assert "unsupported GLM-5.3 packed execution policy" in str(bad)


def test_prompt_decoded_admission_is_prompt_width_and_k_block_only():
    from mlx2.runtime.gguf.prefill_scope import prompt_rows_active
    from mlx2.runtime.gguf.weights import GGUFWeights

    weights = SimpleNamespace(prompt_decoded_policy="decoded_f32_v1")
    admits = GGUFWeights._prompt_decoded
    assert admits(weights, _packed((8192, 128), Q4_0))  # KDA ssm_f_b / ssm_g_b
    assert admits(weights, _packed((64, 512, 256), Q3_K))  # grouped attn_k_b
    assert not admits(weights, _packed((8192, 40), Q4_0))
    assert not admits(weights, _packed((8192, 128), 8))  # Q8_0 has no K-block decoder
    assert not admits(SimpleNamespace(prompt_decoded_policy="reference"), _packed((8192, 128), Q4_0))
    # The scope is the prompt width itself, under the reference dense policy too.
    model = SimpleNamespace(
        weights=SimpleNamespace(dense_prefill_policy="reference", expert_prefill_policy="reference")
    )
    with Glm53GGUFModel._prompt_scope(model, 512):
        assert prompt_rows_active(512) and not prompt_rows_active(4) and not prompt_rows_active(511)
        assert not dense_prefill_active(512)
    assert not prompt_rows_active(512)


def test_prompt_decoded_linear_skips_tile_admitted_projections():
    """Under glm53_dense_prefill=reference the default decoded_f16_v1 must not
    take over projections a NAX tile admits (the documented no-fast-tile set
    is the KDA K=128 low-rank rows); grouped k_b/v_b keep their own admission."""
    from mlx2.runtime.gguf.weights import GGUFWeights

    weights = SimpleNamespace(prompt_decoded_policy="decoded_f16_v1")
    weights._prompt_decoded = lambda w: GGUFWeights._prompt_decoded(weights, w)
    linear = GGUFWeights._prompt_decoded_linear
    assert linear(weights, _packed((8192, 128), Q4_0))  # KDA ssm_f_b / ssm_g_b
    assert GGUFWeights._prompt_decoded(weights, _packed((4096, 4096), Q3_K))
    assert not linear(weights, _packed((4096, 4096), Q3_K))  # a dense projection with a tile
    assert not linear(weights, _packed((8192, 4096), Q4_0))
    # The built projection: a tile-admitted weight stays the packed reference.
    provider = object.__new__(GGUFWeights)
    shapes = {"dense": ((4096, 4096), Q3_K), "kda_f_b": ((8192, 128), Q4_0)}
    provider.tensors = {name: _packed(*shape) for name, shape in shapes.items()}
    provider.packed = lambda name: provider.tensors[name]
    provider.dense_prefill_policy, provider.compute_policy = "reference", "reference"
    provider.row_policy, provider.prompt_decoded_policy = "reference", "decoded_f16_v1"
    assert provider._build_linear("dense") is provider.tensors["dense"]
    assert callable(provider._build_linear("kda_f_b"))


def test_kblock_levers_are_refused_under_the_reference_row_kernel(tmp_path):
    """shared_fold/kda_group/mla_group/tree_batch run only through the K-block
    row kernels; under glm53_row_kernel=reference they would fall back
    silently while the profile named them."""
    for key, value in (("glm53_shared_fold", "kblock"), ("glm53_kda_group", "kblock"),
                       ("glm53_mla_group", "kblock"), ("glm53_tree_batch", "kblock_v1")):
        _, error = _parse({key: value, "glm53_row_kernel": "reference"}, tmp_path)
        assert "require glm53_row_kernel" in str(error), key
        ok, error = _parse({key: value, "glm53_row_kernel": "kblock_rows"}, tmp_path)
        assert "require glm53_row_kernel" not in str(error) and ok.kernel_policy[key] == value


def test_kda_fused_defaults_off_under_reference_math(tmp_path):
    adapter, _ = _parse({"glm53_math": "reference"}, tmp_path)
    assert adapter.kernel_policy["glm53_kda_fused"] == "reference"
    adapter, _ = _parse({"glm53_math": "reference", "glm53_kda_fused": "decode_v1"}, tmp_path)
    assert adapter.kernel_policy["glm53_kda_fused"] == "decode_v1"  # explicit stays explicit
    adapter, _ = _parse({}, tmp_path)
    assert adapter.kernel_policy["glm53_kda_fused"] == "decode_v1"


def test_multi_linear_steps_aside_for_decoded_prompt_projections():
    """glm53_kda_group's joined row launch must not consume the KDA low-rank
    rows at prompt width when decoded_f32_v1 owns them (codex round 1)."""
    from mlx2.runtime.gguf.weights import GGUFWeights

    class Stub:
        prompt_decoded_policy = "decoded_f32_v1"
        _prompt_decoded = GGUFWeights._prompt_decoded
        _prompt_decoded_linear = GGUFWeights._prompt_decoded_linear
        scheduled_dense_calls = {}

        def _multi_plan(self, names, joined):
            return [_packed((8192, 128), Q4_0), _packed((8192, 128), Q4_0)], ((0, 8192), (8192, 16384))

    x = SimpleNamespace(shape=(512, 128), reshape=lambda *s: SimpleNamespace(shape=(512, 128)))
    model = SimpleNamespace(
        weights=SimpleNamespace(dense_prefill_policy="reference", expert_prefill_policy="reference")
    )
    with Glm53GGUFModel._prompt_scope(model, 512):
        assert GGUFWeights.multi_linear(Stub(), [x, x], ["a", "b"]) is None


def test_gathered_f16_prompt_law_parses_and_changes_identity(tmp_path, monkeypatch):
    """glm53_sparse_prefill=gathered_f16_from_<n>: admitted at the nine
    positions, Metal-only, carried in the profile like every sparse value;
    the adapter's default prompt law on a Metal host, gathered_v1_8k without."""
    import mlx.core as mx

    from mlx2.adapters.glm53_gguf import SPARSE_DEFAULTS, sparse_default

    assert SPARSE_DEFAULTS["glm53_sparse_prefill"] == "gathered_f16_from_8192"
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    adapter, error = _parse({"glm53_sparse_prefill": "gathered_f16_from_8192"}, tmp_path)
    assert "unsupported" not in str(error) and "requires Metal" not in str(error)
    assert adapter.kernel_policy["glm53_sparse_prefill"] == "gathered_f16_from_8192"
    assert "gathered_f16_from_8192" in adapter.profile
    default, _ = _parse({}, tmp_path)
    assert default.kernel_policy["glm53_sparse_prefill"] == "gathered_f16_from_8192"
    assert default.profile == adapter.profile
    explicit_v1, _ = _parse({"glm53_sparse_prefill": "gathered_v1_8k"}, tmp_path)
    assert explicit_v1.profile != adapter.profile
    _, bad = _parse({"glm53_sparse_prefill": "gathered_f16_from_9000"}, tmp_path)
    assert "glm53_sparse_prefill" in str(bad) or "unsupported" in str(bad)
    # Without Metal the default is the fp32 gathered law; the explicit f16 value is refused.
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    assert sparse_default("glm53_sparse_prefill") == "gathered_v1_8k"
    fallback, _ = _parse({}, tmp_path)
    assert fallback.kernel_policy["glm53_sparse_prefill"] == "gathered_v1_8k"
    _, refused = _parse({"glm53_sparse_prefill": "gathered_f16_from_8192"}, tmp_path)
    assert "requires Metal" in str(refused)


def test_decoded_f16_and_mla_prompt_policies_parse_and_change_identity(tmp_path):
    adapter, error = _parse({"glm53_prompt_decoded": "decoded_f16_v1", "glm53_mla_prompt": "expanded_f16_v1"}, tmp_path)
    assert "unsupported GLM-5.3 packed execution policy" not in str(error)
    assert adapter.kernel_policy["glm53_prompt_decoded"] == "decoded_f16_v1"
    assert adapter.kernel_policy["glm53_mla_prompt"] == "expanded_f16_v1"
    assert ":prompt_decoded=decoded_f16_v1:mla_prompt=expanded_f16_v1:" in adapter.profile + ":"
    reference, _ = _parse({}, tmp_path)
    assert reference.kernel_policy["glm53_mla_prompt"] == "reference" and "mla_prompt" not in reference.profile
    _, bad = _parse({"glm53_mla_prompt": "expanded_f32_v1"}, tmp_path)
    assert "unsupported GLM-5.3 packed execution policy" in str(bad)
    # fp16 decode needs whole 64-value chunks; f32 needs 32.
    from mlx2.runtime.gguf.weights import GGUFWeights

    admits = GGUFWeights._prompt_decoded
    assert admits(SimpleNamespace(prompt_decoded_policy="decoded_f16_v1"), _packed((8192, 128), Q4_0))
    assert not admits(SimpleNamespace(prompt_decoded_policy="decoded_f16_v1"), _packed((8192, 96), Q4_0))
    assert admits(SimpleNamespace(prompt_decoded_policy="decoded_f32_v1"), _packed((8192, 96), Q4_0))
