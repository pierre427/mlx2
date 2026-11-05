"""Synthetic CPU contracts; no MLX, native kernel, or model execution.

Use --noconftest and disable plugin autoload: the repository-wide conftest
imports MLX. Source baseline: 96dfb0325c2a81098ed5fb02085ce3800f0963e0.
"""

import ast
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.runtime.arithmetic_class import (
    ArithmeticClass,
    attention_class,
    flattened_offset,
    gemm_class,
    lane_cache_fingerprint,
    lane_family_identity,
    lane_implementation_digest,
    tensorfold_family_identity,
    vector_address,
)

ROOT = Path(__file__).resolve().parents[1]
ARCH = "applegpu_g17s"
BUILD = "synthetic-runtime@revision-1"


def attention():
    return attention_class(128, architecture=ARCH, runtime_revision=BUILD)


def gemm():
    return gemm_class(64, 1024, bits=4, group_size=64, split_k=2,
                      architecture=ARCH, runtime_revision=BUILD)


def extracted(relative, name, globals_):
    """Execute one dependency-injected CPU metadata function, never its module."""
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = dict(globals_)
    exec(compile(module, str(path), "exec"), scope)  # noqa: S102 -- checked local AST seam avoids MLX import.
    return scope[name]


def test_round_trip_and_immutability():
    contract = attention()
    metadata = json.loads(json.dumps(contract.metadata()))
    assert ArithmeticClass.from_metadata(metadata) == contract
    contract.require_compatible(metadata)
    with pytest.raises(FrozenInstanceError):
        contract.operation = "changed"
    with pytest.raises(ValueError, match="immutable"):
        replace(contract, dataflow=list(contract.dataflow))
    assert replace(contract, dimensions=contract.dimensions[::-1]).fingerprint == contract.fingerprint


@pytest.mark.parametrize("changes", [
    {"architecture": "applegpu_g17p"}, {"runtime_revision": "other-runtime"},
    {"algorithm_version": 2}, {"output_dtype": "float16"},
    {"dataflow": (("probability", "float32-P"),)},
    {"dimensions": (("head_dim", 128), ("tile_keys", 64), ("chunk_keys", 256))},
])
def test_every_numerical_or_scope_boundary_invalidates_compatibility(changes):
    contract = attention()
    changed = replace(contract, **changes)
    assert changed.fingerprint != contract.fingerprint
    with pytest.raises(ValueError, match="mismatch"):
        contract.require_compatible(changed.metadata())


@pytest.mark.parametrize("dimension", ["M", "batch", "batch_size", "padding", "sm_count", "physical_row"])
def test_scheduling_and_batch_are_not_arithmetic_dimensions(dimension):
    with pytest.raises(ValueError, match="placement"):
        replace(attention(), dimensions=((dimension, 4),))


@pytest.mark.parametrize("changes", [
    {"algorithm_version": True}, {"input_dtype": "float8"}, {"architecture": ""},
    {"dimensions": (("head_dim", 128), ("head_dim", 256))},
    {"dimensions": (("head_dim", 0),)}, {"dataflow": ()},
])
def test_descriptor_refuses_ambiguous_or_incomplete_contracts(changes):
    with pytest.raises(ValueError):
        replace(attention(), **changes)


def test_metadata_tampering_and_unknown_schema_fail_closed():
    metadata = attention().metadata()
    for key, value in (("architecture", "other-chip"), ("fingerprint", "f" * 64),
                       ("schema", "future-schema")):
        changed = {**metadata, key: value}
        with pytest.raises(ValueError):
            ArithmeticClass.from_metadata(changed)
    with pytest.raises(ValueError, match="schema"):
        ArithmeticClass.from_metadata({**metadata, "unknown": "ignored?"})


def test_factories_describe_existing_local_math_and_refuse_unimplemented_shape():
    fields = dict(attention().dataflow)
    assert "float16" in fields["probability"]
    assert "fast-exp" in fields["probability"]
    assert "ascending-absolute" in fields["merge"]
    assert dict(attention().dimensions) == {"head_dim": 128, "tile_keys": 64, "chunk_keys": 512}
    for kwargs in ({"head_dim": 64}, {"head_dim": 128, "tile_keys": 128},
                   {"head_dim": 128, "chunk_keys": 256}):
        with pytest.raises(ValueError):
            attention_class(**kwargs, architecture=ARCH, runtime_revision=BUILD)
    assert dict(gemm().dimensions)["split_k"] == 2
    with pytest.raises(ValueError):
        gemm_class(64, 1024, bits=4, group_size=64, split_k=3,
                   architecture=ARCH, runtime_revision=BUILD)


def test_active_lane_namespace_changes_while_ordinary_namespace_is_preserved():
    metadata = lane_family_identity("mpp", architecture=ARCH, runtime_revision=BUILD)
    receipt = {"covered": {"q4": 2}, "law_id": "lane-matmul-v1", "arithmetic_class": metadata}
    old_namespace = ("artifact", "lane-matmul", "lane-matmul-v1")
    new_namespace = lane_cache_fingerprint("artifact", receipt)
    assert new_namespace != old_namespace
    assert new_namespace[-1] == metadata["fingerprint"]
    assert lane_cache_fingerprint("artifact", {"covered": {}, "law_id": "stock"}) == "artifact"
    assert lane_cache_fingerprint("artifact", None) == "artifact"
    legacy = {"covered": {"q4": 2}, "law_id": "lane-matmul-v1"}
    assert lane_cache_fingerprint("artifact", legacy) not in (old_namespace, new_namespace)
    changed = {**receipt, "arithmetic_class": {**metadata, "runtime_revision": "tampered"}}
    with pytest.raises(ValueError, match="fingerprint"):
        lane_cache_fingerprint("artifact", changed)


def test_runtime_installer_metadata_and_apc_delegate_without_importing_mlx():
    path = "src/mlx2/runtime/lane/installer.py"
    metadata = extracted(path, "_lane_arithmetic_identity", {
        "mx": SimpleNamespace(device_info=lambda: {"architecture": ARCH}),
        "installed_mlx_build": lambda: BUILD,
        "lane_family_identity": lane_family_identity,
        "lane_implementation_digest": lambda: "a" * 64,
    })("mpp")
    receipt = {"covered": {"q4": 1}, "law_id": "lane-matmul-v1", "arithmetic_class": metadata}
    delegate = extracted(path, "apc_lane_fingerprint", {"lane_cache_fingerprint": lane_cache_fingerprint})
    assert delegate("base", receipt) == lane_cache_fingerprint("base", receipt)
    # Pin the actual install->receipt dataflow, rather than just testing the
    # standalone helper. Other live installer branches still need hardware.
    tree = ast.parse((ROOT / path).read_text())
    install = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "install")
    assert any(isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
        and target.value.id == "receipt" and isinstance(target.slice, ast.Constant)
        and target.slice.value == "arithmetic_class" for target in node.targets
    ) and isinstance(node.value, ast.Name) and node.value.id == "arithmetic_identity"
               for node in ast.walk(install))


def test_installer_refuses_missing_device_or_build_identity():
    path = "src/mlx2/runtime/lane/installer.py"
    for info, build in (({}, BUILD), ({"architecture": ARCH}, "unknown")):
        helper = extracted(path, "_lane_arithmetic_identity", {
            "mx": SimpleNamespace(device_info=lambda info=info: info),
            "installed_mlx_build": lambda build=build: build,
            "lane_family_identity": lane_family_identity,
            "lane_implementation_digest": lambda: "a" * 64,
        })
        with pytest.raises(RuntimeError, match="identity"):
            helper("mpp")


def test_identity_failure_precedes_model_or_backend_mutation():
    class Model:
        def __init__(self):
            self.projection = SimpleNamespace(_lane_prepared=SimpleNamespace(backend="simd"))
            self.visits = 0

        def named_modules(self):
            self.visits += 1
            raise AssertionError("identity failure must precede model traversal/preparation")

    def refused(_backend):
        raise RuntimeError("synthetic missing build identity")

    installer = extracted("src/mlx2/runtime/lane/installer.py", "install", {
        "DEFAULT_MAX_ROWS": 32, "MAX_ROWS": 128, "DEFAULT_GROUPS": (),
        "ProjectionGroup": object, "backend": lambda: "mpp",
        "_lane_arithmetic_identity": refused,
        "uninstall": lambda _model: pytest.fail("backend mismatch mutated model before identity validation"),
    })
    model = Model()
    original_projection, original_class = model.projection, type(model.projection)
    with pytest.raises(RuntimeError, match="missing build identity"):
        installer(model)
    assert model.projection is original_projection and type(model.projection) is original_class
    assert model.projection._lane_prepared.backend == "simd" and model.visits == 0


def test_installed_family_binds_cold_source_content_and_caches_it(tmp_path):
    names = ("arithmetic_class.py", "lane/matmul.py", "lane/simd.py",
             "lane/installer.py", "lane/tiled_prefill.py")
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic source: {name}\n")
    lane_implementation_digest.cache_clear()
    first = lane_implementation_digest(tmp_path)
    (tmp_path / "lane/matmul.py").write_text("# changed primitive without version bump\n")
    assert lane_implementation_digest(tmp_path) == first  # one imported-process snapshot
    lane_implementation_digest.cache_clear()
    second = lane_implementation_digest(tmp_path)
    assert first != second
    before = lane_family_identity("mpp", architecture=ARCH, runtime_revision=BUILD, implementation_sha256=first)
    after = lane_family_identity("mpp", architecture=ARCH, runtime_revision=BUILD, implementation_sha256=second)
    assert before["fingerprint"] != after["fingerprint"]
    lane_implementation_digest.cache_clear()


def test_tensorfold_token_includes_arithmetic_recipe_and_runtime_build():
    from mlx2.adapters.qwen38_tensorfold_source import qualification_source_identity

    identity = qualification_source_identity()
    arithmetic = ArithmeticClass.from_metadata(identity["arithmetic_class"])
    assert dict(arithmetic.dimensions) == {"attention_chunk_keys": 512, "attention_tile_keys": 64}
    assert "module_sha256" in identity
    previous = {key: value for key, value in identity.items() if key != "arithmetic_class"}
    assert json.dumps(previous, sort_keys=True) != json.dumps(identity, sort_keys=True)
    assert tensorfold_family_identity(runtime_revision="build1")["fingerprint"] != tensorfold_family_identity(
        runtime_revision="build2"
    )["fingerprint"]
    lane_attention = ast.parse((ROOT / "src/mlx2/runtime/tensorfold_qwen38/lane_attention.py").read_text())
    constants = {node.targets[0].id: node.value.value for node in lane_attention.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant)}
    assert (constants["CHUNK"], constants["TILE"]) == (512, 64)


def tree_sum(values):
    values = np.asarray(values, np.float32).copy()
    while values.shape[-1] > 1:
        values = np.add(values[..., 0::2], values[..., 1::2], dtype=np.float32)
    return values[..., 0]


def zero_score_attention(values, logical_map, position, *, lo=0, chunk=512, reverse_schedule=False):
    """Independent constant-score softmax case of fixed tile/chunk arithmetic.

    Probabilities and rescale factors are exactly one. This isolates the
    non-associative PV/chunk reductions without modelling native MMA/exp.
    """
    parts = {}
    chunks = list(range(lo // chunk, (position + chunk) // chunk))
    for index in chunks[::-1] if reverse_schedule else chunks:
        total = np.zeros(values.shape[1], np.float32)
        count = 0
        for tile in range(index * chunk, (index + 1) * chunk, 64):
            keys = np.arange(tile, tile + 64)
            valid = (keys >= lo) & (keys <= position)
            tile_values = np.zeros((64, values.shape[1]), np.float32)
            tile_values[valid] = values[logical_map[keys[valid]]]
            total = np.add(total, tree_sum(tile_values.T), dtype=np.float32)
            count += int(valid.sum())
        parts[index] = total, count
    total = np.zeros(values.shape[1], np.float32)
    count = 0
    for index in chunks:
        total = np.add(total, parts[index][0], dtype=np.float32)
        count += parts[index][1]
    return total / np.float32(count)


@pytest.mark.parametrize("position,lo", [(63, 0), (64, 0), (511, 0), (512, 0), (1023, 401)])
def test_fixed_absolute_tree_survives_batch_padding_paging_and_split_placement(position, lo):
    rng = np.random.default_rng(8590)
    values = rng.normal(size=(1024, 128)).astype(np.float32)
    dense_map = np.arange(1024)
    shuffled = rng.permutation(1024) + 37
    physical = np.full((1064, 128), np.nan, np.float32)
    physical[shuffled] = values
    ref = zero_score_attention(values, dense_map, position, lo=lo)
    target = (physical, shuffled, position, lo)
    neighbor = (values, dense_map, 63, 0)
    for batch, row in (([target], 0), ([neighbor, target, neighbor], 1)):
        outputs = [zero_score_attention(v, mapping, pos, lo=start, reverse_schedule=True)
                   for v, mapping, pos, start in batch]
        np.testing.assert_array_equal(outputs[row].view(np.uint32), ref.view(np.uint32))


def test_dynamic_chunking_is_a_different_arithmetic_class_not_an_equal_placement():
    values = np.zeros((1024, 128), np.float32)
    values[[0, 128, 256, 384], 0] = [1e8, 1, -1e8, 1]
    logical = np.arange(1024)
    fixed = zero_score_attention(values, logical, 1023, chunk=512)
    dynamic = zero_score_attention(values, logical, 1023, chunk=256)
    assert fixed[0] == np.float32(1 / 1024) and dynamic[0] == 0
    changed = replace(attention(), dimensions=(("head_dim", 128), ("tile_keys", 64), ("chunk_keys", 256)))
    assert attention().fingerprint != changed.fingerprint


def test_fixed_gemm_chunk_tree_is_independent_of_batch_and_padding():
    terms = np.zeros(64, np.float32)
    terms[[0, 16, 32, 48]] = [1e8, 1, -1e8, 1]

    def dot(values, chunks):
        partials = []
        for chunk in np.split(values, chunks):
            partial = np.float32(0)
            for block in np.split(chunk, len(chunk) // 16):
                partial = np.float32(partial + tree_sum(block))
            partials.append(partial)
        total = np.float32(0)
        for partial in partials:
            total = np.float32(total + partial)
        return total

    for batch in (1, 3, 128):
        padded = np.zeros((batch + 2, 64), np.float32)
        padded[1:batch + 1] = terms
        assert all(dot(row, 2) == 0 for row in padded[1:batch + 1])
    assert dot(terms, 1) == 1 and dot(terms, 2) == 0
    assert gemm().fingerprint != replace(gemm(), dimensions=tuple(
        (name, 1 if name == "split_k" else value) for name, value in gemm().dimensions
    )).fingerprint


def test_checked_offsets_widen_before_multiplication_and_do_not_allocate():
    total = 4160 * 64 * 64 * 128
    offset = flattened_offset(4160 * 64 * 64 - 1, 128, 127, total_elements=total)
    assert offset == total - 1 and offset > 2**31
    for kwargs in ({"row": True, "stride": 128, "column": 0, "total_elements": 1024},
                   {"row": 2**62, "stride": 128, "column": 0, "total_elements": 2**63 - 1}):
        with pytest.raises(ValueError):
            flattened_offset(**kwargs)


def test_vector_alignment_extent_and_native_packet_shapes():
    assert vector_address(4096, 136, element_bytes=2, vector_elements=8, total_elements=256) % 16 == 0
    for offset, count in ((129, 256), (128, 130)):
        with pytest.raises(ValueError, match="alignment|extent"):
            vector_address(4096, offset, element_bytes=2, vector_elements=8, total_elements=count)
    with pytest.raises(ValueError, match="power-of-two"):
        vector_address(4096, 0, element_bytes=4, vector_elements=3, total_elements=256)
    with pytest.raises(ValueError, match="power-of-two"):
        vector_address(4096, 0, element_bytes=3, vector_elements=4, total_elements=256)


def test_cpu_metadata_and_source_identity_never_import_mlx():
    script = """
import sys
from mlx2.runtime.arithmetic_class import attention_class
from mlx2.adapters.qwen38_tensorfold_source import qualification_source_identity
attention_class(128,architecture='synthetic-chip',runtime_revision='synthetic-build').metadata()
qualification_source_identity()
assert not any(name in sys.modules for name in ('mlx.core','coremltools','torch','flashinfer'))
"""
    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
