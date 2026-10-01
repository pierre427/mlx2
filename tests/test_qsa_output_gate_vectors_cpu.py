"""CPU-only checks of the host builder in test_qsa_output_gate_build_metal.

Only the standard-library vector builder, geometry, padding and skip gate are
tested here, with mlx, mlx_lm and the project runtime blocked. Nothing in this
file evaluates or qualifies native output-gate bits.
"""
import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

NATIVE = Path(__file__).with_name("test_qsa_output_gate_build_metal.py")
BLOCKED = ("mlx", "mlx_lm", "mlx2.runtime")
DIGESTS = {
    "bfloat16": "c7fc894404e85aa848e467f14c10668ba5aceebb8facb2b473585cd86d55902f",
    "float16": "487b823c4ddef0bea08a340ea6d3194f0d4b81c9f951f74496ddc5e6d97df04b",
    "float32": "6e5c7592e8bab4ec906a34314ac794090ebb84076895f77d2722a226c0e75fc9",
}
COUNTS = {"bfloat16": (65280, 256), "float16": (63488, 2048), "float32": (65148, 388)}
CORRECTIONS = {"bfloat16": 0xC0DB, "float16": 0xC6D8, "float32": 0xC0DB0000}
SKIP_SCRIPT = r"""
import importlib.abc, json, sys
import pytest
BLOCKED = ("mlx", "mlx_lm", "mlx2.runtime")
def blocked(name):
    return any(name == p or name.startswith(p + ".") for p in BLOCKED)
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if blocked(name):
            raise ImportError("blocked: " + name)
sys.meta_path.insert(0, Blocker())
class Reports:
    def __init__(self):
        self.rows = []
    def pytest_runtest_logreport(self, report):
        self.rows.append([report.nodeid, report.when, report.outcome])
reports = Reports()
code = pytest.main([sys.argv[1], "--noconftest", "-p", "no:cacheprovider", "-o", "addopts=", "-q"], plugins=[reports])
print("RESULT " + json.dumps({"code": int(code), "loaded": sorted(n for n in sys.modules if blocked(n)), "rows": reports.rows}))
"""


def _blocked(name):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in BLOCKED)


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if _blocked(name):
            raise ImportError(f"{name} is blocked for the host-only builder")


def _load():
    before = {name for name in sys.modules if _blocked(name)}
    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        spec = importlib.util.spec_from_file_location("qsa_output_gate_build_host", NATIVE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.meta_path.remove(blocker)
    assert {name for name in sys.modules if _blocked(name)} == before
    return module


@pytest.fixture(scope="module")
def gate():
    return _load()


@pytest.fixture(scope="module")
def vectors(gate):
    return {label: gate.gate_vector(label) for label in gate.DTYPES}


def _digest(vector):
    width = 4 if vector["storage"] == "uint32" else 2
    return hashlib.sha256(b"".join(raw.to_bytes(width, "little") for raw in vector["raw"])).hexdigest()


def _sign(value):
    return math.copysign(1.0, value)


def test_import_is_host_only(gate):
    assert not any(_blocked(name) for name in sys.modules)
    assert not hasattr(gate, "mx") and not hasattr(gate, "np")
    assert gate.ROUTES == ("sequential_fused_merge", "native_sdpa_merge")
    assert gate.DTYPES == ("bfloat16", "float16", "float32")
    marks = [mark for mark in gate.test_native_output_gate_epilogue_bits.pytestmark if mark.name == "parametrize"]
    assert sorted(mark.args[0] for mark in marks) == ["dtype_label", "route"]
    assert all(isinstance(value, str) for mark in marks for value in mark.args[1])


@pytest.mark.parametrize(("value", "skipped"), [(None, True), ("0", True), ("1", False)])
def test_skip_gate_follows_only_the_native_switch(monkeypatch, value, skipped):
    if value is None:
        monkeypatch.delenv("MLX2_TEST_OPTIONAL_METAL", raising=False)
    else:
        monkeypatch.setenv("MLX2_TEST_OPTIONAL_METAL", value)
    module = _load()
    assert module.pytestmark.name == "skipif"
    assert module.pytestmark.args[0] is skipped
    assert "exclusive GPU gate" in module.pytestmark.kwargs["reason"]
    assert not any(_blocked(name) for name in sys.modules)


def test_native_collection_is_skip_gated_without_mlx():
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("MLX2_TEST_OPTIONAL_METAL", None)
    run = subprocess.run(
        [sys.executable, "-c", SKIP_SCRIPT, str(NATIVE)],
        cwd=NATIVE.parents[1], env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    lines = [line for line in run.stdout.splitlines() if line.startswith("RESULT ")]
    assert run.returncode == 0 and len(lines) == 1, run.stdout + run.stderr
    result = json.loads(lines[0][len("RESULT "):])
    assert result["code"] == 0 and result["loaded"] == []
    setups = {nodeid: outcome for (nodeid, when, outcome) in result["rows"] if when == "setup"}
    assert len(setups) == 6 and set(setups.values()) == {"skipped"}
    cells = {frozenset(nodeid.split("[", 1)[1].rstrip("]").split("-")) for nodeid in setups}
    assert cells == {
        frozenset((route, dtype))
        for route in ("sequential_fused_merge", "native_sdpa_merge")
        for dtype in ("bfloat16", "float16", "float32")
    }
    assert {when for (_, when, _) in result["rows"]} == {"setup", "teardown"}


@pytest.mark.parametrize("label", ["bfloat16", "float16"])
def test_sixteen_bit_vectors_exhaust_every_finite_encoding(gate, vectors, label):
    vector = vectors[label]
    (unique_count, pad_count) = COUNTS[label]
    assert vector["scope"] == "exhaustive-finite" and vector["storage"] == "uint16"
    assert (vector["unique_count"], vector["pad_count"]) == (unique_count, pad_count)
    assert vector["count"] == len(vector["raw"]) == gate.ELEMENTS == unique_count + pad_count
    unique = vector["raw"][:unique_count]
    finite = [raw for raw in range(1 << 16) if math.isfinite(gate.decode(label, raw))]
    assert unique == finite and len(set(unique)) == unique_count
    assert all(math.isfinite(gate.decode(label, raw)) for raw in vector["raw"])
    edges = vector["edges"]
    assert vector["raw"][unique_count:] == [edges[i % len(edges)] for i in range(pad_count)]
    assert len(set(vector["raw"])) == unique_count
    assert _digest(vector) == DIGESTS[label]


@pytest.mark.parametrize(
    ("label", "min_subnormal", "max_finite"),
    [("bfloat16", 2.0**-133, 3.3895313892515355e38), ("float16", 2.0**-24, 65504.0)],
)
def test_sixteen_bit_edges(gate, vectors, label, min_subnormal, max_finite):
    edges = vectors[label]["edges"]
    (zero, negative_zero) = (gate.decode(label, 0x0000), gate.decode(label, 0x8000))
    assert zero == negative_zero == 0.0 and _sign(zero) == 1.0 and _sign(negative_zero) == -1.0
    assert {0x0000, 0x8000, 0x0001, 0x8001} <= set(edges)
    assert gate.decode(label, 0x0001) == min_subnormal
    assert max(abs(gate.decode(label, raw)) for raw in edges) == max_finite
    correction = CORRECTIONS[label]
    assert gate.encode(label, gate.BF16_CORRECTION) == correction
    assert gate.decode(label, correction) == -6.84375
    neighbours = list(range(correction - gate.NEIGHBOURS, correction + gate.NEIGHBOURS + 1))
    assert set(neighbours) <= set(edges)
    values = [gate.decode(label, raw) for raw in neighbours]
    assert values == sorted(values, reverse=True) and len(set(values)) == len(values)


def test_float32_is_a_bounded_deterministic_finite_sample(gate, vectors):
    vector = vectors["float32"]
    (unique_count, pad_count) = COUNTS["float32"]
    assert vector["scope"] == "sampled-finite" and vector["storage"] == "uint32"
    assert (vector["unique_count"], vector["pad_count"]) == (unique_count, pad_count)
    assert vector["count"] == len(vector["raw"]) == gate.ELEMENTS
    unique = vector["raw"][:unique_count]
    assert len(set(unique)) == unique_count
    assert all(math.isfinite(gate.decode("float32", raw)) for raw in vector["raw"])
    edges = vector["edges"]
    assert unique[: len(edges)] == edges
    assert vector["raw"][unique_count:] == [edges[i % len(edges)] for i in range(pad_count)]
    assert gate.gate_vector("float32")["raw"] == vector["raw"]
    assert _digest(vector) == DIGESTS["float32"]
    for (sign, exponent) in [(s, e) for s in (0, 1) for e in range(255)]:
        assert any(raw >> 23 == (sign << 8) | exponent for raw in unique)
    dense = [raw for raw in unique if ((raw >> 23) & 0xFF) in gate.FP32_DENSE_EXPONENTS]
    assert len(dense) >= 2 * len(gate.FP32_DENSE_EXPONENTS) * gate.FP32_DENSE_PER_EXPONENT
    assert gate.decode("float32", 0x00000001) == 2.0**-149
    assert {0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF} <= set(edges)
    correction = CORRECTIONS["float32"]
    assert gate.encode("float32", gate.BF16_CORRECTION) == correction
    assert {correction + step for step in range(-4, 5)} <= set(edges)
    assert {(0xC0DB + step) << 16 for step in range(-4, 5)} <= set(edges)


def test_geometry_and_cell_mapping(gate, vectors):
    assert (gate.BATCH, gate.HEADS, gate.LENGTH, gate.DIM, gate.PARTIALS) == (2, 2, 64, 256, 128)
    row = gate.BATCH * gate.HEADS * gate.DIM
    largest = max(vector["unique_count"] for vector in vectors.values())
    assert gate.ELEMENTS == gate.LENGTH * row == 65536
    assert (gate.LENGTH - 1) * row < largest <= gate.ELEMENTS
    assert gate.gate_cell(0) == (0, 0, 0, 0)
    assert gate.gate_cell(gate.DIM) == (0, 1, 0, 0)
    assert gate.gate_cell(gate.HEADS * gate.DIM) == (0, 0, 1, 0)
    assert gate.gate_cell(gate.LENGTH * gate.HEADS * gate.DIM) == (1, 0, 0, 0)
    assert gate.gate_cell(gate.ELEMENTS - 1) == (1, 1, 63, 255)
    assert len({gate.gate_cell(index) for index in range(gate.ELEMENTS)}) == gate.ELEMENTS


def test_partial_tables_are_safe_finite_and_exact(gate):
    tables = gate.partial_tables()
    assert len(tables["m"]) == len(tables["l"]) == len(tables["scale"]) == gate.PARTIALS
    assert all(-1.75 <= value <= 0.0 for value in tables["m"])
    assert all(1.0 <= value <= 1.75 for value in tables["l"])
    assert len(tables["amplitude"]) == gate.ELEMENTS
    amplitudes = tables["amplitude"]
    assert any(value > 0 for value in amplitudes) and any(value < 0 for value in amplitudes)
    assert any(value == 0 and _sign(value) > 0 for value in amplitudes)
    assert any(value == 0 and _sign(value) < 0 for value in amplitudes)
    for amplitude in set(gate.AMPLITUDES):
        for scale in set(gate.PARTIAL_SCALES):
            product = amplitude * scale
            assert abs(product) <= 80.0
            for label in ("bfloat16", "float16", "float32"):
                assert gate.decode(label, gate.encode(label, product)) == product
    index = 0
    for b in range(gate.BATCH):
        for h in range(gate.HEADS):
            for l in range(gate.LENGTH):
                assert amplitudes[index : index + gate.DIM] == [gate.amplitude(b, h, l, d) for d in range(gate.DIM)]
                index += gate.DIM


@pytest.mark.parametrize("label", ["bfloat16", "float16", "float32"])
def test_correction_meets_several_nonzero_attention_values(gate, vectors, label):
    raw = vectors[label]["raw"]
    cells = [gate.gate_cell(index) for (index, value) in enumerate(raw) if value == CORRECTIONS[label]]
    paired = {gate.amplitude(*cell) for cell in cells} - {0.0}
    assert len(cells) >= 2 and len(paired) >= 2
