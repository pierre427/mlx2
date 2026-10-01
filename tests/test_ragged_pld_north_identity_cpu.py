"""Host-only tests for the North bounded-header identity gate (scripts/qualify_ragged_pld.py).

No mlx, mlx_lm or mlx2 module may load: a meta-path blocker is installed
before the driver is imported, and every test checks that none is present.

Artifacts are synthetic: small metadata files and SPARSE shards (an 8-byte
length, a real safetensors header for the full 49-layer schema, then a
truncate() to the apparent payload size, so no payload is allocated and no
tensor exists). No real model artifact is read.

HOST REFERENCE (test-only). ``REF`` holds a fixed allowlist of functions
and constants extracted by AST from the pinned production inspector
``src/mlx2/adapters/north_mini_code.py`` (sha256 checked) and executed with
stdlib globals only; its imports (contracts, sampling defaults, the drafter
mixin) are never executed and no MLX or model is involved. It is the
production identity recipe and schema check, run on host fixtures; the
driver's own stdlib reimplementation is compared against it. A
hand-written recompute is checked as well.

The native gate ORDER is executed through the driver's seams with host
fakes (``Q._import``, ``Q.build_identity``, ``Q.loaded_module_files``): that
proves the source order of the gates, not that a real build or model passes.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest -p no:cacheprovider \
      -o addopts= -q tests/test_ragged_pld_north_identity_cpu.py
"""

import ast
import builtins
import copy
import hashlib
import json
import math
import os
import pathlib
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

NATIVE = frozenset({"mlx", "mlx_lm", "mlx2"})


def _native_loaded():
    return sorted(name for name in sys.modules if name.split(".")[0] in NATIVE)


class _NativeBlocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in NATIVE:
            raise ImportError(f"native import blocked in a host test: {name}")


assert not _native_loaded(), _native_loaded()
sys.meta_path.insert(0, _NativeBlocker())
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import qualify_ragged_pld as Q

COMMIT = "a" * 40
BASELINE = "649a1ce21f8a61ef319300067872d1218c04fa63"
NORTH_SOURCE = "src/mlx2/adapters/north_mini_code.py"
# Re-pinned 2026-10-01 after 637ff770 (process-wide TF32 constant, one import)
# changed the file but none of REFERENCE_FUNCTIONS / REFERENCE_CONSTANTS
# (AST-compared byte-identical).  The recorded North evidence in
# qualification/runs/mechanism-intake-20260930 was taken at 7b6cecf3....
NORTH_SHA256 = "8e256ddfef2750589838bf5773891c562baab75d0ea6a6ae09ceb1e26f8cfdee"
REFERENCE_FUNCTIONS = ("_load_json", "_safe_index", "_quantized_shapes", "_expected_weight_headers",
                       "_validate_weight_headers", "_unique_pairs", "inspect_artifact")
REFERENCE_CONSTANTS = ("_SAFETENSORS_HEADER_LIMIT", "_DTYPE_BYTES")
REFERENCE_GLOBALS = {"hashlib": hashlib, "json": json, "math": math, "struct": struct, "Path": Path}
METADATA = ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
            "chat_template.jinja", "generation_config.json")


@pytest.fixture(autouse=True)
def host_only():
    assert not _native_loaded()
    yield
    assert not _native_loaded(), _native_loaded()


# ---- host reference: the pinned production inspector, AST-extracted ----

def _north_tree():
    raw = (ROOT / NORTH_SOURCE).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == NORTH_SHA256, "pinned North inspector source moved"
    return ast.parse(raw)


def _reference():
    nodes = []
    for node in _north_tree().body:
        function = isinstance(node, ast.FunctionDef) and node.name in REFERENCE_FUNCTIONS
        constant = (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in REFERENCE_CONSTANTS)
        if function or constant:
            nodes.append(node)
    names = {getattr(n, "name", None) or n.targets[0].id for n in nodes}
    assert names == set(REFERENCE_FUNCTIONS) | set(REFERENCE_CONSTANTS)
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for node in nodes for n in ast.walk(node))
    defined = set()
    for node in nodes:
        for n in ast.walk(node):
            if isinstance(n, (ast.FunctionDef, ast.Lambda)):
                defined |= {a.arg for a in n.args.args + n.args.kwonlyargs}
                if isinstance(n, ast.FunctionDef):
                    defined.add(n.name)  # nested helpers (unique, add_quantized)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                defined.add(n.id)
    loads = {n.id for node in nodes for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    free = loads - defined - names - set(REFERENCE_GLOBALS) - set(dir(builtins))
    assert not free, free  # nothing outside the allowlist and the stdlib globals
    namespace = {"__builtins__": builtins, "__name__": "north_reference_host_only", **REFERENCE_GLOBALS}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), NORTH_SOURCE, "exec"), namespace)  # noqa: S102
    return namespace


REF = _reference()


def _expected_literal():
    fn = next(n for n in _north_tree().body if isinstance(n, ast.FunctionDef) and n.name == "inspect_artifact")
    node = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "expected")
    return ast.literal_eval(node.value)


# ---- synthetic North artifacts (metadata + sparse shards) ----

def north_config(**update):
    config = {**_expected_literal(), "architectures": ["Cohere2MoeForCausalLM"],
              "layer_types": ["full_attention" if i % 4 == 0 else "sliding_attention" for i in range(49)],
              "quantization": {"group_size": 64, "bits": 4}, "eos_token_id": 3}
    config.update(update)
    return config


def _record(dtype, shape, start):
    size = math.prod(shape) * {"BF16": 2, "U32": 4}[dtype]
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [start, start + size]}


def build_headers(schema, names):
    """Contiguous chunks of the schema per shard, sequential offsets."""
    tensors = list(schema)
    per = math.ceil(len(tensors) / len(names))
    headers = {}
    for k, name in enumerate(names):
        header, offset = {"__metadata__": {"format": "mlx"}}, 0
        for tensor in tensors[k * per:(k + 1) * per]:
            header[tensor] = _record(*schema[tensor], offset)
            offset = header[tensor]["data_offsets"][1]
        headers[name] = header
    return headers


def write_shard(item, raw_header, payload):
    with open(item, "wb") as stream:
        stream.write(struct.pack("<Q", len(raw_header)))
        stream.write(raw_header)
        stream.truncate(8 + len(raw_header) + payload)  # sparse: nothing allocated


def payload_of(header):
    """Apparent payload size: the largest end offset, except deliberately out-of-file ones."""
    ends = [r["data_offsets"][1] for k, r in header.items()
            if k != "__metadata__" and isinstance(r, dict) and isinstance(r.get("data_offsets"), list)
            and len(r["data_offsets"]) == 2 and type(r["data_offsets"][1]) is int and r["data_offsets"][1] < BEYOND]
    return max([0, *ends])


BEYOND = 1 << 48  # an offset past any fixture's end; never used as a file size


def make_north(path, *, names=("model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
                               "model-00003-of-00003.safetensors"),
               config=None, schema_config=None, mutate=None, raw_config=None, raw_index=None):
    """A North artifact; ``mutate(headers, weight_map)`` edits before writing. Returns header bytes."""
    path.mkdir(parents=True)
    config = north_config() if config is None else config
    schema = REF["_expected_weight_headers"](schema_config or config)
    headers = build_headers(schema, names)
    weight_map = {t: name for name, header in headers.items() for t in header if t != "__metadata__"}
    if mutate is not None:
        mutate(headers, weight_map)
    (path / "config.json").write_text(raw_config if raw_config is not None else json.dumps(config))
    (path / "model.safetensors.index.json").write_text(
        raw_index if raw_index is not None else json.dumps({"metadata": {}, "weight_map": weight_map}))
    for name, text in (("tokenizer.json", '{"t": 1}'), ("tokenizer_config.json", '{"c": 2}'),
                       ("chat_template.jinja", "{{ m }}"), ("generation_config.json", '{"g": 3}')):
        (path / name).write_text(text)
    raw = {}
    for name, header in headers.items():
        raw[name] = header if isinstance(header, bytes) else json.dumps(header).encode()
        item = path / name
        item.parent.mkdir(parents=True, exist_ok=True)
        write_shard(item, raw[name], 0 if isinstance(header, bytes) else payload_of(header))
    return raw


def independent_fingerprint(path, raw_headers):
    digest = hashlib.sha256()
    for name in METADATA:
        if (path / name).is_file():
            digest.update(name.encode() + (path / name).read_bytes())
    names = sorted(raw_headers)
    for name in names:
        info = (path / name).stat()
        digest.update(json.dumps([name, info.st_size, info.st_mtime_ns]).encode())
    digest.update(json.dumps([hashlib.sha256(raw_headers[n]).hexdigest() for n in names]).encode())
    return digest.hexdigest()


def refused(path, match):
    """The driver refuses with ``match`` and the production inspector refuses too."""
    with pytest.raises(ValueError, match=match):
        Q.artifact_manifest(path)
    with pytest.raises(Exception):  # noqa: B017 - any refusal by the reference
        REF["inspect_artifact"](path)


@pytest.fixture
def north(tmp_path):
    path = tmp_path / "north"
    return SimpleNamespace(path=path, raw=make_north(path))


# ---- reference pinning ----

def test_driver_constants_are_the_pinned_inspector_values():
    assert Q.NORTH_CONFIG == _expected_literal()
    assert Q.NORTH_HEADER_LIMIT == REF["_SAFETENSORS_HEADER_LIMIT"] == 64 << 20
    assert Q.NORTH_DTYPE_BYTES == REF["_DTYPE_BYTES"]
    fn = ast.unparse(next(n for n in _north_tree().body
                          if isinstance(n, ast.FunctionDef) and n.name == "inspect_artifact"))
    assert repr(tuple(METADATA)) in fn and Q.FAMILIES["north"]["metadata"] == METADATA
    assert "if config.get('architectures') != ['Cohere2MoeForCausalLM']" in fn
    assert "for marker in ('mtp.', 'eagle', 'draft')" in fn and Q.NORTH_SPECULATIVE_MARKERS == ("mtp.", "eagle", "draft")
    assert "digest.update(json.dumps(record).encode())" in fn
    assert "digest.update(json.dumps(header_digests).encode())" in fn
    assert fn.index("json.dumps(record)") < fn.index("json.dumps(header_digests)")
    assert "'header_sha256': header_digests" in fn
    assert Q.NORTH_LAYER_TYPES == north_config()["layer_types"]


def test_north_family_files_are_the_route_sources():
    family = Q.FAMILIES["north"]
    assert family["adapter"] == "mlx2.adapters.north_mini_code.NorthMiniCodeAdapter" and family["headers"] is True
    assert family["files"][0] == NORTH_SOURCE
    for name in family["files"]:
        assert (ROOT / name).is_file(), name
    adapter = (ROOT / NORTH_SOURCE).read_text()
    for needle in ("class NorthMiniCodeAdapter(ExternalDraftAdapterMixin)", "runtime.models.cohere2_moe",
                   "runtime.ubc_evict import load_shards_evicting", "runtime.tokenizer_integrity",
                   'self.identity = artifact["identity"]'):
        assert needle in adapter
    assert "from .switch_layers import SwitchGLU" in (ROOT / "src/mlx2/runtime/models/cohere2_moe.py").read_text()
    registry = (ROOT / "src/mlx2/adapters/registry.py").read_text()
    assert '"cohere2_moe": _north_mini_code' in registry and "module.inspect_artifact(path)" in registry


def test_reference_extraction_is_allowlisted_and_runs_without_native_imports():
    assert set(REFERENCE_FUNCTIONS) <= set(REF) and "ModelDescriptor" not in REF and "Capability" not in REF
    assert REF["__name__"] == "north_reference_host_only" and not _native_loaded()


# ---- valid artifacts: driver == production inspector == hand recompute ----

def test_valid_north_manifest_matches_the_production_inspector(north):
    manifest = Q.artifact_manifest(north.path)
    reference = REF["inspect_artifact"](north.path)["identity"]
    assert manifest["family"] == "north" and manifest["adapter"] == Q.FAMILIES["north"]["adapter"]
    assert manifest["fingerprint"] == reference["fingerprint"] == independent_fingerprint(north.path, north.raw)
    assert manifest["header_sha256"] == reference["header_sha256"]
    assert manifest["shards"] == [list(r) for r in reference["files"]] and manifest["path"] == reference["path"]
    assert manifest["header_sha256"] == [hashlib.sha256(north.raw[n]).hexdigest() for n in sorted(north.raw)]
    assert manifest["schema"] == {"tensors": 1226, "shards": 3}
    assert set(manifest["metadata_sha256"]) == set(METADATA)
    assert "never payload bytes" in manifest["fingerprint_scope"]
    assert "not tensor-payload content verification" in manifest["fingerprint_scope"]
    assert "not tensor values" in manifest["header_scope"]


def test_schema_spot_values_are_independent_of_both_implementations():
    schema = Q.north_expected_headers(north_config())
    assert schema == REF["_expected_weight_headers"](north_config()) and len(schema) == 1226
    assert schema["model.embed_tokens.weight"] == ("U32", [262144, 256])
    assert schema["model.embed_tokens.scales"] == ("BF16", [262144, 32])
    assert schema["model.layers.0.mlp.gate_proj.weight"] == ("U32", [3072, 256])
    assert "model.layers.0.mlp.gate.weight" not in schema
    assert schema["model.layers.1.mlp.gate.weight"] == ("U32", [128, 256])
    assert schema["model.layers.48.mlp.switch_mlp.down_proj.weight"] == ("U32", [128, 2048, 96])
    assert schema["model.layers.7.self_attn.k_proj.biases"] == ("BF16", [512, 32])
    assert schema["model.norm.weight"] == ("BF16", [2048]) and "lm_head.weight" not in schema


@pytest.mark.parametrize("names", [
    ("model.safetensors",),
    ("z-first.safetensors", "a-second.safetensors"),
    ("sub/b.safetensors", "a.safetensors", "c.safetensors", "d.safetensors"),
])
def test_shard_count_and_sorted_order_match_the_inspector(tmp_path, names):
    path = tmp_path / "north"
    raw = make_north(path, names=names)
    manifest = Q.artifact_manifest(path)
    reference = REF["inspect_artifact"](path)["identity"]
    assert [r[0] for r in manifest["shards"]] == sorted(names)
    assert manifest["fingerprint"] == reference["fingerprint"] == independent_fingerprint(path, raw)
    assert manifest["header_sha256"] == reference["header_sha256"]


def test_optional_metadata_absent_is_skipped_like_the_inspector(tmp_path):
    path = tmp_path / "north"
    raw = make_north(path)
    for name in ("chat_template.jinja", "generation_config.json"):
        (path / name).unlink()
    manifest = Q.artifact_manifest(path)
    assert manifest["fingerprint"] == REF["inspect_artifact"](path)["identity"]["fingerprint"]
    assert manifest["fingerprint"] == independent_fingerprint(path, raw)
    assert set(manifest["metadata_sha256"]) == set(METADATA[:4])


# ---- the reader touches only the length prefix and the bounded header ----

def test_reader_never_requests_payload_bytes(north, monkeypatch):
    reads, opened = [], {}
    real_open, real_read = Q._open_fd, Q._read_fd

    def spy_open(item, flags):
        fd = real_open(item, flags)
        opened[fd] = Path(item).name
        return fd

    def spy_read(fd, count):
        data = real_read(fd, count)
        reads.append((opened[fd], count, os.lseek(fd, 0, os.SEEK_CUR)))
        return data

    def no_open(file, *args, **kwargs):
        if str(file).endswith(".safetensors"):
            pytest.fail(f"shard opened outside the bounded reader: {file}")
        return real_builtin_open(file, *args, **kwargs)

    real_builtin_open, real_path_open = builtins.open, pathlib.Path.open
    monkeypatch.setattr(Q, "_open_fd", spy_open)
    monkeypatch.setattr(Q, "_read_fd", spy_read)
    monkeypatch.setattr(builtins, "open", no_open)
    monkeypatch.setattr(pathlib.Path, "open", lambda self, *a, **k: (
        pytest.fail(f"shard opened via Path: {self}") if self.suffix == ".safetensors" else real_path_open(self, *a, **k)))
    Q.artifact_manifest(north.path)
    for name, raw in north.raw.items():
        mine = [r for r in reads if r[0] == name]
        bound = 8 + len(raw)
        assert mine and max(position for _, _, position in mine) == bound  # header end, never beyond
        assert sum(count for _, count, _ in mine) == 2 * bound  # two collections, exact requests only
        assert (north.path / name).stat().st_size > 10 * bound  # a large sparse payload stayed unread
    metadata = [r for r in reads if not r[0].endswith(".safetensors")]
    assert {r[0] for r in metadata} == set(METADATA)  # every metadata byte through the guarded reader


# ---- header bounds, truncation, malformed roots ----

def _rewrite(path, name, data):
    (path / name).write_bytes(data)


@pytest.mark.parametrize("data,match", [
    (b"", "Truncated"),
    (b"\x01\x02\x03", "Truncated"),
    (struct.pack("<Q", 0) + b"{}", "header length"),
    (struct.pack("<Q", 100) + b"{}", "header length"),
    (struct.pack("<Q", 4) + b"[1] ", "header object"),
    (struct.pack("<Q", 4) + b"null", "header object"),
    (struct.pack("<Q", 4) + b"{\"a\"", "not valid JSON"),
    (struct.pack("<Q", 4) + b"\xff\xfe\x00\x01", "not valid JSON"),
    (struct.pack("<Q", 9) + b'{"a":"b"}', "Invalid North tensor metadata"),
])
def test_header_bounds_and_roots_refuse(north, data, match):
    _rewrite(north.path, min(north.raw), data)
    refused(north.path, match)


def test_header_longer_than_the_limit_refuses_after_reading_eight_bytes(north, monkeypatch):
    name = min(north.raw)
    item = north.path / name
    with open(item, "wb") as stream:
        stream.write(struct.pack("<Q", (64 << 20) + 1))
        stream.truncate((64 << 20) + 64)
    counts, opened = [], {}
    real_open, real_read = Q._open_fd, Q._read_fd

    def spy_open(item, flags):
        fd = real_open(item, flags)
        opened[fd] = Path(item).name
        return fd

    monkeypatch.setattr(Q, "_open_fd", spy_open)
    monkeypatch.setattr(Q, "_read_fd", lambda fd, n: counts.append((opened[fd], n)) or real_read(fd, n))
    refused(north.path, "header length")
    assert [c for c in counts if c[0].endswith(".safetensors")] == [(name, 8)]


def test_deep_header_nesting_refuses_explicitly(north):
    body = b"[" * 200000 + b"]" * 200000
    _rewrite(north.path, min(north.raw), struct.pack("<Q", len(body)) + body + b"\0" * 8)
    with pytest.raises(ValueError, match="nests too deeply|not valid JSON|header object"):
        Q.artifact_manifest(north.path)


# ---- full schema from the headers ----

FIRST = "model.embed_tokens.weight"
LAYER = "model.layers.5.self_attn.q_proj.weight"


def _shard_of(headers, tensor):
    return next(name for name, header in headers.items() if tensor in header)


def _edit(tensor, fn):
    def mutate(headers, weight_map):
        fn(headers[_shard_of(headers, tensor)], tensor, headers, weight_map)
    return mutate


def _drop(header, tensor, headers, weight_map):
    del header[tensor]
    del weight_map[tensor]


def _set(key, value):
    def fn(header, tensor, headers, weight_map):
        header[tensor][key] = value
    return fn


def _extra(header, tensor, headers, weight_map):
    header["model.layers.5.self_attn.q_norm.weight"] = _record("BF16", [1], payload_of(header))
    weight_map["model.layers.5.self_attn.q_norm.weight"] = _shard_of(headers, tensor)


def _overlap(header, tensor, headers, weight_map):
    start = header[tensor]["data_offsets"][0]
    header["model.layers.5.input_layernorm.weight"]["data_offsets"] = [start, start + 4096]


def _duplicate_across(header, tensor, headers, weight_map):
    other = next(h for name, h in headers.items() if tensor not in h)
    other[tensor] = copy.deepcopy(header[tensor])


def _index_points_elsewhere(header, tensor, headers, weight_map):
    weight_map[tensor] = next(name for name, h in headers.items() if tensor not in h)


def _index_extra(header, tensor, headers, weight_map):
    weight_map["model.layers.5.self_attn.ghost.weight"] = _shard_of(headers, tensor)


def _record_not_object(header, tensor, headers, weight_map):
    header[tensor] = [1, 2, 3]


def _record_extra_key(header, tensor, headers, weight_map):
    header[tensor]["extra"] = 1


def _byte_size(header, tensor, headers, weight_map):
    start, end = header[tensor]["data_offsets"]
    header[tensor]["data_offsets"] = [start, end - 4]


@pytest.mark.parametrize("tensor,fn,match", [
    (LAYER, _drop, "schema mismatch.*missing=1"),
    (LAYER, _extra, "schema mismatch.*extra=1"),
    (LAYER, _set("dtype", "F32"), "dtype/shape"),
    (LAYER, _set("dtype", ["U32"]), "dtype/shape"),
    (LAYER, _set("shape", [4096, -256]), "dtype/shape"),
    (LAYER, _set("shape", [4096, True]), "dtype/shape"),
    (LAYER, _set("shape", "4096x256"), "dtype/shape"),
    (LAYER, _set("data_offsets", [0, 1, 2]), "offsets"),
    (LAYER, _set("data_offsets", [10, 5]), "offsets"),
    (LAYER, _set("data_offsets", [-4, 4]), "offsets"),
    (LAYER, _set("data_offsets", [0.0, 4.0]), "offsets"),
    (LAYER, _set("data_offsets", [0, BEYOND]), "offsets"),
    (LAYER, _byte_size, "byte size"),
    (LAYER, _record_not_object, "tensor metadata"),
    (LAYER, _record_extra_key, "tensor metadata"),
    (LAYER, _overlap, "Overlapping"),
    (FIRST, _duplicate_across, "Duplicate North tensor across shards"),
    (LAYER, _index_points_elsewhere, "index/shard mismatch"),
    (LAYER, _index_extra, "index does not match shard headers"),
])
def test_schema_violations_refuse(tmp_path, tensor, fn, match):
    path = tmp_path / "north"
    make_north(path, mutate=_edit(tensor, fn))
    refused(path, match)


def test_wrong_shape_with_consistent_bytes_is_a_schema_mismatch(tmp_path):
    def fn(header, tensor, headers, weight_map):
        start = header[tensor]["data_offsets"][0]
        header[tensor] = _record("U32", [256, 4096], start)  # transposed, same bytes
    path = tmp_path / "north"
    make_north(path, mutate=_edit(LAYER, fn))
    refused(path, "schema mismatch.*wrong=1")


def test_duplicate_key_inside_one_header_refuses(tmp_path):
    def mutate(headers, weight_map):
        name = _shard_of(headers, LAYER)
        text = json.dumps(headers[name])
        record = json.dumps({LAYER: headers[name][LAYER]})[1:-1]
        headers[name] = (text[:-1] + ", " + record + "}").encode()
    path = tmp_path / "north"
    make_north(path, mutate=mutate)
    refused(path, "duplicate JSON key")


# ---- topology, norms, layer order, quantization ----

@pytest.mark.parametrize("update,match", [
    ({"rms_norm_eps": None}, "topology"),
    ({"layer_norm_eps": 1e-6}, "topology"),
    ({"num_hidden_layers": 48}, "topology"),
    ({"tie_word_embeddings": True}, "topology"),
    ({"use_qk_norm": True}, "topology"),
    ({"layer_types": ["sliding_attention"] * 49}, "layer order"),
    ({"layer_types": None}, "layer order"),
    ({"architectures": ["CohereForCausalLM"]}, "architecture"),
])
def test_topology_norm_layer_order_and_architecture_refuse(tmp_path, update, match):
    config = north_config(**update)
    assert Q.artifact_family(config)[0] is None and match in Q.artifact_family(config)[1]
    path = tmp_path / "north"
    make_north(path, config=config, schema_config=north_config())
    refused(path, match)


def test_dropped_rms_norm_eps_key_refuses():
    config = north_config()
    del config["rms_norm_eps"]
    assert Q.artifact_family(config)[0] is None


def test_incomplete_cohere2_moe_config_stays_refused():
    family, why = Q.artifact_family({"model_type": "cohere2_moe", "num_hidden_layers": 49, "hidden_size": 2048})
    assert family is None and "topology" in why
    assert Q.artifact_family(north_config()) == ("north", None)
    assert Q.artifact_family({**north_config(), "dflash_config": {}})[0] is None


OVERRIDES = {"group_size": 64, "bits": 4, "model.embed_tokens": {"group_size": 32, "bits": 8},
             "model.layers.3.mlp.switch_mlp.down_proj": {"group_size": 128, "bits": 8}}


def test_per_module_quantization_overrides_set_the_schema(tmp_path):
    config = north_config(quantization=OVERRIDES)
    schema = Q.north_expected_headers(config)
    assert schema == REF["_expected_weight_headers"](config)
    assert schema["model.embed_tokens.weight"] == ("U32", [262144, 512])
    assert schema["model.embed_tokens.scales"] == ("BF16", [262144, 64])
    assert schema["model.layers.3.mlp.switch_mlp.down_proj.weight"] == ("U32", [128, 2048, 192])
    path = tmp_path / "north"
    make_north(path, config=config)
    assert Q.artifact_manifest(path)["fingerprint"] == REF["inspect_artifact"](path)["identity"]["fingerprint"]
    legacy = tmp_path / "legacy"
    make_north(legacy, config={**{k: v for k, v in config.items() if k != "quantization"},
                               "quantization_config": OVERRIDES})
    assert Q.artifact_manifest(legacy)["schema"]["tensors"] == 1226


def test_headers_quantized_differently_from_the_config_refuse(tmp_path):
    path = tmp_path / "north"
    make_north(path, config=north_config(quantization=OVERRIDES), schema_config=north_config())
    refused(path, "schema mismatch.*wrong=")


@pytest.mark.parametrize("quant,match", [
    (None, "quantization metadata"),
    ([64, 4], "quantization metadata"),
    ({"group_size": 64, "bits": 3}, "quantization parameters"),
    ({"group_size": 64, "bits": True}, "quantization parameters"),
    ({"group_size": 64.0, "bits": 4}, "quantization parameters"),
    ({"group_size": 0, "bits": 4}, "quantization parameters"),
    ({"group_size": -64, "bits": 4}, "quantization parameters"),
    ({"group_size": 100, "bits": 4}, "not integral"),
    ({"group_size": 64, "bits": 4, "model.embed_tokens": 8}, "override"),
    ({"group_size": 64, "bits": 4, "model.embed_tokens": {"bits": 8}}, "quantization parameters"),
])
def test_invalid_quantization_metadata_refuses(tmp_path, quant, match):
    config = north_config(quantization=quant)
    with pytest.raises(ValueError, match=match):
        Q.north_expected_headers(config)
    path = tmp_path / "north"
    make_north(path, config=config, schema_config=north_config())
    refused(path, match)


# ---- metadata and index JSON ----

@pytest.mark.parametrize("raw_config,raw_index,match", [
    ('{"model_type": "cohere2_moe", "model_type": "cohere2_moe"}', None, "duplicate JSON key|topology"),
    (None, '{"weight_map": {}, "weight_map": {}}', "duplicate JSON key"),
    (None, "[]", "nonempty weight index"),
    (None, '{"weight_map": {}}', "nonempty weight index"),
    (None, '{"weight_map": []}', "nonempty weight index"),
    (None, '{"weight_map": {"model.embed_tokens.weight": 7}}', "non-string shard"),
    (None, "{not json", "not valid JSON|unreadable"),
    ("[]", None, "not a JSON object"),
])
def test_metadata_json_refuses(tmp_path, raw_config, raw_index, match):
    path = tmp_path / "north"
    make_north(path, raw_config=raw_config, raw_index=raw_index)
    refused(path, match)


def test_duplicate_key_in_a_valid_config_refuses(tmp_path):
    text = json.dumps(north_config())
    path = tmp_path / "north"
    make_north(path, raw_config=text[:-1] + ', "hidden_size": 2048}')
    refused(path, "duplicate JSON key 'hidden_size' in config.json")


@pytest.mark.parametrize("tensor", ["model.mtp.0.weight", "model.layers.1.eagle_fc.weight",
                                    "draft.lm_head.weight", "lm_head.weight"])
def test_embedded_draft_or_untied_head_tensors_refuse(tmp_path, tensor):
    def mutate(headers, weight_map):
        name = max(headers)
        start = payload_of(headers[name])
        headers[name][tensor] = _record("BF16", [2], start)
        weight_map[tensor] = name
    path = tmp_path / "north"
    make_north(path, mutate=mutate)
    refused(path, "speculative|tied embedding")


def test_missing_embedding_refuses(tmp_path):
    path = tmp_path / "north"
    make_north(path, mutate=_edit(FIRST, _drop))
    refused(path, "tied embedding")


# ---- shard paths ----

def _rename_index(path, old, new):
    index = json.loads((path / "model.safetensors.index.json").read_text())
    index["weight_map"] = {k: (new if v == old else v) for k, v in index["weight_map"].items()}
    (path / "model.safetensors.index.json").write_text(json.dumps(index))


def test_unsafe_shard_paths_refuse(tmp_path):
    """Escaping, wrong-suffix and missing shards: the driver and the inspector both refuse."""
    cases = {
        "traversal": lambda p, n: ((p / n).rename(tmp_path / n), _rename_index(p, n, f"../{n}")),
        "suffix": lambda p, n: ((p / n).rename(p / "w.bin"), _rename_index(p, n, "w.bin")),
        "missing": lambda p, n: (p / n).unlink(),
        "escape": lambda p, n: ((p / n).rename(tmp_path / f"outside-{p.name}.safetensors"),
                                (p / n).symlink_to(tmp_path / f"outside-{p.name}.safetensors")),
    }
    matches = {"missing": "missing weight shard"}
    for label, prepare in cases.items():
        path = tmp_path / label
        raw = make_north(path)
        prepare(path, min(raw))
        refused(path, matches.get(label, "not a local .safetensors file"))


@pytest.mark.parametrize("rename", [lambda p, n: str(p / n), lambda p, n: f"../{p.name}/{n}"])
def test_absolute_or_dotdot_names_refuse_even_when_they_land_inside(tmp_path, rename):
    """Stricter than the inspector (which resolves these inside and accepts them): refusal only."""
    path = tmp_path / "north"
    name = min(make_north(path))
    _rename_index(path, name, rename(path, name))
    with pytest.raises(ValueError, match="is not a local"):
        Q.artifact_manifest(path)


def test_two_index_names_for_one_file_refuse(tmp_path):
    path = tmp_path / "north"
    raw = make_north(path, names=("a.safetensors", "b.safetensors"))
    (path / "b.safetensors").unlink()
    (path / "b.safetensors").symlink_to(path / "a.safetensors")
    assert raw
    refused(path, "same file|Duplicate North tensor")
    hard = tmp_path / "hard"
    make_north(hard, names=("a.safetensors", "b.safetensors"))
    (hard / "b.safetensors").unlink()
    os.link(hard / "a.safetensors", hard / "b.safetensors")
    refused(hard, "same file")


def test_metadata_symlink_escaping_the_artifact_refuses(tmp_path):
    path = tmp_path / "north"
    make_north(path)
    outside = tmp_path / "outside-tokenizer.json"
    outside.write_text('{"t": 1}')
    (path / "tokenizer.json").unlink()
    (path / "tokenizer.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks are refused"):
        Q.artifact_manifest(path)


def _guard_opens(monkeypatch):
    opened = []
    real_open = Q._open_fd
    monkeypatch.setattr(Q, "_open_fd", lambda item, flags: opened.append(Path(item).name) or real_open(item, flags))
    return opened


@pytest.mark.parametrize("name", ["config.json", "model.safetensors.index.json", "tokenizer.json",
                                  "generation_config.json"])
@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_metadata_cannot_redirect_a_read_into_shard_payload(tmp_path, monkeypatch, name, link):
    """A metadata name linked to a shard inside the artifact refuses before any byte is read."""
    path = tmp_path / "north"
    raw = make_north(path)
    shard = path / min(raw)
    (path / name).unlink()
    if link == "symlink":
        (path / name).symlink_to(shard)
    else:
        os.link(shard, path / name)
    opened = _guard_opens(monkeypatch)
    with pytest.raises(ValueError, match="symlinks are refused|hard links"):
        Q.artifact_manifest(path)
    assert name not in opened and not any(o.endswith(".safetensors") for o in opened)


def test_initial_config_read_is_guarded_for_every_family(tmp_path, monkeypatch):
    path = tmp_path / "muse"
    path.mkdir()
    (path / "real.json").write_text(json.dumps({"model_type": "muse_glimmer"}))
    (path / "config.json").symlink_to(path / "real.json")
    opened = _guard_opens(monkeypatch)
    with pytest.raises(ValueError, match="config.json unreadable.*symlinks are refused"):
        Q.artifact_manifest(path)
    assert opened == []


def test_oversized_metadata_refuses_before_reading(north, monkeypatch):
    monkeypatch.setattr(Q, "MAX_METADATA_BYTES", 8)
    opened = _guard_opens(monkeypatch)
    with pytest.raises(ValueError, match="exceeds 8 bytes"):
        Q.artifact_manifest(north.path)
    assert opened == []


def test_metadata_replaced_at_its_path_while_read_refuses(north, monkeypatch):
    real_read = Q._read_fd
    item = north.path / "tokenizer.json"
    state = {"done": False}

    def read(fd, count):
        data = real_read(fd, count)
        if not state["done"] and os.fstat(fd).st_ino == item.stat().st_ino:
            state["done"] = True
            info = item.stat()
            replacement = north.path / "replacement.tmp"
            replacement.write_bytes(data)
            os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
            os.replace(replacement, item)  # same bytes, size and mtime; new inode
        return data

    monkeypatch.setattr(Q, "_read_fd", read)
    with pytest.raises(ValueError, match="metadata tokenizer.json changed or was replaced while read"):
        Q.artifact_manifest(north.path)


def test_shard_directory_and_metadata_directory_refuse(tmp_path):
    path = tmp_path / "north"
    raw = make_north(path)
    (path / "chat_template.jinja").unlink()
    (path / "chat_template.jinja").mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        Q.artifact_manifest(path)
    other = tmp_path / "other"
    make_north(other)
    (other / min(raw)).unlink()
    (other / min(raw)).mkdir()
    refused(other, "missing weight shard")


# ---- drift: metadata, header, stat, mid-preflight, during a read ----

def test_metadata_header_and_stat_changes_move_the_identity(north):
    base = Q.artifact_manifest(north.path)
    (north.path / "tokenizer.json").write_text('{"t": 2}')
    tok = Q.artifact_manifest(north.path)
    assert tok["fingerprint"] != base["fingerprint"] and tok["header_sha256"] == base["header_sha256"]
    name = sorted(north.raw)[1]
    item = north.path / name
    info = item.stat()
    raw = north.raw[name].replace(b'"format": "mlx"', b'"format": "xlm"')
    assert len(raw) == len(north.raw[name])
    with open(item, "r+b") as stream:
        stream.seek(8)
        stream.write(raw)
    os.utime(item, ns=(info.st_atime_ns, info.st_mtime_ns))  # same size, restored mtime
    head = Q.artifact_manifest(north.path)
    assert head["shards"] == tok["shards"] and head["header_sha256"] != tok["header_sha256"]
    assert head["fingerprint"] != tok["fingerprint"]
    assert head["fingerprint"] == REF["inspect_artifact"](north.path)["identity"]["fingerprint"]
    os.utime(item, ns=(info.st_atime_ns, info.st_mtime_ns + 1000))
    moved = Q.artifact_manifest(north.path)
    assert moved["shards"] != head["shards"] and moved["fingerprint"] != head["fingerprint"]


def test_payload_rewrite_at_same_size_and_mtime_is_not_seen(north):
    """Documented limitation: header/stat identity is not payload verification."""
    base = Q.artifact_manifest(north.path)
    item = north.path / min(north.raw)
    info = item.stat()
    with open(item, "r+b") as stream:
        stream.seek(8 + len(north.raw[min(north.raw)]) + 4096)
        stream.write(b"\x7f" * 16)
    os.utime(item, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert Q.artifact_manifest(north.path) == base
    assert "payload rewritten at the same size with a restored mtime is not seen" in base["fingerprint_scope"]


def test_change_between_the_two_collections_refuses(north, monkeypatch):
    real = Q._north_collect
    calls = []

    def collect(path):
        calls.append(path)
        if len(calls) == 2:
            (north.path / "generation_config.json").write_text('{"g": 9}')
        return real(path)

    monkeypatch.setattr(Q, "_north_collect", collect)
    with pytest.raises(ValueError, match="changed during the preflight"):
        Q.artifact_manifest(north.path)


def test_header_rewritten_while_read_refuses(north, monkeypatch):
    real_read = Q._read_fd
    target = min(north.raw)
    state = {"done": False}

    def read(fd, count):
        data = real_read(fd, count)
        if count == 8 and not state["done"] and os.fstat(fd).st_ino == (north.path / target).stat().st_ino:
            state["done"] = True
            item = north.path / target
            info = item.stat()
            os.utime(item, ns=(info.st_atime_ns, info.st_mtime_ns + 5000))
        return data

    monkeypatch.setattr(Q, "_read_fd", read)
    with pytest.raises(ValueError, match="changed while its header was read"):
        Q.artifact_manifest(north.path)


def test_shard_replaced_at_its_path_while_read_refuses(north, monkeypatch):
    """Same bytes, size and mtime under a new inode: only the post-read path check sees it."""
    real_read = Q._read_fd
    target = north.path / min(north.raw)
    state = {"done": False}

    def read(fd, count):
        data = real_read(fd, count)
        if count == 8 and not state["done"] and os.fstat(fd).st_ino == target.stat().st_ino:
            state["done"] = True
            info = target.stat()
            replacement = north.path / "copy.tmp"
            write_shard(replacement, north.raw[target.name], info.st_size - 8 - len(north.raw[target.name]))
            os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
            os.replace(replacement, target)
        return data

    monkeypatch.setattr(Q, "_read_fd", read)
    with pytest.raises(ValueError, match="was replaced at its path while its header was read"):
        Q.artifact_manifest(north.path)


def test_identical_replacement_between_collections_is_seen_through_the_inode(north, monkeypatch):
    real = Q._north_collect
    calls = []

    def collect(path):
        calls.append(path)
        if len(calls) == 2:
            target = north.path / min(north.raw)
            info = target.stat()
            replacement = north.path / "copy.tmp"
            write_shard(replacement, north.raw[target.name], info.st_size - 8 - len(north.raw[target.name]))
            os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
            os.replace(replacement, target)
        return real(path)

    monkeypatch.setattr(Q, "_north_collect", collect)
    with pytest.raises(ValueError, match="changed during the preflight"):
        Q.artifact_manifest(north.path)


def test_file_swapped_between_stat_and_open_refuses(north, monkeypatch):
    real = Q._fstat_fd
    monkeypatch.setattr(Q, "_fstat_fd", lambda fd: os.stat_result((*real(fd)[:1], real(fd).st_ino + 1,
                                                                    *real(fd)[2:10])))
    with pytest.raises(ValueError, match="changed between stat and open"):
        Q.artifact_manifest(north.path)


def test_short_read_refuses_as_truncated(north, monkeypatch):
    real_open, real_read = Q._open_fd, Q._read_fd
    shards = set()

    def spy_open(item, flags):
        fd = real_open(item, flags)
        if str(item).endswith(".safetensors"):
            shards.add(fd)
        return fd

    monkeypatch.setattr(Q, "_open_fd", spy_open)
    monkeypatch.setattr(Q, "_read_fd", lambda fd, n: b"" if fd in shards and n != 8 else real_read(fd, n))
    with pytest.raises(ValueError, match="Truncated"):
        Q.artifact_manifest(north.path)


# ---- preflight and run_all: refusals before any native import ----

class FakeGit:
    def __init__(self):
        self.head, self.status = COMMIT, ""

    def __call__(self, *args):
        if args[0] == "rev-parse":
            return self.head + "\n"
        if args[0] == "status":
            return self.status
        if args[0] == "ls-files":
            return "\n".join(args[3:])
        raise AssertionError(f"unexpected git call {args}")


@pytest.fixture
def git(monkeypatch):
    fake = FakeGit()
    monkeypatch.setattr(Q, "_git", fake)
    return fake


def native_args(path):
    return Q.resolve_args(Q.build_parser(), ["--i-own-the-gpu", "--model", str(path),
                                             "--out", str(Path(path).parent / "receipt.json")])


def test_preflight_binds_north_with_its_route_files(north, git):
    gate = Q.preflight(native_args(north.path))
    assert gate["refusals"] == [] and gate["artifact"]["family"] == "north"
    assert gate["required_files"] == list(Q.IDENTITY_FILES) + list(Q.FAMILIES["north"]["files"])
    assert gate["artifact"]["header_sha256"] == REF["inspect_artifact"](north.path)["identity"]["header_sha256"]
    assert not _native_loaded()


@pytest.mark.parametrize("error", [PermissionError(13, "denied"), OSError(5, "io error")])
def test_shard_open_failure_is_a_preflight_refusal(north, git, monkeypatch, error):
    real_open = Q._open_fd

    def fail(item, flags):
        if str(item).endswith(".safetensors"):
            raise error
        return real_open(item, flags)
    monkeypatch.setattr(Q, "_open_fd", fail)
    gate = Q.preflight(native_args(north.path))
    assert gate["artifact"] is None and any(p.startswith("artifact: ") for p in gate["refusals"])


def test_collector_type_error_becomes_a_refusal(north, git, monkeypatch):
    monkeypatch.setattr(Q, "north_expected_headers", lambda config: (_ for _ in ()).throw(TypeError("boom")))
    gate = Q.preflight(native_args(north.path))
    assert any("cannot be bound: TypeError" in p for p in gate["refusals"])


@pytest.mark.parametrize("prepare,match", [
    (lambda p: _rewrite(p, "model-00001-of-00003.safetensors", b"\0\0"), "Truncated"),
    (lambda p: (p / "config.json").write_text(json.dumps(north_config(rms_norm_eps=1e-5))), "topology"),
    (lambda p: (p / "model-00002-of-00003.safetensors").unlink(), "missing weight shard"),
])
def test_run_all_refuses_north_before_any_native_import(north, git, monkeypatch, prepare, match):
    prepare(north.path)
    monkeypatch.setattr(Q, "_import", lambda name: pytest.fail(f"native import {name}"))
    monkeypatch.setattr(Q.Driver, "__init__", lambda *a, **k: pytest.fail("Driver must not be constructed"))
    record = Q.run_all(native_args(north.path))
    assert record["executed"] is False and record["arms_executed"] == [] and record["refused_at"] == Q.STAGE_PREFLIGHT
    assert any(match in p for p in record["refusals"]) and record["results"] == {}


def test_deeply_nested_config_is_an_unexecuted_refusal(north, git, monkeypatch):
    (north.path / "config.json").write_text("[" * 200000 + "]" * 200000)
    monkeypatch.setattr(Q, "_import", lambda name: pytest.fail(f"native import {name}"))
    monkeypatch.setattr(Q.Driver, "__init__", lambda *a, **k: pytest.fail("Driver must not be constructed"))
    record = Q.run_all(native_args(north.path))
    assert record["executed"] is False and record["arms_executed"] == [] and record["refused_at"] == Q.STAGE_PREFLIGHT
    assert "artifact: config.json unreadable: nests too deeply to parse" in record["refusals"]


# ---- Driver gates with host fakes: adapter header identity before arms and after ----

class FakeNorthAdapter:
    """Host stand-in whose identity is the production inspector's own output."""

    tamper = None

    def __init__(self, model):
        self.identity = REF["inspect_artifact"](model)["identity"]
        if FakeNorthAdapter.tamper:
            FakeNorthAdapter.tamper(self.identity)
        self.model = object()
        self.draft_model = None
        self.environment = {}
        self.tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: [ord(c) % 97 for c in text])


FakeNorthAdapter.__module__ = "mlx2.adapters.north_mini_code"
FakeNorthAdapter.__qualname__ = "NorthMiniCodeAdapter"


@pytest.fixture
def world(north, git, monkeypatch, tmp_path):
    events = []
    registry = SimpleNamespace(resolve_adapter=lambda model, **kw: events.append("resolve") or FakeNorthAdapter)
    serving = SimpleNamespace(generation_stop_token_ids=lambda adapter: ())

    def imp(name):
        events.append(f"import {name}")
        return {"mlx.core": SimpleNamespace(name="host mx"), "mlx2.adapters.registry": registry,
                "mlx2.serving": serving}.get(name, SimpleNamespace(name=name))

    build = {"version": "0", "package": "mlx", "device": "Device(gpu, 0)", "path": str(tmp_path),
             "metallib_sha256": "0" * 64}
    modules = {"mlx2": str(ROOT / "src/mlx2/__init__.py"), FakeNorthAdapter.__module__: str(ROOT / NORTH_SOURCE)}
    monkeypatch.setattr(Q, "_import", imp)
    monkeypatch.setattr(Q, "build_identity", lambda mx: events.append("build") or dict(build))
    monkeypatch.setattr(Q, "loaded_module_files", lambda modules_=None: dict(modules))
    monkeypatch.setattr(FakeNorthAdapter, "tamper", None)
    real_init = FakeNorthAdapter.__init__
    monkeypatch.setattr(FakeNorthAdapter, "__init__", lambda self, model: (events.append("construct"),
                                                                          real_init(self, model))[0])
    return SimpleNamespace(path=north.path, raw=north.raw, events=events)


def test_north_driver_binds_adapter_header_identity_before_any_arm(world):
    args = native_args(world.path)
    gate = Q.preflight(args)
    driver = Q.Driver(args, gate)
    snapshot = driver.native_identity["adapter_identity"]
    assert snapshot["header_sha256"] == gate["artifact"]["header_sha256"]
    assert snapshot["fingerprint"] == gate["artifact"]["fingerprint"]
    order = list(dict.fromkeys(e for e in world.events if e in (
        "import mlx.core", "build", "import mlx2.adapters.registry", "resolve", "construct", "import mlx2.serving")))
    assert order == ["import mlx.core", "build", "import mlx2.adapters.registry", "resolve", "construct",
                     "import mlx2.serving"]
    after, refusals = Q.post_run_identity(args, gate, driver)
    assert refusals == [] and after["adapter_identity"] == snapshot
    assert after["artifact"]["header_sha256"] == gate["artifact"]["header_sha256"]


@pytest.mark.parametrize("tamper,match", [
    (lambda i: i["header_sha256"].reverse(), "header digests differ"),
    (lambda i: i.pop("header_sha256"), "header digests differ"),
    (lambda i: i.update(fingerprint="f" * 64), "fingerprint differs"),
])
def test_adapter_header_mismatch_refuses_before_any_arm(world, monkeypatch, tamper, match):
    monkeypatch.setattr(FakeNorthAdapter, "tamper", staticmethod(tamper))
    args = native_args(world.path)
    with pytest.raises(Q.IdentityRefusal) as caught:
        Q.Driver(args, Q.preflight(args))
    assert caught.value.stage == Q.STAGE_ADAPTER and any(match in p for p in caught.value.refusals)
    assert "import mlx2.serving" not in world.events


def test_header_rewrite_during_the_run_refuses_post_run(world):
    args = native_args(world.path)
    gate = Q.preflight(args)
    driver = Q.Driver(args, gate)
    name = sorted(world.raw)[2]
    item = world.path / name
    info = item.stat()
    raw = world.raw[name].replace(b'"format": "mlx"', b'"format": "MLX"')
    with open(item, "r+b") as stream:
        stream.seek(8)
        stream.write(raw)
    os.utime(item, ns=(info.st_atime_ns, info.st_mtime_ns))
    _after, refusals = Q.post_run_identity(args, gate, driver)
    assert "artifact header_sha256 changed during the run" in refusals
    assert "artifact fingerprint changed during the run" in refusals
    assert "artifact shards changed during the run" not in refusals
    assert "artifact shard_file_ids changed during the run" not in refusals


def test_header_corruption_after_the_arms_is_a_post_run_refusal(world):
    args = native_args(world.path)
    gate = Q.preflight(args)
    driver = Q.Driver(args, gate)
    _rewrite(world.path, min(world.raw), b"\0")
    after, refusals = Q.checked_post_run_identity(args, gate, driver)
    assert after["artifact"] is None and any("after the arms: artifact: Truncated" in p for p in refusals)


# ---- preservation and source order ----

def _baseline_tree():
    try:
        text = subprocess.run(["git", "show", f"{BASELINE}:scripts/qualify_ragged_pld.py"], cwd=ROOT,
                              capture_output=True, text=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip("baseline revision not readable")
    return ast.parse(text)


def _definitions(tree):
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = ast.dump(node)
        elif isinstance(node, ast.ClassDef):
            out.update({f"{node.name}.{n.name}": ast.dump(n) for n in node.body if isinstance(n, ast.FunctionDef)})
    return out


CHANGED = {"artifact_family", "artifact_manifest", "adapter_identity_snapshot", "adapter_identity_refusals",
           "post_run_identity",
           # 451f5d30 (survivor continuation after lane removal) changed run_all on purpose
           "run_all"}


def test_every_other_baseline_definition_is_unchanged():
    before, now = _definitions(_baseline_tree()), _definitions(ast.parse((ROOT / "scripts/qualify_ragged_pld.py").read_text()))
    assert set(before) <= set(now)
    moved = sorted(name for name in before if before[name] != now[name])
    assert moved == sorted(CHANGED)
    for name in ("Driver.__init__", "Driver.run", "Driver.continuation", "compare", "coverage",
                 "row_evidence_refusal", "continuation_refusal", "lane_row_evidence", "preflight",
                 "source_identity", "checked_post_run_identity", "build_parser", "resolve_args"):
        assert before[name] == now[name], name


def test_muse_and_qwen_recipes_are_unchanged():
    tree = _baseline_tree()
    families = next(n for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "FAMILIES")
    scope = next(n for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "FINGERPRINT_SCOPE")
    old = {k.value: ast.unparse(v) for k, v in zip(families.value.keys, families.value.values)}
    assert set(old) == {"muse", "qwen38", "qwen36"} and set(Q.FAMILIES) == set(old) | {"north"}
    namespace = {"MUSE_METADATA": Q.MUSE_METADATA, "QWEN_METADATA": Q.QWEN_METADATA}
    for name, source in old.items():
        assert eval(source, namespace) == Q.FAMILIES[name], name  # tuple literals from the baseline
    assert ast.literal_eval(scope.value) == Q.FINGERPRINT_SCOPE


def test_new_north_code_is_stdlib_only_and_never_uses_the_native_seam():
    tree = ast.parse((ROOT / "scripts/qualify_ragged_pld.py").read_text())
    names = {"north_config_refusal", "_unique_object", "_strict_json", "_quantized_shapes",
             "north_expected_headers", "_stat_key", "_read_exact", "north_shard_header", "_check_north_header",
             "_schema_mismatch", "_north_collect", "north_artifact_manifest", "_read_metadata", "_config_json"}
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}
    assert set(functions) == names
    for fn in functions.values():
        text = ast.unparse(fn)
        assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(fn)), fn.name
        assert "_import" not in text and "mlx" not in text.replace("mlx2", ""), fn.name
        assert "importlib" not in text and "exec(" not in text and "eval(" not in text, fn.name
    top = {a.name for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert top == {"annotations", "argparse", "hashlib", "importlib", "json", "math", "os", "struct", "subprocess",
                   "sys", "time", "pairwise", "Path", "S_ISREG"}


def test_artifact_manifest_routes_north_before_the_stat_only_recipe():
    fn = ast.unparse(next(n for n in ast.parse((ROOT / "scripts/qualify_ragged_pld.py").read_text()).body
                          if isinstance(n, ast.FunctionDef) and n.name == "artifact_manifest"))
    assert fn.index("north_artifact_manifest(path)") < fn.index("model.safetensors.index.json")
    collect = ast.unparse(next(n for n in ast.parse((ROOT / "scripts/qualify_ragged_pld.py").read_text()).body
                               if isinstance(n, ast.FunctionDef) and n.name == "_north_collect"))
    assert collect.index("north_config_refusal") < collect.index("north_shard_header")
    assert collect.index("json.dumps(record)") < collect.index("json.dumps(headers)")
