"""Bug sweep 2026-11-21, amoeba-runtime lane (AMRT-1 … AMRT-7). CPU only.

The resident calculator header and the ``prepare`` kernel body are compiled as C++ (the same
construction as tests/test_amoeba_calc_resident.py); no Metal kernel is built or run. Every
Python calculator tile here pins ``arithmetic_backend: cpu`` (the v4 default is ``gpu``).
"""

import ctypes
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mlx2.runtime import amoeba_calc_gpu as G
from mlx2.runtime import amoeba_calc_requests as Q
from mlx2.runtime import amoeba_calc_resident as R
from mlx2.runtime import amoeba_counting as C
from mlx2.runtime import amoeba_det_tiles as D
from mlx2.runtime.amoeba_blocks import layout as L


def _prepare_body(monkeypatch_context):
    """The ``prepare`` kernel body as written in the module, without building a Metal kernel."""
    import mlx.core as mx

    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch_context.setattr(mx.fast, "metal_kernel", capture)
    R.kernel.__wrapped__("prepare")
    assert captured["input_names"][:5] == ["tokens", "pool", "offsets", "lengths", "cues"]
    return captured["source"]


@pytest.fixture(scope="module")
def portable(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sweep1121-resident-cpp")
    source, lib = tmp / "oracle.cpp", tmp / "oracle.dylib"
    with pytest.MonkeyPatch.context() as patch:
        body = _prepare_body(patch)
    source.write_text(
        """#include <cstdint>
using uchar=unsigned char; using uint=unsigned int;
#define thread
#define device
#define constant
#define RESIDENT_ARITHMETIC 1
"""
        + G._HEADER
        + R.header()
        + """
extern "C" int calculate(const uchar *s,int len,int mode,int *out) {
 int program[1025]={}; uchar normalized[4096]={};
 resident_parse(s,len,mode,program,normalized);
 resident_evaluate(program,normalized,out);
 return program[0];
}
extern "C" int prepare(const int *tokens,int nt,const uchar *pool,const int *offsets,
                       const int *lengths,const int *cues,int vocabulary,const uchar *request,
                       const int *meta,const int *state,uchar *source) {
 int tokens_shape[1]={nt},cues_shape[1]={vocabulary},offsets_shape[1]={vocabulary};
 struct { unsigned x; } thread_position_in_grid={0};
 int length[1]={0};
 [&]{
"""
        + body
        + """
 }();
 return length[0];
}
"""
    )
    subprocess.run(["clang++", "-std=c++17", "-O1", "-shared", "-fPIC", str(source), "-o", str(lib)],
                   check=True, capture_output=True)
    dll = ctypes.CDLL(str(lib))
    dll.calculate.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    dll.calculate.restype = ctypes.c_int
    ints, chars = ctypes.POINTER(ctypes.c_int), ctypes.c_char_p
    dll.prepare.argtypes = [ints, ctypes.c_int, chars, ints, ints, ints, ctypes.c_int, chars, ints, ints, chars]
    dll.prepare.restype = ctypes.c_int

    def run(text, mode=0):
        data = text.encode()
        out = (ctypes.c_int * 268)()
        n = dll.calculate(data, len(data), mode, out)
        if n < 1 or out[0]:
            return None
        digits = "".join(str(x) for x in list(out)[4:4 + out[3]][::-1])
        scale = out[2]
        if scale:
            digits = digits.rjust(scale + 1, "0")
            digits = digits[:-scale] + "." + digits[-scale:]
        return ("-" if out[1] < 0 else "") + digits

    def prepare(prompt, generated):
        """Byte-level vocabulary (token id == byte); returns the source handed to the parser."""
        tokens = list(prompt.encode()) + list(generated.encode())
        array = lambda values: (ctypes.c_int * len(values))(*values)  # noqa: E731
        meta = array([len(prompt.encode()), 0, 1, 0, 0])
        out = ctypes.create_string_buffer(4096)
        n = dll.prepare(array(tokens), len(tokens), bytes(range(256)), array(list(range(256))),
                        array([1] * 256), array([1] * 256), 256, b"\0" * 8, meta, array([0] * 288), out)
        return out.raw[:n].decode() if n > 0 else None

    run.prepare = prepare
    return run


def _reference(text):
    request = Q.parse_request(text)
    return None if request is None else Q.evaluate(request, "cpu")[0]


# --- AMRT-1 -------------------------------------------------------------------------------------

# Bare user turns that are ranges, dates, ratios, scores, phones or versions, not arithmetic.
_BARE_REFUSED = [
    "10-20", "5-10", "9-5", "18-25?", "2024-05", "1-800", "555-1234", "50/50", "80/20.", "3-2",
    "1.2-3", "7-11", "12/25", "1/2", "4-6+1", "1-2-3-4", "10−20", "10-20=", "2019 - 2024",
    "2026-11-17", "12/25/2024",
]
# Bare formulas the resident gate accepts on purpose (tests/test_amoeba_calc_resident.py). The
# Python request reference needs a wrapper for them, so their values are compared with it under
# the same "Calculate " wrapper.
_BARE_ACCEPTED = [
    ("0.1+0.2", "0.3"), ("12,345,678,901 * 9", "111111110109"), ("-.25 * 8", "-2"),
    ("2^-3", "0.125"), ("-2^2", "-4"), ("(-2)^2", "4"), ("10!", "3628800"), ("1 / 3 * 3", "1"),
    ("12 × 4 ÷ 3 − 1", "15"), ("((12/7)+(3/4))*(7*4)", "69"), (r"\frac{3}{4}+1", "1.75"),
    (r"\binom{20}{3}", "1140"),
]


@pytest.mark.parametrize("text", _BARE_REFUSED)
def test_amrt1_bare_range_like_turn_is_refused_like_the_reference(portable, text):
    assert _reference(text) is None
    assert portable(text) is None


@pytest.mark.parametrize("text,expected", _BARE_ACCEPTED)
def test_amrt1_bare_formulas_stay_accepted_with_the_reference_value(portable, text, expected):
    assert portable(text) == expected
    assert portable("Calculate " + text) == _reference("Calculate " + text) == expected


@pytest.mark.parametrize("text", [t for t in _BARE_REFUSED if not t.endswith("=")])
def test_amrt1_wrapped_requests_agree_with_the_reference(portable, text):
    """An instruction wrapper makes the same text explicit math: both sides accept or both
    refuse, with one value. (A wrapped "…=" is a separate, older difference: not asserted.)"""
    wrapped = "Calculate " + text
    assert portable(wrapped) == _reference(wrapped)


def test_amrt1_generated_lines_keep_unspaced_operators(portable):
    # Mode 1 (a generated line ending "=") is not a bare user turn: GSM8K writes "48/2=".
    assert portable("48/2=", 1) == "24"
    assert portable("90-40 =", 1) == "50"


# --- AMRT-4 -------------------------------------------------------------------------------------

def test_amrt4_generated_formula_must_start_a_line(portable):
    # The prompt ends inside a line: the generated suffix is not a complete formula.
    assert portable.prepare("so far the total is 100", " - 20 - 30 =") is None
    assert portable.prepare("Start with 5", " + 3 - 1 =") is None
    # A prompt that ends with a newline (chat templates), no prompt, or a newline in the
    # generated text all give a line start.
    assert portable.prepare("Items:\n", "- 20 - 30 =") == "- 20 - 30 ="
    assert portable.prepare("", "3 + 4 =") == "3 + 4 ="
    assert portable.prepare("so far the total is 100", " left.\n90 - 40 =") == "90 - 40 ="
    # The Markdown-bullet rule on a real line start is unchanged.
    assert portable(portable.prepare("Items:\n", "- 90 - 40 ="), 1) == "50"


# --- AMRT-5 -------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("2^3!", None), ("2^{3}!", None), ("2^-3!", None),
    ("2^{3!}", "64"), ("(2^3)!", "40320"), ("3!^2", "36"), ("2*3!", "12"), ("-3!", "-6"),
])
def test_amrt5_factorial_on_an_unbraced_exponent_is_refused(portable, text, expected):
    assert portable(text) == expected
    assert portable(text + " =", 1) == expected
    assert portable("Calculate " + text) == _reference("Calculate " + text) == expected


# --- AMRT-6 -------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("How many ss in Mississippi?", None), ("How many Ss in Mississippi?", None),
    ("How many s's in Mississippi?", "4"), ("How many 's's in Mississippi?", "4"),
    ("How many s in Mississippi?", "4"), ("How many rs in strawberry?", "3"),
    ("How many ll are in hello?", None),
])
def test_amrt6_doubled_s_is_not_read_as_a_plural(portable, text, expected):
    assert portable(text) == expected
    assert _reference(text) == expected
    assert (C.parse_request(text) is None) == (expected is None)


# --- AMRT-2 -------------------------------------------------------------------------------------

class _Tok:
    eos_token_id = 0

    def __init__(self):
        self.v = (["<eos>"] + list("0123456789") + [" ", "=", "+", "-", "*", "/", ".", ",", "\n", "So", " 7",
                                                    ".5", ",000", "5)", "0%", "5x", " 3", " 4", "00", "%",
                                                    ". ", ", and", " apples"])
        self.ix = {s: i for i, s in enumerate(self.v)}

    def __len__(self):
        return len(self.v)

    def decode(self, ids, **_):
        return "".join(self.v[i] for i in ids)

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            for size in (7, 5, 4, 3, 2, 1):
                piece = text[i:i + size]
                if piece in self.ix and piece != "<eos>":
                    out.append(self.ix[piece])
                    i += size
                    break
            else:
                raise ValueError(text[i:])
        return out


_EXTENDS = (".5", ",000", "5)", "0%", "5x", "%", "5", "00")
_ALLOWED = (".", ",", "\n", " ", ". ", ", and", " apples", "<eos>")


@pytest.mark.parametrize("parser", [D.PARSER_ID, D.PARSER_V2, D.PARSER_V4])
@pytest.mark.parametrize("mode", ["force", "constrain"])
def test_amrt2_no_token_may_extend_the_written_result(parser, mode):
    import mlx.core as mx

    tok = _Tok()
    spec = {"id": "calc", "kind": "calculator", "write_mode": mode, "read_layer": 0, "write_layer": 0,
            "parser": parser, "arithmetic_backend": "cpu", "gate": {"bias": 0.0, "threshold": 0.5}, "file": "x"}
    manifest = {"schema": D.SCHEMA, "source": {"fingerprint": "f"}, "layers": 1, "tiles": [spec],
                "files": {"x": "0"}, "_digest": "d", "_dir": "."}
    tiles = D.DetTileSet(manifest, {"calc.probe_w": mx.zeros((4,))}, None)
    proc = D.CalculatorProcessor(tiles, "calc", tok, gate_override=lambda _text: True)
    ids = tok.encode("So 3 + 4 =")
    # A "model" that would rather extend the number than stop.
    prefer = np.zeros(len(tok), np.float32)
    for rank, piece in enumerate(_EXTENDS):
        prefer[tok.ix[piece]] = 10 - rank
    row = None
    for _ in range(6):
        row = np.array(proc(mx.array(ids), mx.array(prefer)[None])[0])
        if proc.counts.get("closed"):
            break
        ids.append(int(np.argmax(row)))
    assert proc.counts.get("closed") == 1 and tok.decode(ids).endswith("7")
    banned = {piece for piece in tok.v if row[tok.ix[piece]] <= proc.MASK / 2}
    assert set(_EXTENDS) <= banned, sorted(set(_EXTENDS) - banned)
    assert not banned & set(_ALLOWED)
    # A lone "." may end the sentence, but nothing digit-led may follow it ("7." + "5)").
    ids.append(tok.ix["."])
    row = np.array(proc(mx.array(ids), mx.array(prefer)[None])[0])
    after_point = {piece for piece in tok.v if row[tok.ix[piece]] <= proc.MASK / 2}
    assert {"5", "00", "5)", "0%", "5x"} <= after_point and "\n" not in after_point


# --- AMRT-3 -------------------------------------------------------------------------------------

def _fact_and_calculator_tiles():
    import mlx.core as mx

    fact = {"id": "facts", "kind": "fact_table", "write_mode": "delta", "read_layer": 0, "write_layer": 0,
            "table": {"recogniser": "residual_dims", "tau": 0.5, "margin": 0.0}, "file": "x"}
    calc = {"id": "calc", "kind": "calculator", "write_mode": "force", "read_layer": 0, "write_layer": 0,
            "parser": D.PARSER_V2, "arithmetic_backend": "cpu", "gate": {"bias": 0.0, "threshold": 0.5},
            "file": "x"}
    manifest = {"schema": D.SCHEMA, "source": {"fingerprint": "f"}, "layers": 1, "tiles": [fact, calc],
                "files": {"x": "0"}, "_digest": "d", "_dir": "."}
    tensors = {"facts.dims": mx.array([0, 1]), "facts.keys": mx.array([[1.0, 0.0]]),
               "facts.values": mx.ones((1, 4)) * 5, "facts.null": mx.array([False]),
               "calc.probe_w": mx.zeros((4,))}
    return D.DetTileSet(manifest, tensors, None)


@pytest.mark.parametrize("name", ["facts", "fact_table"])
def test_amrt3_fact_table_cannot_be_receipted_off_while_it_writes(name):
    import mlx.core as mx

    tiles = _fact_and_calculator_tiles()
    request = {"det_tiles": {name: False}, "messages": [{"role": "user", "content": "hello"}]}
    with pytest.raises(D.DetTileRefused, match="cannot be turned off"):
        D.request_toggles(tiles, request)
    with pytest.raises(D.DetTileRefused):
        D.processors_for(tiles, _Tok(), request=request)
    assert "_mlx2_det_tiles_on" not in request
    # The table does write in the shared forward, which is why "off" is refused.
    x = mx.array([[[1.0, 0.0, 0.0, 0.0]]])
    assert tiles.on_mlp(0, x, mx.zeros((1, 1, 4))).tolist() == [[[5.0, 5.0, 5.0, 5.0]]]
    # A calculator keeps its per-request switch, and an explicit "on" is still fine.
    assert D.request_toggles(tiles, {"det_tiles": {"calculator": False}}) == {"facts": True, "calc": False}
    assert D.request_toggles(tiles, {"det_tiles": {"facts": True, "calc": False}}) == {"facts": True, "calc": False}


def test_amrt3_serving_checks_the_route_when_every_toggle_is_off():
    # Source check only: _prepare_job needs a whole engine. The old condition skipped
    # check_route for an all-off request although the tile set still runs in the forward.
    source = (Path(D.__file__).parents[1] / "serving.py").read_text()
    block = source[source.index("toggles = request_toggles(det_tiles, job.request)"):]
    block = block[:block.index("det_tiles.check_route(served)")]
    assert "if det_tiles is not None:" in block and "any(toggles.values())" not in block


# --- AMRT-7 -------------------------------------------------------------------------------------

def _saved_tables(tmp_path):
    rng = np.random.default_rng(1121)
    graph = L.graph_from_partition_names(["core", "box:phys", "box:chem"])
    owner = rng.integers(0, len(graph.tiles), size=(3, 128))
    layers = {2: L.plan_layer(owner, tiles=len(graph.tiles), group_size=64)}
    L.save_tables(tmp_path, layers, graph, source={"checkpoint": "sweep1121@0"})
    return layers, graph


def test_amrt7_tables_without_a_digest_are_refused(tmp_path):
    _saved_tables(tmp_path)
    assert set(L.load_tables(tmp_path)[0]) == {2}
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    del manifest["layers"]["2"]["digest"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest"):
        L.load_tables(tmp_path)


def test_amrt7_tables_are_validated_even_when_the_digest_matches(tmp_path):
    layers, graph = _saved_tables(tmp_path)
    lb = layers[2]
    # Shorten the first chunk and restamp the digest: the stored bytes match their digest,
    # but the chunk table no longer tiles the packed rows the kernels index.
    lb.chunks[0, 1] -= 1
    arrays = {f"L2.{k}": v for k, v in lb.arrays().items()}
    arrays["tile_box_bits"] = graph.tile_box_bits()
    np.savez(tmp_path / "blocks.npz", **arrays)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["layers"]["2"] = lb.meta()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="chunks"):
        L.load_tables(tmp_path)
    # A table set packed for another tile count than its graph is refused as well.
    manifest["layers"]["2"]["tiles"] = len(graph.tiles) + 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        L.load_tables(tmp_path)
