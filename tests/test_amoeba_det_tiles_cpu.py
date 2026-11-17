"""CPU tests: deterministic amoeba tiles (calculator parser/evaluator, fact table, wrapper, processor)."""
from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx2.runtime import amoeba_det_tiles as D

mx.set_default_device(mx.cpu)


@pytest.mark.parametrize("text,answer", [
    ("48 / 2 =", " 24"), ("So 12 * 34 = ", "408"), ("7 times 8 equals", " 56"), ("Calculate: 5 + 3 =", " 8"), ("0.5 * 0.3 =", " 0.15"),
    ("$5 + $3 =", " $8"), ("1,200 + 300 =", " 1,500"), ("Answer: 15 - 27 =", " -12"),
    ("100 divided by 8 equals", " 12.5"), ("3 x 4 =", " 12"), ("123 * 45 =", " 5535"),
    ("<<72-24=", "48"), ("What is 7 times 12? Reply with the number only.\nassistant\n", "84"),
    ("Q: What is 15 * 4?\nA:", " 60"), ("so the total is <<48/2=", "24"),
    ("$10 / $4 =", " 2.5"), ("72-24=", "48"),
])
def test_calculator_answers_exactly(text, answer):
    got, info = D.calculator_answer(text)
    assert got == answer, info


@pytest.mark.parametrize("text,why", [
    ("10 / 3 =", "non_terminating_quotient"), ("9 / 0 =", "division_by_zero"),
    ("On 12/25 is", "no_expression"), ("Call 555-1234 is", "no_expression"), ("On 12/25/2024 =", "operand_boundary"),
    ("What is 7 times 12? It is 84.", "no_expression"), ("7 times 8 is", "no_expression"),
    ("v3.1 + 2 =", "operand_boundary"), ("at 10:30 + 5 =", "operand_boundary"),
    ("item A12 + 3 =", "operand_boundary"), ("3x4 =", "x_without_spaces"),
    ("2024-05-01 + 1 =", "operand_boundary"), ("the year 1999 is", "no_expression"),
    ("$5 + 3 =", "mixed_units"), ("1.2.3 + 4 =", "operand_boundary"),
    # chained expressions: evaluating only the last step would write a wrong result
    ("2 * 3 + 4 =", "chained_expression"), ("10 - 2 + 3 =", "chained_expression"), ("x + 5 + 3 =", "chained_expression"),
    ("2^3 + 1 =", "chained_expression"), ("1,2345 + 1 =", "operand_boundary"), ("12,34 + 1 =", "operand_boundary"),
    ("<<2*3+4=", "chained_expression"), ("What is 2 * 3 + 4?", "chained_expression"), ("(3 + 4) * 2 =", "no_expression"), ("2 + (3 * 4) =", "no_expression"),
    # ranges, scores, phones, years, dates
    ("between 5 - 10 is", "no_expression"), ("In 2020 - 2021 =", "year_range"), ("score 3 - 2 is", "no_expression"),
    ("Phone: 555 - 1234 =", "phone_like"), ("Call 555-1234 =", "phone_like"), ("On 12/25 =", "date_like"),
    ("pages 10-20 =", "range_like"), ("Call 555 - 1234. Thanks", "no_expression"), ("Ages 20 - 30: what", "no_expression"),
    ("3 + 4 =\n", "no_expression"), ("\u0663 + 4 =", "no_expression"), ("$5 * $3 =", "mixed_units"), ("She sold 48/2=", "lead_not_allowed"), ("$5 * 3 =", "one_sided_dollar"),
])
def test_calculator_refuses(text, why):
    got, info = D.calculator_answer(text)
    assert got is None and info["refused"] == why


def test_decimal_string_is_exact():
    from fractions import Fraction
    assert D._decimal_string(Fraction(1, 8)) == "0.125"
    assert D._decimal_string(Fraction(-5, 2)) == "-2.5"
    assert D._decimal_string(Fraction(1, 3)) is None
    assert D._decimal_string(Fraction(10**20 + 1, 10**20)) == "1.00000000000000000001"


class _MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = mx.eye(d) * 0.5

    def __call__(self, x):
        return x @ self.w


class _Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = _MLP(d)

    def __call__(self, h):
        return h + self.mlp(h)


def _toy_set(tmp_path, *, calc_mode="force", tau=0.9, fingerprint="fp0"):
    d, k = 8, 4
    probe_w = mx.zeros((d,)).at[0].add(10.0)               # gate: x[0] > 0 (logit 10*x0 - 5 > logit(0.5)=0)
    gate_rows = mx.eye(d)[:k]
    up_rows = mx.eye(d)[:k]
    keys = mx.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])
    values = mx.zeros((3, d)).at[0, 7].add(3.0).at[1, 6].add(2.0)
    null = mx.array([False, False, True])
    tensors = {"calc.probe_w": probe_w, "facts.gate_rows": gate_rows, "facts.up_rows": up_rows,
               "facts.keys": keys, "facts.values": values, "facts.null": null}
    tiles = [{"id": "calc", "kind": "calculator", "write_mode": calc_mode, "read_layer": 1, "write_layer": 1,
              "gate": {"bias": -5.0, "threshold": 0.5}},
             {"id": "facts", "kind": "fact_table", "write_mode": "delta", "read_layer": 0, "write_layer": 2,
              "table": {"tau": tau, "margin": 0.1}}]
    path = D.write_manifest(tmp_path / "tiles", source={"fingerprint": fingerprint, "artifact": "toy"}, layers=3,
                            tiles=tiles, tensors=tensors,
                            rows_source=[{"row": i, "method": "test"} for i in range(3)])
    return path


def _relu(x):
    return mx.maximum(x, 0)


def test_install_wraps_and_fact_table_writes(tmp_path):
    path = _toy_set(tmp_path)
    layers = [_Layer(8) for _ in range(3)]
    ts = D.install(layers, path, {"mlp": "mlp", "activation_fn": _relu}, source_fingerprint="fp0")
    assert ts.layers_used() == [0, 1, 2]

    def run(x):
        h = x
        for l in layers:
            h = l(h)
        return h
    # Row 0 keys on neuron 0 (fact row 0 -> +3 at dim 7); row 1 matches nothing above tau; row 2 hits the null row.
    x = mx.zeros((3, 1, 8))
    x = x.at[0, 0, 0].add(1.0).at[1, 0, 0].add(1.0).at[1, 0, 1].add(1.0).at[2, 0, 2].add(1.0)
    ts.enabled = False
    base = run(x)
    ts.enabled = True
    out = run(x)
    diff = out - base
    # The delta lands after layer 2's MLP (no later layer), so it appears unscaled.
    assert float(diff[0, 0, 7]) == pytest.approx(3.0)
    assert float(mx.abs(diff[0]).sum()) == pytest.approx(3.0)
    assert float(mx.abs(diff[1]).sum()) == 0.0        # below tau: exact zero
    assert float(mx.abs(diff[2]).sum()) == 0.0        # null row: falls back, exact zero
    r = ts.receipt()
    assert r["qualified"] is False and r["counters"]["fact_open_rows"] == 1
    assert {t["id"]: t["exact"] for t in r["tiles"]} == {"calc": True, "facts": False}


def test_fail_closed(tmp_path):
    path = _toy_set(tmp_path)
    decl = {"mlp": "mlp", "activation_fn": _relu}
    with pytest.raises(ValueError, match="fingerprint"):
        D.install([_Layer(8) for _ in range(3)], path, decl, source_fingerprint="other")
    with pytest.raises(ValueError, match="layers"):
        D.install([_Layer(8) for _ in range(4)], path, decl, source_fingerprint="fp0")
    with pytest.raises(D.DetTileRefused):
        D.install([_Layer(8) for _ in range(3)], path, decl, source_fingerprint="fp0", route="native_mtp")
    (path.parent / "tiles.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="digest"):
        D.load_manifest(path)


def test_push_mode_is_research_only(tmp_path):
    path = _toy_set(tmp_path, calc_mode="push")
    with pytest.raises(D.DetTileRefused, match="research-only"):
        D.install([_Layer(8) for _ in range(3)], path, {"mlp": "mlp", "activation_fn": _relu}, source_fingerprint="fp0")
    ts = D.build(path, _relu, source_fingerprint="fp0", layers_count=3, allow_research_modes=True)
    assert ts.tiles[0].mode == "push"


def test_nonempty_output_refused(tmp_path):
    _toy_set(tmp_path)
    with pytest.raises(ValueError, match="nonempty"):
        _toy_set(tmp_path)


class _Tok:
    """Character tokenizer: id = ord(char)."""

    def __len__(self):
        return 128

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


@pytest.mark.parametrize('question,expected',[("How many r's are in strawberry?",'3'),('Calculate (3+5)*7.','56')])
def test_complete_literal_request_has_bound_symbolic_gate_and_terminal_answer(tmp_path,question,expected):
    path=_toy_set(tmp_path)
    ts=D.build(path,_relu,source_fingerprint='fp0',layers_count=3)
    tile=next(t for t in ts.tiles if t.id=='calc')
    tile.spec['complete_literal_requests']=True
    tile.arithmetic_backend='cpu'
    ts.stash['calc']=mx.array([[False]])
    ts.stash_shape['calc'],ts.forward_ids['calc']=(1,1),1
    tok=_Tok();tok.eos_token_id=3
    proc=D.CalculatorProcessor(ts,'calc',tok,prompt_length=1,request_text=question)
    proc.bind_row(0)
    assert _drive(proc,'?',len(expected)+1)==expected+chr(3)
    assert proc.events[0]['syntax']=='complete_literal_request'


def test_literal_request_never_uses_unbound_batch_row_or_control_override(tmp_path):
    path=_toy_set(tmp_path)
    ts=D.build(path,_relu,source_fingerprint='fp0',layers_count=3)
    tile=next(t for t in ts.tiles if t.id=='calc');tile.spec['complete_literal_requests']=True
    tile.arithmetic_backend='cpu'
    ts.stash['calc']=mx.array([[False],[False]])
    ts.stash_shape['calc'],ts.forward_ids['calc']=(2,1),1
    tok=_Tok();tok.eos_token_id=3
    p=D.CalculatorProcessor(ts,'calc',tok,request_text='Calculate 2+3.')
    assert _drive(p,'?',1)=='?' and not p.events
    p=D.CalculatorProcessor(ts,'calc',tok,request_text='Calculate 2+3.',gate_override=lambda _:False)
    p.bind_row(0)
    assert _drive(p,'?',1)=='?' and not p.events


def test_serving_factory_passes_actual_user_request_and_records_execution(tmp_path):
    ts=D.build(_toy_set(tmp_path),_relu,source_fingerprint='fp0',layers_count=3)
    tile=next(t for t in ts.tiles if t.id=='calc')
    tile.spec['complete_literal_requests']=True;tile.arithmetic_backend='cpu'
    ts.stash['calc']=mx.array([[False]])
    ts.stash_shape['calc'],ts.forward_ids['calc']=(1,1),1
    # HF fast tokenizers have an internal Rust Tokenizer with no len() or EOS.
    # Serving adds another wrapper around that HF object.
    tok=_Tok();tok.eos_token_id=3;tok._tokenizer=object()
    class ServingTokenizer:
        _tokenizer = tok
    (proc,)=D.processors_for(ts,ServingTokenizer(),prompt_length=1,
                           request={'messages':[{'role':'user','content':'Calculate 9*7.'}]})
    assert proc.tok is tok
    proc.bind_row(0)
    assert _drive(proc,'?',3)=='63'+chr(3)
    assert ts.receipt()['counters']['calc']['plan_executions']=={'cpu_reference':1}


@pytest.mark.parametrize('text,answer',[("Count 'r' in 'strawberry' =",'3'),('Sum [0.1, 0.2, -0.5] =','-0.2')])
def test_count_and_list_sum_at_reasoning_cue(tmp_path,text,answer):
    path=_toy_set(tmp_path)
    ts=D.build(path,_relu,source_fingerprint='fp0',layers_count=3)
    tile=next(t for t in ts.tiles if t.id=='calc')
    tile.parser=D.PARSER_V4;tile.arithmetic_backend='cpu';tile.spec['complete_literal_requests']=True
    ts.stash['calc']=mx.array([[False]])
    ts.stash_shape['calc'],ts.forward_ids['calc']=(1,1),1
    p=D.CalculatorProcessor(ts,'calc',_Tok());p.bind_row(0)
    assert _drive(p,text,len(answer))==answer
    assert p.events[0]['syntax']=='literal_count'


def _drive(proc, prompt, steps):
    ids = [ord(c) for c in prompt]
    for _ in range(steps):
        logits = mx.zeros((1, 128))
        out = proc(mx.array(ids), logits)
        ids.append(int(mx.argmax(out[0]).item()) if float(mx.max(out)) == 0.0 and float(mx.min(out)) < -1e8
                   else ord("?"))
    return "".join(chr(i) for i in ids[len(prompt):])


@pytest.mark.parametrize("mode", ["force", "constrain"])
def test_processor_writes_result_when_gate_and_parse_agree(tmp_path, mode):
    path = _toy_set(tmp_path, calc_mode=mode)
    layers = [_Layer(8) for _ in range(3)]
    ts = D.install(layers, path, {"mlp": "mlp", "activation_fn": _relu}, source_fingerprint="fp0")
    proc = D.CalculatorProcessor(ts, "calc", _Tok())
    ts.stash["calc"] = mx.array([[False], [True]])          # row 1's gate open
    ts.stash_shape["calc"], ts.forward_ids["calc"] = (2, 1), 7
    proc.bind_row(1)
    got = _drive(proc, "So 12 * 34 =", 5)
    assert got[:4] == " 408" and not got[4].isdigit()      # the result, then the number is closed
    assert ts.counters["calc"]["plans"] == 1 and ts.counters["calc"]["closed"] == 1
    proc2 = D.CalculatorProcessor(ts, "calc", _Tok())
    proc2.bind_row(0)                                       # row 0's gate closed: host keeps its tokens
    assert _drive(proc2, "So 12 * 34 =", 2) == "??"
    proc3 = D.CalculatorProcessor(ts, "calc", _Tok())
    proc3.bind_row(1)                                       # gate open but no expression: nothing written
    assert _drive(proc3, "Call v5.5-1234 =", 2) == "??"
    assert ts.counters["calc"]["parse_refused"]["operand_boundary"] >= 1


def test_identity_joins_manifest(tmp_path):
    path = _toy_set(tmp_path)
    m = D.load_manifest(path)
    ident = D.tile_set_identity(m)
    assert ident["manifest_sha256"] == m["_digest"]
    json.dumps(ident)


def test_unbound_row_fails_closed(tmp_path):
    path = _toy_set(tmp_path)
    ts = D.install([_Layer(8) for _ in range(3)], path, {"mlp": "mlp", "activation_fn": _relu}, source_fingerprint="fp0")
    ts.stash["calc"] = mx.array([[True], [True]])
    ts.stash_shape["calc"], ts.forward_ids["calc"] = (2, 1), 3
    proc = D.CalculatorProcessor(ts, "calc", _Tok())       # never bound: width 2 is ambiguous
    assert _drive(proc, "So 12 * 34 =", 1) == "?"
    assert ts.counters["calc"]["row_unbound"] == 1
    ts.stash["calc"] = mx.array([[False, True]])            # one row, prompt-shaped: last position decides
    ts.stash_shape["calc"] = (1, 2)
    assert _drive(D.CalculatorProcessor(ts, "calc", _Tok()), "So 12 * 34 =", 1) == " "


def test_residual_dims_recogniser_with_centre(tmp_path):
    d = 8
    keys = mx.array([[1.0, 0.0], [0.0, 1.0]])
    tensors = {"t.dims": mx.array([2, 5], dtype=mx.int32), "t.center": mx.array([1.0, 1.0]),
               "t.keys": keys, "t.values": mx.zeros((2, d)).at[1, 0].add(4.0), "t.null": mx.array([True, False])}
    tiles = [{"id": "t", "kind": "fact_table", "write_mode": "delta", "read_layer": 0, "write_layer": 0,
              "table": {"tau": 0.9, "margin": 0.1, "recogniser": "residual_dims"}}]
    path = D.write_manifest(tmp_path / "r", source={"fingerprint": "fp"}, layers=1, tiles=tiles, tensors=tensors)
    ts = D.build(path, _relu, source_fingerprint="fp", layers_count=1)
    x = mx.zeros((3, 1, d))
    x = x.at[0, 0, 5].add(3.0).at[0, 0, 2].add(1.0)      # centred (0, 2): row 1 -> open, +4 at dim 0
    x = x.at[1, 0, 2].add(3.0).at[1, 0, 5].add(1.0)      # centred (2, 0): null row 0 -> closed
    x = x.at[2, 0, 2].add(1.0).at[2, 0, 5].add(1.0)      # centred (0, 0): no direction -> closed
    y = ts.on_mlp(0, x, mx.zeros((3, 1, d)))
    assert float(y[0, 0, 0]) == pytest.approx(4.0)
    assert float(mx.abs(y[1:]).sum()) == 0.0
    tiles[0]["table"]["recogniser"] = "bogus"
    with pytest.raises(ValueError, match="recogniser"):
        bad = D.write_manifest(tmp_path / "b", source={"fingerprint": "fp"}, layers=1, tiles=tiles, tensors=tensors)
        D.build(bad, _relu, source_fingerprint="fp", layers_count=1)


def test_adapter_policy_refusals():
    with pytest.raises(D.DetTileRefused, match="ordinary"):
        D.refuse_policy("m.json", {}, require_mtp=True)
    with pytest.raises(D.DetTileRefused, match="draft_model"):
        D.refuse_policy("m.json", {"draft_model": "x"})
    with pytest.raises(D.DetTileRefused, match="streaming"):
        D.refuse_policy("m.json", {}, weight_streaming="early")
    with pytest.raises(ValueError):
        D.refuse_policy("", {})
    D.refuse_policy("m.json", {"amoeba_tiles": "x"})


def test_host_tiles_mismatch_refused(tmp_path):
    path = _toy_set(tmp_path)
    with pytest.raises(ValueError, match="host tiles"):
        D.install_on_adapter([_Layer(8) for _ in range(3)], path, {"mlp": "mlp", "activation_fn": _relu},
                             source_fingerprint="fp0", host_tiles_sha256="abc")


def test_masks_compose_with_a_later_hard_mask(tmp_path):
    """A later processor keeping only one other token (thinking guard) must leave a finite row."""
    path = _toy_set(tmp_path)
    ts = D.install([_Layer(8) for _ in range(3)], path, {"mlp": "mlp", "activation_fn": _relu}, source_fingerprint="fp0")
    ts.stash["calc"] = mx.array([[True]])
    ts.stash_shape["calc"], ts.forward_ids["calc"] = (1, 1), 1
    proc = D.CalculatorProcessor(ts, "calc", _Tok())
    proc.bind_row(0)
    out = proc(mx.array([ord(c) for c in "So 12 * 34 ="]), mx.zeros((1, 128)))
    keep = mx.zeros((128,), dtype=mx.bool_).at[ord("<")].add(True)
    guarded = mx.where(keep[None], out, -mx.inf)
    lp = guarded - mx.logsumexp(guarded, axis=-1, keepdims=True)
    assert bool(mx.isfinite(lp[0, ord("<")]).item())


def test_plan_aborts_when_an_earlier_processor_banned_the_token(tmp_path):
    path = _toy_set(tmp_path)
    ts = D.install([_Layer(8) for _ in range(3)], path, {"mlp": "mlp", "activation_fn": _relu}, source_fingerprint="fp0")
    ts.stash["calc"] = mx.array([[True]])
    ts.stash_shape["calc"], ts.forward_ids["calc"] = (1, 1), 1
    proc = D.CalculatorProcessor(ts, "calc", _Tok())
    proc.bind_row(0)
    upstream = mx.zeros((1, 128)).at[0, ord(" ")].add(-1e9)   # the planned " " is already banned
    out = proc(mx.array([ord(c) for c in "So 12 * 34 ="]), upstream)
    assert ts.counters["calc"]["aborted_upstream_mask"] == 1 and proc.plan is None
    assert float(mx.abs(out - upstream).max()) == 0.0


def test_number_stays_closed_after_a_point(tmp_path):
    path = _toy_set(tmp_path)
    ts = D.install([_Layer(8) for _ in range(3)], path, {"mlp": "mlp", "activation_fn": _relu}, source_fingerprint="fp0")
    ts.stash["calc"] = mx.array([[True]])
    ts.stash_shape["calc"], ts.forward_ids["calc"] = (1, 1), 1
    proc = D.CalculatorProcessor(ts, "calc", _Tok())
    proc.bind_row(0)
    ids = [ord(c) for c in "So 12 * 34 ="]
    for ch in " 408":
        proc(mx.array(ids), mx.zeros((1, 128)))
        ids.append(ord(ch))
    out = proc(mx.array(ids), mx.zeros((1, 128)))           # closing step: digits banned
    assert float(out[0, ord("5")]) < -1e8
    ids.append(ord("."))
    out = proc(mx.array(ids), mx.zeros((1, 128)))           # after "408." still no digit
    assert float(out[0, ord("5")]) < -1e8 and float(out[0, ord(" ")]) == 0.0
    ids.append(ord(" "))
    out = proc(mx.array(ids), mx.zeros((1, 128)))           # released
    assert float(out[0, ord("5")]) == 0.0


def test_fact_write_handles_prefill_tail_rows(tmp_path):
    path = _toy_set(tmp_path)
    ts = D.install([_Layer(8) for _ in range(3)], path, {"mlp": "mlp", "activation_fn": _relu}, source_fingerprint="fp0")
    x = mx.zeros((1, 5, 8)).at[0, -1, 0].add(1.0)
    ts.on_mlp(0, x, mx.zeros((1, 5, 8)))
    y = ts.on_mlp(2, x[:, -1:], mx.zeros((1, 1, 8)))      # tail-skip: write layer sees one row
    assert y.shape == (1, 1, 8) and float(y[0, 0, 7]) == pytest.approx(3.0)


def test_neural_tile_audit_sees_through_det_wrapper():
    from mlx2.runtime import amoeba_tiles as T

    class Inner:
        amoeba_tiled = True

        def projections(self):
            return [("g", "m")]

    class Wrap:
        amoeba_det_wrapped = True

        def __init__(self):
            self.inner = Inner()

    class L:
        mlp = Wrap()
    assert T.tile_projection_modules([L()]) == [("0.mlp.inner.g", "m")]


@pytest.mark.parametrize("text", [
    "2 \\times 3 + 4 =", "2 \\cdot 3 + 4 =", "6 \\div 2 + 1 =", "2 \u00b7 3 + 4 =", "10 \u2013 3 + 2 =",
    "negative 5 + 3 =", "twice 3 + 4 =", "the square root of 16 + 9 =", "20% of 50 + 1 =",
    "10 percent of 50 + 5 =", "2 to the power 3 + 1 =", "log 100 + 1 =", "sqrt 9 + 1 =",
    "What is 2 \\times 3 + 4?", "What is the square root of 16 + 9?",
    "So $5 * 3 =", "Each box costs $12 / 4 =", "Step 3\n- 4 =", "Total 10\n\n+ 2 =",
])
def test_lead_allow_list_refuses_partial_expressions(text):
    got, info = D.calculator_answer(text)
    assert got is None, (text, got)


def test_truncated_window_refuses_operand_at_start():
    assert D.calculator_answer("45 + 6 =")[0] == " 51"
    assert D.calculator_answer("45 + 6 =", truncated=True)[0] is None
    assert D.calculator_answer("Then 45 + 6 =", truncated=True)[0] == " 51"


@pytest.mark.parametrize("text,want", [
    ("so the total is <<12*7=", "84"), ("June computed 156-47=", "109"), ("We need 12 / 4 =", " 3"),
    ("Then $5 + $3 =", " $8"), ("Q: What is 12 times 7?\nA:", " 84"),
])
def test_lead_allow_list_keeps_plain_steps(text, want):
    assert D.calculator_answer(text)[0] == want


# ---- calc-parse-v2: whole expressions --------------------------------------------------------
V2 = D.PARSER_V2


@pytest.mark.parametrize("text,want", [
    ("2 + 3 * 4 =", " 14"), ("(12 - 4) / 2 =", " 4"), ("Calculate: 2 ^ 10 =", " 1024"),
    ("so the total is <<3*4+2=", "14"), ("We need -5 + 3 =", " -2"), ("What is (3 + 4) * 2?", "14"),
    ("What is 2 + 3 * 4? Reply with the number only.", " 14"), ("1,200 + 300 * 2 =", " 1,800"),
    ("10 / 4 * 3 =", " 7.5"), ("2 ** 3 ** 2 =", " 512"), ("1 / 3 * 3 =", " 1"), ("3 - -2 * 4 =", " 11"),
    ("3 + 4 * 2 is equal to", " 11"), ("3 + 4 =", " 7"), ("What is 7 times 12? Reply with the number only.\nassistant\n", "84"),
])
def test_v2_whole_expressions(text, want):
    assert D.calculator_answer(text, parser=V2)[0] == want


@pytest.mark.parametrize("text,why", [
    ("2 / 3 + 1 =", "non_terminating_quotient"), ("2024-05-12 =", "date_or_time"),
    ("555-123-4567 =", "unspaced_dash_or_slash"), ("12/25/2024 =", "date_or_time"),
    ("June computed 5-3+2=", "unspaced_dash_or_slash"), ("9 ^ 9 ^ 9 =", "exponent_not_small_integer"),
    ("Step 3\n- 4 * 2 =", "chained_expression"), ("Notes:\n- 4 * 2 =", "spaced_unary_minus"), ("2 \\times 3 + 4 =", "lead_not_allowed"),
    ("10 – 3 + 2 =", "lead_not_allowed"), ("the square root of 16 + 9 =", "lead_not_allowed"),
    ("x = 3 + 4 * 2 =", "chained_expression"), ("$5 + $3 * 2 =", "chained_expression"),
    ("v1.2 + 3 * 4 =", "chained_expression"), ("(3 + 4 =", "chained_expression"),
    ("2(3 + 4) =", "no_expression"), ("4 / (2 - 2) * 3 =", "division_by_zero"),
    ("What is 3+4*2?", "unspaced_question"),
])
def test_v2_refusals(text, why):
    got, info = D.calculator_answer(text, parser=V2)
    assert got is None and info["refused"] == why, (text, got, info)


def test_v1_default_never_reads_chains():
    for text in ("2 + 3 * 4 =", "(12 - 4) / 2 =", "We need -5 + 3 ="):
        assert D.calculator_answer(text)[0] is None


def test_v2_truncated_window_refuses_cut_operand():
    assert D.calculator_answer("45 + 6 * 2 =", truncated=True, parser=V2)[0] is None
    assert D.calculator_answer("Then 45 + 6 * 2 =", truncated=True, parser=V2)[0] == " 57"


def test_v2_large_results_refused():
    assert D.calculator_answer("99 ^ 64 * 99 ^ 64 =", parser=V2)[0] is None


def test_manifest_parser_field(tmp_path):
    import mlx.core as mx
    tiles = [{"id": "calc", "kind": "calculator", "write_mode": "force", "read_layer": 1, "write_layer": 1,
              "gate": {"bias": 0.0, "threshold": 0.5}, "parser": V2}]
    man = D.write_manifest(tmp_path / "t", source={"artifact": "a", "fingerprint": "f"}, layers=2, tiles=tiles,
                           tensors={"calc.probe_w": mx.zeros((4,))})
    m = D.load_manifest(man)
    ident = D.tile_set_identity(m)
    assert ident["parser"] == V2 and ident["tiles"][0]["parser"] == V2
    old = D.tile_set_identity(D.load_manifest(D.write_manifest(
        tmp_path / "v1", source={"artifact": "a", "fingerprint": "f"}, layers=2,
        tiles=[{k: v for k, v in tiles[0].items() if k != "parser"}], tensors={"calc.probe_w": mx.zeros((4,))})))
    assert old["parser"] == D.PARSER_ID and "parser" not in old["tiles"][0]
    bad = [{**tiles[0], "parser": "calc-parse-v9"}]
    man2 = D.write_manifest(tmp_path / "u", source={"artifact": "a", "fingerprint": "f"}, layers=2, tiles=bad,
                            tensors={"calc.probe_w": mx.zeros((4,))})
    with pytest.raises(ValueError, match="parser"):
        D.load_manifest(man2)


@pytest.mark.parametrize("text,want", [
    ("So 10 divided by 2 =", " 5"), ("What is 10 divided by 2?", "5"),
    ("So 2 multiplied by 3 + 1 =", " 7"), ("So 6 multiplied by 7 =", " 42"),
])
def test_v2_word_operators(text, want):
    assert D.calculator_answer(text, parser=V2)[0] == want


@pytest.mark.parametrize("text", [
    "So 2 +\n3 * 4 + 1 =", "So 2 *\n(3 + 4) =", "So 7 -\n(2) =", "Total: 10 -\n-3 * 2 =", "So 2 +\n-3 =",
    "At 10:30 + 15 * 2 =", "Time 9:05 - 3 * 1 =", "So 12:00 + 1 + 2 =", "Ratio is 3:2 * 4 + 1 =", "<<10:30+1*2=",
    "He is 5'10 + 2 * 3 =", "x=3;2 + 3 * 4 =", "So 2plus3 * 4 =", "So 2times3 + 1 =", "So -3 =", "Step 1: -3 =",
    "So (5) =", "What is -5?", "So 1939 - 1945 + 1 =",
    "So " + "1" + "0" * 70 + " ^ 64 =", "So (" + "9" * 70 + ")^64 =",
])
def test_v2_review_refusals(text):
    got, info = D.calculator_answer(text, parser=V2)
    assert got is None, (text, got, info)


@pytest.mark.parametrize('text,answer', [
    (r'So 109 \times 71 =', ' 7739'),
    (r'So 6 \div 2 + 1 =', ' 4'),
    (r'So 2 \cdot 3 + 4 =', ' 10'),
    ('$$\n' + r'\begin{array}{r}' + '\n109' + r' \\' + '\n' + r'\times 71 \\' + '\n' + r'\hline' + '\n', '7739'),
    (r'\begin{array}{r} 12 \\ \times 34 \\ \hline ', '408'),
])
def test_calculator_v3_latex(text, answer):
    assert D.calculator_answer(text, parser=D.PARSER_V3)[0] == answer


@pytest.mark.parametrize('text', [
    r'\begin{array}{r} 109 \\ \times 71 \\ \hline 7661',
    r'\begin{array}{rr} 109 \\ \times 71 \\ \hline',
    r'\begin{array}{r} 109 & 2 \\ \times 71 \\ \hline',
    r'\begin{array}{r} 109 \\ \times \\ \hline',
    r'\begin{array}{r} 109 \\ \times 71 \\ 10 \\ \hline',
    r'2 \timesfoo 3 + 4 =', r'2 \sqrt 3 + 4 =',
])
def test_calculator_v3_refuses_ambiguous_latex(text):
    assert D.calculator_answer(text, parser=D.PARSER_V3)[0] is None


@pytest.mark.parametrize('override', [None, False])
def test_v3_column_gate_and_control(tmp_path, override):
    path = _toy_set(tmp_path)
    m = json.loads(path.read_text())
    m['tiles'][0]['parser'] = D.PARSER_V3; m['tiles'][0]['arithmetic_backend'] = 'cpu'
    path.write_text(json.dumps(m))
    ts = D.build(path, _relu, source_fingerprint='fp0', layers_count=3)
    ts.stash['calc'] = mx.array([[False]])
    ts.stash_shape['calc'], ts.forward_ids['calc'] = (1, 1), 7
    p = D.CalculatorProcessor(ts, 'calc', _Tok(), gate_override=None if override is None else lambda _: False)
    p.bind_row(0)
    prompt = r'\begin{array}{r}109 \\ \times 71 \\ \hline' + '\n'
    assert _drive(p, prompt, 4) == ('7739' if override is None else '????')
    assert ts.receipt()['parser'] == D.PARSER_V3


def test_v3_does_not_bypass_missing_forward_binding(tmp_path):
    path = _toy_set(tmp_path)
    m = json.loads(path.read_text()); m['tiles'][0]['parser'] = D.PARSER_V3; m['tiles'][0]['arithmetic_backend'] = 'cpu'
    path.write_text(json.dumps(m))
    ts = D.build(path, _relu, source_fingerprint='fp0', layers_count=3)
    p = D.CalculatorProcessor(ts, 'calc', _Tok())
    assert _drive(p, r'\begin{array}{r}109 \\ \times 71 \\ \hline'+'\n', 1) == '?'


def test_processor_accepts_tokenizer_wrapper_without_len(tmp_path):
    class Wrapper:
        _tokenizer = _Tok()
    ts = D.build(_toy_set(tmp_path), _relu, source_fingerprint='fp0', layers_count=3)
    p = D.CalculatorProcessor(ts, 'calc', Wrapper(), gate_override=lambda _: True)
    assert _drive(p, 'So 12 * 34 =', 4) == ' 408'


@pytest.mark.parametrize("line", [
    "2 + 3 =", "Count 'r' in 'strawberry' =", "Sum [1, 2, 3] =",
])
def test_verbatim_request_leaves_a_generated_calculator_cue_unchanged(tmp_path, line):
    path = _toy_set(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["tiles"][0].update(
        parser=D.PARSER_V4, arithmetic_backend="cpu", complete_literal_requests=True,
    )
    path.write_text(json.dumps(manifest))
    ts = D.build(path, _relu, source_fingerprint="fp0", layers_count=3)
    processor = D.CalculatorProcessor(
        ts, "calc", _Tok(), gate_override=lambda _: True,
        request_text=f"Repeat exactly this text, with no explanation:\n{line}",
    )
    logits = mx.arange(128, dtype=mx.float32)[None]
    assert processor(mx.array(_Tok().encode(line)), logits) is logits
    assert ts.counters["calc"]["gate_open"] == 0
    assert not processor.events


def test_v3_force_closes_complete_equation(tmp_path):
    path = _toy_set(tmp_path)
    m=json.loads(path.read_text());m['tiles'][0]['parser']=D.PARSER_V3; m['tiles'][0]['arithmetic_backend']='cpu';path.write_text(json.dumps(m))
    ts=D.build(path,_relu,source_fingerprint='fp0',layers_count=3)
    p=D.CalculatorProcessor(ts,'calc',_Tok(),gate_override=lambda _:True)
    assert _drive(p,'So 3^3 + 78 =',5)==' 105\n'


def test_gpu_parser_builds_ir_without_cpu_arithmetic(monkeypatch):
    def forbid(_):
        raise AssertionError('CPU Fraction conversion used in GPU parser')
    monkeypatch.setattr(D,'_number',forbid)
    expr,why=D.parse_expression_v3('So (3^3 + 78) / 5 =',deferred=True)
    assert why is None and expr['value']._calculator_ir
    from mlx2.runtime.amoeba_calc_gpu import program
    _,ops,_,_=program(expr)
    assert ops==[0,0,5,0,1,0,4]


def test_v3_defaults_to_gpu_in_runtime_identity(tmp_path):
    path=_toy_set(tmp_path)
    m=json.loads(path.read_text());m['tiles'][0]['parser']=D.PARSER_V3;path.write_text(json.dumps(m))
    ts=D.build(path,_relu,source_fingerprint='fp0',layers_count=3)
    assert ts.tiles[0].arithmetic_backend=='gpu'
    assert ts.receipt()['tiles'][0]['arithmetic_backend']=='gpu'


@pytest.mark.parametrize('text,answer', [
    (r'\[\frac{12}{4}=\]', ' 3'),
    (r'$\dfrac{12}{4}=$', ' 3'),
    (r'\(\frac{12}{4}=\)', ' 3'),
    (r'So \frac{1}{3}+\frac{2}{3} =', ' 1'),
    (r'So \sqrt{144} =', ' 12'),
    (r'So 5! =', ' 120'),
    (r'So \binom{10}{3} =', ' 120'),
    (r'So 17\bmod 5 =', ' 2'),
    (r'So \sum_{i=1}^{5}i^2 =', ' 55'),
    (r'So \prod_{i=1}^{5}i =', ' 120'),
])
def test_v4_exact_latex(text,answer):
    assert D.calculator_answer(text,parser=D.PARSER_V4)[0]==answer


def test_v4_normalizes_complete_formula_before_character_cap():
    raw=r'\left('+r'\displaystyle '*20+r'\dfrac{12}{\dfrac{9}{3}}\right)'
    assert len(raw)>160
    assert D.calculator_answer('So '+raw+' =',parser=D.PARSER_V4)[0]==' 4'


@pytest.mark.parametrize('text', [r'So \evil{12}+3 =',r'So \left(555-1234\right) =',r'12\dfrac{1}{3} =',r'\frac{1}{0} ='])
def test_v4_refuses_ambiguous_latex(text):
    assert D.calculator_answer(text,parser=D.PARSER_V4)[0] is None


def test_v4_preserves_cut_operand_evidence():
    assert D.calculator_answer(r'12\frac{1}{3} =',truncated=True,parser=D.PARSER_V4)[0] is None


def test_v4_normalizes_spacing_only_before_cap():
    raw = '12' + r'\,' * 100 + '+ 3'
    assert len(raw) > 160
    assert D.calculator_answer('So ' + raw + ' =', parser=D.PARSER_V4)[0] == ' 15'


@pytest.mark.parametrize('suffix', ['= ', '=$', r'=\)', r'=\]'])
def test_v4_trailing_delimiter_cues(suffix):
    opener = {'=$':'$', r'=\)':r'\(', r'=\]':r'\['}.get(suffix, '')
    text = opener + r'\dfrac{12}{4}' + suffix
    answer, info = D.calculator_answer(text, parser=D.PARSER_V4)
    assert answer is not None and answer.strip() == '3'


@pytest.mark.parametrize('text', [r'\[4 =',r'$$1800 =',r'$16 =$ ',r'\(-12 ='])
def test_v4_refuses_number_only_equation_lhs(text):
    assert D.calculator_answer(text,parser=D.PARSER_V4)[0] is None


def test_v4_force_preserves_unit_continuation(tmp_path):
    path = _toy_set(tmp_path)
    m=json.loads(path.read_text());m['tiles'][0].update(parser=D.PARSER_V4,arithmetic_backend='cpu');path.write_text(json.dumps(m))
    ts=D.build(path,_relu,source_fingerprint='fp0',layers_count=3)
    p=D.CalculatorProcessor(ts,'calc',_Tok(),gate_override=lambda _:True)
    prompt=r'\[\frac{360}{6} ='
    assert _drive(p,prompt,3)==' 60'
    out=p(mx.array([ord(c) for c in prompt+' 60']),mx.zeros((1,128)))
    assert float(out[0,ord('m')])==0 and float(out[0,ord('5')]) < -1e8


def test_v4_closes_arithmetic_rhs_without_forcing_newline(tmp_path):
    path=_toy_set(tmp_path)
    m=json.loads(path.read_text());m['tiles'][0].update(parser=D.PARSER_V4,arithmetic_backend='cpu');path.write_text(json.dumps(m))
    ts=D.build(path,_relu,source_fingerprint='fp0',layers_count=3)
    p=D.CalculatorProcessor(ts,'calc',_Tok(),gate_override=lambda _:True)
    prompt='So 3^3 + 78 ='
    assert _drive(p,prompt,4)==' 105'
    for suffix in ['', ' ', '  ']:
        out=p(mx.array([ord(c) for c in prompt+' 105'+suffix]),mx.zeros((1,128)))
        assert float(out[0,ord('+')]) < -1e8
        assert float(out[0,ord('m')])==0 and float(out[0,ord('\n')])==0
    out=p(mx.array([ord(c) for c in prompt+' 105\n']),mx.zeros((1,128)))
    assert float(out[0,ord('+')])==0
