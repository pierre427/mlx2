"""Position-matched calculator tiles (read point layer_output_capture) on a tiny qwen4 (CPU).

Covers the Flash-Next port: the layer-output reader leaves the forward untouched (bit-exact with
the tile off), the gate is matched to the processor's history position (so a speculative verify
row reads its own position), the calculator writes the same tokens on the ordinary and native
self-MTP generators, counters settle through the verify ledger, and the per-request toggle.
"""
import json

import mlx.core as mx
import pytest

from mlx2.runtime import amoeba_det_tiles as D
from mlx2.runtime.sample_utils import LaneRNG

CHARS = "0123456789 +-*/=?.,:abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#"[:64]


class CharTokenizer:
    """One character per token id (64 ids), enough text for the parser."""

    eos_token_id = None

    def decode(self, ids):
        return "".join(CHARS[int(i)] for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [CHARS.index(ch) for ch in text]

    def __len__(self):
        return len(CHARS)


TOK = CharTokenizer()


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _model():
    from tests.test_batched_mtp import _tiny_qwen4_model

    mx.random.seed(515)
    return _tiny_qwen4_model()


def _manifest(tmp_path, *, bias, name="calc", read_point="layer_output_capture", dim=32):
    return D.write_manifest(
        tmp_path / name, source={"artifact": "tiny-qwen4", "fingerprint": "tiny"}, layers=2,
        tiles=[{"id": "calculator", "kind": "calculator", "write_mode": "force", "read_layer": 0,
                "write_layer": 0, "read_point": read_point,
                "gate": {"bias": bias, "threshold": 0.5, "probe": "test"}}],
        tensors={"calculator.probe_w": mx.zeros((dim,))})


def _install(model, manifest):
    import mlx.nn as nn

    trunk = model.language_model.model
    layers = list(trunk.layers)
    tiles = D.install_on_adapter(layers, str(manifest), {"mlp": "mlp", "activation_fn": nn.silu},
                                 source_fingerprint="tiny", route="native_mtp")
    D.install_layer_output_reader(layers, tiles, trunk._external_capture)
    return tiles


def _scripted(prompt_len, script):
    """Force ``script`` (token ids) right after the prompt; a pure function of the history."""
    def force(tokens, logits):
        ids = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
        while ids and isinstance(ids[0], list):
            ids = [i for row in ids for i in row]
        n = len(ids) - prompt_len
        if 0 <= n < len(script):
            import numpy as np
            m = np.full((logits.shape[-1],), -1.0e9, dtype=np.float32)
            m[script[n]] = 0.0
            return logits + mx.array(m)[None]
        return logits
    return force


def _run(model, prompt, processors, *, mtp, max_tokens=12):
    from mlx2.runtime.generate import BatchGenerator

    kw = {}
    if mtp:
        kw["self_mtp"] = {"num_draft": 2, "persistent": True, "segment_aware_live_tip": True,
                          "segment_aware_cohort_size": 1}
    gen = BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1, prefill_step_size=32, **kw)
    try:
        ins = {"lane_rngs": [LaneRNG(7)], "self_mtp_configs": [{"sampling_temp": 0.0}]} if mtp else {}
        (uid,) = gen.insert([prompt], max_tokens=[max_tokens], logits_processors=[list(processors)], **ins)
        out, receipt = [], None
        for _ in range(200):
            _, responses = gen.next()
            for r in responses:
                out.append(r.token)
                if r.finish_reason:
                    return out, getattr(r, "mtp_receipt", None)
        raise AssertionError("generator did not finish")
    finally:
        gen.close()


PROMPT = TOK.encode("Q: so")
SCRIPT = TOK.encode(" 5 + 7 =")


def test_reader_off_is_bit_exact_and_computes_nothing(tmp_path):
    model = _model()
    ref_ord, _ = _run(model, PROMPT, [], mtp=False)
    ref_mtp, _ = _run(model, PROMPT, [], mtp=True)
    tiles = _install(model, _manifest(tmp_path, bias=10.0))
    assert type(model.language_model.model.layers[0]).__name__ == "DecoderLayer"
    assert getattr(model.language_model.model.layers[0], "amoeba_det_reader", False)
    off_ord, _ = _run(model, PROMPT, [], mtp=False)
    off_mtp, _ = _run(model, PROMPT, [], mtp=True)
    assert off_ord == ref_ord and off_mtp == ref_mtp
    # No live processor: the reader stashed nothing.
    assert tiles.out_forwards == 0 and not tiles.out_stash


@pytest.mark.parametrize("mtp", [False, True])
def test_calculator_writes_after_a_generated_expression(tmp_path, mtp):
    model = _model()
    tiles = _install(model, _manifest(tmp_path, bias=10.0))
    calc = D.CalculatorProcessor(tiles, "calculator", TOK, prompt_length=len(PROMPT))
    out, receipt = _run(model, PROMPT, [_scripted(len(PROMPT), SCRIPT), calc], mtp=mtp)
    if mtp:
        assert receipt["route"] == "segmented_self_mtp", receipt
        assert receipt.get("accepted_tokens", receipt.get("accepted", 1)) != 0
    text = TOK.decode(out)
    assert text.startswith(" 5 + 7 = 12"), text
    assert not text[len(" 5 + 7 = 12"):][:1].isdigit()
    c = tiles.request_receipt([calc.uid])
    assert c["plans"] == 1 and c["gate_open"] == 1 and c["tokens_written"] == 3, c
    assert c.get("gate_unmatched", 0) == 0 and c.get("gate_ambiguous", 0) == 0


def test_ordinary_and_mtp_agree_and_closed_gate_writes_nothing(tmp_path):
    model = _model()
    base_ord, _ = _run(model, PROMPT, [_scripted(len(PROMPT), SCRIPT)], mtp=False)
    tiles = _install(model, _manifest(tmp_path, bias=-10.0))
    outs = []
    for mtp in (False, True):
        calc = D.CalculatorProcessor(tiles, "calculator", TOK, prompt_length=len(PROMPT))
        out, _ = _run(model, PROMPT, [_scripted(len(PROMPT), SCRIPT), calc], mtp=mtp)
        c = tiles.request_receipt([calc.uid])
        assert c["plans"] == 0 and c["gate_closed"] >= 1, c
        outs.append(out)
    assert outs[0] == base_ord
    assert outs[1] == outs[0]


def test_gate_is_matched_to_the_history_position():
    tiles = object.__new__(D.DetTileSet)
    tiles.out_stash = {"calculator": __import__("collections").deque(maxlen=4)}
    ids = mx.array([[11, 12, 13, 14], [21, 12, 13, 24]], mx.uint32)
    gate = mx.array([[False, True, False, True], [False, True, True, False]])
    tiles.out_stash["calculator"].appendleft({"ids": ids, "gate": gate, "host": None})
    # verify row 2 of lane 0: history ends with 11 12 13
    assert tiles.calc_gate_at("calculator", [5, 11, 12, 13]) == (False, None)
    assert tiles.calc_gate_at("calculator", [5, 11, 12, 13, 14]) == (True, None)
    # same last token in both rows; the preceding window tells them apart
    assert tiles.calc_gate_at("calculator", [9, 21, 12, 13]) == (True, None)
    # "12" preceded by nothing in the window: both rows match position 1, both open
    assert tiles.calc_gate_at("calculator", [12]) == (True, None)
    # 13 with only its own token as evidence: rows disagree -> fail closed
    assert tiles.calc_gate_at("calculator", [13]) == (None, "gate_ambiguous")
    assert tiles.calc_gate_at("calculator", [12, 13], row=1) == (True, None)
    assert tiles.calc_gate_at("calculator", [7, 8]) == (None, "gate_unmatched")


def test_verify_ledger_restores_plan_and_counters(tmp_path):
    model = _model()
    tiles = _install(model, _manifest(tmp_path, bias=10.0))
    calc = D.CalculatorProcessor(tiles, "calculator", TOK, prompt_length=0)
    ledger = calc.verify_ledger()
    calc.plan, calc._start = [1, 2], 5
    calc.counts["plans"] += 1
    calc.events.append({"x": 1})
    calc.restore_verify_ledger(ledger)
    assert calc.plan is None and calc._start == 0 and calc.events == []
    assert tiles.request_receipt([calc.uid])["plans"] == 0
    probe = calc._clone(shadow=True)
    probe.counts["plans"] += 5
    assert probe.shadow and probe not in tiles.live and calc in tiles.live
    assert tiles.request_receipt([calc.uid])["plans"] == 0
    snap = __import__("copy").deepcopy(calc)
    assert snap.tiles is calc.tiles and snap.counts is calc.counts


def test_request_toggles(tmp_path):
    model = _model()
    tiles = _install(model, _manifest(tmp_path, bias=10.0))
    assert D.request_toggles(tiles, {}) == {"calculator": True}
    assert D.request_toggles(tiles, {"det_tiles": {"calculator": False}}) == {"calculator": False}
    req = {"det_tiles": {"calculator": False}}
    assert D.processors_for(tiles, TOK, prompt_length=3, request=req) == ()
    assert req["_mlx2_det_tile_uids"] == [] and req["_mlx2_det_tiles_on"] == {"calculator": False}
    req = {}
    (p,) = D.processors_for(tiles, TOK, prompt_length=3, request=req)
    assert req["_mlx2_det_tile_uids"] == [p.uid]
    with pytest.raises(D.DetTileRefused):
        D.request_toggles(tiles, {"det_tiles": {"fact": True}})
    with pytest.raises(ValueError):
        D.request_toggles(tiles, {"det_tiles": {"calculator": 1}})
    with pytest.raises(D.DetTileRefused):
        D.request_toggles(None, {"det_tiles": {"calculator": True}})
    assert D.request_toggles(None, {"det_tiles": {"calculator": False}}) == {}
    rec = D.request_receipt(tiles, req)
    assert rec["on"] == {"calculator": True} and rec["calculator"]["plans"] == 0
    assert rec["apcv2_namespace"]["amoeba_det_tiles"] == tiles.identity["manifest_sha256"]


def test_manifest_read_point_rules_and_identity(tmp_path):
    man = json.loads(_manifest(tmp_path, bias=1.0).read_text())
    assert man["tiles"][0]["read_point"] == "layer_output_capture"
    ident = D.tile_set_identity(D.load_manifest(tmp_path / "calc"))
    assert ident["tiles"][0]["read_point"] == "layer_output_capture"
    legacy = D.tile_set_identity(D.load_manifest(_manifest(tmp_path, bias=1.0, name="legacy",
                                                           read_point="mlp_input")))
    assert "read_point" not in legacy["tiles"][0]
    # An mlp_input calculator is not served on the speculative route.
    with pytest.raises(D.DetTileRefused):
        D.build(tmp_path / "legacy", None, source_fingerprint="tiny", layers_count=2, route="native_mtp")
    bad = D.write_manifest(tmp_path / "bad", source={"artifact": "t", "fingerprint": "tiny"}, layers=2,
                           tiles=[{"id": "c", "kind": "calculator", "write_mode": "force", "read_layer": 0,
                                   "write_layer": 1, "read_point": "layer_output_capture",
                                   "gate": {"bias": 0.0, "threshold": 0.5}}],
                           tensors={"c.probe_w": mx.zeros((32,))})
    with pytest.raises(ValueError):
        D.load_manifest(bad)


def test_capture_tiles_refuse_mlp_input_only_features_and_check_route(tmp_path):
    # Merge of the resident-calculator work: features that read the mlp_input stash or carry device
    # state the position-matched verify ledger does not rewind are refused on capture tiles.
    for i, extra in enumerate(({"trigger_prefilter": "gpu"}, {"complete_literal_requests": True},
                               {"parser": D.PARSER_V5, "arithmetic_backend": "gpu"})):
        gate = {"bias": 0.0, "threshold": 0.5}
        if extra.get("parser") == D.PARSER_V5:
            gate["kind"] = "validated_syntax"
        bad = D.write_manifest(tmp_path / f"x{i}", source={"artifact": "t", "fingerprint": "tiny"}, layers=2,
                               tiles=[{"id": "c", "kind": "calculator", "write_mode": "force", "read_layer": 0,
                                       "write_layer": 0, "read_point": "layer_output_capture",
                                       "gate": gate, **extra}],
                               tensors={"c.probe_w": mx.zeros((32,))})
        with pytest.raises(ValueError, match="layer_output_capture"):
            D.load_manifest(bad)
    # One check_route for every admitted route: capture tiles installed for native_mtp serve both
    # ordinary and native_mtp; an mlp_input set installed for ordinary serves ordinary only.
    ts = D.build(_manifest(tmp_path, bias=1.0, name="ok"), None, source_fingerprint="tiny",
                 layers_count=2, route="native_mtp")
    ts.check_route("ordinary")
    ts.check_route("native_mtp")
    with pytest.raises(D.DetTileRefused):
        ts.check_route("external_draft")
    legacy = D.build(_manifest(tmp_path, bias=1.0, name="legacy2", read_point="mlp_input"), None,
                     source_fingerprint="tiny", layers_count=2)
    legacy.check_route("ordinary")
    with pytest.raises(D.DetTileRefused):
        legacy.check_route("native_mtp")
    # A request served by resident (device) processors has no per-request counters to report.
    rec = D.request_receipt(legacy, {"_mlx2_det_tile_resident": True, "_mlx2_det_tiles_on": {"calculator": True}})
    assert rec["calculator"] is None and rec["calculator_scope"].startswith("process_cumulative_only")
    (p,) = D.processors_for(legacy, TOK, prompt_length=0, request=(req := {}))
    assert "_mlx2_det_tile_resident" not in req and req["_mlx2_det_tile_uids"] == [p.uid]
    assert D.request_receipt(legacy, req)["calculator"]["plans"] == 0


def test_det_tile_flag_resolves_the_manifest(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mlx2 import server

    man = _manifest(tmp_path / "Tiny-Model", bias=1.0, name="calculator")
    monkeypatch.setattr(server, "DET_TILE_ROOT", tmp_path)
    args = SimpleNamespace(det_tile="calculator", model=str(tmp_path / "Tiny-Model"))
    assert server.merge_invocation_det_tile(args, None) == {"amoeba_det_tiles": str(man)}
    args = SimpleNamespace(det_tile=f"calculator={man.parent}", model="/nowhere")
    assert server.merge_invocation_det_tile(args, {"x": 1}) == {"x": 1, "amoeba_det_tiles": str(man)}
    with pytest.raises(ValueError):
        server.merge_invocation_det_tile(SimpleNamespace(det_tile=f"fact={man}", model="/n"), None)
    with pytest.raises(ValueError):
        server.merge_invocation_det_tile(SimpleNamespace(det_tile=f"calculator={man}", model="/n"),
                                         {"amoeba_det_tiles": "other"})
    assert server.merge_invocation_det_tile(SimpleNamespace(det_tile=None, model="/n"), None) is None


def test_metrics_report_loaded_tiles_and_fires(tmp_path):
    from types import SimpleNamespace

    from mlx2 import prometheus

    b = prometheus.PrometheusBuilder()
    prometheus._add_det_tile_metrics(b, SimpleNamespace(adapter=SimpleNamespace(amoeba_det_tiles=None)))
    assert "mlx2_det_tiles_loaded 0" in b.render()
    model = _model()
    tiles = _install(model, _manifest(tmp_path, bias=10.0))
    calc = D.CalculatorProcessor(tiles, "calculator", TOK, prompt_length=len(PROMPT))
    _run(model, PROMPT, [_scripted(len(PROMPT), SCRIPT), calc], mtp=False)
    b = prometheus.PrometheusBuilder()
    prometheus._add_det_tile_metrics(b, SimpleNamespace(adapter=SimpleNamespace(amoeba_det_tiles=tiles)))
    text = b.render()
    assert "mlx2_det_tiles_loaded 1" in text
    assert "mlx2_det_tile_calculator_plans 1" in text
    assert 'read_point="layer_output_capture"' in text


def test_fit_backend_capture_equals_the_served_read(tmp_path):
    """The probe is fitted on exactly what the served reader gates on."""
    from types import SimpleNamespace

    from research.amoeba.det_backend_fn import BackendFN

    model = _model()
    be = object.__new__(BackendFN)
    be.trunk = model.language_model.model
    be.adapter = SimpleNamespace(model=model)
    ids = mx.array([TOK.encode("Q: so 5 + 7 =")])
    _, caps, _ = be.forward(ids, record={0}, stop_after=0)
    fitted = caps[0][0]
    tiles = _install(model, _manifest(tmp_path, bias=0.0))
    seen = {}
    tiles.tiles[0].gate.open = lambda x: (seen.setdefault("x", x), mx.zeros(x.shape[:2], mx.bool_))[1]
    keep = D.CalculatorProcessor(tiles, "calculator", TOK)   # one live processor: the reader runs
    model(ids)
    assert keep in tiles.live
    assert mx.array_equal(seen["x"], fitted).item()


def test_row_binding_holds_for_one_call_only(tmp_path):
    model = _model()
    tiles = _install(model, _manifest(tmp_path, bias=10.0))
    calc = D.CalculatorProcessor(tiles, "calculator", TOK, prompt_length=0)
    seen = []
    tiles.calc_gate_at = lambda tid, hist, row=None: (seen.append(row), (None, "gate_unmatched"))[1]
    logits = mx.zeros((1, 64))
    calc.bind_row(3)
    calc(mx.array(TOK.encode("so 5 + 7 =")), logits)
    calc(mx.array(TOK.encode("so 5 + 8 =")), logits)     # no bind: an unbound (all-rows) search
    assert seen == [3, None]


def test_recovery_copy_stays_live_and_restores_counts(tmp_path):
    import copy as _copy
    import gc

    model = _model()
    tiles = _install(model, _manifest(tmp_path, bias=10.0))
    calc = D.CalculatorProcessor(tiles, "calculator", TOK, prompt_length=0)
    calc.counts["plans"] = 1
    snap = _copy.deepcopy(calc)                  # recovery snapshot
    calc.counts["plans"] = 3                     # rows after the snapshot ...
    restored = _copy.deepcopy(snap)              # ... restore clones the snapshot
    calc.__dict__.clear()
    calc.__dict__.update(restored.__dict__)      # hybrid_speculative restores in place
    del snap, restored
    gc.collect()
    assert calc in tiles.live
    calc(mx.array(TOK.encode("hello")), mx.zeros((1, 64)))
    assert tiles.request_receipt([calc.uid])["plans"] == 1   # replayed rows are not counted twice


def test_mtp_binds_the_verify_row(tmp_path):
    """Two lanes at width 2 under self-MTP: each verify call is bound to its own lane's row."""
    from mlx2.runtime.generate import BatchGenerator

    model = _model()
    tiles = _install(model, _manifest(tmp_path, bias=10.0))
    rows_seen = []
    orig = tiles.calc_gate_at

    def spy(tid, hist, row=None):
        rows_seen.append(row)
        return orig(tid, hist, row)
    tiles.calc_gate_at = spy
    prompts = [TOK.encode("Q: so"), TOK.encode("Q: ok so")]
    gen = BatchGenerator(model, completion_batch_size=2, prefill_batch_size=2, prefill_step_size=32,
                         self_mtp={"num_draft": 2, "persistent": True, "segment_aware_live_tip": True,
                                   "segment_aware_cohort_size": 2})
    calcs = [D.CalculatorProcessor(tiles, "calculator", TOK, prompt_length=len(p)) for p in prompts]
    try:
        uids = gen.insert(prompts, max_tokens=[12, 12],
                          logits_processors=[[_scripted(len(p), SCRIPT), c] for p, c in zip(prompts, calcs)],
                          lane_rngs=[LaneRNG(7), LaneRNG(8)], self_mtp_configs=[{"sampling_temp": 0.0}] * 2)
        out = {u: [] for u in uids}
        done = set()
        for _ in range(200):
            _, responses = gen.next()
            for r in responses:
                out[r.uid].append(r.token)
                if r.finish_reason:
                    done.add(r.uid)
            if len(done) == 2:
                break
    finally:
        gen.close()
    for u in uids:
        assert TOK.decode(out[u]).startswith(" 5 + 7 = 12"), TOK.decode(out[u])
    for c in calcs:
        cnt = tiles.request_receipt([c.uid])
        assert cnt["plans"] == 1 and cnt.get("gate_ambiguous", 0) == 0, cnt


def test_ab_driver_scoring_pairs_and_errors():
    import sys as _sys
    _sys.path.insert(0, "scripts")
    import amoeba_fn_calc_ab as A

    def row(pair, arm, ok, err=None, plans=0, content="x"):
        return {"task": "mul:3:2", "pair": pair, "arm": arm, "ok": ok, "http_error": err, "plans": plans,
                "content": content, "det": {"calculator": {"plans": plans}}, "decode_tps": None, "ttft_s": None}
    rows = [row(0, "on", True, plans=1, content="12"), row(0, "off", False, content="13"),
            row(1, "on", True), row(1, "off", True),
            row(2, "on", False, err="transport"), row(2, "off", True)]
    s = A.summarize_scored(rows)["mul:3:2"]
    assert s["n"] == 2 and s["acc_on"] == 1.0 and s["acc_off"] == 0.5 and s["on_only"] == 1
    assert s["errors"] == 1 and s["pairs_dropped_for_errors"] == 1
    assert s["text_differs"] == 1 and s["differs_without_fire"] == 0 and s["fire_rate_on"] == 0.5
    assert A.paired_bootstrap([], []) == {"n": 0}
