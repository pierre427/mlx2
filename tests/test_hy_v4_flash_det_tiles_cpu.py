"""Hy4-flash deterministic calculator tile (CPU, tiny random checkpoint).

The calculator is a probe-gated, position-matched tile (``read_point``
``layer_output_capture``): the probe reads the f32 mean of the four
hyper-connection streams after its read layer, through
``Model.layer_output_readers``.  On the default DFly2 route the lane's
processors run on every verify row, so the committed tokens must equal
ordinary greedy decode with the same forced spans, for one lane and for a
packed cohort, and the DFly2 drafter's state must stay that of the committed
history.  Refusal paths fail closed before weights load.

The tiny model has random weights, so the probe is a constant (zero weights,
a bias that keeps the gate open or closed); the gate read, the position
matching, the parser and the exact write are the served ones.
"""

import json

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_cpu import (  # noqa: E402
    _drain,
    _write_artifact,
    build,
    checkpoint,
    tiny_config,
)

from mlx2.runtime import amoeba_det_tiles as D  # noqa: E402

READ_LAYER = 3
CHARS = list("0123456789 +-*/=.,?\n") + list("abcdefghijklmnopqrstuvwxyz") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@pytest.fixture(scope="module")
def tiny():
    config = tiny_config()
    W = checkpoint(config)
    model, drafter = build(config, W)
    return config, W, model, drafter


def _char_artifact(path, config, W):
    """The tiny checkpoint with a one-character-per-token tokenizer (digits, operators, letters)."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    path.mkdir(parents=True, exist_ok=True)
    _write_artifact(path, config, W)
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2, "<mask>": 3}
    for c in CHARS:
        vocab[c] = len(vocab)
    while len(vocab) < config["text_config"]["vocab_size"]:
        vocab[f"<f{len(vocab)}>"] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "</s>", "unk_token": "<unk>"}))
    return path


@pytest.fixture(scope="module")
def artifact(tiny, tmp_path_factory):
    config, W, _, _ = tiny
    return _char_artifact(tmp_path_factory.mktemp("hy4-det") / "Hy4-tiny", config, W)


def _fingerprint(path):
    from mlx2.adapters import hy_v4_flash as A

    return A.inspect_artifact(str(path))["identity"]["fingerprint"]


def _manifest(out, artifact, *, bias=20.0, layers=6, fingerprint=None, extra=None, probe_w=None, **tile):
    from mlx2.adapters import hy_v4_flash as A

    spec = {"id": "calculator", "kind": "calculator", "write_mode": "force",
            "read_layer": READ_LAYER, "write_layer": READ_LAYER,
            "gate": {"bias": bias, "threshold": 0.5, "probe": "test constant"},
            "parser": D.PARSER_V2, "read_point": "layer_output_capture", **tile}
    d = tiny_config()["text_config"]["hidden_size"]
    return str(D.write_manifest(
        out, source={"artifact": "Hy4-tiny", "fingerprint": fingerprint or _fingerprint(artifact)},
        layers=layers, tiles=[spec],
        tensors={"calculator.probe_w": mx.zeros((d,)) if probe_w is None else probe_w},
        extra={"capture": A.DET_TILE_CAPTURE, "fit_execution_policy": dict(FIT), **(extra or {})}))


FIT = {"engram_table_format": "bf16", "engram_precision": "transformers", "latent_kv": "native",
       "moe_top_k": None}


@pytest.fixture(scope="module")
def manifests(artifact, tmp_path_factory):
    root = tmp_path_factory.mktemp("hy4-det-tiles")
    return {"open": _manifest(root / "open", artifact, bias=20.0),
            "closed": _manifest(root / "closed", artifact, bias=-20.0)}


def _adapter(artifact, policy):
    from mlx2.adapters import hy_v4_flash as A

    return A.HYV4FlashAdapter(str(artifact), execution_policy=policy)


def _encode(adapter, text):
    return adapter.tokenizer.encode(text, add_special_tokens=False)


def _answer_ids(adapter, text):
    answer, info = D.calculator_answer(text, parser=D.PARSER_V2)
    assert answer is not None, info
    return answer, _encode(adapter, answer)


def _request(text):
    return {"prompt": text}


def _ordinary_reference(adapter, prompt, n, processors):
    """Ordinary greedy decode (cached, one token per step) through ``processors``."""
    model = adapter.model
    cache = model.make_cache()
    ids = list(prompt)
    logits = model(mx.array([prompt]), cache)[:, -1]
    out = []
    for _ in range(n):
        for p in processors:
            if hasattr(p, "bind_row"):
                p.bind_row(0)
            logits = p(mx.array(ids), logits)
        t = int(mx.argmax(logits[0]).item())
        out.append(t)
        ids.append(t)
        logits = model(mx.array([[t]]), cache)[:, -1]
    return out


def _forced_greedy(model, prompt, n, spans):
    """Plain greedy with fixed forced spans ``[(history length, ids)]`` (no calculator)."""
    cache = model.make_cache()
    ids = list(prompt)
    logits = model(mx.array([prompt]), cache)[0, -1]
    out = []
    for _ in range(n):
        t = None
        for start, toks in spans:
            if start <= len(ids) < start + len(toks):
                t = toks[len(ids) - start]
        if t is None:
            t = int(mx.argmax(logits).item())
        out.append(t)
        ids.append(t)
        logits = model(mx.array([[t]]), cache)[0, -1]
    return out


# Arithmetic the v2 parser accepts at the end of a prompt (answer right after it).
EXPRESSIONS = ["so 12*34=", "We need 345 * 67 =", "then 1234+5678="]


# --------------------------------------------------------------- wiring --

def test_ordinary_wiring_receipts_and_namespace(artifact, manifests):
    adapter = _adapter(artifact, {"amoeba_det_tiles": manifests["open"]})
    try:
        tiles = adapter.amoeba_det_tiles
        # An ordinary install never admits the external-draft route.
        assert tiles.route == "ordinary" and "external_draft" not in tiles.served_routes
        with pytest.raises(D.DetTileRefused):
            tiles.check_route("external_draft")
        assert set(adapter.model.layer_output_readers) == {READ_LAYER}
        assert adapter.identity["amoeba_det_tiles_manifest_sha256"] == tiles.identity["manifest_sha256"]
        assert adapter.execution_numerics_contract() == {"amoeba_det_tiles": tiles.identity}
        diag = adapter.diagnostics()["amoeba_det_tiles"]
        assert diag["qualified"] is False and diag["qualification"] == "unqualified"
        assert diag["capture"] == "hc-stream-mean-f32-v1" and diag["implemented"]
        assert diag["fit_execution_policy"] == FIT
        assert "speculative_protocol" not in diag
        request = _request("so 12*34=")
        procs = adapter.request_logits_processors(request, prompt_length=9)
        assert len(procs) == 1 and type(procs[0]) is D.CalculatorProcessor
        receipt = D.request_receipt(tiles, request)
        assert receipt["on"] == {"calculator": True} and receipt["qualification"] == "unqualified"
        off = {"prompt": "x", "det_tiles": {"calculator": False}}
        assert adapter.request_logits_processors(off, prompt_length=1) == ()
        with pytest.raises(D.DetTileRefused, match="unknown tile"):
            adapter.request_logits_processors({"prompt": "x", "det_tiles": {"fact": True}}, prompt_length=1)
        # Nothing is stashed while no calculator processor is live.
        del procs
        import gc

        gc.collect()
        before = tiles.out_forwards
        adapter.model(mx.array([_encode(adapter, "plain text")]))
        assert tiles.out_forwards == before
    finally:
        adapter.close()


def test_dfly2_route_admits_external_draft_only_with_matched_tiles(artifact, manifests):
    adapter = _adapter(artifact, {"amoeba_det_tiles": manifests["open"], "draft_model": "embedded_dfly",
                                  "num_draft": 3})
    try:
        tiles = adapter.amoeba_det_tiles
        assert tiles.route == "external_draft"
        assert tiles.served_routes == ("ordinary", "external_draft")
        tiles.check_route("external_draft")
        with pytest.raises(D.DetTileRefused):
            tiles.check_route("native_mtp")
        receipt = tiles.receipt()
        assert receipt["speculative_protocol"] == "position-matched-calculator-verify-ledger-v1"
        assert "tree_verifier" not in receipt and not tiles.resident_external
        assert tiles.stash_forwards == 4 * 32 + 8
    finally:
        adapter.close()


def test_no_tiles_adapter_has_no_processors_and_no_contract(artifact):
    adapter = _adapter(artifact, {})
    try:
        assert adapter.amoeba_det_tiles is None and not adapter.model.layer_output_readers
        assert adapter.request_logits_processors({"prompt": "1+1="}, prompt_length=4) == ()
        assert adapter.execution_numerics_contract() is None
        assert adapter.diagnostics()["amoeba_det_tiles"]["selected"] is False
        with pytest.raises(D.DetTileRefused, match="loaded no deterministic tiles"):
            adapter.request_logits_processors({"prompt": "x", "det_tiles": {"calculator": True}},
                                              prompt_length=1)
    finally:
        adapter.close()


def test_reader_sees_streams_padding_and_committed_prefix(tiny):
    from mlx2.adapters.hy_v4_flash import det_tile_capture
    from mlx2.runtime.models.hy_v4_flash import HY4FlashBatchLayerCache

    _, _, model, _ = tiny
    seen = []
    model.layer_output_readers[READ_LAYER] = lambda i, s, ids, prefix: seen.append((i, s, ids, prefix))
    try:
        a, b = model.make_cache(), model.make_cache()
        model(mx.array([[4, 5, 6, 7, 8]]), a)
        model(mx.array([[9, 10, 11]]), b)
        seen.clear()
        # A right-padded joint forward (the chain-packed wave shape).
        batch = [HY4FlashBatchLayerCache([x, y]) for x, y in zip(a, b)]
        for c in batch:
            c.prepare(lengths=[3, 1], right_padding=[0, 2])
        model(mx.array([[20, 21, 22], [30, 30, 30]]), batch)
        for c in batch:
            c.finalize()
        (layer, streams, ids, prefix), = seen
        assert layer == READ_LAYER and streams.ndim == 4 and streams.shape[2] == 4
        assert ids.tolist() == [[20, 21, 22], [30, -1, -1]]
        assert prefix[0][-5:] == [4, 5, 6, 7, 8] and prefix[1][-3:] == [9, 10, 11]
        cap = det_tile_capture(streams)
        np.testing.assert_allclose(np.array(cap), np.array(streams.astype(mx.float32)).mean(axis=2),
                                   rtol=1e-6, atol=1e-6)
        with pytest.raises(ValueError):
            det_tile_capture(cap)
    finally:
        model.layer_output_readers.clear()


def test_padding_never_matches_and_prefix_extends_the_window(tiny, artifact, manifests):
    tiles = D.build(manifests["open"], None, source_fingerprint=_fingerprint(artifact), layers_count=6,
                    route="external_draft")
    proc_holder = object.__new__(D.CalculatorProcessor)
    tiles.live.add(proc_holder)            # stash only while a processor is live
    hidden = mx.zeros((2, 3, 4, 32))
    from mlx2.adapters.hy_v4_flash import det_tile_capture

    # Row 1 is padding past its first token; row 0's first token has a committed prefix.
    tiles.on_layer_output(READ_LAYER, hidden, mx.array([[7, 8, 9], [7, -1, -1]]), det_tile_capture,
                          prefix_ids=[[1, 2, 3], [5, 6]])
    assert tiles.calc_gate_at("calculator", [1, 2, 3, 7]) == (True, None)
    assert tiles.calc_gate_at("calculator", [5, 6, 7]) == (True, None)
    # A history that matches no row's real prefix stays unmatched (row 1's pads never match).
    assert tiles.calc_gate_at("calculator", [4, 4, 7]) == (None, "gate_unmatched")
    assert tiles.calc_gate_at("calculator", [7, -1]) == (None, "gate_unmatched")


def test_gate_is_read_from_the_matching_row_and_position(artifact, tmp_path):
    """A probe on stream-mean dimension 0 (open iff it is positive): distinct gates per position, so a
    read from the wrong row, position or prefix offset is caught. Two rows share their first input
    token (the verify anchor of two lanes); their committed prefixes tell them apart."""
    from mlx2.adapters.hy_v4_flash import det_tile_capture

    w = mx.zeros((32,)).at[0].add(1.0)
    man = _manifest(tmp_path / "dim0", artifact, bias=0.0, probe_w=w)
    tiles = D.build(man, None, source_fingerprint=_fingerprint(artifact), layers_count=6,
                    route="external_draft")
    holder = object.__new__(D.CalculatorProcessor)    # a live processor (weakly held by the set)
    tiles.live.add(holder)
    signs = np.array([[+1, -1, +1, -1], [-1, +1, +1, -1]], np.float32)    # row 1 has one pad
    hidden = np.zeros((2, 4, 4, 32), np.float32)
    hidden[:, :, :, 0] = signs[:, :, None] * np.array([0.5, 1.5, -0.25, 0.25])  # stream mean = sign/2
    tiles.on_layer_output(READ_LAYER, mx.array(hidden), mx.array([[30, 31, 32, 33], [30, 41, 42, -1]]),
                          det_tile_capture, prefix_ids=[[1, 2, 3, 4, 5, 6, 7, 8, 9], [11, 12]])
    gate = lambda history: tiles.calc_gate_at("calculator", history)  # noqa: E731
    # Anchor token 30 in both rows: the prefixes pick the row.
    assert gate([7, 8, 9, 30]) == (True, None)
    assert gate([11, 12, 30]) == (False, None)
    # Later positions, with the window running back through the prefix.
    assert gate([8, 9, 30, 31]) == (False, None)
    assert gate([9, 30, 31, 32, 33]) == (False, None)
    assert gate([3, 4, 5, 6, 7, 8, 9, 30, 31, 32]) == (True, None)
    assert gate([12, 30, 41]) == (True, None)
    assert gate([12, 30, 41, 42]) == (True, None)
    # A wrong prefix never borrows another row's gate.
    assert gate([5, 30]) == (None, "gate_unmatched")
    # Without prefixes the shared anchor is ambiguous and fails closed.
    tiles.on_layer_output(READ_LAYER, mx.array(hidden), mx.array([[30, 31, 32, 33], [30, 41, 42, -1]]),
                          det_tile_capture)
    assert gate([7, 8, 9, 30]) == (None, "gate_ambiguous")


# ---------------------------------------------------- forced tokens: ordinary --

def test_forced_span_on_ordinary_is_the_exact_result(artifact, manifests):
    adapter = _adapter(artifact, {"amoeba_det_tiles": manifests["open"]})
    try:
        text = "We need 345 * 67 ="
        prompt = _encode(adapter, text)
        answer, ids = _answer_ids(adapter, text)
        assert answer.strip() == str(345 * 67)
        request = _request(text)
        procs = adapter.request_logits_processors(request, prompt_length=len(prompt))
        out = _ordinary_reference(adapter, prompt, 12, procs)
        assert out[:len(ids)] == ids
        # The forced span is exactly a fixed-span forced decode of the same model.
        assert out == _forced_greedy(adapter.model, prompt, 12, [(len(prompt), ids)])
        calc = D.request_receipt(adapter.amoeba_det_tiles, request)["calculator"]
        assert calc["plans"] == 1 and calc["tokens_written"] == len(ids) and calc["gate_open"] == 1
    finally:
        adapter.close()


@pytest.mark.parametrize("text", ["The story begins on a quiet hill.", "Call 555-1234 on 12/25 at 10:30."])
def test_open_gate_never_fires_without_a_complete_expression(artifact, manifests, text):
    plain = _adapter(artifact, {})
    calc = _adapter(artifact, {"amoeba_det_tiles": manifests["open"]})
    try:
        prompt = _encode(plain, text)
        want = _ordinary_reference(plain, prompt, 10, ())
        request = _request(text)
        got = _ordinary_reference(calc, prompt, 10, calc.request_logits_processors(
            request, prompt_length=len(prompt)))
        assert got == want
        receipt = D.request_receipt(calc.amoeba_det_tiles, request)["calculator"]
        assert receipt["plans"] == 0 and receipt["tokens_written"] == 0
    finally:
        plain.close()
        calc.close()


# ------------------------------------------------- forced tokens: DFly2 / AST --

def _external(adapter, texts, n, *, width=None, max_extra=6):
    batch = adapter.create_external_batch(completion_batch_size=width or len(texts), prefill_step_size=4)
    prompts = [_encode(adapter, t) for t in texts]
    requests = [_request(t) for t in texts]
    procs = [list(adapter.request_logits_processors(r, prompt_length=len(p)))
             for r, p in zip(requests, prompts)]
    uids = batch.insert(prompts, max_tokens=[n + max_extra] * len(texts),
                        sampling_configs=[{"sampling_temp": 0}] * len(texts), logits_processors=procs)
    out = _drain(batch, uids, n)
    return batch, uids, prompts, requests, [out[u][:n] for u in uids]


@pytest.mark.parametrize("policy", [
    {"draft_model": "embedded_dfly", "num_draft": 3},
    {"draft_model": "embedded_dfly", "num_draft": 3, "ast_target_executor": "serial"},
    {"draft_model": "prompt_lookup"},
], ids=["dfly2", "dfly2-serial", "pld"])
@pytest.mark.parametrize("width", [1, 2, 3])
def test_external_route_commits_ordinary_greedy_with_the_same_forced_spans(artifact, manifests,
                                                                          policy, width):
    n = 14
    texts = EXPRESSIONS[:width]
    ordinary = _adapter(artifact, {"amoeba_det_tiles": manifests["open"]})
    try:
        refs, spans = [], []
        for text in texts:
            prompt = _encode(ordinary, text)
            _, ids = _answer_ids(ordinary, text)
            ref = _ordinary_reference(ordinary, prompt, n, ordinary.request_logits_processors(
                _request(text), prompt_length=len(prompt)))
            assert ref[:len(ids)] == ids
            refs.append(ref)
            spans.append(ids)
    finally:
        ordinary.close()
    adapter = _adapter(artifact, {"amoeba_det_tiles": manifests["open"], **policy})
    try:
        batch, uids, _, requests, outs = _external(adapter, texts, n)
        assert outs == refs
        tiles = adapter.amoeba_det_tiles
        for request, ids in zip(requests, spans):
            calc = D.request_receipt(tiles, request)["calculator"]
            # Verify rows past the accept point were settled away: one plan, its tokens once.
            assert calc["plans"] == 1 and calc["tokens_written"] == len(ids), calc
        stats = batch.scheduler_stats
        assert stats.get("external_target_tree_target_rounds", 0) > 0 or policy["draft_model"] == "prompt_lookup"
        if policy["draft_model"] == "embedded_dfly":
            assert adapter.diagnostics()["dfly2"]["observed_used"]
        assert tiles.receipt()["observed_used"] and tiles.receipt()["route"] == "external_draft"
    finally:
        adapter.close()


class _Script:
    """History-pure writer: forces ``ids`` once the history reaches ``start`` (stands in for a model
    that writes an expression mid-answer, so the calculator fires on a verify row)."""

    def __init__(self, start, ids):
        self.start, self.ids = start, ids

    def __call__(self, tokens, logits):
        n = int(tokens.size) - self.start
        if 0 <= n < len(self.ids):
            m = np.full((logits.shape[-1],), -1e9, np.float32)
            m[self.ids[n]] = 0.0
            return logits + mx.array(m)[None]
        return logits


@pytest.mark.parametrize("width", [1, 2])
def test_calculator_fires_mid_generation_inside_dfly2_verify_rows(artifact, manifests, width):
    texts = ["The story begins", "A quiet hill"][:width]
    n = 24
    ordinary = _adapter(artifact, {"amoeba_det_tiles": manifests["open"]})
    try:
        expr = _encode(ordinary, " so 12*34=")
        _, answer = _answer_ids(ordinary, "so 12*34=")
        refs = []
        for text in texts:
            prompt = _encode(ordinary, text)
            script = _Script(len(prompt) + 3, expr)
            procs = [script, *ordinary.request_logits_processors(_request(text), prompt_length=len(prompt))]
            ref = _ordinary_reference(ordinary, prompt, n, procs)
            at = 3 + len(expr)
            assert ref[3:at] == expr and ref[at:at + len(answer)] == answer
            refs.append(ref)
    finally:
        ordinary.close()
    adapter = _adapter(artifact, {"amoeba_det_tiles": manifests["open"], "draft_model": "embedded_dfly",
                                  "num_draft": 3})
    try:
        batch = adapter.create_external_batch(completion_batch_size=width, prefill_step_size=4)
        prompts = [_encode(adapter, t) for t in texts]
        requests = [_request(t) for t in texts]
        procs = [[_Script(len(p) + 3, expr), *adapter.request_logits_processors(r, prompt_length=len(p))]
                 for r, p in zip(requests, prompts)]
        uids = batch.insert(prompts, max_tokens=[n + 4] * width,
                            sampling_configs=[{"sampling_temp": 0}] * width, logits_processors=procs)
        out = _drain(batch, uids, n)
        assert [out[u][:n] for u in uids] == refs
        for request in requests:
            calc = D.request_receipt(adapter.amoeba_det_tiles, request)["calculator"]
            assert calc["plans"] == 1 and calc["tokens_written"] == len(answer), calc
        assert batch.scheduler_stats.get("external_target_tree_target_rounds", 0) > 0
    finally:
        adapter.close()


def test_dfly2_drafter_state_follows_the_committed_history(tiny, artifact, manifests):
    """After a forced span, the drafter's context (its DFlyFeatureCache rows and the shared target
    KV/indexer state it reads) is that of the committed tokens: equal to a fresh target pass."""
    _, _, _, _ = tiny
    adapter = _adapter(artifact, {"amoeba_det_tiles": manifests["open"], "draft_model": "embedded_dfly",
                                  "num_draft": 3})
    try:
        batch, uids, prompts, requests, outs = _external(adapter, ["so 12*34="], 12, max_extra=12)
        _, answer = _answer_ids(adapter, "so 12*34=")
        assert outs[0][:len(answer)] == answer
        assert D.request_receipt(adapter.amoeba_det_tiles, requests[0])["calculator"]["plans"] == 1
        lane = batch.lanes[uids[0]]
        feature = lane.draft_cache[0]
        history = list(lane.history)
        assert history[:len(prompts[0])] == prompts[0]
        generated = history[len(prompts[0]):]
        # The lane's history is the committed stream (the latest bonus token may still be pending).
        assert len(generated) >= 8 and generated[:12] == outs[0][:len(generated[:12])]
        covered = feature.offset
        assert covered > len(prompts[0])
        ref = adapter.model.hidden(mx.array([history[:covered]]))
        k = feature.features.shape[1]
        np.testing.assert_allclose(np.array(feature.features[0]), np.array(ref[0, covered - k:covered]),
                                   rtol=2e-4, atol=2e-4)
        # The shared target cache (KV Share / Index Share source) holds the committed history only:
        # rejected draft rows were trimmed (the bonus token may still be waiting for its forward).
        assert lane.cache[0].offset in (len(history), len(history) - 1)
        assert lane.cache[0].offset >= covered
    finally:
        adapter.close()


def test_closed_gate_on_dfly2_equals_plain_dfly2(artifact, manifests):
    texts = EXPRESSIONS[:2]
    plain = _adapter(artifact, {"draft_model": "embedded_dfly", "num_draft": 3})
    calc = _adapter(artifact, {"amoeba_det_tiles": manifests["closed"], "draft_model": "embedded_dfly",
                               "num_draft": 3})
    try:
        _, _, _, _, want = _external(plain, texts, 12)
        _, _, _, requests, got = _external(calc, texts, 12)
        assert got == want
        for request in requests:
            c = D.request_receipt(calc.amoeba_det_tiles, request)["calculator"]
            assert c["plans"] == 0 and c["gate_closed"] >= 1
    finally:
        plain.close()
        calc.close()


# ------------------------------------------------------ probe-fit backend --

def test_probe_fit_backend_captures_what_serving_reads(artifact, tmp_path):
    """research/amoeba/det_backend_hy4.py: caps[l] is det_tile_capture of layer l's output through
    the served reader hook, and a manifest carrying its fit policy is admitted by the adapter."""
    from mlx2.adapters import hy_v4_flash as A
    from research.amoeba.det_backend_hy4 import BackendHy4

    donor = BackendHy4(path=str(artifact), policy={}, wire=False, watchdog=False, log=lambda *_: None)
    try:
        assert donor.L == 6 and donor.capture == A.DET_TILE_CAPTURE and donor.chat_format is None
        ids = donor.encode("We need 345 * 67 =")
        logits, caps, _ = donor.forward(mx.array([ids]), record={1, READ_LAYER})
        assert set(caps) == {1, READ_LAYER} and not donor.model.layer_output_readers
        seen = {}
        donor.model.layer_output_readers[READ_LAYER] = lambda i, s, *_: seen.setdefault(i, s)
        try:
            ref_logits = donor.model(mx.array([ids]))
        finally:
            donor.model.layer_output_readers.clear()
        np.testing.assert_array_equal(np.array(caps[READ_LAYER][0]),
                                      np.array(A.det_tile_capture(seen[READ_LAYER])))
        np.testing.assert_array_equal(np.array(logits), np.array(ref_logits))
        assert caps[READ_LAYER][0].shape == (1, len(ids), 32)
        fit = donor.fit_execution_policy
        assert fit == FIT
        man = _manifest(tmp_path / "fit", artifact, extra={"fit_execution_policy": fit})
    finally:
        donor.close()
    adapter = _adapter(artifact, {"amoeba_det_tiles": man})
    try:
        assert adapter.amoeba_det_tiles is not None
    finally:
        adapter.close()


class _FakeDonor:
    """Donor-shaped stub: fixed-size captures per forward, no model."""

    def __init__(self, d=256):
        self.d = d

    def forward(self, ids, *, record=(), stop_after=None, **_):
        T = int(ids.shape[1])
        base = mx.arange(T * self.d, dtype=mx.float32).reshape(1, T, self.d) + float(sum(ids.tolist()[0]))
        return None, {l: (base * (l + 1), None) for l in record}, {}


def _peak_bytes(fn):
    import gc
    import tracemalloc

    gc.collect()
    tracemalloc.start()
    try:
        result = fn()
        return tracemalloc.get_traced_memory()[1], result
    finally:
        tracemalloc.stop()


def test_streamed_feature_memory_does_not_grow_with_forwards(tmp_path):
    """det_calc.capture_to_disk keeps one sequence's captures resident: its peak traced memory is
    the same for 10 and 200 forwards, while the in-memory capture (the control) grows with them.
    The files hold exactly what the in-memory capture returns."""
    from research.amoeba import det_calc as C

    donor, layers = _FakeDonor(), [1, 3, 5]
    seqs = lambda n: [[(7 * i + j) % 50 + 1 for j in range(24)] for i in range(n)]  # noqa: E731
    small, _ = _peak_bytes(lambda: C.capture_to_disk(donor, seqs(10), layers, tmp_path / "s10"))
    large, feats = _peak_bytes(lambda: C.capture_to_disk(donor, seqs(200), layers, tmp_path / "s200"))
    control, ref = _peak_bytes(lambda: C.capture(donor, seqs(200), layers))
    per_forward = 24 * donor.d * 4 * len(layers)
    assert large < small + 4 * per_forward, (small, large)
    assert control > 100 * per_forward > large
    for l in layers:
        assert isinstance(feats[l], np.memmap) and not feats[l].flags.writeable
        np.testing.assert_array_equal(np.asarray(feats[l]), ref[l])
    with pytest.raises(FileExistsError):
        C.capture_to_disk(donor, seqs(2), layers, tmp_path / "s10")


def test_backend_streams_features_and_unloads(artifact, tmp_path):
    from research.amoeba import det_calc as C
    from research.amoeba.det_backend_hy4 import BackendHy4

    donor = BackendHy4(path=str(artifact), policy={}, wire=False, watchdog=False, log=lambda *_: None)
    try:
        seqs = [donor.encode("We need 345 * 67 ="), donor.encode("so 12*34=")]
        on_disk = C.capture_to_disk(donor, seqs, [1, READ_LAYER], tmp_path / "f")
        in_mem = C.capture(donor, seqs, [1, READ_LAYER])
        for l in (1, READ_LAYER):
            np.testing.assert_array_equal(np.asarray(on_disk[l]), in_mem[l])
    finally:
        donor.close()
    assert donor.adapter is None and donor.model is None
    donor.close()   # idempotent


def test_watchdog_counter_is_named_and_selectable():
    import sys

    sys.path.insert(0, "scripts")
    import amoeba27b_lib as A

    assert A.AVAILABLE_COUNTERS == ("vm_stat", "serving")
    assert A.available_gb("vm_stat") > 0
    with pytest.raises(ValueError):
        A.available_gb("free")
    with pytest.raises(ValueError):
        A.footprint_watchdog(1000.0, print, counter="free")


# --------------------------------------------------------------- refusals --

def test_refusals_fail_closed_before_weights_load(artifact, tmp_path):
    from mlx2.adapters import hy_v4_flash as A

    cases = {
        "mlp_input": (_manifest(tmp_path / "a", artifact, read_point="mlp_input"), "layer_output_capture"),
        "fingerprint": (_manifest(tmp_path / "b", artifact, fingerprint="sha256:" + "0" * 64),
                        "another checkpoint"),
        "layers": (_manifest(tmp_path / "c", artifact, layers=7), "layers"),
        "capture": (_manifest(tmp_path / "d", artifact, extra={"capture": "hc-stream-0"}), "fitted on"),
        "no_capture": (_manifest(tmp_path / "d2", artifact, extra={"capture": None}), "fitted on"),
        "fit_policy": (_manifest(tmp_path / "e", artifact,
                                 extra={"fit_execution_policy": {**FIT, "engram_table_format": "iq4_nl"}}),
                       "engram_table_format"),
        "fit_policy_partial": (_manifest(tmp_path / "e2", artifact,
                                         extra={"fit_execution_policy": {"engram_table_format": "bf16"}}),
                               "needs fit_execution_policy"),
        "no_fit_policy": (_manifest(tmp_path / "e3", artifact, extra={"fit_execution_policy": None}),
                          "needs fit_execution_policy"),
        "host_tiles": (_manifest(tmp_path / "f", artifact, extra={"host_tiles_manifest_sha256": "ab" * 32}),
                       "neural-tiled"),
    }
    for name, (manifest, match) in cases.items():
        with pytest.raises(D.DetTileRefused, match=match):
            A.HYV4FlashAdapter(str(artifact), execution_policy={"amoeba_det_tiles": manifest})
    good = _manifest(tmp_path / "g", artifact)
    with pytest.raises(D.DetTileRefused, match="ast_tensorfold_rows"):
        A.HYV4FlashAdapter(str(artifact), execution_policy={
            "amoeba_det_tiles": good, "draft_model": "embedded_dfly", "ast_tensorfold_rows": True,
            "ast_tensorfold_rows_backend": "reference"})
    with pytest.raises(ValueError, match="manifest path"):
        A._parse_policy({"amoeba_det_tiles": 3})
    with pytest.raises(ValueError, match="not found"):
        A.HYV4FlashAdapter(str(artifact), execution_policy={"amoeba_det_tiles": str(tmp_path / "nope")})


def test_fact_tables_and_resident_parsers_are_refused(artifact, tmp_path):
    d = tiny_config()["text_config"]["hidden_size"]
    fact = D.write_manifest(
        tmp_path / "fact", source={"artifact": "Hy4-tiny", "fingerprint": _fingerprint(artifact)}, layers=6,
        tiles=[{"id": "facts", "kind": "fact_table", "write_mode": "delta", "read_layer": 2,
                "write_layer": 3, "table": {"tau": 0.9, "margin": 0.1, "recogniser": "residual_dims"}}],
        tensors={"facts.dims": mx.zeros((2,), dtype=mx.int32), "facts.keys": mx.zeros((1, 2)),
                 "facts.values": mx.zeros((1, d)), "facts.null": mx.zeros((1,))})
    with pytest.raises(D.DetTileRefused, match="probe-gated calculators"):
        D.admit_position_matched(str(fact), source_fingerprint=_fingerprint(artifact), layers_count=6)


def test_generic_external_route_refuses_unmatched_tiles(artifact, tmp_path):
    """Only position-matched calculators get the external-draft route; an MLP-input calculator, whose
    gate has no history match, stays ordinary-only."""
    man = _manifest(tmp_path / "mlp", artifact, read_point="mlp_input")
    with pytest.raises(D.DetTileRefused):
        D.build(man, None, source_fingerprint=_fingerprint(artifact), layers_count=6, route="external_draft")
    ts = D.build(man, None, source_fingerprint=_fingerprint(artifact), layers_count=6)
    assert ts.served_routes == ("ordinary",)
    with pytest.raises(D.DetTileRefused):
        ts.check_route("external_draft")
