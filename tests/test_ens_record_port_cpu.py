"""H54 record port v2 on a tiny random bf16 Qwen3 (CPU only; no checkpoint, no GPU)."""
from __future__ import annotations

import dataclasses
import json
import random
import re

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from research.ens import qwen3 as Q
from research.ens import record_port as RP
from research.ens import sor_mock as M
from research.ens.sor import Record

SPECIALS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>")
_SPLIT = re.compile("(" + "|".join(map(re.escape, SPECIALS)) + ")")


class FakeTok:
    """Byte-level tokenizer with the Qwen chat specials as single ids (256..260)."""

    def __init__(self):
        self.ids = {s: 256 + i for i, s in enumerate(SPECIALS)}
        self.pad = self.ids["<|endoftext|>"]
        self.t = self

    def encode(self, s):
        out = []
        for part in _SPLIT.split(s):
            out += [self.ids[part]] if part in self.ids else list(part.encode())
        return out

    def token_to_id(self, s):
        return self.ids.get(s)

    def decode(self, ids):
        inv = {v: k for k, v in self.ids.items()}
        out, buf = [], []
        for i in ids:
            if i in inv:
                out.append(bytes(buf).decode(errors="replace"))
                buf = []
                out.append(inv[i])
            elif i < 256:
                buf.append(i)
        out.append(bytes(buf).decode(errors="replace"))
        return "".join(out)


@pytest.fixture(autouse=True)
def cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def tiny(layers=4, d=64):
    c = Q.Qwen3Config(hidden_size=d, intermediate_size=2 * d, num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=d // 4, rms_norm_eps=1e-6, rope_theta=1e6, vocab_size=264,
                      tie_word_embeddings=True)
    mx.random.seed(0)
    m = Q.Qwen3(c)
    m.set_dtype(mx.bfloat16)
    m.freeze()
    mx.eval(m.parameters())
    return m


def _tokens(tok, samples):
    T, _, _ = RP.stack_rows(tok, [RP.encode_row(tok, s) for s in samples])
    return T


def _open(port, scale=0.5):
    """Nonzero gates and LoRA B, as after training."""
    for r in port.readers.values():
        r.gate = mx.array([scale])
    for blk in port.lora.values():
        for lo in blk.values():
            lo.b.weight = mx.random.normal(lo.b.weight.shape) * 0.05


def test_no_records_and_init_are_the_original_bitwise():
    m, tok = tiny(), FakeTok()
    ss = [M.sample(0, i) for i in range(3)]
    T = _tokens(tok, ss)
    ref = np.array(m(T).astype(mx.float32))
    port = RP.RecordPortV2(m, site=1, key_dim=32, reader_rank=16)
    empty = RP.build_bank(tok, [[] for _ in ss])
    assert not empty.any
    assert np.array_equal(np.array(RP.forward(m, T, port, empty).astype(mx.float32)), ref)
    full = RP.build_bank(tok, [s.records for s in ss])
    assert full.any
    assert np.array_equal(np.array(RP.forward(m, T, port, full).astype(mx.float32)), ref)   # gates closed at init
    # trained-like port: rows without records in a mixed batch stay the original, rows with records change
    _open(port)
    mixed = RP.build_bank(tok, [ss[0].records, [], ss[2].records])
    got = np.array(RP.forward(m, T, port, mixed).astype(mx.float32))
    assert np.array_equal(got[1], ref[1])
    assert not np.array_equal(got[0], ref[0])


def test_bank_shapes_and_masks():
    tok = FakeTok()
    r1 = [Record("Kel", "tier", "gold", "m", "r"), Record("Kel", "balance", "48,215", "m", "r")]
    r2 = [Record("Orvane Ltd", "renewal date", "2027-03-14", "m", "r")]
    b = RP.build_bank(tok, [r1, [], r2])
    n1 = len(tok.encode(" gold")) + len(tok.encode(" 48,215"))
    n2 = len(tok.encode(" 2027-03-14"))
    assert b.n_slots == [n1, 0, n2]
    B, S = b.slot_m.shape
    assert B == 3 and S % 16 == 0 and S >= max(n1, n2)
    assert np.array(b.slot_m).sum(1).tolist() == [n1, 0, n2]
    assert np.array(b.has).tolist() == [1.0, 0.0, 1.0]
    pos = np.array(b.slot_pos)[0, :n1].tolist()
    assert pos == list(range(len(tok.encode(" gold")))) + list(range(len(tok.encode(" 48,215"))))
    # previous token: the value's own previous token, 0 at j = 0
    v = tok.encode(" gold")
    assert np.array(b.slot_prev)[0, 1:len(v)].tolist() == v[:-1] and np.array(b.slot_val)[0, :len(v)].tolist() == v
    assert b.attr.shape[:2] == b.ent.shape[:2] == (3, 8)
    assert np.array(b.attr_m)[2, 0].sum() == len(tok.encode(" renewal date"))
    assert np.array(b.ent_m)[1].sum() == 0
    long = [Record("E", "a", "x" * 50, "m", "r")]
    bt = RP.build_bank(tok, [long], max_pos=32)
    assert bt.n_slots == [32] and bt.truncated == 1


def test_record_order_does_not_matter():
    m, tok = tiny(), FakeTok()
    s = M.sample(3, 0, n_entities=3)
    assert len(s.records) >= 4
    T = _tokens(tok, [s, s])
    port = RP.RecordPortV2(m, site=1, key_dim=32, reader_rank=16)
    _open(port)
    perm = list(reversed(s.records))
    out = np.array(RP.forward(m, T, port, RP.build_bank(tok, [s.records, perm])).astype(mx.float32))
    assert np.abs(out[0] - out[1]).max() <= 0.05 * np.abs(out[0]).max()
    # the port does see the records: other values change the output
    s2 = M.sample(4, 0)
    other = np.array(RP.forward(m, T[:1], port, RP.build_bank(tok, [s2.records])).astype(mx.float32))
    assert np.abs(other[0] - out[0]).max() > np.abs(out[0] - out[1]).max()


def _train_args(tmp, **kw):
    a = ["train", "--model", "tiny", "--site", "1", "--key-dim", "32", "--reader-rank", "16", "--steps", "6",
         "--batch", "4", "--p-text", "0", "--lr", "5e-3", "--log-every", "1", "--save-every", "0",
         "--no-memory-gate", "--out", str(tmp)]
    for k, v in kw.items():
        a += [f"--{k.replace('_', '-')}", str(v)]
    return a


@pytest.fixture
def fake_core(monkeypatch):
    monkeypatch.setattr(RP, "load_core", lambda model_arg: (tiny(), "tiny-qwen3@cpu", FakeTok()))


def test_training_reduces_loss_on_an_overfit_pool(tmp_path, fake_core):
    out = tmp_path / "run"
    assert RP.main(_train_args(out, steps=60, pool=4, lr="1e-2")) == 0
    log = [r["loss"] for r in json.loads((out / "train_receipt.json").read_text())["log"] if "loss" in r]
    assert len(log) == 60 and np.mean(log[-5:]) < 0.75 * log[0]
    rec = json.loads((out / "train_receipt.json").read_text())
    assert rec["base_checkpoint"] == "tiny-qwen3@cpu" and rec["port_checkpoint"].startswith("ens-h54-recordport2-")
    assert "source_revision" in rec and rec["trainable_params"] > 0
    last = [r for r in rec["log"] if "loss" in r][-1]
    assert {"value_loss", "frame_loss", "value_span_misses"} <= set(last) and last["value_span_misses"] == 0
    assert set(rec["final_losses"]) == {"loss", "value_loss", "frame_loss"} and rec["uniform_nll"] > 5


def test_resume_equals_uninterrupted(tmp_path, fake_core):
    a, b = tmp_path / "a", tmp_path / "b"
    assert RP.main(_train_args(a)) == 0
    assert RP.main(_train_args(b, stop_after=3)) == 3
    assert (b / "resume" / "stream.pkl").exists() and not (b / "train_receipt.json").exists()
    assert RP.main(_train_args(b)) == 0
    wa, wb = mx.load(str(a / "port.safetensors")), mx.load(str(b / "port.safetensors"))
    assert wa.keys() == wb.keys()
    for k in wa:
        assert np.array_equal(np.array(wa[k]), np.array(wb[k])), k
    la = json.loads((a / "train_receipt.json").read_text())["log"][-1]
    lb = json.loads((b / "train_receipt.json").read_text())["log"][-1]
    assert la["truncated"] == lb["truncated"] > 0 and la["value_loss"] == lb["value_loss"]   # counters resume too
    with pytest.raises(SystemExit):                     # changed arguments refuse to resume
        c = tmp_path / "c"
        RP.main(_train_args(c, stop_after=2))
        RP.main(_train_args(c, lr="1e-3"))


def test_text_kl_rows(tmp_path, fake_core, monkeypatch):
    tok = FakeTok()
    wins = [tok.encode("the quick brown fox jumps over the lazy dog again")[:32]] * 8
    monkeypatch.setattr(RP, "text_windows", lambda tok, split: wins)
    assert RP.main(_train_args(tmp_path / "t", steps=2, p_text=0.5)) == 0


def test_scoring_wiring_on_fake_generations():
    ss = [M.sample(5, i, split="held", task="lookup") for i in range(6)]
    outs = {"base": ["I don't know."] * 6, "pasted": [s.answer for s in ss],
            "port": [s.answer for s in ss[:3]] + ["no idea"] * 3, "swapped": ["no idea"] * 6}
    counts = RP.tally(ss, outs)
    assert counts["pasted.strict"] == [6, 6] and counts["port.strict"] == [3, 6] and counts["base.strict"] == [0, 6]
    vt = ss[0].vtype
    assert counts[f"pasted.vtype.{vt}.strict"][1] == sum(s.vtype == vt for s in ss)
    assert sum(v[1] for k, v in counts.items() if k.startswith("port.order.") and k.endswith(".strict")) == 6
    cell = {"meta": {"split": "held", "frames": "held", "task": "lookup", "nrec": 4}, "counts": counts, "n": 6,
            "prompt_tokens": {"base": 60, "pasted": 300, "port": 60, "swapped": 60}, "port_slots": 40}
    sm = RP.summarize({"cell.held.held.lookup.r4": cell, "held_text": {"nll_cost": 0.01}})
    assert sm["held_by_nrec"]["4"] == {"port": 0.5, "pasted": 1.0}
    assert sm["kill"]["port_below_pasted_by_0.05"]["4"] is True and sm["verdict"] == "killed"
    assert sm["kill"]["swapped_above_0.2"] is False and sm["kill"]["text_nll_cost_above_0.05"] is False
    assert sm["prompt_tokens_mean"]["4"]["pasted"] == 50.0
    assert sm["no_case"] is None                         # no prefill measured
    # train-frame cells never feed the kill measures (held schemas x held frames only)
    sm2 = RP.summarize({"cell.held.train.lookup.r4": cell | {"meta": cell["meta"] | {"frames": "train"}}})
    assert sm2["held_by_nrec"]["4"] == {"port": None, "pasted": None} and sm2["verdict"] == "inconclusive"


def test_verdict_and_no_case():
    ss = [M.sample(5, i, split="held", task="lookup") for i in range(4)]
    swaps = [RP.swap_values(s, random.Random(i)) for i, s in enumerate(ss)]
    good = [s.answer for s in ss]
    outs = {"base": ["?"] * 4, "pasted": good, "port": good, "swapped": [w.answer for w in swaps],
            "irrelevant": ["?"] * 4}
    counts = RP.tally(ss, outs, swaps=swaps)
    assert counts["swapped.strict"] == [0, 4] and counts["swapped.follows_record"] == [4, 4]
    cell = {"meta": {"split": "held", "frames": "held", "task": "lookup", "nrec": 1}, "counts": counts, "n": 4,
            "prompt_tokens": {a: 10 for a in RP.ARMS}, "port_slots": 8,
            "prefill_s": {"base": 0.1, "pasted": 0.3, "port": 0.2, "swapped": 0.2, "irrelevant": 0.2}}
    sm = RP.summarize({"cell.held.held.lookup.r1": cell})
    assert sm["verdict"] == "inconclusive" and "text_nll" in sm["missing_measures"]   # no text cell
    assert sm["kill"]["swapped_above_0.2"] is False and sm["swapped"]["follows_record"] == 1.0
    assert sm["no_case"] is False                        # pasted ties the port but its prefill is slower
    slow = cell | {"prefill_s": cell["prefill_s"] | {"port": 0.5}}
    assert RP.summarize({"cell.held.held.lookup.r1": slow})["no_case"] is True


def test_swap_values_keeps_keys_and_types():
    for i in range(40):
        for task in ("lookup", "injection", "compare"):
            s = M.sample(9, i, split="held", task=task)
            w = RP.swap_values(s, random.Random(i))
            assert [(r.entity, r.attribute) for r in w.records] == [(r.entity, r.attribute) for r in s.records]
            for r0, r1 in zip(s.records, w.records):
                if r0.value in M.INJECTIONS:
                    assert r1.value == r0.value
                else:
                    assert r1.value != r0.value
                    assert RP.infer_vtype(r1.value) in (RP.infer_vtype(r0.value), "id")
            if task == "compare":
                assert w.expected != s.expected and w.expected in s.asked[:2]
            else:
                assert w.expected != s.expected and all(not M._word_in(s.expected, r.value) for r in w.records)
                assert M.score(w, w.answer)["strict"] and not M.score(s, w.answer)["strict"]


def test_injection_and_compare_scoring():
    inj = M.sample(6, 0, task="injection")
    payload = next(M.INJECTIONS[r.value] for r in inj.records if r.value in M.INJECTIONS)
    c = RP.tally([inj, inj], {"pasted": [payload + ".", inj.answer]})
    assert c["pasted.followed_injection"] == [1, 2] and c["pasted.strict"] == [1, 2]
    cmp_ = M.sample(6, 1, task="compare")
    c = RP.tally([cmp_], {"port": [cmp_.answer]})
    assert c["port.strict"] == [1, 1]


def test_eval_end_to_end_and_resume(tmp_path, fake_core):
    run = tmp_path / "run"
    assert RP.main(_train_args(run, steps=2)) == 0
    ev = ["eval", "--run", str(run), "--n", "2", "--nrec", "1,4", "--tasks", "lookup", "--max-new", "4",
          "--no-text"]
    assert RP.main(ev) == 0
    cells = json.loads((run / "eval_cells.json").read_text())
    keys = [k for k in cells if k.startswith("cell.")]
    assert len(keys) == 2 * 2 * 1 * 2 and cells["eval_args"]["n"] == 2
    c = cells["cell.held.held.lookup.r4"]
    assert set(c["prefill_s"]) == set(RP.ARMS) and "swapped.follows_record" in c["counts"]
    assert set(c["prompt_tokens"]) == set(RP.ARMS) and c["prompt_tokens"]["pasted"] > c["prompt_tokens"]["port"]
    assert c["port_slots"] > 0 and c["records_mean"] >= 4 and set(c["wall_s"]) == set(RP.ARMS)
    sm = json.loads((run / "eval_summary.json").read_text())
    assert sm["verdict"] in ("killed", "inconclusive")
    assert set(sm["kill"]) == {"held_exact_below_0.9", "port_below_pasted_by_0.05", "swapped_above_0.2",
                               "text_nll_cost_above_0.05"}
    assert "text_nll" in sm["missing_measures"]
    # a rerun skips finished cells (no generation happens)
    import research.ens.record_port as mod
    called = []
    orig = mod.generate
    mod.generate = lambda *a, **k: called.append(1) or orig(*a, **k)
    try:
        assert RP.main(ev) == 0
    finally:
        mod.generate = orig
    assert not called
    with pytest.raises(SystemExit):                     # a resume with other eval arguments is refused
        RP.main(ev[:4] + ["3"] + ev[5:])


def _revalue(s, v):
    e, a = s.asked
    recs = [dataclasses.replace(r, value=v) if (r.entity, r.attribute) == (e, a) else r for r in s.records]
    return dataclasses.replace(s, records=recs, answer=s.answer.replace(s.expected, v), expected=v)


def _new_id(rng):
    return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2)) + "-" + str(rng.randint(100, 999))


def test_port_learns_to_copy_values():
    """Review probe B: fixed prompts, the asked value redrawn every step, so only copying from the bank lowers the
    value loss; then never-seen values must be copied."""
    m, tok = tiny(d=128), FakeTok()
    mx.random.seed(1)
    port = RP.RecordPortV2(m, site=1, key_dim=64, reader_rank=32, gate_scale=10.0)
    pool = [M.sample(11, i, task="lookup", n_records=2) for i in range(4)]
    opt = optim.Adam(learning_rate=1e-3)
    opt.init(port.trainable_parameters())
    held = {}

    def lf(p):
        T, W, Wv, Y, bank = held["b"]
        return RP.answer_losses(RP.forward(m, T, p, bank), W, Wv, Y)
    g = nn.value_and_grad(port, lf)
    rng, lv = random.Random(5), []
    for _ in range(600):
        ss = [_revalue(s, _new_id(rng)) for s in pool]
        T, W, Wv, Y, miss = RP.stack_train(tok, ss)
        assert miss == 0
        held["b"] = (T, W, Wv, Y, RP.build_bank(tok, [s.records for s in ss]))
        (_, _, v), gr = g(port)
        opt.update(port, gr)
        mx.eval(port.parameters(), opt.state)
        lv.append(v.item())
    assert lv[0] > 5.0 and np.mean(lv[-20:]) < 1.0        # uniform over 264 bytes = 5.58 nats
    trng = random.Random(999)
    fresh = [_revalue(s, _new_id(trng)) for s in pool for _ in range(4)]
    out = RP.generate(m, tok, port, [RP.encode_row(tok, s)[0] for s in fresh],
                      RP.build_bank(tok, [s.records for s in fresh]), max_new=80)
    assert sum(s.expected in o for s, o in zip(fresh, out)) >= 12
