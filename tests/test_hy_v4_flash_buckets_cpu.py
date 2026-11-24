"""Hy4-flash baked aMoEba box (``runtime/models/hy_v4_flash_buckets.py``) on CPU.

The tiny checkpoint is the mlx-lm-style pack of ``test_hy_v4_flash_mlx_pack_cpu`` (layers 1-2 q2_0 expert banks,
layers 3-5 affine 4-bit, layer 0 a dense MLP), widened to F=256 so 32-neuron blocks are whole quantization groups.

* the baked model's logits equal the zero-mask model's bit for bit (CPU sequential reduction), on the remap path and
  on the sliced (prefill) path;
* the expert planes shrink, the source pack is untouched (its files hash the same), and unchanged files are links;
* the baked artifact refuses an amoeba_mask, and its receipt reports the buckets and their observed use.
"""

import hashlib
import json

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime import amoeba_neuron_mask as M  # noqa: E402
from mlx2.runtime.models import hy_v4_flash_buckets as B  # noqa: E402
from test_hy_v4_flash_cpu import checkpoint, tiny_config  # noqa: E402
from test_hy_v4_flash_mlx_pack_cpu import write_pack  # noqa: E402

PROMPT = [5, 7, 9, 11, 13, 4]
BLOCK, CLASSES = 32, [8, 6, 4]          # F = 256 = 8 blocks of 32


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    from tokenizers import Tokenizer, models

    config = tiny_config(routed_expert_hidden_size=64, moe_intermediate_size=256,
                         engram_n_embed_per_ngram=64, engram_fp8_embedding=True)
    W = checkpoint(config, seed=7, scale=0.2)
    root, _, _ = write_pack(tmp_path_factory.mktemp("pack"), config, W)
    vocab = {f"t{i}": i for i in range(config["text_config"]["vocab_size"])}
    Tokenizer(models.WordLevel(vocab, unk_token="t0")).save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "t2"}))
    return root


def _digest(root):
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def bundle(pack, tmp_path_factory):
    """A box with whole closed blocks (0-4 per expert, so every bucket is used) plus scattered single neurons, and
    ~20% of layer 0's dense neurons."""
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter

    identity, sites = HYV4FlashAdapter.amoeba_mask_sites_for(pack)
    rng = np.random.default_rng(3)
    closed = {}
    for s in sites:
        if s.experts is None:
            closed[s.layer] = rng.choice(s.units, size=s.units // 5, replace=False)
            continue
        nb = s.width // BLOCK
        ids = []
        for e in range(s.experts):
            for b in rng.choice(nb, size=e % 5, replace=False):
                ids.append(np.arange(e * s.width + b * BLOCK, e * s.width + (b + 1) * BLOCK))
        ids.append(rng.choice(s.units, size=7, replace=False))      # partial blocks: zero rows inside kept blocks
        closed[s.layer] = np.unique(np.concatenate(ids))
    out = M.write_bundle(tmp_path_factory.mktemp("mask") / "b", checkpoint=f"tiny@{identity['fingerprint']}",
                         fingerprint=identity["fingerprint"], sites=sites, closed=closed,
                         boxes_off=["lang.cmn"], source={"note": "bucket bake test box"})
    return out


@pytest.fixture(scope="module")
def baked(pack, bundle, tmp_path_factory):
    before = _digest(pack)
    out = tmp_path_factory.mktemp("baked") / "art"
    rec = B.bake(pack, out, M.NeuronMaskBundle.load(bundle), classes=CLASSES, block=BLOCK, log=lambda *_: None)
    assert _digest(pack) == before                                   # the source pack is never written
    return out, rec


def _load(root, mask=None):
    from mlx2.adapters import hy_v4_flash as A

    artifact = A.inspect_artifact(root)
    model, _, readers = A.load_model(root, artifact, engram_table_format="iq4_nl",
                                     amoeba_mask=None if mask is None else M.NeuronMaskBundle.load(mask))
    return model, readers


def _logits(model, tokens):
    return np.array(model(mx.array([tokens])).astype(mx.float32))


def _close(readers):
    for r in readers.values():
        r.close()


def test_assign_buckets_widest_first_and_keeps_open_blocks():
    closed = np.zeros((5, 8), bool)
    closed[1, [0, 3]] = True                 # 6 open -> class 6
    closed[2, :5] = True                     # 3 open -> class 4 (keeps one closed block, the first)
    closed[3, 7] = True                      # 7 open -> class 8
    lay = B.assign_buckets(closed, CLASSES)
    assert lay["widths"] == [8, 6, 4] and lay["counts"] == [3, 1, 1]
    assert list(lay["bucket"]) == [0, 1, 2, 0, 0] and list(lay["slot"]) == [0, 0, 0, 1, 2]
    assert list(lay["keep"][2]) == [0, 5, 6, 7] and list(lay["keep"][1]) == [1, 2, 4, 5, 6, 7]


def test_baked_logits_equal_the_zero_mask_bit_for_bit(pack, bundle, baked, monkeypatch):
    out, rec = baked
    spec = json.loads((out / "config.json").read_text())["mlx2_expert_buckets"]
    for lay in spec["layers"].values():        # every plane scales with the kept width (+1 zero expert per bucket)
        E, nb = sum(lay["counts"]), max(lay["widths"])
        kept = sum((n + 1) * w for n, w in zip(lay["counts"], lay["widths"]))
        assert lay["bytes_after"] * E * nb == lay["bytes_before"] * kept
    zm, rz = _load(pack, bundle)
    bm, rb = _load(out)
    plain, rp = _load(pack)
    try:
        ref = _logits(zm, PROMPT)
        got = _logits(bm, PROMPT)                                    # 18 assignments: the remap path
        assert np.array_equal(got, ref), float(np.abs(got - ref).max())
        assert not np.allclose(ref, _logits(plain, PROMPT))         # the box matters
        long = PROMPT * 2 + [3, 8]                                   # 14 tokens x 3 = 42 assignments: sorted gather
        assert np.array_equal(_logits(bm, long), _logits(zm, long))
        calls = [l.mlp.experts.describe()["calls"] for l in bm.model.layers if hasattr(l.mlp, "experts")]
        assert all(c["remap"] >= 1 and c["sliced"] == 0 for c in calls)
        monkeypatch.setattr(B, "SLICE_MIN_ASSIGNMENTS", 1)           # the prefill path
        sliced = _logits(bm, PROMPT)
        assert np.array_equal(sliced, ref), float(np.abs(sliced - ref).max())
        assert all(l.mlp.experts.describe()["calls"]["sliced"] >= 1
                   for l in bm.model.layers if hasattr(l.mlp, "experts"))
    finally:
        for r in (rz, rb, rp):
            _close(r)


def test_baked_artifact_layout_links_and_refusals(pack, bundle, baked):
    from mlx2.adapters import hy_v4_flash as A

    out, rec = baked
    cfg = json.loads((out / "config.json").read_text())
    spec = cfg["mlx2_expert_buckets"]
    assert spec["schema"] == B.SCHEMA and spec["classes"] == sorted(CLASSES) and spec["block"] == BLOCK
    assert spec["source"]["mask_digest"] == M.NeuronMaskBundle.load(bundle).digest
    assert not any(".mlp.switch_mlp." in k and ".buckets." not in k for k in cfg["quantization"])
    for name in rec["linked"]:
        if (pack / name).is_file():
            assert (out / name).stat().st_ino == (pack / name).stat().st_ino     # hard links, not copies
    a, b = A.inspect_artifact(pack), A.inspect_artifact(out)
    assert a["identity"]["fingerprint"] != b["identity"]["fingerprint"]
    with pytest.raises(ValueError, match="buckets"):
        A.load_model(out, b, engram_table_format="iq4_nl", amoeba_mask=M.NeuronMaskBundle.load(bundle))


def test_adapter_receipt_and_policy_refusals(baked, bundle):
    from mlx2.adapters import hy_v4_flash as A

    out, _ = baked
    with pytest.raises(ValueError, match="buckets"):
        A.HYV4FlashAdapter(str(out), execution_policy={"engram_table_format": "iq4_nl", "moe_fused_gate_up": True})
    adapter = A.HYV4FlashAdapter(str(out), execution_policy={"engram_table_format": "iq4_nl"})
    try:
        r = adapter.diagnostics()["expert_buckets"]
        assert r["selected"] and not r["observed_used"] and r["classes"] == sorted(CLASSES)
        assert "bucketed-zero-mask-v1" in adapter.layout
        adapter.model(mx.array([PROMPT]))
        r = adapter.diagnostics()["expert_buckets"]
        assert r["observed_used"] and r["expert_bytes_after"] > 0
    finally:
        adapter.close()


def test_bake_refuses_a_block_that_splits_a_quantization_group(pack, bundle, tmp_path):
    with pytest.raises(ValueError, match="does not split"):
        B.bake(pack, tmp_path / "x", M.NeuronMaskBundle.load(bundle), classes=[16, 12], block=16,
               log=lambda *_: None)


def test_small_path_sliced_everywhere_stays_bitwise(pack, bundle, baked, monkeypatch):
    out, _ = baked
    zm, rz = _load(pack, bundle)
    bm, rb = _load(out)
    try:
        monkeypatch.setattr(B, "SMALL_PATH", "sliced")
        for tokens in (PROMPT, PROMPT * 2 + [3, 8]):
            assert np.array_equal(_logits(bm, tokens), _logits(zm, tokens))
        calls = [l.mlp.experts.describe()["calls"] for l in bm.model.layers if hasattr(l.mlp, "experts")]
        assert all(c["sliced"] >= 2 and c["remap"] == 0 for c in calls)
    finally:
        _close(rz)
        _close(rb)


def test_small_path_fanout_equals_the_zero_mask_to_reassociation(pack, bundle, baked, monkeypatch):
    out, _ = baked
    zm, rz = _load(pack, bundle)
    bm, rb = _load(out)
    try:
        spec = json.loads((out / "config.json").read_text())["mlx2_expert_buckets"]
        rec = B.prepare(bm, spec, "fanout")
        assert rec["small_path"] == "fanout" and rec["blocks"] > 0
        assert all(l.mlp.experts.buckets == [] for l in bm.model.layers if hasattr(l.mlp, "experts"))
        monkeypatch.setattr(B, "FANOUT_MAX_ROWS", 48)                     # several chunks per call
        for tokens in (PROMPT, PROMPT * 2 + [3, 8]):
            got, ref = _logits(bm, tokens), _logits(zm, tokens)
            assert np.allclose(got, ref, atol=2e-2, rtol=1e-2), float(np.abs(got - ref).max())
            assert np.array_equal(got.argmax(-1), ref.argmax(-1))
        calls = [l.mlp.experts.describe()["calls"] for l in bm.model.layers if hasattr(l.mlp, "experts")]
        assert all(c["fanout"] >= 2 and c["remap"] == 0 and c["sliced"] == 0 for c in calls)
    finally:
        _close(rz)
        _close(rb)
