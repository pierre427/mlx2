"""Hy4-flash route-wide aMoEba neuron mask (``amoeba_mask``) on CPU.

The tiny checkpoint is the mlx-lm-style pack of ``test_hy_v4_flash_mlx_pack_cpu`` (layers 1-2 q2_0 expert banks,
layers 3-5 affine 4-bit, layer 0 a dense MLP), so every storage law the real artifact has is covered.

* the mask zeroes exactly the listed neurons: the masked model's logits equal, bit for bit, a reference that runs
  the unmasked weights and multiplies each listed neuron's activation by 0 (every other neuron untouched), and
  the stored gate/up planes differ from the base only on the listed rows;
* with no ``amoeba_mask`` the model is the base model, bit for bit, and the receipt says not selected;
* a bundle made for another fingerprint, or another geometry, refuses before any weight is read;
* receipts: selected, observed_used only after a forward, the bundle digest in the APCv2 namespace.
"""

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

mx.set_default_device(mx.cpu)

from mlx2.runtime import amoeba_neuron_mask as M  # noqa: E402
from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from test_hy_v4_flash_cpu import checkpoint, tiny_config  # noqa: E402
from test_hy_v4_flash_mlx_pack_cpu import write_pack  # noqa: E402

PROMPT = [5, 7, 9, 11, 13, 4]      # 6 tokens x top-3 = 18 assignments: the unsorted switch path


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    from tokenizers import Tokenizer, models

    config = tiny_config(routed_expert_hidden_size=64, moe_intermediate_size=64,
                         engram_n_embed_per_ngram=64, engram_fp8_embedding=True)
    W = checkpoint(config, seed=7, scale=0.2)
    root, pack_cfg, _ = write_pack(tmp_path_factory.mktemp("pack"), config, W)
    vocab = {f"t{i}": i for i in range(config["text_config"]["vocab_size"])}
    Tokenizer(models.WordLevel(vocab, unk_token="t0")).save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "t2"}))
    return root


def _sites(root):
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter

    return HYV4FlashAdapter.amoeba_mask_sites_for(root)


def _closed(sites, seed=0, frac=0.3):
    """A random box: ~frac of the units of layer 0 (dense) and of every expert layer."""
    rng = np.random.default_rng(seed)
    return {s.layer: rng.choice(s.units, size=max(1, int(frac * s.units)), replace=False) for s in sites}


@pytest.fixture(scope="module")
def bundle(pack, tmp_path_factory):
    identity, sites = _sites(pack)
    out = M.write_bundle(tmp_path_factory.mktemp("mask") / "b", checkpoint=f"tiny@{identity['fingerprint']}",
                         fingerprint=identity["fingerprint"], sites=sites, closed=_closed(sites),
                         boxes_off=["lang.cmn", "lang.jpn", "lang.hin"], source={"note": "random test box"})
    return out, sites


def _load(root, mask=None):
    from mlx2.adapters import hy_v4_flash as A

    artifact = A.inspect_artifact(root)
    bundle = None if mask is None else M.NeuronMaskBundle.load(mask)
    model, _, readers = A.load_model(root, artifact, engram_table_format="iq4_nl", amoeba_mask=bundle)
    return model, readers


def _close(readers):
    for r in readers.values():
        r.close()


class _MaskedSwitch(H.SwitchGLU):
    """Reference: the unmasked bank, the listed neurons' activation multiplied by 0 (unsorted path)."""

    def __call__(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))
        h = self.activation(self.up_proj(x, indices), self.gate_proj(x, indices))
        h = h * self._keep[indices][..., None, :].astype(h.dtype)
        return self.down_proj(h, indices).squeeze(-2)


class _MaskedMLP(H.MLP):
    def __call__(self, x):
        h = nn.silu(self.gate_proj(x)) * self.up_proj(x)
        y = self.down_proj(h * self._keep.astype(h.dtype))
        return self.post_fc2_norm(y)


def _reference(model, sites, closed):
    for s in sites:
        keep = np.ones(s.units, bool)
        keep[closed[s.layer]] = False
        layer = model.model.layers[s.layer]
        if s.experts is None:
            mod, cls, shape = layer.mlp, _MaskedMLP, (s.width,)
        else:
            mod, cls, shape = layer.mlp.experts, _MaskedSwitch, (s.experts, s.width)
        object.__setattr__(mod, "_keep", mx.array(keep.reshape(shape)))
        mod.__class__ = cls
    return model


def _logits(model, tokens):
    return np.array(model(mx.array([tokens])).astype(mx.float32))


def test_mask_zeroes_exactly_the_listed_neurons(pack, bundle):
    path, sites = bundle
    closed = M.NeuronMaskBundle.load(path).closed
    masked, ra = _load(pack, path)
    base, rb = _load(pack)
    try:
        out = _logits(masked, PROMPT)
        ref = _logits(_reference(base, sites, closed), PROMPT)
        assert np.array_equal(out, ref), float(np.abs(out - ref).max())
        plain, rc = _load(pack)
        try:
            assert not np.allclose(out, _logits(plain, PROMPT))      # the box really changed the model
            # stored planes: only the listed rows differ from the base, and they are exact zeros
            pm, pp = dict(tree_flatten(masked.parameters())), dict(tree_flatten(plain.parameters()))
            for s in sites:
                keep = np.ones(s.units, bool)
                keep[closed[s.layer]] = False
                for proj in ("gate_proj", "up_proj"):
                    quantized = f"{s.stem}.{proj}.scales" in pp
                    for part in ("weight", "scales", "biases"):
                        k = f"{s.stem}.{proj}.{part}"
                        if k not in pp:
                            continue
                        a = np.array(pm[k].astype(mx.float32) if pm[k].dtype != mx.uint32 else pm[k])
                        b = np.array(pp[k].astype(mx.float32) if pp[k].dtype != mx.uint32 else pp[k])
                        rows = keep.reshape(a.shape[:-1])
                        assert np.array_equal(a[rows], b[rows]), k
                        if part == "weight" and quantized:
                            assert np.array_equal(a[~rows], b[~rows]), k      # codes untouched
                        else:
                            assert not a[~rows].any(), k                        # exact zero rows
                # down projections and shared experts are never touched
                for k in pp:
                    if k.startswith(f"model.layers.{s.layer}.") and ("down_proj" in k or "shared_experts" in k):
                        assert np.array_equal(np.array(pm[k]), np.array(pp[k])), k
        finally:
            _close(rc)
    finally:
        _close(ra)
        _close(rb)


def test_no_mask_is_the_base_model(pack, bundle, tmp_path):
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter

    base, ra = _load(pack)
    try:
        ref = _logits(base, PROMPT)
    finally:
        _close(ra)
    adapter = HYV4FlashAdapter(str(pack), execution_policy={"engram_table_format": "iq4_nl"})
    try:
        assert np.array_equal(_logits(adapter.model, PROMPT), ref)
        receipt = adapter.diagnostics()["amoeba_mask"]
        assert receipt == {"implemented": True, "qualified": False, "selected": False, "observed_used": False}
        assert "amoeba-mask" not in adapter.layout
    finally:
        adapter.close()
    # an empty box (nothing closed) is the base model too, in its own namespace
    identity, sites = _sites(pack)
    empty = M.write_bundle(tmp_path / "empty", checkpoint="tiny", fingerprint=identity["fingerprint"],
                           sites=sites, closed={1: []}, boxes_off=["lang.cmn"], source={})
    model, rb = _load(pack, empty)
    try:
        assert np.array_equal(_logits(model, PROMPT), ref)
    finally:
        _close(rb)


def test_refuses_another_checkpoint_or_geometry(pack, bundle, tmp_path):
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter

    identity, sites = _sites(pack)
    other = M.write_bundle(tmp_path / "other", checkpoint="other", fingerprint="0" * 16, sites=sites,
                           closed={1: [0, 1]}, boxes_off=["lang.cmn"], source={})
    with pytest.raises(ValueError, match="fingerprint"):
        HYV4FlashAdapter(str(pack), execution_policy={"engram_table_format": "iq4_nl", "amoeba_mask": str(other)})
    # same fingerprint, wrong expert width
    wrong = [M.MaskSite(s.layer, s.stem, s.experts, s.width * 2) if s.experts else s for s in sites]
    bad = M.write_bundle(tmp_path / "geom", checkpoint="tiny", fingerprint=identity["fingerprint"], sites=wrong,
                         closed={1: [0]}, boxes_off=["lang.cmn"], source={})
    with pytest.raises(ValueError, match="width"):
        HYV4FlashAdapter(str(pack), execution_policy={"engram_table_format": "iq4_nl", "amoeba_mask": str(bad)})
    # a tampered mask file
    path, _ = bundle
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    (tampered / "manifest.json").write_text((path / "manifest.json").read_text())
    np.savez_compressed(tampered / M.MASK_FILE, layer_1=np.array([0]))
    with pytest.raises(ValueError, match="digest"):
        M.NeuronMaskBundle.load(tampered)
    with pytest.raises(ValueError, match="amoeba_mask"):
        HYV4FlashAdapter(str(pack), execution_policy={"amoeba_mask": 3})


def test_receipts_and_namespace(pack, bundle):
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter

    path, sites = bundle
    # Band attention off: the mask tag ends the layout (the band default appends its own).
    adapter = HYV4FlashAdapter(str(pack), execution_policy={"engram_table_format": "iq4_nl",
                                                            "amoeba_mask": str(path),
                                                            "sliding_attention_blocks": 0})
    try:
        r = adapter.diagnostics()["amoeba_mask"]
        assert r["selected"] and not r["qualified"] and not r["observed_used"]
        assert r["boxes_off"] == ["lang.cmn", "lang.jpn", "lang.hin"]
        assert r["closed_units"] == sum(len(v) for v in _closed(sites).values())
        assert 0 < r["closed_fraction"] < 1 and r["model_units"] == sum(s.units for s in sites)
        tag = M.NeuronMaskBundle.load(path).numerics_tag()
        assert r["apcv2_namespace"] == tag and adapter.layout.endswith("+" + tag + "+sparse-prefill-v1+mla-fused-in-v1")
        assert adapter.descriptor.cache_layout == adapter.layout
        adapter.model(mx.array([PROMPT]))
        r = adapter.diagnostics()["amoeba_mask"]
        assert r["observed_used"] and r["trunk_forwards"] == 1
    finally:
        adapter.close()


def test_export_cli_from_a_carve_mask(pack, tmp_path):
    identity, sites = _sites(pack)
    width = max(s.units for s in sites)
    box = np.zeros((len(sites), width), np.uint8)
    box[0, :5] = 1
    box[3, [0, 17, 63, 64 * 7 + 1]] = 1
    np.savez_compressed(tmp_path / "partition.npz", box_lang_cjk__mask=box)
    out = tmp_path / "bundle"
    M.main(["export", "--model", str(pack), "--mask", f"{tmp_path / 'partition.npz'}:box_lang_cjk__mask",
            "--box", "lang.cmn", "--box", "lang.jpn", "--box", "lang.hin", "--out", str(out)])
    b = M.NeuronMaskBundle.load(out)
    assert b.manifest["source_fingerprint"] == identity["fingerprint"]
    assert {k: v.tolist() for k, v in b.closed.items()} == {0: [0, 1, 2, 3, 4], 3: [0, 17, 63, 449]}
    with pytest.raises(SystemExit, match="not empty"):
        M.main(["export", "--model", str(pack), "--mask", f"{tmp_path / 'partition.npz'}:box_lang_cjk__mask",
                "--box", "lang.cmn", "--out", str(out)])


# ------------------------------------------------------------------ page-out (the box not resident)

BLOCK = 32          # the pack's down groups are 32 wide (affine 4-bit g32 and q2_0 g32); F = 64 -> 2 blocks per expert


def _block_box(pack, tmp_path, seed=1, frac=0.4):
    """Whole 32-neuron blocks of random experts (paged) plus scattered neurons (zeroed, resident), dense layer too."""
    identity, sites = _sites(pack)
    rng = np.random.default_rng(seed)
    closed = {}
    for s in sites:
        if s.experts is None:
            closed[s.layer] = rng.choice(s.units, 6, replace=False)
            continue
        nb = s.experts * (s.width // BLOCK)
        blocks = rng.choice(nb, size=int(frac * nb), replace=False)
        closed[s.layer] = np.concatenate([np.arange(b * BLOCK, (b + 1) * BLOCK) for b in blocks]
                                         + [rng.choice(s.units, 7, replace=False)])
    return M.write_bundle(tmp_path / "box", checkpoint="tiny", fingerprint=identity["fingerprint"], sites=sites,
                          closed=closed, boxes_off=["lang.cmn", "lang.jpn", "lang.hin"], source={}), closed


def _adapter(pack, spec):
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter

    return HYV4FlashAdapter(str(pack), execution_policy={"engram_table_format": "iq4_nl", "amoeba_mask": spec})


def _greedy(model, prompt, n):
    cache = model.make_cache()
    out = [int(mx.argmax(model(mx.array([prompt]), cache)[0, -1]).item())]
    for _ in range(n - 1):
        out.append(int(mx.argmax(model(mx.array([[out[-1]]]), cache)[0, -1]).item()))
    return out


def test_pageout_equals_the_zero_mask_and_frees_the_paged_blocks(pack, tmp_path):
    from mlx.utils import tree_flatten

    from mlx2.runtime.amoeba_pageout import PagedSwitchGLU, blocks

    path, closed = _block_box(pack, tmp_path)
    zero = _adapter(pack, str(path))
    paged = _adapter(pack, {"bundle": str(path), "mode": "pageout", "block": BLOCK})
    try:
        a, b = _logits(zero.model, PROMPT), _logits(paged.model, PROMPT)
        assert np.allclose(a, b, rtol=1e-5, atol=1e-5), float(np.abs(a - b).max())   # fp32 reassociation only
        assert np.array_equal(a.argmax(-1), b.argmax(-1))
        assert _greedy(zero.model, PROMPT, 8) == _greedy(paged.model, PROMPT, 8)
        r = paged.diagnostics()["amoeba_mask"]
        po = r["pageout"]
        assert r["mode"] == "pageout" and po["law"] == "block-pageout-sequential-fp32-v2"
        assert "pageout-b32-sequential-v2" in paged.layout and paged.layout != zero.layout
        assert po["paged_bank_passes"] >= po["paged_bank_calls"] > 0
        pz = dict(tree_flatten(zero.model.parameters()))
        pp = dict(tree_flatten(paged.model.parameters()))
        freed = sum(v.nbytes for v in pz.values()) - sum(v.nbytes for v in pp.values())
        tables = sum(l.mlp.experts._table.nbytes for l in paged.model.model.layers
                     if isinstance(getattr(l.mlp, "experts", None), PagedSwitchGLU))
        assert po["resident_bytes_freed"] == freed - tables > 0
        _, sites = _sites(pack)
        for s in sites:
            if s.experts is None:
                continue
            paged_b, res, table, kmax = blocks(closed[s.layer], s.experts, s.width, BLOCK)
            rec = next(x for x in po["layers"] if x["layer"] == s.layer)
            assert rec["paged_blocks"] == int(paged_b.sum()) and rec["resident_blocks"] == len(res)
            bank = paged.model.model.layers[s.layer].mlp.experts
            src = zero.model.model.layers[s.layer].mlp.experts
            for proj in ("gate_proj", "up_proj"):                # resident rows: the same bytes as the zero mask's
                for part in ("weight", "scales", "biases"):
                    if part not in src[proj]:
                        continue
                    A_ = np.array(src[proj][part].astype(mx.float32) if src[proj][part].dtype != mx.uint32
                                  else src[proj][part])
                    B_ = np.array(bank[proj][part].astype(mx.float32) if bank[proj][part].dtype != mx.uint32
                                  else bank[proj][part])
                    want = A_.reshape(s.experts, s.width // BLOCK, BLOCK, *A_.shape[2:])[
                        [e for e, _ in res], [b_ for _, b_ in res]]
                    assert np.array_equal(B_[:-1], want) and not B_[-1].any()
        assert po["paged_bank_calls"] > 0 and r["observed_used"]
    finally:
        zero.close()
        paged.close()


def test_pageout_refuses_fused_banks_and_bad_blocks(pack, tmp_path):
    path, _ = _block_box(pack, tmp_path)
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter
    with pytest.raises(ValueError, match="moe_fused_gate_up"):
        HYV4FlashAdapter(str(pack), execution_policy={"engram_table_format": "iq4_nl", "moe_fused_gate_up": True,
                                                      "amoeba_mask": {"bundle": str(path), "mode": "pageout"}})
    with pytest.raises(ValueError, match="does not split|does not divide"):
        _adapter(pack, {"bundle": str(path), "mode": "pageout", "block": 24})
    with pytest.raises(ValueError, match="mode"):
        _adapter(pack, {"bundle": str(path), "mode": "drop"})
    with pytest.raises(ValueError, match="block applies"):
        _adapter(pack, {"bundle": str(path), "mode": "zero", "block": 32})
