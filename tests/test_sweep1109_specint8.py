"""Bug sweep 2026-11-09, lane specint8 (CPU only).

SI-1: Flash-Next's int8 prefill allow-list (8779ce64a) leaked through
inheritance into the dense Qwen adapters (scope ``mlp`` bound nothing, ``all``
skipped the dense MLP) and gave Nemotron 3 Super / Lightning a declaration that
matches none of their modules.

SI-2: the int8 feature record did not bind which projections the adapter's
selector binds; a bound-set digest now goes into the settings and route
receipt and is compared when a record carries one.
"""

import importlib
import pkgutil

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime import int8_prefill as ip

from test_int8_prefill_q8_inplace import _Model, _qlinear, fake_device  # noqa: F401
from test_int8_prefill_auto import (  # noqa: F401  (fixtures)
    AUTO_POLICY,
    _engine,
    _feature_table,
    _q8_serving_model,
    device_ok,
    serving_host,
)


# --------------------------------------------------------------------------
# SI-1: selected projection sets per FlashNextAdapter subclass
# --------------------------------------------------------------------------

H = 256


class _Bag(nn.Module):
    def __init__(self, **children):
        super().__init__()
        for name, child in children.items():
            setattr(self, name, child)


def _q(*names):
    return _Bag(**{name: _qlinear(H, H) for name in names})


def _dense_tree():
    """Served dense Qwen layout: ``language_model.model.layers.N``."""
    gdn = _Bag(linear_attn=_q("in_proj_qkv", "in_proj_z", "out_proj"),
               mlp=_q("gate_proj", "up_proj", "down_proj"))
    full = _Bag(self_attn=_q("q_proj", "k_proj", "v_proj", "o_proj"),
                mlp=_q("gate_proj", "up_proj", "down_proj"))
    inner = _Bag()
    inner.layers = [gdn, full]
    return _Bag(language_model=_Bag(model=inner))


def _flash_next_tree():
    """Flash-Next layout: the trunk MLP is a shared expert."""
    full = _Bag(self_attn=_q("q_proj", "k_proj", "v_proj", "o_proj"),
                mlp=_Bag(shared_expert=_q("gate_proj", "up_proj", "down_proj")))
    inner = _Bag()
    inner.layers = [full]
    return _Bag(language_model=_Bag(model=inner))


P = "language_model.model.layers."
DENSE_MLP = {f"{P}{n}.mlp.{leaf}" for n in (0, 1) for leaf in ("gate_proj", "up_proj", "down_proj")}
DENSE_ALL = DENSE_MLP | {f"{P}0.linear_attn.{leaf}" for leaf in ("in_proj_qkv", "in_proj_z", "out_proj")} | {
    f"{P}1.self_attn.{leaf}" for leaf in ("q_proj", "k_proj", "v_proj", "o_proj")}
FN_MLP = {f"{P}0.mlp.shared_expert.{leaf}" for leaf in ("gate_proj", "up_proj", "down_proj")}
FN_ALL = FN_MLP | {f"{P}0.self_attn.{leaf}" for leaf in ("q_proj", "k_proj", "v_proj", "o_proj")}

DENSE = {"Qwen3827BAdapter", "Qwen3627BAdapter", "Qwen359BAdapter", "Qwen354BAdapter"}
UNDECLARED = {"Qwen3635BA3BAdapter", "Qwen35122BA10BAdapter",
              "Nemotron3SuperAdapter", "Nemotron35LightningAdapter"}
OWN_SELECTOR = {"Nemotron35SuperAdapter", "Nemotron35BoltAdapter", "Nemotron35SuperVLAdapter"}


def _flash_next_subclasses():
    import mlx2.adapters as adapters
    from mlx2.adapters.flash_next import FlashNextAdapter

    for info in pkgutil.iter_modules(adapters.__path__):
        importlib.import_module(f"mlx2.adapters.{info.name}")
    found, stack = {}, [FlashNextAdapter]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub.__module__.startswith("mlx2.adapters.") and sub.__name__ not in found:
                found[sub.__name__] = sub
                stack.append(sub)
    return FlashNextAdapter, found


def _adapter(cls, model):
    adapter = object.__new__(cls)
    adapter.model = model
    return adapter


def _bound(adapter, scope):
    """The projection set the serving install actually binds, plus the
    census the auto resolver counts with the same chooser."""
    policy = {"enabled": True, "scope": scope, "q8_inplace": True, "act_scale": "group64"}
    census = ip.checkpoint_census(adapter, scope)["q8"]
    handle = ip.apply_for_adapter(adapter, policy)
    try:
        return set(handle.modules), census
    finally:
        ip.remove(handle)


def test_si1_every_flash_next_subclass_has_a_decided_int8_selection():
    base, subs = _flash_next_subclasses()
    # Walk the whole tree: a new subclass must be classified here.
    assert set(subs) == DENSE | UNDECLARED | OWN_SELECTOR
    flash_next_select = base.__dict__["int8_prefill_select"]
    for name, cls in subs.items():
        resolved = next(k.__dict__["int8_prefill_select"] for k in cls.__mro__
                        if "int8_prefill_select" in k.__dict__)
        # The allow-list is Flash-Next's alone.
        assert resolved is not flash_next_select, name


@pytest.mark.parametrize("name", sorted(DENSE))
def test_si1_dense_qwen_int8_binds_the_dense_mlp(fake_device, name):  # noqa: F811
    cls = _flash_next_subclasses()[1][name]
    adapter = _adapter(cls, _dense_tree())
    assert ip.adapter_scopes(adapter) == {"mlp", "all"}
    bound, census = _bound(adapter, "mlp")
    assert bound == DENSE_MLP and census == len(DENSE_MLP)
    bound, census = _bound(adapter, "all")
    assert bound == DENSE_ALL and census == len(DENSE_ALL)


def test_si1_flash_next_keeps_its_shared_expert_allow_list(fake_device):  # noqa: F811
    base, _ = _flash_next_subclasses()
    adapter = _adapter(base, _flash_next_tree())
    assert _bound(adapter, "mlp") == (FN_MLP, len(FN_MLP))
    assert _bound(adapter, "all") == (FN_ALL, len(FN_ALL))
    # On the dense layout the allow-list binds no MLP: why it must not leak.
    dense = _adapter(base, _dense_tree())
    with pytest.raises(ip.Int8PrefillError):
        _bound(dense, "mlp")


@pytest.mark.parametrize("name", sorted(UNDECLARED))
def test_si1_undeclared_subclasses_declare_no_int8_scope(fake_device, name):  # noqa: F811
    cls = _flash_next_subclasses()[1][name]
    adapter = _adapter(cls, _dense_tree())
    assert ip.adapter_scopes(adapter) == frozenset()
    assert ip.auto_scope(adapter) is None
    with pytest.raises(ip.Int8PrefillError, match="does not declare int8 prefill scope"):
        ip.apply_for_adapter(adapter, {"enabled": True, "scope": "all",
                                       "q8_inplace": True, "act_scale": "group64"})


# --------------------------------------------------------------------------
# SI-2: bound-set digest
# --------------------------------------------------------------------------

Q8_ALL = {"enabled": True, "scope": "all", "q8_inplace": True, "act_scale": "group64"}


class _Selecting:
    def __init__(self, model, keep):
        self.model, self.keep = model, keep

    def int8_prefill_supported(self):
        return ("mlp", "all")

    def int8_prefill_select(self, scope):
        return lambda path, module: path.rsplit(".", 1)[-1] in self.keep


def _types(model):
    return {path: type(m) for path, m in model.named_modules()}


def test_si2_bind_records_a_bound_set_digest_that_tells_selectors_apart(fake_device):  # noqa: F811
    model = _Model(mlp_bits=8, attn_bits=8)
    stock = _types(model)
    digests = []
    for keep in ({"gate_proj"}, {"gate_proj", "up_proj", "down_proj"}):
        handle, settings = ip.bind_for_serving(
            _Selecting(model, keep), Q8_ALL, max_lanes=1, config={}, speculation="ordinary")
        try:
            digest = settings["bound_set_sha256"]
            assert digest == handle.bound_set_sha256() == handle.receipt()["bound_set_sha256"]
            assert handle.status()["bound_set_sha256"] == digest
            digests.append(digest)
        finally:
            ip.remove(handle)
    # Same policy revision, different projections: the digest differs.
    assert digests[0] != digests[1]

    # A record digest that does not match refuses and leaves nothing bound.
    with pytest.raises(ip.Int8PrefillError, match="not feature-qualified"):
        ip.bind_for_serving(_Selecting(model, {"gate_proj"}), Q8_ALL, max_lanes=1, config={},
                            speculation="ordinary", expected_bound_set_sha256=digests[1])
    assert _types(model) == stock
    # The matching digest binds.
    handle, _ = ip.bind_for_serving(_Selecting(model, {"gate_proj"}), Q8_ALL, max_lanes=1,
                                    config={}, speculation="ordinary",
                                    expected_bound_set_sha256=digests[0])
    ip.remove(handle)


def test_si2_feature_table_digest_is_optional_validated_and_receipted(
    device_ok, tmp_path, monkeypatch  # noqa: F811
):
    with pytest.raises(ValueError, match="bound_set_sha256"):
        ip._feature_table(_feature_table(tmp_path, "fp-a", bound_set_sha256="not-a-digest"))
    monkeypatch.setattr(ip, "FEATURE_TABLE_PATH",
                        _feature_table(tmp_path, "fp-a", bound_set_sha256="ab" * 32))
    adapter = _Selecting(_Model(mlp_bits=8, attn_bits=8), {"gate_proj", "q_proj"})
    policy, receipt = ip.resolve_auto(
        adapter, qualification_mode=False, qualified_settings=None,
        artifact_fingerprint="fp-a", max_prefill_rows=4096)
    assert policy == AUTO_POLICY
    assert receipt["feature_record"]["bound_set_sha256"] == "ab" * 32
    # Without a digest the record still qualifies (fingerprint + revision).
    monkeypatch.setattr(ip, "FEATURE_TABLE_PATH", _feature_table(tmp_path, "fp-a"))
    policy, receipt = ip.resolve_auto(
        adapter, qualification_mode=False, qualified_settings=None,
        artifact_fingerprint="fp-a", max_prefill_rows=4096)
    assert policy == AUTO_POLICY and receipt["feature_record"]["bound_set_sha256"] is None


def test_si2_engine_auto_declines_a_feature_record_whose_bound_set_differs(
    serving_host, device_ok, tmp_path, monkeypatch  # noqa: F811
):
    # Learn the true digest from a record that does not carry one.
    monkeypatch.setattr(ip, "FEATURE_TABLE_PATH", _feature_table(tmp_path, "tiny-hybrid"))
    engine = _engine(_q8_serving_model(), int8_prefill="auto")
    try:
        assert engine.ready.wait(60), engine.error
        true_digest = engine.status()["settings"]["int8_prefill"]["bound_set_sha256"]
    finally:
        engine.close()

    monkeypatch.setattr(ip, "FEATURE_TABLE_PATH", _feature_table(
        tmp_path, "tiny-hybrid", bound_set_sha256="0" * 64))
    engine = _engine(_q8_serving_model(), int8_prefill="auto")
    try:
        assert engine.ready.wait(60), engine.error
        status = engine.status()
        assert "int8_prefill" not in status["settings"]
        auto = status["int8_prefill"]["auto"]
        assert auto["resolved"] == "off" and "not feature-qualified" in auto["reason"]
    finally:
        engine.close()

    monkeypatch.setattr(ip, "FEATURE_TABLE_PATH", _feature_table(
        tmp_path, "tiny-hybrid", bound_set_sha256=true_digest))
    engine = _engine(_q8_serving_model(), int8_prefill="auto")
    try:
        assert engine.ready.wait(60), engine.error
        assert engine.status()["int8_prefill"]["auto"]["resolved"] == "on"
        assert engine.status()["settings"]["int8_prefill"]["bound_set_sha256"] == true_digest
    finally:
        engine.close()


def test_si2_engine_explicit_int8_refuses_a_feature_record_whose_bound_set_differs(
    serving_host, device_ok, tmp_path, monkeypatch  # noqa: F811
):
    explicit = ip.Int8PrefillPolicy(enabled=True, scope="all", q8_inplace=True,
                                    act_scale="group64", q8_only=True)
    monkeypatch.setattr(ip, "FEATURE_TABLE_PATH", _feature_table(
        tmp_path, "tiny-hybrid", policy=explicit, bound_set_sha256="1" * 64))
    engine = _engine(_q8_serving_model(), int8_prefill=explicit.as_dict())
    try:
        engine.ready.wait(60)
        assert engine.error and "not feature-qualified" in str(engine.error)
    finally:
        engine.close()
