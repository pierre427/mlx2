"""CPU tests for knapsack per-layer mixed precision (mlx2.conversion.knapsack)."""
import itertools
import json
import random

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlx2.conversion import knapsack as K
from mlx2.conversion import knapsack_export as E

mx.set_default_device(mx.cpu)


# ------------------------------------------------------------------ solver --

def _random_case(rng, layers, options, *, forbid=0.15):
    costs, sizes = [], []
    for _ in range(layers):
        cr, sr = [], []
        for _ in range(options):
            if rng.random() < forbid:
                cr.append(None)
                sr.append(None)
            else:
                cr.append(round(rng.uniform(-0.2, 5.0), 3))
                sr.append(rng.randint(1, 40))
        costs.append(cr)
        sizes.append(sr)
    return costs, sizes


def test_solver_matches_brute_force_on_small_cases():
    rng = random.Random(1234)
    checked = 0
    for _ in range(300):
        layers, options = rng.randint(1, 5), rng.randint(1, 4)
        costs, sizes = _random_case(rng, layers, options)
        budget = rng.randint(0, 40 * layers)
        ref = K.brute_force(costs, sizes, budget)
        if ref is None:
            with pytest.raises(ValueError):
                K.solve(costs, sizes, budget, resolution=1)
            continue
        got = K.solve(costs, sizes, budget, resolution=1)
        assert got.exact and got.nbytes <= budget
        assert got.cost == pytest.approx(ref.cost, abs=1e-9)
        # same cost, and never more bytes than the brute-force tie-break
        assert got.nbytes <= ref.nbytes or got.cost < ref.cost - 1e-9
        assert all(costs[i][j] is not None for i, j in enumerate(got.choice))
        checked += 1
    assert checked > 150


def test_coarse_resolution_never_exceeds_budget_and_is_near_optimal():
    rng = random.Random(7)
    checked = 0
    for _ in range(100):
        costs, sizes = _random_case(rng, 4, 3, forbid=0.0)
        sizes = [[s * 1000 + rng.randint(0, 999) for s in row] for row in sizes]
        budget = rng.randint(4000, 160_000)
        ref = K.brute_force(costs, sizes, budget)
        try:
            got = K.solve(costs, sizes, budget, resolution=97)
        except ValueError:
            continue  # rounding sizes up may lose a plan that only just fits
        assert got.nbytes <= budget and not got.exact
        assert ref is not None and got.cost >= ref.cost - 1e-9
        checked += 1
    assert checked > 50


def test_default_resolution_is_bounded():
    costs = [[1.0, 0.5, 0.1]] * 96
    sizes = [[10**9 + 1, 2 * 10**9 + 3, 3 * 10**9 + 7]] * 96   # coprime: no exact GCD bucket
    sol = K.solve(costs, sizes, 200 * 10**9)
    assert sol.nbytes <= 200 * 10**9 and sol.resolution > 1 and not sol.exact


def test_gcd_bucket_is_exact_even_at_a_tight_budget():
    rng = random.Random(11)
    for _ in range(200):
        costs, sizes = _random_case(rng, 5, 3, forbid=0.1)
        sizes = [[None if x is None else x * 4096 for x in row] for row in sizes]
        budget = rng.randint(0, 40 * 5) * 4096
        ref = K.brute_force(costs, sizes, budget)
        if ref is None:
            continue
        got = K.solve(costs, sizes, budget, max_cells=1000)   # default resolution
        assert got.exact and got.cost == pytest.approx(ref.cost, abs=1e-9)
    # uniform at exactly its own bytes is feasible
    sizes = [[7 * 4096]] * 6
    assert K.solve([[0.0]] * 6, sizes, 42 * 4096).nbytes == 42 * 4096


def test_infeasible_budget_raises():
    with pytest.raises(ValueError, match="no plan fits"):
        K.solve([[1.0, 2.0]], [[10, 20]], 9, resolution=1)


# ----------------------------------------------------------------- formats --

@pytest.mark.parametrize("name", ["q2g64", "q3g64", "q4g32", "q5g64", "q6g128", "q8g64",
                                  "mxfp4", "nvfp4", "mxfp8"])
def test_format_bytes_match_mlx_storage(name):
    fmt = K.parse_format(name)
    w = mx.random.normal((3, 24, 256)).astype(mx.bfloat16)
    planes = mx.quantize(w, group_size=fmt.group_size, bits=fmt.bits, mode=fmt.mode)
    stored = sum(int(p.nbytes) for p in planes if p is not None)
    if fmt.mode == "affine":
        assert planes[1].dtype == mx.bfloat16
    assert fmt.nbytes(w.shape) == stored
    assert fmt.bpw == pytest.approx(8 * stored / w.size)


def test_parse_format_and_pins():
    assert K.parse_format("affine-q3-g64").name == "q3g64"
    assert K.parse_format("q4").name == "q4g64"
    with pytest.raises(ValueError):
        K.parse_format("q7g64")
    pins = K.parse_pins(["0:>=3", "1-2=q4g64", "0:>=2"])
    assert pins == {0: {"min_bits": 3.0}, 1: {"format": "q4g64"}, 2: {"format": "q4g64"}}
    with pytest.raises(ValueError):
        K.parse_pins(["zero:>=3"])


# ------------------------------------------------------------- tiny model --

class _Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, d, bias=False)
        self.router = nn.Linear(d, 4, bias=False)
        self.mlp = nn.Linear(d, 2 * d, bias=False)
        self.out = nn.Linear(2 * d, d, bias=False)
        self.norm = nn.RMSNorm(d)


class _Tiny(nn.Module):
    def __init__(self, d=128, layers=4, vocab=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = [_Layer(d) for _ in range(layers)]
        self.head = nn.Linear(d, vocab, bias=False)

    @property
    def quant_predicate(self):
        def predicate(path, _module):
            if path.endswith("router"):
                return {"group_size": 64, "bits": 8}
            return True
        return predicate

    def __call__(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = h + layer.attn(layer.norm(h))
            h = h + layer.out(nn.silu(layer.mlp(h)))
        return self.head(h)


def _tiny():
    mx.random.seed(0)
    model = _Tiny()
    mx.eval(model.parameters())
    return model


def test_inventory_roles_and_sizes():
    units = K.inventory(_tiny(), layer_prefix="layers.")
    roles = {u.path: u.role for u in units}
    assert roles["layers.0.attn"] == "planned" and roles["layers.3.out"] == "planned"
    assert roles["layers.1.router"] == "fixed" and roles["head"] == "rest" and roles["embed"] == "rest"
    assert "layers.0.norm" not in roles
    menu = K.parse_menu(["q2g64,q4g64"])
    sizes = K.layer_sizes(units, menu)
    d = 128
    assert sizes[0]["q4g64"] == K.parse_format("q4g64").nbytes((d, d)) + 2 * K.parse_format("q4g64").nbytes((2 * d, d))


def _plan(model, menu, costs, budget, **kw):
    units = K.inventory(model, layer_prefix="layers.")
    return units, K.make_plan(checkpoint="tiny@sha256:00", units=units, menu=K.parse_menu(menu),
                              costs=costs, budget=budget, scorer={"name": "test"},
                              mlx2={"commit": "test", "tracked_dirty": False},
                              rest_format=K.parse_format("q8g64"), layer_prefix="layers.", **kw)


def test_plan_respects_budget_pins_and_round_trips(tmp_path):
    model = _tiny()
    menu = ["q2g64", "q3g64", "q4g64"]
    # layer 0 is cheap to hurt by "MSE", so a naive plan would starve it.
    costs = {0: {"q2g64": 0.01, "q3g64": 0.005, "q4g64": 0.0},
             1: {"q2g64": 1.0, "q3g64": 0.3, "q4g64": 0.1},
             2: {"q2g64": 0.8, "q3g64": 0.2, "q4g64": 0.1},
             3: {"q2g64": 2.0, "q3g64": 0.5, "q4g64": 0.05}}
    units, free = _plan(model, menu, costs, "3.5bpw", resolution=1)
    assert free["layers"][0]["format"] == "q2g64"
    assert free["solution"]["planned_bytes"] <= free["budget"]["bytes"]
    assert free["solution"]["planned_bpw"] <= 3.5
    # brute force over the same table agrees
    sizes = K.layer_sizes(units, K.parse_menu(menu))
    best = min((sum(costs[l][f] for l, f in enumerate(c)), c)
               for c in itertools.product(menu, repeat=4)
               if sum(sizes[l][f] for l, f in enumerate(c)) <= free["budget"]["bytes"])
    assert free["solution"]["predicted_cost"] == pytest.approx(best[0])

    _, pinned = _plan(model, menu, costs, "3.5bpw", pins=K.parse_pins(["0:>=3"]), resolution=1)
    assert pinned["layers"][0]["format"] in ("q3g64", "q4g64")
    assert pinned["layers"][0]["allowed"] == ["q3g64", "q4g64"]
    _, floor = _plan(model, menu, costs, "4.5bpw", min_bits=3, resolution=1)
    assert all(r["format"] != "q2g64" for r in floor["layers"])
    with pytest.raises(ValueError, match="no plan fits"):
        _plan(model, menu, costs, "2.6bpw", min_bits=3)

    path = tmp_path / "plan.json"
    K.save_plan(pinned, path)
    assert K.load_plan(path) == pinned
    with pytest.raises(FileExistsError):
        K.save_plan(pinned, path)
    tampered = json.loads(path.read_text())
    tampered["layers"][0]["format"] = "q2g64"
    (tmp_path / "bad.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="digest"):
        K.load_plan(tmp_path / "bad.json")
    assert set(pinned["uniform_baselines"]) == set(menu)
    assert not pinned["uniform_baselines"]["q2g64"]["satisfies_constraints"]
    assert pinned["uniform_baselines"]["q3g64"]["satisfies_constraints"]
    assert pinned["checkpoint"] == "tiny@sha256:00" and pinned["mlx2"]["commit"] == "test"


def test_plan_refuses_mismatched_inputs():
    model = _tiny()
    good = {l: {"q2g64": 1.0, "q4g64": 0.0} for l in range(4)}
    with pytest.raises(ValueError, match="missing layers"):
        _plan(model, ["q2g64", "q4g64"], {l: good[l] for l in range(3)}, "4.5bpw")
    with pytest.raises(ValueError, match="no q4g64 cost"):
        _plan(model, ["q2g64", "q4g64"], {**good, 1: {"q2g64": 1.0}}, "4.5bpw")
    with pytest.raises(ValueError, match="not in the menu"):
        _plan(model, ["q2g64", "q4g64"], good, "4.5bpw", pins=K.parse_pins(["0=q3g64"]))
    with pytest.raises(ValueError, match="does not have"):
        _plan(model, ["q2g64", "q4g64"], good, "4.5bpw", pins=K.parse_pins(["9:>=4"]))


def test_predicate_and_config_quantize_the_model_as_planned():
    model = _tiny()
    costs = {l: {"q2g64": float(4 - l), "q4g64": 0.0, "nvfp4": 0.05} for l in range(4)}
    units, plan = _plan(model, ["q2g64", "q4g64", "nvfp4"], costs, "3.5bpw", resolution=1)
    quant = K.quantization_config(plan, units)
    assert (quant["bits"], quant["group_size"]) == (8, 64)  # rest format is the default
    nn.quantize(model, class_predicate=K.plan_predicate(plan, units))
    modules = dict(model.named_modules())
    formats = K.layer_formats(plan)
    for u in units:
        m = modules[u.path]
        want = quant[u.path]
        assert (m.bits, m.group_size, m.mode) == (want["bits"], want["group_size"], want["mode"])
        if u.role == "planned":
            assert want == formats[u.layer].quant()
    assert modules["layers.2.router"].bits == 8 and modules["head"].bits == 8
    assert isinstance(modules["layers.0.norm"], nn.RMSNorm)


def test_weight_mse_costs_are_monotone_in_bits():
    model = _tiny()
    units = K.inventory(model, layer_prefix="layers.")
    menu = K.parse_menu(["q2g64,q3g64,q4g64,q8g64"])
    costs = K.weight_mse_costs(model, units, menu)
    rel = K.weight_mse_costs(model, units, menu, relative=True)
    for layer in range(4):
        row = [costs[layer][f.name] for f in menu]
        assert row == sorted(row, reverse=True) and row[-1] > 0
        assert rel[layer]["q2g64"] < costs[layer]["q2g64"]


def _scan(bits, rows, *, ckpt="m@sha256:aa", exact=True, cumulative=False):
    from mlx2.layer_kl import SCAN_SCHEMA

    return {"schema": SCAN_SCHEMA, "reference": {"checkpoint": ckpt, "texts_sha256": "t"},
            "settings": {"bits": bits, "group_size": 64, "mode": "affine", "cumulative": cumulative,
                         "groups": ["x"]},
            "restore_check": {"exact": exact},
            "rows": [{"layer": l, "delta_kl": v} for l, v in enumerate(rows)]}


def test_layer_kl_costs_validate_scans():
    costs = K.layer_kl_costs({"q2g64": _scan(2, [0.5, 0.2]), "q4g64": _scan(4, [0.01, 0.0])},
                             checkpoint="m@sha256:aa")
    assert costs == {0: {"q2g64": 0.5, "q4g64": 0.01}, 1: {"q2g64": 0.2, "q4g64": 0.0}}
    with pytest.raises(ValueError, match="do not match"):
        K.layer_kl_costs({"q3g64": _scan(2, [0.5])})
    with pytest.raises(ValueError, match="different references"):
        K.layer_kl_costs({"q2g64": _scan(2, [0.5]), "q4g64": _scan(4, [0.1], ckpt="other")})
    with pytest.raises(ValueError, match="cumulative"):
        K.layer_kl_costs({"q2g64": _scan(2, [0.5], cumulative=True)})
    with pytest.raises(ValueError, match="restore"):
        K.layer_kl_costs({"q2g64": _scan(2, [0.5], exact=False)})
    with pytest.raises(ValueError, match="not the planned"):
        K.layer_kl_costs({"q2g64": _scan(2, [0.5])}, checkpoint="x@sha256:bb")


def test_budget_specs():
    model = _tiny()
    units = K.inventory(model, layer_prefix="layers.")
    sizes = K.layer_sizes(units, K.parse_menu(["q4g64,nvfp4"]))
    nv = sum(r["nvfp4"] for r in sizes.values())
    assert K.budget_bytes("0.5xnvfp4", sizes, units) == nv // 2
    planned = sum(u.numel for u in units if u.role == "planned")
    assert K.budget_bytes("3bpw", sizes, units) == 3 * planned // 8
    assert K.budget_bytes("12345", sizes, units) == 12345
    with pytest.raises(ValueError):
        K.budget_bytes("lots", sizes, units)


# ------------------------------------------------- export on a tiny Mellum --

def _tiny_mellum(path):
    config = {"model_type": "mellum", "hidden_size": 128, "num_hidden_layers": 4, "intermediate_size": 256,
              "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "num_experts": 4,
              "num_experts_per_tok": 2, "moe_intermediate_size": 128, "vocab_size": 128,
              "max_position_embeddings": 4096, "sliding_window": 64, "tie_word_embeddings": False,
              "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 10000.0},
                                  "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}}}
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    mx.random.seed(3)
    model, _ = E.build_model(path)
    model.set_dtype(mx.bfloat16)  # real checkpoints are bf16; scales then take bf16 too
    mx.eval(model.parameters())
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(model.parameters()))
    # official per-expert layout, so the export goes through the model's own sanitize
    weights = {}
    for key, value in flat.items():
        if ".switch_mlp." in key:
            prefix, rest = key.split(".switch_mlp.")
            for e in range(config["num_experts"]):
                weights[f"{prefix}.experts.{e}.{rest}"] = value[e]
        else:
            weights[key] = value
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    return config


def test_export_round_trip_loads_and_matches(tmp_path):
    src = tmp_path / "dense"
    _tiny_mellum(src)
    model, _ = E.load_dense(src)
    units = K.inventory(model)
    assert {u.role for u in units} == {"planned", "fixed", "rest"}
    menu = K.parse_menu(["q2g64,q3g64,q4g64,mxfp4"])
    costs = K.weight_mse_costs(model, units, menu, relative=True)
    plan = K.make_plan(checkpoint="tiny-mellum@sha256:00", units=units, menu=menu, costs=costs,
                       budget="3.4bpw", scorer={"name": "weight_mse_relative"},
                       mlx2={"commit": "test", "tracked_dirty": False}, rest_format=K.parse_format("q8g64"),
                       pins=K.parse_pins(["0:>=3"]), resolution=1)
    assert len(plan["solution"]["format_counts"]) >= 1 and plan["layers"][0]["format"] != "q2g64"
    out = tmp_path / "planned"
    receipt = E.export(src, out, plan, log=lambda _m: None)
    assert receipt["plan_sha256"] == plan["plan_sha256"]
    config = json.loads((out / "config.json").read_text())
    assert config["knapsack_plan"]["plan_sha256"] == plan["plan_sha256"]
    quant = config["quantization"]
    # load the way mlx2 adapters do: per-module override, else scales present
    reloaded, _ = E.build_model(src)
    weights = {}
    for shard in sorted(out.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    weights = reloaded.sanitize(weights)

    def predicate(name, module):
        override = quant.get(name)
        if isinstance(override, dict):
            return override
        return f"{name}.scales" in weights

    nn.quantize(reloaded, group_size=quant["group_size"], bits=quant["bits"], mode=quant["mode"],
                class_predicate=predicate)
    reloaded.load_weights(list(weights.items()), strict=True)
    # bytes on disk for the planned modules are what the plan said
    planned = sum(int(weights[f"{u.path}.{k}"].nbytes) for u in units if u.role == "planned"
                  for k in ("weight", "scales", "biases") if f"{u.path}.{k}" in weights)
    assert planned == plan["solution"]["planned_bytes"]
    # and the reloaded model computes exactly what the in-memory quantized model computes
    nn.quantize(model, class_predicate=K.plan_predicate(plan, units))
    ids = mx.array([[1, 5, 9, 17, 33, 2, 77, 100]])
    assert mx.array_equal(model(ids), reloaded(ids)).item()
    with pytest.raises(FileExistsError):
        E.export(src, out, plan)


# ---------------------------------------------------- review follow-ups --

def test_pin_errors_are_specific():
    with pytest.raises(ValueError, match="before its start"):
        K.parse_pins(["5-2:>=4"])
    model = _tiny()
    costs = {l: {"q2g64": 1.0, "q4g64": 0.0} for l in range(4)}
    with pytest.raises(ValueError, match="layer 0: no menu format"):
        _plan(model, ["q2g64", "q4g64"], costs, "4.5bpw", pins=K.parse_pins(["0=q2g64", "0:>=3"]))


def test_default_without_rest_format_is_the_widest_override():
    model = _tiny()
    units = K.inventory(model, layer_prefix="layers.")
    plan = K.make_plan(checkpoint="tiny@sha256:00", units=units, menu=K.parse_menu(["q3g64"]),
                       costs={l: {"q3g64": 0.0} for l in range(4)}, budget=str(1 << 40),
                       scorer={"name": "uniform"}, mlx2={"commit": "t", "tracked_dirty": False},
                       rest_format=None, layer_prefix="layers.")
    quant = K.quantization_config(plan, units)
    assert (quant["bits"], quant["group_size"]) == (8, 64)  # the router's fixed format, not q3
    assert quant["layers.0.attn"]["bits"] == 3
    assert quant["head"] is False and quant["embed"] is False   # dense rest stays dense explicitly


def test_layer_kl_costs_refuse_mismatched_scans():
    units = K.inventory(_tiny(), layer_prefix="layers.")
    full = [f"layers.0.{m}" for m in ("attn", "mlp", "out")]
    def scan(bits, modules, **settings):
        doc = _scan(bits, [0.5])
        doc["rows"][0]["modules"] = modules
        doc["settings"].update(settings)
        return doc
    only0 = [u for u in units if u.layer in (None, 0)]
    assert K.layer_kl_costs({"q2g64": scan(2, full)}, units=only0) == {0: {"q2g64": 0.5}}
    with pytest.raises(ValueError, match="not the planned modules"):
        K.layer_kl_costs({"q2g64": scan(2, full[:1])}, units=only0)
    with pytest.raises(ValueError, match="RTN"):
        K.layer_kl_costs({"q2g64": scan(2, full, quantizer="gptq")})
    with pytest.raises(ValueError, match="groups/splits"):
        K.layer_kl_costs({"q2g64": scan(2, full, groups=["a"]), "q4g64": scan(4, full, groups=["b"])})
    doc = scan(2, full)
    doc["checkpoint"] = "other@sha256:bb"
    with pytest.raises(ValueError, match="scanned"):
        K.layer_kl_costs({"q2g64": doc}, checkpoint="m@sha256:aa")


def test_real_layer_kl_scan_covers_exactly_the_planned_modules(tmp_path):
    from mlx2 import layer_kl as L

    src = tmp_path / "dense"
    _tiny_mellum(src)
    model, _ = E.load_dense(src)
    model.set_dtype(mx.float32)  # MLX's CPU GatherMM is float32-only
    texts = [{"id": f"t{i}", "split": "dev", "ids": [(7 * i + 3 * j) % 128 for j in range(24)]} for i in range(2)]
    ref = tmp_path / "ref"
    L.write_reference(ref, texts, L.reference_from_model(model), skip=4, checkpoint="tiny@sha256:00",
                      source={"kind": "test"})
    docs = {}
    for name in ("q2g64", "q4g64"):
        f = K.parse_format(name)
        doc = L.scan(model, ref, layers=[0, 1, 2, 3], groups=["self_attn", "mlp.switch_mlp"], bits=f.bits,
                     group_size=f.group_size, mode=f.mode)
        doc["checkpoint"] = "tiny@sha256:00"
        docs[name] = json.loads(json.dumps(doc))
    units = K.inventory(model)
    costs = K.layer_kl_costs(docs, checkpoint="tiny@sha256:00", units=units)
    assert sorted(costs) == [0, 1, 2, 3]
    assert all(costs[l]["q2g64"] > costs[l]["q4g64"] for l in costs)


def test_export_is_atomic_and_validated(tmp_path):
    src = tmp_path / "dense"
    _tiny_mellum(src)
    model, _ = E.load_dense(src)
    units = K.inventory(model)
    plan = K.make_plan(checkpoint="t@sha256:00", units=units, menu=K.parse_menu(["q4g64"]),
                       costs={l: {"q4g64": 0.0} for l in range(4)}, budget=str(1 << 40),
                       scorer={"name": "uniform"}, mlx2={"commit": "t", "tracked_dirty": False},
                       rest_format=K.parse_format("q8g64"))
    out = tmp_path / "q4"
    seen = []

    def refuse(path):
        seen.append(path)
        raise RuntimeError("adapter refused")

    with pytest.raises(RuntimeError, match="adapter refused"):
        E.export(src, out, plan, log=lambda _m: None, validate=refuse)
    assert seen == [tmp_path / "q4.partial"] and not out.exists()
    with pytest.raises(FileExistsError):  # the failed partial is kept for inspection, never reused
        E.export(src, out, plan, log=lambda _m: None)
    import shutil
    shutil.rmtree(tmp_path / "q4.partial")
    receipt = E.export(src, out, plan, log=lambda _m: None, validate=lambda p: f"ok:{p.name}")
    assert receipt["validated"] == "ok:q4.partial" and out.is_dir() and not (tmp_path / "q4.partial").exists()


def test_uniform_plan_fits_exactly_without_a_gcd_bucket():
    # coprime layer sizes and a tiny max_cells force the rounding fallback; one option per row skips it
    sizes = [[10**9 + 1], [10**9 + 3], [10**9 + 7]]
    sol = K.solve([[0.0]] * 3, sizes, 3 * 10**9 + 11, max_cells=100)
    assert sol.exact and sol.nbytes == 3 * 10**9 + 11
    with pytest.raises(ValueError, match="no plan fits"):
        K.solve([[0.0]] * 3, sizes, 3 * 10**9 + 10, max_cells=100)


def test_user_resolution_that_does_not_divide_sizes_is_not_exact():
    sol = K.solve([[1.0, 0.0]] * 2, [[3, 5]] * 2, 10, resolution=2)
    assert not sol.exact and sol.nbytes <= 10
