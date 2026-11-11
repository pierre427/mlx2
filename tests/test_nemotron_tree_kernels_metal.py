"""Metal bit-exactness of the Nemotron-H Mamba tree kernels.

Production geometry (Super: H128 / Dh64 / 8 groups / Ds128, conv4, C=10240):
every tree node's conv and gated-SSM output must equal the serial S=1 fused
decode kernels along its root-to-node path, and the S=p commit replay must
equal p S=1 decode calls.  Also the missing chain fact: ``fused_short`` at
S=k equals k S=1 calls.
"""

import copy

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Nemotron tree kernels are Metal-only"
)

from mlx2.runtime.models import nemotron_mamba_conv as conv  # noqa: E402
from mlx2.runtime.models import nemotron_mamba_ssm as ssm  # noqa: E402
from mlx2.runtime.models import nemotron_mamba_tree as tree  # noqa: E402

H, DH, G, DS = 128, 64, 8, 128
CHANNELS = H * DH + 2 * G * DS
LIMITS = [(0.0, float("inf")), (0.001, 100.0)]
TOPOLOGIES = [
    [-1],
    [-1, 0, 0, 0],
    [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    [-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 1, 10, 2, 12, 12, 7],
    [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
]


@pytest.fixture(autouse=True)
def _gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def _bits(value):
    mx.eval(value)
    view = mx.uint16 if value.dtype in (mx.bfloat16, mx.float16) else mx.uint32
    return np.asarray(value.view(view))


def _assert_bits(actual, expected):
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


def _inputs(rows, seed):
    keys = mx.random.split(mx.random.key(seed), 13)
    return {
        "conv_input": (mx.random.normal((1, rows, CHANNELS), key=keys[0]) * 2).astype(mx.bfloat16),
        "conv_state": (mx.random.normal((1, 3, CHANNELS), key=keys[1]) * 2).astype(mx.bfloat16),
        "weight": (mx.random.normal((CHANNELS, 4, 1), key=keys[2]) * 0.5).astype(mx.bfloat16),
        "bias": (mx.random.normal((CHANNELS,), key=keys[3]) * 0.5).astype(mx.bfloat16),
        "x": mx.random.normal((1, rows, H, DH), key=keys[4]).astype(mx.bfloat16),
        "B": mx.random.normal((1, rows, G, DS), key=keys[5]).astype(mx.bfloat16),
        "C": mx.random.normal((1, rows, G, DS), key=keys[6]).astype(mx.bfloat16),
        "dt": (mx.random.normal((1, rows, H), key=keys[7]) - 1).astype(mx.bfloat16),
        "gate": (mx.random.normal((1, rows, H * DH), key=keys[8]) * 2).astype(mx.bfloat16),
        "A_log": mx.random.uniform(-1, 2.5, (H,), key=keys[9]).astype(mx.float32),
        "D": mx.random.normal((H,), key=keys[10]).astype(mx.bfloat16),
        "dt_bias": mx.random.normal((H,), key=keys[11]).astype(mx.float32),
        "state": mx.random.normal((1, H, DH, DS), key=keys[12]).astype(mx.float32),
    }


def _serial_ssm_row(v, row, state, limit, gated):
    sl = slice(row, row + 1)
    if gated:
        return ssm.fused_mamba_ssm_gate_short(
            v["x"][:, sl], v["A_log"], v["B"][:, sl], v["C"][:, sl], v["D"],
            v["dt"][:, sl], v["dt_bias"], state, v["gate"][:, sl], limit,
        )
    return ssm.fused_mamba_ssm_short(
        v["x"][:, sl], v["A_log"], v["B"][:, sl], v["C"][:, sl], v["D"],
        v["dt"][:, sl], v["dt_bias"], state, limit,
    )


def test_process_parity_gate_admits_both_tree_kernels():
    tree.reset_parity_for_tests()
    for limit in LIMITS:
        assert tree.tree_kernel_parity(mx.bfloat16, limit) is None
    assert tree.tree_kernel_parity(mx.float32) is not None
    # Verdicts are per tile size and geometry; another geometry probes anew.
    assert tree.tree_kernel_parity(
        mx.bfloat16, LIMITS[0], geometry=(128, 64, 8, 128, 4096), rows=3
    ) is None
    status = tree.parity_status()
    assert any(key.endswith("x4096:S3") for key in status["verdicts"])
    assert sum(key.endswith(":S16") for key in status["verdicts"]) == 2
    assert status["law"] == tree.TREE_KERNEL_LAW
    assert all(value == "exact" for key, value in status["verdicts"].items()
               if key.startswith("mlx.core.bfloat16"))


@pytest.mark.parametrize("parents", TOPOLOGIES)
def test_tree_conv_matches_serial_short_conv_per_node(parents):
    v = _inputs(len(parents), seed=len(parents) * 7 + parents[-1])
    windows = tree.conv_windows(parents)
    output = tree.fused_mamba_conv_tree(
        v["conv_input"], v["conv_state"], v["weight"], v["bias"],
        mx.array(windows, dtype=mx.int32),
    )
    source = mx.concatenate([v["conv_state"], v["conv_input"]], axis=1)
    serial = []
    for row, window in enumerate(windows):
        out, _ = conv.fused_mamba_conv_short(
            v["conv_input"][:, row : row + 1],
            mx.take(source, mx.array(window, dtype=mx.int32), axis=1),
            v["weight"], v["bias"], capture_checkpoints=False,
        )
        serial.append(out)
    _assert_bits(output, mx.concatenate(serial, axis=1))


@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("parents", TOPOLOGIES)
def test_tree_ssm_matches_serial_decode_along_every_path(parents, gated, limit):
    v = _inputs(len(parents), seed=3 + len(parents))
    output = tree.fused_mamba_ssm_tree(
        v["x"], v["A_log"], v["B"], v["C"], v["D"], v["dt"], v["dt_bias"],
        v["state"], mx.array(parents, dtype=mx.int32), limit,
        gate=v["gate"] if gated else None,
    )
    states, serial = [None] * len(parents), []
    for row, parent in enumerate(parents):
        y, states[row] = _serial_ssm_row(
            v, row, v["state"] if parent < 0 else states[parent], limit, gated
        )
        serial.append(y)
    _assert_bits(output, mx.concatenate(serial, axis=1))


@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("gated", [True, False])
def test_short_ssm_at_s_k_equals_k_one_token_calls(gated, limit):
    """The chain fact the commit replay rests on, for every S = 2..16."""
    v = _inputs(16, seed=91)
    states = [v["state"]]
    outputs = []
    for row in range(16):
        y, state = _serial_ssm_row(v, row, states[-1], limit, gated)
        states.append(state)
        outputs.append(y)
    for steps in range(2, 17):
        sl = slice(0, steps)
        if gated:
            y, state = ssm.fused_mamba_ssm_gate_short(
                v["x"][:, sl], v["A_log"], v["B"][:, sl], v["C"][:, sl], v["D"],
                v["dt"][:, sl], v["dt_bias"], v["state"], v["gate"][:, sl], limit,
            )
        else:
            y, state = ssm.fused_mamba_ssm_short(
                v["x"][:, sl], v["A_log"], v["B"][:, sl], v["C"][:, sl], v["D"],
                v["dt"][:, sl], v["dt_bias"], v["state"], limit,
            )
        _assert_bits(state, states[steps])
        _assert_bits(y, mx.concatenate(outputs[:steps], axis=1))


def test_gated_and_plain_short_kernels_commit_the_same_state():
    """Chain partial accepts replay with the plain kernel; decode uses gated."""
    v = _inputs(16, seed=17)
    for steps in (1, 2, 5, 16):
        sl = slice(0, steps)
        _, gated = ssm.fused_mamba_ssm_gate_short(
            v["x"][:, sl], v["A_log"], v["B"][:, sl], v["C"][:, sl], v["D"],
            v["dt"][:, sl], v["dt_bias"], v["state"], v["gate"][:, sl], LIMITS[0],
        )
        _, plain = ssm.fused_mamba_ssm_short(
            v["x"][:, sl], v["A_log"], v["B"][:, sl], v["C"][:, sl], v["D"],
            v["dt"][:, sl], v["dt_bias"], v["state"], LIMITS[0],
        )
        _assert_bits(gated, plain)


def test_short_conv_at_s_k_equals_k_one_token_calls():
    v = _inputs(16, seed=5)
    state = v["conv_state"]
    outputs = []
    for row in range(16):
        out, state = conv.fused_mamba_conv_short(
            v["conv_input"][:, row : row + 1], state, v["weight"], v["bias"],
            capture_checkpoints=False,
        )
        outputs.append(out)
    for steps in (2, 7, 16):
        out, history = conv.fused_mamba_conv_short(
            v["conv_input"][:, :steps], v["conv_state"], v["weight"], v["bias"],
            capture_checkpoints=False,
        )
        _assert_bits(out, mx.concatenate(outputs[:steps], axis=1))


# -- the model-level tree on Metal at production Mamba geometry -------------


def _metal_model():
    from mlx2.runtime.models.nemotron_h import Model, ModelArgs

    mx.random.seed(5)
    args = ModelArgs(
        model_type="nemotron_h", vocab_size=64, hidden_size=256,
        intermediate_size=128, num_hidden_layers=4, max_position_embeddings=4096,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        attention_bias=False, mamba_num_heads=H, mamba_head_dim=DH,
        mamba_proj_bias=False, ssm_state_size=DS, conv_kernel=4, n_groups=G,
        mlp_bias=False, layer_norm_epsilon=1e-5, use_bias=False,
        use_conv_bias=True, hybrid_override_pattern=["M", "*", "-", "M"],
    )
    model = Model(args)
    model.set_dtype(mx.bfloat16)
    # The recurrent parameters stay fp32 as in the served checkpoint.
    for layer in model.layers:
        if layer.block_type == "M":
            layer.mixer.A_log = layer.mixer.A_log.astype(mx.float32)
            layer.mixer.dt_bias = layer.mixer.dt_bias.astype(mx.float32)
    model.set_short_segment_policy("layerwise_short_exact")
    for layer in model.layers:
        if layer.block_type == "M":
            layer.mixer.set_conv_policy("fused_short")
            layer.mixer.set_ssm_policy("fused_short_gate")
            layer.mixer.set_projection_policy("short_row_exact")
            layer.mixer.norm.set_policy("fused_short")
    model.eval()
    return model


def _prefill(model, length):
    cache = model.make_cache()
    tokens = mx.array([[(7 * index + 3) % 64 for index in range(length)]])
    model(tokens, cache)
    for entry in cache:
        if hasattr(entry, "cache") and entry.cache[1] is not None:
            entry.cache[1] = entry.cache[1].astype(mx.float32)
    mx.eval([entry.state for entry in cache])
    return cache


@pytest.mark.parametrize("serial", [False, True])
@pytest.mark.parametrize("prefix", [37, 250, 300])
def test_model_tree_rows_and_commit_match_one_token_decode(prefix, serial, monkeypatch):
    from mlx2.runtime.hybrid_tree_executor import HybridTreeExecutor
    from mlx2.runtime.models.cache import ArraysCache
    from mlx2.runtime.nemotron_tree_executor import NemotronTreeExecutor

    if serial:
        # A parity refusal must leave the serial per-node backend, still exact.
        monkeypatch.setattr(tree, "tree_kernel_parity", lambda *a, **k: "forced refusal")
    model = _metal_model()
    cache = _prefill(model, prefix)
    parents = TOPOLOGIES[3]
    tokens = [(5 * index + 1) % 64 for index in range(len(parents))]
    layers = (0, 3)
    ref_logits, ref_taps, ref_tx = HybridTreeExecutor().forward(
        model, tokens, parents, cache, layers, cached=False
    )
    mx.eval(ref_logits, ref_taps)
    executor = NemotronTreeExecutor()
    for path in ([0], [0, 1, 2, 3, 4, 5, 6, 7, 15], [0, 1, 10, 11], [0, 1, 2, 12, 14]):
        live = copy.deepcopy(cache)
        logits, taps, transaction = executor.forward(
            model, tokens, parents, live, layers, cached=False
        )
        mx.eval(logits, taps)
        _assert_bits(logits, ref_logits)
        _assert_bits(taps, ref_taps)
        transaction.commit_paths([path])
        oracle = copy.deepcopy(cache)
        _l, _t, tx = HybridTreeExecutor().forward(
            model, tokens, parents, oracle, layers, cached=False
        )
        mx.eval(_l, _t)
        tx.commit_paths([path])
        for left, right in zip(live, oracle, strict=True):
            if isinstance(left, ArraysCache):
                for a, e in zip(left.cache, right.cache, strict=True):
                    _assert_bits(a, e)
            else:
                assert left.offset == right.offset
                for a, e in zip(left.state, right.state, strict=True):
                    _assert_bits(a[..., : left.offset, :], e[..., : right.offset, :])
        step = mx.array([[9]])
        _assert_bits(model(step, live), model(step, oracle))
    receipt = executor.receipt()
    expected = "serial_or_mixed" if serial else "fused_tree"
    assert receipt["mamba_backend"] == expected, receipt["backends"]
    ref_tx.abort()
