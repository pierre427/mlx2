"""Bug-sweep 2026-11-06 regressions for the Inkling lane (CPU only)."""

import ast
import copy
import dataclasses
import importlib.util
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.adapters.inkling import validate_policy
from mlx2.runtime.models.inkling import Model
from mlx2.runtime.models.inkling.tree import InklingTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_inkling_model_cpu import args

mx.set_default_device(mx.cpu)


def _d128_model(policy):
    mx.random.seed(3)
    model = Model(dataclasses.replace(args(), head_dim=128, swa_head_dim=128))
    model.eval()
    mx.eval(model.parameters())
    model.configure_kernels(attention=policy)
    return model


@pytest.mark.parametrize("parents", [[-1, 0, 1], [-1, 0, 0]])
def test_sweep1106_ink1_mapped_tree_falls_back_like_ordinary(parents):
    """INK-1: an unsupported mapped launch keeps reference SDPA in the tree.

    Ordinary decode under ``inkling_attention: mapped`` falls back to the
    reference law where the kernel is unsupported (here: CPU); the packed
    tree raised ``ValueError`` instead of following the same law.
    """
    model = _d128_model("mapped")
    live = model.make_cache()
    for token in range(1, 6):
        model(mx.array([[token]]), cache=live)
    expected = copy.deepcopy(live)
    tokens = [7, 8, 9]
    tree = InklingTree(parents, live)
    logits, _ = tree.forward(model, mx.array([tokens]), [1])
    path = tree.paths[-1]
    reference = model(mx.array([[tokens[i] for i in path]]), cache=expected)
    mx.eval(logits, reference)
    assert bool(mx.allclose(logits[:, path[-1]], reference[:, -1], atol=1e-4).item())
    assert model.kernel_receipt()["mapped_attention_calls"] == 0


def test_sweep1106_ink2_shortconv_prefill_requires_fused():
    """INK-2: the wide-row shortconv kernel only exists inside the fused path."""
    with pytest.raises(ValueError, match="requires inkling_fused"):
        validate_policy({"inkling_shortconv_prefill": True})
    kernels, _ = validate_policy(
        {"inkling_shortconv_prefill": True, "inkling_fused": True}
    )
    assert kernels["inkling_shortconv_prefill"] is True


def _probe_module():
    spec = importlib.util.spec_from_file_location(
        "inkling_probe_sweep1106", ROOT / "scripts/probe_inkling.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sweep1106_ink3_source_revision_ignores_the_callers_cwd(tmp_path, monkeypatch):
    """INK-3: receipts name the commit of the tree that ran, not the cwd's."""
    probe = _probe_module()
    monkeypatch.chdir(tmp_path)  # not a git checkout
    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert probe.source_revision() == expected
    env = probe.tree_child_env(MLX2_RUN_GPU_TESTS="1")
    assert env["PYTHONPATH"].split(":")[0] == str(ROOT / "src")
    assert env["MLX2_RUN_GPU_TESTS"] == "1"


def _calls(path, name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == name):
            yield node


@pytest.mark.parametrize(
    "script",
    ["probe_inkling.py", "probe_inkling_divergence.py", "bench_inkling_ops.py"],
)
def test_sweep1106_ink3_git_and_children_are_pinned_to_this_tree(script):
    path = ROOT / "scripts" / script
    for call in _calls(path, "check_output"):
        argv = call.args[0] if call.args else None
        if isinstance(argv, ast.List) and argv.elts and getattr(
            argv.elts[0], "value", None
        ) == "git":
            assert any(k.arg == "cwd" for k in call.keywords), (script, call.lineno)
    for call in _calls(path, "Popen"):
        keywords = {k.arg for k in call.keywords}
        assert {"cwd", "env"} <= keywords, (script, call.lineno)


def test_sweep1106_ink3_caller_relative_paths_are_resolved_before_children():
    """Review follow-up: children run with cwd=ROOT, so --out etc. must be absolute."""
    source = (ROOT / "scripts" / "probe_inkling.py").read_text()
    parsed = source.index("args = ap.parse_args()")
    first_child = source.index("Popen(", parsed)
    block = source[parsed:first_child]
    for needle in ("args.out = args.out.resolve()", "args.model = args.model.resolve()",
                   "\"policy\", \"lane_policy\"", "args.divergence_reports = ["):
        assert needle in block, needle


def test_sweep1106_anchor_rounds_count_their_global_kv_lends():
    """Reported INK item: anchor-only rounds lent global KV but were uncounted."""
    from test_inkling_model_cpu import model as small_model

    from mlx2.runtime.inkling_tree_executor import InklingTreeExecutor

    target = small_model()
    cache = target.make_cache()
    mx.eval(target(mx.array([[1, 2, 1, 2]]), cache=cache))
    executor = InklingTreeExecutor.create()
    executor.forward_many(target, [[7]], [[-1]], [cache], [1], cached=False, configured_limit=1)
    receipt = executor.receipt()
    assert receipt["ordinary_anchor_dispatches"] == 1
    assert receipt["global_kv_lends"] > 0


def _head_model(tied):
    from test_inkling_model_cpu import args as small_args

    mx.random.seed(21)
    value = Model(dataclasses.replace(small_args(), hidden_size=64, tie_word_embeddings=tied))
    value.eval()
    mx.eval(value.parameters())
    return value


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("bits", [8, 4])
def test_sweep1106_mtp_draft_head_is_quantized_for_tied_and_untied_heads(tied, bits):
    """Tied checkpoints silently kept the BF16 head while the receipt said q8/q4."""
    from mlx import nn

    target = _head_model(tied)
    target.attach_mtp_draft_head((bits, 64))
    head = target._mtp_draft_head
    assert isinstance(head, nn.QuantizedLinear) and head.bits == bits
    hidden = mx.random.normal((3, 64))
    full, draft = target._project_logits(hidden), target._project_draft_logits(hidden)
    assert draft.shape == full.shape
    error = float(mx.abs(draft - full).max() / mx.abs(full).max())
    assert error < (0.02 if bits == 8 else 0.15)
    target.attach_mtp_draft_head(None)
    assert target._mtp_draft_head is None


def test_sweep1106_mtp_draft_head_from_a_quantized_source():
    """A quantized head is dequantized first (from_linear would have quantized
    its packed integers); a source no wider than the request is refused."""
    from mlx import nn

    for tied in (False, True):
        target = _head_model(tied)
        nn.quantize(target, group_size=64, bits=8,
                    class_predicate=lambda p, m: p in ("lm_head", "model.embed_tokens"))
        target.attach_mtp_draft_head((4, 64))
        hidden = mx.random.normal((2, 64))
        full, draft = target._project_logits(hidden), target._project_draft_logits(hidden)
        assert float(mx.abs(draft - full).max() / mx.abs(full).max()) < 0.15
        with pytest.raises(ValueError, match="no narrower"):
            target.attach_mtp_draft_head((8, 64))
