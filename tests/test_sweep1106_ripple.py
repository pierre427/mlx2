"""Bug sweep 2026-11-06, Ripple lane: CPU regressions (no weights, no Metal)."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

from mlx2.adapters.ripple import CACHE_LAYOUT, apc_layout, resolved_options

ROOT = Path(__file__).resolve().parents[1]


def test_affine_default_namespace_is_unchanged():
    # All three affine defaults on: metal_glue shares the metal_rmsnorm law.
    assert apc_layout(CACHE_LAYOUT, resolved_options("affine", {})) == (
        CACHE_LAYOUT + "+gqa_folded_attention+metal_rmsnorm")
    assert apc_layout(CACHE_LAYOUT, {}) == CACHE_LAYOUT
    assert apc_layout(CACHE_LAYOUT, {"fused_elementwise": True}) == CACHE_LAYOUT


def test_metal_glue_without_metal_rmsnorm_gets_its_own_namespace():
    # RIPPLE-1: glue alone reorders the Q/K norm FP32 sum while block norms
    # stay eager; it shared the eager namespace and could warm-hit a prefix
    # cached under a different arithmetic law.
    eager = apc_layout(CACHE_LAYOUT, {})
    rmsnorm = apc_layout(CACHE_LAYOUT, {"metal_rmsnorm": True})
    glue = apc_layout(CACHE_LAYOUT, {"metal_glue": True})
    assert len({eager, rmsnorm, glue}) == 3
    explicit_off = resolved_options("affine", {"metal_rmsnorm": False})
    assert apc_layout(CACHE_LAYOUT, explicit_off) == CACHE_LAYOUT + "+gqa_folded_attention+metal_glue"
    assert apc_layout(CACHE_LAYOUT, {"metal_rmsnorm": True, "metal_glue": True}) == rmsnorm


def _script(name):
    spec = importlib.util.spec_from_file_location(f"sweep1106_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("name", "function"), [
    ("probe_ripple_serving", "source_revision"),
    ("convert_ripple_oq4e", "checkout_revision"),
])
def test_receipt_revision_is_the_scripts_checkout_not_the_cwd(tmp_path, monkeypatch, name, function):
    # RIPPLE-2: ``git rev-parse HEAD`` ran in the caller's cwd, so a receipt
    # named whatever checkout the operator stood in (or failed outside one).
    expected = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    module = _script(name)
    monkeypatch.chdir(tmp_path)
    assert getattr(module, function)() == expected


def _serving_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_ripple_serving_sweep1106", ROOT / "scripts/probe_ripple_serving.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ripple_smoke_prompt_is_varied_by_default_and_legacy_reproducible():
    """RIPPLE-4: the smoke filler was one string repeated 160 times."""
    probe = _serving_probe()
    legacy = [
        {"role": "system", "content": "Use at most one short sentence of private reasoning, then give only the requested identifier."},
        {"role": "user", "content": "Archive notes: " + "Inventory checked. " * 160
         + "\nThe shipment identifier is ZX-3187. Reply with the shipment identifier only."},
    ]
    assert probe.messages("legacy-repeated") == legacy  # byte-identical to the 2026-11-04 receipts
    varied = probe.messages()[1]["content"]
    assert varied.endswith("ZX-3187. Reply with the shipment identifier only.")
    lines = [line for line in varied.splitlines() if line.strip()]
    assert len(set(lines)) == len(lines)  # no repeated filler lines
    assert varied.count("ZX-") == 1 and abs(len(varied) - len(legacy[1]["content"])) < 200
    receipt = probe.prompt_receipt("varied-v1")
    assert receipt["version"] == "varied-v1" and len(receipt["messages_sha256"]) == 64
    assert receipt != probe.prompt_receipt("legacy-repeated")
    fixture = (ROOT / "benchmarks/ripple_smoke/filler.md").read_text()
    import hashlib
    assert hashlib.sha256(fixture.encode()).hexdigest() in (ROOT / "benchmarks/ripple_smoke/README.md").read_text()


def test_ripple_reference_child_runs_the_same_prompt():
    source = (ROOT / "scripts/probe_ripple_serving.py").read_text()
    assert '"--reference-only", "--prompt", args.prompt' in source
    assert source.count("messages(args.prompt)") == 2  # reference and server arms


def test_hermes_strict_reader_rejects_repeated_members_at_any_depth():
    """RIPPLE-5: duplicate keys were accepted and the last silently kept."""
    from mlx2.runtime.tool_parsers import hermes_json

    tools = [{"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": {"path": {}, "opts": {}}}}} for name in ("a", "b")]
    for body in ('{"name": "a", "arguments": {}, "name": "b"}',
                 '{"name": "a", "arguments": {"path": "x", "path": "y"}}',
                 '{"name": "a", "arguments": {"opts": {"k": 1, "k": 2}}}'):
        with pytest.raises(hermes_json.DuplicateJSONKey, match="repeats the member"):
            hermes_json.parse_tool_call(body, tools)
    call = hermes_json.parse_tool_call('{"name": "a", "arguments": {"path": "x", "opts": {"k": [1, {"k": 2}]}}}', tools)
    assert call == {"name": "a", "arguments": {"path": "x", "opts": {"k": [1, {"k": 2}]}}}
