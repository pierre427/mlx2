"""Every caller of a contract-bound kernel API names its contract.

Sweep 2026-10-12 L4-F2: e3c19168e made the model-owned contract mandatory
for the fused-GDN probes and builders (a missing one refuses) and a required
keyword of ``qwen4_moe_window.router_topk``.  The Metal gate tests and the
GPU check scripts kept the old contract-less calls, so every gate either
declined, raised ``ValueError: missing fused GDN contract`` or raised
``TypeError`` -- but only on a GPU host, never in CPU CI.

The static scan derives the contract-bound API from the modules themselves and
checks every call in tests/ and scripts/.  The second test runs the call forms
the gates now use against a stubbed ``mx.fast.metal_kernel`` in a fresh
process (the stub must not leak into this process's kernel caches); no Metal
work is dispatched.
"""

import ast
import inspect
import os
import pathlib
import subprocess
import sys

from mlx2.runtime.models import (
    qwen4_fused_gdn,
    qwen4_fused_gdn_verify,
    qwen4_moe_window,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _contract_bound_api():
    """{name: accepts_architecture} for functions that refuse without a contract."""
    bound = {}
    for module in (qwen4_fused_gdn, qwen4_fused_gdn_verify, qwen4_moe_window):
        for name, fn in vars(module).items():
            if not inspect.isfunction(fn) or fn.__module__ != module.__name__:
                continue
            params = inspect.signature(fn).parameters
            contract = params.get("contract")
            if contract is None:
                continue
            source = inspect.getsource(fn)
            if contract.default is inspect.Parameter.empty or (
                "resolve_qwen_gdn_contract(" in source
                and "contract = QWEN4_GDN_CONTRACT" not in source
            ):
                bound[name] = "architecture" in params
    return bound


def _dict_has_contract(node):
    if isinstance(node, ast.Dict):
        return any(
            isinstance(key, ast.Constant) and key.value == "contract" for key in node.keys
        )
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dict":
        return any(keyword.arg == "contract" for keyword in node.keywords)
    return False


def _starred_has_contract(scope, value):
    """Resolve ``**name`` to a dict built in the same function."""
    if not isinstance(value, ast.Name):
        return False
    for node in ast.walk(scope):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == value.id
                for target in node.targets
            )
            and _dict_has_contract(node.value)
        ):
            return True
    return False


def _missing_contract_calls(path, bound):
    tree = ast.parse(path.read_text())
    scopes = [tree] + [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    owner = {}
    for scope in scopes:
        for node in ast.walk(scope):
            owner[id(node)] = scope  # innermost scope wins (walked last)
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name not in bound:
            continue
        keywords = {keyword.arg for keyword in node.keywords}
        if "contract" in keywords or (bound[name] and "architecture" in keywords):
            continue
        starred = [keyword.value for keyword in node.keywords if keyword.arg is None]
        if any(_starred_has_contract(owner[id(node)], value) for value in starred):
            continue
        missing.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    return missing


def test_contract_bound_api_is_what_the_gates_call():
    bound = _contract_bound_api()
    for name in (
        "probe_qwen4_fused_gdn_decode",
        "qwen4_fused_gdn_decode",
        "qwen4_fused_gdn_batch_decode",
        "probe_qwen4_fused_gdn_verify",
        "qwen4_fused_gdn_verify",
        "probe_qwen4_fused_gdn_catchup",
        "qwen4_fused_gdn_catchup",
        "qwen4_fused_gdn_batch_verify",
        "router_topk",
    ):
        assert name in bound, name


def test_every_test_and_script_call_names_its_contract():
    bound = _contract_bound_api()
    missing = []
    for root in ("tests", "scripts"):
        for path in sorted((ROOT / root).rglob("*.py")):
            missing.extend(_missing_contract_calls(path, bound))
    assert not missing, "\n".join(missing)


_STUBBED_CALLS = r'''
import mlx.core as mx
mx.set_default_device(mx.cpu)
def _fake_metal_kernel(**spec):
    def run(*, output_shapes, output_dtypes, **_):
        return [mx.zeros(s, dtype=d) for s, d in zip(output_shapes, output_dtypes)]
    return run
mx.fast.metal_kernel = _fake_metal_kernel
from mlx2.runtime.models import qwen4_fused_gdn as F
from mlx2.runtime.models import qwen4_fused_gdn_verify as V
from mlx2.runtime.models import qwen4_moe_window as W
from mlx2.runtime.models.qwen_gdn_contracts import QWEN4_GDN_CONTRACT as C
from mlx2.runtime.models.qwen_moe_contracts import QWEN4_E512_K10_ROUTING, QWEN36_E256_K8_ROUTING
# This fixture checks API contracts and shapes. Zero-filled fake Metal
# outputs cannot establish served-SiLU equivalence; supply that dependency
# explicitly while retaining the real selector and kernel-header builder.
# test_fused_gdn_served_silu.py covers the numerical oracle and refusals.
F._probe_served_silu = lambda: {F.KERNEL_SILU_EXP: True}
F.fused_gdn_runtime_supported = lambda: True
V.fused_gdn_runtime_supported = lambda: True
W.RD.runtime_supported = lambda: True
z = lambda *s: mx.zeros(s, dtype=mx.bfloat16)
out = {}
out["probe_decode"] = F.probe_qwen4_fused_gdn_decode(mx.bfloat16, contract=C)
out["probe_verify"] = V.probe_qwen4_fused_gdn_verify(mx.bfloat16, 4, contract=C)
out["probe_catchup"] = V.probe_qwen4_fused_gdn_catchup(mx.bfloat16, 4, contract=C)
out["verify"] = type(V.qwen4_fused_gdn_verify(z(1, 4, 8192), z(1, 4, 6144), z(1, 4, 48), z(1, 4, 48), z(1, 3, 8192), z(8192, 4, 1), mx.zeros((48,)), z(48), mx.zeros((1, 48, 128, 128)), z(128), 1e-6, threadgroup_y=4, contract=C)).__name__
out["decode"] = type(F.qwen4_fused_gdn_decode(z(1, 1, 8192), z(1, 1, 6144), z(1, 1, 48), z(1, 1, 48), z(1, 3, 8192), z(8192, 4, 1), mx.zeros((48,)), z(48), mx.zeros((1, 48, 128, 128)), z(128), 1e-6, threadgroup_y=4, contract=C)).__name__
out["router_qwen4"] = type(W.router_topk(z(1, 512), contract=QWEN4_E512_K10_ROUTING)).__name__
out["router_qwen36"] = type(W.router_topk(z(1, 256), top_k=8, contract=QWEN36_E256_K8_ROUTING)).__name__
print(out)
'''


def test_gate_call_forms_build_against_a_stubbed_metal_kernel():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), *filter(None, [env.get("PYTHONPATH")])]
    )
    result = subprocess.run(
        [sys.executable, "-c", _STUBBED_CALLS],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == str(
        {
            "probe_decode": 32,
            "probe_verify": 32,
            "probe_catchup": 32,
            "verify": "tuple",
            "decode": "tuple",
            "router_qwen4": "list",
            "router_qwen36": "list",
        }
    )
