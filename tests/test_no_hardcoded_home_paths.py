"""Scripts and sources must not hard-code one user's home directory.

A literal "/Users/<name>/..." default only works on one machine, and a literal
"~/..." default is never expanded by argparse or Path(), so both break for
anyone else (the public mirror rewrote the first into the second). Defaults
under the home directory are spelled Path.home() / ... instead.
"""

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# scripts/benchmark_adaptive_mtp.py is hash-pinned by scripts/qualify_serving.py
# (APPROVED_ADAPTIVE_BENCHMARK_SHA256); only the qualifier lane may change it
# and re-pin. Remove this entry when that lands.
PINNED_EXCEPTIONS: set = set()


def _offending_literals(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    # Docstrings may quote the lab's literal command lines (some tests pin
    # them); only executable string values are defaults.
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)):
            docstrings.add(id(body[0].value))
    hits = []
    for node in ast.walk(tree):
        if id(node) in docstrings:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if "~" in value or value.startswith("~/"):
                hits.append((node.lineno, value.splitlines()[0][:80]))
    return hits


def test_no_hardcoded_home_paths_in_scripts_or_src():
    offenders = {}
    for top in ("scripts", "src"):
        for path in sorted((ROOT / top).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel in PINNED_EXCEPTIONS:
                continue
            hits = _offending_literals(path)
            if hits:
                offenders[rel] = hits
    assert offenders == {}


def test_named_script_defaults_follow_home(monkeypatch, tmp_path):
    for rel in (
        "scripts/capture_qwen4_gate_inject.py",
        "scripts/trace_qwen4_gate_inject_once.py",
    ):
        source = (ROOT / rel).read_text()
        assert 'default=Path.home() / "mlx-models" / ' in source, rel

    # The probe's default is a module constant: evaluate it under another HOME.
    monkeypatch.setenv("HOME", str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        "_probe_native_mtp_head_d1_home", ROOT / "scripts/probe_native_mtp_head_d1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.DEFAULT_MODEL).parent == tmp_path / "mlx-models"
