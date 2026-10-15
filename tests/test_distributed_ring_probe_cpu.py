import ast
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_distributed_serving_ring.py"


def source():
    return SCRIPT.read_text()


def test_probe_requires_strict_two_rank_ring_and_gpu_ownership():
    text = source()
    assert 'init(strict=True, backend="ring")' in text
    assert "size != 2" in text
    assert "require_qualification_lease" in text
    assert "--cpg-generation" in text


def test_fault_modes_keep_collective_sequence_common():
    tree = ast.parse(source())
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    calls = [ast.unparse(node.func) for node in ast.walk(run) if isinstance(node, ast.Call)]
    assert calls.count("mx.distributed.all_gather") == 2
    assert calls.count("mx.distributed.all_sum") == 1
    # No mode branch may return or raise before all three collectives complete.
    all_sum = next(node for node in ast.walk(run) if isinstance(node, ast.Call)
                   and ast.unparse(node.func) == "mx.distributed.all_sum")
    for node in ast.walk(run):
        if isinstance(node, (ast.Return, ast.Raise)) and node.lineno < all_sum.lineno:
            assert node.lineno in {39, 41}  # strict world-size refusal only


def test_receipt_keeps_transport_observation_separate_from_qualification():
    text = source()
    assert '"state_publishable": publishable' in text
    assert '"qualified": False' in text
    assert '"performance_qualified": False' in text
