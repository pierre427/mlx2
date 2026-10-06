import importlib.util
from pathlib import Path


def _load_scout():
    path = Path(__file__).parents[1] / "scripts" / "research" / "bench_32k_proposal_strategy_scout.py"
    spec = importlib.util.spec_from_file_location("bench_32k_proposal_strategy_scout", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cascade_prunes_paths_outside_accepted_prefix_frontier():
    scout = _load_scout()
    target = (1, 2, 3, 4, 9, 6, 7, 8)
    paths = (
        (1, 2, 3, 4, 5, 6, 7),  # longest first; fails at token 5
        (0, 2, 3, 4, 9, 6),     # invalid after accepted token 1
        (1, 2, 0, 4, 9, 6),     # invalid after accepted token 3
        (1, 2, 3, 4, 9, 6),     # survives token 5's target correction
    )

    advance, rows, launches = scout.cascade_step(paths, target)

    assert (advance, rows, launches) == (7, 10, 2)


def test_cascade_does_not_launch_a_competing_path_with_the_wrong_prefix():
    scout = _load_scout()
    target = (1, 2, 3, 4, 9, 6, 7, 8)
    paths = (
        (1, 2, 3, 4, 5, 6, 7),
        (0, 2, 3, 4, 9, 6),
        (1, 2, 0, 4, 9, 6),
    )

    advance, rows, launches = scout.cascade_step(paths, target)

    assert (advance, rows, launches) == (5, 8, 1)
