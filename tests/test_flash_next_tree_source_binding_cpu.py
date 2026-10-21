"""P2: the Flash-Next tree executor's source binding is computed once per
source state, not hashed and ``git rev-parse``-d for every external batch."""

import os
from types import SimpleNamespace

from mlx2.adapters import flash_next_tree


def test_executor_source_binding_is_cached_until_a_source_changes(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        flash_next_tree.subprocess,
        "check_output",
        lambda *args, **kwargs: calls.append(args) or "abc123\n",
    )
    monkeypatch.setattr(flash_next_tree, "_SOURCE_BINDINGS", {})
    first = flash_next_tree.FlashNextTargetTreeExecutor(SimpleNamespace())
    second = flash_next_tree.FlashNextTargetTreeExecutor(SimpleNamespace())
    assert len(calls) == 1
    assert first.source_identity == second.source_identity
    assert first.source_revision == second.source_revision == "abc123"
    assert first.source_revision_token == second.source_revision_token
    assert len(first.source_identity) == 17

    # A changed source state on disk misses the cache (rehash + rev-parse).
    root = tmp_path
    paths = ("a.py",)
    (root / "a.py").write_text("one")
    identity, _ = flash_next_tree._source_binding(root, paths)
    assert flash_next_tree._source_binding(root, paths)[0] == identity
    assert len(calls) == 2
    (root / "a.py").write_text("two!")
    stat = (root / "a.py").stat()
    os.utime(root / "a.py", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    changed, _ = flash_next_tree._source_binding(root, paths)
    assert changed != identity and len(calls) == 3
