"""Host-only identity validation for the pinned Qwen3.8 TensorFold source."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

EXPECTED_REVISION = "1a5f38e12afbb560d8fc61c88ccb5900f7d5d170"
EXPECTED_TREE = "d9d141b99525a8866f48475858d7011fb192c11b"
TRACKED_SOURCE = "src/tensorfold"
QUALIFICATION_MODULE_SHA256 = {
    "src/tensorfold/families/qwen3_5/__init__.py": (
        "1b3501c5713a3c289f912620359290335b44b3ab8f16d28ec4fc76ded41e9546"
    ),
    "src/tensorfold/kernels/qwen/dense/v1/lane_fuse.py": (
        "c993ee6aa40b1b3b35e8df8095581f1a029b22a70238ea7bca8d0b56ace1abd0"
    ),
    "src/tensorfold/kernels/qwen/dense/v1/lane_glue.py": (
        "6346d180310ed4b066e0615a9009cbe2575f2798e71da1742fa53e083919c715"
    ),
    "src/tensorfold/kernels/qwen/dense/v1/lane_multi.py": (
        "b122e496f9ae9620717bbaa16c23e523ea8adaead5b111bddc75bb0beced5be3"
    ),
    "src/tensorfold/kernels/qwen/dense/v1/row_glue.py": (
        "5da32bedf4bbc9402be5fe68fde10b50ffae5d1c5e97c74cdd297724b0918bdf"
    ),
    "src/tensorfold/kernels/qwen/dense/v1/row_streams.py": (
        "9fa0fb3787f34c817847bc95e3fb7a1b336c01cf1e2df7fd0a50d130d5e8c13d"
    ),
    "src/tensorfold/kernels/qwen/dense/v1/stream_attention.py": (
        "7646c12946cdd990b76265c5664276033ce0597b60d40c11e0c8f9faa6dcd288"
    ),
    "src/tensorfold/kernels/qwen/dense/v1/stream_gdn.py": (
        "53531129f56eacedc83d5778189bae358be46c2832967fc6d92e8f3f4e86a892"
    ),
}


def inspect_source(root: str | Path) -> dict[str, str]:
    """Return the source identity without importing TensorFold or MLX."""

    path = Path(root).expanduser().resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()
    tracked_diff = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            TRACKED_SOURCE,
        ],
        cwd=path,
        text=True,
    ).strip()
    return {
        "root": str(path),
        "revision": revision,
        "tracked_source": TRACKED_SOURCE,
        "tracked_diff": tracked_diff,
    }


def validate_source(root: str | Path) -> dict[str, str]:
    """Require the pinned clean source tree and return its host-only identity."""

    identity = inspect_source(root)
    revision = identity["revision"]
    if revision != EXPECTED_REVISION:
        raise RuntimeError(
            f"TensorFold source revision mismatch: {revision}, "
            f"expected {EXPECTED_REVISION}"
        )
    if identity["tracked_diff"]:
        raise RuntimeError(
            "TensorFold source tree is dirty:\n" + identity["tracked_diff"]
        )
    return identity


def qualification_source_identity(root: str | Path) -> dict:
    """Bind the corrected source tree and qualification-relevant modules."""

    identity = validate_source(root)
    path = Path(identity["root"])
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=path, text=True
    ).strip()
    if tree != EXPECTED_TREE:
        raise RuntimeError(
            f"TensorFold source tree mismatch: {tree}, expected {EXPECTED_TREE}"
        )
    modules = {}
    for name, expected in QUALIFICATION_MODULE_SHA256.items():
        module_path = path / name
        digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(
                f"TensorFold qualification module mismatch: {name}: "
                f"{digest}, expected {expected}"
            )
        modules[name] = digest
    return {**identity, "tree": tree, "module_sha256": modules}
