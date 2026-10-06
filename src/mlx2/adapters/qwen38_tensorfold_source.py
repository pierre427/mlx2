"""Identity validation for mlx2's vendored Qwen3.8 TensorFold executor."""

from __future__ import annotations

import hashlib
from pathlib import Path

EXPECTED_REVISION = "1a5f38e12afbb560d8fc61c88ccb5900f7d5d170"
EXPECTED_TREE = "d9d141b99525a8866f48475858d7011fb192c11b"
VENDORED_SOURCE = "src/mlx2/runtime/tensorfold_qwen38"
VENDORED_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "tensorfold_qwen38"
QUALIFICATION_MODULE_SHA256 = {
    "__init__.py": "95ab120332366704fc5aeaac38a96c4cd85695e96d151eae6f8374028bdb855c",
    "inputs.py": "e13ffabe1a413781c846f50aeb69181ecb668a2ef8860311ae1222f96fcaf32c",
    "lane_attention.py": "fc66173e75b660c069011eb15d63ca51368132f017c55fc49d5c91c6529efd91",
    "lane_fuse.py": "af84f208d1bd718afddbb263a4f049e803764345fee972cc8a87bb81016b5831",
    "lane_glue.py": "aece8b0239e0960e88a929b8abb35ccfedfaab2f557087ab6b2f11003f89f899",
    "lane_multi.py": "d2b7d81038eec7e1cf035eddf340e28d5ff0a776f28feb8ce4252ade67624884",
    "lane_qmm.py": "02ee84c02a46e2b91ca3d77806fc57f0e19003115e766d975dbe9f1eefc158ac",
    "lane_tree.py": "e6de53828708e4833fcf390bc1dd330377cbdd26629da235c9728822770b5839",
    "lane_widen.py": "66369189f5300c06af754c0d867b7f50a55519aadb2342fe05785275767e8b77",
    "stream_attention.py": "cf3560e44bdf4789f7a0dc1cd6fe5c6760364347db0558dc5b97c10945ff63a0",
    "stream_gdn.py": "f120c8c25c5e2712ee214b770200d621b7d0824acb01cafc8f1a9b04ababc9a0",
}


def inspect_source(root: str | Path | None = None) -> dict[str, str]:
    """Return the in-package source identity without importing MLX."""

    path = VENDORED_ROOT if root is None else Path(root).expanduser().resolve()
    if path != VENDORED_ROOT:
        raise RuntimeError(
            f"Qwen3.8 TensorFold source must be the vendored mlx2 package: {VENDORED_ROOT}"
        )
    return {
        "root": str(path),
        "revision": EXPECTED_REVISION,
        "tree": EXPECTED_TREE,
        "tracked_source": VENDORED_SOURCE,
        "tracked_diff": "",
    }


def validate_source(root: str | Path | None = None) -> dict[str, str]:
    """Require every vendored module to match the reviewed source projection."""

    identity = inspect_source(root)
    for name, expected in QUALIFICATION_MODULE_SHA256.items():
        module_path = VENDORED_ROOT / name
        digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(
                f"vendored TensorFold module mismatch: {name}: {digest}, expected {expected}"
            )
    return identity


def qualification_source_identity(root: str | Path | None = None) -> dict:
    """Return the revision, tree and module hashes bound by qualification."""

    identity = validate_source(root)
    return {**identity, "module_sha256": dict(QUALIFICATION_MODULE_SHA256)}
