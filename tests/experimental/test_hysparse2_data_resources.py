import hashlib
import io
import json
import os
import time

import pytest

from mlx2.experimental.hysparse2 import resources
from mlx2.experimental.hysparse2.download import fetch_file
from mlx2.experimental.hysparse2.mixture import Mixture


def test_verified_download_and_no_silent_overwrite(tmp_path):
    data = b"example corpus payload\n"
    item = {
        "path": "data/train.jsonl",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

    def get(url, timeout):
        assert "/" + ("a" * 40) + "/" in url
        return io.BytesIO(data)

    result = fetch_file("org/data", "a" * 40, item, tmp_path, open_url=get)
    assert result["status"] == "downloaded-verified"
    assert (
        fetch_file(
            "org/data",
            "a" * 40,
            item,
            tmp_path,
            open_url=lambda *a, **k: pytest.fail("unexpected network"),
        )["status"]
        == "verified-existing"
    )
    from pathlib import Path

    Path(result["path"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="existing"):
        fetch_file("org/data", "a" * 40, item, tmp_path, open_url=get)
    with pytest.raises(ValueError, match="unsafe"):
        fetch_file("org/data", "a" * 40, {**item, "path": "../outside"}, tmp_path)
    wrong = {**item, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="mismatch"):
        fetch_file("org/other", "a" * 40, wrong, tmp_path, open_url=get)
    assert not (tmp_path / "org--other" / ("a" * 40) / "data/train.jsonl").exists()


def test_mixture_token_weighting_and_determinism(tmp_path):
    np = pytest.importorskip("numpy")
    sources = []
    for name, n, weight in [("short", 40, 0.75), ("long", 4000, 0.25)]:
        path = tmp_path / (name + ".npy")
        np.save(path, np.arange(n, dtype=np.uint32))
        sources.append({"id": name, "path": path.name, "weight": weight})
    plan = tmp_path / "mixture.json"
    plan.write_text(json.dumps({"tokenizer_sha256": "hash", "sources": sources}))
    mixture = Mixture(plan, tokenizer_sha256="hash", sequence=8)
    a, names = mixture.sample(np.random.default_rng(42), 4000)
    b, names2 = mixture.sample(np.random.default_rng(42), 4000)
    assert np.array_equal(a, b) and names == names2
    assert 0.72 < names.count("short") / len(names) < 0.78
    assert a.shape == (4000, 10)
    with pytest.raises(ValueError, match="tokenizer"):
        Mixture(plan, tokenizer_sha256="wrong", sequence=8)
    sources[0]["sha256"] = "0" * 64
    plan.write_text(json.dumps({"tokenizer_sha256": "hash", "sources": sources}))
    with pytest.raises(ValueError, match="hash"):
        Mixture(plan, tokenizer_sha256="hash", sequence=8)


def test_gpu_guard_refuses_busy_and_preserves_foreign_files(tmp_path, monkeypatch):
    import fcntl

    monkeypatch.setattr(resources.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(resources.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(resources, "serving_processes", list)
    lock = tmp_path / "gpu.lock"
    waiters = tmp_path / "waiters"
    waiters.mkdir()
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (
            pytest.raises(RuntimeError, match="locks held"),
            resources.gpu_guard(locks=[str(lock)], waiters=str(waiters)),
        ):
            pytest.fail("must not enter")
    with resources.gpu_guard(locks=[str(lock)], waiters=str(waiters)):
        pass
    waiter = waiters / f"other.{os.getpid()}.{int(time.time()) - 5}"
    waiter.touch()
    with (
        pytest.raises(RuntimeError, match="waiters"),
        resources.gpu_guard(locks=[str(lock)], waiters=str(waiters)),
    ):
        pytest.fail("must not bypass queue")
    assert waiter.exists() and lock.exists()


def test_public_tools_and_magicoder_keep_code(tmp_path):
    from mlx2.experimental.hysparse2.corpus import json_examples

    path = tmp_path / "data.jsonl"
    row = {
        "uuid": "u",
        "messages": [
            {"role": "user", "content": "Write a kernel."},
            {
                "role": "assistant",
                "content": "Testing the kernel.",
                "tool_calls": [
                    {"name": "write_file", "arguments": {"code": "x = x + 1"}}
                ],
            },
            {
                "role": "tool",
                "content": "All tests passed.",
                "tool_call_id": "call_123",
                "name": "test_kernel",
            },
        ],
    }
    path.write_text(json.dumps(row) + "\n")
    text, _ = next(
        json_examples({"path": str(path), "id": "cuda", "kind": "public-tools"})
    )
    assert "x = x + 1" in text and "All tests passed." in text
    assert "call_123" in text and "test_kernel" in text
    path.write_text(
        json.dumps(
            {"problem": "Write a sum.", "solution": "return a+b", "seed": "seed"}
        )
        + "\n"
    )
    text, _ = next(
        json_examples({"path": str(path), "id": "magicoder", "kind": "magicoder"})
    )
    assert "return a+b" in text
