import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

from mlx2.adapters.nemotron35_super_config import SOURCE_REVISION
from mlx2.conversion.calibration_corpus import (
    canonical,
    prepare_corpus,
    select_examples,
    training_rows,
    verify_corpus,
)


def row(index):
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), (index, index // 2, 255 - index)).save(stream, format="PNG")
    return {"images": [{"bytes": stream.getvalue(), "path": None}],
            "texts": [{"user": f"<image> What is sample {index}?", "assistant": str(index), "source": "fixture"}]}


def test_image_rank_selection_is_repeatable_unique_and_excludes_other_cohorts():
    rows = [row(i) for i in range(20)]
    expected, _ = select_examples(rows, cohort="test", count=5)
    reverse, _ = select_examples(list(reversed(rows)), cohort="test", count=5)
    assert [r["image_sha256"] for r, _ in expected] == [r["image_sha256"] for r, _ in reverse]
    first_hash = expected[0][0]["image_sha256"]
    distinct, receipt = select_examples(rows + rows, cohort="test", count=5, excluded_images={first_hash})
    assert first_hash not in {r["image_sha256"] for r, _ in distinct}
    assert receipt["rejected"]["duplicate"] > 0
    assert all("<image>" not in r["question"] for r, _ in distinct)
    for record, _ in expected:
        unsigned = {k: v for k, v in record.items() if k != "input_sha256"}
        assert hashlib.sha256(canonical(unsigned)).hexdigest() == record["input_sha256"]


def test_bad_or_duplicate_images_do_not_pad_an_underfilled_cohort():
    bad = row(3)
    bad["images"][0]["bytes"] = b"not an image"
    with pytest.raises(ValueError, match="only 1 selected"):
        select_examples([row(1), row(1), bad, {"images": [], "texts": []}], cohort="test", count=2)


def make_fixture(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    file = cache / "train-00000.parquet"
    file.write_bytes(b"fixture only")
    shards = (("test", file.name, file.stat().st_size, hashlib.sha256(file.read_bytes()).hexdigest()),)
    destination = tmp_path / "corpus"
    manifest = prepare_corpus(cache, destination, shards=shards, per_cohort=3, reserve_bytes=0,
                              row_reader=lambda _: iter(row(i) for i in range(10)))
    return cache, destination, manifest, shards


def test_atomic_corpus_binds_images_questions_and_source_and_refuses_overwrite(tmp_path):
    cache, destination, manifest, shards = make_fixture(tmp_path)
    loaded, records = verify_corpus(destination, shards=shards, expected_digest=manifest["corpus_sha256"])
    assert loaded == manifest and len(records) == 3
    with pytest.raises(FileExistsError):
        prepare_corpus(cache, destination, shards=shards)
    with pytest.raises(ValueError, match="source binding"):
        verify_corpus(destination)
    with pytest.raises(ValueError, match="manifest binding"):
        verify_corpus(destination, shards=shards, expected_digest="0" * 64)
    image = destination / records[0]["image"]
    image.write_bytes(image.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="sample bytes"):
        verify_corpus(destination, shards=shards)


def test_changed_prompt_and_extra_image_fail_integrity(tmp_path):
    _, destination, manifest, shards = make_fixture(tmp_path)
    records_file = destination / "records.jsonl"
    records = records_file.read_text()
    records_file.write_text(records.replace("What is sample", "Ignore the sample"))
    with pytest.raises(ValueError, match="prompt bytes"):
        verify_corpus(destination, shards=shards)
    records_file.write_text(records)
    (destination / "images/extra.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="closure"):
        verify_corpus(destination, shards=shards)
    assert json.loads((destination / "manifest.json").read_text())["corpus_sha256"] == manifest["corpus_sha256"]


def test_pinned_shard_hash_precedes_corpus_creation(tmp_path):
    cache, destination, _, shards = make_fixture(tmp_path)
    (cache / shards[0][1]).write_bytes(b"wrong source")
    with pytest.raises(ValueError, match="pinned source"):
        prepare_corpus(cache, tmp_path / "other", shards=shards, reserve_bytes=0)
    assert not (tmp_path / "other").exists()
    assert destination.is_dir()


def test_real_parquet_schema_is_streamed_without_dataset_code(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "train-fixture.parquet"
    rows = [row(i) for i in range(4)]
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    assert list(training_rows(path)) == rows
    assert len(select_examples(training_rows(path), cohort="fixture", count=3)[0]) == 3


def test_acquisition_uses_explicit_file_credential_despite_other_hf_token(tmp_path, monkeypatch):
    import huggingface_hub

    script = Path(__file__).parents[1] / "scripts/prepare_nemotron35_calibration.py"
    spec = importlib.util.spec_from_file_location("calibration_prepare_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = tmp_path / "source"
    cache.mkdir()
    source = cache / "train-fixture.parquet"
    source.write_bytes(b"fixed fixture")
    shards = (("test", source.name, source.stat().st_size, hashlib.sha256(source.read_bytes()).hexdigest()),)
    marker = tmp_path / "complete.json"
    marker.write_text(json.dumps({"revision": SOURCE_REVISION, "files": 88, "total_bytes": 248937279780}))
    (tmp_path / ".hf_pr").write_text("file_credential_fixture\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HF_TOKEN", "unrelated_environment_fixture")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    for key in ("TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_XET", "HF_HUB_VERBOSITY", "HF_HUB_DISABLE_PROGRESS_BARS"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.setattr(module, "SHARDS", shards)
    observed = []

    def download(repo, filename, **kwargs):
        assert kwargs["token"] == "file_credential_fixture"
        assert kwargs["repo_type"] == "dataset" and kwargs["revision"] == module.REVISION
        observed.append((repo, filename))
        return source

    def prepare(cache, destination, **kwargs):
        return prepare_corpus(cache, destination, shards=shards, per_cohort=3,
                              row_reader=lambda _: iter(row(i) for i in range(10)), **kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(module, "prepare_corpus", prepare)
    monkeypatch.setattr(module, "verify_corpus", lambda path, **kw: verify_corpus(path, shards=shards, **kw))
    destination, state = tmp_path / "calibration", tmp_path / "state"
    monkeypatch.setattr(sys, "argv", [str(script), str(cache), str(destination), "--state", str(state),
                                     "--source-marker", str(marker), "--reserve-bytes", "0"])
    module.main()
    assert len(observed) == 1
    receipt = (state / "calibration-status.json").read_text()
    assert json.loads(receipt)["state"] == "prepared"
    assert "file_credential_fixture" not in receipt and "unrelated_environment_fixture" not in receipt
