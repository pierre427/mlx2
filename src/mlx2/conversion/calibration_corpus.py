"""Deterministic, content-bound image/text calibration corpus preparation.

Reads pinned local Parquet training shards without executing dataset code.
This corpus is for calibration only and must never be used as a quality test.
No model or accelerator is imported.
"""
from __future__ import annotations

import hashlib
import heapq
import io
import json
import os
import re
import shutil
import uuid
from pathlib import Path

from PIL import Image

from .nvfp4 import sha256

SCHEMA = "mlx2.image-text-calibration-corpus.v1"
DATASET = "HuggingFaceM4/the_cauldron"
REVISION = "847a98a779b1652d65111daf20c972dfcd333605"
SEED = 20261015
MAX_IMAGE_BYTES = 4 << 20
MAX_IMAGE_PIXELS = 16_000_000
MAX_TEXT_CHARACTERS = 8192
SHARDS = (
    ("ai2d", "ai2d/train-00000-of-00001-2ce340398c113b79.parquet", 438136609,
     "a1e05b99801ec73374dff401b7c91645465da955fc85ac62b2b5aaf6c9a18ab5"),
    ("chartqa", "chartqa/train-00000-of-00002-7733d4ca73ccd12e.parquet", 395345697,
     "7d96bf7b4b083f7228446c2c7eb1826f594872d715a33fe269fce1b2fbacf6c9"),
    ("cocoqa", "cocoqa/train-00000-of-00005-c3900470e9cef956.parquet", 481913547,
     "b15a803a5236e0d928833734282502e5e77330132994889be8ee2754bc11a4db"),
    ("docvqa", "docvqa/train-00002-of-00014-05fae2c793fa735b.parquet", 57588301,
     "d7034e65d9e30152b4c4f85bcddc61845fbc721abea41cf18a01443ddfc89071"),
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def image_geometry(data):
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_IMAGE_BYTES:
        raise ValueError("Calibration image bytes exceed the bound")
    with Image.open(io.BytesIO(data)) as image:
        if (image.format not in {"PNG", "JPEG", "WEBP"} or min(image.size) < 16
                or image.width * image.height > MAX_IMAGE_PIXELS
                or getattr(image, "n_frames", 1) != 1):
            raise ValueError("Unsupported calibration image geometry")
        geometry = {"width": image.width, "height": image.height, "format": image.format}
        image.verify()
    # Some decoders validate their compressed stream only when pixels load.
    with Image.open(io.BytesIO(data)) as image:
        image.convert("RGB").load()
    return geometry


def training_rows(path):
    # Optional preparation dependency, deliberately absent from the runtime.
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if not {"images", "texts"} <= set(parquet.schema_arrow.names):
        raise ValueError("Calibration shard lacks images/texts columns")
    for batch in parquet.iter_batches(batch_size=16, columns=["images", "texts"], use_threads=False):
        yield from batch.to_pylist()


def select_examples(rows, *, cohort, count, excluded_images=(), seed=SEED):
    """Select unique images by seeded hash rank, retaining at most count images.

    Ranking images instead of rows avoids oversampling repeated question rows.
    Exactly one valid QA is selected for each retained single-image example.
    """
    if type(count) is not int or count < 1 or not re.fullmatch(r"[a-z0-9_]+", cohort):
        raise ValueError("Invalid calibration selection")
    excluded, seen, selected, heap = set(excluded_images), set(), {}, []
    rejected = {"row": 0, "duplicate": 0, "image": 0}
    eligible = 0
    for row_index, row in enumerate(rows):
        images, texts = row.get("images"), row.get("texts")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(texts, list):
            rejected["row"] += 1
            continue
        raw = images[0].get("bytes") if isinstance(images[0], dict) else None
        if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_IMAGE_BYTES:
            rejected["image"] += 1
            continue
        pairs = []
        for qa_index, qa in enumerate(texts):
            if not isinstance(qa, dict):
                continue
            question, answer = qa.get("user"), qa.get("assistant")
            if not isinstance(question, str) or not isinstance(answer, str):
                continue
            question, answer = question.replace("<image>", "").strip(), answer.strip()
            if (not question or not answer or max(len(question), len(answer)) > MAX_TEXT_CHARACTERS
                    or re.search(r"<(?:img|video|image)[^>]*>", question + answer, re.IGNORECASE)):
                continue
            pairs.append((qa_index, question, answer))
        if not pairs:
            rejected["row"] += 1
            continue
        image_hash = hashlib.sha256(raw).hexdigest()
        if image_hash in excluded or image_hash in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(image_hash)
        rank = int.from_bytes(hashlib.sha256(f"{seed}:{cohort}:{image_hash}".encode()).digest(), "big")
        eligible += 1
        if len(heap) == count and (rank, image_hash) >= (-heap[0][0], heap[0][1]):
            continue
        try:
            geometry = image_geometry(raw)
        except (OSError, ValueError, Image.DecompressionBombError):
            rejected["image"] += 1
            eligible -= 1
            continue
        qa_index, question, answer = pairs[rank % len(pairs)]
        record = {"sample_id": f"{cohort}:{image_hash}", "cohort": cohort,
                  "source_row": row_index, "source_qa": qa_index,
                  "question": question, "answer": answer,
                  "image": f"images/{image_hash}.bin", "image_sha256": image_hash,
                  "image_bytes": len(raw), "geometry": geometry}
        record["input_sha256"] = hashlib.sha256(canonical(record)).hexdigest()
        if len(heap) == count:
            _, old_hash = heapq.heappop(heap)
            del selected[old_hash]
        heapq.heappush(heap, (-rank, image_hash))
        selected[image_hash] = (record, raw)
    if len(selected) != count:
        raise ValueError(f"Calibration cohort {cohort} has only {len(selected)} selected valid unique images; need {count}")
    return [selected[key] for key in sorted(selected)], {"eligible_images": eligible, "rejected": rejected}


def prepare_corpus(cache, destination, *, shards=SHARDS, per_cohort=256,
                   reserve_bytes=50_000_000_000, row_reader=training_rows, progress=None):
    """Publish atomically, retaining an incomplete directory on any failure."""
    destination, cache = Path(destination).absolute(), Path(cache).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if type(per_cohort) is not int or per_cohort < 1 or reserve_bytes < 0:
        raise ValueError("Invalid calibration preparation bounds")
    if len({item[0] for item in shards}) != len(shards) or not shards:
        raise ValueError("Calibration requires distinct source cohorts")
    for _, filename, size, digest in shards:
        file = cache / filename
        if (Path(filename).is_absolute() or ".." in Path(filename).parts
                or not Path(filename).name.startswith("train-")
                or file.is_symlink() or not file.is_file() or file.stat().st_size != size
                or sha256(file) != digest):
            raise ValueError("Calibration training shard bytes differ from the pinned source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < reserve_bytes + len(shards) * per_cohort * MAX_IMAGE_BYTES:
        raise OSError("Calibration preparation would consume the free-space reserve")
    staging = destination.with_name(destination.name + ".incomplete-" + uuid.uuid4().hex)
    (staging / "images").mkdir(parents=True)
    used, records, sources = set(), [], []
    for cohort, filename, size, digest in shards:
        examples, selection = select_examples(row_reader(cache / filename), cohort=cohort,
                                              count=per_cohort, excluded_images=used)
        for record, raw in examples:
            used.add(record["image_sha256"])
            with (staging / record["image"]).open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            records.append(record)
        sources.append({"cohort": cohort, "path": filename, "bytes": size,
                        "sha256": digest, "split": "train", "selected": len(examples), **selection})
        if progress:
            progress({"state": "preparing", "cohort": cohort, "samples": len(records)})
    records_path = staging / "records.jsonl"
    with records_path.open("xb") as handle:
        for record in records:
            handle.write(canonical(record) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    manifest = {"schema": SCHEMA, "purpose": "calibration_only_not_quality_evaluation",
                "dataset": DATASET, "revision": REVISION, "source_license": None,
                "seed": SEED, "selection": "unique single images ranked by seeded SHA256; one QA per image",
                "per_cohort": per_cohort, "samples": len(records), "sources": sources,
                "records_sha256": sha256(records_path),
                "limits": {"image_bytes": MAX_IMAGE_BYTES, "image_pixels": MAX_IMAGE_PIXELS,
                           "text_characters": MAX_TEXT_CHARACTERS}}
    manifest["corpus_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    with (staging / "manifest.json").open("xb") as handle:
        handle.write(canonical(manifest) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    os.rename(staging, destination)
    return manifest


def verify_corpus(path, *, shards=SHARDS, expected_digest=None):
    """Verify all prompt/image bytes before GPU calibration consumes them."""
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    digest = manifest.pop("corpus_sha256")
    if (manifest.get("schema") != SCHEMA or manifest.get("dataset") != DATASET
            or manifest.get("revision") != REVISION
            or manifest.get("purpose") != "calibration_only_not_quality_evaluation"
            or (expected_digest is not None and digest != expected_digest)
            or hashlib.sha256(canonical(manifest)).hexdigest() != digest):
        raise ValueError("Calibration corpus manifest binding differs")
    expected_sources = {(cohort, filename, size, sha) for cohort, filename, size, sha in shards}
    actual_sources = {(item["cohort"], item["path"], item["bytes"], item["sha256"])
                      for item in manifest["sources"] if item["split"] == "train"}
    if actual_sources != expected_sources or len(manifest["sources"]) != len(shards):
        raise ValueError("Calibration training source binding differs")
    manifest["corpus_sha256"] = digest
    records_file = path / "records.jsonl"
    if records_file.is_symlink() or sha256(records_file) != manifest["records_sha256"]:
        raise ValueError("Calibration prompt bytes differ")
    records, images, ids = [], set(), set()
    for line in records_file.read_bytes().splitlines():
        record = json.loads(line)
        image_hash = record["image_sha256"]
        if not isinstance(image_hash, str) or re.fullmatch(r"[0-9a-f]{64}", image_hash) is None:
            raise ValueError("Invalid calibration image identity")
        expected = "images/" + image_hash + ".bin"
        item = path / expected
        unsigned = {key: value for key, value in record.items() if key != "input_sha256"}
        if (record["image"] != expected or item.is_symlink() or not item.is_file()
                or image_hash in images or record["sample_id"] in ids
                or item.stat().st_size != record["image_bytes"] or sha256(item) != image_hash
                or hashlib.sha256(canonical(unsigned)).hexdigest() != record["input_sha256"]):
            raise ValueError("Calibration sample bytes or identity differ")
        images.add(image_hash)
        ids.add(record["sample_id"])
        records.append(record)
    cohorts = {item["cohort"] for item in manifest["sources"]}
    if (len(records) != manifest["samples"] or len(records) != manifest["per_cohort"] * len(cohorts)
            or any(sum(r["cohort"] == cohort for r in records) != manifest["per_cohort"] for cohort in cohorts)
            or (path / "images").is_symlink()
            or {item.name for item in (path / "images").iterdir()} != {key + ".bin" for key in images}):
        raise ValueError("Calibration corpus sample closure differs")
    return manifest, records
