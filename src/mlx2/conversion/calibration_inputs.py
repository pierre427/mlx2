"""Content-bound inputs for owned Super VL calibration; no accelerator imports."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import numpy as np

from .calibration_corpus import canonical


def interleaved_records(records):
    cohorts = {}
    for record in records:
        cohorts.setdefault(record["cohort"], []).append(record)
    result = []
    for index in range(max(map(len, cohorts.values()), default=0)):
        for cohort in sorted(cohorts):
            if index < len(cohorts[cohort]):
                result.append(cohorts[cohort][index])
    return result


def input_digest(tokens, pixels, source_inputs):
    digest = hashlib.sha256(canonical(source_inputs))
    digest.update(np.asarray(tokens, dtype="<i4").tobytes())
    for image in pixels:
        array = np.asarray(image, dtype="<f4")
        digest.update(canonical(list(array.shape)))
        digest.update(array.tobytes())
    return digest.hexdigest()


def image_text_input(adapter, corpus, record):
    corpus = Path(corpus).resolve()
    file = corpus / record["image"]
    if file.is_symlink() or not file.resolve(strict=True).is_relative_to(corpus / "images"):
        raise ValueError("Calibration image escaped its verified corpus")
    raw = file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["image_sha256"]:
        raise ValueError("Calibration image changed after corpus verification")
    subtype = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[record["geometry"]["format"]]
    url = f"data:image/{subtype};base64," + base64.b64encode(raw).decode("ascii")
    request = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": record["question"]}]}], "enable_thinking": False}
    prepared = adapter.prepare_multimodal_request(request)
    # Teacher-force the published answer after the real assistant prefix.
    # No generation or guessed image+MTP alignment enters the statistics.
    tokens = list(adapter.prompt_tokens(prepared))
    tokens += list(adapter.tokenizer.encode(record["answer"], add_special_tokens=False)) + [11]
    pixels = prepared["_mlx2_prefill_inputs"]["image_pixels"]
    if len(pixels) != 1 or not 2 <= len(tokens) <= 8192:
        raise ValueError("Calibration image/text input exceeds its bounded shape")
    digest = input_digest(tokens, pixels, [record["input_sha256"]])
    return tokens, pixels, digest


def _text_packet_details(adapter, records, start, *, max_tokens=512):
    if not records or type(start) is not int or start < 0 or not 2 <= max_tokens <= 1024:
        raise ValueError("Invalid calibration text packet selection")
    tokens, sources, sample_ids = [], [], []
    for offset in range(len(records)):
        record = records[(start + offset) % len(records)]
        text = record["question"] + "\nAnswer: " + record["answer"] + "\n\n"
        encoded = list(adapter.tokenizer.encode(text, add_special_tokens=False)) + [11]
        tokens.extend(encoded)
        sources.append(record["input_sha256"])
        sample_ids.append(record["sample_id"])
        if len(tokens) >= max_tokens:
            break
    tokens = tokens[:max_tokens]
    if len(tokens) < 2:
        raise ValueError("Calibration text packet is empty")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Calibration text packet repeats a source record")
    return tokens, input_digest(tokens, [], sources), tuple(sample_ids)


def text_packet(adapter, records, start, *, max_tokens=512):
    tokens, digest, _sample_ids = _text_packet_details(
        adapter, records, start, max_tokens=max_tokens
    )
    return tokens, digest


def unit_text_packets(
    adapter,
    records,
    *,
    start,
    image_count,
    packet_count,
    max_tokens=512,
):
    """Partition one unit's image records into disjoint MTP text packets.

    A resumed unit advances start by its completed image count, so it cannot
    regenerate a prior unit's packet records. Within a unit, every source
    record belongs to at most one packet. The merge independently enforces
    record-level disjointness across units.
    """

    if (
        type(start) is not int
        or type(image_count) is not int
        or type(packet_count) is not int
        or start < 0
        or image_count < 0
        or packet_count < 0
        or start + image_count > len(records)
    ):
        raise ValueError("Invalid calibration unit text packet range")
    selected = list(records[start : start + image_count])
    count = min(packet_count, len(selected))
    packets = []
    for packet in range(count):
        first = packet * len(selected) // count
        stop = (packet + 1) * len(selected) // count
        tokens, digest, sample_ids = _text_packet_details(
            adapter, selected[first:stop], 0, max_tokens=max_tokens
        )
        packets.append(
            {
                "tokens": tokens,
                "input_sha256": digest,
                "source_record_ids": sample_ids,
            }
        )
    return packets
