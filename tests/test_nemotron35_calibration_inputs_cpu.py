import hashlib
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from mlx2.adapters.nemotron35_super import Nemotron35SuperAdapter
from mlx2.conversion.calibration_inputs import (
    image_text_input,
    interleaved_records,
    text_packet,
    unit_text_packets,
)


class Tokenizer:
    def encode(self, text, **kwargs):
        return [4 + ord(char) % 10 for char in text]

    def apply_chat_template(self, messages, **kwargs):
        placeholders = sum(message["content"].count("<image>") for message in messages)
        return [1] + [18] * placeholders + [2, 3]


def record(index, cohort="fixture"):
    return {"sample_id": f"{cohort}:{index}", "cohort": cohort,
            "question": f"Question {cohort} {index}", "answer": str(index),
            "input_sha256": hashlib.sha256(f"{cohort}:{index}".encode()).hexdigest()}


def test_actual_owned_media_processor_and_assistant_answer_are_content_bound(tmp_path):
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), (128, 128, 128)).save(stream, format="PNG")
    raw = stream.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    (tmp_path / "images").mkdir()
    image = tmp_path / "images" / (digest + ".bin")
    image.write_bytes(raw)
    sample = {**record(1), "image": "images/" + image.name, "image_sha256": digest, "geometry": {"format": "PNG"}}
    adapter = Nemotron35SuperAdapter.__new__(Nemotron35SuperAdapter)
    adapter.tokenizer, adapter.draft_model = Tokenizer(), None
    adapter.source_config = {"norm_mean": [.5] * 3, "norm_std": [.5] * 3}
    adapter.identity = {"fingerprint": "proxy"}
    tokens, pixels, identity = image_text_input(adapter, tmp_path, sample)
    assert tokens.count(18) == 256
    assert tokens[-2:] == adapter.tokenizer.encode("1") + [11]
    assert pixels[0].shape == (1, 3, 512, 512)
    np.testing.assert_allclose(pixels[0], (128 / 255 - .5) / .5, rtol=0, atol=1e-6)
    assert image_text_input(adapter, tmp_path, sample)[2] == identity
    assert image_text_input(adapter, tmp_path, {**sample, "answer": "2"})[2] != identity
    image.write_bytes(raw + b"changed")
    with pytest.raises(ValueError, match="changed"):
        image_text_input(adapter, tmp_path, sample)


def test_interleaving_and_text_packets_are_deterministic_and_token_bound():
    samples = [record(i, cohort) for cohort in ("ai2d", "chartqa", "docvqa") for i in range(2)]
    ordered = interleaved_records(samples)
    assert [sample["cohort"] for sample in ordered] == ["ai2d", "chartqa", "docvqa"] * 2
    adapter = SimpleNamespace(tokenizer=Tokenizer())
    tokens, digest = text_packet(adapter, ordered, 0, max_tokens=32)
    assert len(tokens) == 32
    assert text_packet(adapter, ordered, 0, max_tokens=32) == (tokens, digest)
    assert text_packet(adapter, ordered, 1, max_tokens=32)[1] != digest
    assert text_packet(adapter, ordered, len(ordered), max_tokens=32)[1] == digest


def test_unit_text_packets_partition_source_records_without_overlap():
    records = [record(index) for index in range(12)]
    adapter = SimpleNamespace(tokenizer=Tokenizer())
    first = unit_text_packets(
        adapter, records, start=0, image_count=6, packet_count=4, max_tokens=32
    )
    resumed = unit_text_packets(
        adapter, records, start=6, image_count=6, packet_count=4, max_tokens=32
    )
    first_ids = [
        sample_id for packet in first for sample_id in packet["source_record_ids"]
    ]
    resumed_ids = [
        sample_id for packet in resumed for sample_id in packet["source_record_ids"]
    ]
    assert len(first_ids) == len(set(first_ids))
    assert len(resumed_ids) == len(set(resumed_ids))
    assert set(first_ids).isdisjoint(resumed_ids)


def test_image_path_cannot_traverse_outside_verified_image_directory(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "outside").write_bytes(b"outside")
    sample = {**record(1), "image": "images/../outside"}
    with pytest.raises(ValueError, match="escaped"):
        image_text_input(None, tmp_path, sample)
