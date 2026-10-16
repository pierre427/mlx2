"""A coremltools stand-in importable in a spawned Core ML encoder worker.

CPU contract tests only: it never executes Core ML. Behaviour for the next
calls is read from the file named by ``MLX2_FAKE_COREML_CONTROL`` (absent or
empty means normal), so a test can change it while the worker is alive.
"""

import hashlib
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

__version__ = "fake-coreml-encoder-contract"
ComputeUnit = SimpleNamespace(CPU_AND_NE="CPU_AND_NE", ALL="ALL", CPU_AND_GPU="CPU_AND_GPU", CPU_ONLY="CPU_ONLY")


def fake_vector(ids, mask):
    """Deterministic unit vector of the exact padded inputs."""
    seed = hashlib.sha256(np.asarray(ids).tobytes() + np.asarray(mask).tobytes()).digest()
    vector = np.random.default_rng(int.from_bytes(seed[:8], "little")).standard_normal(384)
    return (vector / np.linalg.norm(vector)).astype(np.float32).reshape(1, 384)


def _mode():
    control = os.environ.get("MLX2_FAKE_COREML_CONTROL")
    try:
        return Path(control).read_text().strip() if control else ""
    except FileNotFoundError:
        return ""


class CompiledMLModel:
    def __init__(self, path, *, compute_units):
        if compute_units != ComputeUnit.CPU_AND_NE:
            raise ValueError(f"fake runtime accepts only CPU_AND_NE, got {compute_units!r}")
        leaked = [name for name in sys.modules if name == "mlx" or name.startswith("mlx.")]
        if leaked:
            raise RuntimeError(f"MLX imported in the Core ML worker: {leaked}")
        if _mode() == "fail_load":
            raise RuntimeError("injected load failure")
        self.length = int(Path(path).name.removeprefix("seq").removesuffix(".mlmodelc"))

    def predict(self, features):
        mode = _mode()
        ids, mask = features["input_ids"], features["attention_mask"]
        if set(features) != {"input_ids", "attention_mask"} or ids.shape != (1, self.length) or (
            ids.dtype != np.int32 or mask.dtype != np.int32 or mask.shape != ids.shape
        ):
            raise ValueError("fake model received inputs outside its fixed bucket")
        if mode == "sleep":
            time.sleep(30)
        if mode == "exit":
            os._exit(7)
        if mode == "nan":
            return {"embedding": np.full((1, 384), np.nan, dtype=np.float32)}
        if mode == "raise":
            raise RuntimeError("injected native failure")
        return {"embedding": fake_vector(ids, mask)}


models = SimpleNamespace(CompiledMLModel=CompiledMLModel)


TOKENIZER_SOURCE = (
    Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
    / "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer.json"
)
TINY_WORDS = ("hello", "world", "goodbye", "the", "word", "sea", "light", "##s", "a", "of", "ship", "lamp")


def tiny_tokenizer_bytes():
    """A small uncased WordPiece tokenizer with BERT special ids (101/102, pad 0)."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    vocab = {"[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102}
    vocab.update({word: 1000 + index for index, word in enumerate(TINY_WORDS)})
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=100))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 101), ("[SEP]", 102)]
    )
    return tokenizer.to_str().encode()


def pin_entry(data):
    import hashlib

    return (len(data), hashlib.sha256(data).hexdigest())


def pin_tokenizer(monkeypatch, data):
    """Make ``data`` the pinned MiniLM tokenizer entry for this test only."""
    from mlx2.adapters import minilm

    monkeypatch.setitem(minilm.ARTIFACT_FILES, "tokenizer.json", pin_entry(data))
    return data


def write_bundle(root, tokenizer_bytes, buckets=(32, 64, 128, 256), *, plan_available=True,
                 manifest_overrides=None, fractions=None):
    """Fabricate a manifest-valid bundle whose compiled trees are fake bytes.

    ``tokenizer_bytes`` must already be the pinned entry (see ``pin_tokenizer``).
    """
    import json

    from mlx2.adapters import minilm
    from mlx2.runtime import coreml_encoder as ce
    from mlx2.runtime.ane_projection import compiled_model_digest

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "tokenizer.json").write_bytes(tokenizer_bytes)
    rows = []
    for length in buckets:
        tree = root / "models" / f"seq{length}.mlmodelc"
        (tree / "weights").mkdir(parents=True)
        (tree / "model.mil").write_bytes(f"fake program {length}".encode())
        (tree / "weights/weight.bin").write_bytes(bytes([length % 251]) * 64)
        rows.append({
            "sequence_length": length, "batch": 1, "path": f"models/seq{length}.mlmodelc",
            "tree_sha256": compiled_model_digest(tree),
            "inputs": {"input_ids": "int32", "attention_mask": "int32"},
            "output": "embedding", "output_shape": [1, 384],
        })
    names = [str(length) for length in buckets]
    fractions = fractions or {}
    per_bucket = {
        name: {"neural_engine": 90, "cpu": 10, "unknown": 0,
               "neural_engine_compute_fraction": fractions.get(int(name), 0.95)}
        for name in names
    } if plan_available else {}
    size, digest = minilm.ARTIFACT_FILES["tokenizer.json"]
    manifest = {
        "schema": ce.SCHEMA,
        "family": "minilm",
        "source": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": minilm.MODEL_REVISION,
            "artifact_fingerprint": ce.source_fingerprint(),
            "model_safetensors_sha256": minilm.ARTIFACT_FILES["model.safetensors"][1],
        },
        "tokenizer": {"path": "tokenizer.json", "bytes": size, "sha256": digest},
        "representation": minilm.representation_contract(),
        "precision": "float16",
        "compute_units": "cpu_and_ne",
        "buckets": rows,
        "builder": {"tool": ce.BUILDER_TOOL, "source_sha256": "0" * 64, "coremltools": "fake",
                    "macos": "fake", "minimum_deployment_target": "macOS14", "graph_layout": "sequence_major",
                    "layer_norm": {"precision": "float16", "epsilon": 2.0 ** -14, "source_epsilon": 1e-12}},
        "build_parity": {
            "reference": "mlx_bert_cpu_float32", "threshold_min_cosine": 0.999,
            "compute_units": ["cpu_only", "cpu_and_ne"], "cases": 2 * len(names),
            "per_bucket": {name: {"cases": 2, "min_cosine": 0.9999, "max_abs_error": 0.001} for name in names},
            "min_cosine": 0.9999, "max_abs_error": 0.001, "passed": True,
        },
        "compute_plan": {
            "available": plan_available, "note": ce.PLAN_NOTE, "per_bucket": per_bucket,
            "placement_gate": ce.expected_placement_gate(per_bucket, 0.9) if plan_available else None,
        },
    }
    for override in manifest_overrides or ():
        override(manifest)
    (root / ce.MANIFEST_NAME).write_text(json.dumps(manifest))
    return root
