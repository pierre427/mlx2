"""Tiny random HySparse2 research checkpoints exported for serving tests (CPU)."""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

VOCAB = 320
CORPUS = [
    "user: Explain what a GPU kernel is in one sentence.\n\nassistant: A small program.",
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
    "The quick brown fox jumps over the lazy dog. The quick brown fox again.",
    "for i in range(10):\n    total = total + i\n",
] * 8


def tiny_config(**overrides):
    from mlx2.experimental.hysparse2.config import Config

    return replace(Config.smoke(), vocab_size=VOCAB, max_context=4096, **overrides)


def train_tokenizer(path: Path) -> str:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(CORPUS, trainer)
    tokenizer.save(str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_export_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_hysparse2_serving.py"
    spec = importlib.util.spec_from_file_location("export_hysparse2_serving", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_export(root: Path, *, seed: int = 11, config=None) -> Path:
    """Save a model-only research checkpoint and export it for serving."""
    from mlx2.experimental.hysparse2.model import Model
    from mlx2.experimental.hysparse2.train import save_checkpoint

    root = Path(root)
    tokenizer = root / "tokenizer.json"
    tokenizer_sha = train_tokenizer(tokenizer)
    mx.random.seed(seed)
    model = Model(config or tiny_config())
    model.eval()
    mx.eval(model.parameters())
    checkpoint = save_checkpoint(
        root / "train", model, None, 7, {"tokenizer_sha256": tokenizer_sha}, mode="model"
    )
    output = root / "export"
    load_export_script().export(checkpoint, tokenizer, output)
    return output


def research_greedy(export: Path, prompt, steps):
    """Greedy tokens and per-step logits from the research model's own path."""
    import json

    from mlx2.experimental.hysparse2.config import Config
    from mlx2.experimental.hysparse2.model import Model
    from mlx2.experimental.hysparse2.train import initialize_from_checkpoint

    state = json.loads((export / "state.json").read_text())
    model = Model(Config(**state["config"]))
    initialize_from_checkpoint(export, model)
    model.eval()
    logits, cache = model.prefill(mx.array([list(prompt)]))
    rows, tokens = [logits[0, -1]], []
    for _ in range(steps):
        token = int(mx.argmax(rows[-1]).item())
        tokens.append(token)
        rows.append(model.decode(mx.array([[token]]), cache)[0, -1])
    return tokens, rows
