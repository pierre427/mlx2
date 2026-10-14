"""CPU checks for the chat-faithful scoring helpers in scripts/chat_quality.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import chat_quality


class WordTokenizer:
    """One token per whitespace-separated piece; newline is its own token."""

    def __init__(self, specials):
        self.vocab, self.words = {}, []
        for word in specials:
            self._id(word)

    def _id(self, word):
        if word not in self.vocab:
            self.vocab[word] = len(self.words)
            self.words.append(word)
        return self.vocab[word]

    def encode(self, text, add_special_tokens=False):
        pieces = text.replace("\n", " \n ").split(" ")
        return [self._id(p) for p in pieces if p]

    def decode(self, ids):
        return " ".join(self.words[i] for i in ids)


def test_chatml_mask_covers_assistant_bodies_and_their_end_marker():
    tok = WordTokenizer(["<|im_start|>", "<|im_end|>"])
    text = ("<|im_start|> system \n be brief <|im_end|> \n "
            "<|im_start|> user \n hi there <|im_end|> \n "
            "<|im_start|> assistant \n hello friend <|im_end|> \n "
            "<|im_start|> tool \n ok <|im_end|> \n "
            "<|im_start|> assistant \n bye <|im_end|>")
    ids = tok.encode(text)
    mask = chat_quality.assistant_mask(tok, ids)
    picked = [tok.words[i] for i, keep in zip(ids, mask) if keep]
    assert picked == ["hello", "friend", "<|im_end|>", "bye", "<|im_end|>"]


def test_gemma_turns_use_the_model_role():
    tok = WordTokenizer(["<start_of_turn>", "<end_of_turn>"])
    ids = tok.encode("<start_of_turn> user \n q <end_of_turn> \n <start_of_turn> model \n a b <end_of_turn>")
    mask = chat_quality.assistant_mask(tok, ids)
    assert [tok.words[i] for i, keep in zip(ids, mask) if keep] == ["a", "b", "<end_of_turn>"]


def test_text_without_turn_markers_has_no_mask():
    tok = WordTokenizer(["<|im_start|>", "<|im_end|>"])
    assert chat_quality.assistant_mask(tok, tok.encode("plain prose only")) is None


def test_compare_restricts_to_masked_rows():
    import mlx.core as mx

    ref = mx.log(mx.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]]))
    arm = mx.log(mx.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8]]))
    assert chat_quality.compare(ref, arm, rows=[True, False, True])["mean_kl"] == 0
    assert chat_quality.compare(ref, arm)["mean_kl"] > 0
    assert chat_quality.compare(ref, arm, rows=[False, False, False]) == {"positions": 0}


def test_real_lfm_template_masks_only_tool_call_turns():
    path = Path.home() / "mlx-models/LFM2.5-24B-A2B-MLX-8bit"
    if not (path / "tokenizer.json").is_file():
        pytest.skip("LFM2.5 artifact not present")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "find rec-1"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "f", "arguments": {"id": "rec-1"}}}]},
        {"role": "tool", "content": '{"ok": true}'},
    ]
    ids = tok.apply_chat_template(messages, tokenize=False)
    ids = list(tok.encode(ids, add_special_tokens=False))
    mask = chat_quality.assistant_mask(tok, ids)
    body = tok.decode([t for t, keep in zip(ids, mask) if keep])
    assert body == "<|tool_call_start|>[f(id='rec-1')]<|tool_call_end|><|im_end|>"
