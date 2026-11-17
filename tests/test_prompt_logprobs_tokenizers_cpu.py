"""Echo ``tokens``/``text_offset`` with real BPE tokenizers (tokenizer files only).

Regression for review finding M1 (BOS shifted offsets; generated offsets went
backwards; multibyte byte-level tokens) and H4 (formatting cost), from the
reviewer's shapes probe.
"""

import glob
import time
from pathlib import Path

import pytest

from mlx2.prompt_logprobs import PromptScores, TokenText, legacy_echo_logprobs, vllm_prompt_logprobs

PROMPT = "The quick brown fox jumps over the lazy dog. Café naïve résumé 日本語 テキスト emoji 🎉 done."
HUB = Path.home() / ".cache/huggingface/hub"


def _hub(repo):
    found = sorted(glob.glob(str(HUB / f"models--{repo}/snapshots/*")))
    return Path(found[0]) if found else HUB / "absent"


TOKENIZERS = {
    "qwen3": Path.home() / "mlx-models/Qwen3-0.6B-BF16-microscaling-source",
    "gemma4": Path.home() / "mlx-models/gemma-4-26B-A4B-it-MLX-8bit",
    # The reviewer's shapes_probe2 tokenizers (Hugging Face cache snapshots).
    "qwen3_hub": _hub("Qwen--Qwen3-0.6B"),
    "gemma3_hub": _hub("mlx-community--gemma-3-12b-it-4bit"),
}


def load(name):
    path = TOKENIZERS[name]
    if not (path / "tokenizer.json").is_file():
        pytest.skip(f"{name} tokenizer files are not on this host")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))


def scores_for(count, top=2):
    return PromptScores([-1.0] * count, [1] * count, [list(range(100, 100 + top))] * count,
                        [[-1.0] * top] * count, 1, 10)


@pytest.mark.parametrize("name,bos", [("qwen3", False), ("gemma4", True),
                                      ("qwen3_hub", False), ("gemma3_hub", True)])
def test_offsets_index_the_echoed_text(name, bos):
    tokenizer = load(name)
    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    if bos:
        ids = [tokenizer.bos_token_id, *ids]  # as the Gemma 4 adapter renders a prompt
    generated_ids = tokenizer.encode(" Fin 🎉", add_special_tokens=False)
    eos = tokenizer.eos_token_id
    generated = [{"id": token, "logprob": -0.5, "top_logprobs": []} for token in [*generated_ids, eos]]
    out = legacy_echo_logprobs(tokenizer, ids, scores_for(len(ids) - 1), 2, generated)
    texts = TokenText(tokenizer)
    text = PROMPT + " Fin 🎉"
    special = [token in texts.special for token in out["token_ids"]]
    joined = "".join(piece for piece, is_special in zip(out["tokens"], special) if not is_special)
    assert joined == text
    for piece, offset, is_special in zip(out["tokens"], out["text_offset"], special):
        if not is_special:
            assert text[offset:offset + len(piece)] == piece
    offsets = out["text_offset"]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0
    if bos:
        assert special[0] and offsets[1] == 0  # BOS has zero width
    assert offsets[len(ids)] == len(PROMPT)  # generation starts at the prompt's end
    assert special[-1] and offsets[-1] == len(text)  # EOS has zero width


def test_formatting_32k_top20_is_bounded():
    tokenizer = load("qwen3")
    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    big = (ids * (32768 // len(ids) + 1))[:32768]
    scores = scores_for(len(big) - 1, top=20)
    started = time.monotonic()
    texts = TokenText(tokenizer)
    legacy_echo_logprobs(tokenizer, big, scores, 20, texts=texts)
    vllm_prompt_logprobs(tokenizer, big, scores, 20, texts=texts)
    # Was 22.9 s for the legacy shape alone before per-id memoization.
    assert time.monotonic() - started < 5.0
