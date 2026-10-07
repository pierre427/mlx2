"""Every adapter that opts into the incremental tokenizer cache renders the
same prompt text through ``render_incremental_prompt`` as its ordinary path.

The cache checks ordinary/reference parity on the first request of an epoch
only.  Nemotron 3 Super inherited Flash-Next's renderer (thinking defaults
off) while its ordinary path defaults thinking on, so after a first request
that set ``enable_thinking`` explicitly, a later request without it would
have been served a thinking-off prompt.
"""

import pytest

from mlx2.adapters.flash_next import FlashNextAdapter
from mlx2.adapters.nemotron3_super import Nemotron3SuperAdapter
from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
from mlx2.runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper


@pytest.fixture(scope="module")
def hf_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen3-0.6B tokenizer is not in the local HF cache")


MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    "cls", [FlashNextAdapter, Qwen3827BAdapter, Nemotron3SuperAdapter]
)
@pytest.mark.parametrize("thinking", [None, True, False])
def test_incremental_renderer_matches_ordinary_render(cls, thinking, hf_tokenizer):
    assert cls.incremental_tokenizer_cache_supported is True
    adapter = object.__new__(cls)
    adapter.tokenizer = TokenizerWrapper(
        hf_tokenizer, detokenizer_class=BPEStreamingDetokenizer
    )
    request = {"messages": MESSAGES}
    if thinking is not None:
        request["enable_thinking"] = thinking
    assert adapter.render_incremental_prompt(hf_tokenizer, request) == (
        adapter.render_prompt(request)
    )
    assert adapter.render_incremental_prompt(hf_tokenizer, {"prompt": "x"}) == "x"


def test_renderer_revisions_differ_where_renderers_differ():
    assert (
        Nemotron3SuperAdapter.incremental_tokenizer_renderer_revision
        != FlashNextAdapter.incremental_tokenizer_renderer_revision
    )
