"""``tool_choice: "none"`` asks for no call.

The engine's capability gate admits tools with ``tool_choice: "none"`` onto a
route without ``Capability.TOOLS`` (agent clients resend their tool list and
set "none" when they want plain text), but adapters that declare no TOOLS
refused any truthy ``tools`` with a 400 after the prompt was tokenized.  They
never render tools into the prompt, so "none" is servable as plain text; an
``auto``, ``required`` or named choice must still fail closed.
"""

from types import SimpleNamespace as NS

import pytest

from mlx2.adapters.agnes_3_flash import Agnes3FlashAdapter
from mlx2.adapters.gpt_oss import GptOssAdapter
from mlx2.adapters.granite_swa import GraniteSWAAdapter
from mlx2.adapters.hy_v3 import HYV3Adapter
from mlx2.adapters.lfm25_vl import LFM25VLAdapter
from mlx2.adapters.pinned_vlm_candidate import PinnedVisionCandidateAdapter
from mlx2.adapters.standard_decoder import StandardDecoderAdapter
from mlx2.server import validate_request

TOOLS = [{"type": "function", "function": {"name": "sum", "description": "", "parameters": {}}}]
CHAT = {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS}


def _tokenizer():
    return NS(
        apply_chat_template=lambda messages, **kw: [1, 2, 3],
        encode=lambda text, **kw: [4],
        decode=lambda ids, **kw: "",
    )


def _adapter(cls):
    value = object.__new__(cls)
    value.tokenizer = _tokenizer()
    value.config = {"model_type": "qwen3"}
    value.direct_final = True
    return value


TEXT_ADAPTERS = (
    HYV3Adapter, GraniteSWAAdapter, Agnes3FlashAdapter, GptOssAdapter, StandardDecoderAdapter,
)
ALL_ADAPTERS = TEXT_ADAPTERS + (LFM25VLAdapter, PinnedVisionCandidateAdapter)


@pytest.mark.parametrize("cls", TEXT_ADAPTERS, ids=lambda cls: cls.__name__)
def test_tools_with_tool_choice_none_are_served_as_plain_text(cls):
    request = validate_request({**CHAT, "tool_choice": "none", "max_tokens": 4})
    assert request["tools"] and request["tool_choice"] == "none"
    adapter = _adapter(cls)
    assert adapter.prompt_tokens(request) == adapter.prompt_tokens(
        {key: value for key, value in request.items() if key not in ("tools", "tool_choice")}
    )
    parser = adapter.output_parser(request)
    # Nothing is parsed as a call: the parser carries no tools.
    assert getattr(parser, "tools", None) in (None, [], ())
    if hasattr(adapter, "tool_constraint"):
        assert adapter.tool_constraint(request) is None


@pytest.mark.parametrize("cls", (LFM25VLAdapter, PinnedVisionCandidateAdapter),
                         ids=lambda cls: cls.__name__)
def test_vision_candidates_parse_tool_choice_none_as_plain_text(cls):
    parser = _adapter(cls).output_parser({**CHAT, "tool_choice": "none"})
    assert getattr(parser, "tools", None) in (None, [], ())


@pytest.mark.parametrize("cls", ALL_ADAPTERS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize(
    "choice", ["auto", "required", {"type": "function", "function": {"name": "sum"}}],
    ids=["auto", "required", "named"],
)
def test_callable_tools_still_fail_closed(cls, choice):
    adapter = _adapter(cls)
    with pytest.raises(ValueError):
        adapter.output_parser({**CHAT, "tool_choice": choice})


@pytest.mark.parametrize("cls", ALL_ADAPTERS, ids=lambda cls: cls.__name__)
def test_an_omitted_choice_with_tools_still_fails_closed(cls):
    with pytest.raises(ValueError):
        _adapter(cls).output_parser(dict(CHAT))
