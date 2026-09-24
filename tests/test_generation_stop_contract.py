from types import SimpleNamespace

import pytest

from mlx2.adapters.flash_next import FlashNextAdapter
from mlx2.adapters.gemma4 import Gemma431BAdapter, Gemma4A4BAdapter
from mlx2.adapters.laguna_xs21 import LagunaXS21Adapter
from mlx2.adapters.mlx_vlm import Gemma3nAdapter, MiniCPMOAdapter
from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter
from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter
from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
from mlx2.adapters.xing import XingAdapter
from mlx2.serving import generation_stop_token_ids
from mlx2.structured_output import make_structured_processor


@pytest.mark.parametrize(
    "adapter_type",
    [
        FlashNextAdapter,
        Qwen3635BA3BAdapter,
        Qwen3827BAdapter,
        MuseGlimmerAdapter,
        NorthMiniCodeAdapter,
        LagunaXS21Adapter,
        XingAdapter,
        Gemma3nAdapter,
        Gemma4A4BAdapter,
        Gemma431BAdapter,
        MiniCPMOAdapter,
    ],
)
def test_structured_terminals_equal_generation_stops_for_every_adapter(adapter_type):
    """Every registered family crosses serving through one terminal contract."""
    adapter = object.__new__(adapter_type)
    pieces = ["x"] * 25
    pieces[2] = "<eos>"
    pieces[3] = "{"
    pieces[4] = "}"
    pieces[24] = "</assistant>"
    tokenizer = SimpleNamespace(
        eos_token_ids={2, 24},
        vocab_size=25,
        decode=lambda ids, **_kwargs: "".join(pieces[token] for token in ids),
        convert_ids_to_tokens=lambda ids: [pieces[token] for token in ids],
    )
    adapter.tokenizer = tokenizer
    stops = generation_stop_token_ids(adapter)
    processor = make_structured_processor(
        tokenizer,
        0,
        response_format={"type": "json_object"},
        generation_stop_token_ids=stops,
    )
    assert stops == (2, 24)
    assert processor.eos_ids == frozenset(stops)
