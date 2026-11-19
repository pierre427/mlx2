"""Anthropic Messages image and document blocks (Claude Code image ``Read``).

Before 2026-11-18 every image or document block was a 400, so a Claude Code
session on a VLM route failed the moment it read a screenshot.  Images now
cross the same multimodal contract as Chat ``image_url`` data URLs; documents
are served as their text.
"""

import base64
import io
import json
import threading
from collections import Counter
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mlx2.anthropic_compat import anthropic_request_to_chat
from mlx2.server import handler_for, validate_request

from test_anthropic_compat import AnthropicEngine


def _png_b64():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), (200, 30, 90)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


PNG = _png_b64()


def _image(data=PNG, media_type="image/png", **extra):
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
        **extra,
    }


def _chat(messages, counts=None):
    return anthropic_request_to_chat(
        {"model": "fixture", "max_tokens": 16, "messages": messages}, counts=counts
    )["messages"]


def test_user_text_and_image_keep_order_as_chat_parts():
    counts = Counter()
    messages = _chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before "},
                    {"type": "text", "text": "image:"},
                    _image(cache_control={"type": "ephemeral"}),
                    {"type": "text", "text": "after"},
                ],
            }
        ],
        counts,
    )
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
                {"type": "text", "text": "after"},
            ],
        }
    ]
    assert counts["anthropic_image_blocks_received"] == 1
    # The translated turn is a valid multimodal Chat request.
    validated = validate_request(
        {"model": "fixture", "max_tokens": 16, "messages": messages}, True
    )
    assert validated["messages"][0]["content"][1]["type"] == "image_url"


def test_all_text_user_turn_stays_a_string():
    assert _chat(
        [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    ) == [{"role": "user", "content": "ab"}]


def test_claude_code_read_image_tool_result_follows_contiguous_tool_turns():
    counts = Counter()
    messages = _chat(
        [
            {"role": "user", "content": "look at both"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_a", "name": "Read", "input": {"file_path": "/s.png"}},
                    {"type": "tool_use", "id": "toolu_b", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    # Claude Code's Read of an image: a tool_result whose
                    # only content is the image block.
                    {"type": "tool_result", "tool_use_id": "toolu_a", "content": [_image()]},
                    {"type": "tool_result", "tool_use_id": "toolu_b", "content": "s.png"},
                    {"type": "text", "text": "<system-reminder>x</system-reminder>"},
                ],
            },
        ],
        counts,
    )
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool", "user"]
    assert messages[2] == {"role": "tool", "tool_call_id": "toolu_a", "content": "[image result follows]"}
    assert messages[3] == {"role": "tool", "tool_call_id": "toolu_b", "content": "s.png"}
    assert messages[4]["content"] == [
        {"type": "text", "text": "[image from tool result toolu_a]"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
        {"type": "text", "text": "<system-reminder>x</system-reminder>"},
    ]
    assert counts["anthropic_tool_result_images_received"] == 1


def test_parallel_image_results_are_labelled_by_call():
    second = _png_b64()
    messages = _chat(
        [
            {"role": "user", "content": "compare"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_a", "name": "Read", "input": {}},
                    {"type": "tool_use", "id": "toolu_b", "name": "Read", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_a", "content": [_image(), _image(data=second)]},
                    {"type": "tool_result", "tool_use_id": "toolu_b", "content": [_image()]},
                ],
            },
        ]
    )
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool", "user"]
    labels = [part["text"] for part in messages[-1]["content"] if part["type"] == "text"]
    assert labels == [
        "[image 1 of 2 from tool result toolu_a]",
        "[image 2 of 2 from tool result toolu_a]",
        "[image from tool result toolu_b]",
    ]
    assert [part["type"] for part in messages[-1]["content"]] == ["text", "image_url"] * 3


def test_tool_result_text_and_image_keep_the_exact_text():
    messages = _chat(
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "shot", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "content": [{"type": "text", "text": "captured"}, _image()],
                    }
                ],
            },
        ]
    )
    assert messages[-2] == {"role": "tool", "tool_call_id": "t", "content": "captured"}
    assert messages[-1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "[image from tool result t]"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
        ],
    }


def test_mid_conversation_system_after_an_image_turn_keeps_alternation():
    from mlx2.agent_compat import AgentCompat

    body = anthropic_request_to_chat(
        {
            "model": "fixture",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": [_image(), {"type": "text", "text": "what"}]},
                {"role": "system", "content": "reminder"},
            ],
        },
        agent_compat=AgentCompat(enabled=True),
    )
    assert [m["role"] for m in body["messages"]] == ["user"]
    assert body["messages"][0]["content"][-1] == {"type": "text", "text": "\n\nreminder"}


def test_text_and_content_documents_render_as_text():
    counts = Counter()
    messages = _chat(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "title": "notes.txt",
                        "context": "from the repo",
                        "citations": {"enabled": False},
                        "source": {"type": "text", "media_type": "text/plain", "data": "alpha"},
                    },
                    {
                        "type": "document",
                        "source": {"type": "content", "content": [{"type": "text", "text": "beta"}]},
                    },
                    {"type": "text", "text": "summarize"},
                ],
            }
        ],
        counts,
    )
    assert messages == [
        {
            "role": "user",
            "content": (
                '\n<document title="notes.txt">\nfrom the repo\n\nalpha\n</document>\n'
                "\n<document>\nbeta\n</document>\n"
                "summarize"
            ),
        }
    ]
    assert counts["anthropic_document_blocks_received"] == 2


def test_pdf_document_uses_its_text_layer():
    pymupdf = pytest.importorskip("pymupdf")
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Quarterly figure 4217")
    data = base64.b64encode(pdf.tobytes()).decode()
    messages = _chat(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {"type": "base64", "media_type": "application/pdf", "data": data},
                    }
                ],
            }
        ]
    )
    assert "Quarterly figure 4217" in messages[0]["content"]
    assert messages[0]["content"].startswith("\n<document>\n## Page 1\n")


def _pdf_document(build):
    pymupdf = pytest.importorskip("pymupdf")
    pdf = pymupdf.open()
    build(pdf)
    data = base64.b64encode(pdf.tobytes()).decode()
    return {
        "type": "document",
        "cache_control": {"type": "ephemeral"},
        "source": {"type": "base64", "media_type": "application/pdf", "data": data},
    }


def test_pdf_over_the_page_limit_is_refused_not_truncated():
    def build(pdf):
        for index in range(101):
            pdf.new_page().insert_text((72, 72), f"page {index}")

    with pytest.raises(ValueError, match="has 101 pages; at most 100"):
        _chat([{"role": "user", "content": [_pdf_document(build)]}])


def test_scanned_pdf_page_is_refused_and_blank_page_is_not():
    from PIL import Image

    def scanned(pdf):
        pdf.new_page().insert_text((72, 72), "cover text")
        buffer = io.BytesIO()
        Image.new("RGB", (40, 40), (10, 10, 10)).save(buffer, format="PNG")
        pdf.new_page().insert_image((72, 72, 200, 200), stream=buffer.getvalue())

    with pytest.raises(ValueError, match="page 2 has no text layer"):
        _chat([{"role": "user", "content": [_pdf_document(scanned)]}])

    def with_blank(pdf):
        pdf.new_page().insert_text((72, 72), "only text")
        pdf.new_page()

    messages = _chat([{"role": "user", "content": [_pdf_document(with_blank)]}])
    assert "only text" in messages[0]["content"]
    assert "Page 2" not in messages[0]["content"]


@pytest.mark.parametrize(
    "block, message",
    [
        ({"type": "image", "source": {"type": "url", "url": "https://example.test/a.png"}}, "source type 'url' is unsupported"),
        ({"type": "image", "source": {"type": "file", "file_id": "file_1"}}, "source type 'file' is unsupported"),
        (_image(media_type="image/tiff"), "image media_type must be"),
        (_image(data=""), "image data must be nonempty text"),
        ({**_image(), "extra": 1}, "unsupported image block fields: extra"),
        (
            {"type": "document", "citations": {"enabled": True}, "source": {"type": "text", "media_type": "text/plain", "data": "x"}},
            "document citations are unsupported",
        ),
        (
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": base64.b64encode(b"not a pdf").decode()}},
            "document PDF data is not a PDF",
        ),
        (
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "%%%"}},
            "document PDF data is not valid base64",
        ),
        (
            {"type": "document", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
            "base64 document media_type must be application/pdf",
        ),
        (
            {"type": "document", "source": {"type": "url", "url": "https://example.test/a.pdf"}},
            "document source type 'url' is unsupported",
        ),
        (
            {"type": "document", "source": {"type": "content", "content": [_image()]}},
            "unsupported document content block fields: source",
        ),
    ],
)
def test_media_blocks_fail_closed(block, message):
    with pytest.raises(ValueError, match=message):
        _chat([{"role": "user", "content": [block]}])


def test_tool_result_media_blocks_fail_closed_too():
    with pytest.raises(ValueError, match="source type 'url' is unsupported"):
        _chat(
            [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "x", "input": {}}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t",
                            "content": [{"type": "image", "source": {"type": "url", "url": "https://example.test/a.png"}}],
                        }
                    ],
                },
            ]
        )
    with pytest.raises(ValueError, match="tool_result supports text, image and document blocks"):
        _chat(
            [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "x", "input": {}}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "search_result"}]}
                    ],
                },
            ]
        )


def test_assistant_media_blocks_stay_refused():
    with pytest.raises(ValueError, match="assistant image and document blocks are unsupported"):
        _chat(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [_image()]},
                {"role": "user", "content": "again"},
            ]
        )


class _MediaEngine(AnthropicEngine):
    def __init__(self, multimodal):
        super().__init__()
        self.multimodal = multimodal

    def supports_multimodal(self):
        return self.multimodal


@pytest.fixture(params=[True, False], ids=["vlm", "text"])
def media_endpoint(request):
    engine = _MediaEngine(request.param)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def test_http_image_reaches_a_vlm_route_and_fails_closed_on_a_text_route(media_endpoint):
    engine, base = media_endpoint
    body = {
        "model": "fixture",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": [_image(), {"type": "text", "text": "what is this"}]}],
    }
    request = Request(
        base + "/v1/messages",
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    if engine.multimodal:
        with urlopen(request) as response:
            payload = json.load(response)
        assert payload["type"] == "message"
        parts = engine.last_request["messages"][-1]["content"]
        assert parts[0] == {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}
        assert engine.counts["anthropic_image_blocks_received"] == 1
    else:
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        detail = json.load(error.value)
        assert error.value.code >= 400 and error.value.code != 500
        assert "image" in json.dumps(detail)
        assert engine.submit_calls == 0
        assert engine.counts["multimodal_rejected"] == 1


def test_http_count_tokens_refuses_media_instead_of_undercounting(media_endpoint):
    engine, base = media_endpoint
    body = {
        "model": "fixture",
        "messages": [{"role": "user", "content": [_image(), {"type": "text", "text": "hello"}]}],
    }
    request = Request(
        base + "/v1/messages/count_tokens",
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    detail = json.dumps(json.load(error.value))
    if engine.multimodal:
        assert error.value.code == 400
        assert "count_tokens does not count image" in detail
    else:
        assert error.value.code not in (400, 500)
        assert "no qualified image" in detail
    assert engine.count_tokens_calls == 0
