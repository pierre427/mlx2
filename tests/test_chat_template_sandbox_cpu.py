from types import SimpleNamespace

import pytest
from jinja2 import Template

from mlx2.runtime.chat_templates import secure_model_chat_templates


class _Tokenizer:
    chat_template = None

    @staticmethod
    def encode(text):
        return [ord(value) for value in text]


def _processor(module, template):
    def apply_chat_template(
        self,
        conversation,
        chat_template=None,
        add_generation_prompt=False,
        tokenize=False,
        **kwargs,
    ):
        raise AssertionError("the unsandboxed renderer must not run")

    apply_chat_template.__module__ = module
    return SimpleNamespace(
        tokenizer=_Tokenizer(),
        chat_template=template,
        apply_chat_template=apply_chat_template,
    )


@pytest.mark.parametrize(
    "module",
    [
        "mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl",
        "mlx_vlm.models.kimi_vl.processing_kimi_vl",
        "mlx_vlm.models.locateanything.processing_locateanything",
        "mlx_vlm.models.molmo.processing_molmo",
        "mlx_vlm.models.molmo2.processing",
        "mlx_vlm.models.phi3_v.processing_phi3_v",
    ],
)
def test_pinned_unsandboxed_processors_use_shared_sandbox(module):
    processor = _processor(
        module,
        "{% for message in messages %}{{ message.role }}={{ message.content }};{% endfor %}"
        "{% if add_generation_prompt %}assistant={% endif %}",
    )
    secure_model_chat_templates(processor)
    messages = [{"role": "user", "content": "hello"}]
    rendered = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    expected = Template(processor.chat_template).render(
        messages=messages, add_generation_prompt=True
    )
    assert rendered == expected == "user=hello;assistant="
    assert processor.apply_chat_template(messages, tokenize=True) == [
        ord(value) for value in "user=hello;"
    ]


@pytest.mark.parametrize(
    "template",
    [
        "{{ messages.__class__.__mro__ }}",
        "{{ messages.__class__.__subclasses__() }}",
    ],
)
def test_hostile_attribute_traversal_is_rejected(template):
    processor = _processor("mlx_vlm.models.kimi_vl.processing_kimi_vl", template)
    secure_model_chat_templates(processor)
    with pytest.raises(Exception) as caught:
        processor.apply_chat_template([], tokenize=False)
    assert caught.type.__module__.startswith("jinja2")
    assert caught.type.__name__ == "SecurityError"


def test_huggingface_renderer_is_not_replaced():
    calls = []

    def renderer(*args, **kwargs):
        calls.append((args, kwargs))
        return "kept"

    renderer.__module__ = "transformers.tokenization_utils_base"
    processor = SimpleNamespace(apply_chat_template=renderer)
    assert secure_model_chat_templates(processor) is processor
    assert processor.apply_chat_template([]) == "kept"
    assert calls


def test_sandbox_preserves_loopcontrols_and_huggingface_helpers():
    processor = _processor(
        "mlx_vlm.models.molmo2.processing",
        "{% for message in messages %}{% if not message.content %}{% continue %}{% endif %}"
        "{{ message.content|tojson }}{% endfor %}|{{ strftime_now('%Y') }}",
    )
    secure_model_chat_templates(processor)
    rendered = processor.apply_chat_template(
        [{"content": ""}, {"content": "<ok>"}], tokenize=False
    )
    value, year = rendered.split("|")
    assert value == '"<ok>"'
    assert year.isdigit() and len(year) == 4


def test_sandbox_preserves_raise_exception_helper():
    processor = _processor(
        "mlx_vlm.models.phi3_v.processing_phi3_v",
        "{{ raise_exception('bad template input') }}",
    )
    secure_model_chat_templates(processor)
    with pytest.raises(Exception, match="bad template input") as caught:
        processor.apply_chat_template([], tokenize=False)
    assert caught.type.__module__.startswith("jinja2")
