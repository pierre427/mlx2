"""Fail-closed chat-template rendering for pinned third-party processors."""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from types import MethodType

_UNSANDBOXED_MLX_VLM_PROCESSORS = frozenset(
    {
        "mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl",
        "mlx_vlm.models.kimi_vl.processing_kimi_vl",
        "mlx_vlm.models.locateanything.processing_locateanything",
        "mlx_vlm.models.molmo.processing_molmo",
        "mlx_vlm.models.molmo2.processing",
        "mlx_vlm.models.phi3_v.processing_phi3_v",
    }
)


@lru_cache(maxsize=128)
def _compile_sandboxed(template: str):
    try:
        import jinja2
        from jinja2.ext import LoopControlExtension
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as error:  # pragma: no cover - serving extra owns Jinja
        raise RuntimeError("chat-template rendering requires jinja2") from error
    if not isinstance(template, str):
        raise TypeError("chat template must be a string")

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[LoopControlExtension],
    )
    environment.filters["tojson"] = tojson
    environment.globals["raise_exception"] = raise_exception
    # Match Transformers' helper: model templates expect host-local wall time.
    environment.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)  # noqa: DTZ005
    return environment.from_string(template)


def _template_for(processor, override):
    if override is not None:
        return override
    template = getattr(processor, "chat_template", None)
    if template is not None:
        return template
    return getattr(getattr(processor, "tokenizer", None), "chat_template", None)


def secure_model_chat_templates(processor):
    """Install a sandbox on vulnerable renderers in the pinned mlx-vlm tree.

    Hugging Face tokenizer/processor renderers already compile with an
    ``ImmutableSandboxedEnvironment``. Six custom renderers in the mlx-vlm
    revision pinned by mlx2 bypass that compiler; replace only those instance
    methods, preserving their argument and tokenization contract. A trusted
    built-in fallback remains with the original method when no model template
    exists.
    """
    if processor is None or getattr(processor, "_mlx2_chat_template_sandbox", False):
        return processor
    method = getattr(processor, "apply_chat_template", None)
    module = getattr(getattr(method, "__func__", method), "__module__", "")
    if module not in _UNSANDBOXED_MLX_VLM_PROCESSORS:
        return processor
    original = method

    def apply_chat_template(
        self,
        conversation,
        chat_template=None,
        add_generation_prompt=False,
        tokenize=False,
        **kwargs,
    ):
        template = _template_for(self, chat_template)
        if template is None:
            return original(
                conversation,
                chat_template=None,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                **kwargs,
            )
        rendered = _compile_sandboxed(template).render(
            messages=conversation,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        if tokenize:
            return self.tokenizer.encode(rendered)
        return rendered

    try:
        processor.apply_chat_template = MethodType(apply_chat_template, processor)
        processor._mlx2_chat_template_sandbox = True
    except (AttributeError, TypeError) as error:
        raise RuntimeError(
            f"cannot sandbox chat-template renderer {type(processor).__module__}."
            f"{type(processor).__qualname__}"
        ) from error
    return processor
