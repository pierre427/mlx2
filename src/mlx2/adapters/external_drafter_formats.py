"""Pick an external drafter artifact's inspect/pin/load functions by its config.

A target adapter whose external route verifies DFlash-backbone heads calls
:func:`drafter_format` with ``draft_model`` instead of importing one format.
The choice is keyed on the artifact's declared ``architectures``, never on a
model name.  An unreadable or unknown config keeps the DFlash2 format, whose
own inspector then reports the artifact's defect exactly as before.
"""

import json
from pathlib import Path
from types import SimpleNamespace


def _dflash2():
    from . import dflash2

    return SimpleNamespace(
        name="dflash2",
        inspect_drafter=dflash2.inspect_drafter,
        content_revision=dflash2.content_revision,
        legacy_content_revision=dflash2._legacy_content_revision,
        validate_runtime_quantization=dflash2.validate_runtime_quantization,
        load_drafter=dflash2.load_drafter,
    )


def _dspark():
    from . import dflash2, dspark

    return SimpleNamespace(
        name="dspark",
        inspect_drafter=dspark.inspect_drafter,
        content_revision=dspark.content_revision,
        legacy_content_revision=dspark.legacy_content_revision,
        validate_runtime_quantization=dflash2.validate_runtime_quantization,
        load_drafter=dspark.load_drafter,
    )


def _eagle4():
    from . import eagle4

    return SimpleNamespace(
        name="eagle4",
        inspect_drafter=eagle4.inspect_drafter,
        content_revision=eagle4.content_revision,
        legacy_content_revision=eagle4.legacy_content_revision,
        validate_runtime_quantization=eagle4.validate_runtime_quantization,
        load_drafter=eagle4.load_drafter,
    )


_FORMATS = {
    "Qwen3DSparkModel": _dspark,
    "LlamaForCausalLMEagle4": _eagle4,
}


def format_for_config(config):
    architectures = config.get("architectures") if isinstance(config, dict) else None
    if isinstance(architectures, list) and len(architectures) == 1:
        factory = _FORMATS.get(architectures[0])
        if factory is not None:
            return factory()
    return _dflash2()


def drafter_format(draft_model):
    """Format of the artifact at ``draft_model`` (DFlash2 when undeterminable)."""
    try:
        config = json.loads(
            (Path(str(draft_model)).expanduser() / "config.json").read_text()
        )
    except (OSError, ValueError, TypeError):
        return _dflash2()
    return format_for_config(config)


def drafter_format_for_record(record):
    """Format of an inspected record (its bound config, not a re-read)."""
    return format_for_config(record.get("config"))


__all__ = ["drafter_format", "drafter_format_for_record", "format_for_config"]
