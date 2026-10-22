"""Pick an external drafter artifact's inspect/pin/load functions by its config.

A target adapter whose external route verifies DFlash-backbone heads calls
:func:`drafter_format` with ``draft_model`` instead of importing one format.
The choice is keyed on the artifact's declared ``architectures``, never on a
model name.  An unreadable or unknown config keeps the DFlash2 format, whose
own inspector then reports the artifact's defect exactly as before.

Ling's UBTree is a separate format: it uses a DFlash (not DFlash2) backbone
and target-embedding factorizer MLPs, so it must never enter the DFlash2
weight schema by fallback.
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


def _ubtree():
    from . import ubtree

    return SimpleNamespace(
        name="ubtree",
        inspect_drafter=ubtree.inspect_drafter,
        content_revision=ubtree.content_revision,
        legacy_content_revision=ubtree.legacy_content_revision,
        validate_runtime_quantization=ubtree.validate_runtime_quantization,
        load_drafter=ubtree.load_drafter,
    )


_FORMATS = {
    "Qwen3DSparkModel": _dspark,
    "Lfm2DSparkDraftModel": _dspark,
    "LlamaForCausalLMEagle4": _eagle4,
    "Ling3UBTreeModel": _ubtree,
}
_FORMAT_NAMES = {"Qwen3DSparkModel": "dspark", "LlamaForCausalLMEagle4": "eagle4"}


def _architecture(config):
    architectures = config.get("architectures") if isinstance(config, dict) else None
    if isinstance(architectures, list) and len(architectures) == 1:
        return architectures[0]
    return None


def format_for_config(config):
    factory = _FORMATS.get(_architecture(config))
    return factory() if factory is not None else _dflash2()


def drafter_format_name(draft_model):
    """Format name of the artifact at ``draft_model``, importing no format module.

    Safe before an adapter pins its import-time environment (CLI policy
    resolution): the EAGLE-4 format module imports ``runtime.models``, so
    resolving it through :func:`drafter_format` at that point made the
    Qwen3.8 27B adapter refuse to load with ImportOrderError.
    """
    try:
        config = json.loads(
            (Path(str(draft_model)).expanduser() / "config.json").read_text()
        )
    except (OSError, ValueError, TypeError):
        return "dflash2"
    return _FORMAT_NAMES.get(_architecture(config), "dflash2")


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


__all__ = ["drafter_format", "drafter_format_for_record", "drafter_format_name", "format_for_config"]
