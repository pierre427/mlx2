"""CPU-only preflight for the pinned offline LFM2.5-VL DSpark pair.

This checks the same target object that the pinned DSpark ``bind`` and DFlash
round loop receive.  It does not make DSpark a serving capability or prove
speculative numerical parity.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager


def _value(config, name):
    return config.get(name) if isinstance(config, Mapping) else getattr(config, name, None)


def require_offline_dspark_target(target_model, draft_config: Mapping) -> None:
    """Reject a structurally incompatible target before loading draft weights.

    The pinned drafter compatibility check reads ``target.language_model``;
    the pinned DFlash round loop also calls its rollback method.  A VL wrapper
    that merely contains a rollback-capable inner model is insufficient.
    """
    if not isinstance(draft_config, Mapping):
        raise TypeError("DSpark draft configuration must be an object")
    dflash = draft_config.get("dflash_config")
    if not isinstance(dflash, Mapping):
        raise ValueError("DSpark draft configuration lacks dflash_config")

    language_model = getattr(target_model, "language_model", target_model)
    config = getattr(language_model, "config", None)
    text_config = _value(config, "text_config") or config
    inner = getattr(language_model, "model", language_model)
    layers = getattr(inner, "layers", None)
    layer_count = len(layers) if layers is not None else _value(text_config, "num_hidden_layers")
    expected = {
        "hidden_size": (draft_config.get("hidden_size"), _value(text_config, "hidden_size")),
        "num_target_layers": (dflash.get("num_target_layers"), layer_count),
        "vocab_size": (draft_config.get("vocab_size"), _value(text_config, "vocab_size")),
    }
    for name, (draft_value, target_value) in expected.items():
        if type(draft_value) is not int or draft_value != target_value:
            raise ValueError(
                f"DSpark offline target {name} mismatch: draft={draft_value!r}, "
                f"target={target_value!r}"
            )

    if not callable(getattr(language_model, "rollback_speculative_cache", None)):
        raise ValueError(
            "DSpark offline target language_model lacks "
            "rollback_speculative_cache; the pinned LFM2.5-VL wrapper "
            "cannot commit rejected convolution state"
        )

    embeddings = getattr(inner, "embed_tokens", None)
    head = getattr(target_model, "lm_head", None) or getattr(language_model, "lm_head", None)
    if embeddings is None or (head is None and not callable(getattr(embeddings, "as_linear", None))):
        raise ValueError("DSpark offline target has no bindable token embedding/head")


def install_offline_dspark_bridge(target_model, *, vl_language_class,
                                  text_language_class, verifier, cache_classes,
                                  output_class):
    """Give the pinned VL target the text model's exact offline draft contract.

    The wrapper is installed only on this already loaded target.  It leaves
    parameters and ordinary calls on the original VL language model, and
    refuses any cache outside the pinned mlx-vlm classes.  mlx2 serving uses
    different checkpoint-aware cache classes and never calls this bridge.
    """
    language = getattr(target_model, "language_model", None)
    if language is None or type(language) is not vl_language_class:
        raise ValueError("DSpark offline bridge requires the pinned VL language class")
    if getattr(language.config, "num_hidden_layers", None) != 30:
        raise ValueError("DSpark offline bridge requires the 30-layer LFM target")
    layers = getattr(getattr(language, "model", None), "layers", None)
    if layers is None or len(layers) != 30 or sum(bool(layer.is_attention_layer) for layer in layers) != 8:
        raise ValueError("DSpark offline bridge requires 22 ShortConv and eight KV layers")
    if not callable(getattr(text_language_class, "rollback_speculative_cache", None)):
        raise ValueError("pinned LFM text rollback contract is absent")
    if not callable(getattr(text_language_class, "_restore_conv_cache", None)):
        raise ValueError("pinned LFM text convolution rollback contract is absent")
    array_cache, kv_cache = cache_classes

    class _TiedVLArgs:
        # The pinned VL language model always uses embed_tokens.as_linear but
        # its TextConfig has no tie_word_embeddings field.  The text verifier
        # needs this explicit fact when selecting the exact output head.
        tie_word_embeddings = True

        def __init__(self, config):
            self._config = config

        def __getattr__(self, name):
            return getattr(self._config, name)

    def check_cache(cache):
        if not isinstance(cache, list) or len(cache) != 30:
            raise ValueError("DSpark offline target requires a 30-plane cache")
        for index, (entry, layer) in enumerate(zip(cache, layers, strict=True)):
            expected = kv_cache if layer.is_attention_layer else array_cache
            if type(entry) is not expected:
                raise ValueError(f"DSpark offline cache class mismatch at layer {index}")

    # Check the source cache factory before changing the loaded model.  The
    # speculative loop later creates its cache from this same factory.
    check_cache(language.make_cache())

    original_call = vl_language_class.__call__

    def bridge_call(self, inputs, mask=None, cache=None, inputs_embeds=None, **kwargs):
        capture = kwargs.pop("capture_layer_ids", None)
        exact = kwargs.pop("speculative_verify", False)
        chunk_size = kwargs.pop("n_to_process", None)
        if chunk_size is not None and (type(chunk_size) is not int or
                                       chunk_size != inputs.shape[1]):
            raise ValueError("DSpark prefill chunk length mismatch")
        if exact is not False and exact is not True:
            raise ValueError("DSpark speculative_verify must be a boolean")
        if kwargs:
            raise ValueError(f"unsupported DSpark target arguments: {sorted(kwargs)}")
        if cache is not None and (capture is not None or exact):
            check_cache(cache)
        if exact:
            if capture is None or len(capture) != len(set(capture)) or any(
                type(index) is not int or index < 0 or index >= 30 for index in capture
            ):
                raise ValueError("DSpark exact verification requires distinct target layers")
            result = verifier(self, inputs, cache=cache,
                              input_embeddings=inputs_embeds,
                              capture_layer_ids=capture)
            if result.hidden_states is None or len(result.hidden_states) != len(capture):
                raise RuntimeError("DSpark exact verifier did not capture target layers")
            if result.gdn_states is None or len(result.gdn_states) != 22:
                raise RuntimeError("DSpark exact verifier did not capture all ShortConv states")
            return result
        if capture is None:
            return original_call(self, inputs, mask=mask, cache=cache,
                                 inputs_embeds=inputs_embeds)
        if len(capture) != len(set(capture)) or any(
            type(index) is not int or index < 0 or index >= 30 for index in capture
        ):
            raise ValueError("DSpark capture requires distinct target layers")
        hidden = []
        output = self.model(inputs, cache, inputs_embeds,
                            capture_layer_ids=capture, hidden_sink=hidden)
        if len(hidden) != len(capture):
            raise RuntimeError("DSpark prefill did not capture target layers")
        return output_class(self.model.embed_tokens.as_linear(output),
                            hidden_states=hidden)

    def bridge_rollback(self, caches, gdn_states, accepted, block_size):
        check_cache(caches)
        if gdn_states is None or len(gdn_states) != 22:
            raise ValueError("DSpark rollback requires all 22 ShortConv snapshots")
        indices = [row[0] for row in gdn_states]
        expected = [i for i, layer in enumerate(layers) if not layer.is_attention_layer]
        if indices != expected:
            raise ValueError("DSpark rollback ShortConv layer order mismatch")
        return text_language_class.rollback_speculative_cache(
            self, caches, gdn_states, accepted, block_size
        )

    bridge_type = type(
        "OfflineLFM25DSparkLanguageBridge", (vl_language_class,),
        {
            "__call__": bridge_call,
            "rollback_speculative_cache": bridge_rollback,
            "_restore_conv_cache": staticmethod(text_language_class._restore_conv_cache),
            "args": property(lambda self: _TiedVLArgs(self.config)),
        },
    )
    try:
        language.__class__ = bridge_type
    except TypeError as exc:
        raise ValueError("pinned VL language instance cannot be bridged") from exc
    return target_model


@contextmanager
def offline_dspark_target_transaction(target_model):
    """Restore the original target class if draft binding or loading fails."""
    language = target_model.language_model
    original_class = type(language)
    try:
        yield
    except Exception:
        if type(language) is not original_class:
            language.__class__ = original_class
        raise


__all__ = [
    "install_offline_dspark_bridge", "offline_dspark_target_transaction",
    "require_offline_dspark_target",
]
