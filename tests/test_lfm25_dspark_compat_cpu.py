"""DSpark offline binding preflight without importing or using real MLX."""

import importlib.abc
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError(f"real MLX import forbidden: {fullname}")
        return None


sys.meta_path.insert(0, BlockMLX())
from mlx2.adapters.lfm25_dspark_compat import (  # noqa: E402
    install_offline_dspark_bridge, require_offline_dspark_target,
    offline_dspark_target_transaction,
)


DRAFT_CONFIG = {
    "hidden_size": 2048,
    "vocab_size": 128000,
    "dflash_config": {"num_target_layers": 30, "target_layer_ids": [2, 9, 17, 21, 27]},
}


class VLTarget:
    """The pinned VL model's object layout, including its missing rollback."""

    def __init__(self, *, rollback=False, layers=30, hidden=2048):
        embeddings = SimpleNamespace(as_linear=lambda x: x)
        inner = SimpleNamespace(layers=[object() for _ in range(layers)], embed_tokens=embeddings)
        text_config = SimpleNamespace(hidden_size=hidden, vocab_size=128000,
                                      num_hidden_layers=layers)
        self.language_model = SimpleNamespace(config=text_config, model=inner)
        if rollback:
            self.language_model.rollback_speculative_cache = lambda *args: None


class OfflineDSparkCompatibilityTest(unittest.TestCase):
    def test_current_pinned_vl_wrapper_fails_before_source_or_draft_weights(self):
        target = VLTarget()
        with self.assertRaisesRegex(ValueError, "lacks rollback_speculative_cache"):
            require_offline_dspark_target(target, DRAFT_CONFIG)

    def test_compatible_fake_passes_structural_preflight(self):
        require_offline_dspark_target(VLTarget(rollback=True), DRAFT_CONFIG)

    def test_wrong_target_geometry_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "hidden_size mismatch"):
            require_offline_dspark_target(VLTarget(rollback=True, hidden=4096), DRAFT_CONFIG)
        with self.assertRaisesRegex(ValueError, "num_target_layers mismatch"):
            require_offline_dspark_target(VLTarget(rollback=True, layers=29), DRAFT_CONFIG)


if __name__ == "__main__":
    unittest.main()


class _ArrayCache:
    def __init__(self):
        self.tokens = []


class _KVCache:
    def __init__(self):
        self.tokens = []


class _FakeInner:
    def __init__(self):
        self.layers = [SimpleNamespace(is_attention_layer=i in {2, 9, 17, 21, 27, 28, 29, 1})
                       for i in range(30)]
        self.embed_tokens = SimpleNamespace(as_linear=lambda value: value)

    def __call__(self, inputs, cache, embeddings, *, capture_layer_ids, hidden_sink):
        hidden_sink.extend((index, inputs) for index in capture_layer_ids)
        return inputs


class _FakeVL:
    def __init__(self):
        self.config = SimpleNamespace(num_hidden_layers=30, hidden_size=2048,
                                      vocab_size=128000, conv_L_cache=3)
        self.model = _FakeInner()

    def __call__(self, inputs, mask=None, cache=None, inputs_embeds=None):
        return SimpleNamespace(logits=inputs, hidden_states=None, gdn_states=None)

    def make_cache(self):
        return [_KVCache() if layer.is_attention_layer else _ArrayCache()
                for layer in self.model.layers]


class _FakeText:
    _restore_conv_cache = staticmethod(lambda *args: None)

    def rollback_speculative_cache(self, caches, snapshots, accepted, block_size):
        for cache, original in zip(caches, self._offline_original, strict=True):
            cache.tokens = original + self._offline_verify_tokens[:accepted + 1]
        return accepted


def _fake_verifier(language, inputs, *, cache, input_embeddings, capture_layer_ids):
    language._offline_original = [list(item.tokens) for item in cache]
    language._offline_verify_tokens = list(inputs)
    for item in cache:
        item.tokens.extend(inputs)
    snapshots = [(index, tuple(language._offline_original[index]))
                 for index, layer in enumerate(language.model.layers)
                 if not layer.is_attention_layer]
    return SimpleNamespace(logits=inputs,
                           hidden_states=[(index, inputs) for index in capture_layer_ids],
                           gdn_states=snapshots)


class OfflineDSparkBridgeTest(unittest.TestCase):
    def make_target(self):
        target = SimpleNamespace(language_model=_FakeVL())
        install_offline_dspark_bridge(
            target, vl_language_class=_FakeVL, text_language_class=_FakeText,
            verifier=_fake_verifier, cache_classes=(_ArrayCache, _KVCache),
            output_class=lambda logits, hidden_states: SimpleNamespace(
                logits=logits, hidden_states=hidden_states),
        )
        cache = target.language_model.make_cache()
        for item in cache:
            item.tokens = [10, 11]
        return target, cache

    def test_capture_and_accepted_or_rejected_hybrid_state(self):
        for accepted in (0, 2):
            target, cache = self.make_target()
            language = target.language_model
            ordinary = language([7], cache=cache)
            self.assertEqual(ordinary.logits, [7])
            captured = language([7], cache=cache,
                                capture_layer_ids=[2, 9, 17, 21, 27])
            self.assertEqual(len(captured.hidden_states), 5)
            verified = language([20, 21, 22, 23], cache=cache,
                                capture_layer_ids=[2, 9, 17, 21, 27],
                                speculative_verify=True)
            self.assertEqual(len(verified.gdn_states), 22)
            self.assertEqual(sum(isinstance(item, _KVCache) for item in cache), 8)
            language.rollback_speculative_cache(cache, verified.gdn_states,
                                                accepted, 4)
            for item in cache:
                self.assertEqual(item.tokens, [10, 11, *[20, 21, 22][:accepted + 1]])

    def test_wrong_cache_and_missing_shortconv_snapshots_fail_closed(self):
        target, cache = self.make_target()
        cache[0] = object()
        with self.assertRaisesRegex(ValueError, "cache class mismatch"):
            target.language_model([1, 2], cache=cache,
                                  capture_layer_ids=[2], speculative_verify=True)
        target, cache = self.make_target()
        with self.assertRaisesRegex(ValueError, "all 22"):
            target.language_model.rollback_speculative_cache(cache, [], 0, 4)

    def test_chunked_prefill_capture_checks_length(self):
        target, cache = self.make_target()
        chunk = SimpleNamespace(shape=(1, 3))
        output = target.language_model(chunk, cache=cache,
                                       capture_layer_ids=[2, 9, 17, 21, 27],
                                       n_to_process=3)
        self.assertEqual(len(output.hidden_states), 5)
        with self.assertRaisesRegex(ValueError, "chunk length mismatch"):
            target.language_model(chunk, cache=cache,
                                  capture_layer_ids=[2], n_to_process=2)

    def test_wrong_factory_class_fails_before_target_mutation(self):
        target = SimpleNamespace(language_model=_FakeVL())
        target.language_model.make_cache = lambda: [object()] * 30
        with self.assertRaisesRegex(ValueError, "cache class mismatch"):
            install_offline_dspark_bridge(
                target, vl_language_class=_FakeVL, text_language_class=_FakeText,
                verifier=_fake_verifier,
                cache_classes=(_ArrayCache, _KVCache), output_class=object,
            )
        self.assertIs(type(target.language_model), _FakeVL)

    def test_failed_draft_bind_restores_loaded_target_class(self):
        target = SimpleNamespace(language_model=_FakeVL())
        with self.assertRaisesRegex(RuntimeError, "draft bind failed"):
            with offline_dspark_target_transaction(target):
                install_offline_dspark_bridge(
                    target, vl_language_class=_FakeVL,
                    text_language_class=_FakeText, verifier=_fake_verifier,
                    cache_classes=(_ArrayCache, _KVCache), output_class=object,
                )
                raise RuntimeError("draft bind failed")
        self.assertIs(type(target.language_model), _FakeVL)
