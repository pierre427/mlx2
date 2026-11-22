"""CPU checks for the opt-in TRAC persistence sampler (fake denoiser, no model)."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from mlx_blocker import install_for_test_case
from test_diffusion_gemma_cpu import fixture

from mlx2.adapters import diffusion_trac
from mlx2.adapters.diffusion_gemma import DiffusionGemmaAdapter
from mlx2.adapters.diffusion_trac import (
    TRACPersistenceConfig, TRACPersistenceState, approx_topk_jsd,
    confidence_transfer_mask, simulate_generate,
)

V = 16


def dist(token, p):
    row = np.full(V, (1.0 - p) / (V - 1))
    row[token] = p
    return row


def paced(pacers, specials):
    """Canvas = specials then pacers.  Pacer j is high (0.95) from step j+1 on
    and otherwise low with a changing argmax, so exactly one new pacer crosses
    the threshold per step and the force-one rule stays idle while they last.
    ``specials``: list of callables step -> probability row."""

    def denoise(block, step, canvas):
        rows = [f(step) for f in specials]
        for j in range(pacers):
            rows.append(dist(j % V, 0.95) if step >= j + 1 else dist((step + j) % V, 0.2))
        return np.stack(rows)

    return denoise


class TRACPolicyTest(unittest.TestCase):
    def setUp(self):
        install_for_test_case(self)

    def test_positions_commit_at_the_persistence_step(self):
        specials = [
            lambda s: dist(3, 0.95),                      # high: step 1 in both
            lambda s: dist(7, 0.6),                       # stable medium
            lambda s: dist(9 if s == 1 else 11, 0.55),    # changes at step 2
        ]
        args = dict(max_tokens=9, canvas_length=9, max_denoising_steps=12)
        base = simulate_generate(paced(6, specials), **args)
        trac = simulate_generate(paced(6, specials), trac=TRACPersistenceConfig(), **args)
        # k = 2 stable transitions: same top-1 on steps 1,2,3 -> commit at 3;
        # position 2 is stable from step 2 -> transitions 2->3, 3->4 -> step 4.
        self.assertEqual(trac.commit_step[0][:3], [1, 3, 4])
        self.assertEqual(trac.early_commits, 2)
        # Baseline: the force-one rule takes them only once pacers run out.
        self.assertEqual(base.commit_step[0][:3], [1, 7, 8])
        self.assertEqual(base.forwards, 8)
        self.assertEqual(trac.forwards, 6)
        self.assertEqual(trac.tokens, base.tokens)
        # A single stable transition is enough with stable_steps=1.
        one = simulate_generate(paced(6, specials), trac=TRACPersistenceConfig(stable_steps=1), **args)
        self.assertEqual(one.commit_step[0][:3], [1, 2, 3])

    def test_low_confidence_and_jsd_gate(self):
        below = [lambda s: dist(3, 0.95), lambda s: dist(7, 0.45)]
        args = dict(max_tokens=8, canvas_length=8, max_denoising_steps=12)
        out = simulate_generate(paced(6, below), trac=TRACPersistenceConfig(), **args)
        self.assertEqual(out.early_commits, 0)
        # Same top-1, but the runner-up mass moves: JSD rejects, persistence-only admits.
        def drifting(s):
            row = np.zeros(V)
            row[7] = 0.55
            row[(s % 2) + 1] = 0.45
            return row
        spec = [lambda s: dist(3, 0.95), drifting]
        self.assertGreater(approx_topk_jsd([[7, 1]], [[0.55, 0.45]], [[7, 2]], [[0.55, 0.45]])[0], 0.04)
        gated = simulate_generate(paced(6, spec), trac=TRACPersistenceConfig(), **args)
        loose = simulate_generate(paced(6, spec), trac=TRACPersistenceConfig(jsd_epsilon=None), **args)
        self.assertEqual(gated.early_commits, 0)
        self.assertEqual(loose.commit_step[0][1], 3)

    def test_disabled_policy_is_bit_identical_to_baseline(self):
        rng = np.random.default_rng(1121)
        for trial in range(25):
            L = int(rng.integers(2, 12))
            steps = int(rng.integers(1, 10))
            table = rng.dirichlet(np.full(V, 0.3), size=(3, steps + 1, L))

            def denoise(block, step, canvas, table=table):
                return table[block % 3, step - 1]

            args = dict(max_tokens=int(rng.integers(1, 3 * L)), canvas_length=L,
                        max_denoising_steps=steps, threshold=float(rng.uniform(0.3, 0.95)))
            base = simulate_generate(denoise, **args)
            off = simulate_generate(denoise, trac=TRACPersistenceConfig(low_threshold=1.0, stable_steps=48), **args)
            self.assertEqual((base.tokens, base.forwards, base.commit_step),
                             (off.tokens, off.forwards, off.commit_step))
            self.assertEqual(off.early_commits, 0)

    def test_forwards_drop_on_a_stable_sequence(self):
        L = 32
        targets = np.arange(L) % V

        def denoise(block, step, canvas):
            return np.stack([dist(int(t), 0.7) for t in targets])

        args = dict(max_tokens=64, canvas_length=L, max_denoising_steps=48)
        base = simulate_generate(denoise, **args)
        trac = simulate_generate(denoise, trac=TRACPersistenceConfig(), **args)
        # Baseline force-commits one position per step; TRAC commits the rest at step 3.
        self.assertEqual(base.steps_per_block, [32, 32])
        self.assertEqual(trac.steps_per_block, [3, 3])
        self.assertLess(trac.forwards, base.forwards)
        self.assertEqual(trac.tokens, base.tokens)

    def test_never_commits_mask_token_early(self):
        spec = [lambda s: dist(3, 0.95), lambda s: dist(5, 0.6)]
        args = dict(max_tokens=8, canvas_length=8, max_denoising_steps=12)
        cfg = TRACPersistenceConfig(mask_token_id=5)
        out = simulate_generate(paced(6, spec), trac=cfg, **args)
        base = simulate_generate(paced(6, spec), **args)
        self.assertEqual(out.early_commits, 0)
        self.assertEqual(out.commit_step, base.commit_step)
        state = TRACPersistenceState(cfg, 2)
        for _ in range(4):
            state.observe([5, 6], [[5, 6], [6, 5]], [[0.6, 0.1], [0.6, 0.1]])
        mask = state.early_mask([True, True], [5, 6], [5, 6], [0.6, 0.6])
        self.assertEqual(mask.tolist(), [False, True])
        # Revealed positions and a sampled non-argmax denoiser token are never early commits.
        self.assertEqual(state.early_mask([True, False], [5, 6], [5, 6], [0.6, 0.6]).tolist(), [False, False])
        self.assertEqual(state.early_mask([True, True], [5, 7], [5, 6], [0.6, 0.6]).tolist(), [False, False])

    def test_block_and_length_limits(self):
        L, steps = 4, 5

        def denoise(block, step, canvas):
            # Position 0 is stable medium; 2 and 3 outrank it for the force-one
            # rule but change their argmax every step.
            return np.stack([dist(1, 0.6), dist(2, 0.95), dist((4 + step) % V, 0.65),
                             dist((9 + step) % V, 0.65)])

        out = simulate_generate(denoise, max_tokens=10, canvas_length=L, max_denoising_steps=steps,
                                trac=TRACPersistenceConfig())
        self.assertEqual(len(out.tokens), 10)
        self.assertEqual(len(out.steps_per_block), 3)
        self.assertTrue(all(1 <= s <= steps for s in out.steps_per_block))
        self.assertEqual(out.forwards, sum(out.steps_per_block))
        # Streaks reset per block: the stable position commits early at step 3 in
        # every block, never at step 1 or 2 carried over from the previous block.
        for block in out.commit_step:
            self.assertEqual(len(block), L)
            self.assertEqual(block[0], 3)
        with self.assertRaises(ValueError):
            TRACPersistenceState(TRACPersistenceConfig(), 3).observe([1, 2])

    def test_transfer_mask_mirrors_force_one_rule(self):
        conf = np.array([0.3, 0.6, 0.2])
        unrev = np.array([True, True, True])
        self.assertEqual(confidence_transfer_mask(conf, unrev, 0.9).tolist(), [False, True, False])
        self.assertEqual(confidence_transfer_mask(conf, unrev, 0.9, early=np.array([True, False, False])).tolist(),
                         [True, False, False])
        self.assertEqual(confidence_transfer_mask(conf, unrev, 0.9, force_all=True).tolist(), [True] * 3)

    def test_config_validation(self):
        for bad in (dict(low_threshold=0), dict(low_threshold=1.5), dict(stable_steps=0),
                    dict(stable_steps=True), dict(jsd_top_k=0), dict(jsd_epsilon=1.0),
                    dict(mask_token_id=-1)):
            with self.assertRaises(ValueError):
                TRACPersistenceConfig(**bad)
        receipt = TRACPersistenceConfig().as_dict()
        self.assertFalse(receipt["qualified"])
        self.assertEqual((receipt["low_threshold"], receipt["stable_steps"], receipt["jsd_top_k"],
                          receipt["jsd_epsilon"]), (0.5, 2, 8, 0.04))


class TRACAdapterWiringTest(unittest.TestCase):
    """The default sampler call is unchanged; TRAC runs only when asked."""

    def setUp(self):
        install_for_test_case(self)
        self.calls = []
        saved = {k: sys.modules.get(k) for k in ("mlx_vlm", "mlx_vlm.generate", "mlx_vlm.prompt_utils")}

        def restore():
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        self.addCleanup(restore)
        pkg = types.ModuleType("mlx_vlm")
        gen = types.ModuleType("mlx_vlm.generate")
        prompt = types.ModuleType("mlx_vlm.prompt_utils")

        def generate(model, processor, formatted, **kwargs):
            self.calls.append((formatted, kwargs))
            return types.SimpleNamespace(text="ok", finish_reason="stop",
                                         diffusion_canvas_tokens=256, diffusion_denoising_steps=7)
        gen.generate = generate
        prompt.apply_chat_template = lambda processor, config, prompt, **kw: f"<{prompt}>"
        sys.modules.update({"mlx_vlm": pkg, "mlx_vlm.generate": gen, "mlx_vlm.prompt_utils": prompt})

    def adapter(self, root):
        fixture(root)
        model = types.SimpleNamespace(config=types.SimpleNamespace())
        return DiffusionGemmaAdapter(root, backend_factory=lambda path: (model, object()))

    def test_default_call_unchanged_and_trac_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self.adapter(Path(tmp))
            out = adapter.generate_text("hello")
            self.assertEqual(self.calls[-1], ("<hello>", {
                "image": None, "video": None, "verbose": False, "max_tokens": 256,
                "max_denoising_steps": 48, "diffusion_sampler": "entropy-bound",
                "temperature": 1.0, "diffusion_full_canvas": True, "seed": 0}))
            self.assertEqual((out.sampler, out.trac), ("entropy-bound", None))
            adapter.generate_text("hello", sampler="confidence-threshold")
            self.assertEqual(self.calls[-1][1]["diffusion_sampler"], "confidence-threshold")
            with self.assertRaisesRegex(ValueError, "trac_config"):
                adapter.generate_text("hello", trac_config=TRACPersistenceConfig())

            entered = []

            class FakeHook:
                def __init__(self, config):
                    entered.append(config)

                def __enter__(self):
                    return {"early_commits": 3, "steps": 5}

                def __exit__(self, *exc):
                    entered.append("exit")
                    return False

            original = diffusion_trac.persistence_hook
            diffusion_trac.persistence_hook = FakeHook
            try:
                out = adapter.generate_text("hello", sampler="trac-persistence", temperature=0)
            finally:
                diffusion_trac.persistence_hook = original
            self.assertEqual(self.calls[-1][1]["diffusion_sampler"], "confidence-threshold")
            self.assertEqual(entered, [TRACPersistenceConfig(), "exit"])
            self.assertEqual(out.sampler, "trac-persistence")
            self.assertEqual(out.trac["early_commits"], 3)
            self.assertFalse(out.trac["qualified"])
            self.assertNotIn("mlx", sys.modules)


def _standin_functions():
    import mlx.core as mx

    def token_probability(processed_logits, token_ids):
        logits = processed_logits.astype(mx.float32)
        tok = mx.take_along_axis(logits, token_ids[..., None], axis=-1).squeeze(-1)
        return mx.exp(tok - mx.logsumexp(logits, axis=-1))

    def transfer(confidence, unrevealed_mask, threshold, *, force_all=False):
        if force_all:
            return unrevealed_mask
        mask = unrevealed_mask & (confidence >= threshold)
        need = mx.any(unrevealed_mask, axis=-1) & ~mx.any(mask, axis=-1)
        best = mx.argmax(mx.where(unrevealed_mask, confidence, -mx.inf), axis=-1)
        pos = mx.arange(confidence.shape[-1])[None, :]
        return mask | ((pos == best[:, None]) & need[:, None])

    return token_probability, transfer


def _three_steps(module, logits, tokens):
    import mlx.core as mx

    unrev = mx.array([[True] * logits.shape[1]])
    masks = []
    for _ in range(3):
        conf = module._diffusion_token_probability(logits, tokens)
        masks.append(np.array(module._diffusion_confidence_transfer_mask(conf, unrev, 0.9)[0]).tolist())
        unrev = unrev & ~mx.array([masks[-1]])
    return masks


class TRACHookRealMLXCPUTest(unittest.TestCase):
    """Hook semantics with real MLX on the CPU device and a stand-in module."""

    def _inputs(self):
        import mlx.core as mx

        # Two stable medium positions: plain force-one takes one at step 3, TRAC both.
        probs = np.stack([dist(3, 0.95), dist(7, 0.6), dist(8, 0.55), dist(1, 0.52)])
        logits = mx.array(np.log(probs)[None].astype(np.float32))
        return logits, mx.argmax(logits, axis=-1)

    def test_other_thread_falls_through_to_originals(self):
        import threading

        token_probability, transfer = _standin_functions()
        module = types.SimpleNamespace(__file__=__file__, _diffusion_token_probability=token_probability,
                                       _diffusion_confidence_transfer_mask=transfer)
        logits, tokens = self._inputs()
        plain = types.SimpleNamespace(_diffusion_token_probability=token_probability,
                                      _diffusion_confidence_transfer_mask=transfer)
        expected_plain = _three_steps(plain, logits, tokens)
        results = {}
        with diffusion_trac.persistence_hook(TRACPersistenceConfig(), module=module,
                                             expected_sha256=None) as counters:
            # A concurrent non-TRAC caller on another thread runs while installed.
            worker = threading.Thread(target=lambda: results.__setitem__("other", _three_steps(module, logits, tokens)))
            worker.start()
            worker.join()
            after_other = dict(counters)
            results["owner"] = _three_steps(module, logits, tokens)
        self.assertEqual(results["other"], expected_plain)
        self.assertEqual(after_other, {"early_commits": 0, "steps": 0})
        self.assertEqual(expected_plain[2], [False, False, True, False])
        self.assertEqual(results["owner"][2], [False, False, True, True])
        self.assertEqual(counters["early_commits"], 2)

    def test_lock_released_when_install_fails(self):
        token_probability, transfer = _standin_functions()

        class Missing:
            __file__ = __file__
            _diffusion_token_probability = staticmethod(token_probability)

        with self.assertRaises(AttributeError):
            with diffusion_trac.persistence_hook(TRACPersistenceConfig(), module=Missing(), expected_sha256=None):
                pass

        class RefusesPatch:
            __file__ = __file__

            def __init__(self):
                object.__setattr__(self, "_diffusion_token_probability", token_probability)
                object.__setattr__(self, "_diffusion_confidence_transfer_mask", transfer)

            def __setattr__(self, name, value):
                if name == "_diffusion_confidence_transfer_mask" and value is not transfer:
                    raise RuntimeError("refused")
                object.__setattr__(self, name, value)

        refusing = RefusesPatch()
        with self.assertRaisesRegex(RuntimeError, "refused"):
            with diffusion_trac.persistence_hook(TRACPersistenceConfig(), module=refusing, expected_sha256=None):
                pass
        self.assertIs(refusing._diffusion_token_probability, token_probability)
        self.assertIs(refusing._diffusion_confidence_transfer_mask, transfer)
        self.assertFalse(diffusion_trac._HOOK_LOCK.locked())
        self.assertIsNone(diffusion_trac._HOOK_OWNER.get())
        # And a later install works.
        module = types.SimpleNamespace(__file__=__file__, _diffusion_token_probability=token_probability,
                                       _diffusion_confidence_transfer_mask=transfer)
        with diffusion_trac.persistence_hook(TRACPersistenceConfig(), module=module, expected_sha256=None):
            with self.assertRaisesRegex(RuntimeError, "already installed"):
                with diffusion_trac.persistence_hook(TRACPersistenceConfig(), module=module,
                                                     expected_sha256=None):
                    pass
        self.assertFalse(diffusion_trac._HOOK_LOCK.locked())

    def test_hook_admits_early_commits_and_restores(self):
        import mlx.core as mx

        def token_probability(processed_logits, token_ids):
            logits = processed_logits.astype(mx.float32)
            tok = mx.take_along_axis(logits, token_ids[..., None], axis=-1).squeeze(-1)
            return mx.exp(tok - mx.logsumexp(logits, axis=-1))

        def transfer(confidence, unrevealed_mask, threshold, *, force_all=False):
            if force_all:
                return unrevealed_mask
            mask = unrevealed_mask & (confidence >= threshold)
            need = mx.any(unrevealed_mask, axis=-1) & ~mx.any(mask, axis=-1)
            best = mx.argmax(mx.where(unrevealed_mask, confidence, -mx.inf), axis=-1)
            pos = mx.arange(confidence.shape[-1])[None, :]
            return mask | ((pos == best[:, None]) & need[:, None])

        module = types.SimpleNamespace(__file__=__file__, _diffusion_token_probability=token_probability,
                                       _diffusion_confidence_transfer_mask=transfer)
        probs = np.stack([dist(3, 0.95), dist(7, 0.6), dist(8, 0.55), dist(1, 0.3)])
        logits = mx.array(np.log(probs)[None].astype(np.float32))
        tokens = mx.argmax(logits, axis=-1)
        cfg = TRACPersistenceConfig()
        with diffusion_trac.persistence_hook(cfg, module=module, expected_sha256=None) as counters:
            unrev = mx.array([[True, True, True, True]])
            masks = []
            for step in range(3):
                conf = module._diffusion_token_probability(logits, tokens)
                masks.append(np.array(module._diffusion_confidence_transfer_mask(conf, unrev, 0.9)[0]).tolist())
                unrev = unrev & ~mx.array([masks[-1]])
        self.assertIs(module._diffusion_token_probability, token_probability)
        self.assertIs(module._diffusion_confidence_transfer_mask, transfer)
        self.assertEqual(masks[0], [True, False, False, False])
        self.assertEqual(masks[1], [False, True, False, False])  # force-one (no early yet)
        self.assertEqual(masks[2], [False, False, True, False])  # step 3: early commit
        self.assertEqual(counters["early_commits"], 1)
        with self.assertRaisesRegex(RuntimeError, "unpinned"):
            with diffusion_trac.persistence_hook(cfg, module=module):
                pass


if __name__ == "__main__":
    unittest.main()
