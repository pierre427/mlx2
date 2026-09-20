"""rm15: prefill chunk size must not change what the engine *means*.

These are CPU tests over a tiny random-weight model with the 27B's own
hybrid architecture (GatedDeltaNet + full attention).  They pin three
separate claims, each of which was established by measurement and any of
which failing would be news:

1. The engine is deterministic for a FIXED chunk size (bit-for-bit).
2. Chunking a dense matmul by ROWS is bit-exact, so the divergence does not
   come from splitting work across rows as such.
3. A row's dense-matmul result DOES depend on how many rows are in the call
   (M), which is the mechanism by which a prefill chunk-size change perturbs
   every linear layer.  This test documents the mechanism; if MLX ever makes
   matmul M-invariant it will fail, and that is a result worth noticing.

It also exercises the GPU harness's own instrumentation on CPU, so a broken
harness is caught without spending a GPU slot.
"""

import sys
import unittest
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mlx2.runtime.models.cache import make_prompt_cache
from mlx2.runtime.models.qwen38_27b import Model, ModelArgs

import measure_prefill_chunk_variance as H


TEXT_CONFIG = dict(
    model_type="qwen3_5_text",
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    vocab_size=128,
    linear_num_value_heads=4,
    linear_num_key_heads=2,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    mtp_num_hidden_layers=0,
    num_experts=0,
    rms_norm_eps=1e-6,
    max_position_embeddings=65536,
    rope_parameters={
        "type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
    },
)


def _tiny_model(seed=0):
    mx.random.seed(seed)
    model = Model(ModelArgs(model_type="qwen3_5", text_config=dict(TEXT_CONFIG)))
    mx.eval(model.parameters())
    return model


def _prefill(model, stream, ctx, step):
    cache = list(make_prompt_cache(model))
    for start in range(0, ctx, step):
        mx.eval(model(stream[:, start : min(start + step, ctx)], cache=cache))
    return cache


class PrefillChunkVarianceTest(unittest.TestCase):
    CTX = 256
    SCORE = 8

    def setUp(self):
        self.model = _tiny_model()
        mx.random.seed(1234)
        self.ids = mx.random.randint(
            0, 128, (1, self.CTX + self.SCORE + 1)
        ).astype(mx.uint32)

    # -- 1. same chunk size is bit-for-bit deterministic -------------------

    def test_same_chunk_size_is_bit_exact(self):
        for step in (self.CTX, 128, 64):
            a = _prefill(self.model, self.ids, self.CTX, step)
            b = _prefill(self.model, self.ids, self.CTX, step)
            for ca, cb in zip(a, b):
                for name, arr in H.cache_fields(ca, 16).items():
                    other = H.cache_fields(cb, 16)[name]
                    self.assertEqual(
                        0.0,
                        float(mx.max(mx.abs(arr.astype(mx.float32)
                                            - other.astype(mx.float32))).item()),
                        f"step={step} field={name} is not reproducible",
                    )

    # -- 2. row-chunking an op is bit-exact -------------------------------

    def test_row_chunking_a_matmul_is_bit_exact(self):
        """Splitting the SAME call's rows into several calls of the same
        shape family does not change any row."""
        mx.random.seed(7)
        x = mx.random.normal((512, 256))
        w = mx.random.normal((256, 256))
        full = x @ w
        mx.eval(full)
        for chunk in (64, 128, 256):
            parts = mx.concatenate(
                [x[i : i + chunk] @ w for i in range(0, 512, chunk)], axis=0
            )
            mx.eval(parts)
            self.assertEqual(0.0, float(mx.max(mx.abs(parts - full)).item()))

    # -- 3. the mechanism: a row's result depends on M --------------------

    def test_matmul_result_for_a_fixed_row_depends_on_M(self):
        """This is the root mechanism behind prefill chunk-size variance.

        The last row of ``x @ W.T`` is not the same number when the call has
        512 rows and when it has 1: the reduction strategy is keyed on M.
        A prefill chunk-size change changes M in every linear layer.
        """
        mx.random.seed(11)
        k, n, s = 1024, 64, 512
        w = mx.random.normal((n, k))
        x = mx.random.normal((1, s, k))
        full = (x @ w.T)[0, -1]
        mx.eval(full)
        one = (x[:, s - 1 : s] @ w.T)[0, -1]
        mx.eval(one)
        delta = float(mx.max(mx.abs(one - full)).item())
        self.assertGreater(
            delta,
            0.0,
            "MLX matmul has become M-invariant; rm15's stated mechanism for "
            "prefill chunk-size variance no longer holds and the finding "
            "should be re-measured",
        )
        # It is rounding-scale, not a logic error: relative to the row's own
        # magnitude it sits near float32 epsilon.
        scale = float(mx.max(mx.abs(full)).item())
        self.assertLess(delta / max(scale, 1e-6), 1e-4)

    # -- the harness itself ------------------------------------------------

    def test_harness_localises_divergence_on_cpu(self):
        capture = {}
        names = H.layer_names(self.model)
        H.instrument(self.model, capture)
        ref = H.run_arm(self.model, self.ids, self.CTX, self.CTX,
                        self.SCORE, capture, names, 16)
        same = H.run_arm(self.model, self.ids, self.CTX, self.CTX,
                         self.SCORE, capture, names, 16)
        control = H.compare(same, ref)
        self.assertEqual(1.0, control["decode_top1_agreement"])
        self.assertEqual(0.0, control["decode_max_abs_logit_delta"])
        self.assertIsNone(control["first_divergent_layer"])
        self.assertEqual(0, control["n_differing_cache_fields"])

        chunked = H.run_arm(self.model, self.ids, self.CTX, 100,
                            self.SCORE, capture, names, 16)
        result = H.compare(chunked, ref)
        self.assertEqual(len(names),
                         len(result["per_layer_last_pos_max_abs_delta"]))
        self.assertIn("decode_kl_mean", result)

    def test_parse_arms(self):
        self.assertEqual(
            [("full", 4096, False), ("2048", 2048, False), ("2048r", 2048, True)],
            H.parse_arms("full,2048,2048r", 4096),
        )



# --------------------------------------------------------------- serving side

STEP = 32
LONG_PROMPT = [(7 * i + 3) % 120 + 2 for i in range(10 * STEP)]


def _tiny_text_model():
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=4,
        mtp_num_hidden_layers=0, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=1 << 14,
    )
    mx.random.seed(7)
    model = TextModel(args)
    model.eval()
    mx.eval(model.parameters())
    return model


class PrefillChunkReceiptTest(unittest.TestCase):
    """The prefill chunk schedule must reach the request's receipt.

    The chunk size a request gets is chosen per round from measured timing and
    current concurrency, and a different chunk size gives a different answer,
    so a receipt that omits it cannot explain why the same prompt answered
    differently.
    """

    def _generator(self, **kw):
        from mlx2.runtime.generate import BatchGenerator

        return BatchGenerator(
            _tiny_text_model(), completion_batch_size=4, prefill_batch_size=2,
            prefill_step_size=STEP, prefill_batch_window=1, **kw
        )

    def _drive(self, gen, prompt, rounds=200):
        uid = gen.insert([prompt], max_tokens=[2])[0]
        for _ in range(rounds):
            prompts, _gen_responses = gen.next()
            if any(r.uid == uid and r.end_of_prompt for r in prompts):
                break
        return uid

    def test_fixed_step_records_one_scheduled_width(self):
        gen = self._generator(adaptive_prefill=False)
        uid = self._drive(gen, LONG_PROMPT)
        trace = gen.pop_prefill_chunk_trace(uid)
        self.assertIsNotNone(trace, "prefill chunk trace was never recorded")
        self.assertEqual("mlx2.prefill-chunk-trace.v1", trace["schema"])
        self.assertEqual(STEP, trace["configured_step"])
        self.assertFalse(trace["adaptive"])
        self.assertFalse(
            trace["varied"],
            f"a fixed-step prefill reported a varying chunk schedule: {trace}",
        )
        self.assertGreater(trace["rounds"], 1)
        # mechanism counter: the recording path actually ran
        self.assertGreater(
            gen.scheduler_stats["prefill_chunk_rounds_recorded"], 0
        )

    def test_trace_is_popped_once(self):
        gen = self._generator(adaptive_prefill=False)
        uid = self._drive(gen, LONG_PROMPT)
        self.assertIsNotNone(gen.pop_prefill_chunk_trace(uid))
        self.assertIsNone(gen.pop_prefill_chunk_trace(uid))

    def test_varied_schedule_is_flagged_and_counted(self):
        """A chunk schedule that changes mid-prefill sets ``varied`` and bumps
        the mechanism counter, so an auditor can tell the two cases apart."""
        gen = self._generator(adaptive_prefill=False)
        uid = 4242
        gen._record_prefill_chunk(uid, 32)
        gen._record_prefill_chunk(uid, 32)
        gen._record_prefill_chunk(uid, 8)
        gen._record_prefill_chunk(uid, 8)
        trace = gen.pop_prefill_chunk_trace(uid)
        self.assertTrue(trace["varied"])
        self.assertEqual({"32": 2, "8": 2}, trace["widths"])
        self.assertEqual(32, trace["first"])
        self.assertEqual(8, trace["last"])
        self.assertEqual(4, trace["rounds"])
        self.assertEqual(1, gen.scheduler_stats["prefill_chunk_varied_requests"])

    def test_single_short_tail_chunk_is_not_called_varied(self):
        """The last chunk of a prompt is short because of the prompt length,
        not because the scheduler changed its mind."""
        gen = self._generator(adaptive_prefill=False)
        uid = 99
        gen._record_prefill_chunk(uid, 32)
        gen._record_prefill_chunk(uid, 32)
        gen._record_prefill_chunk(uid, 5)
        trace = gen.pop_prefill_chunk_trace(uid)
        self.assertFalse(trace["varied"], trace)
        self.assertEqual(0, gen.scheduler_stats["prefill_chunk_varied_requests"])

if __name__ == "__main__":
    unittest.main()


def test_chunk_counters_reach_prometheus_under_their_own_mechanism():
    """The trace is default-on telemetry; a counter nobody can scrape is not
    an observation.  rm15 added the two counters to scheduler_stats but not to
    the exporter's allowlist, so /metrics dropped them silently."""
    from mlx2 import prometheus

    for key in ("prefill_chunk_rounds_recorded", "prefill_chunk_varied_requests"):
        assert key in prometheus._SCHEDULER_EVENTS
        # Its own mechanism: the chunk schedule is not adaptive-prefill's
        # release policy and not decode fairness.
        assert prometheus._scheduler_mechanism(key) == "prefill_chunk"
