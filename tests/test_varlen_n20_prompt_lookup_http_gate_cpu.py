"""CPU-only contract checks for the bounded N3 actual HTTP gate."""
import importlib.abc
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/research")]


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "mlx" or name.startswith("mlx.") or name == "_paged_kv_native":
            raise RuntimeError("device import forbidden")


sys.meta_path.insert(0, Guard())
import varlen_n20_prompt_lookup_http_gate as G


class Tests(unittest.TestCase):
    def test_plan_is_narrow_default_off_and_not_qualified(self):
        value = G.plan()
        self.assertEqual(value["cohort_size"], 3)
        self.assertEqual(value["max_tokens"], 4)
        self.assertEqual(value["max_tokens_by_lane"], [4, 4, 4])
        self.assertFalse(value["gpu_executed"])
        self.assertFalse(value["qualified"])
        self.assertFalse(value["selected_by_default"])
        self.assertFalse(value["performance_claim"])
        self.assertFalse(value["performance_measurement"])
        self.assertFalse(value["diagnostic_stage_profile_enabled"])
        measured = G.plan(
            "full-k2", performance_measurement=True,
            arm_order="ordinary-native")
        self.assertTrue(measured["performance_measurement"])
        self.assertEqual(measured["arm_order"], "ordinary-native")
        profiled = G.plan(
            "full-k2", diagnostic_stage_profile=True,
            ordinary_declared_cohort=True, ordinary_prefill_batch_size=3)
        self.assertTrue(profiled["diagnostic_stage_profile_enabled"])
        self.assertEqual(
            profiled["ordinary_admission"],
            "declared_atomic_n3_research_prefill_override")
        self.assertEqual(profiled["ordinary_prefill_batch_size"], 3)
        experimental = G.plan(
            native_mlp_experiment="single_eval_bf16",
            gdn_eval_wave_max_segments=4,
            gdn_eval_wave_expanded_charge=True)
        self.assertEqual(experimental["native_mlp_experiment"], "single_eval_bf16")
        self.assertEqual(experimental["gdn_eval_wave_max_segments"], 4)
        self.assertTrue(experimental["gdn_eval_wave_expanded_charge"])
        tiled = G.plan(native_mlp_experiment="tiled_q4_swiglu")
        self.assertEqual(tiled["native_mlp_experiment"], "tiled_q4_swiglu")
        packed = G.plan(native_mlp_experiment="packed_gate_up_qmm")
        self.assertEqual(packed["native_mlp_experiment"], "packed_gate_up_qmm")
        prefill = G.plan("prefill-only", performance_measurement=True)
        self.assertEqual(prefill["max_tokens_by_lane"], [1, 1, 1])
        self.assertEqual(prefill["expected_draft_depths"], [0, 0, 0])
        self.assertEqual(prefill["prompt_lookup_depth"], 0)
        self.assertIsNone(prefill["prompt_lookup_ngram"])
        with self.assertRaises(ValueError):
            G.plan(ordinary_prefill_batch_size=1)
        with self.assertRaises(ValueError):
            G.plan(ordinary_prefill_batch_size=3)
        with self.assertRaises(ValueError):
            G.plan(native_mlp_experiment="fast")
        with self.assertRaises(ValueError):
            G.plan(gdn_eval_wave_max_segments=3)
        with self.assertRaises(ValueError):
            G.plan(gdn_eval_wave_expanded_charge=True)

    def test_request_body_preserves_source_and_selects_only_native_arm(self):
        row = {
            "case_id": "case-1", "body": {
                "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 192, "temperature": 0, "enable_thinking": False,
            },
        }
        native = G.request_body(row, model="m", native=True, inputs_sha256="a" * 64)
        ordinary = G.request_body(row, model="m", native=False, inputs_sha256="a" * 64)
        cohort = G.request_body(
            row, model="m", native=False, inputs_sha256="a" * 64,
            declared_cohort=True, cohort_id="ordinary-n3")
        short = G.request_body(row, model="m", native=True,
                               inputs_sha256="a" * 64, cap=2,
                               cohort_id="mixed")
        self.assertEqual(native["max_tokens"], 4)
        self.assertEqual(native["batch_cohort"], {"id": "n20-prompt-lookup-http-n3", "size": 3})
        self.assertTrue(native["paged_native_packed_n20_research"])
        self.assertNotIn("paged_native_packed_n20_research", ordinary)
        self.assertNotIn("batch_cohort", ordinary)
        self.assertEqual(cohort["batch_cohort"], {"id": "ordinary-n3", "size": 3})
        self.assertNotIn("paged_native_packed_n20_research", cohort)
        self.assertEqual(ordinary["native_research_input_id"], "case-1")
        self.assertEqual(ordinary["native_research_inputs_sha256"], "a" * 64)
        self.assertEqual(short["max_tokens"], 2)
        self.assertEqual(short["batch_cohort"]["id"], "mixed")
        one = G.request_body(row, model="m", native=True,
                             inputs_sha256="a" * 64, cap=1)
        self.assertEqual(one["max_tokens"], 1)

    def test_prefill_only_requires_packed_prefill_and_zero_drafts(self):
        row = {"case_id": "x", "prompt_tokens": 2048}
        events = [{"uid": 17, "token": 7, "from_draft": False}]
        receipt = {
            "route": "native_hybrid_packed_n20_research",
            "selected": True, "observed_used": True,
            "qualified": False, "price_usable": False,
            "native_n20_ragged_observed_used": False,
            "draft_proposed": 0, "draft_accepted": 0,
            "speculative_verification": False,
            "prefill_mode": "native_packed_prefill",
            "prefill_layout": "real_rows",
            "native_prefill_observed_used": True,
            "output_token_ids": [7], "prefill_cohort_width": 3,
        }
        body = {
            "choices": [{"finish_reason": "length",
                         "message": {"content": "ok"}}],
            "usage": {"completion_tokens": 1, "prompt_tokens": 2048,
                      "prompt_tokens_details": {"cached_tokens": 0}},
            "mlx2": {"qualification": "unqualified",
                     "route_receipt": receipt},
        }
        value = G.summarize(
            row, (200, body), events, native=True, cap=1,
            expected_draft=0, expected_query_lengths=(1, 1, 1),
            expected_round_widths=(3,), prefill_only=True)
        self.assertEqual(value["tokens"], [7])
        for key, bad_value in (
                ("native_prefill_observed_used", False),
                ("speculative_verification", True),
                ("draft_proposed", 1)):
            bad = json.loads(json.dumps(body))
            bad["mlx2"]["route_receipt"][key] = bad_value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                G.summarize(
                    row, (200, bad), events, native=True, cap=1,
                    expected_draft=0, expected_query_lengths=(1, 1, 1),
                    expected_round_widths=(3,), prefill_only=True)

    def test_final_native_receipt_requires_real_k2_prompt_lookup(self):
        row = {"case_id": "x", "prompt_tokens": 2048}
        events = [{"uid": 17, "token": token, "from_draft": from_draft}
                  for token, from_draft in zip(
                      (7, 8, 9, 10), (False, True, True, False))]
        receipt = {
            "route": "native_hybrid_packed_n20_research",
            "selected": True, "observed_used": True,
            "qualified": False, "price_usable": False,
            "native_n20_ragged_observed_used": True,
            "draft_source": "prompt_lookup", "draft_proposed": 2,
            "draft_accepted": 2, "draft_configured_max_depth": 2,
            "verifier_executed_rows": 3,
            "verifier_round_widths": [3, 3, 3],
            "speculative_verification": True,
            "state_publication": "atomic_selected_executed_prefix",
            "ragged_verify_layout": {
                "lane_uids": [17, 18, 19], "query_lengths": [3, 3, 3]},
            "native_n20_graph_proof": {
                "physical_counters": G.physical_counters((3, 3, 3))},
            "output_token_ids": [7, 8, 9, 10], "prefill_cohort_width": 3,
        }
        body = {
            "choices": [{"finish_reason": "length", "message": {"content": "ok"}}],
            "usage": {"completion_tokens": 4, "prompt_tokens": 2048,
                      "prompt_tokens_details": {"cached_tokens": 0}},
            "mlx2": {"qualification": "unqualified", "route_receipt": receipt},
        }
        summary = G.summarize(row, (200, body), events, native=True)
        self.assertEqual(summary["tokens"], [7, 8, 9, 10])
        for key, value in (("draft_accepted", 1), ("draft_source", "caller"),
                           ("state_publication", "full_query_row_only")):
            bad = json.loads(json.dumps(body))
            bad["mlx2"]["route_receipt"][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                G.summarize(row, (200, bad), events, native=True)

    def test_ordinary_unqualified_string_receipt_is_not_native_selection(self):
        row = {"case_id": "x", "prompt_tokens": 2048}
        events = [{"token": token} for token in (7, 8, 9, 10)]
        body = {
            "choices": [{"finish_reason": "length", "message": {"content": "ok"}}],
            "usage": {"completion_tokens": 4, "prompt_tokens": 2048,
                      "prompt_tokens_details": {"cached_tokens": 0}},
            "mlx2": {"qualification": "unqualified", "route_receipt": "unqualified"},
        }
        value = G.summarize(row, (200, body), events, native=False)
        self.assertEqual(value["route_receipt"], "unqualified")

    def test_mixed_caps_require_k2_k1_k0_and_n3_n2_n1(self):
        caps, drafts, queries, widths = G.cap_contract("shrinking-k2-k1-k0")
        self.assertEqual((caps, drafts, queries, widths),
                         ((4, 3, 2), (2, 1, 0), (3, 2, 1), (3, 2, 1)))
        layout = {"lane_uids": [31, 30, 32], "query_lengths": [2, 3, 1]}
        for lane, (cap, draft, uid) in enumerate(zip(caps, drafts, (30, 31, 32))):
            tokens = list(range(10, 10 + cap))
            events = [{"uid": uid, "token": token, "from_draft": from_draft}
                      for token, from_draft in zip(
                          tokens, [False, *([True] * draft), False])]
            receipt = {
                "route": "native_hybrid_packed_n20_research",
                "selected": True, "observed_used": True,
                "qualified": False, "price_usable": False,
                "native_n20_ragged_observed_used": draft > 0,
                "draft_source": "prompt_lookup", "draft_proposed": draft,
                "draft_accepted": draft, "draft_configured_max_depth": 2,
                "verifier_executed_rows": draft + 1,
                "verifier_round_widths": list(widths),
                "speculative_verification": True,
                "state_publication": "atomic_selected_executed_prefix",
                "ragged_verify_layout": layout,
                "native_n20_graph_proof": {
                    "physical_counters": G.physical_counters(widths)},
                "output_token_ids": tokens, "prefill_cohort_width": 3,
            }
            body = {
                "choices": [{"finish_reason": "length",
                             "message": {"content": "lane" + str(lane)}}],
                "usage": {"completion_tokens": cap, "prompt_tokens": 2048,
                          "prompt_tokens_details": {"cached_tokens": 0}},
                "mlx2": {"qualification": "unqualified",
                         "route_receipt": receipt},
            }
            value = G.summarize(
                {"case_id": str(lane), "prompt_tokens": 2048},
                (200, body), events, native=True, cap=cap,
                expected_draft=draft, expected_query_lengths=queries,
                expected_round_widths=widths)
            self.assertEqual(value["tokens"], tokens)

    def test_import_and_plan_do_not_import_device_runtime(self):
        self.assertEqual(G.COHORT_SIZE, 3)
        self.assertEqual(G.OUTPUT_CAP, 4)

    def test_timing_summary_uses_synchronized_calls_and_client_wall(self):
        def event(uid, token, tick, call, start, draft=False):
            return {
                "uid": uid, "token": token, "from_draft": draft,
                "monotonic_ns": tick, "generator_call_id": call,
                "generator_call_started_ns": start,
                "generator_call_ended_ns": tick,
            }
        streams = {
            "a": [event(0, 1, 1_000_000_000, 0, 500_000_000),
                  event(0, 2, 4_000_000_000, 1, 1_100_000_000, True)],
            "b": [event(1, 1, 1_000_000_000, 0, 500_000_000),
                  event(1, 2, 4_000_000_000, 1, 1_100_000_000, True)],
            "c": [event(2, 1, 1_000_000_000, 0, 500_000_000),
                  event(2, 2, 4_000_000_000, 1, 1_100_000_000, True)],
        }
        value = G.timing_summary(streams, {
            "client_starts_ns": [0, 10_000_000, 20_000_000],
            "client_ends_ns": [5_000_000_000, 5_100_000_000, 5_200_000_000],
        })
        self.assertEqual(value["cohort_http_wall_seconds"], 5.2)
        self.assertEqual(value["client_start_spread_seconds"], .02)
        self.assertEqual(value["time_through_all_first_samples_seconds"], 1)
        self.assertEqual(value["first_sample_to_last_sample_seconds"], 3)
        self.assertEqual(value["per_lane_first_to_last_seconds"], [3, 3, 3])
        self.assertEqual(value["mean_per_lane_first_to_last_seconds"], 3)
        self.assertEqual(value["max_per_lane_first_to_last_seconds"], 3)
        self.assertEqual(value["last_sample_to_http_complete_seconds"], 1.2)
        self.assertEqual(value["generator_calls"], 2)
        self.assertEqual(value["generator_call_wall_seconds"], 3.4)
        self.assertEqual(value["accepted_draft_tokens"], 3)
        self.assertEqual(value["execution_widths"], [1])

    def test_stage_receipt_delta_is_per_call_and_nonnegative(self):
        before = {"stages": {
            "mlp.gate_up": {"calls": 2, "rows": 20, "elapsed_ns": 200}}}
        after = {"stages": {
            "mlp.gate_up": {"calls": 5, "rows": 50, "elapsed_ns": 800},
            "mlp.down": {"calls": 1, "rows": 10, "elapsed_ns": 300}}}
        self.assertEqual(G.stage_receipt_delta(before, after), {
            "mlp.down": {"calls": 1, "rows": 10, "elapsed_ns": 300},
            "mlp.gate_up": {"calls": 3, "rows": 30, "elapsed_ns": 600},
        })
        with self.assertRaises(RuntimeError):
            G.stage_receipt_delta(after, before)


if __name__ == "__main__":
    unittest.main()
