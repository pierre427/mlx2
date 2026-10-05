"""Pure host checks for the Qwen3.8 served A/B campaign contract."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ab_qwen38_dflash2.py"
SPEC = importlib.util.spec_from_file_location("ab_qwen38_dflash2", SCRIPT)
AB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AB)

SUMMARY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/summarize_qwen38_dflash2.py"
)
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_qwen38_dflash2", SUMMARY_SCRIPT
)
SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
SUMMARY_SPEC.loader.exec_module(SUMMARY)


def status(route, speculation, mtp, num_draft, *, pairwise=0):
    return {
        "profile": "test-profile",
        "artifact": "artifact",
        "runtime": {"source_sha256": "source"},
        "settings": {
            "route": route,
            "speculation": speculation,
            "mtp": mtp,
            "route_selection_source": "test",
            "host_prompt_cache_entries": 0,
            "host_prompt_cache_tokens": 0,
            "execution_policy": {
                "num_draft": num_draft,
                **(
                    {
                        "backend": "external_draft",
                        "pairwise_selection": "batched",
                    }
                    if speculation == "external_draft"
                    else {}
                ),
            },
        },
        "scheduler": {"external_pairwise_selection_groups": pairwise},
    }


def controls(sampling):
    return {
        "sampling": dict(sampling),
        "effective_sampling": dict(sampling),
        "sampling_defaults": {},
        "skip_writing_prefix_cache": True,
    }


class ArmContractTests(unittest.TestCase):
    def test_campaign_shape_is_exactly_b1_and_b4(self):
        AB.validate_campaign_shape([1, 4], 4, [0, 0.7])
        for widths in ([1], [4], [4, 1], [1, 2, 4], [1, 5]):
            with self.assertRaisesRegex(ValueError, "exactly 1,4"):
                AB.validate_campaign_shape(widths, 4, [0, 0.7])
        for count in (0, 1, 2, 3, AB.MAX_B1_PROMPTS + 1):
            with self.assertRaisesRegex(ValueError, "exactly 4"):
                AB.validate_campaign_shape([1, 4], count, [0, 0.7])
        for temperatures, message in (
            ([0, 0], "unique"),
            ([float("nan")], "finite"),
            ([float("inf")], "finite"),
            ([-0.1], "finite"),
            ([], "finite"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                AB.validate_campaign_shape([1, 4], 4, temperatures)
        self.assertEqual(len(AB.workload_batch("code", 1, 4)), 4)
        self.assertEqual(len(AB.workload_batch("code", 4, 4)), 4)

    def test_server_process_disables_host_prompt_cache(self):
        args = mock.Mock(model="model", max_context=32768)
        command = AB.server_command(args, "dk4v", 18731)
        self.assertEqual(
            command[command.index("--host-prompt-cache-entries") + 1], "0"
        )
        self.assertEqual(
            command[command.index("--host-prompt-cache-tokens") + 1], "0"
        )
        self.assertIn("--qualification-mode", command)

    def test_startup_refuses_a_live_host_prompt_cache(self):
        live = status("ordinary", "ordinary", False, None)
        live["settings"]["host_prompt_cache_entries"] = 1
        with self.assertRaisesRegex(RuntimeError, "host_prompt_cache_entries"):
            AB.validate_startup_route("ord", live)

    def test_served_model_identity_is_exact_not_a_substring(self):
        self.assertEqual(
            AB.validate_served_model(
                {"model": "Qwen3.8-27B-oQ4e-mtp"},
                "/models/Qwen3.8-27B-oQ4e-mtp",
            ),
            "Qwen3.8-27B-oQ4e-mtp",
        )
        with self.assertRaisesRegex(RuntimeError, "server reports model"):
            AB.validate_served_model(
                {"model": "Qwen3.8-27B-oQ4e-mtp-old"},
                "/models/Qwen3.8-27B-oQ4e-mtp",
            )

    def test_mtp2_is_explicit_and_mtp3_is_current_default(self):
        mtp2 = AB.arm_route_args("mtp2")
        self.assertEqual(mtp2[0], "--execution-policy")
        self.assertEqual(Path(mtp2[1]).name, "qwen38-27b-mtp2.json")
        self.assertEqual(AB.arm_route_args("mtp3"), [])
        self.assertEqual(AB.arm_contract("mtp2")["num_draft"], 2)
        self.assertEqual(AB.arm_contract("mtp3")["num_draft"], 3)

    def test_startup_identity_comes_from_nested_settings(self):
        live = status("native_mtp", "self_mtp", True, 3)
        live.update({"route": "wrong-top-level", "execution_policy": {"num_draft": 99}})
        snapshot = AB.validate_startup_route("mtp3", live)
        self.assertEqual(snapshot["route"], "native_mtp")
        self.assertEqual(snapshot["execution_policy"]["num_draft"], 3)

    def test_startup_depth_mismatch_fails_before_requests(self):
        with self.assertRaisesRegex(RuntimeError, "num_draft"):
            AB.validate_startup_route(
                "mtp2", status("native_mtp", "self_mtp", True, 3)
            )
        AB.validate_startup_route(
            "dk4", status("external_draft", "external_draft", False, 4)
        )

    def test_external_startup_requires_batched_pairwise_policy(self):
        live = status("external_draft", "external_draft", False, 4)
        live["settings"]["execution_policy"]["pairwise_selection"] = "host"
        with self.assertRaisesRegex(RuntimeError, "pairwise_selection"):
            AB.validate_startup_route("dk4", live)

    def test_only_existing_varlen_policy_arms_are_accepted(self):
        self.assertEqual(
            Path(AB.arm_route_args("dk4v")[-1]).name,
            "qwen38-27b-dflash2-k4-varlen-ingress.json",
        )
        self.assertEqual(
            Path(AB.arm_route_args("dk7v")[-1]).name,
            "qwen38-27b-dflash2-k7-varlen-ingress.json",
        )
        for arm in ("dk3v", "dk5v", "dk6v"):
            with self.assertRaisesRegex(ValueError, "only dk4v and dk7v"):
                AB.arm_route_args(arm)
        for arm in ("dk0", "dk1", "dk2", "dk8", "dk9"):
            with self.assertRaisesRegex(ValueError, "dk3 through dk7"):
                AB.arm_route_args(arm)

    def test_campaign_arm_list_is_nonempty_unique_and_validated(self):
        self.assertEqual(
            AB.parse_arms("ord, mtp3,dk4v"), ["ord", "mtp3", "dk4v"]
        )
        for raw in ("", ",", "ord,", ",ord", "ord,,mtp3"):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                AB.parse_arms(raw)
        with self.assertRaisesRegex(ValueError, "unique"):
            AB.parse_arms("ord,ord")
        with self.assertRaisesRegex(ValueError, "unknown arm"):
            AB.parse_arms("ord,unknown")

    def test_varlen_startup_requires_selected_ingress_contract(self):
        live = status("external_draft", "external_draft", False, 4)
        policy = live["settings"]["execution_policy"]
        policy["external_varlen_prefill"] = {"enabled": True}
        with self.assertRaisesRegex(RuntimeError, "ingress cohort policy"):
            AB.validate_startup_route("dk4v", live)
        policy["ingress_cohort"] = {
            "enabled": True,
            "mechanism": "external_varlen_prefill",
            "maximum_wait_ms": 250,
            "minimum_prompt_tokens": 1,
            "target_lanes": 4,
        }
        live["settings"]["apcv2_reuse"] = {
            "enabled": False,
            "reason": "external_varlen_prefill_not_batch_invariant",
        }
        AB.validate_startup_route("dk4v", live)
        policy["ingress_cohort"]["target_lanes"] = 2
        with self.assertRaisesRegex(RuntimeError, "invalid ingress cohort"):
            AB.validate_startup_route("dk4v", live)
        policy["ingress_cohort"]["target_lanes"] = 4
        policy["ingress_cohort"]["maximum_wait_ms"] = 249
        with self.assertRaisesRegex(RuntimeError, "invalid ingress cohort"):
            AB.validate_startup_route("dk4v", live)
        policy["ingress_cohort"]["maximum_wait_ms"] = 250
        policy["ingress_cohort"]["minimum_prompt_tokens"] = 2
        with self.assertRaisesRegex(RuntimeError, "invalid ingress cohort"):
            AB.validate_startup_route("dk4v", live)


class SamplingAndReceiptTests(unittest.TestCase):
    def test_sampling_profiles_name_pairwise_boundary_truthfully(self):
        greedy = AB.sampling_controls(0)
        sampled = AB.sampling_controls(0.7)
        self.assertEqual(greedy["presence_penalty"], 0.0)
        self.assertEqual(sampled["temperature"], 0.7)
        self.assertEqual(sampled["top_p"], 0.8)
        self.assertEqual(sampled["presence_penalty"], 1.5)
        self.assertTrue(AB.pairwise_control_eligibility(greedy)["eligible"])
        ineligible = AB.pairwise_control_eligibility(sampled)
        self.assertFalse(ineligible["eligible"])
        self.assertEqual(ineligible["blockers"], ["presence_penalty"])

    def test_native_request_receipt_must_match_named_depth(self):
        sampling = AB.sampling_controls(0)
        receipt = {
            "route": "native_mtp",
            "cached_tokens": 0,
            "mtp": {"num_draft": 2},
            "request_controls": controls(sampling),
        }
        AB.validate_request_receipt("mtp2", receipt, sampling)
        receipt["mtp"]["num_draft"] = 3
        with self.assertRaisesRegex(RuntimeError, "num_draft"):
            AB.validate_request_receipt("mtp2", receipt, sampling)

    def test_dflash_request_must_execute_without_ordinary_fallback(self):
        sampling = AB.sampling_controls(0)
        receipt = {
            "route": "external_draft",
            "cached_tokens": 0,
            "speculation": {
                "kind": "external_dflash2",
                "execution": "external_draft_verify",
                "ordinary_fallback": False,
            },
            "request_controls": controls(sampling),
        }
        AB.validate_request_receipt("dk4", receipt, sampling)
        receipt["speculation"]["ordinary_fallback"] = True
        with self.assertRaisesRegex(RuntimeError, "did not stay"):
            AB.validate_request_receipt("dk4", receipt, sampling)
        receipt["speculation"].pop("ordinary_fallback")
        with self.assertRaisesRegex(RuntimeError, "did not stay"):
            AB.validate_request_receipt("dk4", receipt, sampling)

    def test_varlen_dflash_request_must_disclose_disabled_apcv2_reuse(self):
        sampling = AB.sampling_controls(0)
        receipt = {
            "route": "external_draft",
            "cached_tokens": 0,
            "speculation": {
                "kind": "external_dflash2",
                "execution": "external_draft_verify",
                "ordinary_fallback": False,
            },
            "request_controls": controls(sampling),
        }
        with self.assertRaisesRegex(RuntimeError, "disabled APCv2 reuse"):
            AB.validate_request_receipt("dk4v", receipt, sampling)
        receipt["apcv2_reuse"] = {
            "enabled": False,
            "reason": "external_varlen_prefill_not_batch_invariant",
        }
        AB.validate_request_receipt("dk4v", receipt, sampling)

    def test_request_captures_effective_controls(self):
        sampling = AB.sampling_controls(0.7)
        captured = {}

        def post(_url, body, _timeout):
            captured.update(body)
            return {
                "mlx2": {
                    "route": "ordinary",
                    "cached_tokens": 0,
                    "completion_tokens": 2,
                    "prompt_tokens": 3,
                    "ttft_seconds": 0.1,
                    "elapsed_seconds": 0.2,
                    "request_controls": controls(sampling),
                },
                "choices": [
                    {
                        "message": {"content": "ok", "reasoning_content": ""},
                        "finish_reason": "length",
                    }
                ],
            }

        with mock.patch.object(AB, "_post", post):
            row = AB._request(
                "http://unused",
                "hello",
                arm="ord",
                max_tokens=2,
                temperature=0.7,
                timeout=1,
                nonce="n",
                seed=7,
            )
        self.assertEqual(captured["presence_penalty"], 1.5)
        self.assertEqual(captured["temperature"], 0.7)
        self.assertEqual(captured["seed"], 7)
        self.assertIs(captured["skip_writing_prefix_cache"], True)
        self.assertEqual(row["request_controls"]["effective_sampling"], sampling)

    def test_request_receipt_refuses_any_apcv2_reuse(self):
        sampling = AB.sampling_controls(0)
        receipt = {
            "route": "ordinary",
            "cached_tokens": 1,
            "request_controls": controls(sampling),
        }
        with self.assertRaisesRegex(RuntimeError, "reused APCv2 state"):
            AB.validate_request_receipt("ord", receipt, sampling)
        receipt["cached_tokens"] = 0
        receipt["request_controls"]["skip_writing_prefix_cache"] = False
        with self.assertRaisesRegex(RuntimeError, "suppress APCv2 writes"):
            AB.validate_request_receipt("ord", receipt, sampling)


class PairwiseEvidenceTests(unittest.TestCase):
    def test_presence_penalty_is_reported_ineligible_without_false_claim(self):
        sampled = AB.sampling_controls(0.7)
        rows = [{"request_controls": controls(sampled)} for _ in range(4)]
        observation = AB.pairwise_cell_observation(
            "dk4",
            4,
            rows,
            status("external_draft", "external_draft", False, 4, pairwise=5),
            status("external_draft", "external_draft", False, 4, pairwise=5),
        )
        self.assertTrue(observation["selected"])
        self.assertFalse(observation["controls_eligible"])
        self.assertFalse(observation["expected_engagement"])
        self.assertFalse(observation["observed_used"])
        self.assertEqual(observation["ineligibility_reasons"], ["presence_penalty"])

    def test_ineligible_cell_refuses_pairwise_counter_contamination(self):
        sampled = AB.sampling_controls(0.7)
        rows = [{"request_controls": controls(sampled)} for _ in range(4)]
        before = status("external_draft", "external_draft", False, 4, pairwise=5)
        after = status("external_draft", "external_draft", False, 4, pairwise=6)
        with self.assertRaisesRegex(RuntimeError, "outside eligible"):
            AB.pairwise_cell_observation("dk4", 4, rows, before, after)

    def test_eligible_concurrent_cell_requires_positive_counter_delta(self):
        greedy = AB.sampling_controls(0)
        rows = [{"request_controls": controls(greedy)} for _ in range(4)]
        before = status("external_draft", "external_draft", False, 4, pairwise=2)
        after = status("external_draft", "external_draft", False, 4, pairwise=3)
        observation = AB.pairwise_cell_observation(
            "dk4", 4, rows, before, after
        )
        self.assertTrue(observation["expected_engagement"])
        self.assertEqual(observation["groups_delta"], 1)
        self.assertTrue(observation["observed_used"])
        with self.assertRaisesRegex(RuntimeError, "did not engage"):
            AB.pairwise_cell_observation("dk4", 4, rows, before, before)

    def test_eligible_singleton_cell_requires_and_claims_pairwise_use(self):
        greedy = AB.sampling_controls(0)
        row = {"request_controls": controls(greedy)}
        observation = AB.pairwise_cell_observation(
            "dk4",
            1,
            [row],
            status("external_draft", "external_draft", False, 4),
            status("external_draft", "external_draft", False, 4, pairwise=1),
        )
        self.assertTrue(observation["controls_eligible"])
        self.assertFalse(observation["concurrent"])
        self.assertTrue(observation["expected_engagement"])
        self.assertTrue(observation["observed_used"])

    def test_pairwise_counter_is_strict_and_cannot_advance_outside_gate(self):
        greedy = AB.sampling_controls(0)
        rows = [{"request_controls": controls(greedy)}]
        missing = status("external_draft", "external_draft", False, 4)
        del missing["scheduler"]["external_pairwise_selection_groups"]
        with self.assertRaisesRegex(RuntimeError, "counter is missing"):
            AB.pairwise_cell_observation("dk4", 1, rows, missing, missing)

        before = status("external_draft", "external_draft", False, 4)
        with self.assertRaisesRegex(RuntimeError, "did not engage"):
            AB.pairwise_cell_observation("dk4", 1, rows, before, before)

        ordinary_before = status("ordinary", "ordinary", False, None)
        ordinary_after = status("ordinary", "ordinary", False, None)
        ordinary_after["scheduler"]["external_pairwise_selection_groups"] = 1
        with self.assertRaisesRegex(RuntimeError, "outside eligible"):
            AB.pairwise_cell_observation(
                "ord", 1, rows, ordinary_before, ordinary_after
            )


class IngressEvidenceTests(unittest.TestCase):
    @staticmethod
    def snapshot(target_reached):
        value = status("external_draft", "external_draft", False, 4)
        value["settings"]["execution_policy"]["ingress_cohort"] = {
            "enabled": True,
            "target_lanes": 4,
        }
        value["counts"] = {
            "ingress_cohort_target_reached": target_reached,
        }
        # A lifetime scheduler maximum is intentionally irrelevant: warm-up
        # can populate it before any measured cell runs.
        value["scheduler"][
            "external_batched_prefill_max_cohort_width"
        ] = 4
        return value

    def test_warmup_only_ingress_evidence_is_refused(self):
        warm = self.snapshot(3)
        with self.assertRaisesRegex(RuntimeError, "timed cells"):
            AB.ingress_timed_observation("dk4v", warm, self.snapshot(3))

    def test_requested_ingress_policy_cannot_be_silently_absent(self):
        absent = status("external_draft", "external_draft", False, 4)
        absent["counts"] = {"ingress_cohort_target_reached": 0}
        with self.assertRaisesRegex(RuntimeError, "was not selected"):
            AB.ingress_timed_observation("dk4v", absent, absent)

    def test_requested_ingress_counter_is_strict(self):
        missing = self.snapshot(0)
        del missing["counts"]["ingress_cohort_target_reached"]
        with self.assertRaisesRegex(RuntimeError, "counter is missing"):
            AB.ingress_timed_observation("dk4v", missing, missing)
        malformed = self.snapshot(0)
        malformed["counts"]["ingress_cohort_target_reached"] = -1
        with self.assertRaisesRegex(RuntimeError, "non-negative integer"):
            AB.ingress_timed_observation("dk4v", malformed, malformed)

    def test_timed_target_cohort_delta_is_recorded(self):
        observation = AB.ingress_timed_observation(
            "dk4v", self.snapshot(3), self.snapshot(4)
        )
        self.assertTrue(observation["selected"])
        self.assertTrue(observation["requested"])
        self.assertTrue(observation["observed_used"])
        self.assertEqual(observation["target_reached_delta"], 1)

    def test_each_varlen_b4_cell_requires_physical_target_receipts(self):
        receipt = {
            "selected": True,
            "mechanism": "external_varlen_prefill",
            "target_lanes": 4,
            "admission_width": 4,
            "execution_width": 4,
            "formation_target_reached": True,
            "target_reached": True,
            "member_observed_used": True,
            "observed_used": True,
            "expired": False,
            "maximum_wait_ms": 250,
        }
        rows = [{"ingress_cohort": dict(receipt)} for _ in range(4)]
        observation = AB.ingress_cell_observation("dk4v", 4, rows)
        self.assertTrue(observation["target_reached"])
        rows[0]["ingress_cohort"]["execution_width"] = 2
        with self.assertRaisesRegex(RuntimeError, "did not physically execute"):
            AB.ingress_cell_observation("dk4v", 4, rows)

    def test_varlen_b1_cell_is_an_unbatched_control(self):
        rows = [
            {
                "ingress_cohort": {
                    "selected": True,
                    "mechanism": "external_varlen_prefill",
                    "target_lanes": 4,
                    "admission_width": 1,
                    "execution_width": 0,
                    "formation_target_reached": False,
                    "target_reached": False,
                    "member_observed_used": False,
                    "observed_used": False,
                    "expired": True,
                    "maximum_wait_ms": 250,
                }
            }
        ]
        observation = AB.ingress_cell_observation("dk4v", 1, rows)
        self.assertFalse(observation["observed_used"])
        rows[0]["ingress_cohort"]["observed_used"] = True
        with self.assertRaisesRegex(RuntimeError, "expired width-one"):
            AB.ingress_cell_observation("dk4v", 1, rows)

    def test_warmup_refuses_partial_b4_or_missing_varlen_engagement(self):
        receipt = {
            "selected": True,
            "mechanism": "external_varlen_prefill",
            "target_lanes": 4,
            "maximum_wait_ms": 250,
            "admission_width": 4,
            "execution_width": 2,
            "formation_target_reached": True,
            "target_reached": False,
            "member_observed_used": True,
            "observed_used": True,
            "expired": False,
        }
        cells = [
            (
                4,
                [{"ingress_cohort": dict(receipt)} for _ in range(4)],
                {},
                {},
            )
        ]
        warm = VarlenEvidenceTests.snapshot(1, 4)
        hot = VarlenEvidenceTests.snapshot(2, 8)
        with self.assertRaisesRegex(RuntimeError, "did not physically execute"):
            AB.warmup_observation("dk4v", warm, hot, cells)


class VarlenEvidenceTests(unittest.TestCase):
    @staticmethod
    def snapshot(calls, padding, *, selected=True):
        return {
            "execution": {
                "varlen_dense_mlp": {
                    "selected": selected,
                    "observed_used": bool(calls and padding),
                    "counters": {
                        "mlp_compaction_calls": calls,
                        "padding_token_rows": padding,
                    },
                }
            }
        }

    def test_warmup_only_varlen_evidence_is_refused(self):
        warm = self.snapshot(5, 100)
        with self.assertRaisesRegex(RuntimeError, "timed cells"):
            AB.varlen_timed_observation("dk4v", warm, self.snapshot(5, 100))
        with self.assertRaisesRegex(RuntimeError, "warm-up cells"):
            AB.varlen_timed_observation(
                "dk4v", warm, self.snapshot(5, 100), phase="warm-up"
            )

    def test_pre_execution_warmup_allows_empty_initial_counters_only(self):
        cold = self.snapshot(0, 0)
        cold["execution"]["varlen_dense_mlp"]["counters"] = {}
        hot = self.snapshot(2, 8)
        observation = AB.varlen_timed_observation(
            "dk4v", cold, hot, phase="warm-up"
        )
        self.assertTrue(observation["observed_used"])
        with self.assertRaisesRegex(RuntimeError, "counter is missing"):
            AB.varlen_timed_observation("dk4v", cold, hot)

    def test_timed_varlen_requires_compaction_and_padding_deltas(self):
        warm = self.snapshot(5, 100)
        observation = AB.varlen_timed_observation(
            "dk4v", warm, self.snapshot(7, 140)
        )
        self.assertTrue(observation["selected"])
        self.assertTrue(observation["observed_used"])
        self.assertEqual(
            observation["counter_deltas"],
            {"mlp_compaction_calls": 2, "padding_token_rows": 40},
        )
        with self.assertRaisesRegex(RuntimeError, "with padding"):
            AB.varlen_timed_observation(
                "dk4v", warm, self.snapshot(7, 100)
            )

    def test_each_varlen_b4_cell_requires_its_own_counter_deltas(self):
        before = self.snapshot(5, 100)
        observation = AB.varlen_cell_observation(
            "dk4v", 4, before, self.snapshot(7, 140)
        )
        self.assertTrue(observation["target_expected"])
        self.assertTrue(observation["observed_used"])
        with self.assertRaisesRegex(RuntimeError, "B4 cell did not engage"):
            AB.varlen_cell_observation(
                "dk4v", 4, before, self.snapshot(7, 100)
            )

    def test_varlen_b1_cell_must_not_advance_target_counters(self):
        before = self.snapshot(5, 100)
        observation = AB.varlen_cell_observation("dk4v", 1, before, before)
        self.assertFalse(observation["target_expected"])
        self.assertFalse(observation["observed_used"])
        with self.assertRaisesRegex(RuntimeError, "B1/control cell"):
            AB.varlen_cell_observation(
                "dk4v", 1, before, self.snapshot(6, 120)
            )

    def test_first_warmup_b1_cell_allows_absent_zero_counters(self):
        cold = self.snapshot(0, 0)
        cold["execution"]["varlen_dense_mlp"]["counters"] = {}
        observation = AB.varlen_cell_observation(
            "dk4v", 1, cold, cold, phase="warm-up"
        )
        self.assertEqual(
            observation["counter_deltas"],
            {"mlp_compaction_calls": 0, "padding_token_rows": 0},
        )

    def test_requested_varlen_counters_are_strict(self):
        missing = self.snapshot(1, 1)
        del missing["execution"]["varlen_dense_mlp"]["counters"][
            "padding_token_rows"
        ]
        with self.assertRaisesRegex(RuntimeError, "counter is missing"):
            AB.varlen_timed_observation("dk4v", missing, missing)
        malformed = self.snapshot(1, -1)
        with self.assertRaisesRegex(RuntimeError, "non-negative integer"):
            AB.varlen_timed_observation("dk4v", malformed, malformed)


class CellIsolationTests(unittest.TestCase):
    def test_warmup_and_timed_nonce_generator_has_exact_geometry(self):
        warmup = [
            AB.request_nonce("test", 2, "code", 4, 0.7, index)
            for index in range(4)
        ]
        timed = [
            AB.request_nonce("test", 2, "code", 4, 0.7, index)
            for index in range(4)
        ]
        self.assertEqual(warmup, timed)
        self.assertEqual(len(set(timed)), 4)
        self.assertTrue(all(len(value) == 16 for value in timed))

    def test_serial_b1_requests_use_distinct_arm_invariant_nonces(self):
        args = mock.Mock(
            nonce_salt="test",
            b1_prompts=4,
            max_tokens=2,
            timeout=1,
        )

        def collect(arm):
            seen = []

            def request(_url, _item, **kwargs):
                seen.append(kwargs["nonce"])
                return {"completion_tokens": 1}

            with mock.patch.object(AB, "_request", request):
                AB.run_cell(
                    "http://unused",
                    arm,
                    "code",
                    1,
                    0,
                    args,
                    seed_base=10,
                    rep=2,
                )
            return seen

        ordinary = collect("ord")
        external = collect("dk4v")
        self.assertEqual(ordinary, external)
        self.assertEqual(len(set(ordinary)), 4)


class SummaryGateTests(unittest.TestCase):
    @staticmethod
    def cell(arm, digest, *, rep=1):
        return {
            "arm": arm,
            "temperature": 0,
            "rep": rep,
            "workload": "code",
            "width": 1,
            "rows": [
                {
                    "prompt_index": 0,
                    "output_sha256": digest,
                    "output": digest,
                }
            ],
        }

    def test_ordv_is_compared_to_ord_not_used_as_the_reference(self):
        gate = SUMMARY.greedy_gate(
            [self.cell("ord", "ordinary"), self.cell("ordv", "varlen")]
        )
        self.assertEqual(len(gate), 1)
        self.assertEqual(gate[0]["arm"], "ordv")
        self.assertFalse(gate[0]["equal"])

    def test_missing_ordinary_reference_fails_the_gate_explicitly(self):
        gate = SUMMARY.greedy_gate([self.cell("dk4v", "candidate")])
        self.assertEqual(gate[0]["reason"], "missing_ordinary_reference")
        self.assertFalse(gate[0]["equal"])

    def test_duplicate_ordinary_reference_is_ambiguous(self):
        with self.assertRaisesRegex(ValueError, "duplicate ordinary"):
            SUMMARY.greedy_gate(
                [self.cell("ord", "a"), self.cell("ord", "a")]
            )


if __name__ == "__main__":
    unittest.main()
