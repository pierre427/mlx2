import importlib.util
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
SPEC = importlib.util.spec_from_file_location(
    "arbitrary_traffic",
    ROOT / "scripts/research/prepare_deterministic_arbitrary_traffic.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
CAMPAIGN_SPEC = importlib.util.spec_from_file_location(
    "arbitrary_campaign",
    ROOT / "scripts/research/deterministic_arbitrary_http_campaign.py",
)
CAMPAIGN = importlib.util.module_from_spec(CAMPAIGN_SPEC)
sys.modules[CAMPAIGN_SPEC.name] = CAMPAIGN
CAMPAIGN_SPEC.loader.exec_module(CAMPAIGN)


def test_plan_is_replay_stable_and_multiturn():
    left = MODULE.traffic_plan(17)
    right = MODULE.traffic_plan(17)
    assert left == right
    assert left != MODULE.traffic_plan(18)
    assert len(left) == 18
    assert len({item["user"] for item in left}) == 5
    assert max(item["turn"] for item in left) == 4
    for user in MODULE.USERS:
        lengths = [
            item["target_prompt_tokens"] for item in left if item["user"] == user
        ]
        assert lengths == sorted(lengths)


def test_plan_has_fixed_waves_and_speculation():
    plan = MODULE.traffic_plan()
    assert {item["wave"] for item in plan} == set(range(6))
    assert all(item["draft_depth"] == 2 for item in plan)
    assert all(0 <= item["arrival_offset_ms"] <= 400 for item in plan)
    assert all(1025 <= item["target_prompt_tokens"] <= 8000 for item in plan)


def test_ordinary_string_receipt_is_a_supported_product_shape():
    row = {"case_id": "case", "prompt_tokens": 128}
    response = (
        200,
        {
            "choices": [{"finish_reason": "length", "message": {"content": "x"}}],
            "usage": {"prompt_tokens": 128, "completion_tokens": 4},
            "mlx2": {"route_receipt": "ordinary", "scheduler": "unavailable"},
        },
    )
    summary = CAMPAIGN.response_summary(row, response, native=False)
    assert summary["route_receipt"] == "ordinary"
    assert summary["draft_accepted"] == 0
    assert summary["scheduler"] == {}


def test_scheduler_snapshot_uses_engine_status_mechanism_counters():
    class Engine:
        def __init__(self):
            self.values = deque(
                [
                    {"scheduler": "loading"},
                    {"scheduler": {"batch_geometry_rounds": 3}},
                ]
            )

        def status(self):
            if len(self.values) > 1:
                return self.values.popleft()
            return self.values[0]

    assert CAMPAIGN.wait_for_scheduler_snapshot(Engine(), timeout=0.2) == {
        "batch_geometry_rounds": 3
    }
