from mlx2.contracts import Capability
from mlx2.qualification import effective_route_capabilities


def test_one_lane_profile_does_not_select_continuous_batch():
    implemented = frozenset(
        {
            Capability.TEXT,
            Capability.CONTINUOUS_BATCH,
            Capability.EXTERNAL_DRAFT,
            Capability.MTP,
            Capability.SEGMENTED_MTP,
            Capability.PROMPT_LOOKUP,
        }
    )
    selected = effective_route_capabilities(
        implemented,
        mtp=False,
        external_draft=True,
        prompt_lookup=False,
        max_lanes=1,
    )
    assert selected == frozenset({Capability.TEXT, Capability.EXTERNAL_DRAFT})


def test_two_lane_profile_can_select_continuous_batch():
    implemented = frozenset({Capability.TEXT, Capability.CONTINUOUS_BATCH})
    assert (
        effective_route_capabilities(
            implemented,
            mtp=False,
            external_draft=False,
            prompt_lookup=False,
            max_lanes=2,
        )
        == implemented
    )
