from types import SimpleNamespace

from mlx2.runtime.models import nemotron_moe_router as router


def test_router_candidate_constants_match_super_geometry():
    assert router.NUM_EXPERTS == 512
    assert router.TOP_K == 22
    assert router.MAX_ROWS == 16
    assert router.ROUTED_SCALE == 5.0


def test_router_admission_rejects_non_arrays_before_metal_checks():
    logits = SimpleNamespace(shape=(1, 1, 511), size=511, dtype=None)
    bias = SimpleNamespace(shape=(512,), dtype=None)
    admission = router.admit(
        logits,
        bias,
        top_k=22,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=5.0,
    )
    assert not admission.accepted
    assert admission.reason == "logits must end in 512"
