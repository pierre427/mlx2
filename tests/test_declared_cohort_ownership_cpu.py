"""Pure CPU contracts for model-neutral declared-cohort identity."""

import pytest

from mlx2.runtime.cohort_ownership import declared_cohort_key


def test_declared_cohort_key_is_tenant_and_width_scoped():
    assert declared_cohort_key(None) is None
    assert declared_cohort_key(
        {"tenant_id": "tenant-a", "id": "request-1", "size": 4}
    ) == ("tenant-a", "request-1", 4)
    assert declared_cohort_key(
        {"tenant_id": "tenant-b", "id": "request-1", "size": 4}
    ) != declared_cohort_key(
        {"tenant_id": "tenant-a", "id": "request-1", "size": 4}
    )


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ([], TypeError),
        ({"id": "c", "size": 2}, ValueError),
        ({"tenant_id": "", "id": "c", "size": 2}, ValueError),
        ({"tenant_id": "t", "id": "", "size": 2}, ValueError),
        ({"tenant_id": "t", "id": "c", "size": True}, ValueError),
        ({"tenant_id": "t", "id": "c", "size": 0}, ValueError),
    ],
)
def test_present_malformed_cohort_never_falls_back_to_undeclared(value, error):
    with pytest.raises(error):
        declared_cohort_key(value)
