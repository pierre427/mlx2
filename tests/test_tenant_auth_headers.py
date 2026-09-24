"""Tenant authentication header checks without importing the MLX runtime."""

from email.message import Message

import pytest

from mlx2.tenant_auth import (
    TenantAuthenticator,
    TenantAuthError,
    TenantPrincipal,
    hash_api_key,
)


@pytest.mark.parametrize("name", ["Authorization", "x-api-key", "X-Tenant-ID"])
@pytest.mark.parametrize("second", ["same", "different"])
def test_duplicate_identity_headers_fail_closed(name, second):
    key = "mlx2k_fixture"
    auth = TenantAuthenticator(
        keys={hash_api_key(key): (TenantPrincipal("tenant-a", "api_key"), False)}
    )
    values = {
        "Authorization": f"Bearer {key}",
        "x-api-key": key,
        "X-Tenant-ID": "tenant-a",
    }
    headers = Message()
    headers["Authorization"] = values["Authorization"]
    if name != "Authorization":
        headers[name] = values[name]
    headers[name] = values[name] if second == "same" else "untrusted"
    with pytest.raises(TenantAuthError) as caught:
        auth.authenticate(headers)
    assert caught.value.reason == "ambiguous"
    assert auth.status()["verified"]["api_key"] == 0


def test_distinct_headers_with_identical_credential_remain_supported():
    key = "mlx2k_fixture"
    auth = TenantAuthenticator(
        keys={hash_api_key(key): (TenantPrincipal("tenant-a", "api_key"), False)}
    )
    headers = Message()
    headers["Authorization"] = f"Bearer {key}"
    headers["x-api-key"] = key
    assert auth.authenticate(headers).tenant == "tenant-a"
