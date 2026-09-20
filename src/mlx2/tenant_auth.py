"""Opt-in authenticated tenant identity for the HTTP surface.

Without this module a tenant is whatever ``X-Tenant-ID`` a client sends.  When
an operator configures an API-keys file and/or a tenant-token secret, the
tenant instead comes from a verified credential and every guarded request
without one fails closed.

Two credential kinds are accepted, in ``Authorization: Bearer`` or
``x-api-key``:

* API keys (``mlx2k_...``): only their SHA-256 is stored, in a JSON keys file
  mapping each key to a tenant.  Keys are 256-bit random, so an unsalted digest
  is the right storage (the vLLM ``--api-key`` / GitHub-token pattern).
* Tenant tokens (``mlx2t1.<payload>.<mac>``): HMAC-SHA256 over the payload with
  a server secret; the payload carries ``sub`` (tenant), ``iat``, ``exp`` and
  optional ``scp`` scopes.  The MAC is checked before the payload is parsed.

Nothing here logs, counts or returns credential material or tenant names in
metrics: failure reasons come from a fixed vocabulary.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import stat
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("mlx2.tenant_auth")

TENANT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
KEY_ID_PATTERN = TENANT_PATTERN
KEY_PREFIX = "mlx2k_"
TOKEN_PREFIX = "mlx2t1."
TOKEN_MAC_DOMAIN = b"mlx2-tenant-token-v1."
MAX_CREDENTIAL_BYTES = 4096
DEFAULT_TOKEN_MAX_TTL = 7 * 24 * 3600
TOKEN_IAT_SKEW_SECONDS = 30
SCOPES = frozenset({"inference", "adapters"})
DEFAULT_SCOPES = ("inference",)
HEADER_POLICIES = ("must-match", "ignore")

# Fixed failure vocabulary: (HTTP status, public message).  Messages never
# echo the presented credential or the claimed tenant.
FAILURES = {
    "missing": (401, "tenant authentication required"),
    "malformed": (401, "malformed tenant credential"),
    "ambiguous": (401, "conflicting Authorization and x-api-key credentials"),
    "unknown_key": (401, "invalid tenant credential"),
    "disabled": (401, "invalid tenant credential"),
    "bad_signature": (401, "invalid tenant credential"),
    "expired": (401, "tenant credential expired"),
    "tenant_mismatch": (403, "X-Tenant-ID does not match the authenticated tenant"),
    "scope": (403, "tenant credential lacks the required scope"),
}


class TenantAuthError(Exception):
    """A request failed tenant authentication; ``reason`` is in FAILURES."""

    def __init__(self, reason: str):
        if reason not in FAILURES:
            raise ValueError(f"unknown tenant auth failure {reason!r}")
        self.reason = reason
        self.status, self.public_message = FAILURES[reason]
        super().__init__(self.public_message)


@dataclass(frozen=True)
class TenantPrincipal:
    tenant: str
    method: str  # "api_key" | "token"
    key_id: str | None = None
    scopes: frozenset = frozenset(DEFAULT_SCOPES)
    # Validated but uninterpreted per-key metadata (e.g. an rm05 adapter
    # allowlist); kept out of receipts, logs and metrics.
    extra: dict = field(default_factory=dict, compare=False, hash=False)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("invalid base64url")
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    return KEY_PREFIX + _b64(secrets.token_bytes(32))


def validate_tenant(tenant) -> str:
    if not isinstance(tenant, str) or not TENANT_PATTERN.match(tenant):
        raise ValueError(
            "tenant ids must match [A-Za-z0-9][A-Za-z0-9._:-]{0,127}"
        )
    return tenant


def _validate_scopes(value, where) -> frozenset:
    if value is None:
        return frozenset(DEFAULT_SCOPES)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) for item in value)
    ):
        raise ValueError(f"{where} scopes must be a nonempty list of strings")
    unknown = set(value) - SCOPES
    if unknown:
        raise ValueError(f"{where} has unknown scopes: {', '.join(sorted(unknown))}")
    return frozenset(value)


def _check_owner_file(path: Path, *, secret: bool, flag: str):
    metadata = os.stat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{flag} must be a regular file")
    if metadata.st_uid != os.getuid():
        raise ValueError(f"{flag} must be owned by the process uid")
    mode = stat.S_IMODE(metadata.st_mode)
    if secret and mode != 0o600:
        raise ValueError(f"{flag} permissions must be exactly 0600")
    if not secret and mode & 0o022:
        raise ValueError(f"{flag} must not be group- or world-writable")


def load_keys_file(path) -> dict:
    """Return ``{sha256_hex: TenantPrincipal}`` from a validated keys file."""
    candidate = Path(path).expanduser().resolve()
    _check_owner_file(candidate, secret=False, flag="--tenant-auth-keys-file")
    document = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"version", "keys"}:
        raise ValueError("tenant keys file must be {version, keys}")
    if document["version"] != 1:
        raise ValueError("tenant keys file version must be 1")
    if not isinstance(document["keys"], list) or not document["keys"]:
        raise ValueError("tenant keys file must list at least one key")
    entries = {}
    key_ids = set()
    allowed = {"key_id", "tenant", "sha256", "scopes", "disabled", "adapters"}
    for index, item in enumerate(document["keys"]):
        where = f"tenant key #{index}"
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be an object")
        unknown = set(item) - allowed
        if unknown:
            raise ValueError(f"{where} has unknown fields: {', '.join(sorted(unknown))}")
        missing = {"key_id", "tenant", "sha256"} - set(item)
        if missing:
            raise ValueError(f"{where} is missing: {', '.join(sorted(missing))}")
        key_id = item["key_id"]
        if not isinstance(key_id, str) or not KEY_ID_PATTERN.match(key_id):
            raise ValueError(f"{where} has an invalid key_id")
        if key_id in key_ids:
            raise ValueError(f"duplicate tenant key_id {key_id!r}")
        key_ids.add(key_id)
        tenant = validate_tenant(item["tenant"])
        digest = item["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{where} sha256 must be 64 lowercase hex digits")
        if digest in entries:
            raise ValueError(f"{where} duplicates another key's digest")
        disabled = item.get("disabled", False)
        if not isinstance(disabled, bool):
            raise ValueError(f"{where} disabled must be a boolean")
        extra = {}
        if "adapters" in item:
            adapters = item["adapters"]
            if not isinstance(adapters, list) or not all(
                isinstance(name, str) and name for name in adapters
            ):
                raise ValueError(f"{where} adapters must be a list of names")
            extra["adapters"] = tuple(adapters)
        principal = TenantPrincipal(
            tenant=tenant,
            method="api_key",
            key_id=key_id,
            scopes=_validate_scopes(item.get("scopes"), where),
            extra=extra,
        )
        entries[digest] = (principal, disabled)
    return entries


def load_token_secret(*, path=None, env=None) -> bytes | None:
    if path and env:
        raise ValueError("configure only one tenant token secret source")
    if path:
        candidate = Path(path).expanduser().resolve()
        _check_owner_file(
            candidate, secret=True, flag="--tenant-auth-token-secret-file"
        )
        secret = candidate.read_bytes().strip()
    elif env:
        secret = os.environ.get(str(env), "").encode()
    else:
        return None
    if len(secret) < 32:
        raise ValueError("tenant token secret must contain at least 32 bytes")
    return secret


def mint_token(
    secret: bytes,
    tenant: str,
    *,
    ttl_seconds: int,
    scopes=DEFAULT_SCOPES,
    now: float | None = None,
) -> str:
    validate_tenant(tenant)
    _validate_scopes(list(scopes), "token")
    issued = int(time.time() if now is None else now)
    payload = _b64(
        json.dumps(
            {
                "sub": tenant,
                "iat": issued,
                "exp": issued + int(ttl_seconds),
                "scp": sorted(scopes),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    mac = hmac.new(secret, TOKEN_MAC_DOMAIN + payload.encode(), hashlib.sha256)
    return f"{TOKEN_PREFIX}{payload}.{_b64(mac.digest())}"


class TenantAuthenticator:
    """Verify tenant credentials and keep sync-free, secret-free counters."""

    def __init__(
        self,
        *,
        keys=None,
        token_secret: bytes | None = None,
        header_policy: str = "must-match",
        token_max_ttl: int = DEFAULT_TOKEN_MAX_TTL,
        cache_isolation: str = "tenant",
        clock=time.time,
    ):
        if not keys and token_secret is None:
            raise ValueError("tenant auth needs a keys file or a token secret")
        if header_policy not in HEADER_POLICIES:
            raise ValueError(f"tenant header policy must be one of {HEADER_POLICIES}")
        if not 60 <= int(token_max_ttl) <= 366 * 24 * 3600:
            raise ValueError("tenant token max TTL must be 60 s to 366 days")
        self._keys = dict(keys or {})
        self._token_secret = token_secret
        self.header_policy = header_policy
        self.token_max_ttl = int(token_max_ttl)
        self.cache_isolation = cache_isolation
        self._clock = clock
        self._lock = threading.Lock()
        self.verified = {"api_key": 0, "token": 0}
        self.failures = {reason: 0 for reason in FAILURES}
        self.header_counts = {"tenant_header_matched": 0, "tenant_header_ignored": 0}

    @property
    def methods(self):
        methods = []
        if self._keys:
            methods.append("api_key")
        if self._token_secret is not None:
            methods.append("token")
        return methods

    def uses_secret(self, secret) -> bool:
        if self._token_secret is None or secret is None:
            return False
        if isinstance(secret, str):
            secret = secret.encode()
        return hmac.compare_digest(self._token_secret, secret)

    # -- verification -----------------------------------------------------

    def _fail(self, reason):
        with self._lock:
            self.failures[reason] += 1
        return TenantAuthError(reason)

    @staticmethod
    def _presented(headers):
        authorization = headers.get("Authorization")
        api_key = headers.get("x-api-key")
        bearer = None
        if authorization is not None:
            scheme, _, value = authorization.partition(" ")
            if scheme.lower() != "bearer" or not value.strip():
                return None, "malformed"
            bearer = value.strip()
        if api_key is not None:
            api_key = api_key.strip()
            if not api_key:
                return None, "malformed"
        if bearer is not None and api_key is not None:
            if not hmac.compare_digest(bearer.encode(), api_key.encode()):
                return None, "ambiguous"
        credential = bearer if bearer is not None else api_key
        if credential is None:
            return None, "missing"
        if len(credential.encode("utf-8", "surrogatepass")) > MAX_CREDENTIAL_BYTES:
            return None, "malformed"
        return credential, None

    def _verify_key(self, credential):
        digest = hash_api_key(credential)
        # Lookup is keyed by the SHA-256 of the presented key, never by the
        # key itself: any hash/compare timing depends on a digest the attacker
        # cannot steer toward a stored digest without a preimage (the same
        # argument as vLLM's digest compare).  O(1) in the number of keys.
        match = self._keys.get(digest)
        if match is None:
            raise self._fail("unknown_key")
        principal, disabled = match
        if disabled:
            raise self._fail("disabled")
        return principal

    def _verify_token(self, credential):
        if self._token_secret is None:
            raise self._fail("unknown_key")
        body = credential[len(TOKEN_PREFIX) :]
        payload_b64, dot, mac_b64 = body.partition(".")
        if not dot or "." in mac_b64:
            raise self._fail("malformed")
        try:
            presented_mac = _unb64(mac_b64)
        except ValueError:
            raise self._fail("malformed") from None
        expected = hmac.new(
            self._token_secret,
            TOKEN_MAC_DOMAIN + payload_b64.encode("ascii", "replace"),
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(presented_mac, expected):
            raise self._fail("bad_signature")
        # Only authenticated bytes are parsed below.
        try:
            claims = json.loads(_unb64(payload_b64))
            if not isinstance(claims, dict) or set(claims) - {"sub", "iat", "exp", "scp"}:
                raise ValueError("claims")
            tenant = validate_tenant(claims["sub"])
            issued, expires = claims["iat"], claims["exp"]
            if not all(
                isinstance(v, int) and not isinstance(v, bool) for v in (issued, expires)
            ):
                raise ValueError("times")
            scopes = _validate_scopes(claims.get("scp"), "token")
        except (KeyError, ValueError, TypeError):
            raise self._fail("malformed") from None
        now = self._clock()
        if (
            expires <= now
            or expires - issued > self.token_max_ttl
            or issued > now + TOKEN_IAT_SKEW_SECONDS
        ):
            raise self._fail("expired")
        return TenantPrincipal(tenant=tenant, method="token", scopes=scopes)

    def authenticate(self, headers, *, required_scope="inference") -> TenantPrincipal:
        """Return the verified principal or raise :class:`TenantAuthError`."""
        credential, problem = self._presented(headers)
        if problem is not None:
            raise self._fail(problem)
        if credential.startswith(TOKEN_PREFIX):
            principal = self._verify_token(credential)
        else:
            principal = self._verify_key(credential)
        claimed = headers.get("X-Tenant-ID")
        if claimed is not None:
            if self.header_policy == "must-match":
                if not hmac.compare_digest(
                    claimed.encode("utf-8", "surrogatepass"),
                    principal.tenant.encode("utf-8"),
                ):
                    raise self._fail("tenant_mismatch")
                header_event = "tenant_header_matched"
            else:
                header_event = "tenant_header_ignored"
        else:
            header_event = None
        if required_scope not in principal.scopes:
            raise self._fail("scope")
        with self._lock:
            self.verified[principal.method] += 1
            if header_event is not None:
                self.header_counts[header_event] += 1
        return principal

    # -- reporting ----------------------------------------------------------

    def status(self) -> dict:
        tenants = {principal.tenant for principal, _ in self._keys.values()}
        with self._lock:
            return {
                "enabled": True,
                "methods": self.methods,
                "header_policy": self.header_policy,
                "cache_isolation": self.cache_isolation,
                "token_max_ttl_seconds": self.token_max_ttl,
                "configured_keys": len(self._keys),
                "configured_key_tenants": len(tenants),
                "verified": dict(self.verified),
                "failures": dict(self.failures),
                **self.header_counts,
            }

    def prometheus(self) -> str:
        with self._lock:
            verified = dict(self.verified)
            failures = dict(self.failures)
            header_counts = dict(self.header_counts)
        lines = [
            "# HELP mlx2_tenant_auth_enabled Whether authenticated tenant identity is enforced.",
            "# TYPE mlx2_tenant_auth_enabled gauge",
            "mlx2_tenant_auth_enabled 1",
            "# HELP mlx2_tenant_auth_total Requests whose tenant was verified, by credential method.",
            "# TYPE mlx2_tenant_auth_total counter",
        ]
        lines += [
            f'mlx2_tenant_auth_total{{outcome="verified",method="{method}"}} {count}'
            for method, count in sorted(verified.items())
        ]
        lines += [
            "# HELP mlx2_tenant_auth_failures_total Requests refused by tenant authentication, by reason.",
            "# TYPE mlx2_tenant_auth_failures_total counter",
        ]
        lines += [
            f'mlx2_tenant_auth_failures_total{{reason="{reason}"}} {count}'
            for reason, count in sorted(failures.items())
        ]
        lines += [
            "# HELP mlx2_tenant_auth_header_total X-Tenant-ID headers seen on authenticated requests.",
            "# TYPE mlx2_tenant_auth_header_total counter",
        ]
        lines += [
            f'mlx2_tenant_auth_header_total{{event="{event}"}} {count}'
            for event, count in sorted(header_counts.items())
        ]
        return "\n".join(lines) + "\n"


def configured(
    *,
    keys_file=None,
    token_secret_file=None,
    token_secret_env=None,
    header_policy="must-match",
    token_max_ttl=DEFAULT_TOKEN_MAX_TTL,
    cache_isolation="tenant",
):
    """Build an authenticator from CLI settings, or ``None`` when auth is off."""
    keys = load_keys_file(keys_file) if keys_file else None
    secret = load_token_secret(path=token_secret_file, env=token_secret_env)
    if keys is None and secret is None:
        return None
    return TenantAuthenticator(
        keys=keys,
        token_secret=secret,
        header_policy=header_policy,
        token_max_ttl=token_max_ttl,
        cache_isolation=cache_isolation,
    )


def _main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m mlx2.tenant_auth",
        description="mint mlx2 tenant API keys and tokens",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    new_key = commands.add_parser(
        "new-key", help="print a fresh API key and its keys-file entry"
    )
    new_key.add_argument("--tenant", required=True)
    new_key.add_argument("--key-id", required=True)
    new_key.add_argument("--scope", action="append", choices=sorted(SCOPES))
    token = commands.add_parser("mint-token", help="print a signed tenant token")
    token.add_argument("--tenant", required=True)
    token.add_argument("--ttl-seconds", type=int, default=24 * 3600)
    token.add_argument("--scope", action="append", choices=sorted(SCOPES))
    source = token.add_mutually_exclusive_group(required=True)
    source.add_argument("--secret-file")
    source.add_argument("--secret-env")
    args = parser.parse_args(argv)
    if args.command == "new-key":
        validate_tenant(args.tenant)
        key = generate_api_key()
        entry = {
            "key_id": args.key_id,
            "tenant": args.tenant,
            "sha256": hash_api_key(key),
            "scopes": sorted(args.scope or DEFAULT_SCOPES),
        }
        # The key goes to stdout once; only the entry belongs in the file.
        print(json.dumps({"api_key": key, "keys_file_entry": entry}, indent=2))
        return 0
    secret = load_token_secret(path=args.secret_file, env=args.secret_env)
    print(
        mint_token(
            secret,
            args.tenant,
            ttl_seconds=args.ttl_seconds,
            scopes=tuple(args.scope or DEFAULT_SCOPES),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
