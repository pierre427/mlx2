#!/usr/bin/env python3
"""CPU microbench: per-request cost of tenant credential verification.

Pure Python, no mlx; safe to run anywhere.  Refuses to report an arm whose
verification counter did not move (mechanism gate).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx2.tenant_auth import (  # noqa: E402
    TenantAuthenticator,
    TenantPrincipal,
    generate_api_key,
    hash_api_key,
    mint_token,
)


def _arm(auth, headers, iterations):
    before = sum(auth.status()["verified"].values())
    started = time.perf_counter()
    for _ in range(iterations):
        auth.authenticate(headers)
    elapsed = time.perf_counter() - started
    moved = sum(auth.status()["verified"].values()) - before
    if moved != iterations:
        raise SystemExit(f"mechanism counter moved {moved}, expected {iterations}")
    return elapsed / iterations * 1e6


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    secret = b"x" * 48
    results = {}
    for count in (1, 100, 1000):
        keys = {}
        last = None
        for index in range(count):
            last = generate_api_key()
            keys[hash_api_key(last)] = (
                TenantPrincipal(tenant=f"t{index}", method="api_key", key_id=f"k{index}"),
                False,
            )
        auth = TenantAuthenticator(keys=keys, token_secret=secret)
        results[f"api_key_{count}_keys_us"] = round(
            _arm(auth, {"Authorization": f"Bearer {last}"}, args.iterations), 3
        )
    auth = TenantAuthenticator(token_secret=secret)
    token = mint_token(secret, "tenant", ttl_seconds=3600)
    results["token_us"] = round(
        _arm(auth, {"x-api-key": token}, args.iterations), 3
    )
    results["iterations"] = args.iterations
    text = json.dumps(results, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
