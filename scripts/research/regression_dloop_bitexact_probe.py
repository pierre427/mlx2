#!/usr/bin/env python3
"""Run the frozen first-divergence producer with a recorded QMV diagnostic mode.

Keep this wrapper outside the frozen checkout on the test host. Its own digest,
mode lifetime and dispatch evidence live in a companion receipt; the original
producer's output and source identity remain intact. No qualification claim.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--diagnostic-receipt", type=Path, required=True)
    args, producer_args = parser.parse_known_args()
    sys.path.insert(0, str(args.checkout / "scripts"))
    from dloop_first_divergence_probe import main as run_probe
    from mlx2.runtime.verify_bitexact import VerifyBitexactPolicy, bind_for_serving
    from mlx2.serving import runtime_identity

    runtime_before = runtime_identity()
    report = {
        "schema": "mlx2.dloop-bitexact-diagnostic.v1",
        "qualification_claim": False,
        "performance_claim": False,
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "producer_arguments": producer_args,
        "runtime_before": runtime_before,
        "completed": False,
    }
    handle = None
    before = None
    try:
        handle, settings = bind_for_serving(VerifyBitexactPolicy(True))
        report["settings"] = settings
        before = handle.begin_request(explicit=True)
        run_probe(producer_args)
        report["completed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["runtime_after"] = runtime_identity()
        report["source_stable"] = report["runtime_after"] == runtime_before
        if handle is not None:
            if before is not None:
                report["mode_evidence"] = handle.request_receipt(before)
            handle.remove()
        with args.diagnostic_receipt.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
