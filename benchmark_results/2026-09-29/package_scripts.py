#!/usr/bin/env python3
"""Copy the exact campaign script set, removing only the local home prefix.

The manifest records hashes of both the run-time originals and published
copies. The home-prefix rewrite keeps model path defaults user-independent.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = HERE / "scripts"
SOURCES = {
    "requal": [
        "check_flash_tensorfold_qmv.py", "flash_memory_probe.py",
        "run_flash_moe_isolation.py",
        "queue_features.py", "queue_perf.py",
        "queue_smoke.py", "queue_stress.py", "run.py", "run_features.py",
        "run_perf.py", "run_preflight.py", "run_stress.py",
        "trace_stress_swap.py", "monitor_host_memory.py",
    ],
    "series": [
        "campaign_config.py", "concurrency_probe.py", "experimental_job.py",
        "extra_common.py", "ladder.py", "sanity_20x20.py", "thermal_ladder.py",
    ],
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    manifest = {"schema": "mlx2.public-benchmark-script-snapshot.v1", "files": []}
    for group, names in SOURCES.items():
        source_dir = ROOT / "qualification/runs" / (
            "requal-20260928" if group == "requal" else "series-20260924"
        )
        target_dir = DEST / group
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            source = source_dir / name
            original = source.read_bytes()
            text = original.decode("utf-8")
            # These defaults are path examples, not evidence. At runtime the
            # user's own home replaces the campaign host's absolute prefix.
            home_prefix_rewritten = bool(re.search(r'"/Users/(?!Shared/)[^/]+/', text))
            local_service_label = f"com.{Path.home().name}."
            local_service_label_rewritten = local_service_label in text
            public = re.sub(r'"/Users/(?!Shared/)[^/]+/', 'str(Path.home()) + "/', text)
            public = public.replace(local_service_label, "com.localuser.")
            if re.search(r"/Users/(?!Shared/)[^/]+/", public):
                raise RuntimeError(f"unredacted home path in {source}")
            data = public.encode("utf-8")
            target = target_dir / name
            target.write_bytes(data)
            manifest["files"].append({
                "source": str(source.relative_to(ROOT)),
                "published": str(target.relative_to(HERE)),
                "source_sha256": digest(original),
                "published_sha256": digest(data),
                "home_prefix_rewritten": home_prefix_rewritten,
                "local_service_label_rewritten": local_service_label_rewritten,
            })
    policy_groups = {
        "series": sorted((ROOT / "qualification/runs/series-20260924/policies").glob("*.json")),
        "requal": [
            ROOT / "qualification/runs/requal-20260928/policies/flash-next-tensorfold-qmv-candidate.json",
            ROOT / "qualification/runs/requal-20260928/policies/muse-lane-no-group-candidate.json",
            ROOT / "qualification/runs/requal-20260928/policies/muse-pld-rotating-candidate.json",
            ROOT / "qualification/runs/requal-20260928/policies/nemotron35-lightning-mtp2-width8.json",
        ],
    }
    for group, sources in policy_groups.items():
        policy_target = DEST / group / "policies"
        policy_target.mkdir(parents=True, exist_ok=True)
        for source in sources:
            original = source.read_bytes()
            public = re.sub(r"/Users/(?!Shared/)[^/]+/", "${HOME}/", original.decode("utf-8"))
            if re.search(r"/Users/(?!Shared/)[^/]+/", public):
                raise RuntimeError(f"unredacted home path in {source}")
            data = public.encode("utf-8")
            target = policy_target / source.name
            target.write_bytes(data)
            manifest["files"].append({
                "source": str(source.relative_to(ROOT)),
                "published": str(target.relative_to(HERE)),
                "source_sha256": digest(original),
                "published_sha256": digest(data),
                "home_prefix_rewritten": data != original,
            })
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Packaged {len(manifest['files'])} scripts and policies")


if __name__ == "__main__":
    main()
