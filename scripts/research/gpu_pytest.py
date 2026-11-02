"""Run Metal-marked pytest files under the GPU lease recipe (bench.sh).

    bench.sh <commit> <label> gpu_pytest.py <pytest args...>

bench.sh passes --out/--source-revision first; the rest goes to pytest with
MLX2_RUN_GPU_TESTS=1 and PYTHONPATH at the snapshot's src. The pytest output
and return code land in <out>/pytest.txt and <out>/result.json.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    argv = sys.argv[1:]
    out = Path(argv[argv.index("--out") + 1])
    revision = argv[argv.index("--source-revision") + 1]
    rest = [a for i, a in enumerate(argv)
            if a not in ("--out", "--source-revision") and (i == 0 or argv[i - 1] not in ("--out", "--source-revision"))]
    if out.exists() and any(out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, MLX2_RUN_GPU_TESTS="1", PYTHONPATH=str(root / "src"))
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", *rest], cwd=root, env=env, check=False,
                          capture_output=True, text=True)
    (out / "pytest.txt").write_text(proc.stdout + proc.stderr)
    (out / "result.json").write_text(json.dumps({"source_revision": revision, "args": rest,
                                                 "returncode": proc.returncode}) + "\n")
    print(proc.stdout[-4000:], proc.stderr[-2000:], flush=True)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
