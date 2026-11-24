#!/usr/bin/env python3
"""Build (and, after its GPU self-test, install) the native step-replay module.

Separate from the package build; CPU only; never imports the module.

    scripts/build_step_replay.py [--python VENV_PYTHON]
        configure + build native/step_replay with cmake/clang into
        build/step_replay/ (gitignored), stage the .so in
        build/step_replay/stage/, and check it statically:
          * nanobind internals key of the build equals the MLX wheel's
            (``nanobind_..._a1_v22``, no ``_stable``);
          * links ``@rpath/libmlx.dylib`` and carries only relative rpaths;
          * arm64, built for the running CPython ABI.
        Writes build/step_replay/stage/build-receipt.json.

    scripts/build_step_replay.py --install SELF_TEST_RECEIPT
        copy the staged module beside the ``mlx`` package
        (<site-packages>/mlx2_step_replay/) by an atomic rename, only when
        SELF_TEST_RECEIPT (written by scripts/step_replay_selftest.py on the
        GPU) passed for the same module bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "step_replay"
BUILD = ROOT / "build" / "step_replay"
STAGE = BUILD / "stage"
KEY_RE = re.compile(rb"nanobind[a-z0-9_]*_a\d+_v\d+(?:_stable)?")


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def probe(python):
    code = ("import json, sys, sysconfig, mlx, mlx.core as mx; "
            "print(json.dumps({'mlx_root': list(mlx.__path__)[0], 'mlx': mx.__version__, "
            "'py': list(sys.version_info[:2]), 'ext': sysconfig.get_config_var('EXT_SUFFIX'), "
            "'cache_tag': sys.implementation.cache_tag}))")
    return json.loads(run([python, "-c", code]).stdout)


def internals_keys(path):
    return sorted({m.decode() for m in KEY_RE.findall(Path(path).read_bytes())})


def otool(args, path):
    return run(["otool", *args, str(path)]).stdout


def rpaths(path):
    out = otool(["-l"], path).splitlines()
    found = []
    for i, line in enumerate(out):
        if "cmd LC_RPATH" in line:
            for follow in out[i + 1: i + 4]:
                follow = follow.strip()
                if follow.startswith("path "):
                    found.append(follow.split()[1])
    return found


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head():
    try:
        return run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def build(python):
    info = probe(python)
    if info["py"] != [3, 12]:
        raise SystemExit(f"serving venv is CPython 3.12; got {info['py']}")
    mlx_root = Path(info["mlx_root"])
    wheel_keys = internals_keys(next(mlx_root.glob("core.*.so")))
    nb_prefix = run(["brew", "--prefix", "nanobind"]).stdout.strip()
    nb_dir = Path(nb_prefix) / "share" / "nanobind" / "cmake"
    if not nb_dir.is_dir():
        raise SystemExit(f"nanobind cmake dir missing: {nb_dir}")
    cmake_dir = BUILD / "cmake"
    run(["cmake", "-S", str(SOURCE), "-B", str(cmake_dir), "-DCMAKE_BUILD_TYPE=Release",
         f"-DPython_EXECUTABLE={python}", f"-Dnanobind_DIR={nb_dir}", f"-DMLX_ROOT={mlx_root}"])
    run(["cmake", "--build", str(cmake_dir), "--config", "Release", "-j"])
    built = sorted(cmake_dir.glob("_step_replay*.so"))
    if len(built) != 1:
        raise SystemExit(f"expected one built module, found {built}")
    STAGE.mkdir(parents=True, exist_ok=True)
    staged = STAGE / ("_step_replay" + info["ext"])
    tmp = staged.with_suffix(".tmp")
    shutil.copy2(built[0], tmp)
    os.replace(tmp, staged)
    keys = internals_keys(staged)
    links = otool(["-L"], staged)
    paths = rpaths(staged)
    arch = run(["lipo", "-archs", str(staged)]).stdout.split()
    checks = {
        "internals_key_matches_wheel": keys == wheel_keys and len(keys) == 1,
        "no_stable_abi_suffix": not any(k.endswith("_stable") for k in keys),
        # NB_DOMAIN mlx: the domain string beside the key in both binaries.
        "domain_mlx": b"\0mlx\0" in staged.read_bytes(),
        "links_rpath_libmlx": "@rpath/libmlx.dylib" in links,
        "no_absolute_rpath": bool(paths) and all(not p.startswith("/") for p in paths),
        "arm64_only": arch == ["arm64"],
        "abi_suffix": staged.name.endswith(info["ext"]),
    }
    receipt = {
        "module": str(staged.relative_to(ROOT)), "sha256": sha256(staged),
        "mlx": info["mlx"], "python": info["cache_tag"], "nanobind_prefix": nb_prefix,
        "nanobind_keys": keys, "wheel_keys": wheel_keys, "rpaths": paths,
        "links": [line.strip() for line in links.splitlines()[1:]], "arch": arch,
        "checks": checks, "passed": all(checks.values()), "source_revision": git_head(),
        "loaded": False,
        "note": "static checks only; the module was not imported. Install needs the GPU "
                "self-test receipt (scripts/step_replay_selftest.py).",
    }
    (STAGE / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 1


def install(python, receipt_path):
    receipt = json.loads(Path(receipt_path).read_text())
    info = probe(python)
    staged = STAGE / ("_step_replay" + info["ext"])
    if not staged.is_file():
        raise SystemExit("nothing staged; build first")
    if not receipt.get("passed") or receipt.get("module_sha256") != sha256(staged):
        raise SystemExit("self-test receipt did not pass for the staged module bytes")
    target_dir = Path(info["mlx_root"]).parent / "mlx2_step_replay"
    target_dir.mkdir(exist_ok=True)
    target = target_dir / staged.name
    tmp = target.with_name(target.name + ".tmp")
    shutil.copy2(staged, tmp)
    os.replace(tmp, target)  # a working module is always on disk
    print(f"installed {target}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--install", metavar="SELF_TEST_RECEIPT")
    args = parser.parse_args()
    if args.install:
        return install(args.python, args.install)
    return build(args.python)


if __name__ == "__main__":
    sys.exit(main())
