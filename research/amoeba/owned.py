"""Run one amoeba job inside an owned GPU window: a CPG desktop lease plus both GPU locks.

    owned.py --out DIR --job-id ID -- <command...>

Wrap it in scripts/run_with_service_paused.py when a model service holds the GPU.
The evidence directory must be empty; the job sees AMOEBA_OWNED_OUT=DIR.
"""
import argparse, os, signal, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--job-id", required=True)
    ap.add_argument("--checkpoint", default="Qwen/Qwen3-0.6B@c1899de2")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    rev = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt()))
    with DesktopCPGLease(out, a.job_id, checkpoint=a.checkpoint, source_revision=rev) as lease:
        locks = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"), "--session", lease.owner["cpg_session"],
                 "--label", a.job_id, "--desktop-cpg-lease", str(lease.path), "--receipt", str(out / "paired-locks.json"), "--"]
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), AMOEBA_OWNED_OUT=str(out))
        child = subprocess.Popen(locks + cmd, cwd=ROOT, env=env, start_new_session=True)
        try:
            while child.poll() is None:
                lease.check(); time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=90)


if __name__ == "__main__":
    raise SystemExit(main())
