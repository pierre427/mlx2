"""Require the cpg_job parent lease and hold the existing /tmp advisory lock.

Invoke only via cpg_job.py run --lease --lock --require-radio -- this.py COMMAND.
The wrapper owns and releases /Users/Shared/mlxuag/gpu.lock; this child holds
fcntl flock on /tmp/gpu.lock without deleting or replacing the existing file.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

owner = json.loads(Path('/Users/Shared/mlxuag/gpu.lock/owner.json').read_text())
if owner.get('pid') != os.getppid() or not owner.get('cpg_generation'):
    raise SystemExit('Refusing: lock is not owned by this cpg_job parent')
with open('/tmp/gpu.lock', 'a+') as handle:
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print(json.dumps({'type': 'ownership', 'owner': owner, 'tmp_lock': 'fcntl exclusive'}), flush=True)
    subprocess.run(sys.argv[1:], check=True)
