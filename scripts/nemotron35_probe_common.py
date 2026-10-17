"""CPU-only evidence/ownership helpers for bounded Nemotron component probes."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import qualification_gpu_ownership as ownership
import run_qualification_matrix as matrix

ROOT = Path(__file__).resolve().parents[1]
# Optional: refuse to run unless the checkout is on this branch.
BRANCH = os.environ.get("MLX2_NEMOTRON_PROBE_BRANCH", "")
GIB = 1024**3
GPU_PROCESS = re.compile(
    r"(?:-m\s+mlx2\.(?:server|decisions\.server)|mlx_lm\.server|mlx_vlm\.server|"
    r"llama-server|ollama\s+(?:serve|runner)|rapid-mlx|ComfyUI|ltx-2-mlx|"
    r"music3_server|yue2_server|ruach_server|audiocpp_server)", re.IGNORECASE)
GPU_SERVICE = re.compile(r"mlx2|mlx-uag|ollama|llama|music3|yue2|ruach", re.IGNORECASE)
SUPPORT_SERVICES = {"com.example.mlx2-wiki", "com.example.mlx2-image-mcp", "com.example.yue2-mcp"}
atomic_json = matrix.atomic_json


def require_owner(cpg_lock):
    owner = json.loads((Path(cpg_lock) / "owner.json").read_text())
    return ownership.require_qualification_lease(task_id=owner["cpg_task"],
        generation=owner["cpg_generation"], cpg_owner_lock=Path(cpg_lock))


def source_identity():
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if BRANCH and branch != BRANCH:
        raise RuntimeError("Probe is no longer running on the requested branch")
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard",
                                     "src", "scripts"], cwd=ROOT, text=True).splitlines()
    files = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
             for name in sorted(set(names)) if name.endswith((".py", ".swift"))}
    return {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "branch": branch, "files": files, "sha256": matrix.digest(files),
            "tracked_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                                          cwd=ROOT, text=True).strip())}


def runtime_dependencies():
    """Bind installed versions and available MLX native payloads without import."""
    versions, native = {}, {}
    for package in ("mlx", "mlx-metal", "torch", "transformers", "numpy", "Pillow"):
        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
            continue
        versions[package] = distribution.version
        if package not in {"mlx", "mlx-metal"}:
            continue
        for item in distribution.files or ():
            if str(item).endswith((".so", ".dylib", ".metallib")):
                path = distribution.locate_file(item)
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 << 20), b""):
                        digest.update(block)
                native[package + ":" + str(item)] = digest.hexdigest()
    return {"versions": versions, "mlx_native_files": native}


def parse_memory(text):
    page_size = re.search(r"page size of (\d+) bytes", text)
    pages = {name: int(count) for name, count in re.findall(r"^([^:\n]+):\s+(\d+)\.", text, re.MULTILINE)}
    required = {"Pages free", "Pages inactive", "Pages speculative", "Swapouts"}
    if page_size is None or not required <= pages.keys():
        raise RuntimeError("Host memory/swap counters are unavailable")
    size = int(page_size.group(1))
    return {"page_bytes": size, "pages": pages, "swapouts": pages["Swapouts"],
            "available_bytes": sum(pages[key] for key in ("Pages free", "Pages inactive", "Pages speculative")) * size}


def cpu_seconds(value):
    days, _, clock = value.partition("-")
    parts = [float(part) for part in (clock or days).split(":")]
    return (int(days) * 86400 if clock else 0) + sum(part * 60**i for i, part in enumerate(reversed(parts)))


def idle_audio_service(command, rss_kib):
    # The installed lazy audio.cpp API has no model resident while idle
    # (verified from its executable, RSS and /health). Keep observing it.
    return Path(command.split(None, 1)[0]).name == "audiocpp_server" and rss_kib < 256 * 1024


def remote_ssh_client(command):
    """Return true for a local SSH client whose model process runs remotely."""
    executable = Path(command.split(None, 1)[0]).name.lower()
    return executable in {"ssh", "ssh.exe"}


def process_snapshot():
    rows = subprocess.check_output(["/bin/ps", "-Ao", "pid=,ppid=,rss=,time=,command="], text=True)
    processes = {}
    for row in rows.splitlines():
        fields = row.strip().split(None, 4)
        if len(fields) == 5:
            processes[int(fields[0])] = (int(fields[1]), int(fields[2]), cpu_seconds(fields[3]), fields[4])
    own = ownership._ancestor_pids(os.getpid())
    # Descendants of this runner belong to its server/reference process group;
    # siblings of an ancestor remain foreign.
    children, changed = {os.getpid()}, True
    while changed:
        added = {pid for pid, (parent, _, _, _) in processes.items() if parent in children} - children
        children.update(added)
        changed = bool(added)
    own.update(children)
    foreign, lazy, remote = [], [], []
    for pid, (parent, rss, cpu, command) in processes.items():
        match = GPU_PROCESS.search(command)
        if pid not in own and match:
            # Never persist arbitrary argv, which could contain credentials.
            row = {"pid": pid, "ppid": parent, "rss_kib": rss, "cpu_seconds": cpu,
                   "executable": command.split(None, 1)[0], "surface": match.group(0)}
            if remote_ssh_client(command):
                remote.append(row)
            else:
                (lazy if idle_audio_service(command, rss) else foreign).append(row)
    services = []
    for row in subprocess.check_output(["launchctl", "list"], text=True).splitlines():
        fields = row.split()
        if len(fields) == 3 and GPU_SERVICE.search(fields[2]):
            services.append({"pid": int(fields[0]) if fields[0].isdigit() else None, "label": fields[2]})
    live_services = []
    for item in services:
        pid = item["pid"]
        if pid is None or pid in own:
            continue
        process = processes.get(pid)
        if process is not None and (idle_audio_service(process[3], process[1])
                                   or item["label"] in SUPPORT_SERVICES and process[1] < 256 * 1024):
            continue
        live_services.append(item)
    waiters = Path("/Users/Shared/mlxuag/gpu.lock.waiters")
    return {"foreign_model_processes": foreign, "foreign_live_services": live_services,
            "observed_idle_audio_services": lazy, "observed_remote_model_clients": remote,
            "services": services,
            "waiter_files": sorted(item.name for item in waiters.iterdir()) if waiters.is_dir() else []}


def host_snapshot():
    memory = parse_memory(subprocess.check_output(["/usr/bin/vm_stat"], text=True))
    pressure = int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True))
    return {"memory": memory, "memory_pressure_level": pressure, "swap": matrix.sample_swap(),
            "thermal": matrix.sample_thermal(), "load_average": list(os.getloadavg()), **process_snapshot()}


def admission_errors(snapshot, *, minimum_available_bytes, performance=False):
    errors = []
    if snapshot["foreign_model_processes"] or snapshot["foreign_live_services"]:
        errors.append("foreign model process or service is live")
    if snapshot["memory_pressure_level"] != 1:
        errors.append("host memory pressure is not normal")
    if snapshot["memory"]["available_bytes"] < minimum_available_bytes:
        errors.append("insufficient available memory for this bounded probe")
    thermal = snapshot["thermal"]
    if any(thermal.get(key) is not True for key in (
            "pmset_no_thermal_warning", "pmset_no_performance_warning", "pmset_no_cpu_power_warning")):
        errors.append("host reports a thermal or performance warning")
    if int(thermal.get("thermal_state", 3)) > (0 if performance else 1):
        errors.append("thermal state exceeds admission")
    if performance and (thermal.get("battery_temperature_c", float("inf")) > 40.
                        or thermal.get("virtual_temperature_c", float("inf")) > 45.):
        errors.append("performance temperature ceiling exceeded or unavailable")
    return errors


def contamination(before, after):
    reasons = admission_errors(after, minimum_available_bytes=4 * GIB)
    if after["memory"]["swapouts"] > before["memory"]["swapouts"]:
        reasons.append("swap-outs increased")
    if after["swap"]["used_bytes"] > before["swap"]["used_bytes"]:
        reasons.append("swap usage increased")
    idle_before = {row["pid"]: row for row in before.get("observed_idle_audio_services", [])}
    for row in after.get("observed_idle_audio_services", []):
        prior = idle_before.get(row["pid"])
        if prior is None or row["cpu_seconds"] - prior["cpu_seconds"] > .5:
            reasons.append("otherwise idle audio service became active")
    return reasons


def group_running(pgid):
    rows = subprocess.check_output(["/bin/ps", "-Ao", "pid=,pgid=,stat="], text=True)
    return [int(fields[0]) for row in rows.splitlines()
            if len(fields := row.split()) == 3 and int(fields[1]) == pgid
            and not fields[2].startswith(("Z", "X"))]


def stop_group(process):
    """Clean the whole owned group even when its original leader has exited."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait(timeout=15)
    deadline = time.monotonic() + 2
    while group_running(process.pid):
        if time.monotonic() >= deadline:
            raise RuntimeError("Owned process group still has live members after cleanup")
        time.sleep(.05)
