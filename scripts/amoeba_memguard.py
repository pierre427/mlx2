#!/usr/bin/env python3
"""memguard.py <model_dir> [margin_gb]: exit 0 if available memory >= model size + margin, else 1.
Available = free + inactive + speculative + purgeable pages (vm_stat). Also refuses if any other
process has a physical footprint over 8 GB (top MEM; a foreign model load)."""
import os, re, subprocess, sys
d, margin = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".safetensors")) / 1e9
vm = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True).stdout
page = int(re.search(r"page size of (\d+)", vm).group(1))
get = lambda k: int(re.search(rf"{k}:\s+(\d+)", vm).group(1)) if re.search(rf"{k}:\s+(\d+)", vm) else 0
avail = (get("Pages free") + get("Pages inactive") + get("Pages speculative") + get("Pages purgeable")) * page / 1e9
# Physical footprint from top (RSS misses Metal/wired and compressed memory; 2026-11-13 a 108 GB
# MLX process read small in RSS).
top = subprocess.run(["/usr/bin/top", "-l", "1", "-o", "mem", "-n", "15", "-stats", "pid,mem,command"],
                     capture_output=True, text=True).stdout.splitlines()
def downloader(pid):
    # hf/huggingface-cli downloads hold large page-cache footprints, not model weights (2026-11-13: an
    # 8.6 GB `hf download` refused every load in a window). top truncates commands, so read ps.
    cmd = subprocess.run(["/bin/ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout
    return bool(re.search(r"\b(hf|huggingface-cli)\b.*\bdownload\b", cmd))


def gui_app(pid, gb):
    # A desktop app bundle under 16 GB is UI/texture memory, already counted in avail, not a foreign
    # model load (2026-11-13: the 9.7 GB MLX2.app window refused the host screen with 113 GB free).
    cmd = subprocess.run(["/bin/ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout
    return gb < 16 and cmd.startswith("/Applications/") and ".app/Contents/MacOS/" in cmd


unit = {"K": 1e-6, "M": 1e-3, "G": 1.0, "T": 1e3}
big = []
for l in top[top.index(next(x for x in top if x.startswith("PID"))) + 1:]:
    f = l.split(None, 2)
    m = re.match(r"([\d.]+)([KMGT])", f[1]) if len(f) >= 2 else None
    gb = float(m.group(1)) * unit[m.group(2)] if m else 0
    if gb > 8 and f[0] not in (str(os.getpid()), str(os.getppid())) and not downloader(f[0]) \
            and not gui_app(f[0], gb):
        big.append(f"{f[0]} {f[1]} {f[2] if len(f) > 2 else ''}".strip())
ok = avail >= size + margin and not big
print(f"memguard model {size:.1f} GB avail {avail:.1f} GB margin {margin} big_procs {big} -> {'OK' if ok else 'REFUSE'}")
sys.exit(0 if ok else 1)
