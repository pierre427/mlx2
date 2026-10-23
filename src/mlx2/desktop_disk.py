"""Shared free-space reservations for concurrent acquisition/conversion jobs."""
from __future__ import annotations

import shutil
import threading
from pathlib import Path


class DiskBudget:
    def __init__(self):
        self.lock = threading.RLock()
        self.reservations = {}

    def reserve(self, job_id, path, required, reserve=10 * 1024**3):
        if type(required) is not int or required < 0 or type(reserve) is not int or reserve < 0:
            raise ValueError("Disk requirements must be nonnegative byte counts")
        path = Path(path).resolve(strict=True)
        device = path.stat().st_dev
        with self.lock:
            if job_id in self.reservations:
                raise ValueError("This job already has a disk reservation")
            outstanding = sum(row["remaining"] for row in self.reservations.values() if row["device"] == device)
            safety = max([reserve, *[row["reserve"] for row in self.reservations.values() if row["device"] == device]])
            available = shutil.disk_usage(path).free - outstanding
            if available < required + safety:
                raise ValueError("Insufficient free space after active job reservations and the safety reserve")
            self.reservations[job_id] = {"device": device, "remaining": required, "required": required, "reserve": reserve, "path": str(path)}

    def consumed(self, job_id, size):
        if type(size) is not int or size < 0:
            raise ValueError("Consumed bytes must be nonnegative integers")
        with self.lock:
            row = self.reservations.get(job_id)
            if row:
                if size > row["remaining"]:
                    raise ValueError("Write exceeds this job's disk reservation")
                row["remaining"] -= size

    def used(self, job_id, current_bytes):
        if type(current_bytes) is not int or current_bytes < 0:
            raise ValueError("Used bytes must be nonnegative integers")
        with self.lock:
            row = self.reservations[job_id]
            if current_bytes > row["required"]:
                raise ValueError("Job output exceeds its disk reservation")
            row["remaining"] = row["required"] - current_bytes

    def check(self, job_id, next_bytes=0):
        with self.lock:
            row = self.reservations[job_id]
            if type(next_bytes) is not int or next_bytes < 0 or next_bytes > row["remaining"]:
                raise ValueError("Write exceeds this job's disk reservation")
            safety = max(r["reserve"] for r in self.reservations.values() if r["device"] == row["device"])
            outstanding = sum(r["remaining"] for r in self.reservations.values() if r["device"] == row["device"])
            if shutil.disk_usage(row["path"]).free - outstanding < safety:
                raise ValueError("Free space fell below the configured safety reserve")

    def release(self, job_id):
        with self.lock:
            self.reservations.pop(job_id, None)
