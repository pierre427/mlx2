"""Bounded verified-trace feedback and isolated CPU LiLiCoRR shadow training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .lilicorr_training import TeacherBuffer


@dataclass(frozen=True)
class LiLiCorrFeedbackPolicy:
    directory: str
    max_examples: int = 32
    max_bytes: int = 16 << 20
    min_examples: int = 8
    train_every: int = 16
    steps: int = 25
    learning_rate: float = 1e-3
    timeout_seconds: int = 120
    max_artifacts: int = 4
    max_training_bytes: int = 512 << 20

    @classmethod
    def from_value(cls, value):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("invalid lilicorr_feedback policy")
        if not isinstance(value.get("directory"), str) or not value["directory"]:
            raise ValueError("lilicorr_feedback requires an output directory")
        policy = cls(**value)
        limits = {
            "max_examples": 128,
            "max_bytes": 64 << 20,
            "min_examples": 128,
            "train_every": 1024,
            "steps": 1000,
            "timeout_seconds": 600,
            "max_artifacts": 16,
            "max_training_bytes": 2 << 30,
        }
        for name, maximum in limits.items():
            number = getattr(policy, name)
            if type(number) is not int or not 1 <= number <= maximum:
                raise ValueError(f"lilicorr_feedback {name} must be in [1,{maximum}]")
        if policy.min_examples > policy.max_examples:
            raise ValueError("lilicorr_feedback min_examples exceeds max_examples")
        if (
            type(policy.learning_rate) not in (int, float)
            or not math.isfinite(policy.learning_rate)
            or not 0 < policy.learning_rate <= 0.1
        ):
            raise ValueError("invalid lilicorr_feedback learning_rate")
        return policy


def estimate_training_bytes(parameter_count, teacher_bytes, examples, config):
    """Conservative launch estimate, not an allocator-enforced memory ceiling."""
    width = config.lilicorr_hidden_size or config.hidden_size
    nodes = (config.block_size - 1) * config.lilicorr_candidate_topk
    activations = (
        examples
        * 4
        * 8
        * (
            nodes * width * (config.lilicorr_num_layers + 4)
            + nodes
            * nodes
            * config.lilicorr_num_heads
            * max(config.lilicorr_num_layers, 1)
        )
    )
    return int(parameter_count * 4 * 8 + teacher_bytes * 4 + activations)


def training_budget_available(estimated, maximum):
    import psutil

    return estimated <= maximum and estimated <= int(
        psutil.virtual_memory().available * 0.5
    )


class LiLiCorrFeedbackManager:
    """Only committed labels enter the FIFO; training never mutates serving.

    The inference process captures bounded host arrays. A separately owned
    child sets its own CPU device and trains a clone; it cannot alter the
    inference process's default device, request weights or APCv2 revision.
    """

    def __init__(self, drafter, policy, *, target_revision, draft_revision, binding):
        self.policy = (
            policy
            if isinstance(policy, LiLiCorrFeedbackPolicy)
            else LiLiCorrFeedbackPolicy.from_value(policy)
        )
        if not hasattr(drafter, "lilicorr"):
            raise ValueError("LiLiCoRR feedback requires a resident correlator head")
        self.drafter, self.binding = drafter, binding
        self.buffer = TeacherBuffer(
            drafter.config,
            target_revision=target_revision,
            draft_revision=draft_revision,
            max_examples=self.policy.max_examples,
            max_bytes=self.policy.max_bytes,
        )
        namespace = hashlib.sha256(binding.encode()).hexdigest()
        self.directory = (
            Path(self.policy.directory).expanduser().resolve()
            / namespace
            / uuid.uuid4().hex
        )
        self.child = None
        self.child_started = 0.0
        self.child_directory = None
        self._nonce = 0
        self._last_launch = 0
        self._issued = OrderedDict()
        self.last_committed_trace = None
        self.closed = False
        self.stats = {
            "captured": 0,
            "committed": 0,
            "dropped": 0,
            "training_started": 0,
            "training_completed": 0,
            "training_failed": 0,
            "training_budget_skipped": 0,
        }
        self.latest_shadow = None
        self.last_error = None

    def capture_lattice(
        self,
        candidate_ids,
        token_embeddings,
        candidate_log_probs,
        pass_hidden,
        anchor_hidden,
        anchor_valid,
    ):
        if self.closed:
            return [None] * len(candidate_ids)
        # Reject the batch before exporting any tensor if it would exceed
        # the pending capture bound. No inference graph is retained.
        total_bytes = sum(
            value.size * 4
            for value in (
                candidate_ids,
                token_embeddings,
                candidate_log_probs,
                pass_hidden,
                anchor_hidden,
                anchor_valid,
            )
        )
        batch = len(candidate_ids)
        if total_bytes > self.policy.max_bytes:
            self.stats["dropped"] += batch
            return [None] * batch
        import mlx.core as mx

        arrays = []
        for value in (
            candidate_ids,
            token_embeddings,
            candidate_log_probs,
            pass_hidden,
            anchor_hidden,
            anchor_valid,
        ):
            converted = (
                value
                if mx.issubdtype(value.dtype, mx.integer)
                else value.astype(mx.float32)
            )
            mx.eval(converted)
            arrays.append(np.array(converted, copy=True))
        names = (
            "candidate_ids",
            "token_embeddings",
            "candidate_log_probs",
            "pass_hidden",
            "anchor_hidden",
            "anchor_valid",
        )
        rows = []
        for index in range(batch):
            self._nonce += 1
            row = {
                name: np.array(value[index], copy=True)
                for name, value in zip(names, arrays)
            }
            row["anchor_valid"] = bool(row["anchor_valid"])
            for value in row.values():
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
            row.update(binding=self.binding, nonce=self._nonce)
            self._issued[self._nonce] = self._capture_digest(row)
            if len(self._issued) > 4096:
                self._issued.popitem(last=False)
            rows.append(row)
        self.stats["captured"] += batch
        return rows

    def submit_verified(
        self,
        payload,
        teacher_tokens,
        *,
        first_rejected_position=None,
        request_id=None,
        round_id=None,
    ):
        """Called only after the whole verification transaction succeeds."""
        if payload is None or self.closed:
            return False
        if (
            payload.get("binding") != self.binding
            or type(payload.get("nonce")) is not int
        ):
            raise ValueError("LiLiCoRR feedback revision/capture mismatch")
        nonce = payload["nonce"]
        issued = self._issued.get(nonce)
        if issued is None:
            raise ValueError(
                "LiLiCoRR feedback capture unissued, expired or already consumed"
            )
        if issued != self._capture_digest(payload):
            raise ValueError("LiLiCoRR feedback capture contents changed")
        values = {
            name: payload[name]
            for name in (
                "candidate_ids",
                "token_embeddings",
                "candidate_log_probs",
                "pass_hidden",
                "anchor_hidden",
                "anchor_valid",
            )
        }
        kept = self.buffer.add(
            **values,
            teacher_tokens=list(teacher_tokens),
            first_rejected_position=first_rejected_position,
        )
        del self._issued[nonce]
        self.last_committed_trace = {
            "capture_nonce": nonce,
            "request_id_sha256": hashlib.sha256(str(request_id).encode()).hexdigest(),
            "round_id": round_id,
        }
        self.stats["committed"] += 1
        self.stats["dropped"] += int(not kept)
        return kept

    @staticmethod
    def _capture_digest(payload):
        digest = hashlib.sha256()
        digest.update(str(payload["binding"]).encode())
        digest.update(str(payload["nonce"]).encode())
        for name in (
            "candidate_ids",
            "token_embeddings",
            "candidate_log_probs",
            "pass_hidden",
            "anchor_hidden",
            "anchor_valid",
        ):
            value = np.asarray(payload[name])
            digest.update(
                json.dumps([name, str(value.dtype), list(value.shape)]).encode()
            )
            digest.update(value.tobytes())
        return digest.hexdigest()

    def _terminal_failure(self, job, error):
        self.last_error = error
        if job is not None:
            (job / "result.json").write_text(
                json.dumps(
                    {
                        "passed": False,
                        "binding": self.binding,
                        "error": error,
                        "qualified": False,
                        "selected": False,
                    }
                )
                + "\n"
            )

    def _prune_jobs(self):
        import shutil

        if not self.directory.exists():
            return
        jobs = sorted(
            self.directory.glob("shadow-*"), key=lambda path: path.stat().st_mtime
        )
        inactive = [
            path for path in jobs if path != self.child_directory or self.child is None
        ]
        for old in inactive[: max(0, len(jobs) - self.policy.max_artifacts)]:
            # This instance owns a unique directory; never delete another manager's job.
            shutil.rmtree(old)

    def _poll(self):
        if self.child is None:
            return
        if self.child.poll() is None:
            if time.monotonic() - self.child_started <= self.policy.timeout_seconds:
                return
            self.child.kill()
            self.child.wait(timeout=10)
            self._terminal_failure(self.child_directory, "CPU shadow worker timed out")
        child, job = self.child, self.child_directory
        try:
            result = job / "result.json"
            if child.returncode != 0 or not result.exists():
                raise ValueError(
                    f"CPU shadow worker exited {child.returncode} without a valid receipt"
                )
            report = json.loads(result.read_text())
            if (
                not isinstance(report, dict)
                or not report.get("passed")
                or report.get("binding") != self.binding
            ):
                raise ValueError("invalid shadow worker receipt")
            self.stats["training_completed"] += 1
            self.latest_shadow = report
        except Exception as error:  # noqa: BLE001 - worker diagnostics cannot change inference
            self.stats["training_failed"] += 1
            self._terminal_failure(job, f"{type(error).__name__}: {error}")
        finally:
            self.child = None
            self.child_directory = None
            self._prune_jobs()

    def settle_round(self):
        """Poll/launch one bounded worker, after every request has committed."""
        try:
            self._settle_round()
        except Exception as error:  # noqa: BLE001 - optional trainer cannot invalidate committed inference
            self.last_error = f"{type(error).__name__}: {error}"
            self.stats["training_failed"] += 1
            # Serialization can fail before a child exists. Bound all owned
            # inactive directories, including jobs without a worker receipt.
            if self.child is None:
                try:
                    self._terminal_failure(self.child_directory, self.last_error)
                except OSError:
                    pass
                finally:
                    self.child_directory = None
                    try:
                        self._prune_jobs()
                    except OSError:
                        pass

    def _settle_round(self):
        if self.closed:
            return
        self._poll()
        if (
            self.child is not None
            or len(self.buffer.examples) < self.policy.min_examples
            or self.stats["committed"] - self._last_launch < self.policy.train_every
        ):
            return
        import mlx.core as mx
        from mlx.utils import tree_flatten

        weights = dict(tree_flatten(self.drafter.lilicorr.parameters()))
        estimated = estimate_training_bytes(
            sum(value.size for value in weights.values()),
            self.buffer.nbytes,
            len(self.buffer.examples),
            self.buffer.config,
        )
        self._last_launch = self.stats["committed"]
        if not training_budget_available(estimated, self.policy.max_training_bytes):
            self.stats["training_budget_skipped"] += 1
            self.last_error = f"CPU shadow estimated memory {estimated} exceeds available training budget"
            return
        number = self.stats["training_started"] + self.stats["training_failed"] + 1
        self.directory.mkdir(parents=True, exist_ok=True)
        job = Path(tempfile.mkdtemp(prefix=f"shadow-{number:06d}-", dir=self.directory))
        self.child_directory = job
        mx.eval(list(weights.values()))
        mx.save_safetensors(str(job / "initial.safetensors"), weights)
        tensors, examples = {}, []
        for index, example in enumerate(self.buffer.examples):
            for field in (
                "candidate_ids",
                "token_embeddings",
                "candidate_log_probs",
                "pass_hidden",
                "anchor_hidden",
            ):
                tensors[f"{index}.{field}"] = getattr(example, field)
            examples.append(
                {
                    "anchor_valid": example.anchor_valid,
                    "teacher_tokens": [
                        int(example.candidate_ids[position, column])
                        for position, column in enumerate(example.teacher_columns)
                    ],
                    "first_rejected_position": example.first_rejected_position,
                }
            )
        np.savez(job / "teachers.npz", **tensors)
        data = {
            "binding": self.binding,
            "config": asdict(self.buffer.config),
            "target_revision": self.buffer.target_revision,
            "draft_revision": self.buffer.draft_revision,
            "examples": examples,
            "policy": asdict(self.policy),
            "files_sha256": {
                name: hashlib.sha256((job / name).read_bytes()).hexdigest()
                for name in ("initial.safetensors", "teachers.npz")
            },
        }
        (job / "job.json").write_text(json.dumps(data, indent=2) + "\n")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        try:
            with (job / "worker.log").open("w") as log:
                self.child = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "mlx2.runtime.lilicorr_feedback_worker",
                        "--job",
                        str(job / "job.json"),
                    ],
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        except Exception as error:
            self._terminal_failure(job, f"{type(error).__name__}: {error}")
            self._prune_jobs()
            raise
        self.child_directory, self.child_started = job, time.monotonic()
        self.stats["training_started"] += 1
        self._prune_jobs()

    def receipt(self):
        return {
            "enabled": True,
            "binding": self.binding,
            "device": "isolated_cpu_worker",
            "stats": dict(self.stats),
            "buffer_examples": len(self.buffer.examples),
            "buffer_bytes": self.buffer.nbytes,
            "pending_capture_tickets": len(self._issued),
            "last_committed_trace": self.last_committed_trace,
            "training_running": self.child is not None,
            "latest_shadow": self.latest_shadow,
            "last_error": self.last_error,
            "live_head_changed": False,
            "qualified": False,
            "selected_shadow": False,
        }

    def close(self):
        self.closed = True
        self._poll()
        if self.child is not None:
            self.child.terminate()
            try:
                self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait(timeout=5)
            self.stats["training_failed"] += 1
            self._terminal_failure(
                self.child_directory, "CPU shadow worker closed by owner"
            )
            self.child = None
            self.child_directory = None
        self._issued.clear()
        self._prune_jobs()
