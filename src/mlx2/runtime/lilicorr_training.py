"""Explicit, bounded CPU training of a separate LiLiCorr shadow head.

This research helper never captures requests automatically, changes a serving
model, or publishes APC state. Inputs must be frozen draft features from a
known target/draft revision and labels from an actually emitted target prefix.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TeacherExample:
    candidate_ids: np.ndarray
    token_embeddings: np.ndarray
    candidate_log_probs: np.ndarray
    pass_hidden: np.ndarray
    anchor_hidden: np.ndarray
    anchor_valid: bool
    teacher_columns: tuple[int, ...]
    first_rejected_position: int | None

    @property
    def nbytes(self):
        return sum(
            value.nbytes
            for value in (
                self.candidate_ids,
                self.token_embeddings,
                self.candidate_log_probs,
                self.pass_hidden,
                self.anchor_hidden,
            )
        )


class TeacherBuffer:
    """FIFO bounded examples tied to one immutable target/draft geometry.

    The rejection position may supply its emitted correction token as a label;
    all later slots are censored. If a teacher token leaves the candidate set,
    subsequent positions cannot establish a ground-truth lattice path either.
    """

    def __init__(
        self,
        config,
        *,
        target_revision,
        draft_revision,
        max_examples=128,
        max_bytes=64 << 20,
    ):
        for value in (target_revision, draft_revision):
            if not isinstance(value, str) or not re.fullmatch(
                r"[0-9a-f]{40}|[0-9a-f]{64}", value
            ):
                raise ValueError(
                    "Teacher capture requires pinned target/draft revisions"
                )
        if (
            type(max_examples) is not int
            or max_examples <= 0
            or type(max_bytes) is not int
            or max_bytes <= 0
        ):
            raise ValueError("Teacher buffer budgets must be positive integers")
        self.config = config
        self.target_revision = target_revision
        self.draft_revision = draft_revision
        self.max_examples = max_examples
        self.max_bytes = max_bytes
        self.examples = []
        self.nbytes = 0
        self.dropped = 0
        self.geometry_revision = hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True).encode()
        ).hexdigest()

    def add(
        self,
        *,
        candidate_ids,
        token_embeddings,
        candidate_log_probs,
        pass_hidden,
        anchor_hidden,
        anchor_valid,
        teacher_tokens,
        first_rejected_position=None,
    ):
        config = self.config
        slots, k, h = (
            config.block_size - 1,
            config.lilicorr_candidate_topk,
            config.hidden_size,
        )
        expected_bytes = 4 * (slots * k + slots * k * h + slots * k + slots * h + h)
        if expected_bytes > self.max_bytes:
            raise ValueError("One teacher example exceeds the byte budget")
        specs = [
            ("candidate_ids", candidate_ids, (slots, k), np.int32),
            ("token_embeddings", token_embeddings, (slots, k, h), np.float32),
            ("candidate_log_probs", candidate_log_probs, (slots, k), np.float32),
            ("pass_hidden", pass_hidden, (slots, h), np.float32),
            ("anchor_hidden", anchor_hidden, (h,), np.float32),
        ]
        arrays = {}
        for name, value, shape, dtype in specs:
            source = np.asarray(value)
            if source.shape != shape:
                raise ValueError(f"Teacher {name} geometry mismatch")
            if name == "candidate_ids":
                if (
                    not np.issubdtype(source.dtype, np.integer)
                    or (source < 0).any()
                    or (source >= config.vocab_size).any()
                    or any(len(set(row.tolist())) != k for row in source)
                ):
                    raise ValueError("Invalid teacher candidate IDs")
            elif not np.isfinite(source).all():
                raise ValueError(f"Teacher {name} must be finite")
            stored = np.array(source, dtype=dtype, copy=True)
            stored.setflags(write=False)
            arrays[name] = stored
        if (arrays["candidate_log_probs"] > 1e-6).any() or (
            np.exp(arrays["candidate_log_probs"]).sum(-1) > 1.00001
        ).any():
            raise ValueError(
                "Teacher candidate features require full-vocabulary log probabilities"
            )
        if type(anchor_valid) is not bool:
            raise ValueError("Teacher anchor validity must be explicit")
        if first_rejected_position is not None and (
            type(first_rejected_position) is not int
            or not 0 <= first_rejected_position < slots
        ):
            raise ValueError("Invalid first rejection boundary")
        limit = min(
            slots,
            len(teacher_tokens),
            slots if first_rejected_position is None else first_rejected_position + 1,
        )
        columns = []
        for position in range(limit):
            token = teacher_tokens[position]
            if type(token) is not int or not 0 <= token < config.vocab_size:
                raise ValueError("Invalid emitted teacher token")
            matches = np.flatnonzero(arrays["candidate_ids"][position] == token)
            if not len(matches):
                break
            columns.append(int(matches[0]))
        if not columns:
            self.dropped += 1
            return False
        example = TeacherExample(
            **arrays,
            anchor_valid=anchor_valid,
            teacher_columns=tuple(columns),
            first_rejected_position=first_rejected_position,
        )
        if example.nbytes > self.max_bytes:
            raise ValueError("One teacher example exceeds the byte budget")
        while self.examples and (
            len(self.examples) >= self.max_examples
            or self.nbytes + example.nbytes > self.max_bytes
        ):
            self.nbytes -= self.examples.pop(0).nbytes
            self.dropped += 1
        self.examples.append(example)
        self.nbytes += example.nbytes
        return True


@dataclass
class ShadowTrainingResult:
    head: object
    config: object
    target_revision: str
    draft_revision: str
    geometry_revision: str
    initial_loss: float
    final_loss: float
    steps: int
    supervised_positions: int
    example_count: int
    initial_head_sha256: str
    teacher_examples_sha256: str
    trained_head_sha256: str


def _head_sha256(head):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    digest = hashlib.sha256()
    for name, value in tree_flatten(head.parameters()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        raw = value.view(mx.uint16) if value.dtype == mx.bfloat16 else value
        digest.update(np.asarray(raw).tobytes())
    return digest.hexdigest()


def _examples_sha256(examples):
    digest = hashlib.sha256()
    for example in examples:
        digest.update(
            json.dumps(
                [
                    example.anchor_valid,
                    example.teacher_columns,
                    example.first_rejected_position,
                ]
            ).encode()
        )
        for value in (
            example.candidate_ids,
            example.token_embeddings,
            example.candidate_log_probs,
            example.pass_hidden,
            example.anchor_hidden,
        ):
            digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def train_shadow(initial_head, buffer, *, steps=25, learning_rate=1e-3):
    """Clone and train only head parameters; enforce CPU and bounded work.

    The caller must supply explicit verified examples. A serving drafter is
    neither modified nor passed to this helper.
    """
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx import nn
    from mlx.utils import tree_flatten

    from .drafters.lilicorr import LiLiCorrHead

    if mx.default_device() != mx.cpu:
        raise ValueError(
            "Shadow LiLiCorr training currently requires explicit CPU device"
        )
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError("Shadow training steps must be in [1,1000]")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not np.isfinite(learning_rate)
        or not 0 < learning_rate <= 0.1
    ):
        raise ValueError("Invalid shadow training learning rate")
    if not buffer.examples:
        raise ValueError("Shadow training requires verified teacher examples")
    current_geometry = hashlib.sha256(
        json.dumps(asdict(buffer.config), sort_keys=True).encode()
    ).hexdigest()
    if current_geometry != buffer.geometry_revision:
        raise ValueError("Teacher geometry changed since capture")
    initial_head_sha256 = _head_sha256(initial_head)
    teacher_examples_sha256 = _examples_sha256(buffer.examples)
    head = LiLiCorrHead(buffer.config)
    head.load_weights(
        [
            (name, mx.array(value).astype(mx.float32))
            for name, value in tree_flatten(initial_head.parameters())
        ],
        strict=True,
    )
    batch = []
    for example in buffer.examples:
        frozen = [
            mx.stop_gradient(mx.array(value)[None])
            for value in (
                example.token_embeddings,
                example.candidate_log_probs,
                example.pass_hidden,
                example.anchor_hidden,
                np.array(example.anchor_valid),
            )
        ]
        # Each row is conditioned on the ground-truth predecessor column.
        count = len(example.teacher_columns)
        previous = (0,) + example.teacher_columns[:-1]
        batch.append(
            (
                frozen,
                mx.array(list(range(count))),
                mx.array(previous),
                mx.array(example.teacher_columns),
            )
        )
    total_positions = sum(len(e.teacher_columns) for e in buffer.examples)

    def loss(model):
        total = mx.array(0.0, dtype=mx.float32)
        for values, positions, previous, labels in batch:
            table = model(*values)[0]
            scores = table[positions, previous]
            total = total + mx.sum(
                nn.losses.cross_entropy(scores, labels, reduction="none")
            )
        return total / total_positions

    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(head, loss)
    initial = float(loss(head).item())
    for _ in range(steps):
        value, grads = loss_and_grad(head)
        optimizer.update(head, grads)
        mx.eval(head.parameters(), optimizer.state, value)
        if not np.isfinite(value.item()):
            raise ValueError("Nonfinite shadow training loss")
    final = float(loss(head).item())
    if not np.isfinite(final):
        raise ValueError("Nonfinite shadow training loss")
    return ShadowTrainingResult(
        head,
        buffer.config,
        buffer.target_revision,
        buffer.draft_revision,
        buffer.geometry_revision,
        initial,
        final,
        steps,
        total_positions,
        len(buffer.examples),
        initial_head_sha256,
        teacher_examples_sha256,
        _head_sha256(head),
    )


def export_shadow(result, destination):
    """Write a revision-bound, explicitly unselected head-only candidate.

    The full serving artifact inspector refuses this directory because it
    lacks the co-trained backbone. Selection needs separate qualification and
    an explicitly constructed full source-bound artifact.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    if mx.default_device() != mx.cpu:
        raise ValueError("Shadow export currently requires CPU device")
    geometry = hashlib.sha256(
        json.dumps(asdict(result.config), sort_keys=True).encode()
    ).hexdigest()
    if geometry != result.geometry_revision:
        raise ValueError("Shadow geometry changed since training")
    if _head_sha256(result.head) != result.trained_head_sha256:
        raise ValueError("Shadow parameters changed since training")
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    tensor_path = destination / "head.safetensors"
    weights = {
        "lilicorr." + name: value
        for name, value in tree_flatten(result.head.parameters())
    }
    mx.eval(list(weights.values()))
    mx.save_safetensors(str(tensor_path), weights)
    digest = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    manifest = {
        "artifact_role": "lilicorr-shadow-head",
        "format_version": 1,
        "target_revision": result.target_revision,
        "draft_revision": result.draft_revision,
        "geometry_revision": result.geometry_revision,
        "geometry": asdict(result.config),
        "revision_authority": "caller-declared pins; teacher inputs require trusted target verification",
        "training_lineage": {
            "initial_head_sha256": result.initial_head_sha256,
            "teacher_examples_sha256": result.teacher_examples_sha256,
            "trained_head_sha256": result.trained_head_sha256,
        },
        "files": [
            {
                "path": "head.safetensors",
                "sha256": digest,
                "bytes": tensor_path.stat().st_size,
            }
        ],
        "training": {
            "steps": result.steps,
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
            "supervised_positions": result.supervised_positions,
            "example_count": result.example_count,
            "censoring": "after_first_rejection_or_teacher_outside_candidates",
        },
        "implemented": True,
        "qualified": False,
        "selected": False,
        "observed_used": False,
        "qualification_required": [
            "teacher-integrity",
            "held-out-acceptance",
            "ordinary-reference",
            "target-rejection-law",
            "runtime-overhead",
        ],
    }
    manifest["revision"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
