"""Artifact-bound reduced vocabulary for native-MTP proposal heads.

The target head and every verification pass remain full-vocabulary.  This
module only constructs a smaller, separately owned view of an untied MTP
proposal head and scatters its logits back into a full-width row with ``-inf``
outside the admitted token ids.  Rejection sampling therefore sees a valid
restricted-support proposal distribution without changing the target law.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
from mlx import nn

SCHEMA = "mlx2.mtp-draft-vocab.v1"
MANIFEST_NAME = "mtp_draft_vocab.json"
IDS_NAME = "mtp_draft_vocab.ids"
LICENSE_NAME = "mtp_draft_vocab.ids.LICENSE"
MIN_TOKEN_COUNT = 4096
TOKEN_COUNT_ALIGNMENT = 64


class DraftVocabError(ValueError):
    """The requested reduced-vocabulary artifact is absent or malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_token_ids(text: str, vocab_size: int) -> tuple[int, ...]:
    """Parse one integer id per line, allowing comments and blank lines."""
    if type(vocab_size) is not int or vocab_size <= 0:
        raise DraftVocabError("vocab_size must be a positive integer")
    values = set()
    for line_number, raw in enumerate(text.splitlines(), 1):
        value = raw.split("#", 1)[0].strip()
        if not value:
            continue
        try:
            token_id = int(value, 10)
        except ValueError:
            raise DraftVocabError(
                f"line {line_number}: expected one integer token id"
            ) from None
        if not 0 <= token_id < vocab_size:
            raise DraftVocabError(
                f"line {line_number}: token id {token_id} outside [0, {vocab_size})"
            )
        values.add(token_id)
    if not values:
        raise DraftVocabError("token id list is empty")
    return tuple(sorted(values))


def validate_token_count(count: int, vocab_size: int) -> None:
    if count > vocab_size:
        raise DraftVocabError("draft vocabulary exceeds the model vocabulary")
    if count < MIN_TOKEN_COUNT:
        raise DraftVocabError(
            f"draft vocabulary has {count} ids; minimum is {MIN_TOKEN_COUNT}"
        )
    if count % TOKEN_COUNT_ALIGNMENT:
        raise DraftVocabError(
            f"draft vocabulary size must be a multiple of {TOKEN_COUNT_ALIGNMENT}"
        )


@dataclass(frozen=True)
class DraftVocabManifest:
    schema: str
    vocab_size: int
    token_count: int
    ids_sha256: str
    license_sha256: str
    config_sha256: str
    index_sha256: str
    tokenizer_sha256: str
    source_repository: str
    source_revision: str
    corpus_profile: str

    def receipt(self) -> dict:
        return asdict(self)


def load_manifest(model_path: Path) -> tuple[DraftVocabManifest, tuple[int, ...]]:
    """Load and bind the fixed-name sidecars to the exact model artifact."""
    model_path = Path(model_path).expanduser().resolve()
    manifest_path = model_path / MANIFEST_NAME
    ids_path = model_path / IDS_NAME
    try:
        raw = json.loads(manifest_path.read_text())
    except FileNotFoundError:
        raise DraftVocabError(f"missing {manifest_path}") from None
    except (OSError, json.JSONDecodeError) as error:
        raise DraftVocabError(f"cannot read {manifest_path}: {error}") from error
    fields = set(DraftVocabManifest.__dataclass_fields__)
    if set(raw) != fields:
        missing = sorted(fields - set(raw))
        extra = sorted(set(raw) - fields)
        raise DraftVocabError(
            f"draft vocabulary manifest fields differ (missing={missing}, extra={extra})"
        )
    manifest = DraftVocabManifest(**raw)
    if manifest.schema != SCHEMA:
        raise DraftVocabError(f"unsupported draft vocabulary schema {manifest.schema!r}")
    if any(
        type(value) is not str or not value
        for value in (
            manifest.ids_sha256,
            manifest.license_sha256,
            manifest.config_sha256,
            manifest.index_sha256,
            manifest.tokenizer_sha256,
            manifest.source_repository,
            manifest.source_revision,
            manifest.corpus_profile,
        )
    ):
        raise DraftVocabError("draft vocabulary manifest strings must be non-empty")
    if type(manifest.vocab_size) is not int or manifest.vocab_size <= 0:
        raise DraftVocabError("manifest vocab_size must be a positive integer")
    try:
        config = json.loads((model_path / "config.json").read_text())
        artifact_vocab_size = int(config.get("text_config", config)["vocab_size"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DraftVocabError(f"cannot resolve artifact vocab_size: {error}") from error
    if manifest.vocab_size != artifact_vocab_size:
        raise DraftVocabError(
            f"manifest vocab_size {manifest.vocab_size} != artifact {artifact_vocab_size}"
        )
    validate_token_count(manifest.token_count, manifest.vocab_size)
    required = {
        ids_path: manifest.ids_sha256,
        model_path / LICENSE_NAME: manifest.license_sha256,
        model_path / "config.json": manifest.config_sha256,
        model_path / "model.safetensors.index.json": manifest.index_sha256,
        model_path / "tokenizer.json": manifest.tokenizer_sha256,
    }
    for path, expected in required.items():
        if not path.is_file():
            raise DraftVocabError(f"missing bound artifact file {path}")
        observed = _sha256(path)
        if observed != expected:
            raise DraftVocabError(
                f"draft vocabulary binding mismatch for {path.name}: {observed} != {expected}"
            )
    ids = parse_token_ids(ids_path.read_text(), manifest.vocab_size)
    if len(ids) != manifest.token_count:
        raise DraftVocabError(
            f"manifest token_count {manifest.token_count} != parsed count {len(ids)}"
        )
    return manifest, ids


class ReducedMTPHead(nn.Module):
    """A row-sliced float or affine-quantized head owned by the MTP drafter."""

    def __init__(self, source, token_ids: Iterable[int], vocab_size: int):
        super().__init__()
        token_ids = tuple(int(value) for value in token_ids)
        if not token_ids or token_ids != tuple(sorted(set(token_ids))):
            raise DraftVocabError("token ids must be non-empty, sorted, and unique")
        if token_ids[0] < 0 or token_ids[-1] >= vocab_size:
            raise DraftVocabError("token ids are outside the source vocabulary")
        if int(source.weight.shape[0]) != int(vocab_size):
            raise DraftVocabError("source head rows do not match vocab_size")
        index = mx.array(token_ids, dtype=mx.uint32)
        self.token_ids = index
        self.vocab_size = int(vocab_size)
        self.weight = mx.take(source.weight, index, axis=0)
        self.bias = mx.take(source.bias, index, axis=0) if "bias" in source else None
        self.quantized = hasattr(source, "scales")
        if self.quantized:
            self.scales = mx.take(source.scales, index, axis=0)
            biases = getattr(source, "biases", None)
            self.biases = None if biases is None else mx.take(biases, index, axis=0)
            self.group_size = int(source.group_size)
            self.bits = int(source.bits)
            self.mode = str(source.mode)
        self.freeze()

    def __call__(self, hidden):
        if self.quantized:
            reduced = mx.quantized_matmul(
                hidden,
                self.weight,
                scales=self.scales,
                biases=self.biases,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
        else:
            reduced = hidden @ self.weight.T
        if self.bias is not None:
            reduced = reduced + self.bias
        full = mx.full((*reduced.shape[:-1], self.vocab_size), -mx.inf, reduced.dtype)
        indices = mx.broadcast_to(self.token_ids, reduced.shape)
        return mx.put_along_axis(full, indices, reduced, axis=-1)

    def receipt(self) -> dict:
        return {
            "enabled": True,
            "vocab_size": self.vocab_size,
            "token_count": int(self.token_ids.size),
            "quantized": bool(self.quantized),
            **(
                {
                    "bits": self.bits,
                    "group_size": self.group_size,
                    "mode": self.mode,
                }
                if self.quantized
                else {}
            ),
        }


def install_reduced_mtp_head(model, manifest, token_ids) -> dict:
    """Install an adapter-requested proposal head, refusing tied/no-MTP models."""
    language_model = getattr(model, "language_model", None)
    if getattr(model, "mtp", None) is None:
        raise DraftVocabError("reduced MTP vocabulary requires an MTP sidecar")
    if getattr(getattr(language_model, "args", None), "tie_word_embeddings", False):
        raise DraftVocabError("reduced MTP vocabulary requires an untied lm_head")
    head = getattr(language_model, "lm_head", None)
    if head is None:
        raise DraftVocabError("model has no lm_head")
    reduced = ReducedMTPHead(head, token_ids, manifest.vocab_size)
    model.mtp_draft_head = reduced
    model.mtp_draft_vocab_manifest = manifest.receipt()
    model.mtp_draft_vocab_calls = 0
    model.mtp_draft_vocab_full_bypasses = 0
    model.mtp_draft_vocab_enabled = True
    receipt = {**manifest.receipt(), **reduced.receipt()}
    return receipt
