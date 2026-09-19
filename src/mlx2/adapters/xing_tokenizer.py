# SPDX-License-Identifier: MIT
"""Xing4.0 tokenizer runtime: parity-stamped fast loader, exact decode, streaming.

The Xing4.0 checkpoint ships only a SentencePiece model and a custom slow
tokenizer that needs ``trust_remote_code``.  ``scripts/xing4_0_tokenizer.py``
builds a ``tokenizer.json`` that reproduces it id-for-id and stamps the artifact
with ``xing4_0_tokenizer_parity.json`` only after a parity check against the
live reference passes.  This module:

* loads that fast tokenizer with ``trust_remote_code=False`` after verifying
  the stamp against the files on disk (fail closed on any mismatch);
* replaces the fast decode with an exact emulation of the reference decode,
  which ``tokenizer.json`` decoders cannot express (see ``decode_ids``);
* falls back to the slow reference tokenizer only when explicitly enabled;
* provides ``XingStreamingDetokenizer`` whose text always equals
  ``decode(tokens, skip_special_tokens=False)``;
* provides ``render_prompt`` for chat prompts.

Runtime needs neither ``sentencepiece`` nor remote code on the fast path.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

VOCAB_SIZE = 131072
SPIECE = "▁"
SPIECE_ID = 124361
UNK_ID = 0
UNK_SURFACE = " ⁇ "
BYTE_OFFSET = 140  # <0x00> .. <0xFF> are ids 140..395
EOS_ID = 2
THINK_START_ID = 9
THINK_END_ID = 10

# Registered (HF added, special) tokens of the reference tokenizer.
SPECIAL_TOKENS = {
    "<_start>": 1,
    "<_end>": 2,
    "<_pad>": 3,
    "<_user>": 4,
    "<_bot>": 5,
    "<_system>": 6,
    "<think>": 9,
    "</think>": 10,
    "<tool_call>": 11,
    "</tool_call>": 12,
    "<tool_response>": 13,
    "</tool_response>": 14,
}
SPECIAL_IDS = frozenset(SPECIAL_TOKENS.values())
# SentencePiece CONTROL pieces the reference does NOT register.  Their text is
# encoded as literal pieces; their ids decode to "" (SentencePiece semantics).
CONTROL_PIECES = {
    7: "<_observation>",
    8: "<_sep>",
    **{14 + k: f"<_control{k}>" for k in range(1, 10)},
}
CONTROL_IDS = frozenset(CONTROL_PIECES)

TOKENIZER_JSON = "tokenizer.json"
STAMP_NAME = "xing4_0_tokenizer_parity.json"
STAMP_SCHEMA = "mlx2.xing4_0.tokenizer-parity/1"
STAMPED_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "chat_template.jinja")
FAST_TOKENIZER_CLASS = "TokenizersBackend"

_NORMAL, _SPECIAL, _CONTROL, _UNK, _BYTE = 0, 1, 2, 3, 4


class XingTokenizerError(RuntimeError):
    """The Xing4.0 tokenizer cannot be loaded with proven parity."""


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Exact reference decode
# --------------------------------------------------------------------------


class _Tables:
    """Per-id piece text and kind, validated against the known Xing4.0 layout."""

    def __init__(self, pieces: list[str]):
        if len(pieces) != VOCAB_SIZE:
            raise XingTokenizerError(f"expected {VOCAB_SIZE} pieces, got {len(pieces)}")
        expected = {UNK_ID: "<_unk>", SPIECE_ID: SPIECE, **CONTROL_PIECES}
        expected.update({i: s for s, i in SPECIAL_TOKENS.items()})
        expected.update({BYTE_OFFSET + b: f"<0x{b:02X}>" for b in range(256)})
        wrong = [i for i, s in expected.items() if pieces[i] != s]
        if wrong:
            raise XingTokenizerError(f"vocabulary layout is not Xing4.0 (ids {wrong[:8]})")
        kinds = bytearray(VOCAB_SIZE)
        for i in SPECIAL_IDS:
            kinds[i] = _SPECIAL
        for i in CONTROL_IDS:
            kinds[i] = _CONTROL
        kinds[UNK_ID] = _UNK
        for b in range(256):
            kinds[BYTE_OFFSET + b] = _BYTE
        self.pieces = pieces
        self.kinds = bytes(kinds)

    @classmethod
    def of(cls, tokenizer) -> "_Tables":
        tables = getattr(tokenizer, "_xing_tables", None)
        if tables is None:
            base = getattr(tokenizer, "_tokenizer", None)  # TokenizerWrapper
            if base is not None and not hasattr(base, "backend_tokenizer"):
                base = None
            source = base if base is not None else tokenizer
            tables = getattr(source, "_xing_tables", None)
            if tables is None:
                tables = cls(list(source.convert_ids_to_tokens(list(range(VOCAB_SIZE)))))
                try:
                    object.__setattr__(source, "_xing_tables", tables)
                except (AttributeError, TypeError):
                    pass
        return tables


def _utf8_width(lead: int) -> int:
    if lead < 0x80:
        return 1
    if lead & 0xE0 == 0xC0:
        return 2
    if lead & 0xF0 == 0xE0:
        return 3
    if lead & 0xF8 == 0xF0:
        return 4
    return 0


class _SpmRun:
    """SentencePiece ``DecodeIds`` for one run of pieces between specials.

    Mirrors sentencepiece 0.2 exactly: one leading U+2581 is consumed from the
    first non-control piece of the run; control pieces decode to "" and keep
    that expectation; ``<_unk>`` decodes to `` ⁇ ``; consecutive byte pieces
    are decoded as UTF-8 with one U+FFFD per invalid byte.  Incremental: bytes
    are held only while the pending sequence is still undetermined.
    """

    __slots__ = ("tables", "bos", "pending")

    def __init__(self, tables: _Tables):
        self.tables = tables
        self.bos = True
        self.pending = bytearray()

    def _drain(self, final: bool) -> str:
        out = []
        pending = self.pending
        while pending:
            width = _utf8_width(pending[0])
            if width == 1:
                out.append(chr(pending[0]))
                del pending[0]
                continue
            if width == 0:
                out.append("�")
                del pending[0]
                continue
            if len(pending) < width:
                if not final:
                    break
                out.append("�")
                del pending[0]
                continue
            try:
                out.append(bytes(pending[:width]).decode("utf-8"))
                del pending[:width]
            except UnicodeDecodeError:
                out.append("�")
                del pending[0]
        return "".join(out)

    def push(self, token_id: int) -> str:
        kind = self.tables.kinds[token_id]
        if kind == _BYTE:
            self.bos = False
            self.pending.append(token_id - BYTE_OFFSET)
            return self._drain(False)
        if kind == _CONTROL:
            return self._drain(True)
        head = self._drain(True)
        if kind == _UNK:
            self.bos = False
            return head + UNK_SURFACE
        piece = self.tables.pieces[token_id]
        if self.bos and piece.startswith(SPIECE):
            piece = piece[1:]
        self.bos = False
        return head + piece.replace(SPIECE, " ")

    def end(self) -> str:
        text = self._drain(True)
        self.bos = True
        return text


def _check_id(token_id) -> int:
    token_id = int(token_id)
    if not 0 <= token_id < VOCAB_SIZE:
        raise IndexError(f"token id {token_id} is out of range for Xing4.0")
    return token_id


def decode_ids(tables: _Tables, token_ids, skip_special_tokens: bool = False) -> str:
    """Exact ``Xing4_0Tokenizer.decode`` (``clean_up_tokenization_spaces=False``).

    The reference splits the id list into runs at registered special tokens
    and runs ``sp.decode`` on each run; with ``skip_special_tokens`` the
    specials are removed first, so adjacent runs merge into one.
    """
    run = _SpmRun(tables)
    out = []
    kinds = tables.kinds
    for token_id in token_ids:
        token_id = _check_id(token_id)
        if kinds[token_id] == _SPECIAL:
            if skip_special_tokens:
                continue
            out.append(run.end())
            out.append(tables.pieces[token_id])
            continue
        out.append(run.push(token_id))
    out.append(run.end())
    return "".join(out)


def _exact_decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=None, **kwargs):
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    if isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if clean_up_tokenization_spaces:
        raise ValueError("Xing4.0 reference decode does not clean up tokenization spaces")
    return decode_ids(_Tables.of(self), token_ids, skip_special_tokens)


def _install_exact_decode(tokenizer):
    cls = type(tokenizer)
    if not getattr(cls, "_xing_exact_decode", False):
        cls = type("Xing4_0FastTokenizer", (cls,), {"_decode": _exact_decode, "_xing_exact_decode": True})
    tokenizer.__class__ = cls
    _Tables.of(tokenizer)
    return tokenizer


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


class XingStreamingDetokenizer:
    """Streaming decode whose ``text`` equals ``decode(tokens)`` at every step.

    Linear time.  Text is final as soon as it is emitted: only an incomplete
    UTF-8 byte sequence is ever held back.  Special tokens appear as their
    markup (the reference decode with ``skip_special_tokens=False``), which the
    output parser consumes.
    """

    def __init__(self, tokenizer):
        self._tables = _Tables.of(tokenizer)
        self.reset()

    def reset(self):
        self.offset = 0
        self.text = ""
        self.tokens = []
        self._run = _SpmRun(self._tables)

    def copy(self) -> "XingStreamingDetokenizer":
        clone = object.__new__(type(self))
        clone._tables = self._tables
        clone.reset()
        return clone

    def add_token(self, token):
        token = _check_id(token)
        self.tokens.append(token)
        if self._tables.kinds[token] == _SPECIAL:
            self.text += self._run.end() + self._tables.pieces[token]
        else:
            self.text += self._run.push(token)

    def finalize(self):
        self.text += self._run.end()

    @property
    def last_segment(self):
        text = self.text
        segment = text[self.offset :]
        self.offset = len(text)
        return segment


def make_tokenizer_wrapper(tokenizer, eos_token_ids=None):
    """The mlx2 ``TokenizerWrapper`` with the exact Xing streaming detokenizer."""
    from ..runtime.tokenizer_utils import TokenizerWrapper

    return TokenizerWrapper(
        tokenizer,
        detokenizer_class=XingStreamingDetokenizer,
        eos_token_ids=list(eos_token_ids) if eos_token_ids is not None else [EOS_ID],
    )


# --------------------------------------------------------------------------
# Stamp, fast loader, reference fallback
# --------------------------------------------------------------------------


def fast_tokenizer_config(reference_config: dict) -> dict:
    """The artifact ``tokenizer_config.json`` for the fast tokenizer."""
    config = copy.deepcopy(reference_config)
    config.pop("auto_map", None)
    config["tokenizer_class"] = FAST_TOKENIZER_CLASS
    config["use_fast"] = True
    config["add_bos_token"] = False
    config["add_eos_token"] = False
    config["clean_up_tokenization_spaces"] = False
    config.pop("sp_model_kwargs", None)
    return config


def make_stamp(artifact, *, report: dict, versions: dict, generated_at: str, reference_code=None) -> dict:
    artifact = Path(artifact)
    reference_code = Path(reference_code) if reference_code else artifact / "tokenization_xing4_0.py"
    return {
        "schema": STAMP_SCHEMA,
        "status": report["status"],
        "files": {name: sha256_file(artifact / name) for name in STAMPED_FILES},
        "reference": "tokenization_xing4_0.Xing4_0Tokenizer (slow, trust_remote_code)",
        "reference_sha256": sha256_file(reference_code) if reference_code.exists() else None,
        "counts": report["counts"],
        "seed": report["seed"],
        "versions": versions,
        "generated_at": generated_at,
        "generator": "scripts/xing4_0_tokenizer.py",
    }


def verify_parity_stamp(model_path) -> dict:
    """Fail closed unless the on-disk tokenizer files are the ones that passed parity."""
    path = Path(model_path)
    stamp_path = path / STAMP_NAME
    if not stamp_path.exists():
        raise XingTokenizerError(f"no tokenizer parity stamp at {stamp_path}")
    try:
        stamp = json.loads(stamp_path.read_text())
    except ValueError as exc:
        raise XingTokenizerError(f"unreadable tokenizer parity stamp: {exc}") from None
    if stamp.get("schema") != STAMP_SCHEMA or stamp.get("status") != "pass":
        raise XingTokenizerError("tokenizer parity stamp is not a passing stamp of the expected schema")
    files = stamp.get("files") or {}
    for name in STAMPED_FILES:
        target = path / name
        if not target.exists():
            raise XingTokenizerError(f"stamped tokenizer file missing: {name}")
        if files.get(name) != sha256_file(target):
            raise XingTokenizerError(f"{name} changed since the tokenizer parity check")
    if (stamp.get("counts") or {}).get("fuzz_strings", 0) < 20000:
        raise XingTokenizerError("tokenizer parity stamp has an undersized fuzz corpus")
    return stamp


def load_fast_unverified(model_path):
    """Load ``tokenizer.json`` without the stamp check (build/parity use only)."""
    # The fast backend class directly: ``AutoTokenizer`` would first parse the
    # model ``config.json`` (custom ``xing4_0`` type with an ``auto_map``) and
    # demand remote code.  No remote code is imported on this path.
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_path), local_files_only=True)
    if not hasattr(tokenizer, "backend_tokenizer"):
        raise XingTokenizerError(f"expected a fast tokenizer, got {type(tokenizer).__name__}")
    if len(tokenizer) != VOCAB_SIZE or tokenizer.add_bos_token or tokenizer.add_eos_token:
        raise XingTokenizerError("fast tokenizer configuration diverges from the reference")
    return _install_exact_decode(tokenizer)


def load_reference_tokenizer(model_path, reference_code=None):
    """The slow reference ``Xing4_0Tokenizer``; executes the checkpoint's Python.

    Uses the original tokenizer config (``tokenizer_config.reference.json``,
    kept by the converter) so the reference sees exactly its shipped settings.
    """
    import shutil
    import tempfile

    path = Path(model_path)
    code = Path(reference_code) if reference_code else path / "tokenization_xing4_0.py"
    for required in (path / "tokenizer.model", code):
        if not required.exists():
            raise XingTokenizerError(f"reference tokenizer file missing: {required}")
    try:
        import sentencepiece  # noqa: F401
    except ImportError:
        raise XingTokenizerError("reference tokenizer fallback needs sentencepiece") from None
    config_path = path / "tokenizer_config.reference.json"
    if not config_path.exists():
        config_path = path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    if config.get("tokenizer_class") != "Xing4_0Tokenizer":
        raise XingTokenizerError(f"{config_path.name} does not describe the reference tokenizer")
    from transformers import AutoTokenizer

    with tempfile.TemporaryDirectory() as tmp:
        for name in ("tokenizer.model", "chat_template.jinja"):
            if (path / name).exists():
                shutil.copy2(path / name, Path(tmp) / name)
        shutil.copy2(code, Path(tmp) / "tokenization_xing4_0.py")
        (Path(tmp) / "tokenizer_config.json").write_text(json.dumps(config))
        tokenizer = AutoTokenizer.from_pretrained(tmp, trust_remote_code=True)
    tokenizer.vocab_file = str(path / "tokenizer.model")
    if [tokenizer.convert_tokens_to_ids(s) for s in SPECIAL_TOKENS] != list(SPECIAL_TOKENS.values()):
        raise XingTokenizerError("reference tokenizer special ids diverge")
    return tokenizer


def load_tokenizer(model_path, *, allow_reference_fallback: bool = False):
    """Load the Xing4.0 tokenizer for serving.

    Returns ``(tokenizer, receipt)``.  The fast, parity-stamped tokenizer is the
    only default route.  When the stamp is absent or stale the load fails
    closed, unless ``allow_reference_fallback=True`` explicitly opts in to the
    slow reference tokenizer (executes the checkpoint's Python; needs
    sentencepiece).
    """
    try:
        stamp = verify_parity_stamp(model_path)
        tokenizer = load_fast_unverified(model_path)
        return tokenizer, {
            "route": "fast-tokenizer-json",
            "trust_remote_code": False,
            "parity_counts": stamp.get("counts"),
            "tokenizer_json_sha256": stamp["files"]["tokenizer.json"],
        }
    except XingTokenizerError as exc:
        if not allow_reference_fallback:
            raise
        reason = str(exc)
    tokenizer = load_reference_tokenizer(model_path)
    return tokenizer, {"route": "reference-slow", "trust_remote_code": True, "fallback_reason": reason}


# --------------------------------------------------------------------------
# Chat prompt
# --------------------------------------------------------------------------


def normalize_messages(messages: list[dict]) -> list[dict]:
    """OpenAI chat messages -> the shapes the Xing chat template reads.

    Tool-call ``arguments`` arrive as JSON strings from OpenAI clients; the
    template iterates ``arguments.items()``, so they must be objects.  Content
    part lists are text-only (the template drops anything else silently, so
    reject it instead).
    """
    messages = copy.deepcopy(list(messages))
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and message.get("role") != "tool":
            for part in content:
                if not (isinstance(part, str) or (isinstance(part, dict) and part.get("type") == "text")):
                    raise ValueError("Xing4.0 accepts text content parts only")
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments.strip() else {}
            if not isinstance(arguments, dict):
                raise TypeError("Xing4.0 tool-call arguments must be a JSON object")
            function["arguments"] = arguments
    return messages


def render_prompt(tokenizer, messages, tools=None, enable_thinking=True) -> list[int]:
    """Token ids of the Xing chat prompt with the generation prompt appended.

    ``enable_thinking`` True ends the prompt with ``<_bot><think>\\n`` (output
    starts inside reasoning); False ends it with ``<_bot></think>``.
    Identical to ``apply_chat_template(..., tokenize=True)`` of the reference.
    """
    text = tokenizer.apply_chat_template(
        normalize_messages(messages),
        tools=list(tools) if tools else None,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True if enable_thinking is None else bool(enable_thinking),
    )
    return list(tokenizer.encode(text, add_special_tokens=False))
