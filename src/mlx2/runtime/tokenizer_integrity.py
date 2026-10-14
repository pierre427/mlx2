"""Keep a pack's pre-tokenizer on the split rule its tokenizer_config declares.

Qwen3.5+ packs declare ``pretokenize_regex`` in ``tokenizer_config.json``; the
current rule keeps combining marks (``\\p{M}``) inside a word.  Two transformers
defects drop it (transformers#49066 / #49077):

* ``Qwen2Tokenizer.__init__`` (v5) rebuilds the pre-tokenizer from a hardcoded
  pre-``\\p{M}`` regex, ignoring both ``tokenizer.json`` and the config;
* ``save_pretrained`` then writes that stale regex into ``tokenizer.json``.

Hindi, Thai and vowelled Arabic then split mid-grapheme: 1.2-2.1x the tokens,
and a segmentation the model was not trained on.  :func:`enforce_declared_regex`
repairs a loaded tokenizer, after :func:`restore_file_pretokenization` has put
back what a tokenizer class rebuilt instead of reading the file
(:func:`repair_loaded_tokenizer` runs both); :func:`check_pack` /
:func:`repair_pack` keep the files consistent at conversion time.  A pack that declares no regex is left
alone.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def declared_regex(model_path) -> str | None:
    config = Path(model_path) / "tokenizer_config.json"
    if not config.is_file():
        return None
    value = json.loads(config.read_text()).get("pretokenize_regex")
    return value if isinstance(value, str) and value else None


def _split_nodes(pre_tokenizer: dict | None) -> list[dict]:
    if not pre_tokenizer:
        return []
    if pre_tokenizer.get("type") == "Sequence":
        nodes = pre_tokenizer.get("pretokenizers") or []
    else:
        nodes = [pre_tokenizer]
    return [
        node
        for node in nodes
        if node.get("type") == "Split" and "Regex" in (node.get("pattern") or {})
    ]


def _retarget(pre_tokenizer: dict, want: str) -> dict | None:
    """Return ``pre_tokenizer`` with its single regex Split set to ``want``.

    ``None`` means there is nothing unambiguous to repair: no regex Split, or
    more than one (a layout this rule was not written for).
    """
    fixed = json.loads(json.dumps(pre_tokenizer))
    nodes = _split_nodes(fixed)
    if len(nodes) != 1:
        return None
    nodes[0]["pattern"]["Regex"] = want
    return fixed


def enforce_declared_regex(tokenizer, model_path) -> dict:
    """Repair a loaded HF tokenizer in place; return a receipt of what was done."""
    want = declared_regex(model_path)
    if want is None:
        return {"status": "undeclared"}
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or backend.pre_tokenizer is None:
        return {"status": "unsupported", "reason": "no fast backend pre-tokenizer"}
    current = json.loads(backend.pre_tokenizer.__getstate__())
    nodes = _split_nodes(current)
    if len(nodes) == 1 and nodes[0]["pattern"]["Regex"] == want:
        return {"status": "consistent"}
    fixed = _retarget(current, want)
    if fixed is None:
        return {"status": "unsupported", "reason": f"{len(nodes)} regex splits"}
    backend.pre_tokenizer = _materialize_pre_tokenizer(fixed)
    return {"status": "repaired", "previous": nodes[0]["pattern"]["Regex"] if nodes else None}


def _materialize_pre_tokenizer(serialized: dict):
    """A live pre-tokenizer from its JSON form, structure kept exactly.

    Built through a vocabulary-free carrier so the rest of its structure
    (ByteLevel flags, split behavior) survives unchanged.
    """
    from tokenizers import Tokenizer

    carrier = Tokenizer.from_str(
        json.dumps(
            {
                "version": "1.0",
                "model": {"type": "BPE", "vocab": {}, "merges": []},
                "pre_tokenizer": serialized,
            }
        )
    )
    return carrier.pre_tokenizer


# The encode-side components a tokenizer class may rebuild instead of reading
# them from ``tokenizer.json``.  The decoder is left alone: the drift seen on
# it is ByteLevel flags that do not change decoded bytes.
_FILE_COMPONENTS = ("normalizer", "pre_tokenizer")


def restore_file_pretokenization(tokenizer, model_path, *, reference=None) -> dict:
    """Put ``tokenizer.json``'s normalizer and pre-tokenizer back on a loaded tokenizer.

    Some transformers v5 tokenizer classes rebuild these from hardcoded rules
    instead of reading the checkpoint's file: ``Qwen2Tokenizer`` (the
    pre-``\\p{M}`` regex above), ``GPT2Tokenizer`` (a bare ByteLevel split that
    drops a pack's ``\\p{N}{1,3}`` digit rule) and ``CohereTokenizerFast``
    (transformers#49192).  Token ids then differ from the ones the model was
    trained on.  The file is authoritative, except where the config asks for
    ``add_prefix_space``, which transformers applies to the pre-tokenizer on
    purpose.
    """
    path = Path(model_path) / "tokenizer.json"
    if not path.is_file():
        return {"status": "missing", "reason": "no tokenizer.json"}
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return {"status": "unsupported", "reason": "no fast backend"}
    config = Path(model_path) / "tokenizer_config.json"
    if config.is_file() and json.loads(config.read_text()).get("add_prefix_space") is True:
        return {"status": "config_override", "reason": "add_prefix_space"}
    if reference is None:
        from tokenizers import Tokenizer

        reference = Tokenizer.from_file(str(path))
    # Both sides go through the same serializer, so formatting never reads
    # as drift.
    want = json.loads(reference.to_str())
    have = json.loads(backend.to_str())
    drift = [name for name in _FILE_COMPONENTS if have.get(name) != want.get(name)]
    if not drift:
        return {"status": "consistent"}
    for name in drift:
        setattr(backend, name, getattr(reference, name))
    return {"status": "restored", "components": drift}


# Probe texts for the load-time parity check: combining marks, NFC/NFD pairs,
# CJK, emoji ZWJ sequences, digit runs, edge whitespace, trailing newlines
# (transformers#49579's dropped normalizer) and special-token-looking text.
PARITY_PROBES = (
    "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e04\u0e23\u0e31\u0e1a",
    "\u0645\u064e\u0631\u0652\u062d\u064e\u0628\u064b\u0627",
    "caf\u00e9 cafe\u0301 \u00c5 A\u030a \u1100\u1161\u11a8 \uac01",
    "\u4f60\u597d\uff0c\u4e16\u754c\u3002\u65e5\u672c\u8a9e \ud55c\uad6d\uc5b4",
    "\U0001f468\u200d\U0001f469\u200d\U0001f467 \U0001f44d\U0001f3fd \ufb01 \uff21\uff11",
    "Order 1234567 today 3.14159 2026-10-14 \u0661\u0662\u0663",
    "  hello  ",
    "hello\n",
    "\n\nfoo\n\t bar\r\n",
    "<|im_start|>user\nhello<|im_end|>\n<think>\n</think>\n",
    "def f(x):\n    return x**2  # z\u00e4hlen\n",
)


class TokenizerParityError(ValueError):
    """The loaded tokenizer encodes or decodes differently from its tokenizer.json."""


def _file_reference(model_path, reference=None):
    """``tokenizer.json`` as the tokenizers library reads it, under the declared rule."""
    if reference is None:
        from tokenizers import Tokenizer

        reference = Tokenizer.from_file(str(Path(model_path) / "tokenizer.json"))
    # Serving encodes one prompt at a time; a file's padding or truncation
    # settings are batch options the loaded tokenizer does not apply.
    reference.no_padding()
    reference.no_truncation()
    want = declared_regex(model_path)
    if want is not None:
        fixed = _retarget(json.loads(reference.pre_tokenizer.__getstate__()), want)
        if fixed is not None:
            reference.pre_tokenizer = _materialize_pre_tokenizer(fixed)
    return reference


def _outcome(encode, text):
    # A vocabulary without an unknown token cannot encode every probe; the
    # file and the loaded tokenizer must then fail alike.
    try:
        return ("ids", list(encode(text)))
    except Exception as error:  # noqa: BLE001 - compared, not handled
        return ("error", str(error))


def _parity_mismatches(tokenizer, reference) -> list[tuple[str, str]]:
    backend = tokenizer.backend_tokenizer
    found = []
    for text in PARITY_PROBES:
        want = _outcome(lambda t: reference.encode(t, add_special_tokens=False).ids, text)
        got = _outcome(lambda t: tokenizer.encode(t, add_special_tokens=False), text)
        if got != want:
            found.append(("encode", text))
        if want[0] == "ids" and backend.decode(want[1], skip_special_tokens=False) != (
            reference.decode(want[1], skip_special_tokens=False)
        ):
            found.append(("decode", text))
    return found


def check_tokenizer_parity(tokenizer, model_path, *, reference=None) -> dict:
    """Require the loaded tokenizer to behave like ``tokenizer.json``.

    Encodes the :data:`PARITY_PROBES` with the loaded tokenizer and with the
    file (under the declared split rule), and decodes the file's ids with
    both backends.  A tokenizer class that rebuilt its backend from defaults
    (transformers#49579: 5.18's Qwen3_5Tokenizer drops the file's normalizer,
    pre-tokenizer and decoder) then fails here instead of serving ids or text
    the model was not trained on.  A decoder that differs is restored from
    the file; any other difference raises :class:`TokenizerParityError`.
    """
    reference = _file_reference(model_path, reference)
    mismatches = _parity_mismatches(tokenizer, reference)
    restored = []
    if any(kind == "decode" for kind, _ in mismatches):
        tokenizer.backend_tokenizer.decoder = reference.decoder
        restored.append("decoder")
        mismatches = _parity_mismatches(tokenizer, reference)
    if mismatches:
        import transformers

        detail = "; ".join(f"{kind} {text!r}" for kind, text in mismatches[:4])
        raise TokenizerParityError(
            f"the loaded {type(tokenizer).__name__} (transformers "
            f"{transformers.__version__}) differs from {Path(model_path) / 'tokenizer.json'} "
            f"on {len(mismatches)} probe(s): {detail}. A tokenizer class rebuilt "
            "its backend instead of reading the file; refusing to serve ids the "
            "model was not trained on."
        )
    return {"status": "consistent", "probes": len(PARITY_PROBES), "restored": restored}


def repair_loaded_tokenizer(tokenizer, model_path, *, file_authoritative=True) -> dict:
    """Every load-time repair, in order; the receipt names what each did.

    The file comes first, then the declared rule: ``save_pretrained`` may have
    written a stale regex into the file, which the declared one overrides.
    ``file_authoritative=False`` is for a load that corrects the file on
    purpose (``fix_mistral_regex``), which restoring it would undo.  Last,
    :func:`check_tokenizer_parity` verifies the result against the file and
    raises if the loaded tokenizer still behaves differently; it is skipped
    where the load changes the file on purpose or has no file to compare.
    """
    reference = None
    path = Path(model_path) / "tokenizer.json"
    if file_authoritative and path.is_file():
        from tokenizers import Tokenizer

        # Parsed once: the restore reads it, then the parity check owns it.
        reference = Tokenizer.from_file(str(path))
    receipt = {
        "file": (
            restore_file_pretokenization(tokenizer, model_path, reference=reference)
            if file_authoritative
            else {"status": "skipped", "reason": "load corrects the file"}
        ),
        "declared": enforce_declared_regex(tokenizer, model_path),
    }
    if receipt["file"]["status"] in ("consistent", "restored"):
        receipt["parity"] = check_tokenizer_parity(tokenizer, model_path, reference=reference)
    else:
        receipt["parity"] = {
            "status": "skipped",
            "reason": receipt["file"].get("reason", receipt["file"]["status"]),
        }
    return receipt


def check_pack(model_path) -> dict:
    """Compare ``tokenizer.json``'s split rule with the declared one on disk."""
    path = Path(model_path)
    want = declared_regex(path)
    if want is None:
        return {"status": "undeclared"}
    tokenizer_json = path / "tokenizer.json"
    if not tokenizer_json.is_file():
        return {"status": "missing", "reason": "no tokenizer.json"}
    nodes = _split_nodes(json.loads(tokenizer_json.read_text()).get("pre_tokenizer"))
    if len(nodes) != 1:
        return {"status": "unsupported", "reason": f"{len(nodes)} regex splits"}
    have = nodes[0]["pattern"]["Regex"]
    return {"status": "consistent" if have == want else "mismatch", "have": have, "want": want}


def repair_pack(model_path) -> dict:
    """Rewrite ``tokenizer.json``'s split rule to the declared one.

    Only the regex string is replaced, so the rest of the file stays
    byte-identical; the write is atomic.  A symlinked file is repaired at its
    target.
    """
    report = check_pack(model_path)
    if report["status"] != "mismatch":
        return report
    tokenizer_json = (Path(model_path) / "tokenizer.json").resolve()
    raw = tokenizer_json.read_text()
    old = json.dumps(report["have"], ensure_ascii=False)
    new = json.dumps(report["want"], ensure_ascii=False)
    if raw.count(old) != 1:
        return {**report, "status": "unsupported", "reason": "regex not uniquely located"}
    patched = raw.replace(old, new)
    if _split_nodes(json.loads(patched).get("pre_tokenizer"))[0]["pattern"]["Regex"] != report["want"]:
        return {**report, "status": "unsupported", "reason": "patch did not take"}
    tmp = tokenizer_json.with_name(tokenizer_json.name + ".tmp")
    tmp.write_text(patched)
    os.replace(tmp, tokenizer_json)
    return {**report, "status": "repaired"}
