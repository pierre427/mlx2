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
    from tokenizers import Tokenizer

    # Materialize the pre-tokenizer through a vocabulary-free carrier so the
    # rest of its structure (ByteLevel flags, split behavior) is kept exactly.
    carrier = Tokenizer.from_str(
        json.dumps(
            {
                "version": "1.0",
                "model": {"type": "BPE", "vocab": {}, "merges": []},
                "pre_tokenizer": fixed,
            }
        )
    )
    backend.pre_tokenizer = carrier.pre_tokenizer
    return {"status": "repaired", "previous": nodes[0]["pattern"]["Regex"] if nodes else None}


# The encode-side components a tokenizer class may rebuild instead of reading
# them from ``tokenizer.json``.  The decoder is left alone: the drift seen on
# it is ByteLevel flags that do not change decoded bytes.
_FILE_COMPONENTS = ("normalizer", "pre_tokenizer")


def restore_file_pretokenization(tokenizer, model_path) -> dict:
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


def repair_loaded_tokenizer(tokenizer, model_path, *, file_authoritative=True) -> dict:
    """Every load-time repair, in order; the receipt names what each did.

    The file comes first, then the declared rule: ``save_pretrained`` may have
    written a stale regex into the file, which the declared one overrides.
    ``file_authoritative=False`` is for a load that corrects the file on
    purpose (``fix_mistral_regex``), which restoring it would undo.
    """
    return {
        "file": (
            restore_file_pretokenization(tokenizer, model_path)
            if file_authoritative
            else {"status": "skipped", "reason": "load corrects the file"}
        ),
        "declared": enforce_declared_regex(tokenizer, model_path),
    }


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
