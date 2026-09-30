import json

import pytest

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from mlx2.runtime.tokenizer_integrity import check_pack, enforce_declared_regex, repair_pack

OLD = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
NEW = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
THAI = "ประเทศไทยเป็นประเทศ"


def _pre_tokenizer(regex):
    return {
        "type": "Sequence",
        "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": regex}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
        ],
    }


def _pack(tmp_path, file_regex, declared=NEW):
    tok = {"version": "1.0", "model": {"type": "BPE", "vocab": {}, "merges": []},
           "pre_tokenizer": _pre_tokenizer(file_regex)}
    (tmp_path / "tokenizer.json").write_text(json.dumps(tok, indent=2))
    config = {"tokenizer_class": "PreTrainedTokenizerFast"}
    if declared:
        config["pretokenize_regex"] = declared
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    return tmp_path


def _pieces(backend):
    return [p for p, _ in backend.pre_tokenizer.pre_tokenize_str(THAI)]


def test_repair_pack_rewrites_only_the_regex(tmp_path):
    pack = _pack(tmp_path, OLD)
    before = (pack / "tokenizer.json").read_text()
    assert check_pack(pack)["status"] == "mismatch"
    assert repair_pack(pack)["status"] == "repaired"
    after = (pack / "tokenizer.json").read_text()
    assert check_pack(pack)["status"] == "consistent"
    assert after == before.replace(json.dumps(OLD), json.dumps(NEW))


def test_enforce_repairs_a_loaded_tokenizer(tmp_path):
    pack = _pack(tmp_path, OLD)
    tok = transformers.PreTrainedTokenizerFast(tokenizer_file=str(pack / "tokenizer.json"))
    broken = _pieces(tok.backend_tokenizer)
    assert enforce_declared_regex(tok, pack)["status"] == "repaired"
    fixed = _pieces(tok.backend_tokenizer)
    # Combining vowels no longer split Thai words apart.
    assert len(fixed) < len(broken)
    assert enforce_declared_regex(tok, pack)["status"] == "consistent"


def test_undeclared_pack_is_left_alone(tmp_path):
    pack = _pack(tmp_path, OLD, declared=None)
    tok = transformers.PreTrainedTokenizerFast(tokenizer_file=str(pack / "tokenizer.json"))
    before = _pieces(tok.backend_tokenizer)
    assert enforce_declared_regex(tok, pack) == {"status": "undeclared"}
    assert check_pack(pack) == {"status": "undeclared"}
    assert _pieces(tok.backend_tokenizer) == before


# ---------------------------------------------------------------------------
# A tokenizer class that rebuilds the pre-tokenizer instead of reading the file
# (transformers#49192 for Cohere; GPT2Tokenizer and Qwen2Tokenizer do the same).
# ---------------------------------------------------------------------------

from mlx2.runtime.tokenizer_integrity import (  # noqa: E402
    repair_loaded_tokenizer,
    restore_file_pretokenization,
)

DIGITS = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
NUMBER = "Order 1234567 today"


def _byte_level_pack(tmp_path, regex, tokenizer_class, **config):
    """A byte-level BPE pack whose tokenizer.json splits with ``regex``."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    backend = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(sorted(alphabet))}, merges=[]))
    tok = json.loads(backend.to_str())
    tok["pre_tokenizer"] = {
        "type": "Sequence",
        "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": regex}, "behavior": "Removed", "invert": True},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
        ],
    }
    tok["decoder"] = {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True}
    (tmp_path / "tokenizer.json").write_text(json.dumps(tok))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": tokenizer_class, **config})
    )
    return tmp_path


def _words(tokenizer, text):
    return [p for p, _ in tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)]


def test_gpt2_class_rebuild_is_restored_from_the_file(tmp_path):
    """GPT2Tokenizer (transformers v5) swaps the pack's Split rule for a bare
    ByteLevel split, so "1234567" is one piece instead of 123|456|7."""
    pack = _byte_level_pack(tmp_path, DIGITS, "GPT2Tokenizer", add_prefix_space=False)
    tok = transformers.AutoTokenizer.from_pretrained(str(pack), local_files_only=True)
    from tokenizers import Tokenizer

    reference = Tokenizer.from_file(str(pack / "tokenizer.json"))
    want = [p for p, _ in reference.pre_tokenizer.pre_tokenize_str(NUMBER)]
    if _words(tok, NUMBER) == want:
        pytest.skip("this transformers keeps the file's pre-tokenizer")
    receipt = restore_file_pretokenization(tok, pack)
    assert receipt == {"status": "restored", "components": ["pre_tokenizer"]}
    assert _words(tok, NUMBER) == want
    assert tok.encode(NUMBER, add_special_tokens=False) == reference.encode(NUMBER).ids
    assert restore_file_pretokenization(tok, pack) == {"status": "consistent"}


def test_rebuilt_normalizer_is_restored_too(tmp_path):
    pack = _byte_level_pack(tmp_path, DIGITS, "PreTrainedTokenizerFast")
    tok = transformers.PreTrainedTokenizerFast(tokenizer_file=str(pack / "tokenizer.json"))
    from tokenizers import normalizers, pre_tokenizers

    tok.backend_tokenizer.normalizer = normalizers.Lowercase()
    tok.backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    receipt = restore_file_pretokenization(tok, pack)
    assert receipt == {"status": "restored", "components": ["normalizer", "pre_tokenizer"]}
    assert tok.backend_tokenizer.normalizer is None
    assert tok.encode("ABC 1234", add_special_tokens=False) == tok.backend_tokenizer.encode("ABC 1234").ids


def test_add_prefix_space_config_is_left_to_transformers(tmp_path):
    pack = _byte_level_pack(tmp_path, DIGITS, "PreTrainedTokenizerFast", add_prefix_space=True)
    tok = transformers.PreTrainedTokenizerFast(tokenizer_file=str(pack / "tokenizer.json"))
    assert restore_file_pretokenization(tok, pack)["status"] == "config_override"


def test_repair_runs_the_file_then_the_declared_rule(tmp_path):
    # save_pretrained left the stale rule in the file; the config declares the
    # current one.  Restoring the file alone would keep the stale rule.
    pack = _pack(tmp_path, OLD)
    tok = transformers.PreTrainedTokenizerFast(tokenizer_file=str(pack / "tokenizer.json"))
    from tokenizers import pre_tokenizers

    tok.backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    receipt = repair_loaded_tokenizer(tok, pack)
    assert receipt["file"]["status"] == "restored"
    assert receipt["declared"]["status"] == "repaired"
    fixed = json.loads(tok.backend_tokenizer.pre_tokenizer.__getstate__())
    assert fixed["pretokenizers"][0]["pattern"]["Regex"] == NEW
    skipped = repair_loaded_tokenizer(tok, pack, file_authoritative=False)
    assert skipped["file"]["status"] == "skipped"
    assert skipped["declared"]["status"] == "consistent"


def test_adapters_that_correct_the_file_keep_their_correction():
    """fix_mistral_regex corrects tokenizer.json's regex on purpose; restoring
    the file afterwards would undo it (Laguna XS/S 2.1)."""
    from pathlib import Path

    adapters = Path(__file__).resolve().parents[1] / "src" / "mlx2" / "adapters"
    fixing = [p for p in adapters.glob("*.py") if "fix_mistral_regex=True" in p.read_text()]
    assert fixing, "expected the Laguna adapters to load with fix_mistral_regex"
    for path in fixing:
        assert "file_authoritative=False" in path.read_text(), path.name


def test_pack_check_fails_a_declared_rule_without_tokenizer_json(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"pretokenize_regex": NEW}))
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_tokenizer_pretokenize.py"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)], capture_output=True, text=True,
        env={"PYTHONPATH": str(script.parents[1] / "src")},
    )
    assert "missing" in result.stdout and result.returncode == 1
