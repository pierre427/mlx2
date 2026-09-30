import json

import pytest

from mlx2.experimental.hysparse2.corpus import (
    archive_examples,
    clean,
    prepare,
    tokenize,
)


def test_secret_and_scaffold_filter():
    assert clean("example " + "sk-" + "a" * 32 + " content" * 20) is None
    assert clean("<environment_context>" + "context " * 30) is None
    text = clean(
        "Develop /Users/alice/project/kernel.py with contact alice@example.org and improve the normalization kernel."
    )
    assert "/Users/USER/" in text and "[EMAIL]" in text and "alice" not in text


def test_group_holdout_exact_dedup_and_existing_eval(tmp_path):
    path = tmp_path / "source.jsonl"
    rows = [
        {
            "pair_id": str(i // 2),
            "split": "train",
            "messages": [
                {"role": "user", "content": f"Explain example {i}: " + "logic " * 20},
                {"role": "assistant", "content": "A careful conclusion."},
            ],
        }
        for i in range(200)
    ]
    rows += [rows[0], {"split": "test", "text": "MUST NOT LEAK " * 20}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = prepare(
        {
            "sources": [
                {
                    "id": "logic",
                    "path": str(path),
                    "kind": "jsonl",
                    "status": "local-approved",
                    "license": "private",
                }
            ]
        },
        tmp_path / "out",
    )
    train = [
        json.loads(l) for l in (tmp_path / "out/train.jsonl").read_text().splitlines()
    ]
    valid = [
        json.loads(l) for l in (tmp_path / "out/valid.jsonl").read_text().splitlines()
    ]
    assert result["counts"]["duplicate"] == 1 and len(train) + len(valid) == 200
    assert {r["group"] for r in train}.isdisjoint(r["group"] for r in valid)
    assert all("MUST NOT LEAK" not in r["text"] for r in train + valid)


def test_archive_excludes_tools_analysis_and_other_projects(tmp_path):
    def item(role, text, channel=None):
        return {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": role,
                "channel": channel,
                "content": [{"type": "text", "text": text}],
            },
        }

    data = [
        {"type": "session_meta", "payload": {"cwd": "/repo/mlx2", "id": "abc"}},
        item("developer", "system secret " * 20),
        item(
            "user",
            "Explain causal masking in tiled attention and how a future token should be excluded.",
        ),
        item("assistant", "PRIVATE ANALYSIS", "analysis"),
        {
            "type": "response_item",
            "payload": {"type": "function_call_output", "output": "TOOL SECRET"},
        },
        item(
            "assistant",
            "Use absolute query and key positions to mask all future keys before the softmax.",
            "final",
        ),
    ]
    (tmp_path / "session.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
    source = {"path": str(tmp_path), "projects": ["/mlx2"]}
    found = list(archive_examples(source))
    assert len(found) == 1
    assert "PRIVATE ANALYSIS" not in found[0][0] and "SECRET" not in found[0][0]
    data[-1]["payload"].pop("channel")
    data[-1]["payload"]["phase"] = "final_answer"
    (tmp_path / "session.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
    assert len(list(archive_examples(source))) == 1
    source["projects"] = ["/other"]
    assert list(archive_examples(source)) == []


def test_tokenizer_roundtrip_and_npy(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer

    root = tmp_path / "corpus"
    root.mkdir()
    for split in ("train", "valid"):
        (root / (split + ".jsonl")).write_text(
            json.dumps({"text": "A test of λ, café, 日本語 and code: x += 1."}) + "\n"
        )
    (root / "receipt.json").write_text("{}")
    result = tokenize(root, tmp_path / "tokens", vocab_size=300)
    tokenizer = Tokenizer.from_file(str(tmp_path / "tokens/tokenizer.json"))
    text = "English and λ 日本語"
    assert tokenizer.decode(tokenizer.encode(text).ids) == text
    for split in ("train", "valid"):
        values = np.load(tmp_path / "tokens" / (split + ".npy"))
        assert values.dtype == np.uint32 and len(values) == result["tokens"][split]
        assert values[-1] == tokenizer.token_to_id("<eos>")


def test_reused_tokenizer_preserves_ids_and_source_shards(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("tokenizers")

    root = tmp_path / "corpus"
    root.mkdir()
    for split in ("train", "valid"):
        (root / (split + ".jsonl")).write_text(
            json.dumps(
                {
                    "text": "Stable technical vocabulary: tensor, cache, Python and λ.",
                    "source": "code",
                }
            )
            + "\n"
        )
    (root / "receipt.json").write_text("{}")
    first = tokenize(root, tmp_path / "first", vocab_size=300)
    second = tokenize(
        root,
        tmp_path / "second",
        vocab_size=300,
        tokenizer_path=tmp_path / "first/tokenizer.json",
    )
    assert first["tokenizer_sha256"] == second["tokenizer_sha256"]
    for split in ("train", "valid"):
        file = second["source_files"][split]["code"]["path"]
        assert np.array_equal(
            np.load(tmp_path / "first" / (split + ".npy")),
            np.load(tmp_path / "second" / file),
        )
