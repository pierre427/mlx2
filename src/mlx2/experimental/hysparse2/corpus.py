"""Local, provenance-bearing corpus preparation. No network or GPU access.

Inputs are explicitly selected in a JSON manifest. Private examples remain in
an external output directory; only aggregate receipts should enter source control.
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from contextlib import ExitStack
from pathlib import Path


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def text_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            v.get("text", "")
            for v in value
            if isinstance(v, dict)
            and v.get("type") in ("text", "input_text", "output_text")
        )
    return ""


SECRET = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?i:authorization\s*:\s*bearer\s+)\S+|(?i:(?:api[_-]?key|password|secret|access[_-]?token)\s*[=:]\s*[\x22\x27]?)[A-Za-z0-9/+_.-]{16,})"
)
SCAFFOLD = (
    "<environment_context>",
    "<permissions instructions>",
    "<system-reminder>",
    "# AGENTS.md instructions",
    "<oai-mem-citation>",
    "<skills_instructions>",
    "<system_instructions>",
)


def clean(text):
    # Drop entire credential-bearing rows; replacing a single token can leave
    # nearby fragments or teach invalid credentials as part of code solutions.
    if SECRET.search(text) or any(marker in text for marker in SCAFFOLD):
        return None
    text = re.sub(r"/Users/[^/\s]+/", "/Users/USER/", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    return text.strip() if len(text.strip()) >= 64 else None


def messages_text(messages):
    return "\n\n".join(
        f"{m['role']}: {text_content(m.get('content', ''))}"
        for m in messages
        if m.get("role") in ("user", "assistant") and text_content(m.get("content", ""))
    )


def rows(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def json_examples(source):
    for row in rows(Path(os.path.expandvars(source["path"])).expanduser()):
        split = str(row.get("split", "train")).lower()
        if split not in ("train", "training"):
            continue
        if source["kind"] == "magicoder":
            text = "user: " + row["problem"] + "\n\nassistant: " + row["solution"]
            group = digest(row.get("seed") or row["problem"])
        elif source["kind"] == "public-tools":
            parts = []
            for message in row["messages"]:
                role = message.get("role")
                if role not in ("user", "assistant", "tool"):
                    continue
                content = text_content(message.get("content"))
                if message.get("reasoning_content"):
                    content = (
                        "reasoning: " + message["reasoning_content"] + "\n" + content
                    )
                if message.get("tool_calls"):
                    content += "\ntool_calls: " + json.dumps(
                        message["tool_calls"], ensure_ascii=False
                    )
                if role == "tool":
                    identity = {
                        k: message[k] for k in ("name", "tool_call_id") if k in message
                    }
                    content = json.dumps(identity, ensure_ascii=False) + "\n" + content
                parts.append(role + ": " + content)
            text = "\n\n".join(parts)
            # Keep multiple trajectories for the same task in one split.
            group = digest(
                next(
                    (
                        text_content(m.get("content"))
                        for m in row["messages"]
                        if m.get("role") == "user"
                    ),
                    row["uuid"],
                )
            )
        elif source["kind"] == "parquet-text":
            text = row[source.get("text_field", "text")]
            group = digest(str(row.get(source.get("group_field", "prompt")) or text))
        elif source["kind"] == "nkos":
            text = json.dumps(
                {
                    k: row[k]
                    for k in (
                        "state",
                        "claim",
                        "witness",
                        "verdict",
                        "why_checkable",
                        "next_action",
                        "label",
                        "code",
                        "lang",
                    )
                    if k in row
                },
                ensure_ascii=False,
            )
            group = str(row.get("source") or row.get("cwe") or digest(text))
        else:
            messages = row.get("messages") or row.get("prompt", []) + row.get(
                "completion", []
            )
            text = messages_text(messages) if isinstance(messages, list) else ""
            if not text:
                text = str(row.get("text", ""))
            group = str(
                row.get("pair_id")
                or row.get("session")
                or row.get("id")
                or digest(text)
            )
        yield text, source.get("group_namespace", source["id"]) + ":" + group


def archive_examples(source):
    """Only explicit technical project sessions and user/final-assistant text."""
    root = Path(os.path.expandvars(source["path"])).expanduser()
    allowed = source["projects"]
    files = sorted(root.rglob("*.jsonl"))
    for path in files:
        if path.stat().st_size > source.get("max_file_bytes", 32 << 20):
            continue
        with path.open(encoding="utf-8") as f:
            first = f.readline()
        try:
            meta = json.loads(first).get("payload", {})
        except (ValueError, AttributeError):
            continue
        if not any(
            project.lower() in str(meta.get("cwd", "")).lower() for project in allowed
        ):
            continue
        session = str(meta.get("id") or meta.get("session_id") or path.stem)
        pending = []
        for row in rows(path):
            item = row.get("payload", {})
            if row.get("type") != "response_item" or item.get("type") != "message":
                continue
            role = item.get("role")
            text = text_content(item.get("content"))
            if role == "user":
                if clean(text) is not None:
                    pending.append({"role": "user", "content": text})
                else:
                    pending = []
            elif (
                role == "assistant"
                and (
                    item.get("channel") == "final"
                    or item.get("phase") == "final_answer"
                )
                and pending
            ):
                yield (
                    messages_text(pending + [{"role": "assistant", "content": text}]),
                    "session:" + session,
                )
                pending = []


def repository_examples(source):
    root = Path(os.path.expandvars(source["path"])).expanduser()
    paths = subprocess.check_output(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", source["revision"]],
        text=True,
    ).splitlines()
    for name in paths:
        if not any(name.startswith(prefix) for prefix in source["prefixes"]) or Path(
            name
        ).suffix not in (".py", ".metal", ".md", ".rst"):
            continue
        if any(
            part in name.lower()
            for part in ("/test", "benchmark", "qualification", "/runs/")
        ):
            continue
        data = subprocess.check_output(
            ["git", "-C", str(root), "show", source["revision"] + ":" + name]
        )
        if len(data) > source.get("max_file_bytes", 128 << 10):
            continue
        # Hold out whole files, including all chunks of a large source file.
        text = data.decode("utf-8")
        for start in range(0, len(text), source.get("chunk_chars", 16000)):
            yield (
                f"File: {name}\n\n{text[start : start + source.get('chunk_chars', 16000)]}",
                source["id"] + ":" + name,
            )


def prepare(manifest, output, *, limit=2000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seen = set()
    counts = Counter()
    sources = []
    with (
        (output / "train.jsonl").open("w") as train,
        (output / "valid.jsonl").open("w") as valid,
    ):
        for source in manifest["sources"]:
            if source.get("status") != "local-approved":
                continue
            kind = source["kind"]
            reader = (
                archive_examples
                if kind == "codex"
                else repository_examples
                if kind == "repository"
                else json_examples
            )
            accepted = 0
            for raw, group in reader(source):
                if accepted >= limit:
                    break
                value = clean(raw)
                if value is None:
                    counts["rejected"] += 1
                    continue
                if len(value) > 200000:
                    counts["oversize"] += 1
                    continue
                key = digest(re.sub(r"\s+", " ", value))
                if key in seen:
                    counts["duplicate"] += 1
                    continue
                seen.add(key)
                split = "valid" if int(digest(group)[:8], 16) % 100 < 5 else "train"
                item = {
                    "text": value,
                    "sha256": key,
                    "group": digest(group),
                    "source": source["id"],
                    "license": source["license"],
                    "quality": source.get("quality", "unverified"),
                }
                (valid if split == "valid" else train).write(
                    json.dumps(item, ensure_ascii=False) + "\n"
                )
                counts[split] += 1
                counts[source["id"]] += 1
                accepted += 1
            source_receipt = dict(source)
            if kind not in ("codex", "repository"):
                h = hashlib.sha256()
                with (
                    Path(os.path.expandvars(source["path"]))
                    .expanduser()
                    .open("rb") as f
                ):
                    for block in iter(lambda: f.read(8 << 20), b""):
                        h.update(block)
                source_receipt["file_sha256"] = h.hexdigest()
            sources.append(source_receipt)
    receipt = {
        "schema": "mlx2.hysparse2-corpus.v1",
        "counts": dict(counts),
        "sources": sources,
        "policy": "Exact normalized-text dedup; 5% hash group holdout; source train splits only; secret/scaffold rejection. No automatic near-duplicate or benchmark-content decontamination claim.",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def tokenize(corpus, output, *, vocab_size=32768, tokenizer_path=None):
    """Train byte-level BPE on train only; stream EOS-delimited uint32 NPYs."""
    import numpy as np
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    corpus, output = Path(corpus), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if tokenizer_path is not None:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        if (
            tokenizer.get_vocab_size() > vocab_size
            or tokenizer.token_to_id("<eos>") is None
        ):
            raise ValueError("incompatible existing tokenizer")
    else:
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(
            (r["text"] for r in rows(corpus / "train.jsonl")), trainer
        )
    tokenizer.save(str(output / "tokenizer.json"))
    sha = hashlib.sha256((output / "tokenizer.json").read_bytes()).hexdigest()
    eos = tokenizer.token_to_id("<eos>")
    counts = {}
    source_files = {}
    source_tokens = Counter()
    for split in ("train", "valid"):
        raw = output / (split + ".u32")
        source_handles = {}
        with ExitStack() as stack, raw.open("wb") as f:
            for row in rows(corpus / (split + ".jsonl")):
                encoded = tokenizer.encode(row["text"]).ids + [eos]
                source_tokens[row.get("source", "unknown")] += len(encoded)
                encoded = np.asarray(encoded, dtype=np.uint32)
                encoded.tofile(f)
                source = row.get("source", "unknown")
                if not re.fullmatch(r"[A-Za-z0-9_-]+", source):
                    raise ValueError("unsafe source ID")
                if source not in source_handles:
                    name = f"{split}-{source}.u32"
                    source_handles[source] = stack.enter_context(
                        (output / name).open("wb")
                    )
                encoded.tofile(source_handles[source])
        count = raw.stat().st_size // 4
        if not count:
            raise ValueError(f"{split} split has no tokens")
        values = np.memmap(raw, dtype=np.uint32, mode="r")
        final = np.lib.format.open_memmap(
            output / (split + ".npy"), mode="w+", dtype=np.uint32, shape=(count,)
        )
        for start in range(0, count, 1 << 20):
            final[start : start + (1 << 20)] = values[start : start + (1 << 20)]
        final.flush()
        del values, final
        raw.unlink()
        counts[split] = count
        source_files[split] = {}
        for source in source_handles:
            source_raw = output / f"{split}-{source}.u32"
            n = source_raw.stat().st_size // 4
            source_array = np.memmap(source_raw, dtype=np.uint32, mode="r")
            name = f"{split}-{source}.npy"
            result = np.lib.format.open_memmap(
                output / name, mode="w+", dtype=np.uint32, shape=(n,)
            )
            for start in range(0, n, 1 << 20):
                result[start : start + (1 << 20)] = source_array[
                    start : start + (1 << 20)
                ]
            result.flush()
            del source_array, result
            source_raw.unlink()
            source_files[split][source] = {"path": name, "tokens": n}
    receipt = {
        "schema": "mlx2.hysparse2-tokens.v1",
        "tokenizer_sha256": sha,
        "vocab_size": tokenizer.get_vocab_size(),
        "tokens": counts,
        "source_tokens": dict(source_tokens),
        "source_files": source_files,
        "reused_tokenizer": tokenizer_path is not None,
        "source_receipt_sha256": hashlib.sha256(
            (corpus / "receipt.json").read_bytes()
        ).hexdigest(),
        "format": "uint32 NPY; documents separated by EOS; causal stream training",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "tokenize"])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=2000, help="accepted rows per source")
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument(
        "--tokenizer", type=Path, help="Reuse a tokenizer without changing token IDs"
    )
    a = p.parse_args(argv)
    if a.limit < 1 or a.vocab_size < 260:
        p.error("positive row limit and vocabulary >=260 required")
    result = (
        prepare(json.loads(a.input.read_text()), a.output, limit=a.limit)
        if a.action == "prepare"
        else tokenize(
            a.input, a.output, vocab_size=a.vocab_size, tokenizer_path=a.tokenizer
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
