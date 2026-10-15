#!/usr/bin/env python3
"""Bounded public SciFact retrieval diagnostic; no qualification or GPU work.

Original claims/annotations: CC BY 4.0; abstracts: ODC-By 1.0.
https://github.com/allenai/scifact/blob/master/LICENSE.md
BEIR packaging and exact file revisions are recorded in the output.
Dataset bytes and output belong outside the repository. No download is implicit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

FILES = {
    "corpus.jsonl.gz": "2f4a2a264485e5b5b95ae9cd5534817447757423991a4767d6da811539c0853e",
    "queries.jsonl.gz": "c29574b9cb8fdf043df8494f689de15f4604ac4b585eb34bda9b6a389aa6c4d3",
    "test.tsv": "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
}


def stable_order(ids):
    return sorted(ids, key=lambda value: (hashlib.sha256(str(value).encode()).digest(), str(value)))


def metrics(ranking, relevant, cutoff=10):
    """Binary SciFact qrels, including relevant documents absent from candidates."""
    if not relevant or len(set(ranking)) != len(ranking):
        raise ValueError("metrics require nonempty judgments and unique ranked ids")
    ranks = [index + 1 for index, doc in enumerate(ranking[:cutoff]) if doc in relevant]
    ideal = sum(1 / math.log2(index + 2) for index in range(min(len(relevant), cutoff)))
    return {
        "recall_at_10": len(ranks) / len(relevant),
        "mrr_at_10": 1 / ranks[0] if ranks else 0.0,
        "ndcg_at_10": sum(1 / math.log2(rank + 1) for rank in ranks) / ideal,
    }


def load_dataset(path):
    for name, digest in FILES.items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"public dataset content mismatch: {name}")
    tables = []
    for name in ("corpus.jsonl.gz", "queries.jsonl.gz"):
        with gzip.open(path / name, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        table = {str(row["_id"]): row for row in rows}
        if len(table) != len(rows):
            raise ValueError("duplicate public dataset ids")
        tables.append(table)
    relevant = defaultdict(set)
    with (path / "test.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if int(row["score"]) > 0:
                relevant[row["query-id"]].add(row["corpus-id"])
    corpus, queries = tables
    if any(query not in queries or not docs <= corpus.keys() for query, docs in relevant.items()):
        raise ValueError("public judgments have missing query or corpus ids")
    return corpus, queries, relevant


def lexical_scores(corpus, queries):
    """Untrained TF-IDF cosine baseline over the full original corpus."""
    terms = {key: Counter(re.findall(r"\w+", text.lower())) for key, text in corpus.items()}
    frequencies = Counter(term for counts in terms.values() for term in counts)
    idf = {term: math.log((1 + len(corpus)) / (1 + frequency)) + 1
           for term, frequency in frequencies.items()}
    inverted = defaultdict(list)
    for doc, counts in terms.items():
        weights = {term: (1 + math.log(count)) * idf[term] for term, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
        for term, value in weights.items():
            inverted[term].append((doc, value / norm))
    result = {}
    for key, text in queries.items():
        scores = defaultdict(float)
        weights = {term: (1 + math.log(count)) * idf[term]
                   for term, count in Counter(re.findall(r"\w+", text.lower())).items() if term in idf}
        norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
        for term, value in weights.items():
            for doc, weight in inverted[term]:
                scores[doc] += weight * value / norm
        result[key] = scores
    return result


def evaluate(args):
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    import numpy as np

    from mlx2.adapters.minilm import MiniLMAdapter
    from mlx2.adapters.minilm_reranker import MiniLMRerankerAdapter

    if args.output_dir.exists():
        raise ValueError("refusing to overwrite evaluation evidence")
    args.output_dir.mkdir(parents=True)
    corpus, queries, qrels = load_dataset(args.dataset_dir)
    query_ids = stable_order(qrels)[:24]
    texts = {key: (row["title"] + " " + row["text"]).strip() for key, row in corpus.items()}
    query_texts = {key: queries[key]["text"] for key in query_ids}
    lexical = lexical_scores(texts, query_texts)
    candidates = set(stable_order(corpus)[:128])
    for query in query_ids:
        candidates.update(qrels[query])
        candidates.update(sorted(corpus, key=lambda doc: (-lexical[query].get(doc, 0), doc))[:10])
    doc_ids = sorted(candidates)
    embedder = MiniLMAdapter(args.model)
    reranker = MiniLMRerankerAdapter(args.reranker) if args.reranker else None
    try:
        vectors, input_tokens = [], 0
        for start in range(0, len(doc_ids), 32):
            part, count = embedder.embed([texts[doc] for doc in doc_ids[start:start + 32]])
            vectors.extend(part)
            input_tokens += count
            print(f"embedded {min(start + 32, len(doc_ids))}/{len(doc_ids)} documents", flush=True)
        matrix = np.asarray(vectors, dtype=np.float32)
        query_vectors, count = embedder.embed(list(query_texts.values()))
        input_tokens += count
        rows, rerank_tokens = [], 0
        for query, vector in zip(query_ids, query_vectors):
            dense = matrix @ np.asarray(vector, dtype=np.float32)
            ranking = sorted(range(len(doc_ids)), key=lambda i: (-float(dense[i]), doc_ids[i]))
            ranked = [doc_ids[index] for index in ranking]
            baseline = sorted(doc_ids, key=lambda doc: (-lexical[query].get(doc, 0), doc))
            row = {"query_id": query, "relevant_ids": sorted(qrels[query]),
                   "embedding": metrics(ranked, qrels[query]), "lexical": metrics(baseline, qrels[query])}
            if reranker is not None:
                scores, count = reranker.rerank(query_texts[query], [texts[doc] for doc in ranked[:10]])
                rerank_tokens += count
                order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
                reranked = [ranked[index] for index in order]
                row["reranked_embedding_top10"] = metrics(reranked, qrels[query])
            rows.append(row)
            print(f"scored query {len(rows)}/{len(query_ids)}", flush=True)
        methods = [key for key in rows[0] if key not in {"query_id", "relevant_ids"}]
        result = {
            "schema": "mlx2.bounded-retrieval-evaluation.v1", "device": "cpu",
            "dataset": "SciFact BEIR test subset", "query_count": len(query_ids),
            "corpus_candidates": len(doc_ids), "original_corpus_size": len(corpus),
            "selection": "24 SHA256-ordered test query ids; all their gold documents + each query's full-corpus TF-IDF top10 +128 SHA256-ordered corpus ids; no model-dependent selection",
            "corpus_revision": "a75ae049398addde9b70f6b268875f5cbce99089",
            "qrels_revision": "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
            "dataset_sha256": FILES, "query_ids": query_ids, "document_ids": doc_ids,
            "embedding_artifact": embedder.identity,
            "reranker_artifact": reranker.identity if reranker else None,
            "embedding_input_tokens": input_tokens, "reranker_input_tokens": rerank_tokens,
            "metrics": {method: {metric: sum(row[method][metric] for row in rows) / len(rows)
                                  for metric in rows[0][method]} for method in methods},
            "per_query": rows,
            "limits": ["Reduced corpus includes gold documents, making recall optimistic relative to full BEIR.",
                       "No tuning, full benchmark, task-wide quality qualification or performance measurement.",
                       "Reranking only reorders embedding top10; missing retrieved relevance remains missing.",
                       "Model-native truncation: embeddings256 and query/document pairs512 tokens."],
        }
        (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["metrics"], indent=2), flush=True)
        return result
    finally:
        embedder.close()
        if reranker is not None:
            reranker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reranker", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
