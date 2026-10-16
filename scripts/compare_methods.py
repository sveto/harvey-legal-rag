"""Four-mode comparison on the same queries; outputs JSON for later evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.embeddings import load_dense
from legal_rag.io import read_jsonl
from legal_rag.retrieval import LegalRetriever, load_reranker

QUERIES = [
    "Which agreements require consent to assignment or change of control?",
    "Does TerraNode have termination rights if Aldersgate is acquired by a competitor?",
    "What happens to Pinnacle's exclusive technology license after a change of control?",
    "Which credit agreements require repayment or acceleration upon a change of control?",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifacts")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    directory = Path(args.output)
    chunks = read_jsonl(directory / "legal_chunks_v2.jsonl")
    index, encoder = load_dense(chunks, directory, device=args.device)
    retriever = LegalRetriever(chunks, index, encoder, load_reranker(device=args.device))
    rows = []
    for query in QUERIES:
        print("\nQUERY:", query)
        for method in ("dense", "bm25", "hybrid", "hybrid_rerank"):
            results = retriever.search(query, method=method)
            print(" ", method, f"({results[0]['latency_ms']} ms)")
            for r in results:
                print(f"   {r['rank']}. {r['source']} §{r.get('legal_ref') or r['section_id']} {r['section_title']}")
                rows.append({"query": query, "method": method, "rank": r["rank"],
                             "source": r["source"], "section_id": r["section_id"],
                             "chunk_id": r["chunk_id"], "score": r["score"],
                             "latency_ms": r["latency_ms"]})
    (directory / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
