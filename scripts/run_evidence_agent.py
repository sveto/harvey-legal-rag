"""Run local LangGraph evidence retrieval against an existing index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.embeddings import load_dense
from legal_rag.evidence_agent import build_agent, evidence_report
from legal_rag.io import read_jsonl
from legal_rag.retrieval import LegalRetriever, load_reranker


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--output", default="artifacts")
    p.add_argument("--source", default=None, help="Optional exact DOCX filename")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--max-rounds", type=int, default=2)
    p.add_argument("--save", default=None, help="Optional output JSON path")
    args = p.parse_args()
    chunks = read_jsonl(Path(args.output) / "legal_chunks_v2.jsonl")
    index, encoder = load_dense(chunks, args.output, device=args.device)
    reranker = None if args.no_rerank else load_reranker(device=args.device)
    retriever = LegalRetriever(chunks, index, encoder, reranker)
    state = build_agent(retriever, max_rounds=args.max_rounds).invoke({"question": args.question, "source": args.source})
    report = evidence_report(state)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved: {path}")
    print("SOURCE:", report["source"] or "unresolved (corpus-wide search)")
    print("STATUS:", report["status"], "| missing:", report["missing"])
    for facet, items in report["evidence"].items():
        print(f"\n[{facet}] {len(items)} candidate(s)")
        for item in items[:4]:
            print(f"  {item['source']} §{item['legal_ref']} {item['section_title']} ({item['score']:.3f})")
            print("  ", item["text"].replace("\n", " ")[:230])
    print("\nTRACE:", json.dumps(report["trace"], ensure_ascii=False))


if __name__ == "__main__":
    main()
