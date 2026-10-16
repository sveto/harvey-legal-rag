from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import make_chunks, make_token_aware_chunks
from .embeddings import DEFAULT_MODEL, build_dense, load_dense, token_diagnostics
from .io import read_jsonl, write_jsonl
from .parser import parse_directory
from .retrieval import LegalRetriever, load_reranker, write_bm25_metadata


def main() -> None:
    p = argparse.ArgumentParser(description="Structure-aware legal contract retrieval")
    sub = p.add_subparsers(dest="command", required=True)
    parse = sub.add_parser("parse", help="Parse DOCX documents into sections and chunks")
    parse.add_argument("--input", default="data/docx")
    parse.add_argument("--output", default="artifacts")
    parse.add_argument("--max-words", type=int, default=500)
    parse.add_argument("--token-aware", action="store_true", help="Use BGE tokenizer to enforce 480-token passage limit")
    parse.add_argument("--max-tokens", type=int, default=480)
    build = sub.add_parser("index", help="Build FAISS index and BM25 metadata")
    build.add_argument("--output", default="artifacts")
    build.add_argument("--device", default=None)
    build.add_argument("--batch-size", type=int, default=32)
    search = sub.add_parser("search", help="Query indexed documents")
    search.add_argument("query")
    search.add_argument("--output", default="artifacts")
    search.add_argument("--method", choices=["dense", "bm25", "hybrid", "hybrid_rerank"], default="hybrid_rerank")
    search.add_argument("--device", default=None)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--candidate-k", type=int, default=50)
    search.add_argument("--rerank-k", type=int, default=40)
    diag = sub.add_parser("diagnose", help="Check rank of a target section")
    diag.add_argument("query")
    diag.add_argument("--source", required=True)
    diag.add_argument("--section", required=True)
    diag.add_argument("--output", default="artifacts")
    diag.add_argument("--device", default=None)
    args = p.parse_args()
    out = Path(args.output)
    if args.command == "parse":
        sections, stats = parse_directory(args.input)
        if args.token_aware:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
            chunks = make_token_aware_chunks(sections, tokenizer, max_tokens=args.max_tokens)
        else:
            chunks = make_chunks(sections, args.max_words)
        write_jsonl(out / "legal_sections.jsonl", sections)
        write_jsonl(out / "legal_chunks_v2.jsonl", chunks)
        print(json.dumps({"documents": len(stats), "sections": len(sections),
                          "chunks": len(chunks), "oversized": sum(c["oversized"] for c in chunks),
                          "documents_detail": stats}, indent=2))
        print("NOTICE: re-run `legal-rag index` after parsing; previous index is stale.")
        return
    chunks = read_jsonl(out / "legal_chunks_v2.jsonl")
    if args.command == "index":
        _, encoder = build_dense(chunks, out, batch_size=args.batch_size, device=args.device)
        write_bm25_metadata(chunks, out)
        print(json.dumps(token_diagnostics(chunks, encoder), indent=2))
        return
    idx, encoder = load_dense(chunks, out, device=args.device)
    needs_rerank = args.command == "diagnose" or args.method == "hybrid_rerank"
    reranker = load_reranker(device=args.device) if needs_rerank else None
    retriever = LegalRetriever(chunks, idx, encoder, reranker)
    if args.command == "search":
        result = retriever.search(args.query, method=args.method, top_k=args.top_k,
                                  candidate_k=args.candidate_k, rerank_k=args.rerank_k)
        for item in result:
            print(f"{item['rank']:>2}. {item['score']:>8.3f} | {item['source']} | "
                  f"§ {item.get('legal_ref') or item['section_id']} | {item['section_title']}")
    else:
        print(json.dumps(retriever.diagnostic(args.query, args.source, args.section), indent=2))


if __name__ == "__main__":
    main()
