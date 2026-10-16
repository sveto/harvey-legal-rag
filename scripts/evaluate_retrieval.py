"""Compare five retrievers on evaluation-only evidence labels, on CPU or Colab."""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from legal_rag.evaluation import (
    CLAUDE_PREFIX,
    METHODS,
    file_sha256,
    load_benchmark,
    render_markdown,
    run_benchmark,
    validate_benchmark,
)
from legal_rag.io import digest_chunks, read_jsonl
from legal_rag.types import Record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts", help="Existing parsed corpus and FAISS index")
    parser.add_argument("--benchmark", default="eval/benchmark.json")
    parser.add_argument("--save-dir", default="artifacts/evaluation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS),
                        help="Non-Claude methods; pass '--methods' with a subset to skip the rest")
    parser.add_argument("--claude-models", nargs="*", default=[],
                        help="Claude agents to run (needs ANTHROPIC_API_KEY, spends credits). "
                             "Use SEARCH+ANSWER (e.g. claude-haiku-5-5+claude-sonnet-5-5) for separate models")
    parser.add_argument("--claude-max-cost-usd", type=float, default=0.75, help="Per-question cap")
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--pool-k", type=int, default=20)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--validate-only", action="store_true", help="Check labels without loading models")
    args = parser.parse_args()
    if args.threads <= 0 or not args.cutoffs or min(args.cutoffs) <= 0 or args.pool_k < max(args.cutoffs):
        parser.error("threads/cutoffs must be positive and pool-k must cover all cutoffs")
    args.methods = args.methods + [CLAUDE_PREFIX + m for m in args.claude_models]
    if not 0 <= args.max_rounds <= 3 or len(set(args.methods)) != len(args.methods):
        parser.error("max-rounds must be 0..3 and methods must be unique")
    benchmark = load_benchmark(args.benchmark)
    chunks = read_jsonl(Path(args.artifacts) / "legal_chunks_v2.jsonl")
    validate_benchmark(benchmark, chunks)
    print(f"Validated {len(benchmark['cases'])} questions against {len(chunks)} chunks.", flush=True)
    if args.validate_only:
        return

    import torch

    from legal_rag.embeddings import load_dense
    from legal_rag.retrieval import LegalRetriever, load_reranker
    torch.set_num_threads(args.threads)
    index, encoder = load_dense(chunks, args.artifacts, device=args.device)
    needs_reranker = bool(set(args.methods) & {"hybrid_rerank", "agentic_v2"}) or bool(args.claude_models)
    reranker = load_reranker(device=args.device) if needs_reranker else None
    retriever = LegalRetriever(chunks, index, encoder, reranker)
    claude_agents = {}
    if args.claude_models:
        from run_claude_agent import load_colab_key

        from legal_rag.claude_agent import (
            ClaudeAgentConfig,
            ClaudeEvidenceAgent,
            make_client,
        )
        from legal_rag.claude_agent import config_hash as claude_hash
        load_colab_key()
        for model in args.claude_models:
            search_model, _, answer_model = model.partition("+")
            cfg = ClaudeAgentConfig(model=search_model, answer_model=answer_model or None,
                                    max_cost_usd=args.claude_max_cost_usd)
            claude_agents[CLAUDE_PREFIX + model] = ClaudeEvidenceAgent(retriever, make_client(cfg), cfg)
            print(f"Claude agent {model}: config_hash={claude_hash(cfg)}", flush=True)
    # Warm up the shared encoder/cross-encoder outside the latency measurements.
    retriever.search("assignment consent termination", method="hybrid_rerank" if reranker else "hybrid", top_k=1)

    output = Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queries.jsonl").open("w", encoding="utf-8") as trace_file:
        def progress(row: Record) -> None:
            trace_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            trace_file.flush()
            print(f"{row['case_id']:24} {row['method']:14} {row['latency_ms']:8.0f} ms"
                  + (f" ERROR {row['error']}" if row["error"] else ""), flush=True)

        report = run_benchmark(retriever, benchmark, methods=args.methods, cutoffs=args.cutoffs,
                               pool_k=args.pool_k, max_rounds=args.max_rounds, progress=progress, claude_agents=claude_agents)
    packages = {}
    for name in ("torch", "sentence-transformers", "transformers", "faiss-cpu", "rank-bm25", "langgraph"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    root = Path(__file__).resolve().parents[1]
    report["metadata"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(args.benchmark), "benchmark_sha256": file_sha256(args.benchmark),
        "corpus_sha256": digest_chunks(chunks), "chunk_count": len(chunks),
        "document_count": len({c['source'] for c in chunks}),
        "index_config": json.loads((Path(args.artifacts) / "dense_index_config.json").read_text()),
        "device": args.device, "cpu_threads": args.threads,
        "hardware": torch.cuda.get_device_name() if args.device.startswith("cuda") else platform.machine(),
        "python": platform.python_version(), "packages": packages,
        "methods": args.methods, "cutoffs": args.cutoffs, "pool_k": args.pool_k,
        "candidate_k": 50, "rerank_k": {"baseline": 40, "agent": 24},
        "max_rounds": args.max_rounds, "per_facet": 6,
        "claude_models": args.claude_models,
        "claude_config": {m: __import__("dataclasses").asdict(a.config) for m, a in claude_agents.items()},
        "scope": "baselines: corpus-wide; agent: existing question-only source inference",
        "agent_order": "round robin over inferred facets, then unique sections; no score sorting",
        "latency": "single run after shared model warmup; excludes load/download, includes agent graph construction",
        "source_sha256": {str(p.relative_to(root)): file_sha256(p) for p in
                          [root / "src/legal_rag" / name for name in
                           ("evaluation.py", "retrieval.py", "evidence_agent.py", "embeddings.py",
                            "claude_agent.py", "claude_schemas.py")]
                          + [Path(__file__).resolve()]},
    }
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Saved benchmark report to {output / 'report.md'}", flush=True)
    if any(row["error"] for row in report["records"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
