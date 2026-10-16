"""Run the Claude-backed evidence agent on one question or on benchmark questions.

Needs ANTHROPIC_API_KEY (environment or Colab Secrets) and spends API credits.
Only question text is sent; benchmark labels are never read into prompts.
Use --dry-run to check indexes, config and the credential without any API call.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from legal_rag.claude_agent import (
    ClaudeAgentConfig,
    ClaudeEvidenceAgent,
    config_hash,
    make_client,
)
from legal_rag.embeddings import load_dense
from legal_rag.io import read_jsonl
from legal_rag.retrieval import LegalRetriever, load_reranker


def load_colab_key() -> None:
    """In Colab, copy the ANTHROPIC_API_KEY secret into this process's environment only."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
        os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
    except Exception:  # not in Colab or secret missing; make_client reports it
        pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?", help="Question text; omit with --benchmark")
    p.add_argument("--benchmark", default=None, help="eval/benchmark.json: run every case's question only")
    p.add_argument("--limit", type=int, default=None, help="Run only the first N benchmark cases")
    p.add_argument("--output", default="artifacts")
    p.add_argument("--save-dir", default=None, help="Default: artifacts/claude-runs/<model>-<UTC timestamp>")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--model", default=ClaudeAgentConfig.model)
    p.add_argument("--answer-model", default=None, help="Model for the final answer step (default: --model)")
    p.add_argument("--max-turns", type=int, default=ClaudeAgentConfig.max_turns)
    p.add_argument("--max-tool-calls", type=int, default=ClaudeAgentConfig.max_tool_calls)
    p.add_argument("--max-cost-usd", type=float, default=ClaudeAgentConfig.max_cost_usd)
    p.add_argument("--max-total-tokens", type=int, default=ClaudeAgentConfig.max_total_tokens)
    p.add_argument("--refusal-fallback", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if bool(args.question) == bool(args.benchmark):
        p.error("give either a question or --benchmark")

    config = ClaudeAgentConfig(model=args.model, answer_model=args.answer_model, max_turns=args.max_turns, max_tool_calls=args.max_tool_calls,
                               max_cost_usd=args.max_cost_usd, max_total_tokens=args.max_total_tokens,
                               refusal_fallback=args.refusal_fallback)
    if args.benchmark:
        cases = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))["cases"]
        questions = [(c["id"], c["question"]) for c in cases][: args.limit]
    else:
        questions = [("adhoc", args.question)]

    load_colab_key()
    print(f"model={config.model} config_hash={config_hash(config)} questions={len(questions)} "
          f"key_present={bool(os.environ.get('ANTHROPIC_API_KEY'))}")
    if args.dry_run:
        return
    client = make_client(config)
    chunks = read_jsonl(Path(args.output) / "legal_chunks_v2.jsonl")
    index, encoder = load_dense(chunks, args.output, device=args.device)
    reranker = None if args.no_rerank else load_reranker(device=args.device)
    agent = ClaudeEvidenceAgent(LegalRetriever(chunks, index, encoder, reranker), client, config)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_dir = Path(args.save_dir or f"artifacts/claude-runs/{config.model}-{stamp}")
    save_dir.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for case_id, question in questions:
        report = agent.run(question)
        (save_dir / f"{case_id}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        u = report["usage"]
        total += u["cost_usd"] or 0.0
        print(f"\n[{case_id}] {report['answer']['conclusion_type']} | tools={u['tool_calls']} "
              f"tokens={u['input_tokens']}+{u['output_tokens']} cost~${u['cost_usd']} api={u['api_ms']:.0f}ms")
        print(report["answer"]["answer"])
        for c in report["citations"]:
            print(f"  cite {c['evidence_id']}: {c['source']} §{c['legal_ref']} {c['section_title']}")
        for gap in report["answer"]["evidence_gaps"]:
            print(f"  gap: {gap}")
    print(f"\nTotal estimated cost: ${total:.4f}  (traces in {save_dir})")


if __name__ == "__main__":
    main()
