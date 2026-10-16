# Project report: Harvey change-of-control RAG with a Claude agent

Single-page summary of what was built, what was measured, what is known to be weak, and a self-assessment against the
four assignment criteria. Details live in the linked documents; numbers here come from the committed result folders.

## 1. What was built

Task: RAG over a subset of the Harvey legal dataset (Corporate M&A, change of control; 19 contracts, all files of one task
folder), runnable on a free Colab GPU, with advanced RAG techniques, an agentic module, evaluation and observability.

```text
DOCX contracts -> rules-based parser (legal sections) -> token-aware child chunks (480 tokens, exact parent rebuild)
   -> BGE embeddings + FAISS, BM25 -> RRF fusion -> BGE cross-encoder rerank          (local, GPU on Colab)
   -> Claude agent (LangGraph: plan -> investigate -> answer)                         (Claude API)
        plan: issues and queries     investigate: search_corpus / lookup_reference tools, bounded
        answer: structured JSON, citations by evidence id, explicit gaps, validated in code
   -> evaluation: ESR/CER benchmark, cost and latency, traces, citation checks, answer review
```

| Part | Where | Notes |
|---|---|---|
| Parser, chunker | `src/legal_rag/parser.py`, `chunking.py` | Tuned to this corpus; a new corpus needs its own parser (README) |
| Retrieval | `retrieval.py`, `embeddings.py` | Dense, BM25, RRF hybrid, rerank; four baselines |
| Deterministic agent (control) | `evidence_agent.py` | Agent v2: fixed facets and rules, no LLM |
| Claude agent | `claude_agent.py`, `claude_schemas.py` | Direct Claude API + LangGraph; typed schemas and validators |
| Benchmark | `eval/benchmark.json`, `evaluation.py`, `scripts/evaluate_retrieval.py` | 12 silver-labeled questions; methods incl. `claude:<model>` |
| Citation checks | `citation_check.py`, `scripts/check_citations.py` | Scope, numeric, support judge, relevance judge |
| Answer review | `review.py`, `scripts/build_review_packets.py`, `eval/EVALUATION_PROMPT.md` | Blind packets and a rubric |
| Tests | `tests/` (46) | Offline; fake Claude client; no corpus, GPU or network needed |

Design decisions and rejected options (why direct API instead of the Agent SDK; why not Ragas or DeepEval):
[claude-integration.md](claude-integration.md). Corpus choice and parser limits: [README](../README.md).

## 2. Advanced RAG techniques used

1. **Structure-aware chunking** with legal references and document regions kept in metadata; child chunks rebuild their parent exactly.
2. **Hybrid retrieval**: dense + BM25 fused by reciprocal rank fusion.
3. **Cross-encoder reranking** (BGE reranker) over the fused pool.
4. **Cross-reference expansion**: "Section X" references are followed inside the same document (rule-based in Agent v2; a tool the model can call in the Claude agent).
5. **LLM query planning and iterative evidence gathering** with explicit evidence gaps (the agentic module below).
6. **Grounded generation**: answers must cite evidence ids; unknown ids are removed and an answer without a valid citation is downgraded.

## 3. Why an agent, and where

Fixed facet rules failed on mixed questions, payment/notice/exclusivity and negative-right questions (Agent v2: 63.9% ESR@10, worse than plain
retrieval). Repeated local search cannot fix a plan that omits the requested issue, so the model plans the issues, picks queries,
decides whether to follow references or search again, and states what is still missing. Retrieval itself stays local. Guardrails:
only two read-only tools; contract text is treated as untrusted data; caps on turns, tool calls, tokens and cost (per question); the
model never sees labels. Haiku searches (cheap, thorough) and Sonnet writes the final answer (the split model setting).

## 4. Measured results (12 silver-labeled questions; one run per configuration)

Evidence Set Recall (ESR) is the share of required clauses found in the first k cited sections; Complete Evidence Rate (CER) is the share of
questions with all of them (definitions in [evaluation.md](evaluation.md)). Full tables, caveats and per-question misses:
[evaluation-results.md](evaluation-results.md).

| Method | ESR@5 | ESR@10 | CER@10 | Median latency | Cost/question |
|---|---:|---:|---:|---:|---:|
| BM25 | 47.9% | 56.9% | 33.3% | 10 ms | 0 |
| Dense | 79.2% | 83.3% | 66.7% | 15 ms (T4) | 0 |
| Hybrid + Rerank | 72.9% | 85.4% | 66.7% | 1.07 s (T4) | 0 |
| Agent v2 (rules) | 63.9% | 63.9% | 33.3% | 1.15 s (T4) | 0 |
| Claude Haiku only | 86.1% | 100% | 100% | 58 s (CPU) | about $0.008 |
| Claude Sonnet only | 93.8% | 93.8% | 75.0% | 41 s (CPU) | about $0.083 |
| **Haiku search + Sonnet answer** (Colab T4) | **95.8%** | **100%** | **100%** | 36 s | about $0.046 |

- **Repeat run** of the split configuration (CPU, then T4): ESR@10, CER@10 and cost reproduced; CER@5 moved by one question; the share of
  citations from files without labels for the question went from 13% to 3%.
- **GPU effect** (T4 vs CPU): baseline metrics identical; reranking about 6.5x faster; local time per Claude question 25.5 s to 3.6 s; the
  remaining latency is Claude API time.
- **Citations** (81 claims, Haiku judge): 68% supported, 31% partial (mostly compound claims that go beyond the passage), 1% not supported.
- **Independent answer review** (fresh LLM session): 7 correct, 4 mostly correct, 1 incomplete, 0 wrong; mean 92.5% of the rubric maximum.
- **API spend** for the whole project, by token-based upper estimates: about $2.4.

## 5. Observability

Per query: JSON traces with tool calls and arguments, result ids, one-sentence decision notes, per-call model, stop reason, tokens and latency;
run metadata with corpus, benchmark and source hashes, config hash, hardware and package versions; per-model usage and estimated cost;
validation problems; failures counted in metric denominators. No hidden reasoning is logged and keys never enter traces. There is no hosted tracing
service (for example Phoenix or Langfuse); the traces are local files.

## 6. Known weaknesses and open items

- **Scale and independence of evaluation.** 12 questions, silver labels written by us, one run per configuration, and the benchmark was visible
  while the system was developed (one validator fix was made after looking at results). No held-out questions. Do not read differences of one to
  three questions as real. Results are for this corpus only.
- **No independent legal review.** Answers were reviewed by LLMs (the assistant that helped build the system, then a fresh session), not a lawyer. The
  review rubric relies on short notes; claims beyond the notes could not be verified. Answers are not legal advice.
- **Citation judge** is from the same model family as the answerer; its verdicts were only spot-checked.
- **`conclusion_type` is not reliable** (almost always `conditional_right`); do not use it as a label.
- **Run-to-run variation** in completeness and in which files are cited (see section 4); one answer omitted a point the earlier run included.
- **Internal note leak**: the investigator's note appeared in 2 of 36 answers; the prompt now forbids it, but the recorded results were not re-run.
- **Mixed contracts**: the agent can cite a sibling contract (rare in the T4 run, 3 of 115 citations).
- **Reference expansion** reads indexed child chunks, not full parent sections (not implemented).
- **Parser is corpus-specific.**
- **Unverified paths**: `--refusal-fallback` and Opus models were not run; Haiku 5.5 and Sonnet 5.5 were. Prices in the cost estimate come from a hard-coded table
  without cache pricing.
- **Colab packaging**: [notebooks/COLAB.md](../notebooks/COLAB.md) holds the cells, but there is no committed `.ipynb`. The T4 indexing step was run earlier
  without recorded timings.
- **Not tried**: more questions, repeat runs for variance, other search/answer model pairs, a chunk-level relevance judge for all retrieved passages
  (only cited off-label passages are judged).

## 7. Reproduce

Local and Colab commands: [README](../README.md), [README_LOCAL.md](../README_LOCAL.md), [notebooks/COLAB.md](../notebooks/COLAB.md). Key steps:

```bash
pip install -e '.[dev,claude]' && pytest -q && python scripts/evaluate_retrieval.py --validate-only
python scripts/evaluate_retrieval.py --device cuda --claude-models claude-haiku-5-5+claude-sonnet-5-5 --claude-max-cost-usd 0.15 --save-dir <out>
python scripts/check_citations.py <out>/results.json
python scripts/build_review_packets.py --results <out>/results.json   # then grade with eval/EVALUATION_PROMPT.md
```

Result folders: `eval/results/cpu-v1.json` (historical, unchanged), `claude-v1` (separate Haiku and Sonnet, CPU), `claude-v2-split` (split models, CPU),
`claude-v3-t4` (split models, Colab T4, with citation checks and the independent review). Never put `eval/` or any answer key in the retrieval corpus.

## 8. Self-assessment (author's view, 0 to 25 per criterion)

This is my honest estimate from the evidence above, not a grade; a grader weighing things differently may score higher or lower.

| Criterion | Score | Why | What holds it back |
|---|---:|---|---|
| RAG, runnable on Colab GPU | **19** | Full pipeline from DOCX to cited LLM answers; GPU used for embeddings and reranking; a full run on a free Colab T4 is recorded with timings and a 6.5x rerank speedup; reuses the index; per-question cost about 5 cents | Colab is instructions, not a committed notebook; GPU indexing time not recorded; the setup needs manual steps (secrets, index copy); evaluation is on one small corpus |
| Advanced RAG techniques | **23** | Structure-aware chunks with exact parent rebuild and token budget, hybrid RRF, cross-encoder rerank, cross-reference expansion, LLM planning, grounded and validated generation; every choice documented and compared with controls | Parser is corpus-specific; no full parent-section restoration; no query rewriting or HyDE explored |
| Agentic module | **21** | LangGraph with a reasoned placement (where fixed rules failed), bounded tool loop, typed schemas, budgets, split models, measured gain over the rule-based agent and baselines, a documented control | Graph is linear (the loop is manual inside one node); no variance estimate; some run-to-run inconsistency; `conclusion_type` weak |
| Evaluation and observability | **21** | Labeled benchmark with leakage controls, five baselines plus agents at equal cutoffs, latency, tokens, cost, traces and hashes, citation checks, blind packets with a rubric, independent review, honest caveats | Silver labels, 12 questions, single runs, no held-out set, LLM-only review, no hosted tracing |
| **Total** | **84 / 100** | | |

Biggest remaining gains per effort: a committed Colab notebook; a held-out question set with repeated runs; a lawyer's review of a sample.
