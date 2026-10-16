# Harvey Legal RAG

Reproducible, structure-aware retrieval for the Harvey Corporate M&A change-of-control task.

## Current status

Research/demo prototype for the Harvey Corporate M&A change-of-control task:
structure-aware chunking, hybrid retrieval with reranking, a **Claude-backed
LangGraph agent** that plans, searches and writes cited answers, and an evidence
benchmark with cost and trace logging. Local BGE models do embeddings and
reranking (GPU on Colab); Claude (API key, `claude-haiku-5-5` by default) does
the reasoning steps. The deterministic Agent v2 and four plain retrievers remain
as controls. Results: [measured results](docs/evaluation-results.md).

## Design choices at a glance

| Area | Choice | Why |
|---|---|---|
| Chunking | DOCX parsed into legal sections, token-aware child chunks (480 tokens incl. metadata) that rebuild the parent exactly | Contracts are cited by section; nothing may be silently truncated by the 512-token BGE limit |
| Retrieval | Dense (BGE + FAISS) + BM25, fused with RRF, cross-encoder rerank | Legal text needs both meaning and exact terms ("Direct Competitor", "§9.2") |
| Cross-references | Deterministic and agent-requested lookup of "Section X" in the same document | A rule is often modified by a referenced exception |
| GPU use | Embeddings and reranker on CUDA, indexed once and reused; FAISS on CPU | Scarce free-tier GPU time goes to the only heavy local models; Claude runs remotely |
| Agentic step | Claude plans the issues, then runs a bounded search loop with two tools (`search_corpus`, `lookup_reference`) | Fixed keyword facets missed payment, notice, exclusivity and "no right" questions; a model can name the missing issue |
| Runtime | Direct Claude API + LangGraph, not the Agent SDK | Two custom tools, explicit budgets, works in Colab with an API key; see [integration guide](docs/claude-integration.md) |
| Answers | Structured JSON, citations by evidence id with `primary`/`supporting` role, explicit gaps; invalid citations are stripped | Grounding is checked in code, not trusted |
| Safety | Contract text treated as untrusted data; only two read-only tools; caps on turns, tool calls, tokens and cost | Prompt-injection and runaway-cost containment |
| Evaluation | 12 silver-labeled questions, Evidence Set Recall / Complete Evidence Rate at 5 and 10 sections, plus latency, tokens, cost and JSON traces | Compares all methods at equal section cutoffs without leaking labels |

Details and limits: [architecture](docs/architecture.md), [evaluation protocol](docs/evaluation.md). Why we do not use the Claude Agent SDK, Ragas or DeepEval: [decisions](docs/claude-integration.md#decisions-what-we-deliberately-do-not-use).

## Claude agent quick start

```bash
pip install -e '.[dev,claude]'
export ANTHROPIC_API_KEY=...        # Colab: use Secrets, see notebooks/COLAB.md
python scripts/run_claude_agent.py --dry-run --benchmark eval/benchmark.json
python scripts/run_claude_agent.py "Does TerraNode have termination rights if Aldersgate is acquired by a competitor?"
```

This spends API credits (about a cent per question on Haiku). Each run is saved with its
model and a timestamp under `artifacts/claude-runs/`. Compare all methods:

```bash
python scripts/evaluate_retrieval.py --device cuda --claude-models claude-haiku-5-5 claude-sonnet-5-5 \
  --save-dir eval/results/claude-v1
```

Use `SEARCH+ANSWER` (for example `claude-haiku-5-5+claude-sonnet-5-5`) to search with one
model and write the answer with another.

### Ask your own question

With the index built (`artifacts/`) and `ANTHROPIC_API_KEY` set, ask any English question about the indexed contracts:

```bash
python scripts/run_claude_agent.py "Does TerraNode have termination rights if Aldersgate is acquired by a competitor?" \
  --model claude-haiku-5-5 --answer-model claude-sonnet-5-5 --device cuda     # --device cpu without a GPU
```

It prints the conclusion type, the answer with its conditions, citations (file, section, title), evidence gaps,
tool calls, tokens and estimated cost (about 4-5 cents and a minute per question for Haiku + Sonnet). The full result
with retrieved passages and the trace is saved to `artifacts/claude-runs/<model>-<time>/adhoc.json`. Useful flags:
`--no-rerank`, `--max-cost-usd`, `--max-tool-calls`, `--max-turns`. In Colab, set the key in a notebook cell first
([instructions](notebooks/COLAB.md)).

Without Claude (free, returns passages, not an answer): `legal-rag search "<question>" --output artifacts --method hybrid_rerank`
or `python scripts/run_evidence_agent.py "<question>" --output artifacts`.

Limits: the corpus is fixed to the 19 indexed contracts; a question about another document gets
`insufficient_evidence` or unrelated citations. The agent chooses the contract from the question text
(`run_claude_agent.py` has no `--source`). Answers are model-generated, not legal advice: check the cited clauses.

Start with [local setup](README_LOCAL.md) or the [Colab runner](notebooks/COLAB.md).
Claude Code contributors should read [CLAUDE.md](CLAUDE.md). Python requires
version 3.10 or newer; installation and model downloads require network access.

## Project layout

```text
src/legal_rag/     Library: DOCX parser, chunker, embeddings, retrieval, CLI
scripts/           Corpus download, diagnostics, agent and evaluation runners
tests/             Synthetic regression tests (no GPU required)
docs/              Architecture, evaluation protocol and measured results
eval/              Evaluation-only evidence labels (never index)
notebooks/         Colab entry point instructions
CLAUDE.md          Contributor rules and Claude Code handoff
```

## Google Colab / Tesla T4

```bash
!git clone <YOUR_REPOSITORY_URL>
%cd harvey-legal-rag
%pip install -e '.[dev]'
```

If using this downloadable ZIP instead of GitHub, upload/extract it to `/content/harvey-legal-rag`, then `%cd` there. In Colab's Runtime settings, enable T4 GPU.

### Corpus

The selected dataset is the Harvey [Corporate M&A change-of-control task](https://github.com/harveyai/harvey-labs/tree/main/tasks/corporate-ma/analyze-change-of-control-provisions-across-targets-material-contracts).
Put its 19 DOCX files in `data/docx/`. If you already have them in `/content/harvey_coc/`, you can use that path directly. Optional downloader (requires network):

```bash
python scripts/download_corpus.py --output data/docx
```

**Do not include `task.json`, evaluation rubrics or answer keys in the retrieval corpus.** The downloader fetches only `.docx` files.

### Why this corpus

The assignment asks for a subset of the Harvey legal dataset. We use one task folder, Corporate M&A: Change of
Control, and take every DOCX file in it (19 documents, 1,380 chunks); no file was hand-picked.

The author chose it because it is well suited to showing the difference between plain semantic search and an
advanced RAG pipeline. Per the author, the task's own evaluation criteria (not part of this repository and never
indexed) turn on provisions such as these, which also appear in the contracts:

- **TerraNode:** limits on assignment to a competitor and a consent requirement, with a competitor definition,
  notice and cure (benchmark: `terranode_competitor`, `terranode_notice_affiliate`).
- **Pinnacle:** the exclusive license converts automatically to non-exclusive on a change of control of the licensee
  (benchmark: `pinnacle_licensee`, `pinnacle_licensor`).
- **First Continental Bank:** change of control accelerates and prepays the debt, with a make-whole premium
  (benchmark: `continental_acceleration`, `continental_make_whole`).
- **Apex:** the customer's change-of-control termination right and the provider's Transition Services obligation
  after termination (§12.3). The benchmark question on Apex covers the termination right and its periods;
  **the transition-services obligation is in the corpus but not labeled or tested.**

The answers sit in several clauses that must be combined (a consent rule, its exception, a definition, a cure
period, a cross-reference), and similar terms mean different things in different contracts, so a single nearest
passage often misses the rule. The set is also small enough for a free Colab GPU.

Coverage note: all 19 files are indexed and searched, but benchmark labels cover only 8 of them. The other 11
(including the memos and a 10-K excerpt) have no required-evidence labels: they act as distractors for the metrics
(29% of the top-10 sections of Hybrid + Rerank come from them, 3-8% for the Claude agents), though the deal memos also
carry transaction facts that answers can legitimately use. Results on this one task do not show how the pipeline generalizes
to other contract types or corpora (see the parser note below).

### Build

```bash
legal-rag parse --input /content/harvey_coc --output artifacts --token-aware --max-tokens 480
legal-rag index --output artifacts --device cuda --batch-size 32
python scripts/check_boundaries.py --output artifacts
pytest -q
```

If the documents are stored under `data/docx`, substitute `--input data/docx`.

### Query and compare

```bash
legal-rag search "Which agreements require consent to assignment or change of control?" --output artifacts --method dense --device cuda
legal-rag search "Which agreements require consent to assignment or change of control?" --output artifacts --method bm25 --device cuda
legal-rag search "Which agreements require consent to assignment or change of control?" --output artifacts --method hybrid --device cuda
legal-rag search "Which agreements require consent to assignment or change of control?" --output artifacts --method hybrid_rerank --device cuda
python scripts/compare_methods.py --output artifacts --device cuda
legal-rag diagnose "Does TerraNode have termination rights if Aldersgate is acquired by a competitor?" --source terranode-isa.docx --section 9.2 --output artifacts --device cuda
```

Run the deterministic evidence-gap workflow (see [Agent v2](README_AGENT_V2.md)):

```bash
python scripts/run_evidence_agent.py \
  'Does TerraNode have termination rights if Aldersgate is acquired by a competitor?' \
  --source terranode-isa.docx --output artifacts --device cuda \
  --save artifacts/terranode_evidence_v2.json
```

Omit `--source` for question-based source inference. The agent's `--no-rerank`
option avoids loading the cross-encoder. `candidate_evidence_found` and
`missing: []` indicate lexical coverage, not a verified legal conclusion.

### Outputs

`artifacts/legal_sections.jsonl`, `artifacts/legal_chunks_v2.jsonl`, `artifacts/legal_embeddings.npy`, `artifacts/legal_dense.faiss`, `artifacts/dense_index_config.json`, `artifacts/bm25_index_config.json`, and optionally `artifacts/comparison.json`.

Artifacts are ignored by Git. Save them to Drive if you need persistence between Colab sessions. Hugging Face public models can be downloaded without `HF_TOKEN`, though rate limits may apply.

### Evidence benchmark and local traces

```bash
python scripts/evaluate_retrieval.py --validate-only
python scripts/evaluate_retrieval.py --device cuda
```

The 12-question benchmark compares Dense, BM25, Hybrid, Hybrid + Rerank and Agentic Retrieval v2 using Evidence Set Recall and Complete Evidence Rate at 5 and 10 cited sections. Use `--device cpu` locally. Results, missing evidence and agent traces are saved under `artifacts/evaluation/`. See [evaluation protocol](docs/evaluation.md) and [measured results](docs/evaluation-results.md). Labels stay outside the retrieval corpus.

## Known limitations

This is a reconstructed, modular version of the notebook code, not a byte-for-byte copy of earlier Colab cells. **Chunk counts and ranks may change** after parser repair. Enable `--token-aware` to avoid embedding truncation; query/passage truncation in the 512-token reranker remains a limitation. LangGraph evidence retrieval, the Claude agent, an evidence benchmark and local JSON traces are implemented. Generated-answer quality is only reviewed by hand (no automatic faithfulness metric), hosted tracing is not implemented, and the Claude results are single runs on 12 silver-labeled questions. A full benchmark run on a Colab Tesla T4 is recorded in `eval/results/claude-v3-t4/`; indexing on GPU was not timed.


## Contributor checks

```bash
python -m pytest -q
ruff check src scripts tests
python scripts/evaluate_retrieval.py --validate-only
```

All function parameters and return values must have type annotations, including
tests and nested functions (`self`/`cls` are implicit). Ruff enforces their
presence. Shared JSON aliases and retrieval protocols are in
`src/legal_rag/types.py`; no strict whole-project static type checker is configured.
Unit tests need no corpus, network or GPU; benchmark validation needs parsed chunks.

## Parser and artifact conventions

**The parser is tailored to this corpus.** `src/legal_rag/parser.py` is a rules-based parser (regular expressions,
bold-run ratios, signature and exhibit markers, special handling of the 19 Harvey contracts' headings, memos and
spurious "by and between" lines), tuned and tested on exactly these DOCX files, where most paragraphs use the `Normal`
style instead of Word headings. It is not a general contract parser. Contracts that number sections, articles or
exhibits differently, use real heading styles, or are PDFs or scans (only `.docx` is read) may be split wrongly, and
wrong section boundaries and legal references degrade retrieval and citations silently. For a new corpus, write or adapt
a parser for its conventions, then check boundaries (`scripts/check_boundaries.py`), add labeled questions and
re-run `parse`, `index` and the benchmark before trusting results.

Use a single project layout: `data/docx/` for source DOCX files and `artifacts/` for derived data.

`--token-aware` is required to prevent BGE's 512-token truncation; the 480-token budget includes the document and section context. The parser preserves `document_region`, `parent_path` and `legal_ref` for each section and chunk. Child chunks concatenate exactly to their full parent section.

The CLI retains the filename `legal_chunks_v2.jsonl` for compatibility; the metadata format has been extended. Re-run `index` after every parse. DOCX and index files are excluded from source control; a local workspace may already contain downloaded/generated copies.

Full parent sections are stored, but the current agent's reference lookup returns
indexed child chunks. Automatic restoration of complete parent evidence is not
implemented. See [architecture](docs/architecture.md) for design choices and limits.

## Documentation map

- [Local setup and troubleshooting](README_LOCAL.md).
- [Agent v2 behavior and limits](README_AGENT_V2.md).
- [Architecture and design decisions](docs/architecture.md).
- [Evaluation protocol](docs/evaluation.md) and [measured CPU results](docs/evaluation-results.md).
- [Colab instructions](notebooks/COLAB.md).
- [Claude integration guide](docs/claude-integration.md) (implementation, auth, why not Agent SDK / Ragas / DeepEval) and [contributor instructions](CLAUDE.md).
- [Project report](docs/final-report.md): what was built, measured results, weaknesses, self-assessment.
- [Answer review rubric](eval/EVALUATION_PROMPT.md); `scripts/build_review_packets.py` and `scripts/check_citations.py` (see the evaluation protocol).
