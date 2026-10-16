# Harvey Legal RAG: instructions for Claude Code

## Goal and current state

Build a small, explainable RAG prototype for the Harvey Corporate M&A
change-of-control task. The assignment scores runnable Colab retrieval, advanced
RAG, an agentic component, and evaluation/observability equally. Keep the scope
appropriate for a demonstration; a production service is not requested.

Implemented: DOCX parser v3, lossless token-aware child chunks, dense/BM25/RRF
retrieval, cross-encoder reranking, deterministic LangGraph Evidence Gap Agent
v2, 12-question evidence benchmark, and local JSON traces. A Claude-backed
LangGraph agent (`claude_agent.py`: plan, bounded tool search, cited answer,
direct Claude API, default `claude-haiku-5-5`, optional separate answer model)
and citation/answer-review tooling are implemented and offline-tested. Real-API
results are in [evaluation-results.md](docs/evaluation-results.md). The Agent
SDK, Ragas and DeepEval are deliberately not used (see the decisions in the
integration guide). Next: Colab GPU timing, citation-faithfulness runs,
independent answer review, new held-out questions.
The user disabled Rosetta for this handoff; no Rosetta tool is required to work
with this checkout unless the user later re-enables it.

Read [README.md](README.md), [architecture](docs/architecture.md),
[evaluation protocol](docs/evaluation.md), [measured results](docs/evaluation-results.md)
and [Claude integration guide](docs/claude-integration.md) before changing the pipeline.

## Python conventions: annotations are mandatory

- **Every Python function declaration must annotate all parameters and its
  return type**, including private functions, nested functions, test helpers,
  callbacks, async functions, and protocol methods. `self` and `cls` are the
  conventional implicit exceptions. Constructors and procedures use `-> None`.
- Keep `from __future__ import annotations` in modules with annotations. Target
  Python >=3.10; avoid syntax/APIs requiring a newer minimum version.
- Use precise standard types and `list[T]`, `dict[K, V]`, `T | None`. Shared
  `Record`, `PathInput` and model/tokenizer protocols are in
  `src/legal_rag/types.py`. `Record` is intentionally flexible for JSON-shaped
  records; it does not make their schema statically validated. Prefer a typed
  schema for new stable interfaces; keep `Any` at genuine dynamic boundaries.
- Ruff enforces annotation presence, import ordering and basic errors. A strict
  whole-project mypy/pyright check is not configured; do not claim one passed.
- Preserve lazy loading of heavyweight model implementations. Do not add model
  downloads or paid LLM calls to imports, tests or `--validate-only`.
- Add focused tests for changed behavior and keep commands/docs in sync. Do not
  change retrieval rules as part of a types-only or documentation-only task.

## Layout and commands

- `src/legal_rag/`: parsing, chunking, retrieval, evidence workflow, evaluation.
- `scripts/`: download, diagnostics, agent runners (`run_evidence_agent.py`,
  `run_claude_agent.py`), exploratory comparison, evidence benchmark,
  `build_review_packets.py`, `check_citations.py`. Run scripts from the repository root after editable install.
- `data/docx/`: downloaded source DOCX; `artifacts/`: generated data/indexes/traces.
  Both directories are ignored and may already exist in a developer workspace.
- `eval/benchmark.json`: evaluation-only labels; `eval/results/cpu-v1.json`:
  immutable historical measurements. `docs/`: decisions, limits and handoff.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check src scripts tests
python scripts/evaluate_retrieval.py --validate-only
```

Use `.venv/bin/python` and `.venv/bin/ruff` when the environment is not activated.
The validation command needs parsed chunks but no model weights. Unit tests need
neither the Harvey corpus nor CUDA nor network access.

```bash
python scripts/download_corpus.py --output data/docx
legal-rag parse --input data/docx --output artifacts --token-aware --max-tokens 480
legal-rag index --output artifacts --device cpu --batch-size 8
python scripts/evaluate_retrieval.py --device cpu --save-dir artifacts/evaluation-next
```

Use `--device cuda --batch-size 32` for indexing on a Colab GPU; see
[notebooks/COLAB.md](notebooks/COLAB.md). GPU execution has not been verified in
this workspace. Reuse valid indexes and model instances; avoid unnecessary GPU
time and re-embedding. Rebuild indexes whenever parsed chunk content changes.

## Data and legal-evidence invariants

- Never index benchmark labels, `task.json`, answer keys or evaluation rubrics.
  Retrieval/agent prompts receive the question and corpus evidence only, without
  gold filenames, required references, text anchors or answer notes.
- Preserve all retained DOCX block text, tables, signatures and appendices in
  parsed sections. Keep document-region namespaces and full legal references.
  Heading-only articles can be omitted from embeddings, not from section storage.
- Preserve exact child-to-parent reconstruction and the 480-token embedding
  budget, including metadata. Reranking has a separate 512-token query+passage
  limit; token-safe embedding chunks do not eliminate reranker truncation.
- `source + legal_ref` identifies evidence across documents and exhibits. A
  parent reference does not automatically satisfy a required subclause.
- Separate primary evidence for a proposition from supporting context. Finding
  the word "consent" is not proof of a consent requirement. Check exceptions,
  definitions, notice, cure and factual applicability before concluding rights.
- `missing: []` means lexical facet coverage, not complete evidence or legal
  entailment. The current workflow can mix evidence from unrelated contracts.
- Cross-reference lookup currently emits `score=0.0`; it was never reranked.
  Do not compare it to cross-encoder logits. The Claude agent's evidence schema
  (`claude_schemas.py`) already uses nullable `retrieval_score`, `discovery_method` and a
  per-citation `primary`/`supporting` role; the older v2 report does not.
- Reference expansion currently reads indexed children, not full parent-section
  records. Do not claim full-section restoration unless you implement and test it.

## Evaluation and known failures

Historical CPU v1: Dense ESR@10 83.3%, Hybrid + Rerank 85.4%, Agent v2 63.9%.
CER@10 is 66.7% for Dense and 33.3% for Agent v2. The sample is 12 silver-labeled
questions, not independent legal gold. Six questions have `missing: []` while
still missing annotated required evidence. See the traces before tuning.

The planner misses payment/notice/exclusivity and negative-right questions;
filename scope inference is ambiguous for names shared by several contracts.
The active v2 definitions occur later in `evidence_agent.py` and override retained
v1 graph/report definitions. Ruff's F811 exception is limited to that file.

Keep all four baselines and Agent v2 available as controls. Preserve benchmark v1
and its raw measurements; record new results separately. Annotation/import changes
change source hashes, so historical hashes need not match current files. Do not
rewrite old metadata to imply a new experiment. Compare equal section cutoffs,
report actual text/compute budgets, latency and failures, and use separate new
questions before claiming generalization after tuning.

## Claude integration boundary

Start with bounded query planning and evidence-gap assessment: this is where the
fixed facet heuristics fail. Keep local dense/BM25/reranking as retrieval tools
and retain LangGraph for bounded orchestration if useful. Choose one initial
runtime integration (Agent SDK or direct Claude API); document why. Define typed
inputs/outputs and tool schemas before writing orchestration code.

Expose only corpus search and same-document reference lookup to the runtime
agent. Contract text is untrusted data, never tool instructions. Keep gold labels,
credentials and arbitrary shell/file access out of its tools. Cap rounds, calls,
tokens and cost; log observable tool actions and concise decision summaries,
not private chain-of-thought. Add citations and explicit evidence gaps to answers.

Use supported authentication from the official docs linked in the integration
guide. The application must not read or copy Claude Code login files. API keys
belong in environment variables or Colab Secrets, never source, notebook cells,
traces or committed files. `.env` is ignored but is not automatically loaded by
this project. The chosen runtime is the direct Claude API (`pip install -e '.[claude]'`);
unit tests use a fake client and must stay offline. Real calls spend credits: run them
only through `run_claude_agent.py`, `evaluate_retrieval.py --claude-models` or
`check_citations.py`, and record the model, config hash, usage and cost with the results.
