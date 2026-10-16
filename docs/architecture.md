# Architecture and design decisions

## Corpus and boundaries

The selected Harvey Corporate M&A task has 19 DOCX documents. Most paragraphs are styled `Normal`; therefore, relying on Word Heading styles or XML numbering alone fails. `parser.py` uses textual headings, run boldness, and block-level recognition. It retains tables as row blocks and retains appendices as evidence. It distinguishes standalone `EXHIBIT A` from a sentence starting `Schedule 4.17 sets forth...`.

The parser is rules-based and tuned to this corpus (see the README note: a new corpus needs its own parser). It creates **full sections** and **retrieval child chunks**, linked by
`parent_section_uid`. `legal_sections.jsonl` stores parent text for inspection and
future full-parent expansion; `legal_chunks_v2.jsonl` is the retrieval unit.
Current agent reference expansion looks up matching child chunks, rather than
loading these parent records. Reconstruction assertions ensure no retained block
disappears when segmenting. A heading-only ARTICLE is not embedded but remains
in section storage. Block extraction normalizes whitespace, so losslessness is
relative to retained normalized block text, not the original DOCX bytes.

## Retrieval

- Dense: `BAAI/bge-base-en-v1.5`, 768-dimensional normalized vectors, FAISS exact inner product.
- Sparse: `rank-bm25` over tokenized contextual passages.
- Hybrid: Reciprocal Rank Fusion, k=60, over top 50 from each retriever.
- Reranked: `BAAI/bge-reranker-base` scores the top 40 fused **child** chunks; parent deduplication is performed **after** reranking.
- Default reranker max input length: 512 tokens, including query and passage. Truncation remains a known limitation.

`dense_index_config.json` includes a SHA-256 hash of the ordered corpus. Search refuses stale FAISS indexes after re-parsing.

## Why these choices fit the prototype

DOCX layout and numbered provisions matter for legal citations. Structure-aware
chunking limits boundary contamination and keeps citations tied to parent
sections. Contextual metadata (document, reference and parent titles) is included
in each embedded passage. The 480-token budget covers that metadata as well as
the body; the tokenizer/model limit is 512.

Dense retrieval handles paraphrases; BM25 preserves exact party names and legal
references. RRF combines their rank positions without assuming that their scores
share a scale. A local cross-encoder evaluates query/passage relevance on a small
candidate pool. These are the implemented advanced retrieval techniques; the
benchmark shows where they help and where they do not.

The FAISS index uses CPU exact inner-product search: for this small corpus,
keeping index search on CPU avoids GPU-index setup and leaves Colab GPU memory
for neural inference. BGE embeddings are normalized, so inner product acts as
cosine similarity. Models load once per runner and the existing index can be
reused. No quantization or approximate-vector-search service is needed for this
demonstration. The published measurements are CPU-only; free-Colab GPU throughput
and session availability are not established by them.

The original agent applies bounded follow-up retrieval to evidence gaps because
multi-clause legal questions often need a chain of provisions. LangGraph makes
the plan/search/expand/assess loop explicit and limits repeated searches. Its
v2 decision rules are deterministic; the measured failures motivated the
Claude-backed planning, assessment and cited answers (`claude_agent.py`) described in
[the integration guide](claude-integration.md). That agent keeps the local retrieval tools
and uses LangGraph (`plan -> investigate -> answer`) with a bounded tool loop inside
`investigate`; v2 remains as a control.

## Observed baseline behavior (prior to parser repair)

On a question about TerraNode competitor acquisition and termination, §9.2 ranked 5th in dense, 12th in BM25, 6th in RRF and 10th after reranking. The clause concerns **consent**; the actual termination chain involves §§9.1, 9.2 and 5.5. The section was 296 reranker tokens, so truncation was not the cause. §13.9 incorrectly contained signatures and Exhibit A, contaminating retrieval. This repository fixes those structural boundaries; re-run experiments rather than assuming old metrics persist.

## Agent and evaluation

The local LangGraph evidence-gap agent v2 performs role checks, explicit reference
expansion and targeted retrieval. It can infer a document from a distinctive name
in the question. Its deterministic checks identify candidate evidence and do not
prove a legal outcome. The v1 graph definitions remain in the module but the v2
definitions are the active implementation.

The evidence benchmark compares the four baselines, the existing v2 agent and, optionally,
Claude agents (`--claude-models`) without changing retrieval heuristics. Required evidence groups, equivalent
clauses and optional supporting evidence live in evaluation-only JSON. Per-query
and per-node graph traces, hashes, latency and failures are written locally. See
`evaluation.md` for the metric definitions and comparison limits. Hosted tracing
(Phoenix/OpenTelemetry) and an independent, automatic answer-quality evaluation remain future
work. Claude answers are graded by a manual rubric ([EVALUATION_PROMPT.md](../eval/EVALUATION_PROMPT.md))
and a citation-support check (`scripts/check_citations.py`); see `evaluation.md`.


### Parser v3 normalization

- Document region: `Main Agreement`, `Signature Page`, `Exhibit A`, `Schedule 4.17`, etc.
- The exhibit/schedule identifier is carried into descendant section `parent_path` and `legal_ref`.
- Very short bold conjunctions and signature-field labels do not become headings.
- Retrieval child chunks use a strict 480-token budget with BGE tokenizer including metadata.
- Text is preserved exactly within parent sections. The parent-section ID is stable for the same parsed block range.
- Token-aware mode uses token-based splitting and may split a single long
  paragraph at whitespace boundaries. Children concatenate exactly with `""`
  in token-aware mode; the legacy word splitter reconstructs with `"\n"`.
  Full parent records are available for an explicit future restoration step.

## Types and implementation boundaries

All function declarations in the library, scripts and tests have argument and
return annotations. `types.py` supplies shared JSON/path aliases and structural
protocols for the encoder, tokenizer, FAISS index and reranker. JSON-shaped
records remain `dict[str, Any]` at dynamic boundaries; these annotations do not
constitute JSON-schema validation or a strict static type-checking guarantee.
Ruff enforces annotation presence and basic lint rules. Python >=3.10 is the
supported syntax baseline. Model implementations remain lazily imported by
loaders, and tests/benchmark label validation do not download models.

## Runtime limits and remaining work

The agent currently keeps only evidence passing lexical role checks. A relevant
retrieved passage can therefore be discarded if its role is missing from the
plan. Reference expansion follows only the supported single numeric references
from current evidence and needs a resolved document; it is not a general legal
cross-reference parser. Filename inference can be ambiguous when a party has
several agreements. Parent deduplication can retain only one child of a long
section, and reranker input truncation can hide qualifications.

The score fields also have distinct meanings: dense inner product, BM25, RRF
and cross-encoder logits are not directly comparable. Reference lookup's zero
is a placeholder, not measured relevance. Preserve provenance and use a nullable
score plus an explicit evidence role in the next integration.

There is no external tracing service, and no independent or fully automatic
generated-answer evaluation. The Claude agent (default `claude-haiku-5-5`, optional separate
answer model) is implemented with token, cost and tool-call caps. Local trace files record
observable workflow events, model usage and per-query timing; they are sufficient to reproduce
the diagnostic results. See [evaluation.md](evaluation.md) for leakage
controls, budgets and known comparison limits.
