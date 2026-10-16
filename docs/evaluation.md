# Evidence retrieval evaluation

The runnable benchmark is `eval/benchmark.json`; run it with
`scripts/evaluate_retrieval.py`. It compares Dense, BM25, Hybrid,
Hybrid + Rerank and the existing Agentic Retrieval v2. Retrieval rules are
unchanged: measurement comes before further heuristic tuning.

## Labels and isolation

There are 12 questions across eight source contracts, including a two-document
comparison, exceptions, negative termination rights, consent, remedies, notice,
fees and acceleration. Three previously explored questions are marked `dev`;
nine new probes are marked `eval`. These are small **silver** labels authored
from the local corpus, not independent expert gold or an independently held-out
test set. Freeze v1 before tuning and add separate questions for future testing.

Each required evidence group has a role and one or more equivalent acceptable
clauses (`any_of`). A clause label contains the exact filename, full legal
reference and a short text anchor. Optional `supporting_evidence` never satisfies
a required group. For example, TerraNode's consent requirement is §9.1; a
reference to consent in §5.5 does not satisfy it. §§5.5 and 9.3 are alternatives
for the notice-and-cure remedy, avoiding a penalty for omitting a duplicate rule.
The competitor definition in Article 1 is separately required for the TerraNode
applicability question. The benchmark records answer notes for human review;
it does not generate or grade answers.

Labels were checked against the 1,380 indexed child chunks and all 44 label
anchors were also checked directly against the eight source DOCX files. Every
alternative must exist in the indexed corpus before an experiment can start.
`--validate-only` repeats the indexed-corpus validation without loading models.
The corpus fingerprint identifies this local snapshot; the downloader's upstream
URL follows `main` and is not a pinned dataset release.

The retrieval boundary accepts only `(retriever, question, method, budgets)`.
It receives neither gold source filenames, required references, text anchors
nor answer notes. `eval/` stays outside `data/docx/`; the parser accepts DOCX
files only. Rubrics and `task.json` remain evaluation-only.

## Metrics

For question q, let G(q) be the required evidence groups. A group is covered
when at least one returned chunk has an acceptable **source + exact reference**
and contains its text anchor (case and whitespace insensitive).

`ESR@k(q) = covered required groups in the first k sections / |G(q)|`.

**Evidence Set Recall** is the macro average of ESR across questions: each
question has equal weight. **Complete Evidence Rate** is the fraction of
questions with every required group covered at that cutoff. Retrieval errors
receive zero coverage and remain in both denominators; they are also counted
separately. Repeated chunks and supporting-only clauses earn no extra credit.

References are exact: `7.09` does not satisfy `7.09(b)`, and `Exhibit A / 9.1`
does not satisfy main-agreement `9.1`. An unnumbered article uses a typed fallback
such as `article:1`. Its text anchor prevents an unrelated child of a long
definitions article from satisfying the competitor-definition requirement.

A text anchor is a reproducible retrieval check, not an entailment test or a
guarantee that all qualifications survived chunking. CER means the annotated
evidence set was found, not that a legal conclusion is true. This evaluation
does not measure generated-answer accuracy, factual competitor status or
citation faithfulness. For the five retrieval methods no LLM judge or paid API is involved; the
optional Claude methods and the citation judge (below) do spend API credits.

## Budgets and comparison limits

The primary table uses the same **5 and 10 unique cited sections** for every
method. Baselines retrieve a pool of up to 20 parent-deduplicated sections with
their existing candidate budget of 50; Hybrid + Rerank reranks 40 child chunks.
The agent retains its existing settings: initial top-12 search, up to two rounds,
six candidates per facet and rerank pool 24 per search. Its final evidence pool
is capped at 20 sections for reporting.

Agent output has no shared numeric ranking scale. Its evaluation order is
label-blind round robin across the inferred facets, preserving order within each
facet, followed by section deduplication. Cross-reference `score=0.0` is never
sorted against cross-encoder logits. Only actually returned child text is joined
when the same section appears more than once; there is no gold-guided expansion
or automatic restoration of full parent text. The runtime agent's original
score/report schema and role heuristics are unchanged in this experiment.

Section cutoffs do not equalize compute or tokens. The agent can make multiple
searches and infer a document scope from the question; baselines search the
whole corpus. No gold filename is passed to the agent, including on the
multi-document question. Thus this compares complete existing retrieval methods,
including their scope inference, rather than isolating the causal effect of
graph orchestration. `mean_returned_words`, `mean_returned_sections`, per-query
latency, agent rounds and trace events make these differences inspectable. A
future ablation can share scope selection or impose a token/compute budget.

`pool_metrics` reports coverage at each method's actual pool size (at most 20).
Treat this as a diagnostic, not an equal-budget headline result. Pool size and
returned words are recorded alongside coverage. There is no artificial padding
when an agent returns fewer than k sections.

## Run locally or in Colab

After parsing with `--token-aware` and building indexes as in the main README:

```bash
python scripts/evaluate_retrieval.py --validate-only
python scripts/evaluate_retrieval.py --device cuda --threads 4
```

Use `--device cpu` locally. Models load once, with a shared encoder/reranker
warmup before timing. This is a single pass; p50/p95 describe variation across
questions, not repeated-run confidence intervals. Load/download time is excluded;
agent graph construction is included. GPU timings require a separate Colab run.
If weights are cached, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` prevents network
access. `--methods bm25` selects a subset, though the shared existing retriever
still loads its dense encoder/index. `--cutoffs` and `--pool-k` configure budgets.

Outputs under `artifacts/evaluation/`:

- `report.md`: comparison and missing evidence by question/method/cutoff.
- `results.json`: aggregate and per-split metrics, per-query metrics, returned
  clause/chunk IDs, actual pool coverage, errors and full agent traces.
- `queries.jsonl`: one record flushed after each completed query/method, preserving
  diagnostics if a run is interrupted. It is overwritten at the next run; use
  `--save-dir artifacts/evaluation-next` to retain another experiment.

Metadata records corpus, benchmark and implementation SHA-256 hashes, model
configuration, package/Python versions, device, thread count and retrieval
budgets. Full query and agent traces are local files; no external telemetry
service receives document text or credentials. This is lightweight observability
for the prototype, not a hosted tracing deployment. Failures still produce a
report and cause exit status 2.

The checked-in CPU v1 run is historical. Later annotation/import/documentation
changes alter implementation file hashes without changing the measured retrieval
algorithm; do not update the old run's hashes or present it as a new experiment.
Function annotation presence is now enforced by `ruff check src scripts tests`.
Claude agents run as extra methods named `claude:<model>` or `claude:<search>+<answer>`
(`--claude-models`). They receive only the question text. Their evidence order is label-blind (cited
primary, cited supporting, then uncited by discovery); the report adds cited-only ESR/CER, tokens,
tool calls and estimated cost. They use more compute and text than the baselines. See
[the Claude integration guide](claude-integration.md).

See `docs/evaluation-results.md` for the measured local CPU run. Do not infer a
method ranking from the old exploratory probes or from these labels without
reading the misses and the small-sample limitations.


## Citation faithfulness check (Claude agents)

`scripts/check_citations.py` checks, per claim, that the cited passages support it. Runs written by the
current code store each claim with the full text of its cited passages (`cited_claims` in `results.json`);
runs made earlier (`claude-v1`, `claude-v2-split`) do not, so they cannot be checked and must be re-run.

- **Numeric check (free, deterministic):** numbers (digits or words such as "thirty") and section references
  in a claim must appear in its cited passages. A flag is a warning, not a verdict: a paraphrase such as
  "$1 billion" for "$1,000,000,000" is flagged. It catches invented periods, amounts and section numbers.
- **LLM judge (about a cent per ten claims on Haiku):** sees only the claim and its cited passages and returns
  `supported`, `partial` or `not_supported` with one sentence. It uses no outside knowledge, ignores
  instructions in contract text, and is capped by `--max-cost-usd`.

```bash
python scripts/check_citations.py eval/results/<run>/results.json --dry-run     # counts, no API
python scripts/check_citations.py eval/results/<run>/results.json --no-judge    # numeric check only
python scripts/check_citations.py eval/results/<run>/results.json               # + judge, needs ANTHROPIC_API_KEY
```

**Cited sections outside a question's labeled files.** All 19 files are searched, but labels cover 8. Two further checks
look at cited sections from files with no label for that question (often deal memos, sometimes another contract):

- **Scope (free, runs on every Claude run, old ones included):** share of cited sections from files outside the question's
  own labeled files, and from files labeled for no question. Labels are used only to measure.
- **Relevance judge (Haiku, cents, needs `cited_claims`):** for each such cited passage the judge sees the question and the
  passage (never the labels) and returns `needed`, `useful_context`, `irrelevant` or `wrong_contract`. This is a
  context-precision signal for evidence that ESR/CER cannot see: memos with deal facts should come out as `useful_context`,
  a sibling contract used for the wrong question as `wrong_contract`.

Output goes to `citation-check.json` next to the results (`scope`, `support`, `relevance`, `rows`).

Limits: this tests support of a claim by its own citations, not legal correctness, and not whether needed
clauses were omitted (ESR/CER and the review rubric cover those). The judge is from the same model family as the
answerer, so it can share its blind spots; judge a sample by hand before trusting the percentages.
