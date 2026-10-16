# Measured retrieval results: benchmark v1

This is a preserved historical experiment. Function annotations, typing helpers
and contributor documentation were added afterwards; the raw run retains its
original implementation hashes. Current files may have different hashes without
the old metrics being relabeled as measurements of the current checkout.

Local CPU run on 2026-10-16: 12 questions, five methods, 60 successful query/method pairs. The index contains 19 documents and 1,380 chunks; labels cover eight source contracts. The existing retrievers and v2 agent heuristics were left unchanged.

The agent in this run is a deterministic LangGraph workflow without an LLM. BGE builds embeddings and reranks passages. This section describes the historical `cpu-v1` run only; the Claude agent runs and manual answer review are in the sections at the end of this file.

## All 12 questions

| Method | ESR@5 | CER@5 | ESR@10 | CER@10 | CPU p50 |
|---|---:|---:|---:|---:|---:|
| Dense | 79.2% | 66.7% | 83.3% | 66.7% | 0.027 s |
| BM25 | 47.9% | 25.0% | 56.9% | 33.3% | 0.004 s |
| Hybrid | 74.3% | 58.3% | 81.2% | 66.7% | 0.031 s |
| Hybrid + Rerank | 72.9% | 58.3% | 85.4% | 66.7% | 8.089 s |
| Agentic Retrieval v2 (rules) | 63.9% | 33.3% | 63.9% | 33.3% | 8.856 s |

ESR is the macro average fraction of required evidence groups found; CER is the fraction of questions with all groups found. These silver labels measure retrieval coverage, not correct legal conclusions.

Dense retrieves complete evidence for 8/12 questions at k=5. At k=10, Dense, Hybrid and Hybrid + Rerank each complete 8/12; the reranker has the highest ESR but does not improve CER over Dense on this set. Agent v2 completes 4/12 at both cutoffs. This sample does not establish a general method ranking.

## Nine new evaluation probes

Three previously explored questions are excluded from this table. The nine new probes still form a small authored set, not independently held-out legal gold.

| Method | ESR@10 | CER@10 |
|---|---:|---:|
| Dense | 88.9% | 77.8% |
| BM25 | 63.9% | 44.4% |
| Hybrid | 86.1% | 77.8% |
| Hybrid + Rerank | 86.1% | 77.8% |
| Agentic Retrieval v2 (rules) | 60.2% | 22.2% |

## Agent misses

| Question | Required evidence missing at k=10 |
|---|---|
| pinnacle_licensee | control_definition, unchanged_fees, no_termination |
| pinnacle_licensor | licensor_control_rule |
| wellstone_options | transaction_notice |
| meridian_competitor | assignment_consent, deemed_assignment, consent_discretion |
| novabridge_silence_pricing | pricing_threshold_and_reset |
| apex_msa_successor | assignment_subject_to_control |
| terraverde_recapture | control_assignment_and_recapture |
| compare_competitor_remedies | terranode_consent_rule, terranode_notice_and_cure |

The agent reports no missing lexical facets for 10/12 questions, but 6 of those still miss annotated required evidence. `missing: []` is therefore not a reliable completeness signal.

Trace inspection identifies concrete failure modes:

- Mixed questions can be reduced to termination/cure facets. The Pinnacle questions do not plan for exclusivity, fees or a negative termination rule; the Meridian and Apex questions omit the consent facet.
- Filename-based scope inference is ambiguous for Pinnacle, Meridian and Apex because each name occurs in more than one document. Corpus-wide targeted searches then fill facets with clauses from other agreements.
- The Wellstone and NovaBridge traces spend two rounds searching for competitor and cure facets while missing requested notice or pricing provisions.
- TerraVerde plans only consent; it returns section 21 and drops the requested recapture evidence in section 22.
- The multi-document comparison accepts deal-memo summaries for some facets and stops without the required TerraNode consent and notice/cure clauses. The benchmark intentionally requires the underlying contractual provisions.

These are observations for designing later changes, not heuristics patched during this measurement. Add separate test questions before tuning the planner.

## Budget and runtime context

| Method | Mean sections at k=10 | Mean words at k=10 |
|---|---:|---:|
| Dense | 10.0 | 1522 |
| BM25 | 10.0 | 1809 |
| Hybrid | 10.0 | 1796 |
| Hybrid + Rerank | 10.0 | 1860 |
| Agentic Retrieval v2 (rules) | 5.5 | 872 |

Cutoffs cap unique sections; they do not equalize tokens or compute. The agent frequently returns fewer than ten sections; this contributes to its lower recall and is visible above. Its order is round robin across facets without comparing structural lookup scores with reranker logits. Baselines search globally; the agent retains its existing question-only source inference.

Timing is one pass after shared model warmup, using CPU with four PyTorch threads. It excludes model loading/download. No GPU was available here; these numbers are not Colab/T4 performance claims. The run used cached weights with network access disabled for Hugging Face.

## Reproduce and inspect

```bash
python scripts/evaluate_retrieval.py --validate-only
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/evaluate_retrieval.py --device cpu --threads 4
```

For Colab, use `--device cuda` and omit the offline variables on the first model download.

- [Raw measured run](../eval/results/cpu-v1.json): all per-query results, missing evidence, agent traces, pool metrics, environment and source hashes.
- [Benchmark labels](../eval/benchmark.json).
- [Protocol and limitations](evaluation.md).

Benchmark SHA-256: `05319ae4748fa87098983ee6dd05bc740b8d894780f512d5bd98c0aebeb21a3f`.
Corpus SHA-256: `4292554abe3f515ee73473fca13ac4860943cee43c80a260ae601f2fae9de20b`.


# Claude agent runs (`claude-v1`, 2026-10-16)

Raw data: [eval/results/claude-v1/](../eval/results/claude-v1/) (`report.md`, `results.json`, `queries.jsonl`).
Same benchmark v1, CPU, one run per method; the five retrieval baselines were re-measured in this run, so
their latencies differ from `cpu-v1.json`. Claude agents are `claude_agent_v1` (see
[claude-integration.md](claude-integration.md)): Claude plans, searches with up to 8 tool calls, then answers
with citations. Only the question text is sent. Models: Haiku 5.5 and Sonnet 5.5 (one model for all steps).

| Method | ESR@5 | CER@5 | ESR@10 | CER@10 | p50 latency | Cost/question |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 79.2% | 66.7% | 83.3% | 66.7% | 0.04 s | 0 |
| BM25 | 47.9% | 25.0% | 56.9% | 33.3% | 0.004 s | 0 |
| Hybrid | 74.3% | 58.3% | 81.2% | 66.7% | 0.03 s | 0 |
| Hybrid + Rerank | 72.9% | 58.3% | 85.4% | 66.7% | 7 s | 0 |
| Agent v2 (deterministic) | 63.9% | 33.3% | 63.9% | 33.3% | 7.7 s | 0 |
| Claude Haiku 5.5 | 86.1% | 66.7% | 100% | 100% | 58 s | ~$0.008 |
| Claude Sonnet 5.5 | 93.8% | 75.0% | 93.8% | 75.0% | 41 s | ~$0.083 |

Evidence order for Claude is label-blind: cited primary sections, cited supporting, then uncited retrieved
sections. Counting only sections the answer actually cites: Haiku ESR 83.3% / CER 83.3%, Sonnet 93.8% / 75.0%.
Haiku averaged 6.8 tool calls (55.9k tokens), Sonnet 2.8 (26.7k). Total cost of both runs: about $1.09
(upper estimate from token usage, no cache pricing).

**Caveats.** 12 silver-labeled questions, one stochastic run: a one-to-three question difference is noise.
Claude gets far more compute and text than the baselines (equal section cutoffs only). The metrics grade
retrieval, not answer correctness. The `conclusion_type` field is not informative: nearly every answer was
labeled `conditional_right`, including "no right found" cases.

## Manual answer review

One reviewer (Claude, an LLM, so not independent) compared all 24 answers with the `answer_notes`.

- **Both models give substantively correct, appropriately conditional answers on all 12 questions.** None
  asserts an unconditional termination right from an acquisition alone.
- **Haiku is more complete, sometimes noisy.** It found definitions Sonnet missed (TerraNode "Direct
  Competitor") and kept "sole and absolute discretion" for Meridian consent, which Sonnet dropped. It
  also flagged the notice-breach route to termination in the Pinnacle licensee case. Cost: longer answers,
  occasional irrelevant material from other contracts, internal ids such as "(E7, E12)" in prose, and
  tool-call-limit stops in 4 of 12 questions.
- **Sonnet is more precise and better at stating the rule.** It answered the multi-document comparison
  as the notes expect (Meridian immediate right, TerraNode notice and cure) where Haiku hedged "neither is
  a clean immediate case". It stated the make-whole date-of-control rule where Haiku left it as an
  unresolved conflict. Its weakness is shallow search (about 3 tool calls): it reports the missing
  definitions as gaps (competitor, Change of Control) instead of fetching them.
- **Honesty about gaps is good in both.** Evidence gaps name real missing items (definitions, Section 5.3,
  deal facts) rather than inventing them.
- **Defect found: citation format.** In 2 Haiku answers (`continental_acceleration`, `terraverde_recapture`)
  the model wrote the whole evidence header as the id (`E6 source=... ref=7.09(a)`). The validator removed
  those citations and downgraded two correct answers to `insufficient_evidence`, which lowers Haiku's
  cited-only numbers above. The validator now extracts the leading `E<n>`; these results were not re-run, so the
  table above reflects the old behavior.
- **Overall judgment:** for this task Sonnet's answers are the better written and legally tighter; Haiku
  retrieves more completely at one tenth of the cost. Neither judgment is verified by an independent
  lawyer or a held-out question set.

Not yet measured: Haiku-search + Sonnet-answer (`--claude-models claude-haiku-5-5+claude-sonnet-5-5`),
repeat runs for variance, and new held-out questions.


## Haiku searches, Sonnet answers (`claude-v2-split`, 2026-10-16)

Raw data: [eval/results/claude-v2-split/](../eval/results/claude-v2-split/). Same benchmark, CPU, one run;
`claude-haiku-5-5` plans and searches, `claude-sonnet-5-5` writes the final answer
(`--claude-models claude-haiku-5-5+claude-sonnet-5-5`). Hybrid + Rerank was re-run as a control.

| Method | ESR@5 | CER@5 | ESR@10 | CER@10 | Cited ESR | Cited CER | Cited sections | p50 latency | Cost/question |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hybrid + Rerank | 72.9% | 58.3% | 85.4% | 66.7% | - | - | - | 6.9 s | 0 |
| Haiku only (v1) | 86.1% | 66.7% | 100% | 100% | 83.3% | 83.3% | 8.5 | 58 s | ~$0.008 |
| Sonnet only (v1) | 93.8% | 75.0% | 93.8% | 75.0% | 93.8% | 75.0% | 7.2 | 41 s | ~$0.083 |
| **Haiku + Sonnet answer** | 95.8% | 83.3% | 100% | 100% | 100% | 100% | 10.4 | 58 s | ~$0.046 |

Total cost of the 12 questions: about $0.55. Tool calls averaged 7.1; 6 of 12 questions reached the 8-call limit.
No validation problems occurred and no answer was downgraded.

**Reading it.**
- The search stage is the same Haiku in the first and last rows, so ESR@10 is unchanged at 100%; what changed is the
  answer step, at roughly half the price of Sonnet alone because Sonnet reads the evidence once without searching.
- Two changes are mixed in the improvement over the v1 rows: the stronger answer model and the evidence-id
  normalization added after v1 (which removed Haiku's two downgraded answers). Cited ESR/CER of 100% is not a
  clean model effect, and citing more sections (10.4 vs 7.2 to 8.5) raises a coverage metric by itself.
- 12 silver-labeled questions, one stochastic run: treat 100% versus 94% as indistinguishable.

**Manual read of the split answers** (same single LLM reviewer, not independent): they keep Haiku's completeness
and Sonnet's precision. The Meridian answer keeps "sole and absolute discretion" (dropped by Sonnet alone). The
make-whole answer states the date-of-control rule and applies it to the memo's expected closing date
(unresolved in Haiku alone). The multi-document comparison matches the notes (Meridian immediate right,
TerraNode notice and cure). The Pinnacle licensor question is labeled `no_right_found` and the TerraVerde
question `right_found`; `conclusion_type` now varies, but it was not validated against labels.

**Current recommendation:** Haiku for search and Sonnet for the answer is the best measured quality-per-cost
configuration here. Missing: repeat runs, an independent answer review, an automatic citation-faithfulness check.


## Cited sections outside the question's labeled files

From `scripts/check_citations.py` (scope check, no model involved; saved as `citation-check.json` in each run folder).

| Run / method | Cited sections | From files with no label for the question | Share |
|---|---:|---:|---:|
| `claude-v1` Haiku | 102 | 16 | 16% |
| `claude-v1` Sonnet | 86 | 2 | 2% |
| `claude-v2-split` Haiku + Sonnet | 125 | 16 | 13% |

Many are deal memos (transaction facts the questions' answers legitimately use), but some come from another contract
(for example NovaBridge Partnership §11.2 for a TerraNode question, Pinnacle E-commerce for a Pinnacle licensor question).
Whether these citations helped or mixed contracts is **not yet judged**: the relevance judge needs the cited passage text,
which these runs did not store; it runs on the next re-run. Retrieval ESR/CER do not see this evidence at all.
