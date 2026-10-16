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


## Colab T4 run, same configuration repeated (`claude-v3-t4`, 2026-10-16)

Raw data: [eval/results/claude-v3-t4/](../eval/results/claude-v3-t4/) (`report.md`, `results.json` incl. `cited_claims`,
`queries.jsonl`, `citation-check.json`). Colab free tier, **Tesla T4 (CUDA)**, existing index reused, five baselines plus
`claude-haiku-5-5+claude-sonnet-5-5` (per-question cost cap $0.15). It is a second independent run of the `claude-v2-split`
configuration, so it also shows run-to-run variation.

| Method | ESR@5 | CER@5 | ESR@10 | CER@10 | p50 latency (T4) | p50 latency (CPU) |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 79.2% | 66.7% | 83.3% | 66.7% | 15 ms | 42 ms |
| BM25 | 47.9% | 25.0% | 56.9% | 33.3% | 10 ms | 4 ms |
| Hybrid | 74.3% | 58.3% | 81.2% | 66.7% | 22 ms | 30 ms |
| Hybrid + Rerank | 72.9% | 58.3% | 85.4% | 66.7% | 1.07 s | 6.9-7.0 s |
| Agent v2 | 63.9% | 33.3% | 63.9% | 33.3% | 1.15 s | 7.7 s |
| Haiku + Sonnet answer | 95.8% | 91.7% | 100% | 100% | 36.4 s | 57.8 s |

**GPU effect.** Baseline retrieval metrics are identical to the CPU runs; only latency changed. Reranking is about 6.5x faster
on the T4. For the Claude agent, mean local retrieval time per question fell from 25.5 s (CPU, `claude-v2-split`) to 3.6 s,
while API time stayed about 31-33 s, so the T4 helps the local models and the Claude latency is now dominated by API calls.
(Hardware, packages and hashes are in `results.json` metadata; one run per setting, so treat speedups as indicative.)

**Repeatability of the Claude configuration (v2-split vs v3-t4, same settings).**

| | v2-split (CPU) | v3-t4 (T4) |
|---|---:|---:|
| ESR@5 / CER@5 | 95.8% / 83.3% | 95.8% / 91.7% |
| ESR@10 / CER@10 | 100% / 100% | 100% / 100% |
| Cited ESR / CER | 100% / 100% | 100% / 100% |
| Cited sections per question | 10.4 | 9.6 |
| Tool calls per question (8-call limit hit) | 7.1 (5 of 12) | 6.9 (3 of 12) |
| Cost per question | $0.046 | $0.0455 (total $0.546) |
| Cited sections outside the question's labeled files | 13% | 3% |

Retrieval metrics at cutoff 10 and cost reproduced; CER@5 moved by one question; how often the agent cites unlabeled files
varied noticeably (13% vs 3%). This is two runs on 12 silver-labeled questions, not a variance estimate.

**Citation checks** (`scripts/check_citations.py`, 81 claims, judge `claude-haiku-5-5`, cost about $0.017):

| Check | Result |
|---|---|
| Support judge | 68% supported (55), 31% partial (25), 1% not supported (1 claim) |
| Cited sections outside the question's labeled files | 3 of 115 (3%) |
| Relevance judge on those 3 passages | 1 needed, 2 useful_context, 0 irrelevant, 0 wrong_contract |
| Numeric/section check | 11 flagged with the first version of the check (see below) |

How to read them (a same-family judge, partly checked by hand on a sample of reasons):
- `partial` is mostly compound claims where the cited passage supports most elements but not all (for example a claim that
  also asserts that notices do not delay acceleration, which the text does not state), so it flags inference beyond the text,
  and part of the 31% reflects claim granularity rather than errors. Questions with the most partial verdicts:
  `compare_competitor_remedies` (5 of 8), `terranode_notice_affiliate` (4 of 7). No claim in `terraverde_recapture` and
  `continental_make_whole` was partial. The single `not_supported` claim says a sibling agreement "does not govern this
  question", which the passages do not state.
- The first numeric check was noisy: of 11 flags, 7 were claims the judge rated supported (dotted section numbers such as
  `12.3(a)`, "Article 1" and "one or a series" were read as quantities). The check was then fixed (part references, dotted
  section numbers without "Section", and "one" are no longer read as quantities); re-evaluated on these claims it flags 5, and
  the one inspected (a "100% stock purchase" in a claim about Change of Control) is a real flag: the figure is not in the
  cited passage. `citation-check.json` in the run folder keeps the original 11 flags.
- The three off-scope passages were relevant (one needed, two useful context), unlike the irrelevant 10-K and fund-overview
  citations seen in the one-question smoke run, so off-scope citations vary between runs.


## Independent answer review of `claude-v3-t4`

Verdicts: [answer-review.json](../eval/results/claude-v3-t4/review/answer-review.json). A fresh LLM session with no project
context (model not recorded) graded the 12 blind packets (question, `answer_notes`, answer) with the rubric in
[EVALUATION_PROMPT.md](../eval/EVALUATION_PROMPT.md); `citation_support` was not scored because packets carry no passages.
This is more independent than the earlier review by the assistant that helped build the system, but still an LLM, not a lawyer.

| Verdict | Questions |
|---|---:|
| correct | 7 |
| mostly_correct | 4 (`pinnacle_licensee`, `terraverde_recapture`, `continental_acceleration`, `wellstone_options`) |
| incomplete | 1 (`pinnacle_licensor`) |
| wrong | 0 |

Mean criterion scores (0-2): coverage 1.92, correctness 1.92, conditions 1.92, no_overclaim 1.75, gaps_honest 1.75 (92.5% of the
maximum). Two answers had missed required points; five had claims beyond the notes.

**What the reviewer found.**
- `pinnacle_licensor`: the answer correctly says there is no conversion and no termination right but does not state that the
  agreement and exclusivity continue and bind the successor. The same configuration's earlier answer (`claude-v2-split`)
  did say so, so completeness varies between runs.
- `continental_acceleration`: acceleration is distinguished correctly from the §8.01 declaration mechanism, but the added
  statement that the make-whole applies only if the Change of Control date is before 1 November 2025 was judged an
  oversimplified side claim.
- `wellstone_options`: all deadlines present, but the answer ended with extraneous caveats and a reference to an
  "investigator's note", an internal working input that should not appear in an answer.
- Most "unsupported claims" mean "not in the notes", not "wrong": the notes are short and the contracts contain more terms
  (security deposit, parent guaranty, insolvency termination). The reviewer could not check them. Do not read the 5 of 12 as 5 errors.

**Action taken.** The leak of the internal note (1 of 12 answers here, 1 of 12 in `claude-v2-split`) was a prompt defect:
the answer prompt now tells the model never to mention the investigator's note. This changes the configuration hash and
has not been re-run, so the recorded results still contain the leak.

**Agreement with the assistant's own review of the split run.** The assistant found no substantive error on the earlier
run; the independent reviewer found one omission and one oversimplification in this run. Different runs and different
reviewers: the honest summary is that most answers match the notes and details beyond the notes need a lawyer.
