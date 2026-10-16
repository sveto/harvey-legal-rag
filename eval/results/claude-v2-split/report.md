# Retrieval benchmark

Run: 2026-10-16T12:44:33.787013+00:00.

Silver labels; retrieval coverage only. Complete evidence does not establish a legal conclusion.

Equal section cutoffs; agent rounds can use more searches. Word counts expose differing text budgets.

Device: `cpu`; corpus SHA-256: `4292554abe3f515ee73473fca13ac4860943cee43c80a260ae601f2fae9de20b`.

| Method | ESR@5 | CER@5 | ESR@10 | CER@10 | p50 ms | p95 ms | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid_rerank | 72.9% | 58.3% | 85.4% | 66.7% | 6937 | 8537 | 0 |
| claude:claude-haiku-5-5+claude-sonnet-5-5 | 95.8% | 83.3% | 100.0% | 100.0% | 57841 | 66586 | 0 |

## Claude agents

Pool = all retrieved evidence (cited first); Cited = only sections the answer cites. Cost is an upper estimate from token usage.

| Method | Cited ESR | Cited CER | Cited sections | Tool calls | Tokens | Cost/q | Total cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| claude:claude-haiku-5-5+claude-sonnet-5-5 | 100.0% | 100.0% | 10.4 | 7.1 | 51399 | $0.0460 | $0.552 |

## Incomplete evidence

| Question | Method | Cutoff | Missing required evidence |
|---|---|---:|---|
| terranode_competitor | hybrid_rerank | 5 | competitor_definition, consent_requirement, competitor_carveout |
| terranode_competitor | hybrid_rerank | 10 | consent_requirement |
| pinnacle_licensee | hybrid_rerank | 5 | control_definition, exclusivity_conversion, unchanged_fees |
| pinnacle_licensee | hybrid_rerank | 10 | control_definition |
| pinnacle_licensee | claude:claude-haiku-5-5+claude-sonnet-5-5 | 5 | control_definition |
| continental_make_whole | hybrid_rerank | 5 | mandatory_prepayment |
| meridian_competitor | hybrid_rerank | 5 | assignment_consent, deemed_assignment, consent_discretion |
| meridian_competitor | hybrid_rerank | 10 | assignment_consent, deemed_assignment, consent_discretion |
| compare_competitor_remedies | hybrid_rerank | 5 | terranode_consent_rule, terranode_notice_and_cure |
| compare_competitor_remedies | hybrid_rerank | 10 | terranode_consent_rule, terranode_notice_and_cure |
| compare_competitor_remedies | claude:claude-haiku-5-5+claude-sonnet-5-5 | 5 | terranode_notice_and_cure |

Full per-query metrics, pool coverage, split summaries and agent traces are in results.json.
