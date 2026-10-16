# Retrieval benchmark

Run: 2026-10-16T12:22:43.951569+00:00.

Silver labels; retrieval coverage only. Complete evidence does not establish a legal conclusion.

Equal section cutoffs; agent rounds can use more searches. Word counts expose differing text budgets.

Device: `cpu`; corpus SHA-256: `4292554abe3f515ee73473fca13ac4860943cee43c80a260ae601f2fae9de20b`.

| Method | ESR@5 | CER@5 | ESR@10 | CER@10 | p50 ms | p95 ms | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 79.2% | 66.7% | 83.3% | 66.7% | 42 | 72 | 0 |
| bm25 | 47.9% | 25.0% | 56.9% | 33.3% | 4 | 9 | 0 |
| hybrid | 74.3% | 58.3% | 81.2% | 66.7% | 30 | 37 | 0 |
| hybrid_rerank | 72.9% | 58.3% | 85.4% | 66.7% | 6996 | 8460 | 0 |
| agentic_v2 | 63.9% | 33.3% | 63.9% | 33.3% | 7662 | 26982 | 0 |
| claude:claude-haiku-5-5 | 86.1% | 66.7% | 100.0% | 100.0% | 58067 | 73948 | 0 |
| claude:claude-sonnet-5-5 | 93.8% | 75.0% | 93.8% | 75.0% | 41349 | 65583 | 0 |

## Claude agents

Pool = all retrieved evidence (cited first); Cited = only sections the answer cites. Cost is an upper estimate from token usage.

| Method | Cited ESR | Cited CER | Cited sections | Tool calls | Tokens | Cost/q | Total cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| claude:claude-haiku-5-5 | 83.3% | 83.3% | 8.5 | 6.8 | 55894 | $0.0081 | $0.097 |
| claude:claude-sonnet-5-5 | 93.8% | 75.0% | 7.2 | 2.8 | 26688 | $0.0829 | $0.995 |

## Incomplete evidence

| Question | Method | Cutoff | Missing required evidence |
|---|---|---:|---|
| terranode_competitor | dense | 5 | competitor_definition, consent_requirement, termination_notice_and_cure |
| terranode_competitor | dense | 10 | competitor_definition, consent_requirement, termination_notice_and_cure |
| terranode_competitor | bm25 | 5 | competitor_definition, consent_requirement, competitor_carveout, termination_notice_and_cure |
| terranode_competitor | bm25 | 10 | competitor_definition, consent_requirement, competitor_carveout, termination_notice_and_cure |
| terranode_competitor | hybrid | 5 | competitor_definition, consent_requirement, competitor_carveout, termination_notice_and_cure |
| terranode_competitor | hybrid | 10 | competitor_definition, consent_requirement, termination_notice_and_cure |
| terranode_competitor | hybrid_rerank | 5 | competitor_definition, consent_requirement, competitor_carveout |
| terranode_competitor | hybrid_rerank | 10 | consent_requirement |
| terranode_competitor | claude:claude-sonnet-5-5 | 5 | competitor_definition |
| terranode_competitor | claude:claude-sonnet-5-5 | 10 | competitor_definition |
| pinnacle_licensee | dense | 5 | control_definition |
| pinnacle_licensee | dense | 10 | control_definition |
| pinnacle_licensee | bm25 | 5 | control_definition, exclusivity_conversion |
| pinnacle_licensee | bm25 | 10 | control_definition |
| pinnacle_licensee | hybrid | 5 | control_definition |
| pinnacle_licensee | hybrid | 10 | control_definition |
| pinnacle_licensee | hybrid_rerank | 5 | control_definition, exclusivity_conversion, unchanged_fees |
| pinnacle_licensee | hybrid_rerank | 10 | control_definition |
| pinnacle_licensee | agentic_v2 | 5 | control_definition, unchanged_fees, no_termination |
| pinnacle_licensee | agentic_v2 | 10 | control_definition, unchanged_fees, no_termination |
| pinnacle_licensee | claude:claude-haiku-5-5 | 5 | no_termination |
| pinnacle_licensee | claude:claude-sonnet-5-5 | 5 | control_definition |
| pinnacle_licensee | claude:claude-sonnet-5-5 | 10 | control_definition |
| continental_acceleration | bm25 | 5 | automatic_acceleration, general_remedies_exception |
| continental_acceleration | bm25 | 10 | automatic_acceleration, general_remedies_exception |
| continental_acceleration | claude:claude-haiku-5-5 | 5 | automatic_default_no_cure, automatic_acceleration |
| terranode_notice_affiliate | bm25 | 5 | general_consent, notice_and_silence |
| terranode_notice_affiliate | bm25 | 10 | general_consent |
| pinnacle_licensor | agentic_v2 | 5 | licensor_control_rule |
| pinnacle_licensor | agentic_v2 | 10 | licensor_control_rule |
| continental_make_whole | bm25 | 5 | mandatory_prepayment |
| continental_make_whole | hybrid_rerank | 5 | mandatory_prepayment |
| wellstone_options | bm25 | 5 | termination_window, transaction_notice |
| wellstone_options | bm25 | 10 | termination_window, transaction_notice |
| wellstone_options | hybrid | 5 | transaction_notice |
| wellstone_options | agentic_v2 | 5 | transaction_notice |
| wellstone_options | agentic_v2 | 10 | transaction_notice |
| meridian_competitor | dense | 5 | assignment_consent, deemed_assignment, consent_discretion |
| meridian_competitor | dense | 10 | assignment_consent |
| meridian_competitor | bm25 | 5 | assignment_consent, deemed_assignment, consent_discretion |
| meridian_competitor | bm25 | 10 | assignment_consent, deemed_assignment, consent_discretion |
| meridian_competitor | hybrid | 5 | assignment_consent, deemed_assignment, consent_discretion |
| meridian_competitor | hybrid | 10 | assignment_consent, deemed_assignment |
| meridian_competitor | hybrid_rerank | 5 | assignment_consent, deemed_assignment, consent_discretion |
| meridian_competitor | hybrid_rerank | 10 | assignment_consent, deemed_assignment, consent_discretion |
| meridian_competitor | agentic_v2 | 5 | assignment_consent, deemed_assignment, consent_discretion |
| meridian_competitor | agentic_v2 | 10 | assignment_consent, deemed_assignment, consent_discretion |
| meridian_competitor | claude:claude-haiku-5-5 | 5 | assignment_consent |
| meridian_competitor | claude:claude-sonnet-5-5 | 5 | consent_discretion |
| meridian_competitor | claude:claude-sonnet-5-5 | 10 | consent_discretion |
| novabridge_silence_pricing | agentic_v2 | 5 | pricing_threshold_and_reset |
| novabridge_silence_pricing | agentic_v2 | 10 | pricing_threshold_and_reset |
| apex_msa_successor | bm25 | 5 | assignment_subject_to_control |
| apex_msa_successor | bm25 | 10 | assignment_subject_to_control |
| apex_msa_successor | agentic_v2 | 5 | assignment_subject_to_control |
| apex_msa_successor | agentic_v2 | 10 | assignment_subject_to_control |
| terraverde_recapture | agentic_v2 | 5 | control_assignment_and_recapture |
| terraverde_recapture | agentic_v2 | 10 | control_assignment_and_recapture |
| compare_competitor_remedies | dense | 5 | terranode_consent_rule, terranode_competitor_rule, terranode_notice_and_cure |
| compare_competitor_remedies | dense | 10 | terranode_consent_rule, terranode_competitor_rule, terranode_notice_and_cure |
| compare_competitor_remedies | bm25 | 5 | terranode_consent_rule, terranode_competitor_rule, terranode_notice_and_cure, meridian_immediate_right |
| compare_competitor_remedies | bm25 | 10 | terranode_consent_rule, terranode_competitor_rule, terranode_notice_and_cure, meridian_immediate_right |
| compare_competitor_remedies | hybrid | 5 | terranode_consent_rule, terranode_competitor_rule, terranode_notice_and_cure |
| compare_competitor_remedies | hybrid | 10 | terranode_consent_rule, terranode_competitor_rule, terranode_notice_and_cure |
| compare_competitor_remedies | hybrid_rerank | 5 | terranode_consent_rule, terranode_notice_and_cure |
| compare_competitor_remedies | hybrid_rerank | 10 | terranode_consent_rule, terranode_notice_and_cure |
| compare_competitor_remedies | agentic_v2 | 5 | terranode_consent_rule, terranode_notice_and_cure |
| compare_competitor_remedies | agentic_v2 | 10 | terranode_consent_rule, terranode_notice_and_cure |
| compare_competitor_remedies | claude:claude-haiku-5-5 | 5 | terranode_consent_rule, terranode_competitor_rule |

Full per-query metrics, pool coverage, split summaries and agent traces are in results.json.
