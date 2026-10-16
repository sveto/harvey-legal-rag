# Evidence Gap Agent v2: rule-validated evidence and reference expansion

Agent v2 is part of this repository. It is a deterministic LangGraph workflow:
Python rules select facets, recognize candidate evidence and decide whether to
search again. It uses no generative LLM. BGE embeddings and the optional BGE
cross-encoder provide local retrieval and reranking. It is kept as a measured control;
the Claude-backed agent that replaces its fixed facets is described in
[claude-integration.md](docs/claude-integration.md) (`scripts/run_claude_agent.py`).

## Colab

```python
%cd /content/harvey-legal-rag
%pip install -e '.[dev]'
!python -m pytest -q tests/test_evidence_agent.py
!python scripts/run_evidence_agent.py \
    'Does TerraNode have termination rights if Aldersgate is acquired by a competitor?' \
    --source terranode-isa.docx --output artifacts --device cuda \
    --save artifacts/terranode_evidence_v2.json
```

This assumes the corpus was parsed with `--token-aware` and indexed under
`artifacts/`; follow [Colab setup](notebooks/COLAB.md) first. Locally use the same
script with `--device cpu`. Omit `--source` for automatic filename-based source
inference. `--max-rounds` accepts 0..3 (default 2); `--no-rerank` skips the
cross-encoder. `--save` writes evidence text and graph events to a JSON file.

The graph runs `plan -> initial -> expand -> assess`, then repeats
`targeted -> expand -> assess` for missing facets within the round limit.
Expansion also runs before the first assessment, even if lexical facets already
appear covered. With no resolved source, the current expansion node does no
cross-reference lookup. The retained v1 definitions earlier in the module are
overridden by the later v2 `build_agent` and `evidence_report` definitions.

## Changes

1. Role-specific, issue-conditioned lexical checks prevent SLA chronic-failure termination from satisfying an assignment/change-of-control termination facet.
2. `expand` parses explicit references such as `Notwithstanding Section 9.1` and loads matching indexed child chunks from the resolved document, without embeddings or external LLM calls. It does not restore a full parent section.
3. The graph checks gaps after reference expansion, and runs targeted retrieval only for missing facets, at most `max_rounds` times.
4. Search trace records `expand_reference` events with source and destination legal refs.

## Known limitations

- These are conservative lexical heuristics, **not legal entailment**. `candidate_evidence_found` is not a legal conclusion.
- Reference extraction currently supports a single numeric legal reference after `Section`/`Sections` and does not parse ranges, comma-separated references, or exhibit references.
- Role coverage can be falsely satisfied by unrelated clauses or contracts. The
  [implemented benchmark](docs/evaluation.md) checks exact source/reference and
  text anchors. In the recorded run, six questions had no missing lexical facets
  while still lacking required evidence; see [measured results](docs/evaluation-results.md).
- The existing `scoped_search` reranker is optional, but embedding/reranking still requires the local model weights.
- A structural lookup uses the placeholder `score=0.0`; it is not a reranker
  score. The evaluation adapter never sorts it with cross-encoder logits. A
  separate score/provenance/primary-versus-supporting evidence schema is planned.
- The workflow returns candidates, not a cited natural-language legal answer.
  Its fixed facets and filename source inference are known failure points.

The Claude-backed planning, evidence assessment and cited answers are implemented in
[claude-integration.md](docs/claude-integration.md); v2 stays unchanged as a measured control. All new/modified Python function
parameters and return values must remain annotated, as required by [CLAUDE.md](CLAUDE.md).
