# Claude Code and runtime integration

## Current boundary

Claude Code is the development tool; the runtime agent is separate code
(`claude_agent.py`) that calls the Claude API with an API key. The original deterministic
workflow (`build_agent`, Agent v2) is unchanged and stays a measured control. Local BGE still
does embeddings and reranking. Measurements of both are in
[evaluation-results.md](evaluation-results.md). Importing the package, running tests and
`--validate-only` need neither the `anthropic` package nor a key.

Start Claude Code from the repository root so it can read
[CLAUDE.md](../CLAUDE.md). That file contains the project constraints, mandatory
function annotations, commands, legal-evidence invariants and the next work.
Claude Code's project instruction mechanism is documented in its official
[memory guide](https://code.claude.com/docs/en/memory).

## Runtime options considered (decision: direct Claude API, see below)

| Option | What it supplies | Project responsibility |
|---|---|---|
| Python Claude Agent SDK (`claude-agent-sdk`) | Claude Code agent loop and tool infrastructure embedded in Python | Expose narrow retrieval tools, restrict runtime permissions, validate structured results and bound the loop |
| Claude Python client SDK (`anthropic`) | Calls to the Claude API | Implement the tool loop or orchestrate decisions through LangGraph, validate tool requests and persist state |

The Agent SDK is closest to the original assignment's suggested approach. A
direct API client inside LangGraph is also a reasonable small prototype when
explicit control over each decision is preferred. This is an integration choice
for the implementation stage; the choice made is recorded below. See the official
[Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview),
[Python Agent SDK repository](https://github.com/anthropics/claude-agent-sdk-python)
and [Python API client documentation](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python).

Verify the SDK's current supported authentication and the actual account/billing
configuration before relying on it. The Claude Code subscription does not fund
Console API calls (separate prepaid billing). Use the documented Agent SDK authentication rather
than extracting or copying a Claude Code login token; consult the official
[SDK quickstart](https://code.claude.com/docs/en/agent-sdk/quickstart) and
[Claude Code authentication guide](https://code.claude.com/docs/en/authentication).
These references were last checked for the 2026-10-16 handoff and were not
re-verified when the implementation was written.

For API-key authentication, the client reads `ANTHROPIC_API_KEY` from the
environment. Supply it outside source control, preferably through a local secret
manager or Colab Secrets. The project does not load `.env` automatically. The model
is an explicit option (`ClaudeAgentConfig.model`, `answer_model`; `--model`, `--answer-model`,
`--claude-models`); the model ids, config hash and usage are recorded in each report.

## First implementation slice (status: implemented as `claude_agent_v1`)

Apply LLM decisions to planning and evidence-gap assessment. Measurements show
that the current hard-coded facets ignore parts of mixed questions and
misinterpret negative rights. Repeated local search cannot fix a question plan
that omits the requested issue. Keep embeddings and reranking local so scarce
Colab GPU time serves retrieval, while Claude receives only the question and
selected evidence through the supported service.

1. Add typed schemas for a question plan, source selection, tool arguments,
   evidence records and the final cited answer. Every new Python function must
   annotate its parameters and return value; Ruff checks this requirement.
2. Wrap the current local retrieval and same-document reference lookup in narrow
   tools. `LegalRetriever.search` provides the four baselines;
   `scoped_search` and `lookup_section` in `evidence_agent.py` provide existing
   agent helpers. A scope must come from the question/corpus, never gold labels.
3. Replace or supplement `infer_facets` with a Claude decision that can express
   definitions, exceptions, consent, notice, cure, payments, exclusivity and
   explicit absence of a right. Validate the structured output before use.
4. Let Claude assess concrete evidence gaps, request a bounded follow-up search
   or reference lookup, and stop with explicit uncertainty when evidence remains
   insufficient. Configure limits on rounds, tool calls, tokens and cost.
5. Add answer synthesis with source filenames and full legal references. Distinguish
   primary evidence from supporting context and contractual conditions from
   unresolved transaction facts. No unconditional termination conclusion from
   acquisition alone.

Use nullable `retrieval_score`, `discovery_method` and `evidence_role` in the new
evidence schema. Reference lookup has no reranker score. Preserve a retriever's
raw score type/provenance rather than sorting lookup placeholders with logits.
If full parent evidence is needed, explicitly load it from
`legal_sections.jsonl` by parent UID and report the additional text budget:
the current reference-expansion helper returns child chunks only.

## Implemented slice: `claude_agent_v1` (offline-tested; measured on the real API, see [results](evaluation-results.md))

Runtime choice: **direct Claude API** (`anthropic` SDK, optional extra `pip install -e '.[claude]'`),
not the Agent SDK. The agent needs two custom tools and explicit budgets; the Agent SDK would add a
CLI dependency and built-in file/shell tools we must then disable, and its subscription login is not
available in Colab. Authentication is `ANTHROPIC_API_KEY` from the environment or Colab Secrets.

- Code: `src/legal_rag/claude_agent.py` (LangGraph `plan -> investigate -> answer`),
  `claude_schemas.py` (typed records, JSON schemas, validators), `scripts/run_claude_agent.py`.
- Plan and answer use structured outputs (`output_config.format`); investigation is a manual tool loop
  over `search_corpus` and `lookup_reference` only. No forced `tool_choice` (rejected by current models).
- Evidence records carry nullable `retrieval_score`, `discovery_method` and a per-citation
  `primary`/`supporting` role. Citations to ids absent from the ledger are removed; an answer with no
  valid cited claim is downgraded to `insufficient_evidence`.
- Limits (`ClaudeAgentConfig`): turns, tool calls, total tokens, estimated cost. Investigation stops at
  70% of the token/cost cap, reserving the rest for the answer; if the cap is already hit the answer call is skipped.
  Cost is an upper estimate from a hard-coded price table (no cache pricing).
- Trace: tool calls with args and result ids, one-sentence decision notes, per-call latency/tokens, config hash.
  No thinking content is logged. Contract text is passed inside `<contract_text>` tags as untrusted data.
- Default model is `claude-haiku-5-5`. `answer_model` (`--answer-model`, or `SEARCH+ANSWER` in `--claude-models`) writes the
  final answer with another model; Haiku search + Sonnet answer was the best measured configuration
  (about $0.046 per question; see results for caveats). Cost is tracked per model.
- Citation handling: the validator keeps the leading `E<n>` of a cited id (models sometimes echo the whole header).
  Runs store each claim with its cited passage text (`cited_claims`) for `scripts/check_citations.py`.
- Review tooling: [EVALUATION_PROMPT.md](../eval/EVALUATION_PROMPT.md) rubric and `scripts/build_review_packets.py`.
  `--refusal-fallback` enables the server-side fallback beta; it is untested here and off by default.
- Not done: independent or automatic generated-answer grading, full parent-section loading, repeat-run variance,
  held-out questions. A Colab T4 run of this code is recorded (`eval/results/claude-v3-t4/`). The `--claude-models` option adds Claude methods
  to `evaluate_retrieval.py` (`claude:<model>` or `claude:<search>+<answer>`).
  Smoke test: `python scripts/run_claude_agent.py --benchmark eval/benchmark.json --limit 1`.

## Decisions: what we deliberately do not use

Scope is a demonstration. These are judgments for this prototype, not claims that the tools are bad.
Vendor behavior below was not re-verified against current documentation; check it before relying on it.

**Claude Agent SDK.** Not used; the agent calls the Messages API directly under LangGraph.
- It drives the Claude Code CLI as a subprocess, an extra runtime to install and keep working on Colab. Our agent
  needs two custom read-only tools; the SDK's value (file, shell, edit tools, sessions, subagents) is not needed here
  and would have to be switched off to keep contract text away from arbitrary tool access.
- Authentication: the original idea was to reuse a Claude Code subscription. A Colab runtime has no logged-in
  Claude Code session, and Anthropic's guidance (as we understand it) is that products built on the SDK use API keys
  rather than subscription login. The project also forbids reading Claude Code login files. So an API key is needed
  either way, which removes the SDK's main attraction.
- Control and measurement: per-step model choice (Haiku searches, Sonnet answers), structured-output schemas per step,
  per-call token and cost accounting, and hard caps on turns, tool calls and spend are plain code in `claude_agent.py`
  and are unit-tested offline with a fake client. This is what the evaluation and observability criterion needs.
- Revisit if the agent must read or edit files, run code, or use subagents or MCP servers, or if the project moves to
  a machine where a supported SDK login is available.

**Ragas and DeepEval.** Not used for evaluation.
- Our primary metrics (Evidence Set Recall, Complete Evidence Rate) compare returned sections with labeled required
  clauses deterministically, at equal cutoffs, with no LLM judge. Ragas-style context recall and precision need
  reference answers or LLM judgments that we do not have.
- The metric they would add is answer faithfulness (does a cited passage support the claim). It needs an LLM judge:
  extra credits and time, a judge of the same model family as the answerer, and heavy dependencies that default to
  other providers' APIs and would need reconfiguration for Claude. For 12 questions this adds more setup than signal.
- Instead: a written rubric ([EVALUATION_PROMPT.md](../eval/EVALUATION_PROMPT.md)), blind review packets
  (`scripts/build_review_packets.py`) and cost, token and latency logging in the benchmark report.
- Citation faithfulness is covered by a small purpose-built check (`scripts/check_citations.py`: a free numeric
  check plus a judge that sees only a claim and its cited passages), cheaper than adopting a framework. Passages
  are stored in runs made after it was added; earlier runs must be re-run. Revisit these frameworks if the benchmark grows to many questions with
  reference answers or needs standard dashboards.

## Validation and experiment plan

Keep the current `agentic_v2` and all four baselines intact. Add the Claude-backed
method with its own name rather than relabeling old measurements (done: `claude:<model>`
methods in `evaluate_retrieval.py`, with metadata, cost and tests). Do not pass required
labels, answer notes, gold filenames or text anchors across the retrieval boundary; the review
packets and rubric use `answer_notes` only after the run, for grading.

Use mocked client responses for unit tests of structured-output validation,
tool dispatch, round limits, unavailable credentials, timeout/rate-limit handling,
insufficient evidence and citation provenance. Unit tests must stay offline and
must not spend API credits. Make a separate explicit smoke run with real access;
record model ID, prompts/configuration hash, usage, cost where available, local
retrieval timing, API timing, tool trace and errors. Do not log keys or hidden
chain-of-thought; concise decisions and observable tool calls are sufficient.

Compare ESR/CER at the same section cutoffs and record actual context size and
compute. Add generated-answer checks for cited evidence, unsupported legal
claims and stated conditions: retrieval CER alone cannot grade an answer.
Preserve [benchmark v1](../eval/benchmark.json) and
[raw CPU v1 results](../eval/results/cpu-v1.json); add separately authored
questions before claiming improvements generalize beyond the tuning set.

## Handoff checks

```bash
python -m pip install -e '.[dev]'
ruff check src scripts tests
python -m pytest -q
python scripts/evaluate_retrieval.py --validate-only
```

The last command requires existing parsed chunks. If those are missing, follow
[local setup](../README_LOCAL.md) or [Colab setup](../notebooks/COLAB.md).
Reuse a valid index; do not rebuild it because function annotations changed.
Historical result hashes identify the code measured at that time and are
expected to differ after this handoff. No full benchmark rerun is necessary for
annotations/docs alone; run a new experiment when retrieval behavior changes.
