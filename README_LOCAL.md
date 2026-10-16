# Harvey Legal RAG: local setup

The source includes parser v3, Evidence Gap Agent v2 and the evaluation pipeline.
Downloaded DOCX files and generated indexes are excluded from source control;
your workspace may already contain them. Check `data/docx/` and `artifacts/`
before downloading or rebuilding. The canonical overview is [README.md](README.md).

## Linux / VS Code

```bash
cd /path/to/harvey-legal-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Clone/extract the project first if needed. In VS Code, run `code .` and choose
`.venv/bin/python` in **Python: Select Interpreter**. Python >=3.10 is required;
use an interpreter supported by the installed PyTorch/FAISS packages. The recorded
CPU benchmark used Python 3.14.4; exact versions are in
[the raw run](eval/results/cpu-v1.json). CPU works without CUDA. Model/package
downloads can be large; no API key is needed for the local BGE workflow (only the Claude agent and the citation judge need one).

For Claude Code, launch it from this directory and read [CLAUDE.md](CLAUDE.md)
and [the integration guide](docs/claude-integration.md). The Claude agent is
implemented (`scripts/run_claude_agent.py`); it needs `pip install -e '.[dev,claude]'`
and `ANTHROPIC_API_KEY`, and spends API credits.

## Corpus and index

```bash
python scripts/download_corpus.py --output data/docx
legal-rag parse --input data/docx --output artifacts --token-aware --max-tokens 480
pytest -q
ruff check src scripts tests
python scripts/check_boundaries.py --output artifacts
legal-rag index --output artifacts --device cpu --batch-size 8
python scripts/compare_methods.py --output artifacts --device cpu
python scripts/evaluate_retrieval.py --validate-only
python scripts/evaluate_retrieval.py --device cpu
```

The downloader fetches `.docx` files from the public Harvey GitHub task path. If the upstream URL changes or the dataset is unavailable, put the 19 DOCX files in `data/docx/` manually. Check corpus count before parsing: `find data/docx -name '*.docx' | wc -l`. Do not put answer keys or task JSON in the retrieval corpus. If the existing index matches the chunks, skip parse/index and proceed to queries or evaluation.

## Agent v2

```bash
python scripts/run_evidence_agent.py \
  'Does TerraNode have termination rights if Aldersgate is acquired by a competitor?' \
  --source terranode-isa.docx --output artifacts --device cpu \
  --save artifacts/terranode_evidence_v2.json
```

Use `--no-rerank` for a quicker, lower-memory CPU run. Models download on first use. `artifacts/` and `data/` are ignored by Git. Index files are regenerated from DOCX.

## Notes

- The parser preserves numbered subclause references such as `7.09(a)`/`7.09(b)`
  and excludes spurious `by and between` headings. Run tests after parser changes.
- Agent v2 performs deterministic lexical checks, explicit-reference expansion
  and targeted retrieval via LangGraph. It uses no generative LLM and does not
  prove legal entailment. See [agent details](README_AGENT_V2.md).
- The complete 60-query/method CPU benchmark was executed successfully against
  the local corpus and model weights; see [results](docs/evaluation-results.md).
  Colab GPU performance is not measured by that run.
- Mandatory function annotations and Ruff checks are described in
  [CLAUDE.md](CLAUDE.md). Unit tests run offline without the Harvey corpus.

## Troubleshooting and artifact reuse

- `Corpus changed`: re-run `legal-rag index` after parsing. Annotation/documentation
  changes alone do not invalidate the chunk fingerprint or require re-indexing.
- `chunks exceed 512 tokens`: parse with `--token-aware --max-tokens 480`, then
  rebuild the index. The word-count splitter can leave oversized paragraphs.
- No `.docx` files: check the exact `--input` directory; the parser reads files
  directly in that directory, not recursively.
- Missing models/network: first use downloads weights. If already cached, use
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for an offline run. The
  `--validate-only` benchmark option needs no model loading.
- Slow CPU reranking: use a small thread count as in the benchmark (`--threads 4`)
  or `--no-rerank` for the agent runner. Do not compare its speed/quality with a
  reranked run without recording the changed configuration.
- Evaluation output is written to `artifacts/evaluation/`; use `--save-dir` for
  a new experiment. Source and benchmark hashes in historical CPU v1 identify
  the original run, even when current code hashes later change.

Keep API keys outside source and traces. `.env` is ignored but no automatic
dotenv loading exists: export `ANTHROPIC_API_KEY` in the shell (or set it in the
Colab kernel from Secrets). Runtime choice and authentication are covered in the
[integration guide](docs/claude-integration.md).

## Claude agent and answer review

```bash
python -m pip install -e '.[dev,claude]'
export ANTHROPIC_API_KEY=...
python scripts/run_claude_agent.py --benchmark eval/benchmark.json --dry-run      # no API call
python scripts/run_claude_agent.py "<question>" --device cpu
python scripts/evaluate_retrieval.py --device cpu --claude-models claude-haiku-5-5+claude-sonnet-5-5 \
  --save-dir eval/results/<new-run>
python scripts/build_review_packets.py --results eval/results/<new-run>/results.json   # blind packets
python scripts/check_citations.py eval/results/<new-run>/results.json --dry-run        # then without --dry-run
```

`SEARCH+ANSWER` model syntax uses one model for search and another for the final answer. These
commands spend credits except `--dry-run` and `--no-judge`. See [evaluation.md](docs/evaluation.md),
[EVALUATION_PROMPT.md](eval/EVALUATION_PROMPT.md) and [measured results](docs/evaluation-results.md).
