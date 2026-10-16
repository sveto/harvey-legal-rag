# Minimal Colab runner

These are notebook cell instructions, not a generated `.ipynb` artifact. In a
free Colab runtime, select a T4/GPU when available. Clone the source repository
or upload/extract it to `/content/harvey-legal-rag`, then execute these cells:

```python
%cd /content/harvey-legal-rag
%pip -q install -e '.[dev]'
```

Confirm that the runtime actually provides CUDA:

```python
import torch
print(torch.cuda.is_available())
```

If CUDA is unavailable, change `--device cuda` to `--device cpu`. Download the
DOCX files if they are not already present:

```python
!python scripts/download_corpus.py --output /content/harvey_coc
```

```python
!legal-rag parse --input /content/harvey_coc --output /content/harvey_index --token-aware --max-tokens 480
!pytest -q
!ruff check src scripts tests
!legal-rag index --output /content/harvey_index --device cuda --batch-size 32
```

Start with batch size 32; if GPU memory is insufficient for indexing, retry with
8. The FAISS index stays on CPU while the encoder/reranker use the selected
device. Reuse the valid index and Hugging Face cache between experiments rather
than re-embedding for every question. A Colab runtime is ephemeral: retain source
DOCX, parsed artifacts, index configuration and reports in Drive if needed.

```python
!python scripts/check_boundaries.py --output /content/harvey_index
!python scripts/compare_methods.py --output /content/harvey_index --device cuda
```

```python
!legal-rag diagnose 'Does TerraNode have termination rights if Aldersgate is acquired by a competitor?' --source terranode-isa.docx --section 9.2 --output /content/harvey_index --device cuda
```

Evaluate the same five methods, including Agent v2, on the evidence benchmark:

```python
!python scripts/evaluate_retrieval.py --artifacts /content/harvey_index --validate-only
!python scripts/evaluate_retrieval.py --artifacts /content/harvey_index --device cuda --save-dir /content/harvey_evaluation
```

Inspect `/content/harvey_evaluation/report.md` and save that directory to Drive
before the runtime expires. `results.json` includes per-question missing evidence,
split summaries, configuration and agent traces; `queries.jsonl` is flushed while
the experiment runs. These 12 silver-labeled questions measure retrieval coverage,
not answer correctness. Keep `eval/benchmark.json` outside the DOCX corpus.

## Claude agent in Colab

Install the extra and keep the key in Colab Secrets (key icon in the left sidebar: add
`ANTHROPIC_API_KEY` and switch on "Notebook access"). Set it in the *notebook kernel*, so
that `!python ...` commands inherit it, and never print it:

```python
%pip -q install -e '.[dev,claude]'
import os
from google.colab import userdata
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
```

```python
!python scripts/run_claude_agent.py --benchmark eval/benchmark.json --dry-run
!python scripts/run_claude_agent.py --benchmark eval/benchmark.json --limit 1 --output /content/harvey_index --device cuda
!python scripts/evaluate_retrieval.py --artifacts /content/harvey_index --device cuda \
  --claude-models claude-haiku-5-5 claude-sonnet-5-5 --save-dir /content/harvey_claude_eval
```

The GPU is used only by the local embedding model and reranker; Claude calls are remote, so
total time is dominated by API latency (about one minute per question). Copy the results
directory to Drive before the runtime expires. Do not share a notebook with output that
echoes the key, and do not commit it. The recorded benchmarks in this repository ran on CPU;
no T4 timings are claimed. Every new notebook-exported Python function should have parameter
and return annotations, matching [CLAUDE.md](../CLAUDE.md).
