"""Label-isolated, evidence-oriented retrieval evaluation (no model downloads)."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING

from .types import PathInput, Record

if TYPE_CHECKING:
    from .claude_agent import ClaudeEvidenceAgent
    from .retrieval import LegalRetriever

METHODS = ("dense", "bm25", "hybrid", "hybrid_rerank", "agentic_v2")
CLAUDE_PREFIX = "claude:"  # method name "claude:<model id>"; needs an agent passed to run_benchmark


def evidence_key(item: Record) -> tuple[str, str]:
    """Keep document and exhibit namespaces; never match by a bare suffix."""
    ref = item.get("legal_ref")
    if not ref:
        ref = f"{item.get('section_kind', 'section')}:{item.get('section_id', '')}"
        region = item.get("document_region")
        if region and region != "Main Agreement":
            ref = f"{region} / {ref}"
    return item["source"], ref


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def matches_label(item: Record, label: Record) -> bool:
    return (evidence_key(item) == (label["source"], label["legal_ref"])
            and _normalized(label["contains"]) in _normalized(item.get("text", "")))


def load_benchmark(path: PathInput) -> Record:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not payload.get("cases"):
        raise ValueError("Expected benchmark schema_version=1 and nonempty cases")
    seen = set()
    for case in payload["cases"]:
        if not case.get("id") or case["id"] in seen:
            raise ValueError("Case IDs must be nonempty and unique")
        seen.add(case["id"])
        if not case.get("question", "").strip() or case.get("split") not in {"dev", "eval"}:
            raise ValueError(f"Invalid question or split: {case['id']}")
        required = case.get("required_evidence", [])
        if not required or len({x.get('id') for x in required}) != len(required):
            raise ValueError(f"Required evidence must be nonempty with unique IDs: {case['id']}")
        for group in required:
            if not group.get("id") or not group.get("role") or not group.get("any_of"):
                raise ValueError(f"Invalid evidence group: {case['id']}")
        labels = [label for group in required for label in group["any_of"]]
        labels += case.get("supporting_evidence", [])
        for label in labels:
            if not all(isinstance(label.get(k), str) and label[k].strip()
                       for k in ("source", "legal_ref", "contains")):
                raise ValueError(f"Evidence labels need source, legal_ref, contains: {case['id']}")
    return payload


def validate_benchmark(benchmark: Record, chunks: list[Record]) -> None:
    """Reject labels absent from this corpus, including obsolete parser refs."""
    by_key = defaultdict(list)
    for chunk in chunks:
        by_key[evidence_key(chunk)].append(chunk)
    errors = []
    for case in benchmark["cases"]:
        labels = [label for group in case["required_evidence"] for label in group["any_of"]]
        labels += case.get("supporting_evidence", [])
        for label in labels:
            candidates = by_key[(label["source"], label["legal_ref"])]
            if not any(matches_label(c, label) for c in candidates):
                errors.append(f"{case['id']}: missing {label['source']} {label['legal_ref']} / {label['contains']!r}")
    if errors:
        raise ValueError("Benchmark does not match indexed chunks:\n" + "\n".join(errors))


def collapse_evidence(items: list[Record]) -> list[Record]:
    """One budget slot per cited section; join only text actually returned."""
    output, seen = [], {}
    for item in items:
        key = evidence_key(item)
        if key not in seen:
            row = {**item, "retrieved_chunks": []}
            seen[key] = row
            output.append(row)
        row = seen[key]
        if not any(c["text"] == item.get("text", "") for c in row["retrieved_chunks"]):
            row["retrieved_chunks"].append({
                "chunk_id": item.get("chunk_id"), "text": item.get("text", ""),
                "score": item.get("score"),
                "discovery_method": item.get("retrieval_method", "ranked_retrieval"),
            })
            row["text"] = "\n".join(c["text"] for c in row["retrieved_chunks"])
    return output


def agent_candidates(state: Record, chunks: list[Record]) -> list[Record]:
    """Label-blind round robin across facets, preserving each facet's order.

    Structural lookup scores are not ranks and must not be sorted with logits.
    Restore metadata dropped by the v2 report, without retrieving extra text.
    """
    streams = [state["evidence"].get(f, []) for f in state["facets"]]
    items = []
    for batch in zip_longest(*streams):
        for item in batch:
            if item is not None:
                items.append({**chunks[item["idx"]], **item})
    return collapse_evidence(items)


def score_evidence(case: Record, items: list[Record]) -> Record:
    groups = case["required_evidence"]
    found = [g["id"] for g in groups
             if any(matches_label(item, label) for label in g["any_of"] for item in items)]
    return {
        "evidence_set_recall": len(found) / len(groups),
        "complete_evidence": len(found) == len(groups),
        "found": found, "missing": [g["id"] for g in groups if g["id"] not in found],
        "returned_sections": len(items),
        "returned_words": sum(len(item.get("text", "").split()) for item in items),
    }


def claude_candidates(report: Record, chunks: list[Record]) -> tuple[list[Record], list[Record]]:
    """Order Claude's evidence without labels: cited primary, cited supporting, then uncited by discovery.

    Returns (ordered pool, cited sections only). Lookup items keep their null score.
    """
    by_id = {c.get("chunk_id"): c for c in chunks}
    ledger = {e["evidence_id"]: e for e in report["evidence"]}

    def restore(evidence_id: str) -> Record:
        e = ledger[evidence_id]
        return {**by_id.get(e["chunk_id"], {}), **e, "score": e["retrieval_score"],
                "retrieval_method": e["discovery_method"]}

    refs = [r for claim in report["answer"]["claims"] for r in claim["evidence"]]
    order = [r["evidence_id"] for role in ("primary", "supporting") for r in refs if r["role"] == role]
    cited = list(dict.fromkeys(order))
    rest = [e for e in ledger if e not in cited]
    return (collapse_evidence([restore(e) for e in cited + rest]),
            collapse_evidence([restore(e) for e in cited]))


def retrieve(retriever: LegalRetriever, question: str, method: str, *, pool_k: int = 20, max_rounds: int = 2,
             claude_agents: Mapping[str, ClaudeEvidenceAgent] | None = None) -> tuple[list[Record], Record]:
    """Retrieval boundary: accepts only the question, never labels or gold source."""
    if method.startswith(CLAUDE_PREFIX):
        from .citation_check import (
            cited_claims,  # lazy: pulls in the Claude agent module
        )
        report = (claude_agents or {})[method].run(question)
        pool, cited = claude_candidates(report, retriever.chunks)
        return pool[:pool_k], {
            "cited_sections": [list(evidence_key(c)) for c in cited], "_cited_items": cited,
            "conclusion_type": report["answer"]["conclusion_type"], "usage": report["usage"],
            "evidence_gaps": report["answer"]["evidence_gaps"], "trace": report["trace"],
            "validation_problems": report["validation_problems"], "answer": report["answer"]["answer"],
            "model": report["model"], "config_hash": report["config_hash"],
            "cited_claims": cited_claims(report),  # claims + cited passage text, for scripts/check_citations.py
        }
    if method == "agentic_v2":
        from .evidence_agent import build_agent
        state = build_agent(retriever, max_rounds=max_rounds).invoke(
            {"question": question, "source": None})
        return agent_candidates(state, retriever.chunks)[:pool_k], {
            "trace": state["trace"], "source": state["source"],
            "missing_facets": state["missing"], "rounds": state["rounds"],
        }
    return collapse_evidence(retriever.search(question, method=method, top_k=pool_k)), {}


def _percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def _usage_summary(rows: list[Record]) -> Record:
    used = [r["usage"] for r in rows if r.get("usage")]
    if not used:
        return {}
    costs = [u["cost_usd"] for u in used if u["cost_usd"] is not None]
    cited = [r["cited_metrics"] for r in rows if r.get("cited_metrics")]
    return {"mean_cost_usd": statistics.mean(costs) if costs else None, "total_cost_usd": sum(costs) if costs else None,
            "mean_tokens": statistics.mean(u["input_tokens"] + u["output_tokens"] for u in used),
            "mean_tool_calls": statistics.mean(u["tool_calls"] for u in used),
            "cited_esr": statistics.mean(c["evidence_set_recall"] for c in cited) if cited else None,
            "cited_cer": statistics.mean(c["complete_evidence"] for c in cited) if cited else None,
            "cited_mean_sections": statistics.mean(c["returned_sections"] for c in cited) if cited else None}


def summarize(records: list[Record], cutoffs: Sequence[int]) -> Record:
    output = {}
    for method in dict.fromkeys(r["method"] for r in records):
        rows = [r for r in records if r["method"] == method]
        # Failed queries remain in metric denominators with empty evidence.
        output[method] = {
            "questions": len(rows), "failures": sum(r["error"] is not None for r in rows),
            "latency_ms_p50": statistics.median(r["latency_ms"] for r in rows),
            "latency_ms_p95": _percentile([r["latency_ms"] for r in rows], .95),
            **_usage_summary(rows),
            "metrics": {str(k): {
                "evidence_set_recall": statistics.mean(r["metrics"][str(k)]["evidence_set_recall"] for r in rows),
                "complete_evidence_rate": statistics.mean(r["metrics"][str(k)]["complete_evidence"] for r in rows),
                "mean_returned_sections": statistics.mean(r["metrics"][str(k)]["returned_sections"] for r in rows),
                "mean_returned_words": statistics.mean(r["metrics"][str(k)]["returned_words"] for r in rows),
            } for k in cutoffs},
        }
    return output


def run_benchmark(retriever: LegalRetriever, benchmark: Record, *, methods: Sequence[str] = METHODS, cutoffs: Sequence[int] = (5, 10), pool_k: int = 20, max_rounds: int = 2, progress: Callable[[Record], None] | None = None, claude_agents: Mapping[str, ClaudeEvidenceAgent] | None = None) -> Record:
    unknown = {m for m in methods if m not in METHODS and not (m.startswith(CLAUDE_PREFIX) and m in (claude_agents or {}))}
    if not methods or len(set(methods)) != len(methods) or unknown:
        raise ValueError(f"Choose unique supported methods; unknown: {sorted(unknown)}")
    if not cutoffs or any(k <= 0 for k in cutoffs) or pool_k < max(cutoffs):
        raise ValueError("Positive cutoffs must fit within pool_k")
    if not 0 <= max_rounds <= 3:
        raise ValueError("max_rounds must be between 0 and 3")
    validate_benchmark(benchmark, retriever.chunks)
    records = []
    for case in benchmark["cases"]:
        for method in methods:
            started = time.perf_counter()
            error, extra, items = None, {}, []
            try:
                extra_args: Record = {"claude_agents": claude_agents} if method.startswith(CLAUDE_PREFIX) else {}
                items, extra = retrieve(retriever, case["question"], method, pool_k=pool_k,
                                        max_rounds=max_rounds, **extra_args)
            except Exception as exc:  # noqa: BLE001 -- isolate failures and keep them in metric denominators
                error = f"{type(exc).__name__}: {exc}"
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            cited_items = extra.pop("_cited_items", None)
            if cited_items is not None:
                extra["cited_metrics"] = score_evidence(case, cited_items)
            record = {
                "case_id": case["id"], "split": case["split"], "question": case["question"],
                "method": method, "latency_ms": elapsed, "error": error,
                "metrics": {str(k): score_evidence(case, items[:k]) for k in cutoffs},
                "pool_metrics": score_evidence(case, items),
                "results": [{"rank": n, "source": evidence_key(item)[0],
                             "legal_ref": evidence_key(item)[1],
                             "chunk_ids": [c["chunk_id"] for c in item["retrieved_chunks"]]}
                            for n, item in enumerate(items, 1)],
                **extra,
            }
            records.append(record)
            if progress:
                progress(record)
    return {
        "summary": summarize(records, cutoffs),
        "by_split": {split: summarize([r for r in records if r["split"] == split], cutoffs)
                     for split in sorted({r["split"] for r in records})},
        "records": records,
    }


def file_sha256(path: PathInput) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_markdown(report: Record) -> str:
    metadata = report["metadata"]
    lines = ["# Retrieval benchmark", "", f"Run: {metadata['created_at']}.", "",
             "Silver labels; retrieval coverage only. Complete evidence does not establish a legal conclusion.", "",
             "Equal section cutoffs; agent rounds can use more searches. Word counts expose differing text budgets.", "",
             f"Device: `{metadata['device']}`; corpus SHA-256: `{metadata['corpus_sha256']}`.", "",
             "| Method | ESR@5 | CER@5 | ESR@10 | CER@10 | p50 ms | p95 ms | Failures |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    # The CLI defaults to 5 and 10; render arbitrary cutoffs as well.
    cutoffs = metadata["cutoffs"]
    lines[-2] = "| Method | " + " | ".join(f"ESR@{k} | CER@{k}" for k in cutoffs) + " | p50 ms | p95 ms | Failures |"
    lines[-1] = "|---|" + "---:|" * (2 * len(cutoffs) + 3)
    for method, summary in report["summary"].items():
        metrics = " | ".join(f"{summary['metrics'][str(k)]['evidence_set_recall']:.1%} | "
                             f"{summary['metrics'][str(k)]['complete_evidence_rate']:.1%}" for k in cutoffs)
        lines.append(f"| {method} | {metrics} | {summary['latency_ms_p50']:.0f} | "
                     f"{summary['latency_ms_p95']:.0f} | {summary['failures']} |")
    claude = {m: v for m, v in report["summary"].items() if "mean_cost_usd" in v}
    if claude:
        lines += ["", "## Claude agents", "",
                  "Pool = all retrieved evidence (cited first); Cited = only sections the answer cites. "
                  "Cost is an upper estimate from token usage.", "",
                  "| Method | Cited ESR | Cited CER | Cited sections | Tool calls | Tokens | Cost/q | Total cost |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for method, v in claude.items():
            cost = "n/a" if v["mean_cost_usd"] is None else f"${v['mean_cost_usd']:.4f}"
            total = "n/a" if v["total_cost_usd"] is None else f"${v['total_cost_usd']:.3f}"
            lines.append(f"| {method} | {v['cited_esr']:.1%} | {v['cited_cer']:.1%} | {v['cited_mean_sections']:.1f} | "
                         f"{v['mean_tool_calls']:.1f} | {v['mean_tokens']:.0f} | {cost} | {total} |")
    lines += ["", "## Incomplete evidence", "",
              "| Question | Method | Cutoff | Missing required evidence |", "|---|---|---:|---|"]
    for row in report["records"]:
        for k in cutoffs:
            missing = row["metrics"][str(k)]["missing"]
            if missing:
                lines.append(f"| {row['case_id']} | {row['method']} | {k} | {', '.join(missing)} |")
    lines += ["", "Full per-query metrics, pool coverage, split summaries and agent traces are in results.json.", ""]
    return "\n".join(lines)
