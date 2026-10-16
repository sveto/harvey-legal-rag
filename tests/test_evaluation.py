from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pytest import MonkeyPatch

import legal_rag.evaluation as ev
from legal_rag.types import Record


def label(source: str = "a.docx", ref: str = "9.1", contains: str = "consent required") -> Record:
    return {"source": source, "legal_ref": ref, "contains": contains}


def item(source: str = "a.docx", ref: str | None = "9.1", text: str = "consent required", **extra: object) -> Record:
    return {"source": source, "legal_ref": ref, "text": text, "chunk_id": ref, **extra}


def benchmark() -> Record:
    return {"schema_version": 1, "cases": [{
        "id": "q1", "question": "What consent is needed?", "split": "eval",
        "required_evidence": [
            {"id": "consent", "role": "consent", "any_of": [label()]},
            {"id": "cure", "role": "cure", "any_of": [label(ref="5.5", contains="cure period"),
                                                         label(ref="9.3", contains="cure period")]},
        ], "supporting_evidence": [label(ref="9.4", contains="notice period")],
    }]}


def test_exact_evidence_set_and_alternatives() -> None:
    case = benchmark()["cases"][0]
    partial = ev.score_evidence(case, [item(), item(), item(ref="9.4", text="notice period")])
    assert partial["evidence_set_recall"] == .5  # duplicates and supporting refs add no credit
    assert partial["missing"] == ["cure"]
    assert not partial["complete_evidence"]
    for ref in ("5.5", "9.3"):
        full = ev.score_evidence(case, [item(), item(ref=ref, text="cure period")])
        assert full["evidence_set_recall"] == 1
        assert full["complete_evidence"]


def test_matching_rejects_wrong_document_exhibit_parent_and_text() -> None:
    target = label(ref="7.09(b)", contains="automatic acceleration")
    for row in [item(ref="7.09"),
                item(source="b.docx", ref="7.09(b)", text="automatic acceleration"),
                item(ref="Exhibit A / 7.09(b)", text="automatic acceleration"),
                item(ref="7.09(b)", text="irrelevant child chunk")]:
        assert not ev.matches_label(row, target)
    assert ev.matches_label(item(ref="7.09(b)", text="AUTOMATIC\n acceleration"), target)


def test_unnumbered_article_and_partial_chunk_anchor() -> None:
    target = label(ref="article:1", contains="fifteen percent")
    assert ev.matches_label(item(ref=None, section_kind="article", section_id="1", text="fifteen percent"), target)
    assert not ev.matches_label(item(ref=None, section_kind="article", section_id="1", text="Other definition"), target)


def test_collapse_joins_only_retrieved_children() -> None:
    rows = ev.collapse_evidence([item(text="one"), item(text="two"), item(text="one")])
    assert len(rows) == 1
    assert rows[0]["text"] == "one\ntwo"
    assert len(rows[0]["retrieved_chunks"]) == 2


def test_agent_round_robin_ignores_incomparable_scores() -> None:
    chunks = [item(ref="9.2", text="competitor", idx=0, score=-4),
              item(ref="9.1", idx=1, score=0, retrieval_method="cross_reference"),
              item(ref="5.5", text="cure period", idx=2, score=-8)]
    state = {"facets": ["consent", "cure"], "evidence": {"consent": chunks[:2], "cure": [chunks[2]]}}
    rows = ev.agent_candidates(state, chunks)
    assert [r["legal_ref"] for r in rows] == ["9.2", "5.5", "9.1"]


def test_labels_validated_before_search() -> None:
    with pytest.raises(ValueError, match="missing a.docx"):
        ev.validate_benchmark(benchmark(), [item()])


def test_run_keeps_failures_in_denominator_and_isolates_labels(monkeypatch: MonkeyPatch) -> None:
    data = benchmark()
    second = copy.deepcopy(data["cases"][0])
    second.update(id="q2", question="Second question?")
    data["cases"].append(second)
    chunks = [item(), item(ref="9.3", text="cure period"), item(ref="9.4", text="notice period"),
              item(ref="5.5", text="cure period")]

    class Fake:
        def __init__(self) -> None:
            self.chunks = chunks

    calls = []
    def retrieve(retriever: object, question: str, method: str, *, pool_k: int, max_rounds: int) -> tuple[list[Record], Record]:
        calls.append((question, method))  # only question crosses the boundary, no oracle source/labels
        if question == "Second question?":
            raise RuntimeError("simulated backend failure")
        return ev.collapse_evidence(chunks[:2]), {}

    monkeypatch.setattr(ev, "retrieve", retrieve)
    result = ev.run_benchmark(Fake(), data, methods=["dense"], cutoffs=[1, 2], pool_k=2)
    assert calls == [("What consent is needed?", "dense"), ("Second question?", "dense")]
    summary = result["summary"]["dense"]
    assert summary["failures"] == 1
    assert summary["metrics"]["1"]["evidence_set_recall"] == .25
    assert summary["metrics"]["2"]["complete_evidence_rate"] == .5
    assert result["records"][1]["metrics"]["2"]["missing"] == ["consent", "cure"]


def test_agent_retrieval_never_receives_gold_source(monkeypatch: MonkeyPatch) -> None:
    import legal_rag.evidence_agent as ea
    calls = []
    class Graph:
        def invoke(self, state: Record) -> Record:
            calls.append(state)
            return {"facets": [], "evidence": {}, "trace": [], "source": None, "missing": [], "rounds": 0}
    class Fake:
        def __init__(self) -> None:
            self.chunks = []
    monkeypatch.setattr(ea, "build_agent", lambda *a, **kw: Graph())
    ev.retrieve(Fake(), "Compare two agreements", "agentic_v2")
    assert calls == [{"question": "Compare two agreements", "source": None}]


def test_shipped_benchmark_has_diverse_cases() -> None:
    data = ev.load_benchmark(Path(__file__).resolve().parents[1] / "eval/benchmark.json")
    assert 10 <= len(data["cases"]) <= 15
    sources = {x["source"] for c in data["cases"] for g in c["required_evidence"] for x in g["any_of"]}
    assert len(sources) >= 8
    assert {c["split"] for c in data["cases"]} == {"dev", "eval"}
