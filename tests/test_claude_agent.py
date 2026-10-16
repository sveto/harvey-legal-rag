from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pytest import MonkeyPatch

import legal_rag.claude_agent as ca
from legal_rag.claude_agent import (
    ClaudeAgentConfig,
    ClaudeEvidenceAgent,
    CredentialsError,
    make_client,
)
from legal_rag.claude_schemas import SchemaError, validate_answer, validate_plan
from legal_rag.types import Record

SRC = "terranode-isa.docx"
PLAN = {"issues": [{"name": "consent", "description": "assignment consent", "queries": ["assignment consent"]}],
        "relevant_sources": [SRC, "not-in-corpus.docx"], "notes": ""}


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_block(call_id: str, name: str, args: Record) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=args)


def reply(blocks: list[SimpleNamespace], stop: str = "end_turn", tokens: tuple[int, int] = (1000, 200)) -> SimpleNamespace:
    return SimpleNamespace(content=blocks, stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]))


class FakeClient:
    """Replays scripted responses and records every request."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = list(responses)
        self.requests: list[Record] = []
        self.messages = self

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


class FakeRetriever:
    chunks: list[Record] = [
        {"source": SRC, "legal_ref": "9.1", "section_title": "Assignment", "chunk_id": "c1",
         "text": "Assignment requires prior written consent. See Section 9.2.", "document_region": "Main Agreement"},
        {"source": SRC, "legal_ref": "9.2", "section_title": "Competitor Carve-Out", "chunk_id": "c2",
         "text": "Consent may be withheld for a direct competitor.", "document_region": "Main Agreement"},
    ]


def hit(i: int, score: float) -> Record:
    c = FakeRetriever.chunks[i]
    return {**c, "score": score}


ANSWER = {"conclusion_type": "conditional_right", "answer": "Conditional on a competitor acquirer.",
          "claims": [{"statement": "Consent is required.", "evidence": [{"evidence_id": "E1", "role": "primary"},
                                                                          {"evidence_id": "E99", "role": "supporting"}]},
                     {"statement": "Uncited claim.", "evidence": [{"evidence_id": "E42", "role": "primary"}]}],
          "conditions": ["consent"], "unresolved_facts": ["is the acquirer a competitor"], "evidence_gaps": []}


def make_agent(client: FakeClient, monkeypatch: MonkeyPatch, **cfg: Any) -> ClaudeEvidenceAgent:
    monkeypatch.setattr(ca, "scoped_search", lambda r, q, s, top_k=8: [hit(0, 3.2)])
    return ClaudeEvidenceAgent(FakeRetriever(), client, ClaudeAgentConfig(**cfg))  # type: ignore[arg-type]


def test_full_run_validates_citations_and_records_provenance(monkeypatch: MonkeyPatch) -> None:
    client = FakeClient([
        reply([text_block(json.dumps(PLAN))]),
        reply([text_block("Need the consent clause."), tool_block("t1", "search_corpus", {"query": "consent", "source": SRC})], "tool_use"),
        reply([text_block("Follow the reference."), tool_block("t2", "lookup_reference", {"source": SRC, "reference": "9.2"})], "tool_use"),
        reply([text_block("Competitor facts still unknown.")]),
        reply([text_block(json.dumps(ANSWER))]),
    ])
    report = make_agent(client, monkeypatch).run("Does TerraNode have termination rights?")
    assert report["plan"]["relevant_sources"] == [SRC]
    items = {e["evidence_id"]: e for e in report["evidence"]}
    assert items["E1"]["discovery_method"] == "scoped_search" and items["E1"]["retrieval_score"] == 3.2
    assert items["E2"]["discovery_method"] == "cross_reference" and items["E2"]["retrieval_score"] is None
    assert [c["evidence_id"] for c in report["citations"]] == ["E1"]
    assert len(report["answer"]["claims"]) == 1 and report["validation_problems"]
    assert report["usage"]["api_calls"] == 5 and report["usage"]["tool_calls"] == 2
    assert [s["step"] for s in report["trace"] if s["step"] == "decision"] == ["decision", "decision"]
    # tool output is wrapped as untrusted data and the key is nowhere in the report
    tool_msg = client.requests[2]["messages"][-1]["content"][0]["content"]
    assert tool_msg.startswith("<contract_text>")
    assert "ANTHROPIC_API_KEY" not in json.dumps(report, default=str)


def test_requests_use_structured_output_and_no_forced_tool_choice(monkeypatch: MonkeyPatch) -> None:
    client = FakeClient([reply([text_block(json.dumps(PLAN))]), reply([text_block("done")])])
    make_agent(client, monkeypatch).run("q")
    plan_req, inv_req = client.requests[0], client.requests[1]
    assert plan_req["output_config"]["format"]["type"] == "json_schema"
    assert "tool_choice" not in plan_req and "tool_choice" not in inv_req
    assert {t["name"] for t in inv_req["tools"]} == {"search_corpus", "lookup_reference"}


def test_tool_call_limit_stops_investigation(monkeypatch: MonkeyPatch) -> None:
    looping = [reply([tool_block(f"t{i}", "search_corpus", {"query": "x", "source": None})], "tool_use") for i in range(6)]
    client = FakeClient([reply([text_block(json.dumps(PLAN))]), *looping, reply([text_block(json.dumps(ANSWER))])])
    report = make_agent(client, monkeypatch, max_tool_calls=2).run("q")
    assert report["usage"]["tool_calls"] == 2
    assert any(s.get("stop") == "tool call limit" for s in report["trace"])
    assert any("tool call limit" in g for g in report["answer"]["evidence_gaps"])


def test_turn_limit_and_gap_reported(monkeypatch: MonkeyPatch) -> None:
    looping = [reply([tool_block(f"t{i}", "search_corpus", {"query": "x", "source": None})], "tool_use") for i in range(2)]
    client = FakeClient([reply([text_block(json.dumps(PLAN))]), *looping, reply([text_block(json.dumps(ANSWER))])])
    report = make_agent(client, monkeypatch, max_turns=2).run("q")
    assert any(s.get("stop") == "turn limit" for s in report["trace"])


def test_budget_exhaustion_skips_answer_call(monkeypatch: MonkeyPatch) -> None:
    client = FakeClient([reply([text_block(json.dumps(PLAN))], tokens=(90_000, 70_000))])
    report = make_agent(client, monkeypatch, max_total_tokens=100_000).run("q")
    assert client.requests and len(client.requests) == 1
    assert report["answer"]["conclusion_type"] == "insufficient_evidence"
    assert report["usage"]["cost_usd"] is not None


def test_no_evidence_means_insufficient_without_answer_call(monkeypatch: MonkeyPatch) -> None:
    client = FakeClient([reply([text_block(json.dumps(PLAN))]), reply([text_block("nothing to search")])])
    report = make_agent(client, monkeypatch).run("q")
    assert len(client.requests) == 2
    assert report["answer"]["conclusion_type"] == "insufficient_evidence"


def test_unusable_answer_degrades_to_insufficient(monkeypatch: MonkeyPatch) -> None:
    client = FakeClient([
        reply([text_block(json.dumps(PLAN))]),
        reply([tool_block("t1", "search_corpus", {"query": "c", "source": None})], "tool_use"),
        reply([text_block("ok")]),
        reply([text_block("not json")]),
    ])
    report = make_agent(client, monkeypatch).run("q")
    assert report["answer"]["conclusion_type"] == "insufficient_evidence"
    assert report["validation_problems"]


def test_tool_arguments_are_validated(monkeypatch: MonkeyPatch) -> None:
    agent = make_agent(FakeClient([]), monkeypatch)
    state: Any = {"ledger": {}, "usage": ca.new_usage(), "trace": []}
    assert agent.run_tool(state, "search_corpus", {"query": "x", "source": "other.docx"})[1]
    assert agent.run_tool(state, "search_corpus", {"query": "", "source": None})[1]
    assert agent.run_tool(state, "lookup_reference", {"source": "other.docx", "reference": "9.1"})[1]
    assert agent.run_tool(state, "run_shell", {"cmd": "ls"})[1]
    assert state["ledger"] == {}


def test_validate_plan_requires_issues() -> None:
    with pytest.raises(SchemaError):
        validate_plan({"issues": [], "relevant_sources": [], "notes": ""}, [])


def test_validate_answer_downgrades_when_nothing_is_cited() -> None:
    bad = {**ANSWER, "claims": [{"statement": "x", "evidence": [{"evidence_id": "E7", "role": "primary"}]}]}
    cleaned, problems = validate_answer(bad, {})
    assert cleaned["conclusion_type"] == "insufficient_evidence" and problems


def test_missing_key_raises_without_importing_sdk(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(CredentialsError):
        make_client(ClaudeAgentConfig())


def test_claude_candidates_order_cited_primary_first_without_labels() -> None:
    from legal_rag.evaluation import claude_candidates
    chunks = [{"chunk_id": f"c{i}", "source": SRC, "legal_ref": f"9.{i}", "text": f"t{i}"} for i in range(1, 4)]
    def ev(i: int, method: str, score: float | None) -> Record:
        return {"evidence_id": f"E{i}", "chunk_id": f"c{i}", "source": SRC, "legal_ref": f"9.{i}", "section_title": None,
                "text": f"t{i}", "retrieval_score": score, "discovery_method": method}
    report = {"evidence": [ev(1, "scoped_search", 1.0), ev(2, "cross_reference", None), ev(3, "scoped_search", 0.5)],
              "answer": {"claims": [{"statement": "s", "evidence": [{"evidence_id": "E3", "role": "supporting"},
                                                                     {"evidence_id": "E2", "role": "primary"}]}]}}
    pool, cited = claude_candidates(report, chunks)
    assert [p["legal_ref"] for p in pool] == ["9.2", "9.3", "9.1"]
    assert [c["legal_ref"] for c in cited] == ["9.2", "9.3"]
    assert pool[0]["retrieved_chunks"][0]["discovery_method"] == "cross_reference"


def test_header_echoed_as_evidence_id_is_normalized() -> None:
    ledger: Any = {"E6": {"evidence_id": "E6"}}
    obj = {**ANSWER, "claims": [{"statement": "s", "evidence": [
        {"evidence_id": "E6 source=x.docx ref=7.09(a)", "role": "primary"}]}]}
    cleaned, problems = validate_answer(obj, ledger)
    assert cleaned["claims"][0]["evidence"][0]["evidence_id"] == "E6" and not problems


def test_separate_answer_model_is_used_and_costed_per_model(monkeypatch: MonkeyPatch) -> None:
    client = FakeClient([
        reply([text_block(json.dumps(PLAN))]),
        reply([tool_block("t1", "search_corpus", {"query": "c", "source": None})], "tool_use"),
        reply([text_block("ok")]),
        reply([text_block(json.dumps(ANSWER))], tokens=(1_000_000, 0)),
    ])
    agent = make_agent(client, monkeypatch, model="claude-haiku-5-5", answer_model="claude-sonnet-5-5")
    report = agent.run("q")
    assert [r["model"] for r in client.requests] == ["claude-haiku-5-5"] * 3 + ["claude-sonnet-5-5"]
    assert report["model"] == "claude-haiku-5-5+claude-sonnet-5-5"
    assert set(report["usage"]["by_model"]) == {"claude-haiku-5-5", "claude-sonnet-5-5"}
    assert report["usage"]["cost_usd"] > 2.0  # 1M Sonnet input tokens at $2/M dominates


def test_review_packets_are_blind_shuffled_and_keyed() -> None:
    from legal_rag.review import build_packets, render_packet
    bench = {"cases": [{"id": "a", "question": "Qa?", "answer_notes": "Na"}, {"id": "b", "question": "Qb?", "answer_notes": "Nb"}]}
    recs = [{"case_id": c, "method": m, "answer": f"{m}-{c}", "error": None}
            for c in "ab" for m in ("claude:haiku", "claude:sonnet", "dense")]
    recs.append({"case_id": "a", "method": "claude:opus", "answer": "", "error": "boom"})
    results = {"records": recs}
    packets, key = build_packets(results, bench, ["claude:haiku", "claude:sonnet", "claude:opus"], seed=1)
    assert len(packets) == 4 and sorted(key) == ["P01", "P02", "P03", "P04"]
    assert all("claude" not in render_packet(p) or "claude:" not in render_packet(p).split("Candidate answer:")[0] for p in packets)
    assert {v["method"] for v in key.values()} == {"claude:haiku", "claude:sonnet"}
    assert packets == build_packets(results, bench, ["claude:haiku", "claude:sonnet", "claude:opus"], seed=1)[0]


def _claim(statement: str, text: str, ref: str = "5.5") -> Record:
    return {"statement": statement, "citations": [{"evidence_id": "E1", "role": "primary", "source": SRC,
                                                    "legal_ref": ref, "section_title": "t", "text": text}]}


def test_cited_claims_attach_passage_text() -> None:
    from legal_rag.citation_check import cited_claims
    report = {"evidence": [{"evidence_id": "E1", "source": SRC, "legal_ref": "5.5", "section_title": "T", "text": "body"}],
              "answer": {"claims": [{"statement": "s", "evidence": [{"evidence_id": "E1", "role": "primary"},
                                                                     {"evidence_id": "E9", "role": "supporting"}]}]}}
    out = cited_claims(report)
    assert out[0]["citations"][0]["text"] == "body" and len(out[0]["citations"]) == 1


def test_numeric_check_catches_invented_numbers_and_accepts_words() -> None:
    from legal_rag.citation_check import numeric_check
    text = "Either party may terminate on thirty (30) days written notice; cure within sixty days. See Section 9.3."
    ok = numeric_check(_claim("Termination needs 30 days notice and a 60-day cure under Section 9.3.", text))
    assert not ok["flagged"]
    bad = numeric_check(_claim("Termination needs 45 days notice under Section 9.9.", text))
    assert bad["missing_numbers"] == ["45"] and bad["missing_section_refs"] == ["9.9"] and bad["flagged"]
    assert numeric_check({"statement": "x", "citations": []})["flagged"]


def test_judge_parses_verdict_and_degrades_on_bad_output() -> None:
    from legal_rag.citation_check import judge_claim, summarize_checks
    good = reply([text_block(json.dumps({"verdict": "partial", "reason": "cure period missing"}))], tokens=(2000, 50))
    bad = reply([text_block(json.dumps({"verdict": "maybe", "reason": "?"}))])
    client = FakeClient([good, bad])
    a = judge_claim(client, "claude-haiku-5-5", _claim("s", "t"))  # type: ignore[arg-type]
    b = judge_claim(client, "claude-haiku-5-5", _claim("s", "t"))  # type: ignore[arg-type]
    assert a["verdict"] == "partial" and a["cost_usd"] > 0 and b["verdict"] == "error"
    assert "tool_choice" not in client.requests[0] and client.requests[0]["output_config"]["format"]["type"] == "json_schema"
    rows = [{"method": "m", "numeric": {"flagged": True}, **a}, {"method": "m", "numeric": {"flagged": False}, **b}]
    s = summarize_checks(rows)["m"]
    assert s["claims"] == 2 and s["partial"] == 0.5 and s["judge_errors"] == 1 and s["numeric_flagged"] == 0.5


def test_scope_report_counts_citations_outside_own_labeled_files() -> None:
    from legal_rag.citation_check import scope_report
    bench = {"cases": [
        {"id": "a", "required_evidence": [{"any_of": [{"source": "x.docx"}]}], "supporting_evidence": []},
        {"id": "b", "required_evidence": [{"any_of": [{"source": "y.docx"}]}], "supporting_evidence": []}]}
    rec = lambda case, cited: {"case_id": case, "method": "claude:m", "cited_sections": cited}  # noqa: E731
    results = {"records": [rec("a", [["x.docx", "1"], ["memo.docx", "I"], ["y.docx", "2"]]), rec("b", [["y.docx", "3"]]),
                           {"case_id": "a", "method": "dense"}]}
    out = scope_report(results, bench)["claude:m"]
    assert out["cited"] == 4 and out["unlabeled_files"] == 1 and out["off_gold"] == 2
    assert out["off_gold_share"] == 0.5 and out["off_gold_items"][1]["source"] == "y.docx"


def test_off_gold_passages_dedupe_and_relevance_judge() -> None:
    from legal_rag.citation_check import (
        judge_relevance,
        off_gold_passages,
        summarize_relevance,
    )
    p = {"source": "memo.docx", "legal_ref": "I", "section_title": "t", "text": "closing Oct 15", "evidence_id": "E1", "role": "supporting"}
    record = {"cited_claims": [{"statement": "s", "citations": [p, dict(p)]},
                               {"statement": "t", "citations": [{**p, "source": "x.docx"}]}]}
    passages = off_gold_passages(record, {"x.docx"})
    assert len(passages) == 1 and passages[0]["source"] == "memo.docx"
    client = FakeClient([reply([text_block(json.dumps({"verdict": "useful_context", "reason": "deal date"}))]),
                         reply([text_block("nope")])])
    ok = judge_relevance(client, "claude-haiku-5-5", "Q?", passages[0])  # type: ignore[arg-type]
    bad = judge_relevance(client, "claude-haiku-5-5", "Q?", passages[0])  # type: ignore[arg-type]
    assert ok["verdict"] == "useful_context" and bad["verdict"] == "error"
    summary = summarize_relevance([{"method": "m", **ok}, {"method": "m", **bad}])["m"]
    assert summary["useful_context"] == 1 and summary["error"] == 1
