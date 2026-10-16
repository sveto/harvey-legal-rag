from __future__ import annotations

from pytest import MonkeyPatch

from legal_rag.evidence_agent import (
    build_agent,
    infer_facets,
    infer_source,
    supports_facet,
)
from legal_rag.types import Record


def test_terranode_facets() -> None:
    assert infer_facets("Does TerraNode have termination rights if Aldersgate is acquired by a competitor?") == ["consent", "competitor", "termination", "cure"]


def test_source_is_not_guessed_without_name() -> None:
    chunks = [{"source": "terranode-isa.docx"}, {"source": "pinnacle-license.docx"}]
    assert infer_source("Does TerraNode have termination rights?", chunks) == "terranode-isa.docx"
    assert infer_source("What happens upon change of control?", chunks) is None


def test_facet_lexical_evidence_not_legal_conclusion() -> None:
    assert supports_facet({"section_title": "Termination", "text": "No right to terminate exists."}, "termination")
    assert not supports_facet({"section_title": "Services", "text": "Provider hosts servers."}, "termination")


def test_graph_terminates_when_no_evidence(monkeypatch: MonkeyPatch) -> None:
    import legal_rag.evidence_agent as ea
    class Fake:
        chunks = [{"source": "terranode-isa.docx"}]
    monkeypatch.setattr(ea, "scoped_search", lambda *a, **kw: [])
    result = build_agent(Fake(), max_rounds=2).invoke({"question": "Does TerraNode have termination rights?", "source": None})
    assert result["rounds"] == 2
    assert result["missing"] == ["termination", "cure"]
    assert len([x for x in result["trace"] if x["step"] == "targeted"]) == 4


def test_v2_excludes_unrelated_sla_termination() -> None:
    from legal_rag.evidence_agent import role_matches
    item = {"legal_ref": "Exhibit B / 4", "section_title": "Chronic Failure",
            "text": "Customer shall have the right to terminate this Agreement for availability failures. Cure period applies."}
    q = "Does TerraNode have termination rights if Aldersgate is acquired by a competitor?"
    assert not role_matches(item, "termination", q)
    assert not role_matches(item, "cure", q)


def test_v2_reference_extraction_and_lookup() -> None:
    from legal_rag.evidence_agent import lookup_section, reference_targets
    item = {"legal_ref": "9.2", "text": "Notwithstanding Section 9.1, consent may be withheld."}
    assert reference_targets(item) == ["9.1"]
    class Fake:
        chunks = [{"source": "terranode-isa.docx", "legal_ref": "9.1", "section_title": "Assignment",
                   "text": "Prior written consent is required for assignment.", "chunk_id": "a", "parent_section_uid": "p",
                   "document_region": "Main Agreement"},
                  {"source": "other.docx", "legal_ref": "9.1", "section_title": "Assignment",
                   "text": "Unrelated", "chunk_id": "b", "parent_section_uid": "q"}]
    found = lookup_section(Fake(), "terranode-isa.docx", "9.1")
    assert len(found) == 1 and found[0]["legal_ref"] == "9.1"


def test_v2_graph_expands_reference_even_when_facets_covered(monkeypatch: MonkeyPatch) -> None:
    import legal_rag.evidence_agent as ea
    question = "Does TerraNode have termination rights if Aldersgate is acquired by a competitor?"
    def chunk(ref: str, title: str, text: str) -> Record:
        return {"source": "terranode-isa.docx", "legal_ref": ref, "section_title": title,
                "text": text, "chunk_id": ref, "parent_section_uid": ref,
                "document_region": "Main Agreement", "score": -1.0}
    source_chunks = [
        chunk("9.1", "General Restriction on Assignment", "Assignment requires prior written consent."),
        chunk("9.2", "Competitor Carve-Out", "Notwithstanding Section 9.1, consent to assignment may be withheld for a direct competitor."),
        chunk("5.5", "Termination Upon Unauthorized Assignment", "For unauthorized assignment, either party may terminate this Agreement."),
        chunk("9.3", "Cure Period", "Unauthorized assignment may be cured by written consent."),
    ]
    class Fake:
        chunks = source_chunks
    monkeypatch.setattr(ea, "scoped_search", lambda *a, **kw: source_chunks[1:])
    result = ea.build_agent(Fake()).invoke({"question": question, "source": "terranode-isa.docx"})
    assert any(x["step"] == "expand_reference" and x["reference"] == "9.1" for x in result["trace"])
    assert any(x["legal_ref"] == "9.1" for x in result["evidence"]["consent"])
    assert result["missing"] == []
