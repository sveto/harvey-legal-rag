"""Typed records, JSON schemas and validators for the Claude-backed evidence agent.

No Anthropic import here: schemas are plain data so they can be tested offline.
Model output is validated by hand before use; invalid output raises SchemaError.
"""
from __future__ import annotations

import re
from typing import Literal, TypedDict

from .types import Record

DiscoveryMethod = Literal["scoped_search", "cross_reference"]
EvidenceRole = Literal["primary", "supporting"]
ConclusionType = Literal["right_found", "conditional_right", "no_right_found", "insufficient_evidence"]

MAX_ISSUES = 6
MAX_QUERIES_PER_ISSUE = 3
CONCLUSIONS: tuple[str, ...] = ("right_found", "conditional_right", "no_right_found", "insufficient_evidence")
ROLES: tuple[str, ...] = ("primary", "supporting")


class SchemaError(ValueError):
    """Model output or tool arguments did not match the expected schema."""


class PlanIssue(TypedDict):
    name: str
    description: str
    queries: list[str]


class QuestionPlan(TypedDict):
    issues: list[PlanIssue]
    relevant_sources: list[str]
    notes: str


class EvidenceItem(TypedDict):
    """One retrieved passage. `retrieval_score` is None when nothing scored it."""
    evidence_id: str
    source: str
    legal_ref: str | None
    section_title: str | None
    text: str
    retrieval_score: float | None
    discovery_method: DiscoveryMethod
    chunk_id: str | None


class EvidenceRef(TypedDict):
    evidence_id: str
    role: EvidenceRole


class Claim(TypedDict):
    statement: str
    evidence: list[EvidenceRef]


class CitedAnswer(TypedDict):
    conclusion_type: ConclusionType
    answer: str
    claims: list[Claim]
    conditions: list[str]
    unresolved_facts: list[str]
    evidence_gaps: list[str]


def _strings(description: str) -> Record:
    return {"type": "array", "items": {"type": "string"}, "description": description}


PLAN_SCHEMA: Record = {
    "type": "object",
    "additionalProperties": False,
    "required": ["issues", "relevant_sources", "notes"],
    "properties": {
        "issues": {
            "type": "array",
            "description": f"Distinct legal issues the question needs evidence for (at most {MAX_ISSUES}).",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "description", "queries"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string",
                                    "description": "What clause or definition would settle this issue, incl. exceptions."},
                    "queries": _strings(f"1-{MAX_QUERIES_PER_ISSUE} search queries in contract language."),
                },
            },
        },
        "relevant_sources": _strings("Filenames from the provided corpus list that the question concerns; [] if unclear."),
        "notes": {"type": "string", "description": "One or two sentences on ambiguity or needed facts."},
    },
}

ANSWER_SCHEMA: Record = {
    "type": "object",
    "additionalProperties": False,
    "required": ["conclusion_type", "answer", "claims", "conditions", "unresolved_facts", "evidence_gaps"],
    "properties": {
        "conclusion_type": {"type": "string", "enum": list(CONCLUSIONS)},
        "answer": {"type": "string", "description": "Direct answer, stating its conditions. No unconditional conclusion from acquisition alone."},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["statement", "evidence"],
                "properties": {
                    "statement": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["evidence_id", "role"],
                            "properties": {"evidence_id": {"type": "string"},
                                           "role": {"type": "string", "enum": list(ROLES)}},
                        },
                    },
                },
            },
        },
        "conditions": _strings("Contractual conditions: consent, notice, cure, exceptions, definitions."),
        "unresolved_facts": _strings("Transaction facts not given in the question that change the outcome."),
        "evidence_gaps": _strings("What the retrieved evidence does not establish."),
    },
}

SEARCH_TOOL: Record = {
    "name": "search_corpus",
    "description": ("Search the contract corpus (dense + BM25 + reranker). Returns passages with evidence ids. "
                    "Optionally restrict to one source filename."),
    "strict": True,
    "input_schema": {
        "type": "object", "additionalProperties": False, "required": ["query", "source"],
        "properties": {"query": {"type": "string"},
                       "source": {"type": ["string", "null"], "description": "Exact filename or null for all contracts."}},
    },
}

LOOKUP_TOOL: Record = {
    "name": "lookup_reference",
    "description": ("Fetch passages of one section of one contract by its reference (e.g. '9.1'), "
                    "typically to follow a 'Section X' cross-reference found in evidence."),
    "strict": True,
    "input_schema": {
        "type": "object", "additionalProperties": False, "required": ["source", "reference"],
        "properties": {"source": {"type": "string", "description": "Exact filename."},
                       "reference": {"type": "string", "description": "Section reference such as 9.1 or 4.2(a)."}},
    },
}

TOOLS: list[Record] = [SEARCH_TOOL, LOOKUP_TOOL]


def _require_str_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise SchemaError(f"{field} must be a list of strings")
    return [x.strip() for x in value if x.strip()]


def validate_plan(obj: object, sources: list[str]) -> QuestionPlan:
    if not isinstance(obj, dict):
        raise SchemaError("plan must be an object")
    raw_issues = obj.get("issues")
    if not isinstance(raw_issues, list) or not raw_issues:
        raise SchemaError("plan.issues must be a non-empty list")
    issues: list[PlanIssue] = []
    for raw in raw_issues[:MAX_ISSUES]:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not isinstance(raw.get("description"), str):
            raise SchemaError("each issue needs string name and description")
        queries = _require_str_list(raw.get("queries"), "issue.queries")[:MAX_QUERIES_PER_ISSUE]
        issues.append({"name": raw["name"], "description": raw["description"], "queries": queries})
    known = set(sources)
    relevant = [s for s in _require_str_list(obj.get("relevant_sources"), "relevant_sources") if s in known]
    notes = obj.get("notes")
    return {"issues": issues, "relevant_sources": relevant, "notes": notes if isinstance(notes, str) else ""}


EVIDENCE_ID_RE = re.compile(r"\s*\[?(E\d+)\b")


def normalize_evidence_id(raw: str) -> str:
    """Models sometimes echo the whole rendered header ('E6 source=... ref=...'); keep the id."""
    match = EVIDENCE_ID_RE.match(raw)
    return match.group(1) if match else raw


def validate_answer(obj: object, ledger: dict[str, EvidenceItem]) -> tuple[CitedAnswer, list[str]]:
    """Check shape and citation provenance.

    Citations to ids absent from the ledger are removed; a claim left without
    evidence is dropped. Returns the cleaned answer and human-readable problems.
    """
    if not isinstance(obj, dict):
        raise SchemaError("answer must be an object")
    if obj.get("conclusion_type") not in CONCLUSIONS:
        raise SchemaError("invalid conclusion_type")
    if not isinstance(obj.get("answer"), str) or not isinstance(obj.get("claims"), list):
        raise SchemaError("answer and claims are required")
    problems: list[str] = []
    claims: list[Claim] = []
    for raw in obj["claims"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("statement"), str) or not isinstance(raw.get("evidence"), list):
            raise SchemaError("malformed claim")
        refs: list[EvidenceRef] = []
        for ref in raw["evidence"]:
            if not isinstance(ref, dict) or ref.get("role") not in ROLES or not isinstance(ref.get("evidence_id"), str):
                raise SchemaError("malformed evidence reference")
            evidence_id = normalize_evidence_id(ref["evidence_id"])
            if evidence_id in ledger:
                refs.append({"evidence_id": evidence_id, "role": ref["role"]})
            else:
                problems.append(f"unknown evidence id {ref['evidence_id']!r} removed")
        if refs:
            claims.append({"statement": raw["statement"], "evidence": refs})
        else:
            problems.append(f"claim dropped, no valid evidence: {raw['statement'][:80]!r}")
    answer: CitedAnswer = {
        "conclusion_type": obj["conclusion_type"], "answer": obj["answer"], "claims": claims,
        "conditions": _require_str_list(obj.get("conditions"), "conditions"),
        "unresolved_facts": _require_str_list(obj.get("unresolved_facts"), "unresolved_facts"),
        "evidence_gaps": _require_str_list(obj.get("evidence_gaps"), "evidence_gaps"),
    }
    if answer["conclusion_type"] != "insufficient_evidence" and not claims:
        problems.append("conclusion had no valid cited claims; downgraded to insufficient_evidence")
        answer["conclusion_type"] = "insufficient_evidence"
        answer["evidence_gaps"].append("No claim could be tied to retrieved evidence.")
    return answer, problems
