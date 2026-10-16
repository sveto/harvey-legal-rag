"""Deterministic evidence-gap retrieval on top of LegalRetriever. No LLM/API calls."""
from __future__ import annotations

import re
from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .embeddings import QUERY_PREFIX
from .retrieval import LegalRetriever, rrf_fusion, tokenize
from .types import Record

# Search facets are prompts for retrieval, not assertions about contract terms.
FACETS = {
    "consent": {
        "query": "assignment transfer change of control prior written consent approval",
        "terms": (r"\bconsent\b", r"\bapproval\b", r"\bassign(?:ment|ed)?\b"),
    },
    "competitor": {
        "query": "change of control acquisition direct competitor withhold consent competitor carve-out",
        "terms": (r"\bcompetitor\b", r"\bcompeting\b", r"\bcompetitive\b"),
    },
    "termination": {
        "query": "termination right terminate agreement unauthorized assignment change of control breach",
        "terms": (r"\bterminat(?:e|ion|ed)\b", r"\bright to terminate\b"),
    },
    "cure": {
        "query": "cure period notice unauthorized assignment termination written consent remedy",
        "terms": (r"\bcure\b", r"\bnotice\b", r"\bremed(?:y|ies)\b"),
    },
    "exclusivity": {
        "query": "exclusive license change of control automatic conversion non-exclusive",
        "terms": (r"\bexclusiv(?:e|ity)\b", r"\bnon-exclusive\b"),
    },
    "acceleration": {
        "query": "change of control event of default automatic acceleration obligations immediately due payable",
        "terms": (r"\baccelerat(?:ion|e|ed)\b", r"\bdue and payable\b", r"\bprepay(?:ment)?\b"),
    },
}


class AgentState(TypedDict):
    question: str
    source: str | None
    facets: list[str]
    evidence: dict[str, list[Record]]
    missing: list[str]
    trace: list[Record]
    rounds: int


def infer_facets(question: str) -> list[str]:
    q = question.lower()
    if re.search(r"terminat|right to end|cancel", q):
        return ["consent", "competitor", "termination", "cure"] if re.search(r"competitor|acquir|change of control", q) else ["termination", "cure"]
    if re.search(r"exclusiv|licen[sc]e", q):
        return ["exclusivity", "consent"]
    if re.search(r"repay|accelerat|credit|loan", q):
        return ["acceleration", "cure"]
    if re.search(r"assign|consent|change of control", q):
        return ["consent"]
    return ["consent", "termination"]


def infer_source(question: str, chunks: list[Record]) -> str | None:
    """Only select a document if its distinctive filename stem occurs in question."""
    q = question.lower()
    sources = sorted({c["source"] for c in chunks})
    hits = []
    for source in sources:
        stem = source.lower().removesuffix(".docx")
        tokens = [t for t in re.split(r"[-_\W]+", stem) if len(t) >= 5 and t not in {"agreement", "license", "credit", "services", "contract", "subscription", "renewal"}]
        matched = [t for t in tokens if re.search(r"\b" + re.escape(t) + r"\b", q)]
        if matched:
            hits.append((len(matched), source))
    hits.sort(reverse=True)
    return hits[0][1] if hits and (len(hits) == 1 or hits[0][0] > hits[1][0]) else None


def _eligible(chunk: Record, source: str | None) -> bool:
    if source and chunk["source"] != source:
        return False
    region = (chunk.get("document_region") or "").lower()
    if "signature" in region:
        return False
    if chunk.get("section_kind") == "signature":
        return False
    return True


def scoped_search(retriever: LegalRetriever, query: str, source: str | None, top_k: int = 8, rerank_k: int = 24) -> list[Record]:
    """Rank *within* the selected document, not global top-50 then filter.

    Uses existing FAISS vectors and BM25, and optionally the local cross-encoder.
    """
    ids = [i for i, c in enumerate(retriever.chunks) if _eligible(c, source)]
    if not ids:
        return []
    vector = retriever.encoder.encode([QUERY_PREFIX + query], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False).astype("float32")[0]
    # FAISS IndexFlatIP supports reconstruct; no extra embeddings or index rebuild needed.
    dense_scores = [(i, float(np.dot(vector, retriever.index.reconstruct(i)))) for i in ids]
    dense_scores.sort(key=lambda p: (-p[1], p[0]))
    sparse = retriever.bm25.get_scores(tokenize(query))
    sparse_ids = sorted(ids, key=lambda i: (-float(sparse[i]), i))
    limit = min(50, len(ids))
    lists = {
        "dense": [{"idx": i, "score": score, "rank": rank} for rank, (i, score) in enumerate(dense_scores[:limit], 1)],
        "bm25": [{"idx": i, "score": float(sparse[i]), "rank": rank} for rank, i in enumerate(sparse_ids[:limit], 1)],
    }
    ranked = rrf_fusion(lists)
    if retriever.reranker is not None:
        ranked = retriever.rerank(query, ranked[:rerank_k])
    # Dedup section parents, not legal refs: each facet can retrieve a different subclause.
    output, seen = [], set()
    for item in ranked:
        c = retriever.chunks[item["idx"]]
        uid = c.get("parent_section_uid", c["chunk_id"])
        if uid in seen:
            continue
        seen.add(uid)
        output.append({"idx": item["idx"], "score": item.get("rerank_score", item["score"]),
                       "source": c["source"], "legal_ref": c.get("legal_ref"),
                       "section_title": c.get("section_title"), "text": c["text"],
                       "chunk_id": c.get("chunk_id"), "parent_section_uid": uid})
        if len(output) >= top_k:
            break
    return output


def supports_facet(item: Record, facet: str) -> bool:
    text = (item.get("section_title") or "") + "\n" + item["text"]
    return any(re.search(p, text, re.I) for p in FACETS[facet]["terms"])


def build_agent(retriever: LegalRetriever, *, max_rounds: int = 2, per_facet: int = 6) -> CompiledStateGraph:
    """Graph: plan -> initial -> assess -> targeted -> assess -> finish.

    'Supported' means a lexical evidence candidate was found, NOT legal entailment.
    """
    if not 0 <= max_rounds <= 3:
        raise ValueError("max_rounds must be between 0 and 3")

    def plan(state: AgentState) -> Record:
        facets = infer_facets(state["question"])
        source = state.get("source") or infer_source(state["question"], retriever.chunks)
        return {"facets": facets, "source": source, "evidence": {f: [] for f in facets},
                "missing": facets[:], "trace": [{"step": "plan", "source": source, "facets": facets}], "rounds": 0}

    def initial(state: AgentState) -> Record:
        results = scoped_search(retriever, state["question"], state["source"], top_k=12)
        evidence = {f: [x for x in results if supports_facet(x, f)][:per_facet] for f in state["facets"]}
        return {"evidence": evidence, "trace": state["trace"] + [{"step": "initial", "retrieved": len(results), "source": state["source"]}]}

    def assess(state: AgentState) -> Record:
        missing = [f for f in state["facets"] if not state["evidence"].get(f)]
        return {"missing": missing, "trace": state["trace"] + [{"step": "assess", "missing": missing}]}

    def targeted(state: AgentState) -> Record:
        evidence = {f: list(items) for f, items in state["evidence"].items()}
        trace = list(state["trace"])
        for facet in state["missing"]:
            query = f"{state['question']} {FACETS[facet]['query']}"
            results = scoped_search(retriever, query, state["source"], top_k=per_facet + 4)
            seen = {x["parent_section_uid"] for x in evidence[facet]}
            for item in results:
                if supports_facet(item, facet) and item["parent_section_uid"] not in seen:
                    evidence[facet].append(item)
                    seen.add(item["parent_section_uid"])
                if len(evidence[facet]) >= per_facet:
                    break
            trace.append({"step": "targeted", "facet": facet, "retrieved": len(results),
                          "candidates": len(evidence[facet])})
        return {"evidence": evidence, "trace": trace, "rounds": state["rounds"] + 1}

    def route(state: AgentState) -> str:
        return "targeted" if state["missing"] and state["rounds"] < max_rounds else END

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("initial", initial)
    graph.add_node("assess", assess)
    graph.add_node("targeted", targeted)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "initial")
    graph.add_edge("initial", "assess")
    graph.add_conditional_edges("assess", route, {"targeted": "targeted", END: END})
    graph.add_edge("targeted", "assess")
    return graph.compile()


def evidence_report(state: AgentState) -> Record:
    return {
        "question": state["question"], "source": state["source"],
        "status": "candidate_evidence_found" if not state["missing"] else "incomplete_evidence",
        "warning": "Lexical facet coverage is not a legal conclusion. Verify all cited clauses and conditions.",
        "missing": state["missing"], "rounds": state["rounds"],
        "evidence": {facet: [{k: item[k] for k in ("source", "legal_ref", "section_title", "score", "text")}
                             for item in items] for facet, items in state["evidence"].items()},
        "trace": state["trace"],
    }

# V2: issue-conditioned evidence validation and deterministic cross-reference expansion.
# These override the V1 graph and report while retaining the existing retrieval backend.
ISSUE_RE = re.compile(r"change of control|acquir(?:e|ed|es|ing|isition)|competitor|assign(?:ment|ed)?|transfer", re.I)
REFERENCE_RE = re.compile(r"\b(?:sections?|§§?)\s+((?:\d{1,3}\.)?\d{1,3}(?:\([a-z0-9ivx]+\))?)", re.I)


def role_matches(item: Record, role: str, question: str) -> bool:
    """Conservative lexical checks, not legal entailment.

    Require the relevant *issue* in remedy/cure text, preventing SLA outage
    termination from being counted for an assignment/change-of-control question.
    """
    text = "\n".join((item.get("section_title") or "", item.get("text") or ""))
    t = text.lower()
    is_assignment = bool(ISSUE_RE.search(question))
    issue = bool(re.search(r"assign|transfer|change of control|competitor|unauthori[sz]ed", t))
    if role == "consent":
        return bool(re.search(r"\b(?:prior\s+)?(?:written\s+)?consent\b|\bapproval\b", t)
                    and re.search(r"\b(?:shall|must|requires?|without|subject to|prohibit|restriction|may not|cannot)\b", t)
                    and re.search(r"assign|transfer|change of control", t))
    if role == "competitor":
        return bool(re.search(r"\b(?:direct\s+)?competitor\b|\bcompeting\b", t))
    if role == "termination":
        return bool(re.search(r"\b(?:right to terminate|may terminate|entitled to terminate|termination upon|terminate this agreement)\b", t)
                    and (not is_assignment or issue))
    if role == "cure":
        return bool(re.search(r"\bcure\b|\bremed(?:y|ied)\b", t)
                    and (not is_assignment or issue))
    return supports_facet(item, role)


def reference_targets(item: Record) -> list[str]:
    """Extract explicit section references, excluding self, preserving order."""
    text = item.get("text") or ""
    self_ref = (item.get("legal_ref") or "").split(" / ")[-1]
    refs = []
    for match in REFERENCE_RE.finditer(text):
        ref = match.group(1)
        if ref != self_ref and ref not in refs:
            refs.append(ref)
    return refs


def lookup_section(retriever: LegalRetriever, source: str, ref: str) -> list[Record]:
    """Load referenced section directly from indexed chunks, without embedding calls."""
    matches = []
    for i, c in enumerate(retriever.chunks):
        if c.get("source") != source or not _eligible(c, source):
            continue
        legal = (c.get("legal_ref") or "").split(" / ")[-1]
        if legal != ref:
            continue
        matches.append((i, c))
    return [{"idx": i, "score": 0.0, "source": c["source"],
             "legal_ref": c.get("legal_ref"), "section_title": c.get("section_title"),
             "text": c["text"], "chunk_id": c.get("chunk_id"),
             "parent_section_uid": c.get("parent_section_uid", c.get("chunk_id")),
             "retrieval_method": "cross_reference"} for i, c in matches]


def _merge_evidence(evidence: dict[str, list[Record]], role: str, item: Record, limit: int) -> None:
    items = evidence.setdefault(role, [])
    key = item.get("chunk_id") or (item.get("source"), item.get("legal_ref"), item.get("text"))
    if all((x.get("chunk_id") or (x.get("source"), x.get("legal_ref"), x.get("text"))) != key for x in items):
        if len(items) < limit:
            items.append(item)


def build_agent(retriever: LegalRetriever, *, max_rounds: int = 2, per_facet: int = 6) -> CompiledStateGraph:
    """V2 graph: role validation, cross-reference expansion, targeted search.

    Explicit references are followed without using an LLM. Coverage still means
    'candidate evidence', not proof of the legal outcome.
    """
    if not 0 <= max_rounds <= 3:
        raise ValueError("max_rounds must be between 0 and 3")

    def plan(state: AgentState) -> Record:
        facets = infer_facets(state["question"])
        source = state.get("source") or infer_source(state["question"], retriever.chunks)
        return {"facets": facets, "source": source, "evidence": {f: [] for f in facets},
                "missing": facets[:], "trace": [{"step": "plan", "source": source, "facets": facets}], "rounds": 0}

    def initial(state: AgentState) -> Record:
        results = scoped_search(retriever, state["question"], state["source"], top_k=12)
        evidence = {f: [] for f in state["facets"]}
        for item in results:
            for f in state["facets"]:
                if role_matches(item, f, state["question"]):
                    _merge_evidence(evidence, f, item, per_facet)
        return {"evidence": evidence, "trace": state["trace"] +
                [{"step": "initial", "retrieved": len(results), "source": state["source"]}]}

    def expand(state: AgentState) -> Record:
        evidence = {f: list(v) for f, v in state["evidence"].items()}
        trace = list(state["trace"])
        if not state["source"]:
            return {"evidence": evidence, "trace": trace}
        # Follow references from all initial evidence, even if every facet appears covered.
        seed = {item.get("chunk_id") or (item.get("legal_ref"), item.get("text")): item
                for items in evidence.values() for item in items}
        seen_refs = set()
        for item in list(seed.values()):
            for ref in reference_targets(item):
                if ref in seen_refs:
                    continue
                seen_refs.add(ref)
                matches = lookup_section(retriever, state["source"], ref)
                if matches:
                    trace.append({"step": "expand_reference", "from": item.get("legal_ref"),
                                  "reference": ref, "chunks": len(matches)})
                for found in matches:
                    for f in state["facets"]:
                        if role_matches(found, f, state["question"]):
                            _merge_evidence(evidence, f, found, per_facet)
        return {"evidence": evidence, "trace": trace}

    def assess(state: AgentState) -> Record:
        missing = [f for f in state["facets"] if not state["evidence"].get(f)]
        return {"missing": missing, "trace": state["trace"] + [{"step": "assess", "missing": missing}]}

    def targeted(state: AgentState) -> Record:
        evidence = {f: list(items) for f, items in state["evidence"].items()}
        trace = list(state["trace"])
        for f in state["missing"]:
            query = f"{state['question']} {FACETS[f]['query']}"
            results = scoped_search(retriever, query, state["source"], top_k=per_facet + 8)
            for item in results:
                if role_matches(item, f, state["question"]):
                    _merge_evidence(evidence, f, item, per_facet)
            trace.append({"step": "targeted", "facet": f, "retrieved": len(results),
                          "candidates": len(evidence[f])})
        return {"evidence": evidence, "trace": trace, "rounds": state["rounds"] + 1}

    def route(state: AgentState) -> str:
        return "targeted" if state["missing"] and state["rounds"] < max_rounds else END

    graph = StateGraph(AgentState)
    for name, fn in (("plan", plan), ("initial", initial), ("expand", expand),
                     ("assess", assess), ("targeted", targeted)):
        graph.add_node(name, fn)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "initial")
    graph.add_edge("initial", "expand")
    graph.add_edge("expand", "assess")
    graph.add_conditional_edges("assess", route, {"targeted": "targeted", END: END})
    graph.add_edge("targeted", "expand")
    return graph.compile()


def evidence_report(state: AgentState) -> Record:
    return {"question": state["question"], "source": state["source"],
            "status": "candidate_evidence_found" if not state["missing"] else "incomplete_evidence",
            "warning": "Role-specific lexical coverage is not legal entailment; verify the exact conditions and citations.",
            "missing": state["missing"], "rounds": state["rounds"],
            "evidence": {f: [{k: item.get(k) for k in ("source", "legal_ref", "section_title", "score", "text")}
                             for item in items] for f, items in state["evidence"].items()},
            "trace": state["trace"]}
