"""Section-preserving child chunks. Never silently discard legal text."""
from __future__ import annotations

from .io import short_id
from .parser import DEFINITION, SUBCLAUSE
from .types import Record, Tokenizer


def split_section(section: Record, max_words: int = 500, min_words: int = 35) -> list[str]:
    lines = section["text"].splitlines()
    if len(section["text"].split()) <= max_words:
        return [section["text"]]
    # Keep paragraphs intact whenever possible; split only at paragraph boundaries.
    # The original DOCX block sequence is retained as newline-delimited text.
    groups, current, count = [], [], 0
    for line in lines:
        words = len(line.split())
        if current and (count + words > max_words or
                        ((SUBCLAUSE.match(line) or DEFINITION.match(line)) and count >= min_words)):
            groups.append(current)
            current, count = [], 0
        current.append(line)
        count += words
    if current:
        groups.append(current)
    # Merge tiny trailing parts where it does not create a huge combined chunk.
    if len(groups) > 1 and sum(len(x.split()) for x in groups[-1]) < min_words:
        if sum(len(x.split()) for x in groups[-2] + groups[-1]) <= max_words:
            groups[-2].extend(groups.pop())
    result = ["\n".join(group) for group in groups]
    assert "\n".join(result) == section["text"]
    return result


def make_chunks(sections: list[Record], max_words: int = 500) -> list[Record]:
    chunks = []
    for section in sections:
        # Heading-only article is retained in sections but does not need a retrieval vector.
        # All other content, including signatures and exhibits, remains indexed.
        if section["section_kind"] == "article" and len(section["text"].splitlines()) == 1:
            continue
        parts = split_section(section, max_words=max_words)
        for i, text in enumerate(parts):
            chunks.append({
                "chunk_id": short_id(section["section_uid"], i, text),
                "parent_section_uid": section["section_uid"],
                "source": section["source"],
                "section_id": section["section_id"],
                "section_title": section["section_title"],
                "section_kind": section["section_kind"],
                "document_region": section.get("document_region", "Main Agreement"),
                "legal_ref": section.get("legal_ref"),
                "parent_path": section["parent_path"],
                "part_index": i,
                "part_count": len(parts),
                "text": text,
                "word_count": len(text.split()),
                "start_block": section["start_block"],
                "end_block": section["end_block"],
                "oversized": len(text.split()) > max_words,
            })
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    by_parent = {}
    for c in chunks:
        by_parent.setdefault(c["parent_section_uid"], []).append(c)
    for s in sections:
        if s["section_uid"] in by_parent:
            children = sorted(by_parent[s["section_uid"]], key=lambda c: c["part_index"])
            assert "\n".join(c["text"] for c in children) == s["text"]
    return chunks


def split_section_tokens(section: Record, tokenizer: Tokenizer, max_tokens: int = 480) -> list[str]:
    """Lossless token-budget splitting. Parts concatenate byte-for-byte to source text.

    Token budget includes retrieval metadata, not just clause body.
    """
    import re

    from .embeddings import passage_text

    text = section["text"]
    meta = {**section, "text": ""}

    def fits(part: str) -> bool:
        return len(tokenizer(passage_text({**meta, "text": part}), truncation=False)["input_ids"]) <= max_tokens

    if fits(text):
        return [text]
    # Prefer paragraph ends; if a single paragraph exceeds the budget, use word ends.
    positions = sorted(set([m.end() for m in re.finditer(r"\n|\s+", text)] + [len(text)]))
    parts = []
    start = 0
    while start < len(text):
        lo, hi = 0, len(positions)
        while lo < hi:
            mid = (lo + hi) // 2
            if positions[mid] <= start:
                lo = mid + 1
            elif fits(text[start:positions[mid]]):
                lo = mid + 1
            else:
                hi = mid
        good = lo - 1
        if good < 0 or positions[good] <= start:
            raise ValueError(f"Cannot fit a single token with metadata: {section['source']}")
        end = positions[good]
        # Prefer the latest newline within the last 25% of the available window.
        tail = text[start + int((end-start)*0.75):end]
        last_nl = tail.rfind("\n")
        if last_nl >= 0:
            proposed = start + int((end-start)*0.75) + last_nl + 1
            if proposed > start:
                end = proposed
        parts.append(text[start:end])
        start = end
    assert "".join(parts) == text
    assert all(fits(p) for p in parts)
    return parts


def make_token_aware_chunks(sections: list[Record], tokenizer: Tokenizer, max_tokens: int = 480) -> list[Record]:
    """Construct lossless retrieval chunks with a strict BGE token budget."""
    chunks = []
    for section in sections:
        if section["section_kind"] == "article" and len(section["text"].splitlines()) == 1:
            continue
        parts = split_section_tokens(section, tokenizer, max_tokens)
        for i, text in enumerate(parts):
            chunks.append({
                "chunk_id": short_id(section["section_uid"], i, text),
                "parent_section_uid": section["section_uid"],
                "source": section["source"],
                "section_id": section["section_id"],
                "section_title": section["section_title"],
                "section_kind": section["section_kind"],
                "document_region": section.get("document_region", "Main Agreement"),
                "legal_ref": section.get("legal_ref"),
                "parent_path": section["parent_path"],
                "part_index": i,
                "part_count": len(parts),
                "text": text,
                "word_count": len(text.split()),
                "start_block": section["start_block"],
                "end_block": section["end_block"],
                "oversized": False,
            })
    by_parent = {}
    for c in chunks:
        by_parent.setdefault(c["parent_section_uid"], []).append(c)
    for s in sections:
        if s["section_uid"] in by_parent:
            assert "".join(c["text"] for c in by_parent[s["section_uid"]]) == s["text"]
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    return chunks
