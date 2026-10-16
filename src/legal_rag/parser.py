"""DOCX structural parsing. Preserve all substantive blocks, including appendices.

The source corpus often uses Normal styles instead of Word Heading styles.
We therefore combine textual headings, run formatting, and standalone blocks.
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document

from .io import short_id
from .types import PathInput, Record

ARTICLE = re.compile(r"^ARTICLE\s+([IVXLCDM]+|\d+)\b\s*[-—:.]?\s*(.*)$", re.I)
SECTION = re.compile(r"^Section\s+(\d+(?:\.\d+)*[a-z]?)\b\s*[-—:.]?\s*(.*)$", re.I)
SPECIAL = re.compile(r"^(EXHIBIT|SCHEDULE|APPENDIX)\s+([A-Z0-9IVXLC]+(?:[.-][A-Z0-9]+)*)(?:\s*[-—:]\s*(.*))?$", re.I)
SIGNATURE_NOTICE = re.compile(r"^\[\s*Signature Page Follows\s*\]$", re.I)
SIGNATURE_PAGE = re.compile(r"^SIGNATURE PAGE$", re.I)
WITNESS = re.compile(r"^IN WITNESS WHEREOF\b", re.I)
MEMO = re.compile(r"^([IVXLCDM]+(?:\.[A-Z])?)\.\s+(.+)$")
NUMBERED = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+(.+)$")
INTEGER = re.compile(r"^(\d+)\.\s+(.+)$")
LETTERED = re.compile(r"^([A-Z])\.\s+(.+)$")
SUBCLAUSE = re.compile(r"^\s*(?:\([a-z]\)|[a-z]\)|\([ivxlcdm]+\))\s+", re.I)
DEFINITION = re.compile(r'^\s*["“][^"”]{2,100}["”]\s+(?:means|shall mean|refers to)\b', re.I)


def clean(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def extract_blocks(path: PathInput) -> list[Record]:
    document = Document(str(path))
    blocks = []
    for element in document.iter_inner_content():
        if element.__class__.__name__ == "Paragraph":
            value = clean(element.text)
            if not value or "Right-click to update Table of Contents" in value:
                continue
            visible = [run for run in element.runs if clean(run.text)]
            bold_chars = sum(len(run.text) for run in visible if run.bold)
            all_chars = sum(len(run.text) for run in visible)
            blocks.append({"text": value, "bold_ratio": bold_chars / all_chars if all_chars else 0,
                           "style": element.style.name if element.style else "", "kind": "paragraph"})
        else:
            for row in element.rows:
                value = " | ".join(clean(cell.text) for cell in row.cells)
                if value.strip(" |"):
                    blocks.append({"text": value, "bold_ratio": 0, "style": "", "kind": "table_row"})
    return blocks


def heading(text: str, bold_ratio: float = 0, style: str = "", kind: str = "paragraph", in_appendix: bool = False) -> Record | None:
    """Return heading metadata, or None. Avoid matching cross-references in prose."""
    if kind != "paragraph":
        return None
    t = clean(text)
    if SIGNATURE_NOTICE.fullmatch(t):
        return {"section_kind": "signature_notice", "section_id": None,
                "section_title": "Signature Page Notice", "level": 0}
    if SIGNATURE_PAGE.fullmatch(t) or WITNESS.match(t):
        return {"section_kind": "signature", "section_id": None,
                "section_title": "Signature Page", "level": 0}
    m = SPECIAL.fullmatch(t)
    if m:
        label, ident, title = m.groups()
        return {"section_kind": label.lower(), "section_id": ident,
                "section_title": clean(title) or f"{label.title()} {ident}", "level": 0}
    m = ARTICLE.match(t)
    if m and (len(t) < 220 or bold_ratio >= 0.5):
        return {"section_kind": "article", "section_id": m[1], "section_title": clean(m[2]), "level": 1}
    m = SECTION.match(t)
    if m:
        return {"section_kind": "section", "section_id": m[1],
                "section_title": clean(m[2])[:160], "level": 2}
    # Numbered clauses within appendices are NOT treated as article boundaries.
    m = MEMO.match(t)
    if m and (bold_ratio >= 0.45 or len(t) <= 110) and not in_appendix:
        return {"section_kind": "memo", "section_id": m[1], "section_title": m[2], "level": 1}
    m = NUMBERED.match(t)
    if m and bold_ratio >= 0.45 and len(t) <= 180:
        return {"section_kind": "numbered", "section_id": m[1], "section_title": m[2], "level": 2}
    m = INTEGER.match(t)
    if m and bold_ratio >= 0.45 and len(t) <= 150:
        return {"section_kind": "integer", "section_id": m[1], "section_title": m[2], "level": 2}
    m = LETTERED.match(t)
    if m and bold_ratio >= 0.6 and len(t) <= 130 and not in_appendix:
        return {"section_kind": "memo", "section_id": m[1], "section_title": m[2], "level": 2}
    # Lettered legal subclauses retain their parent section number.
    m = re.match(r"^\(([a-z])\)\s+(.+)$", t, re.I)
    if m and len(t) <= 180:
        return {"section_kind": "subclause", "section_id": m.group(1).lower(),
                "section_title": m.group(2), "level": 3}
    if t.lower() == "by and between":
        return None
    # Bold alone is insufficient: standalone conjunctions and signatures are not headings.
    is_formatted = style.lower().startswith("heading") or bold_ratio >= 0.85
    has_letters = len(re.findall(r"[A-Za-z]", t)) >= 4
    is_single_conjunction = t.lower() in {"and", "or", "by", "the", "of", "not", "name", "date", "title"}
    if is_formatted and has_letters and not is_single_conjunction and 5 <= len(t) <= 100 and not t.endswith(".") and not t.endswith(":") and not re.match(r"^(?:By|Name|Title|Date)\s*:", t, re.I):
        return {"section_kind": "unnumbered", "section_id": None, "section_title": t, "level": 2}
    return None


def build_sections(source: str, blocks: list[Record]) -> list[Record]:
    sections = []
    current = None
    hierarchy = []
    in_appendix = False
    region = None
    active_clause = None

    def flush(end: int) -> None:
        nonlocal current
        if current is None:
            return
        current["end_block"] = end
        current["text"] = "\n".join(current.pop("_lines"))
        current["word_count"] = len(current["text"].split())
        current["section_uid"] = short_id(source, current["start_block"], end,
                                          current["section_kind"], current["section_id"])
        sections.append(current)
        current = None

    for i, block in enumerate(blocks):
        detected = heading(block["text"], block["bold_ratio"], block["style"],
                           block["kind"], in_appendix=in_appendix)
        if detected:
            flush(i)
            level = detected["level"]
            kind = detected["section_kind"]
            if level == 0:
                hierarchy.clear()
                active_clause = None
                if kind in {"exhibit", "schedule", "appendix"}:
                    region = f"{kind.title()} {detected['section_id']}"
                    in_appendix = True
                elif kind in {"signature", "signature_notice"}:
                    # Signature notices must not destroy the preceding appendix context.
                    if not in_appendix:
                        region = "Signature Page"
                else:
                    region = None
                    in_appendix = False
            else:
                hierarchy[:] = [item for item in hierarchy if item[0] < level]
            parents = ([region] if region and kind not in {"exhibit", "schedule", "appendix"} else []) + [name for _, name in hierarchy]
            label = " ".join(x for x in [str(detected["section_id"] or ""), detected["section_title"]] if x)
            if level > 0:
                hierarchy.append((level, label))
            if kind in {"section", "numbered", "integer"}:
                active_clause = str(detected["section_id"])
            elif kind in {"article", "memo"}:
                active_clause = None
            legal_ref = (f"{region} / {active_clause}" if region and active_clause and level > 0
                         else region if region and level > 0
                         else f"{kind.title()} {detected['section_id']}" if kind in {"exhibit", "schedule", "appendix"}
                         else active_clause if kind in {"section", "numbered", "integer"}
                         else None)
            if kind == "subclause" and active_clause:
                legal_ref = (f"{region} / " if region else "") + f"{active_clause}({detected['section_id']})"
            current = {"source": source, **detected, "parent_path": parents,
                       "document_region": region or "Main Agreement", "legal_ref": legal_ref,
                       "start_block": i, "_lines": []}
        elif current is None:
            current = {"source": source, "section_kind": "preamble", "section_id": None,
                       "section_title": "Document preamble", "level": 0, "parent_path": [],
                       "document_region": "Main Agreement", "legal_ref": None,
                       "start_block": i, "_lines": []}
        current["_lines"].append(block["text"])
    flush(len(blocks))
    # Every retained source block belongs to exactly one section.
    assert sum(s["end_block"] - s["start_block"] for s in sections) == len(blocks)
    assert "\n".join(s["text"] for s in sections) == "\n".join(b["text"] for b in blocks)
    return sections


def parse_directory(input_dir: PathInput) -> tuple[list[Record], dict[str, dict[str, int]]]:
    all_sections = []
    stats = {}
    for path in sorted(Path(input_dir).glob("*.docx")):
        blocks = extract_blocks(path)
        parsed = build_sections(path.name, blocks)
        all_sections.extend(parsed)
        stats[path.name] = {"blocks": len(blocks), "sections": len(parsed)}
    if not stats:
        raise FileNotFoundError(f"No .docx files found in {input_dir}")
    return all_sections, stats
