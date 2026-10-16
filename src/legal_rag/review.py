"""Blind, shuffled review packets for grading generated answers (see eval/EVALUATION_PROMPT.md)."""
from __future__ import annotations

import random

from .types import Record


def build_packets(results: Record, benchmark: Record, methods: list[str], seed: int = 0) -> tuple[list[Record], Record]:
    """Return (packets, key). Packets hide the method; the key maps packet id to (case_id, method)."""
    cases = {c["id"]: c for c in benchmark["cases"]}
    pairs = [(r["case_id"], r["method"], r["answer"]) for r in results["records"]
             if r["method"] in methods and not r.get("error") and r.get("answer") and r["case_id"] in cases]
    random.Random(seed).shuffle(pairs)
    packets, key = [], {}
    for n, (case_id, method, answer) in enumerate(pairs, 1):
        packet_id = f"P{n:02d}"
        case = cases[case_id]
        packets.append({"packet_id": packet_id, "case_id": case_id, "question": case["question"],
                        "reviewer_notes": case["answer_notes"], "answer": answer})
        key[packet_id] = {"case_id": case_id, "method": method}
    return packets, key


def render_packet(packet: Record) -> str:
    return (f"packet_id: {packet['packet_id']}\ncase_id: {packet['case_id']}\n"
            f"Question: {packet['question']}\nReviewer notes: {packet['reviewer_notes']}\n"
            f"Candidate answer: {packet['answer']}\n")
