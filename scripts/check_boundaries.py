"""Report sections with embedded signature/exhibit markers after parsing."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from legal_rag.io import read_jsonl

PATTERN = re.compile(r"^(?:\[Signature Page Follows\]|IN WITNESS WHEREOF|(?:EXHIBIT|SCHEDULE|APPENDIX)\s+[A-Z0-9])", re.I)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    sections = read_jsonl(Path(args.output) / "legal_sections.jsonl")
    suspects = []
    for s in sections:
        for i, line in enumerate(s["text"].splitlines()[1:], 1):
            if PATTERN.match(line):
                suspects.append((s["source"], s["section_id"], s["section_kind"], i, line[:120]))
    print("Potential embedded boundaries:", len(suspects))
    for item in suspects:
        print(*item, sep=" | ")
    print("NOTE: references such as 'Schedule 4.17 sets forth...' can be false positives.")


if __name__ == "__main__":
    main()
