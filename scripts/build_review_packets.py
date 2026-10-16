"""Build blind, shuffled answer-review packets from a benchmark results.json.

Output (in --out): packets.md and packets.jsonl (give these to the reviewer together with the prompt
in eval/EVALUATION_PROMPT.md) and key.json (packet id -> method; keep it away from the reviewer).
No API calls. Packets contain answer_notes, which are evaluation-only labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.review import build_packets, render_packet


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, help="e.g. eval/results/claude-v1/results.json")
    p.add_argument("--benchmark", default="eval/benchmark.json")
    p.add_argument("--out", default=None, help="Default: <results dir>/review")
    p.add_argument("--methods", nargs="*", default=None, help="Default: every claude:* method in the results")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    methods = args.methods or sorted({r["method"] for r in results["records"] if r["method"].startswith("claude:")})
    if not methods:
        p.error("no Claude methods found; pass --methods")
    packets, key = build_packets(results, benchmark, methods, args.seed)
    out = Path(args.out) if args.out else Path(args.results).parent / "review"
    out.mkdir(parents=True, exist_ok=True)
    (out / "packets.md").write_text("\n---\n\n".join(render_packet(x) for x in packets), encoding="utf-8")
    (out / "packets.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in packets), encoding="utf-8")
    (out / "key.json").write_text(json.dumps({"seed": args.seed, "methods": methods, "key": key}, indent=2) + "\n",
                                  encoding="utf-8")
    print(f"{len(packets)} packets for {methods} -> {out} (seed {args.seed})")


if __name__ == "__main__":
    main()
