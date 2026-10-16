"""Check the citations in generated answers.

Input: a benchmark results.json, or a single agent report from artifacts/claude-runs/.
Free checks (always run, no API call):
  * scope: share of cited sections from files outside the question's own labeled files (results.json only;
    works on every Claude run, including old ones);
  * numeric: numbers and section references in a claim must appear in its cited passages (needs `cited_claims`).
LLM checks with Haiku (need ANTHROPIC_API_KEY and `cited_claims`; skipped with --no-judge or --dry-run):
  * support: does a claim's cited passages support it;
  * relevance: for each cited passage from outside the question's labeled files: needed, useful_context,
    irrelevant or wrong_contract. Labels select which passages to judge and are never shown to the judge.
`cited_claims` exist only in runs made after these checks were added; older runs get the scope check only.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from legal_rag.citation_check import (
    cited_claims,
    judge_claim,
    judge_relevance,
    numeric_check,
    off_gold_passages,
    scope_report,
    summarize_checks,
    summarize_relevance,
)
from legal_rag.claude_agent import ClaudeAgentConfig, make_client
from legal_rag.types import Record


def claim_rows(data: Record) -> list[Record]:
    if "records" in data:
        return [{"case_id": r["case_id"], "method": r["method"], "claim": c}
                for r in data["records"] if r["method"].startswith("claude:") for c in r.get("cited_claims", [])]
    return [{"case_id": "report", "method": data.get("model", "claude"), "claim": c} for c in cited_claims(data)]


def relevance_rows(data: Record, benchmark: Record) -> list[Record]:
    gold = {c["id"]: {x["source"] for g in c["required_evidence"] for x in g["any_of"]}
            | {x["source"] for x in c.get("supporting_evidence", [])} for c in benchmark["cases"]}
    return [{"case_id": r["case_id"], "method": r["method"], "question": r["question"], "passage": p}
            for r in data.get("records", []) if r["method"].startswith("claude:") and r["case_id"] in gold
            for p in off_gold_passages(r, gold[r["case_id"]])]


def load_colab_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
        os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
    except Exception:  # noqa: BLE001 -- not in Colab; make_client reports the missing key
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="results.json or a single agent report")
    p.add_argument("--benchmark", default="eval/benchmark.json", help="Used for scope and relevance (evaluation only)")
    p.add_argument("--out", default=None, help="Default: <input dir>/citation-check.json")
    p.add_argument("--judge-model", default="claude-haiku-5-5")
    p.add_argument("--no-judge", action="store_true", help="Free checks only")
    p.add_argument("--dry-run", action="store_true", help="Count items and estimate tokens; no API call")
    p.add_argument("--max-cost-usd", type=float, default=0.50)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    path = Path(args.input)
    data = json.loads(path.read_text(encoding="utf-8"))
    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    result: Record = {"judge_model": None}

    if "records" in data:
        result["scope"] = scope = scope_report(data, benchmark)
        for method, s in scope.items():
            print(f"[scope] {method}: cited={s['cited']} from unlabeled files={s['unlabeled_files']} "
                  f"({s['unlabeled_files_share']:.0%}), outside the question's labeled files={s['off_gold']} ({s['off_gold_share']:.0%})")

    claims = claim_rows(data)
    rel = relevance_rows(data, benchmark)
    for row in claims:
        row["numeric"] = numeric_check(row["claim"])
    if not claims:
        print("No cited_claims in this input (run made before these checks): only the scope check is available. "
              "Re-run the Claude agents with the current code for the numeric, support and relevance checks.")
    else:
        chars = sum(len(c["text"]) for r in claims for c in r["claim"]["citations"]) + sum(len(r["passage"]["text"]) for r in rel)
        print(f"{len(claims)} claims, {len(rel)} off-scope passages; ~{chars // 4 + 400 * (len(claims) + len(rel))} judge input tokens; "
              f"numeric flagged: {sum(r['numeric']['flagged'] for r in claims)}")

    if claims and not args.no_judge and not args.dry_run:
        load_colab_key()
        client = make_client(ClaudeAgentConfig(model=args.judge_model))
        result["judge_model"] = args.judge_model
        spent = 0.0

        def run(row: Record, call: object) -> None:
            nonlocal spent
            if spent >= args.max_cost_usd:
                row.update(verdict=None, reason="cost cap reached")
                return
            row.update(call())  # type: ignore[operator]
            spent += row.get("cost_usd") or 0.0

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda r: run(r, lambda: judge_claim(client, args.judge_model, r["claim"])), claims))
            list(pool.map(lambda r: run(r, lambda: judge_relevance(client, args.judge_model, r["question"], r["passage"])), rel))
        print(f"judge cost ~${spent:.4f}")

    if claims:
        result["support"] = summarize_checks(claims)
        for method, s in result["support"].items():
            verdicts = "n/a" if not s["judged"] else " ".join(f"{v}={s[v]:.0%}" for v in ("supported", "partial", "not_supported"))
            print(f"[support] {method}: claims={s['claims']} {verdicts} errors={s['judge_errors']} numeric_flagged={s['numeric_flagged']:.0%}")
        judged_rel = [r for r in rel if r.get("verdict") is not None]
        if judged_rel:
            result["relevance"] = summarize_relevance(rel)
            for method, s in result["relevance"].items():
                print(f"[relevance] {method}: " + " ".join(f"{k}={v}" for k, v in s.items()))
    result["rows"] = {"claims": [{k: v for k, v in r.items() if k != "claim"} | {"statement": r["claim"]["statement"]} for r in claims],
                      "off_scope_passages": [{k: v for k, v in r.items() if k != "passage"}
                                             | {"source": r["passage"]["source"], "legal_ref": r["passage"]["legal_ref"]} for r in rel]}
    out = Path(args.out) if args.out else path.parent / "citation-check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
