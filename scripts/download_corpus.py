"""Download the selected Harvey DOCX task files from GitHub's public API.

Run: python scripts/download_corpus.py --output data/docx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

API = ("https://api.github.com/repos/harveyai/harvey-labs/contents/"
       "tasks/corporate-ma/analyze-change-of-control-provisions-across-targets-material-contracts")


def fetch(url: str) -> bytes:
    with urlopen(Request(url, headers={"User-Agent": "harvey-legal-rag/0.1"}), timeout=30) as response:
        return response.read()


def walk(url: str, dest: Path) -> None:
    entries = json.loads(fetch(url))
    for entry in entries:
        if entry["type"] == "dir":
            walk(entry["url"], dest)
        elif entry["name"].lower().endswith(".docx"):
            path = dest / entry["name"]
            if path.exists():
                print("Exists:", path)
                continue
            path.write_bytes(fetch(entry["download_url"]))
            print("Saved:", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/docx")
    args = parser.parse_args()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    walk(API, target)
