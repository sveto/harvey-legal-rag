from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from .types import PathInput, Record


def read_jsonl(path: PathInput) -> list[Record]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: PathInput, records: Iterable[Record]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def digest_chunks(chunks: list[Record]) -> str:
    payload = json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def short_id(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]
