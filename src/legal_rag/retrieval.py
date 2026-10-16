"""Four comparable retrieval methods: dense, BM25, hybrid, hybrid + rerank."""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import QUERY_PREFIX, passage_text
from .io import digest_chunks
from .types import DenseEncoder, DenseIndex, PassageReranker, PathInput, Record

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", re.I)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def rrf_fusion(result_lists: dict[str, list[Record]], rrf_k: int = 60) -> list[Record]:
    fused = defaultdict(float)
    provenance = defaultdict(dict)
    for method, results in result_lists.items():
        for r in results:
            fused[r["idx"]] += 1.0 / (rrf_k + r["rank"])
            provenance[r["idx"]][method] = {"rank": r["rank"], "score": r["score"]}
    return [{"idx": idx, "score": fused[idx], "methods": provenance[idx]}
            for idx in sorted(fused, key=lambda i: (-fused[i], i))]


def deduplicate_parents(items: list[Record], chunks: list[Record], k: int) -> list[Record]:
    selected, seen = [], set()
    for item in items:
        uid = chunks[item["idx"]]["parent_section_uid"]
        if uid not in seen:
            seen.add(uid)
            selected.append(item)
        if len(selected) >= k:
            break
    return selected


class LegalRetriever:
    def __init__(self, chunks: list[Record], faiss_index: DenseIndex, encoder: DenseEncoder, reranker: PassageReranker | None = None) -> None:
        self.chunks = chunks
        self.index = faiss_index
        self.encoder = encoder
        self.reranker = reranker
        self.bm25 = BM25Okapi([tokenize(passage_text(c)) for c in chunks])
        if self.index.ntotal != len(chunks):
            raise ValueError("FAISS vector count differs from chunk count")

    def dense_candidates(self, query: str, k: int = 50) -> list[Record]:
        vector = self.encoder.encode([QUERY_PREFIX + query], normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False).astype("float32")
        scores, indices = self.index.search(vector, min(k, len(self.chunks)))
        return [{"idx": int(idx), "score": float(score), "rank": rank}
                for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1) if idx >= 0]

    def bm25_candidates(self, query: str, k: int = 50) -> list[Record]:
        scores = self.bm25.get_scores(tokenize(query))
        ids = np.argsort(-scores, kind="stable")[:min(k, len(scores))]
        return [{"idx": int(idx), "score": float(scores[idx]), "rank": rank}
                for rank, idx in enumerate(ids, 1)]

    def candidates(self, query: str, method: str = "hybrid", candidate_k: int = 50, rrf_k: int = 60) -> list[Record]:
        if method == "dense":
            raw = self.dense_candidates(query, candidate_k)
            return [{"idx": r["idx"], "score": r["score"],
                     "methods": {"dense": {"rank": r["rank"], "score": r["score"]}}} for r in raw]
        if method == "bm25":
            raw = self.bm25_candidates(query, candidate_k)
            return [{"idx": r["idx"], "score": r["score"],
                     "methods": {"bm25": {"rank": r["rank"], "score": r["score"]}}} for r in raw]
        if method in {"hybrid", "hybrid_rerank"}:
            return rrf_fusion({"dense": self.dense_candidates(query, candidate_k),
                               "bm25": self.bm25_candidates(query, candidate_k)}, rrf_k)
        raise ValueError(f"Unknown method: {method}")

    def rerank(self, query: str, candidates: list[Record], batch_size: int = 16) -> list[Record]:
        if self.reranker is None:
            raise RuntimeError("Reranker not loaded. Pass --rerank to the search CLI.")
        if not candidates:
            return []
        pairs = [(query, passage_text(self.chunks[item["idx"]])) for item in candidates]
        scores = np.asarray(self.reranker.predict(pairs, batch_size=batch_size,
                                                  show_progress_bar=False, convert_to_numpy=True)).reshape(-1)
        ranked = [{**item, "rerank_score": float(score)} for item, score in zip(candidates, scores)]
        return sorted(ranked, key=lambda item: (-item["rerank_score"], item["idx"]))

    def search(self, query: str, method: str = "hybrid_rerank", candidate_k: int = 50, rerank_k: int = 40, top_k: int = 5, dedup: bool = True) -> list[Record]:
        start = time.perf_counter()
        ranked = self.candidates(query, method, candidate_k)
        if method == "hybrid_rerank":
            ranked = self.rerank(query, ranked[:rerank_k])
        ranked = deduplicate_parents(ranked, self.chunks, top_k) if dedup else ranked[:top_k]
        elapsed = round((time.perf_counter() - start) * 1000, 1)
        return [{"rank": rank, "idx": item["idx"], "score": item.get("rerank_score", item["score"]),
                 "fusion_score": item["score"], "rerank_score": item.get("rerank_score"),
                 "methods": item["methods"], "latency_ms": elapsed,
                 **self.chunks[item["idx"]]} for rank, item in enumerate(ranked, 1)]

    def diagnostic(self, query: str, source: str, section_id: str, candidate_k: int = 50, rerank_k: int = 40) -> Record:
        def ranks(items: list[Record]) -> list[int]:
            return [rank for rank, item in enumerate(items, 1)
                    if self.chunks[item["idx"]]["source"] == source
                    and str(self.chunks[item["idx"]]["section_id"]) == str(section_id)]
        dense = self.dense_candidates(query, len(self.chunks))
        sparse = self.bm25_candidates(query, len(self.chunks))
        fused = rrf_fusion({"dense": dense[:candidate_k], "bm25": sparse[:candidate_k]})
        pool = fused[:rerank_k]
        output = {"dense_ranks": ranks(dense), "bm25_ranks": ranks(sparse),
                  "rrf_ranks": ranks(fused), "in_rerank_pool": bool(ranks(pool))}
        if self.reranker is not None:
            output["reranker_ranks"] = ranks(self.rerank(query, pool))
        return output


def load_reranker(device: str | None = None, model_name: str = "BAAI/bge-reranker-base") -> PassageReranker:
    import torch
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, device=device, max_length=512,
                        activation_fn=torch.nn.Identity())


def write_bm25_metadata(chunks: list[Record], output_dir: PathInput) -> Path:
    path = Path(output_dir) / "bm25_index_config.json"
    path.write_text(json.dumps({"tokenizer": "legal_regex_v1", "chunk_count": len(chunks),
                                "chunk_sha256": digest_chunks(chunks)}, indent=2), encoding="utf-8")
    return path
