"""BGE embeddings and FAISS index, with corpus fingerprint validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io import digest_chunks
from .types import DenseEncoder, DenseIndex, PathInput, Record

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def passage_text(chunk: Record) -> str:
    context = [f"Document: {chunk['source']}"]
    if chunk.get("legal_ref"):
        context.append(f"Legal reference: {chunk['legal_ref']}")
    if chunk.get("parent_path"):
        context.append("Parent sections: " + " > ".join(chunk["parent_path"]))
    if chunk.get("section_id"):
        context.append(f"Section identifier: {chunk['section_id']}")
    context.append(chunk["text"])
    return "\n".join(context)


def load_encoder(model_name: str = DEFAULT_MODEL, device: str | None = None) -> DenseEncoder:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 512
    return model


def build_dense(chunks: list[Record], output_dir: PathInput, model_name: str = DEFAULT_MODEL, batch_size: int = 32, device: str | None = None) -> tuple[DenseIndex, DenseEncoder]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = load_encoder(model_name, device)
    texts = [passage_text(c) for c in chunks]
    # Fail instead of silently truncating legally relevant text.
    over = [i for i, t in enumerate(texts) if len(encoder.tokenizer(t, truncation=False)["input_ids"]) > encoder.max_seq_length]
    if over:
        raise ValueError(f"{len(over)} chunks exceed {encoder.max_seq_length} tokens; run parse with --token-aware first. First indices: {over[:10]}")
    vectors = encoder.encode(texts, batch_size=batch_size,
                             normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=True).astype("float32")
    import faiss
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(np.ascontiguousarray(vectors))
    faiss.write_index(idx, str(output_dir / "legal_dense.faiss"))
    np.save(output_dir / "legal_embeddings.npy", vectors)
    config = {"model": model_name, "dimension": int(vectors.shape[1]),
              "chunk_count": len(chunks), "chunk_sha256": digest_chunks(chunks),
              "normalized": True, "similarity": "inner_product", "max_seq_length": 512}
    (output_dir / "dense_index_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return idx, encoder


def load_dense(chunks: list[Record], index_dir: PathInput, device: str | None = None) -> tuple[DenseIndex, DenseEncoder]:
    index_dir = Path(index_dir)
    config = json.loads((index_dir / "dense_index_config.json").read_text(encoding="utf-8"))
    if config["chunk_sha256"] != digest_chunks(chunks):
        raise ValueError("Corpus changed: rebuild dense and BM25 indexes before searching")
    import faiss
    idx = faiss.read_index(str(index_dir / "legal_dense.faiss"))
    if idx.ntotal != len(chunks) or idx.d != config["dimension"]:
        raise ValueError("FAISS index incompatible with chunk corpus")
    return idx, load_encoder(config["model"], device)


def token_diagnostics(chunks: list[Record], encoder: DenseEncoder) -> Record:
    lengths = [len(encoder.tokenizer(passage_text(c), truncation=False)["input_ids"]) for c in chunks]
    return {"total": len(lengths), "over_limit": sum(n > encoder.max_seq_length for n in lengths),
            "max_tokens": max(lengths, default=0), "median_tokens": float(np.median(lengths)) if lengths else 0}
