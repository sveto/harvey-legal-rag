"""Shared types for JSON records and the local retrieval model interfaces.

Records deliberately remain flexible at JSON/model boundaries; protocols describe
the operations used here without importing heavyweight model implementations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

Record: TypeAlias = dict[str, Any]
PathInput: TypeAlias = str | Path


class Tokenizer(Protocol):
    def __call__(self, text: str, *, truncation: bool = False) -> Record: ...


class DenseEncoder(Protocol):
    tokenizer: Tokenizer
    max_seq_length: int

    def encode(self, sentences: list[str], **kwargs: Any) -> NDArray[np.float32]: ...


class DenseIndex(Protocol):
    ntotal: int
    d: int

    def search(self, vectors: NDArray[np.float32], k: int) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...

    def reconstruct(self, key: int) -> NDArray[np.float32]: ...


class PassageReranker(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> NDArray[np.float32]: ...
