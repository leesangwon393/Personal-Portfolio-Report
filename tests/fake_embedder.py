"""Deterministic, offline stand-in for SentenceTransformer used only in
tests (Section 24: "GPU/API가 필요한 테스트는 mocking으로 대체").

A simple hashing-trick bag-of-words embedder: each word hashes into one of
`dim` buckets, so texts that share words end up with higher cosine
similarity than texts that don't — enough to exercise ranking/dedup logic
without downloading real model weights or hitting a GPU.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

FAKE_DIM = 64


def _hash_embed(text: str, dim: int = FAKE_DIM) -> np.ndarray:
    v = np.zeros(dim, dtype="float32")
    for word in re.findall(r"[a-z0-9]+", (text or "").lower()):
        idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dim
        v[idx] += 1.0
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


class FakeEmbedder:
    """Minimal stand-in for sentence_transformers.SentenceTransformer."""

    def __init__(self, dim: int = FAKE_DIM):
        self.dim = dim
        self.max_seq_length = 512

    def encode(self, texts, normalize_embeddings: bool = True):
        return [_hash_embed(t, self.dim) for t in texts]
