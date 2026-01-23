"""FAISS helpers for cosine similarity computations."""

from __future__ import annotations

import faiss
import numpy as np


def build_ip_index(vectors: np.ndarray) -> faiss.Index:
    """Build an IndexFlatIP assuming vectors are L2-normalized."""
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def all_pairs_cosine(vectors: np.ndarray) -> np.ndarray:
    """Compute clamped cosine similarity matrix for normalized vectors."""
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    # Assume vectors are already L2-normalized by embedding_manager
    # Inner product of normalized vectors is cosine similarity
    sims = vectors @ vectors.T
    sims = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
    return sims
