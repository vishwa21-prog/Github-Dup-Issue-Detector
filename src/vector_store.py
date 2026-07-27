"""FAISS IndexFlatIP (cosine similarity via normalized vectors) + a parallel
JSON metadata store, so a similarity hit can be mapped back to an issue
number/title/URL. Same pattern as a standard RAG vector store, applied to
issues instead of document chunks.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np


class IssueVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        if vectors.shape[0] != len(metadata):
            raise ValueError("vectors and metadata must have the same length")
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        scores, idxs = self.index.search(query_vec.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(out_dir, "issues.faiss"))
        with open(os.path.join(out_dir, "issues_metadata.json"), "w") as f:
            json.dump({"dim": self.dim, "metadata": self.metadata}, f, indent=2)

    @classmethod
    def load(cls, in_dir: str) -> "IssueVectorStore":
        with open(os.path.join(in_dir, "issues_metadata.json")) as f:
            payload = json.load(f)
        store = cls(dim=payload["dim"])
        store.index = faiss.read_index(os.path.join(in_dir, "issues.faiss"))
        store.metadata = payload["metadata"]
        return store
