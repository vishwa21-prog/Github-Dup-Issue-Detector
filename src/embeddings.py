"""Embedding backends for issue text.

Two interchangeable backends behind one interface:
- "sentence-transformers": dense semantic embeddings, catches paraphrased
  duplicates ("app crashes on launch" vs "crashes immediately on startup").
  Needs a one-time model download from the HF Hub.
- "tfidf": sparse lexical vectors fit on the repo's own issue corpus. Zero
  downloads, works fully offline (useful in air-gapped CI runners), but
  only catches duplicates that share vocabulary, not pure paraphrases.

Both expose `.fit(texts)` (tfidf needs this; sentence-transformers no-ops)
and `.encode(texts) -> np.ndarray` of L2-normalized row vectors, so cosine
similarity is just a dot product — which is what `IndexFlatIP` in
src/vector_store.py computes.
"""
from __future__ import annotations

from typing import List, Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def fit(self, texts: List[str]) -> None: ...
    def encode(self, texts: List[str]) -> np.ndarray: ...
    def save(self, out_dir: str) -> None: ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. Lazily imports/loads so the
    rest of the codebase doesn't require the (heavier) dependency unless
    this backend is actually selected.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def fit(self, texts: List[str]) -> None:
        # No fitting needed for a pretrained transformer encoder.
        return

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(vecs.astype("float32"))

    def save(self, out_dir: str) -> None:
        # Nothing to persist: encode() is a pure function of the pretrained
        # model name, which is already recorded in config/.env. Reloading
        # just re-downloads (or reads from the local HF cache) by name.
        return


class TfidfEmbedder:
    """Offline fallback: TF-IDF vectors fit on the repo's own issues,
    projected to dense float32 arrays for a uniform interface with
    SentenceTransformerEmbedder.
    """

    def __init__(self, max_features: int = 20_000):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), min_df=1, stop_words="english"
        )
        self._fitted = False
        self.dim = 0

    def fit(self, texts: List[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True
        self.dim = len(self._vectorizer.vocabulary_)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit(texts) must be called before encode().")
        mat = self._vectorizer.transform(texts).toarray().astype("float32")
        return _l2_normalize(mat)

    def save(self, out_dir: str) -> None:
        # Unlike sentence-transformers, TF-IDF vectors depend on a
        # vocabulary fit on THIS repo's issues, so the fitted vectorizer
        # itself must be persisted for a later process (e.g. the CI
        # workflow checking a single new issue) to encode compatible
        # vectors against the saved FAISS index.
        import os
        import pickle

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(self._vectorizer, f)

    @classmethod
    def load(cls, in_dir: str) -> "TfidfEmbedder":
        import os
        import pickle

        obj = cls()
        with open(os.path.join(in_dir, "tfidf_vectorizer.pkl"), "rb") as f:
            obj._vectorizer = pickle.load(f)
        obj._fitted = True
        obj.dim = len(obj._vectorizer.vocabulary_)
        return obj


def build_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend!r} (expected 'sentence-transformers' or 'tfidf')")


def load_embedder(backend: str, model_name: str, index_dir: str) -> Embedder:
    """Like build_embedder, but for tfidf reloads the vectorizer that was
    fit (and saved) when the index was built — required for a separate
    process (e.g. the per-issue CI check) to produce compatible vectors.
    """
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    if backend == "tfidf":
        return TfidfEmbedder.load(index_dir)
    raise ValueError(f"Unknown embedding backend: {backend!r} (expected 'sentence-transformers' or 'tfidf')")
