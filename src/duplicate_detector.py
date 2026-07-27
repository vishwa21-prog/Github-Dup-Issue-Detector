"""Core duplicate-detection logic: embeds an incoming issue, retrieves the
most similar existing issues, and classifies it as a likely duplicate if
the top match clears `similarity_threshold`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.embeddings import Embedder
from src.text_utils import clean_issue_text
from src.vector_store import IssueVectorStore


@dataclass
class DuplicateCandidate:
    number: int
    title: str
    url: str
    score: float


@dataclass
class DuplicateVerdict:
    is_duplicate: bool
    candidates: List[DuplicateCandidate] = field(default_factory=list)

    @property
    def best(self) -> Optional[DuplicateCandidate]:
        return self.candidates[0] if self.candidates else None


class DuplicateDetector:
    def __init__(self, embedder: Embedder, store: IssueVectorStore,
                 threshold: float = 0.86, top_k: int = 5,
                 max_body_chars: int = 2000):
        self.embedder = embedder
        self.store = store
        self.threshold = threshold
        self.top_k = top_k
        self.max_body_chars = max_body_chars

    @classmethod
    def build_index(cls, embedder: Embedder, issues: List[Dict[str, Any]],
                     threshold: float = 0.86, top_k: int = 5,
                     max_body_chars: int = 2000) -> "DuplicateDetector":
        """Fits the embedder (needed for tfidf; no-op for sentence
        transformers) and builds a fresh vector store from `issues`
        (raw GitHub API issue dicts).
        """
        texts = [
            clean_issue_text(i.get("title", ""), i.get("body", ""), max_body_chars)
            for i in issues
        ]
        embedder.fit(texts)
        vectors = embedder.encode(texts) if texts else None

        store = IssueVectorStore(dim=embedder.dim)
        if texts:
            metadata = [
                {"number": i["number"], "title": i.get("title", ""), "url": i.get("html_url", "")}
                for i in issues
            ]
            store.add(vectors, metadata)

        return cls(embedder, store, threshold=threshold, top_k=top_k, max_body_chars=max_body_chars)

    def classify(self, title: str, body: Optional[str],
                 exclude_number: Optional[int] = None) -> DuplicateVerdict:
        text = clean_issue_text(title, body, self.max_body_chars)
        query_vec = self.embedder.encode([text])[0]

        raw_hits = self.store.search(query_vec, top_k=self.top_k + 1)
        candidates = [
            DuplicateCandidate(number=m["number"], title=m["title"], url=m["url"], score=score)
            for m, score in raw_hits
            if exclude_number is None or m["number"] != exclude_number
        ][: self.top_k]

        is_dup = bool(candidates) and candidates[0].score >= self.threshold
        return DuplicateVerdict(is_duplicate=is_dup, candidates=candidates)
