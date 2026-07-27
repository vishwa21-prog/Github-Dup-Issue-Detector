"""FastAPI demo/testing surface. The production path is the GitHub Action
(scripts/check_new_issue.py) triggered on `issues: [opened]`, but this API
lets you try the detector locally or integrate it with a different CI
system that prefers calling an HTTP endpoint over shelling out to Python.
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.duplicate_detector import DuplicateDetector
from src.embeddings import build_embedder, load_embedder
from src.github_client import GitHubClient
from src.vector_store import IssueVectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Duplicate GitHub Issue Detector", version="1.0.0")

_detector: DuplicateDetector | None = None


class CheckRequest(BaseModel):
    title: str
    body: str = ""


class CheckResponse(BaseModel):
    is_duplicate: bool
    candidates: list


def _load_detector() -> DuplicateDetector:
    global _detector
    if _detector is not None:
        return _detector
    if not os.path.isdir(settings.index_dir):
        raise HTTPException(
            status_code=409,
            detail=f"No index found at {settings.index_dir}. "
                    "Run `python scripts/build_index.py` first, or POST /reindex.",
        )
    embedder = load_embedder(settings.embedding_backend, settings.embedding_model_name, settings.index_dir)
    store = IssueVectorStore.load(settings.index_dir)
    _detector = DuplicateDetector(
        embedder, store,
        threshold=settings.similarity_threshold,
        top_k=settings.top_k,
        max_body_chars=settings.max_body_chars,
    )
    return _detector


@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": _detector is not None}


@app.post("/check", response_model=CheckResponse)
def check(req: CheckRequest):
    detector = _load_detector()
    verdict = detector.classify(req.title, req.body)
    return CheckResponse(
        is_duplicate=verdict.is_duplicate,
        candidates=[
            {"number": c.number, "title": c.title, "url": c.url, "score": round(c.score, 4)}
            for c in verdict.candidates
        ],
    )


@app.post("/reindex")
def reindex():
    global _detector
    client = GitHubClient()
    issues = client.list_issues(state=settings.issue_state_to_index)
    embedder = build_embedder(settings.embedding_backend, settings.embedding_model_name)
    detector = DuplicateDetector.build_index(
        embedder, issues,
        threshold=settings.similarity_threshold,
        top_k=settings.top_k,
        max_body_chars=settings.max_body_chars,
    )
    detector.store.save(settings.index_dir)
    embedder.save(settings.index_dir)
    _detector = detector
    return {"status": "reindexed", "n_issues": len(issues)}
