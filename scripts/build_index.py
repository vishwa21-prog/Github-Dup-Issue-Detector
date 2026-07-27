"""CLI: fetches every issue from the configured GitHub repo, embeds it, and
writes a FAISS index + metadata to `INDEX_DIR` for later duplicate checks.

Usage:
    python scripts/build_index.py
"""
from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, ".")

from src.config import settings
from src.duplicate_detector import DuplicateDetector
from src.embeddings import build_embedder
from src.github_client import GitHubClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default=None, help="owner/repo, overrides GITHUB_REPO")
    parser.add_argument("--state", type=str, default=None, choices=["open", "closed", "all"])
    args = parser.parse_args()

    repo = args.repo or settings.repo
    state = args.state or settings.issue_state_to_index

    logger.info("Fetching issues for %s (state=%s)...", repo, state)
    client = GitHubClient(repo=repo)
    issues = client.list_issues(state=state)
    logger.info("Fetched %d issues.", len(issues))

    embedder = build_embedder(settings.embedding_backend, settings.embedding_model_name)
    detector = DuplicateDetector.build_index(
        embedder, issues,
        threshold=settings.similarity_threshold,
        top_k=settings.top_k,
        max_body_chars=settings.max_body_chars,
    )
    detector.store.save(settings.index_dir)
    embedder.save(settings.index_dir)
    logger.info("Index written to %s (%d vectors, backend=%s).",
                settings.index_dir, detector.store.index.ntotal, settings.embedding_backend)


if __name__ == "__main__":
    main()
