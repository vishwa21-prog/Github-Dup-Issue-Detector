"""CLI: checks ONE issue (by number) against the saved duplicate-detection
index and, if it looks like a duplicate, comments + labels it. This is the
entrypoint the GitHub Action workflow calls whenever a new issue is opened.

Usage:
    python scripts/check_new_issue.py --issue-number 42
"""
from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, ".")

from src.bot_actions import act_on_verdict
from src.config import settings
from src.duplicate_detector import DuplicateDetector
from src.embeddings import load_embedder
from src.github_client import GitHubClient
from src.vector_store import IssueVectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-number", type=int, required=True)
    parser.add_argument("--repo", type=str, default=None)
    args = parser.parse_args()

    client = GitHubClient(repo=args.repo)
    issue = client.get_issue(args.issue_number)

    embedder = load_embedder(settings.embedding_backend, settings.embedding_model_name, settings.index_dir)
    store = IssueVectorStore.load(settings.index_dir)
    detector = DuplicateDetector(
        embedder, store,
        threshold=settings.similarity_threshold,
        top_k=settings.top_k,
        max_body_chars=settings.max_body_chars,
    )

    verdict = detector.classify(issue.get("title", ""), issue.get("body", ""), exclude_number=args.issue_number)

    if verdict.is_duplicate:
        logger.info("Issue #%s flagged as possible duplicate of #%s (score=%.3f)",
                     args.issue_number, verdict.best.number, verdict.best.score)
    else:
        logger.info("Issue #%s: no duplicate found above threshold %.2f",
                     args.issue_number, settings.similarity_threshold)

    act_on_verdict(client, args.issue_number, verdict)


if __name__ == "__main__":
    main()
