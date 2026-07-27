"""Turns a DuplicateVerdict into actual GitHub side effects (comment +
label) — or just logs them when `dry_run` is set, so the same code path
is safely testable without touching a real repo.
"""
from __future__ import annotations

import logging

from src.config import settings
from src.duplicate_detector import DuplicateVerdict
from src.github_client import GitHubClient

logger = logging.getLogger(__name__)


def format_comment(verdict: DuplicateVerdict) -> str:
    lines = [
        "🔎 **Possible duplicate detected**",
        "",
        "This issue looks similar to existing issue(s):",
        "",
    ]
    for c in verdict.candidates:
        lines.append(f"- #{c.number} — {c.title} (similarity: {c.score:.2f}) — {c.url}")
    lines.append("")
    lines.append(
        "_Automated suggestion from the duplicate-issue-detector bot — a maintainer "
        "will confirm and close/merge if this is indeed a duplicate._"
    )
    return "\n".join(lines)


def act_on_verdict(client: GitHubClient, issue_number: int, verdict: DuplicateVerdict) -> None:
    if not verdict.is_duplicate:
        logger.info("Issue #%s: no duplicate above threshold, no action taken.", issue_number)
        return

    comment_body = format_comment(verdict)

    if settings.dry_run:
        logger.info("[DRY RUN] Would comment on #%s:\n%s", issue_number, comment_body)
        if settings.add_label:
            logger.info("[DRY RUN] Would add label %r to #%s", settings.duplicate_label, issue_number)
        return

    if settings.post_comment:
        client.add_comment(issue_number, comment_body)
    if settings.add_label:
        client.add_labels(issue_number, [settings.duplicate_label])
