"""Text normalization for GitHub issues before embedding.

Issue bodies are messy: markdown formatting, code blocks, stack traces,
issue-template boilerplate ("### Steps to reproduce"), and HTML comments
from hidden template instructions. Stripping this noise measurably helps
embedding-based similarity, since otherwise two unrelated issues that both
happen to include a large shared stack trace or template boilerplate can
look artificially similar.
"""
from __future__ import annotations

import re

_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MD_HEADER_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+")
_WHITESPACE_RE = re.compile(r"\s+")
_CHECKBOX_RE = re.compile(r"- \[[ xX]\]\s*")


def clean_issue_text(title: str, body: str | None, max_body_chars: int = 2000) -> str:
    """Returns a single normalized string combining title + cleaned body,
    suitable for embedding. The title is repeated/weighted implicitly by
    being placed first, since it's usually the highest-signal part of an
    issue for duplicate detection.
    """
    title = (title or "").strip()
    body = body or ""

    body = _CODE_BLOCK_RE.sub(" ", body)
    body = _HTML_COMMENT_RE.sub(" ", body)
    body = _INLINE_CODE_RE.sub(" ", body)
    body = _MD_LINK_RE.sub(r"\1", body)
    body = _URL_RE.sub(" ", body)
    body = _MD_HEADER_RE.sub(" ", body)
    body = _CHECKBOX_RE.sub(" ", body)
    body = _WHITESPACE_RE.sub(" ", body).strip()

    if max_body_chars and len(body) > max_body_chars:
        body = body[:max_body_chars]

    combined = f"{title}. {body}".strip()
    return _WHITESPACE_RE.sub(" ", combined)
