"""Unit tests. Deliberately use the tfidf embedding backend so the whole
suite runs offline with no model downloads — CI-friendly and fast.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bot_actions import format_comment
from src.duplicate_detector import DuplicateDetector, DuplicateCandidate, DuplicateVerdict
from src.embeddings import TfidfEmbedder, build_embedder
from src.text_utils import clean_issue_text
from src.vector_store import IssueVectorStore


# --------------------------------------------------------------------------
# text_utils
# --------------------------------------------------------------------------
def test_clean_issue_text_strips_code_blocks_and_urls():
    body = "See ```print('secret_token_abc')``` and https://example.com/x for details."
    cleaned = clean_issue_text("Bug: crash", body)
    assert "secret_token_abc" not in cleaned
    assert "example.com" not in cleaned
    assert cleaned.startswith("Bug: crash.")


def test_clean_issue_text_strips_checkboxes_and_headers():
    body = "### Steps to reproduce\n- [ ] open app\n- [x] click button"
    cleaned = clean_issue_text("t", body)
    assert "###" not in cleaned
    assert "[ ]" not in cleaned
    assert "[x]" not in cleaned


def test_clean_issue_text_truncates_long_bodies():
    body = "word " * 1000
    cleaned = clean_issue_text("t", body, max_body_chars=50)
    assert len(cleaned) < 100  # title + truncated body, well under the untruncated ~5000 chars


# --------------------------------------------------------------------------
# vector_store
# --------------------------------------------------------------------------
def test_vector_store_add_search_roundtrip():
    embedder = TfidfEmbedder()
    texts = ["app crashes on launch", "cannot log in with valid password", "please add dark mode"]
    embedder.fit(texts)
    vecs = embedder.encode(texts)

    store = IssueVectorStore(dim=embedder.dim)
    store.add(vecs, [{"number": i + 1, "title": t, "url": ""} for i, t in enumerate(texts)])

    query = embedder.encode(["the app crashes immediately when I launch it"])[0]
    results = store.search(query, top_k=2)
    assert len(results) == 2
    assert results[0][0]["number"] == 1  # nearest neighbor should be the crash-on-launch issue
    assert results[0][1] >= results[1][1]  # sorted by descending score


def test_vector_store_save_load_roundtrip():
    embedder = TfidfEmbedder()
    texts = ["memory leak over time", "typo in docs"]
    embedder.fit(texts)
    vecs = embedder.encode(texts)

    store = IssueVectorStore(dim=embedder.dim)
    store.add(vecs, [{"number": 1, "title": texts[0], "url": ""}, {"number": 2, "title": texts[1], "url": ""}])

    with tempfile.TemporaryDirectory() as tmp:
        store.save(tmp)
        loaded = IssueVectorStore.load(tmp)
        assert loaded.index.ntotal == 2
        assert loaded.metadata[0]["title"] == texts[0]


# --------------------------------------------------------------------------
# duplicate_detector (end-to-end with tfidf backend, no network)
# --------------------------------------------------------------------------
def _fake_issues():
    return [
        {"number": 1, "title": "App crashes on startup", "body": "Crashes every time I open it.", "html_url": ""},
        {"number": 2, "title": "Cannot log in", "body": "Invalid credentials error with correct password.", "html_url": ""},
        {"number": 3, "title": "Please add dark mode", "body": "Would be nice to have a dark theme.", "html_url": ""},
    ]


def test_duplicate_detector_flags_near_duplicate():
    embedder = build_embedder("tfidf", "unused")
    detector = DuplicateDetector.build_index(embedder, _fake_issues(), threshold=0.2, top_k=3)

    verdict = detector.classify(
        "Application crashes immediately at startup",
        "Every time I launch the app it crashes right away.",
    )
    assert verdict.is_duplicate
    assert verdict.best.number == 1


def test_duplicate_detector_does_not_flag_unrelated_issue():
    embedder = build_embedder("tfidf", "unused")
    detector = DuplicateDetector.build_index(embedder, _fake_issues(), threshold=0.6, top_k=3)

    verdict = detector.classify(
        "Support exporting reports as PDF",
        "It would help to export the monthly report directly as a PDF file.",
    )
    assert not verdict.is_duplicate


def test_duplicate_detector_excludes_self_when_requested():
    embedder = build_embedder("tfidf", "unused")
    detector = DuplicateDetector.build_index(embedder, _fake_issues(), threshold=0.2, top_k=3)

    verdict = detector.classify("App crashes on startup", "Crashes every time I open it.", exclude_number=1)
    # The issue's own (near-identical) text is excluded, so the best match
    # should NOT be issue #1 even though it would trivially match itself.
    assert verdict.best is None or verdict.best.number != 1


# --------------------------------------------------------------------------
# bot_actions
# --------------------------------------------------------------------------
def test_format_comment_lists_all_candidates():
    verdict = DuplicateVerdict(
        is_duplicate=True,
        candidates=[
            DuplicateCandidate(number=5, title="Crash on launch", url="https://x/5", score=0.93),
            DuplicateCandidate(number=9, title="App won't start", url="https://x/9", score=0.81),
        ],
    )
    comment = format_comment(verdict)
    assert "#5" in comment and "#9" in comment
    assert "0.93" in comment
