"""Evaluates duplicate detection quality against a labeled dataset of issue
clusters (data/eval/duplicate_pairs.json): each cluster is a group of
issues that are true duplicates of each other; issues in different
clusters are not duplicates. Singleton clusters (size 1) represent
genuinely unique issues that should NOT be flagged.

Evaluation protocol is leave-one-out: for every issue, build the index from
every OTHER issue, then query. This avoids the trivial "issue matches
itself with score 1.0" degenerate case.

Metrics reported:
- recall@k (k in {1,3,5}): of issues that DO have a true duplicate
  somewhere in the dataset, what fraction have at least one true duplicate
  in their top-k retrieved candidates? (ranking quality, threshold-free)
- precision / recall / F1 @ threshold: treating "top-1 candidate score >=
  threshold" as a duplicate prediction, compared against ground truth.
  This is the operating point the bot actually uses in production.
- a small threshold sweep, to show precision/recall trade-off and justify
  the configured SIMILARITY_THRESHOLD.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from src.duplicate_detector import DuplicateDetector
from src.embeddings import Embedder
from src.text_utils import clean_issue_text
from src.vector_store import IssueVectorStore


@dataclass
class FlatIssue:
    number: int
    cluster_id: str
    title: str
    body: str


def load_eval_dataset(path: str) -> List[FlatIssue]:
    with open(path) as f:
        payload = json.load(f)

    flat: List[FlatIssue] = []
    number = 1
    for cluster in payload["clusters"]:
        for issue in cluster["issues"]:
            flat.append(FlatIssue(
                number=number,
                cluster_id=cluster["cluster_id"],
                title=issue["title"],
                body=issue.get("body", ""),
            ))
            number += 1
    return flat


def _build_full_index(embedder: Embedder, issues: List[FlatIssue], max_body_chars: int) -> IssueVectorStore:
    texts = [clean_issue_text(i.title, i.body, max_body_chars) for i in issues]
    embedder.fit(texts)
    vectors = embedder.encode(texts)
    store = IssueVectorStore(dim=embedder.dim)
    metadata = [{"number": i.number, "title": i.title, "url": "", "cluster_id": i.cluster_id} for i in issues]
    store.add(vectors, metadata)
    return store


def evaluate_detector(
    embedder_factory: Callable[[], Embedder],
    issues: List[FlatIssue],
    threshold: float,
    k_values: List[int] = [1, 3, 5],
    max_body_chars: int = 2000,
) -> Dict[str, Any]:
    """Leave-one-out evaluation. `embedder_factory` is called once and the
    SAME embedder/index is reused for every query (excluding the query
    issue itself from its own results) — cheaper than truly rebuilding the
    index per issue, and equivalent for this evaluation's purposes since
    embedders here don't depend on which single issue is excluded.
    """
    embedder = embedder_factory()
    store = _build_full_index(embedder, issues, max_body_chars)
    cluster_size = {}
    for i in issues:
        cluster_size[i.cluster_id] = cluster_size.get(i.cluster_id, 0) + 1

    max_k = max(k_values)
    hits_at_k = {k: 0 for k in k_values}
    has_true_duplicate = 0

    tp = fp = fn = tn = 0

    texts = [clean_issue_text(i.title, i.body, max_body_chars) for i in issues]
    query_vectors = embedder.encode(texts)

    for idx, issue in enumerate(issues):
        raw_hits = store.search(query_vectors[idx], top_k=max_k + 1)
        candidates = [(m, s) for m, s in raw_hits if m["number"] != issue.number][:max_k]

        is_true_positive_issue = cluster_size[issue.cluster_id] > 1
        if is_true_positive_issue:
            has_true_duplicate += 1
            for k in k_values:
                topk = candidates[:k]
                if any(c[0]["cluster_id"] == issue.cluster_id for c in topk):
                    hits_at_k[k] += 1

        # Threshold-based classification using only the single best candidate.
        predicted_dup = bool(candidates) and candidates[0][1] >= threshold
        actual_dup_top1_correct = (
            predicted_dup and candidates[0][0]["cluster_id"] == issue.cluster_id
        )

        if predicted_dup and actual_dup_top1_correct:
            tp += 1
        elif predicted_dup and not actual_dup_top1_correct:
            fp += 1
        elif not predicted_dup and is_true_positive_issue:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    recall_at_k = {
        f"recall@{k}": round(hits_at_k[k] / has_true_duplicate, 4) if has_true_duplicate else None
        for k in k_values
    }

    return {
        "n_issues": len(issues),
        "n_clusters": len(cluster_size),
        "n_issues_with_true_duplicate": has_true_duplicate,
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        **recall_at_k,
    }


def sweep_thresholds(
    embedder_factory: Callable[[], Embedder],
    issues: List[FlatIssue],
    thresholds: List[float],
    max_body_chars: int = 2000,
) -> List[Dict[str, Any]]:
    return [
        evaluate_detector(embedder_factory, issues, threshold=t, max_body_chars=max_body_chars)
        for t in thresholds
    ]
