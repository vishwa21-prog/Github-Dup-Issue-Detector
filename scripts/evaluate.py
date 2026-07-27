"""CLI: evaluates duplicate-detection quality against
data/eval/duplicate_pairs.json and writes results to
data/eval/results/metrics.json.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --sweep            # also sweep thresholds
    python scripts/evaluate.py --backend tfidf     # force a specific backend
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, ".")

from src.config import settings
from src.embeddings import build_embedder
from src.evaluation import evaluate_detector, load_eval_dataset, sweep_thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=settings.eval_dataset_path)
    parser.add_argument("--backend", type=str, default=settings.embedding_backend,
                         choices=["sentence-transformers", "tfidf"])
    parser.add_argument("--threshold", type=float, default=settings.similarity_threshold)
    parser.add_argument("--sweep", action="store_true", help="Also sweep thresholds 0.5-0.95")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    issues = load_eval_dataset(args.dataset)
    print(f"Loaded {len(issues)} issues across "
          f"{len(set(i.cluster_id for i in issues))} clusters from {args.dataset}")

    def embedder_factory():
        return build_embedder(args.backend, settings.embedding_model_name)

    metrics = evaluate_detector(embedder_factory, issues, threshold=args.threshold)
    print("\n=== Evaluation @ configured threshold ===")
    print(json.dumps(metrics, indent=2))

    results_dir = settings.results_dir
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "metrics.json")

    payload = {"backend": args.backend, "at_threshold": metrics}

    if args.sweep:
        # Wide range on purpose: TF-IDF and sentence-transformers cosine
        # scores live in very different bands (see README "Choosing the
        # threshold"), so a narrow high-only range would silently show
        # zero detections for TF-IDF instead of the actual best operating
        # point.
        thresholds = [round(0.05 + 0.05 * i, 2) for i in range(19)]
        sweep = sweep_thresholds(embedder_factory, issues, thresholds)
        payload["threshold_sweep"] = sweep
        print("\n=== Threshold sweep ===")
        print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>8} {'recall@1':>10} {'recall@3':>10}")
        for row in sweep:
            print(f"{row['threshold']:>10} {row['precision']:>10} {row['recall']:>10} "
                  f"{row['f1']:>8} {row.get('recall@1'):>10} {row.get('recall@3'):>10}")

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
