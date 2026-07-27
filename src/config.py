"""Central configuration, driven by environment variables (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _list(name: str, default: List[str]) -> List[str]:
    val = os.getenv(name)
    if not val:
        return default
    return [x.strip() for x in val.split(",") if x.strip()]


@dataclass
class Settings:
    # --- GitHub ---
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    repo: str = os.getenv("GITHUB_REPO", "owner/repo")  # "owner/repo"
    api_base_url: str = os.getenv("GITHUB_API_BASE_URL", "https://api.github.com")

    # --- Embeddings ---
    # "sentence-transformers" (better quality, needs a model download) or
    # "tfidf" (zero-download offline fallback, good enough for small repos
    # and for CI environments without internet access to the HF Hub).
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # --- Duplicate detection ---
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.86"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    issue_state_to_index: str = os.getenv("ISSUE_STATE_TO_INDEX", "all")  # open|closed|all
    max_body_chars: int = int(os.getenv("MAX_BODY_CHARS", "2000"))

    # --- Bot behavior ---
    post_comment: bool = _bool("POST_COMMENT", True)
    add_label: bool = _bool("ADD_LABEL", True)
    duplicate_label: str = os.getenv("DUPLICATE_LABEL", "possible-duplicate")
    dry_run: bool = _bool("DRY_RUN", False)  # log actions instead of calling GitHub

    # --- Storage ---
    index_dir: str = os.getenv("INDEX_DIR", "data/processed")

    # --- Evaluation ---
    eval_dataset_path: str = os.getenv("EVAL_DATASET_PATH", "data/eval/duplicate_pairs.json")
    results_dir: str = os.getenv("RESULTS_DIR", "data/eval/results")


settings = Settings()
