"""Thin wrapper around the GitHub REST API (issues, comments, labels).

Deliberately dependency-light (raw `requests`) rather than PyGithub, so the
same client works unmodified inside a GitHub Action runner where only the
built-in GITHUB_TOKEN is available.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from src.config import settings


class GitHubClient:
    def __init__(self, token: Optional[str] = None, repo: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.token = token or settings.github_token
        self.repo = repo or settings.repo
        self.base_url = (base_url or settings.api_base_url).rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.token}" if self.token else "",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    # ---------------------------------------------------------------- #
    # Reads
    # ---------------------------------------------------------------- #
    def list_issues(self, state: str = "all", per_page: int = 100,
                     max_pages: int = 20) -> List[Dict[str, Any]]:
        """Lists issues (GitHub's /issues endpoint also returns PRs; we
        filter those out since duplicate-detection only applies to issues).
        """
        issues: List[Dict[str, Any]] = []
        page = 1
        while page <= max_pages:
            resp = self._session.get(
                f"{self.base_url}/repos/{self.repo}/issues",
                params={"state": state, "per_page": per_page, "page": page},
                timeout=30,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            issues.extend(i for i in batch if "pull_request" not in i)
            if len(batch) < per_page:
                break
            page += 1
        return issues

    def get_issue(self, number: int) -> Dict[str, Any]:
        resp = self._session.get(
            f"{self.base_url}/repos/{self.repo}/issues/{number}", timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    # ---------------------------------------------------------------- #
    # Writes (bot actions)
    # ---------------------------------------------------------------- #
    def add_comment(self, number: int, body: str) -> Dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/repos/{self.repo}/issues/{number}/comments",
            json={"body": body},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def add_labels(self, number: int, labels: List[str]) -> Dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/repos/{self.repo}/issues/{number}/labels",
            json={"labels": labels},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
