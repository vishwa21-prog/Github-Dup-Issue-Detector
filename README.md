# Duplicate GitHub Issue Detector

**Problem:** Active repos accumulate duplicate issues faster than maintainers
can triage them — the same bug gets reported five different ways ("app
crashes on launch" / "crash on startup after update" / "closes right after
opening it"), each one re-discussed, re-labeled, and sometimes re-fixed from
scratch, before someone notices and manually closes it as a duplicate. On a
busy repo that's real, recurring maintainer time spent on something a
machine can flag in milliseconds.

**Solution:** A bot that embeds every new issue, retrieves the most
semantically similar existing issues via vector search, and — if the top
match clears a calibrated similarity threshold — automatically comments
with the likely duplicate(s) and applies a `possible-duplicate` label,
leaving the actual close/merge decision to a human maintainer. Ships as a
GitHub Action that runs on every `issues: opened` event, plus a small
FastAPI service for local testing or non-GitHub-Actions CI.

**Outcome (measured, not assumed):** evaluated on a 28-issue / 15-cluster
labeled set (`data/eval/duplicate_pairs.json`, paraphrased near-duplicates +
genuinely unique issues) with the offline TF-IDF backend:

| Threshold | Precision | Recall | F1 | Recall@1 |
|---|---|---|---|---|
| 0.86 (naive default, tuned for dense embeddings) | 0.0 | 0.0 | 0.0 | 1.0 |
| **0.10 (calibrated for this backend)** | **1.00** | **0.95** | **0.98** | 1.0 |

The naive threshold row is included on purpose: it's the actual result of
copying a "reasonable-sounding" 0.86 cosine threshold without calibrating it
for the backend in use — a realistic mistake, and the reason
`scripts/evaluate.py --sweep` exists. See
[Choosing the threshold](#choosing-the-similarity-threshold) below. Run
`python scripts/evaluate.py --sweep` yourself to reproduce these numbers or
regenerate them against your own repo's issue history.

**Tech stack:** Python, FAISS, scikit-learn (TF-IDF) / sentence-transformers,
FastAPI, GitHub REST API, GitHub Actions.

---

## How it works

```
New issue opened
      │
      ▼
┌─────────────────┐  strip markdown/code-blocks/URLs/checkboxes/
│  Text cleaning   │  template boilerplate (src/text_utils.py)
└─────────────────┘
      │
      ▼
┌─────────────────┐  sentence-transformers (dense, paraphrase-aware) or
│   Embedding      │  TF-IDF (offline, lexical) — src/embeddings.py
└─────────────────┘
      │
      ▼
┌─────────────────┐  FAISS IndexFlatIP over every existing issue
│  Vector search   │  (built ahead of time by scripts/build_index.py)
└─────────────────┘
      │  top-k most similar issues + cosine scores
      ▼
┌─────────────────┐  top-1 score >= SIMILARITY_THRESHOLD → duplicate
│  Classification  │  (src/duplicate_detector.py)
└─────────────────┘
      │
      ▼
┌─────────────────┐  comment listing candidate(s) + apply a label
│   Bot action     │  (src/bot_actions.py; DRY_RUN=true just logs)
└─────────────────┘
```

## Project structure

```
dup-issue-detector/
├── src/
│   ├── config.py             # env-driven settings
│   ├── github_client.py      # GitHub REST API wrapper (issues/comments/labels)
│   ├── text_utils.py         # markdown/boilerplate stripping
│   ├── embeddings.py         # sentence-transformers + TF-IDF backends
│   ├── vector_store.py       # FAISS index + JSON metadata, save/load
│   ├── duplicate_detector.py # retrieval + threshold classification
│   ├── bot_actions.py        # verdict -> GitHub comment + label
│   └── evaluation.py         # precision/recall/F1/recall@k harness
├── scripts/
│   ├── build_index.py        # CLI: index every issue in a repo
│   ├── check_new_issue.py    # CLI: check ONE issue (used by the Action)
│   └── evaluate.py           # CLI: run the eval harness, optional threshold sweep
├── api/main.py                # FastAPI: /check /reindex /health (local demo)
├── .github/workflows/
│   └── duplicate-check.yml   # the actual deployed bot
├── data/
│   ├── processed/             # FAISS index + metadata get written here
│   └── eval/duplicate_pairs.json  # labeled clusters for evaluation
├── tests/test_pipeline.py     # unit tests, offline (TF-IDF backend)
├── requirements.txt / Dockerfile / .env.example
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: GITHUB_TOKEN, GITHUB_REPO=owner/repo
```

`EMBEDDING_BACKEND=sentence-transformers` (the default) downloads
`all-MiniLM-L6-v2` on first use. Set `EMBEDDING_BACKEND=tfidf` for a
zero-download, fully offline alternative — this is what the shipped GitHub
Action uses, since Action runners re-downloading a model on every single
new issue would be slow and wasteful.

## Usage

### 1. Try it locally against a real repo

```bash
python scripts/build_index.py                 # indexes every issue in GITHUB_REPO
uvicorn api.main:app --reload --port 8000
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{"title": "App crashes on startup", "body": "Crashes immediately every time I open it."}'
```

### 2. Deploy the bot to a real repo

Copy `.github/workflows/duplicate-check.yml` into your repo. That's it —
no server to host. On every new issue it downloads the latest saved index
(rebuilt nightly by the scheduled job in the same workflow), classifies the
new issue, and comments + labels if it looks like a duplicate. Set
`DRY_RUN=true` in the workflow env while you trust-verify it on your repo
before letting it post real comments.

### 3. Evaluate detection quality

```bash
python scripts/evaluate.py --sweep --backend tfidf
python scripts/evaluate.py --sweep --backend sentence-transformers
```

Writes `data/eval/results/metrics.json` and prints a precision/recall/F1
table across a threshold sweep — this is how the numbers in this README
were produced. Extend `data/eval/duplicate_pairs.json` with real
issues mined from your own repo for a number that's representative of your
actual duplicate patterns, rather than this hand-authored starter set.

### 4. Run unit tests

```bash
pytest tests/ -v
```

## Choosing the similarity threshold

This isn't a cosmetic detail — a backend's raw cosine similarity distribution
determines what a "good" threshold even means:

- **Dense embeddings (sentence-transformers)**: paraphrases of the same
  issue land at very high cosine similarity (often 0.8+), because the model
  was trained to place semantically similar sentences close together
  regardless of exact wording. A threshold around 0.80–0.90 is a reasonable
  starting point.
- **TF-IDF**: similarity is driven by literal shared vocabulary. Two
  well-written paraphrases of the same bug can share almost no words
  ("crashes on launch" vs. "closes right after opening it") and still score
  quite low — 0.08–0.15 in this project's eval set — even though they're
  unambiguous duplicates. Reusing a dense-embedding threshold here silently
  produces zero detections (see the 0.86 row in the Outcome table above).

**Always run `scripts/evaluate.py --sweep` against your chosen backend
before deploying**, rather than trusting a threshold that "sounds right."

## Design notes / things to highlight when discussing this project

- **Two swappable embedding backends** behind one interface, chosen so the
  bot can run either fully offline in CI (TF-IDF, zero downloads, fast) or
  with higher-quality paraphrase matching when that's worth the one-time
  model download (sentence-transformers).
- **Boilerplate-aware text cleaning**: issue templates, checkboxes, stack
  traces in code blocks, and HTML comments are stripped before embedding —
  without this, two unrelated issues that both happen to include the same
  template scaffolding can look artificially similar.
- **Leave-one-out evaluation protocol**: every issue is queried against an
  index of every *other* issue, avoiding the trivial "an issue always
  matches itself perfectly" degenerate case that would otherwise make the
  evaluation meaningless.
- **A calibration bug caught by the eval harness, not left in**: the
  Outcome table above intentionally shows the failure mode of an
  uncalibrated threshold (0 detections at 0.86 with TF-IDF) next to the
  fix, because that's a realistic and easy mistake to make with
  similarity-based systems, and the harness is what catches it.
- **Runs as a GitHub Action, not a hosted service**: no server to deploy,
  scale, or pay for — the whole bot lives inside the target repo's own CI.
- **Human-in-the-loop by design**: the bot flags and labels, it never closes
  or merges issues itself.

## Possible extensions

- Combine TF-IDF (lexical) and sentence-transformers (semantic) scores for
  a hybrid ranker — often more robust than either alone.
- Cluster the label graph over time to catch *chains* of duplicates (A
  dup-of B, B dup-of C) rather than only pairwise nearest neighbor.
- Add a `/feedback` endpoint so maintainers dismissing a false-positive
  label feeds back into threshold tuning.
- Extend to cross-repo duplicate detection for organizations with many
  related repos.

## License

MIT — see `LICENSE`.
