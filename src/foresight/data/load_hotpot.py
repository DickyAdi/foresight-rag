"""Load and filter HotPotQA (distractor config) for eval.

HuggingFace stores supporting_facts and context as column-oriented parallel arrays,
not as lists of dicts. This module adapts them to the row-oriented shape the rest
of the codebase expects, filters to bridge+hard, and caches to a committed
`cases.json` so repeat runs are fast, offline, and insulated from datasets-version
load-path churn (datasets 5.x is parquet-based). See design/07-data-pipeline-hotpotqa.md.

Expected output shape per row:
  {
    "id": str,
    "question": str,
    "answer": str,                # gold — eval only
    "type": "bridge",
    "level": "hard",
    "supporting_facts": [{"title": str, "sent_id": int}, ...],   # gold — eval only
    "context": [{"title": str, "sentences": [str, ...]}, ...],   # retrieval pool
  }
"""
from __future__ import annotations

import json
from pathlib import Path

_CACHE_PATH = Path(__file__).parent / "cases.json"

_DATASET = "hotpotqa/hotpot_qa"
_CONFIG = "distractor"
_SPLIT = "validation"
_TYPE = "bridge"
_LEVEL = "hard"


def load_hotpot(n: int = 20, *, use_cache: bool = True) -> list[dict]:
    """Return the first `n` bridge+hard validation cases (row-oriented).

    Reads the committed `cases.json` when present and large enough; otherwise
    rebuilds from HuggingFace and rewrites the cache.
    """
    if use_cache and _CACHE_PATH.exists():
        cases = json.loads(_CACHE_PATH.read_text())
        if len(cases) >= n:
            return cases[:n]

    cases = _build_from_hf(n)
    _CACHE_PATH.write_text(json.dumps(cases, indent=2, ensure_ascii=False))
    return cases


def _build_from_hf(n: int) -> list[dict]:
    # Lazy import: keeps the module (and offline tests against the cache) free of
    # the datasets dependency and the network until an actual rebuild is needed.
    from datasets import load_dataset

    ds = load_dataset(_DATASET, _CONFIG, split=_SPLIT)
    cases: list[dict] = []
    for row in ds:
        if row["type"] == _TYPE and row["level"] == _LEVEL:
            cases.append(_adapt_row(row))
            if len(cases) >= n:
                break
    return cases


def _adapt_row(row: dict) -> dict:
    """Adapt one HF column-oriented row to the canonical row-of-dicts shape."""
    sf = row["supporting_facts"]
    supporting_facts = [
        {"title": title, "sent_id": sent_id}
        for title, sent_id in zip(sf["title"], sf["sent_id"])
    ]
    ctx = row["context"]
    context = [
        {"title": title, "sentences": list(sentences)}
        for title, sentences in zip(ctx["title"], ctx["sentences"])
    ]
    return {
        "id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "supporting_facts": supporting_facts,
        "context": context,
    }
