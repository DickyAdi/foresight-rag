"""Sanity tests for the HotPotQA loader and HF schema adapter.

These run against the committed `cases.json` cache and stay fully offline — they
skip (rather than hit the network) until the cache has been built by one online run.
"""
import pytest

from foresight.data.load_hotpot import _CACHE_PATH, load_hotpot

pytestmark = pytest.mark.skipif(
    not _CACHE_PATH.exists(),
    reason="cases.json not built yet — run load_hotpot() once online to cache",
)


def test_returns_list_of_dicts():
    cases = load_hotpot(n=3)
    assert isinstance(cases, list)
    assert len(cases) == 3
    assert all(isinstance(c, dict) for c in cases)


def test_all_bridge_hard():
    cases = load_hotpot(n=3)
    assert all(c["type"] == "bridge" and c["level"] == "hard" for c in cases)


def test_supporting_facts_row_oriented():
    case = load_hotpot(n=1)[0]
    sf = case["supporting_facts"]
    assert isinstance(sf, list)
    assert all(set(item) == {"title", "sent_id"} for item in sf)


def test_context_row_oriented():
    case = load_hotpot(n=1)[0]
    ctx = case["context"]
    assert isinstance(ctx, list)
    assert all(set(item) == {"title", "sentences"} for item in ctx)
    assert all(isinstance(item["sentences"], list) for item in ctx)


def test_required_fields_present():
    case = load_hotpot(n=1)[0]
    required = {"id", "question", "answer", "type", "level", "supporting_facts", "context"}
    assert required <= set(case)
