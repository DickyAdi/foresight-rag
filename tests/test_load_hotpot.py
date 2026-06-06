"""Sanity tests for the HotPotQA loader and HF schema adapter."""
import pytest

# TODO: from foresight.data.load_hotpot import load_hotpot


def test_returns_list_of_dicts():
    # TODO: cases = load_hotpot(n=3)
    # TODO: assert isinstance(cases, list) and len(cases) == 3
    pytest.skip("not implemented yet")


def test_all_bridge_hard():
    # TODO: all rows are type=="bridge" and level=="hard"
    pytest.skip("not implemented yet")


def test_supporting_facts_row_oriented():
    # TODO: row["supporting_facts"] is list of dicts with keys "title" and "sent_id"
    # TODO: NOT {"title": [...], "sent_id": [...]}
    pytest.skip("not implemented yet")


def test_context_row_oriented():
    # TODO: row["context"] is list of dicts with keys "title" and "sentences"
    pytest.skip("not implemented yet")


def test_required_fields_present():
    # TODO: each row has: id, question, answer, type, level, supporting_facts, context
    pytest.skip("not implemented yet")
