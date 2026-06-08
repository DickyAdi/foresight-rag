"""Offline unit tests for eval metrics (pure functions, no network)."""
from foresight.eval.metrics import (
    exact_match,
    normalize_answer,
    retrieval_precision,
    retrieval_recall,
    token_f1,
)


def test_normalize_strips_articles_punct_case():
    assert normalize_answer("The Beatles!") == "beatles"
    assert normalize_answer("  an  Apple. ") == "apple"


def test_exact_match_is_normalized():
    assert exact_match("The  Beatles", "beatles") == 1.0
    assert exact_match("Rolling Stones", "Beatles") == 0.0


def test_token_f1_partial_overlap():
    # prediction shares 2 of its 2 tokens with a 3-token gold -> P=1.0, R=2/3 -> F1=0.8
    assert token_f1("Guido van", "Guido van Rossum") == 0.8


def test_token_f1_no_overlap_and_exact():
    assert token_f1("cat", "dog") == 0.0
    assert token_f1("Ada Lovelace", "ada lovelace") == 1.0


def test_retrieval_recall():
    assert retrieval_recall(["A", "B", "X"], ["A", "B"]) == 1.0
    assert retrieval_recall(["A", "X"], ["A", "B"]) == 0.5
    assert retrieval_recall([], ["A"]) == 0.0


def test_retrieval_precision():
    assert retrieval_precision(["A", "B"], ["A", "B"]) == 1.0
    assert retrieval_precision(["A", "X", "Y"], ["A", "B"]) == 1 / 3
    assert retrieval_precision([], ["A"]) == 0.0
