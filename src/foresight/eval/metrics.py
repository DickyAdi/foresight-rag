"""Answer quality and retrieval quality metrics for HotPotQA evaluation.

Pure functions, gold used ONLY here (design/08, design/11) and applied identically
across arms. Answer metrics use the standard SQuAD/HotPotQA normalization; retrieval
metrics compare accumulated retrieved article titles against gold supporting-fact titles.
"""
from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, strip articles, collapse whitespace (SQuAD style)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between normalized prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        # Both empty -> perfect; exactly one empty -> 0.
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def retrieval_recall(retrieved_titles: list[str], required_titles: list[str]) -> float:
    """|retrieved ∩ required| / |required| — did we hit the needed source articles?"""
    required = set(required_titles)
    if not required:
        return 0.0
    return len(set(retrieved_titles) & required) / len(required)


def retrieval_precision(retrieved_titles: list[str], required_titles: list[str]) -> float:
    """|retrieved ∩ required| / |retrieved| — did we avoid distractors?"""
    retrieved = set(retrieved_titles)
    if not retrieved:
        return 0.0
    return len(retrieved & set(required_titles)) / len(retrieved)
