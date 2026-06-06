"""Answer quality and retrieval quality metrics for HotPotQA evaluation."""

# TODO: implement normalize_answer(s: str) -> str
#   Standard HotPotQA normalization: lowercase, strip articles ("a", "an", "the"),
#   strip punctuation, normalize whitespace.

# TODO: implement exact_match(prediction: str, ground_truth: str) -> float
#   Returns 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth), else 0.0.

# TODO: implement token_f1(prediction: str, ground_truth: str) -> float
#   Token-level F1 between normalized prediction and ground truth token sets.

# TODO: implement retrieval_recall(retrieved_titles: list[str], required_titles: list[str]) -> float
#   |retrieved ∩ required| / |required|

# TODO: implement retrieval_precision(retrieved_titles: list[str], required_titles: list[str]) -> float
#   |retrieved ∩ required| / |retrieved| (or 0.0 if retrieved is empty)
