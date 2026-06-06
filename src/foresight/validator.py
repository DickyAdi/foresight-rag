"""Layer 3 — External State Validator (pure deterministic code, no LLM).

Run structural check first (cheap), then semantic only if structural passes.
Structural catches hard failures; semantic catches soft/relevance failures.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresight.state import PredictedPostState, ValidationResult


# TODO: implement validate_state_structural(predicted: PredictedPostState, actual_output: dict) -> tuple[bool, list[str]]
#   errors: EMPTY_RETRIEVAL, MISSING_SOURCE:<title>, INSUFFICIENT_CHUNKS

# TODO: implement validate_state_semantic(query: str, actual_output: dict, threshold: float) -> tuple[bool, list[str]]
#   uses SentenceTransformer("all-MiniLM-L6-v2") + cosine_similarity
#   errors: LOW_RELEVANCE:<score>, POOR_RETRIEVAL_QUALITY:<score>

# TODO: implement validate_state(predicted, actual_output, query) -> ValidationResult
#   compose: structural first, early-return on hard fail, else semantic
