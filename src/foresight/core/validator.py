"""Layer 3 — External State Validator (pure deterministic code, no LLM).

Gold-free in the loop (design/04, design/11). Structural first (cheap), then the
per-question relative relevance check via RelevanceScorer.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresight.core.relevance import RelevanceScorer
    from foresight.core.types import Chunk, ValidationResult

# TODO: class Validator:
#   def __init__(self, relevance: RelevanceScorer, min_chunks: int): ...
#
#   def validate_structural(self, accumulated: list[Chunk]) -> tuple[bool, list[str]]:
#     # EMPTY_RETRIEVAL, INSUFFICIENT_CHUNKS  (MISSING_SOURCE dropped -> metrics only)
#
#   def validate(self, query_emb, retrieved, retrieved_embs, pool_embs,
#                accumulated) -> ValidationResult:
#     # structural first; early-return on hard fail; else relevance gate.
#     # populate relevance_score (continuous) for grounded-rollout ranking reuse.
#
# Tests (tests/test_validator.py) must lazy-import the embedding stack to stay offline.
