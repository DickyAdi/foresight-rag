"""Layer 3 — External State Validator (pure deterministic code, no LLM).

Gold-free in the loop (design/04, design/11). Structural checks (cheap) run first
against accumulated working memory; on a hard fail we early-return and skip the
relevance gate. Otherwise the per-question relative relevance check runs on the
just-retrieved chunks vs the current query, and its continuous score is surfaced
on the result so the grounded-rollout planner can reuse it for ranking.

`MISSING_SOURCE` is intentionally absent — it needs gold, so it lives in metrics.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from foresight.core.types import ValidationResult

if TYPE_CHECKING:
    from foresight.core.relevance import RelevanceScorer
    from foresight.core.types import Chunk

EMPTY_RETRIEVAL = "EMPTY_RETRIEVAL"
INSUFFICIENT_CHUNKS = "INSUFFICIENT_CHUNKS"
LOW_RELEVANCE = "LOW_RELEVANCE"


class Validator:
    def __init__(self, relevance: RelevanceScorer, min_chunks: int) -> None:
        self._relevance = relevance
        self._min_chunks = min_chunks

    def validate_structural(self, accumulated: list[Chunk]) -> tuple[bool, list[str]]:
        """Coverage checks over accumulated working memory (design/04)."""
        if len(accumulated) == 0:
            return False, [EMPTY_RETRIEVAL]
        if len(accumulated) < self._min_chunks:
            return False, [INSUFFICIENT_CHUNKS]
        return True, []

    def validate(
        self,
        *,
        query_emb,
        retrieved_embs,
        pool_embs,
        accumulated: list[Chunk],
    ) -> ValidationResult:
        """Structural first (hard fail → early return), then the relevance gate."""
        ok, errors = self.validate_structural(accumulated)
        if not ok:
            return ValidationResult(ok=False, errors=errors, relevance_score=None)

        score = self._relevance.score(query_emb, retrieved_embs, pool_embs)
        if not self._relevance.passes(score):
            return ValidationResult(ok=False, errors=[LOW_RELEVANCE], relevance_score=score)
        return ValidationResult(ok=True, errors=[], relevance_score=score)
