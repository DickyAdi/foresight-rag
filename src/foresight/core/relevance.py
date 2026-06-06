"""RelevanceScorer — per-question relative relevance (design/04).

Pure core logic: cosine via numpy, depends on the Embedder port. Used by BOTH the
validator (as a gate at quantile Q) and the grounded-rollout planner (to rank
candidate queries). One function, two consumers — exposes a continuous score.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresight.ports import Embedder

# TODO: class RelevanceScorer:
#   def __init__(self, embedder: Embedder, quantile_q: float): ...
#
#   def score(self, query_emb, retrieved_embs, pool_embs) -> float:
#     # cosine(query, each pool doc) -> distribution
#     # cosine(query, best retrieved chunk) -> top
#     # return the quantile position of `top` within the pool distribution (0..1)
#     # (reuse pool vectors from the store via get_embeddings() — no re-encode)
#
#   def passes(self, score: float) -> bool:
#     return score >= self.quantile_q
#
# Note: cosine with numpy only (no scikit-learn).
