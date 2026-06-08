"""RelevanceScorer — per-question relative relevance (design/04).

Pure core logic: cosine via numpy, depends on the Embedder port. Used by BOTH the
validator (as a gate at quantile Q) and the grounded-rollout planner (to rank
candidate queries). One function, two consumers — exposes a continuous score.

The score is the empirical quantile position of the best retrieved chunk within the
query↔pool similarity distribution: "the top chunk must be more relevant than a
fraction Q of the available pool, else retrieval failed." No global magic threshold,
no test leakage — it calibrates per question against that question's own pool.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from foresight.ports import Embedder


class RelevanceScorer:
    def __init__(self, embedder: Embedder, quantile_q: float) -> None:
        # The embedder is the shared single embedding space (design/06). Methods here
        # take precomputed vectors (pool reuse), so it is held as the dependency seam
        # rather than called per scoring.
        self._embedder = embedder
        self.quantile_q = quantile_q

    def score(self, query_emb, retrieved_embs, pool_embs) -> float:
        """Empirical quantile position (0..1) of the best retrieved chunk in the pool.

        1.0 ⇒ the top retrieved chunk is at least as similar to the query as every
        pool doc; 0.0 ⇒ nothing was retrieved (or an empty pool).
        """
        if len(retrieved_embs) == 0 or len(pool_embs) == 0:
            return 0.0
        pool_sims = self._cosine_to_query(query_emb, pool_embs)
        top = float(self._cosine_to_query(query_emb, retrieved_embs).max())
        return float(np.mean(pool_sims <= top))

    def passes(self, score: float) -> bool:
        return score >= self.quantile_q

    @staticmethod
    def _cosine_to_query(query_emb, embs) -> np.ndarray:
        q = np.asarray(query_emb, dtype=float)
        m = np.asarray(embs, dtype=float)
        denom = np.linalg.norm(m, axis=1) * np.linalg.norm(q)
        return (m @ q) / np.where(denom == 0.0, 1.0, denom)
