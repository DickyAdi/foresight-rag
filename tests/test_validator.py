"""Unit tests for the Layer 3 state validator (deterministic, no LLM, no network).

RelevanceScorer takes precomputed vectors, so these never import the embedding
stack — they stay offline and fast by construction (design/04).
"""
from foresight.core.relevance import RelevanceScorer
from foresight.core.types import Chunk
from foresight.core.validator import (
    EMPTY_RETRIEVAL,
    INSUFFICIENT_CHUNKS,
    LOW_RELEVANCE,
    Validator,
)


def _chunks(n: int) -> list[Chunk]:
    return [Chunk(title=f"T{i}", text=f"sentence {i}", sent_id=i) for i in range(n)]


# A pool with one clearly-relevant doc ([1,0]), one orthogonal, one opposite, one
# diagonal. Similarities to query [1,0] are [1.0, 0.0, -1.0, 0.707].
_POOL = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.7071, 0.7071]]
_QUERY = [1.0, 0.0]


def _scorer(q: float = 0.7) -> RelevanceScorer:
    return RelevanceScorer(embedder=None, quantile_q=q)


# --- RelevanceScorer ---

def test_relevance_top_beats_whole_pool():
    score = _scorer().score(_QUERY, retrieved_embs=[[1.0, 0.0]], pool_embs=_POOL)
    assert score == 1.0
    assert _scorer().passes(score)


def test_relevance_irrelevant_chunk_fails_gate():
    score = _scorer().score(_QUERY, retrieved_embs=[[-1.0, 0.0]], pool_embs=_POOL)
    assert score == 0.25  # only beats itself in the pool distribution
    assert not _scorer().passes(score)


def test_relevance_empty_retrieval_scores_zero():
    assert _scorer().score(_QUERY, retrieved_embs=[], pool_embs=_POOL) == 0.0


# --- validate_structural ---

def test_structural_empty_retrieval():
    ok, errors = Validator(_scorer(), min_chunks=2).validate_structural([])
    assert not ok
    assert errors == [EMPTY_RETRIEVAL]


def test_structural_insufficient_chunks():
    ok, errors = Validator(_scorer(), min_chunks=3).validate_structural(_chunks(1))
    assert not ok
    assert errors == [INSUFFICIENT_CHUNKS]


def test_structural_passes():
    ok, errors = Validator(_scorer(), min_chunks=2).validate_structural(_chunks(2))
    assert ok
    assert errors == []


# --- validate (composed) ---

def test_composed_passes_both():
    result = Validator(_scorer(), min_chunks=2).validate(
        query_emb=_QUERY,
        retrieved_embs=[[1.0, 0.0]],
        pool_embs=_POOL,
        accumulated=_chunks(2),
    )
    assert result.ok
    assert result.errors == []
    assert result.relevance_score == 1.0


def test_composed_low_relevance_flag():
    result = Validator(_scorer(), min_chunks=2).validate(
        query_emb=_QUERY,
        retrieved_embs=[[-1.0, 0.0]],
        pool_embs=_POOL,
        accumulated=_chunks(2),
    )
    assert not result.ok
    assert result.errors == [LOW_RELEVANCE]
    assert result.relevance_score == 0.25  # continuous score still surfaced


def test_composed_hard_fail_skips_relevance():
    class _ExplodingRelevance:
        quantile_q = 0.7

        def score(self, *a, **k):
            raise AssertionError("relevance must not run on a structural hard fail")

        def passes(self, s):
            raise AssertionError("passes must not run on a structural hard fail")

    result = Validator(_ExplodingRelevance(), min_chunks=2).validate(
        query_emb=_QUERY,
        retrieved_embs=[[1.0, 0.0]],
        pool_embs=_POOL,
        accumulated=[],  # empty → structural hard fail, relevance never reached
    )
    assert not result.ok
    assert result.errors == [EMPTY_RETRIEVAL]
    assert result.relevance_score is None
