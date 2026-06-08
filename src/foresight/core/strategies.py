"""Foresight strategies (design/03) — strategy pattern, injected into Planner.

Both return a single committed PlanDecision so the graph topology is identical
across strategies; only the internals differ:

- Imagined (cheap): one cheap-model call -> K candidate queries with LLM-imagined
  viability; prune (non-empty-query precondition) -> commit top viability.
- Grounded (deep): same K candidates, then actually retrieve each and rank by the
  SHARED Layer-3 relevance function scored against the QUESTION (design/04's "one
  relevance function, two consumers"): the candidate whose real retrieval is most
  question-relevant wins. Exploratory retrieves are scoring-only and discarded; the
  winner re-flows through the executor (the embedder's query cache avoids re-embed).

Both honour the planner's elective-finalize signal (`ready_to_answer`).

Note: grounded scores against the question, not each candidate's own query — scoring
against the retrieving query saturates the quantile to 1.0 for every candidate, which
would make the ranker uniform (and the validator gate a no-op). Retrieved-chunk vectors
are re-embedded with the same title-prefix as the pool (pool-vector reuse deferred).
"""
from __future__ import annotations
from typing import Protocol, TYPE_CHECKING

from foresight.core.prompts import render
from foresight.core.types import CandidateQuery, PlanDecision, PlannerCandidates

if TYPE_CHECKING:
    from foresight.config import Config
    from foresight.core.relevance import RelevanceScorer
    from foresight.core.types import Chunk
    from foresight.ports import ChatModel, Embedder, VectorStore


class PlanningStrategy(Protocol):
    def decide(self, question: str, working_memory: list["Chunk"],
               feedback: str | None) -> "PlanDecision": ...


def _propose(chat: ChatModel, cfg: Config, question: str,
             working_memory: list[Chunk], feedback: str | None) -> PlannerCandidates:
    """One cheap-model call -> K candidate queries (+ elective-finalize signal)."""
    messages = [
        {"role": "system", "content": render("strips_tools_system")},
        {"role": "user", "content": render(
            "planner", question=question, working_memory=working_memory,
            beam_width=cfg.beam_width, feedback=feedback,
        )},
    ]
    return chat.complete(messages, schema=PlannerCandidates)


def _viable(candidates: list[CandidateQuery]) -> list[CandidateQuery]:
    """Precondition prune (code, not LLM): a non-empty, specific query must exist."""
    return [c for c in candidates if c.query and c.query.strip()]


def _commit(candidate: CandidateQuery, viability: float) -> PlanDecision:
    return PlanDecision(
        action="retrieve",
        query=candidate.query,
        predicted_post_state=candidate.predicted_post_state,
        viability=viability,
        reasoning=candidate.reasoning,
    )


def _finalize() -> PlanDecision:
    """Elective finalize — the Planner's gate coerces this back to retrieve if the
    working memory is empty and allow_direct_answer is off."""
    return PlanDecision(action="answer", reasoning="planner: evidence sufficient")


def _fallback(question: str) -> PlanDecision:
    """No usable candidate came back — keep the loop alive by retrieving the question."""
    return PlanDecision(
        action="retrieve", query=question, viability=0.0,
        reasoning="fallback: no viable candidate; retrieving with the question",
    )


def _prefixed(chunk: Chunk, cfg: Config) -> str:
    return f"{chunk.title}: {chunk.text}" if cfg.embed_title_prefix else chunk.text


class ImaginedStrategy:
    """Cheap foresight: commit the candidate with the highest LLM-imagined viability."""

    def __init__(self, chat: ChatModel, cfg: Config) -> None:
        self._chat = chat
        self._cfg = cfg

    def decide(self, question: str, working_memory: list[Chunk],
               feedback: str | None = None) -> PlanDecision:
        proposal = _propose(self._chat, self._cfg, question, working_memory, feedback)
        if proposal.ready_to_answer:
            return _finalize()
        viable = _viable(proposal.candidates)
        if not viable:
            return _fallback(question)
        best = max(viable, key=lambda c: c.viability)
        return _commit(best, best.viability)


class GroundedStrategy:
    """Deep foresight: retrieve each candidate, commit the one whose real retrieval is
    most relevant to the QUESTION (the shared Layer-3 relevance score)."""

    def __init__(self, chat: ChatModel, embedder: Embedder, store: VectorStore,
                 relevance: RelevanceScorer, cfg: Config) -> None:
        self._chat = chat
        self._embedder = embedder
        self._store = store
        self._relevance = relevance
        self._cfg = cfg

    def decide(self, question: str, working_memory: list[Chunk],
               feedback: str | None = None) -> PlanDecision:
        proposal = _propose(self._chat, self._cfg, question, working_memory, feedback)
        if proposal.ready_to_answer:
            return _finalize()
        viable = _viable(proposal.candidates)
        if not viable:
            return _fallback(question)

        question_emb = self._embedder.embed_query(question)  # cached
        pool_embs = self._store.get_embeddings()

        best: PlanDecision | None = None
        best_viability = -1.0
        for candidate in viable:
            cand_emb = self._embedder.embed_query(candidate.query)  # cached for the winner
            chunks = self._store.query(cand_emb, k=self._cfg.top_k)  # scoring-only, discarded
            retrieved_embs = (
                self._embedder.embed_documents([_prefixed(c, self._cfg) for c in chunks])
                if chunks else []
            )
            viability = self._relevance.score(question_emb, retrieved_embs, pool_embs)
            if viability > best_viability:
                best_viability = viability
                best = _commit(candidate, viability)
        return best
