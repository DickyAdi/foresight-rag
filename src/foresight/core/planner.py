"""Layer 2 — Planner / Foresight (design/03).

Iterative MPC: commit one action at a time. Delegates candidate generation/scoring
to the injected PlanningStrategy and owns the two deterministic (non-LLM) guards:

  - allow_direct_answer: at an empty working memory the planner may NOT answer
    (default), forcing retrieval — protects the retrieval A/B from parametric shortcuts.
  - max_hops: once reached, force a terminal answer (non-termination guard).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from foresight.core.types import PlanDecision

if TYPE_CHECKING:
    from foresight.config import Config
    from foresight.core.strategies import PlanningStrategy
    from foresight.core.types import Chunk


class Planner:
    def __init__(self, strategy: PlanningStrategy, cfg: Config) -> None:
        self._strategy = strategy
        self._cfg = cfg

    def plan(self, question: str, working_memory: list[Chunk],
             feedback: str | None = None, hop: int = 0) -> PlanDecision:
        if hop >= self._cfg.max_hops:
            return PlanDecision(action="answer", reasoning="max_hops reached; finalizing")

        decision = self._strategy.decide(question, working_memory, feedback)

        # Precondition (code-enforced): cannot answer from empty memory unless allowed.
        if (decision.action == "answer" and not working_memory
                and not self._cfg.allow_direct_answer):
            return PlanDecision(
                action="retrieve", query=decision.query or question,
                reasoning="answer forbidden from empty memory; retrieving instead",
            )
        return decision
