"""Foresight strategies (design/03) — strategy pattern, injected into Planner.

Both return a single committed PlanDecision so the graph topology is identical
across strategies; only the internals differ.
"""
from __future__ import annotations
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from foresight.core.types import Chunk, PlanDecision
    from foresight.ports import ChatModel


class PlanningStrategy(Protocol):
    def decide(self, question: str, working_memory: list["Chunk"],
               feedback: str | None) -> "PlanDecision": ...


# TODO: class ImaginedStrategy:  (cheap)
#   one cheap-model call -> K candidate queries + imagined post-states + viability;
#   prune (preconditions in code -> confidence gate); commit top-1.

# TODO: class GroundedStrategy:  (deep)
#   generate K candidate queries (one cheap call) -> actually retrieve each ->
#   score each by RelevanceScorer (continuous) -> commit best.
#   exploratory retrieves are scoring-only/discarded; winner re-flows via executor
#   (query->embedding cache avoids re-embed).
