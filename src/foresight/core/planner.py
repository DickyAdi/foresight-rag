"""Layer 2 — Planner / Foresight (design/03).

Iterative MPC: commit one action at a time. Delegates candidate generation/scoring
to the injected PlanningStrategy. Enforces preconditions in code (the one non-LLM
gate) and the allow_direct_answer / max_hops guards.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresight.config import Config
    from foresight.core.strategies import PlanningStrategy
    from foresight.core.types import Chunk, PlanDecision
    from foresight.ports import ChatModel

# TODO: class Planner:
#   def __init__(self, chat: ChatModel, strategy: PlanningStrategy, cfg: Config): ...
#
#   def plan(self, question, working_memory, feedback=None, hop=0) -> PlanDecision:
#     # precondition prune: if working_memory empty and not cfg.allow_direct_answer,
#     #   forbid 'answer'; if hop >= cfg.max_hops, force 'answer'.
#     # else delegate to strategy.decide(...)
