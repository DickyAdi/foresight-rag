"""Layer 4 — LLM Reflector (design/05).

Strong-model alignment check. Gold-blind. Observational in v1 (records score,
triggers no re-plan).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresight.core.types import Chunk, Reflection
    from foresight.ports import ChatModel

# TODO: class Reflector:
#   def __init__(self, chat: ChatModel): ...
#
#   def reflect(self, question: str, final_answer: str,
#               working_memory: list[Chunk]) -> Reflection:
#     # render("reflector", ...) -> structured output {alignment_score, reasoning}
#     # NEVER receives the gold answer.
