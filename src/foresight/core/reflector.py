"""Layer 4 — LLM Reflector (design/05).

Strong-model semantic alignment check ("did we answer what was asked?"). Gold-blind:
never receives the gold answer. Observational in v1 — records a score, triggers no
re-plan. Eval correlates this score with answer correctness.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from foresight.core.prompts import render
from foresight.core.types import Reflection

if TYPE_CHECKING:
    from foresight.core.types import Chunk
    from foresight.ports import ChatModel


class Reflector:
    def __init__(self, chat: ChatModel) -> None:
        self._chat = chat

    def reflect(self, question: str, final_answer: str,
                working_memory: list[Chunk]) -> Reflection:
        prompt = render(
            "reflector", question=question, final_answer=final_answer,
            working_memory=working_memory,
        )
        return self._chat.complete([{"role": "user", "content": prompt}], schema=Reflection)
