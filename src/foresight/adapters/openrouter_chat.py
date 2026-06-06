"""ChatModel adapter — OpenRouter via langchain-openai ChatOpenAI.

The ONLY place langchain-openai is imported. Implements the ChatModel port so core
never sees ChatOpenAI. with_structured_output is used when a schema is requested.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresight.config import Config

# TODO: class OpenRouterChat:  # implements ports.ChatModel
#   def __init__(self, model: str, cfg: Config):
#     self._llm = ChatOpenAI(model=model, base_url=cfg.openrouter_base_url,
#                            api_key=<env>, temperature=cfg.temperature,
#                            max_retries=cfg.max_retries)
#   def complete(self, messages, *, schema=None):
#     llm = self._llm.with_structured_output(schema) if schema else self._llm
#     return llm.invoke(messages)
