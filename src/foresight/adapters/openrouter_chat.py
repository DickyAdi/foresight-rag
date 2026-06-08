"""ChatModel adapter — OpenRouter via langchain-openai ChatOpenAI.

The ONLY place langchain-openai is imported. Implements the ChatModel port so core
never sees ChatOpenAI. with_structured_output is used when a schema is requested.
See design/09-models-and-config.md and design/12-implementation-architecture.md.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

from foresight.adapters._secrets import require_api_key

if TYPE_CHECKING:
    from foresight.config import Config


class OpenRouterChat:
    """ChatModel port impl backed by ChatOpenAI pointed at OpenRouter."""

    def __init__(self, model: str, cfg: Config) -> None:
        self._llm = ChatOpenAI(
            model=model,
            base_url=cfg.openrouter_base_url,
            api_key=require_api_key(),
            temperature=cfg.temperature,
            max_retries=cfg.max_retries,
        )

    def complete(self, messages: list[dict], *, schema: type | None = None) -> object:
        """Invoke the model. With a pydantic `schema`, return a structured instance."""
        llm = self._llm.with_structured_output(schema) if schema else self._llm
        return llm.invoke(messages)
