"""Build adapters from Config. Secrets come from .env via adapters._secrets.

This is where concrete adapters are wired to the ports; orchestration asks the
factory for ChatModel/Embedder/VectorStore instances and never touches langchain.
See design/12-implementation-architecture.md.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

from foresight.adapters._secrets import require_api_key
from foresight.adapters.chroma_store import ChromaStore
from foresight.adapters.openrouter_chat import OpenRouterChat
from foresight.adapters.openrouter_embeddings import OpenRouterEmbedder

if TYPE_CHECKING:
    from foresight.config import Config
    from foresight.ports import ChatModel, Embedder, VectorStore


def make_planner_chat(cfg: Config) -> ChatModel:
    return OpenRouterChat(cfg.planner_model, cfg)


def make_executor_chat(cfg: Config) -> ChatModel:
    return OpenRouterChat(cfg.executor_model, cfg)


def make_reflector_chat(cfg: Config) -> ChatModel:
    return OpenRouterChat(cfg.reflector_model, cfg)


def make_embedder(cfg: Config) -> Embedder:
    return OpenRouterEmbedder(cfg)


def make_store(collection_name: str) -> VectorStore:
    return ChromaStore(collection_name)


def make_raw_chat(model: str, cfg: Config) -> ChatOpenAI:
    """Raw ChatOpenAI for the baseline ReAct agent, which needs native tool-calling
    (bind_tools) that the ChatModel port deliberately doesn't expose. Orchestration-only;
    the framework arm uses the port. Mirrors OpenRouterChat's construction."""
    return ChatOpenAI(
        model=model, base_url=cfg.openrouter_base_url, api_key=require_api_key(),
        temperature=cfg.temperature, max_retries=cfg.max_retries,
    )
