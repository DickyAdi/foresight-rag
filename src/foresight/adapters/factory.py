"""Build adapters from Config. Loads OPENROUTER_API_KEY from .env (python-dotenv).

This is where concrete adapters are wired to the ports; orchestration asks the
factory for ChatModel/Embedder/VectorStore instances and never touches langchain.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresight.config import Config
    from foresight.ports import ChatModel, Embedder, VectorStore

# TODO: load_dotenv() + read OPENROUTER_API_KEY (raise clear error if missing)
# TODO: def make_planner_chat(cfg) -> ChatModel:  OpenRouterChat(cfg.planner_model, cfg)
# TODO: def make_executor_chat(cfg) -> ChatModel: OpenRouterChat(cfg.executor_model, cfg)
# TODO: def make_reflector_chat(cfg) -> ChatModel: OpenRouterChat(cfg.reflector_model, cfg)
# TODO: def make_embedder(cfg) -> Embedder:        OpenRouterEmbedder(cfg)
# TODO: def make_store(collection_name) -> VectorStore: ChromaStore(collection_name)
