"""Embedder adapter — OpenRouter embeddings via the raw openai SDK.

The ONLY place the openai SDK is imported for embeddings. Implements the Embedder
port. Verify the /embeddings endpoint responds before building on it (design/06).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresight.config import Config

# TODO: class OpenRouterEmbedder:  # implements ports.Embedder
#   def __init__(self, cfg: Config):
#     self._client = OpenAI(base_url=cfg.openrouter_base_url, api_key=<env>)
#     self._model = cfg.embedding_model
#     self._query_cache: dict[str, list[float]] = {}   # query->embedding cache
#   def embed_documents(self, texts): ...   # client.embeddings.create(model, input=texts)
#   def embed_query(self, text): ...        # cache by string to avoid re-embedding
