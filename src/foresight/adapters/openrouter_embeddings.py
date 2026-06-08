"""Embedder adapter — OpenRouter embeddings via the raw openai SDK.

The ONLY place the openai SDK is imported for embeddings. Implements the Embedder
port. Same embedding space is used by the store and the validator — never mix
models/dims. A query->embedding cache avoids re-embedding identical query strings
(notably the grounded-rollout winner re-flowing through the executor). design/06.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from openai import OpenAI

from foresight.adapters._secrets import require_api_key

if TYPE_CHECKING:
    from foresight.config import Config


class OpenRouterEmbedder:
    """Embedder port impl backed by the raw openai SDK pointed at OpenRouter."""

    def __init__(self, cfg: Config) -> None:
        self._client = OpenAI(base_url=cfg.openrouter_base_url, api_key=require_api_key())
        self._model = cfg.embedding_model
        self._query_cache: dict[str, list[float]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        # API preserves input order; sort defensively in case a provider does not.
        data = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in data]

    def embed_query(self, text: str) -> list[float]:
        cached = self._query_cache.get(text)
        if cached is not None:
            return cached
        resp = self._client.embeddings.create(model=self._model, input=[text])
        vec = resp.data[0].embedding
        self._query_cache[text] = vec
        return vec
