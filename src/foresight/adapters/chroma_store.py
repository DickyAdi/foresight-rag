"""VectorStore adapter — per-question store via raw chromadb (EphemeralClient).

The ONLY place chromadb is imported. Raw chromadb (not langchain-chroma) so we can
add() precomputed embeddings and get(include=["embeddings"]) them back for the
validator's vector reuse (design/06). Fresh unique collection per question; dropped
after the case.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresight.core.types import Chunk

# TODO: class ChromaStore:  # implements ports.VectorStore
#   def __init__(self, collection_name: str):
#     self._client = chromadb.EphemeralClient()
#     self._col = self._client.create_collection(collection_name)
#   def add(self, ids, embeddings, documents, metadatas): ...   # precomputed embeddings
#   def query(self, embedding, k) -> list[Chunk]: ...           # -> Chunk via metadata
#   def get_embeddings(self) -> list[list[float]]: ...          # col.get(include=["embeddings"])
#   def drop(self) -> None: ...                                 # delete collection after case
