"""VectorStore adapter — per-question store via raw chromadb (EphemeralClient).

The ONLY place chromadb is imported. Raw chromadb (not langchain-chroma) so we can
add() precomputed embeddings and get(include=["embeddings"]) them back for the
validator's vector reuse (design/06). Fresh unique collection per question; dropped
after the case. Cosine space, so query `score` = 1 - distance is cosine similarity.

The adapter is intentionally dumb: the title-prefix embedding and metadata shape
(title/sent_id/article_idx/raw_sentence) are built by the caller (orchestration/tools).
"""
from __future__ import annotations

import chromadb

from foresight.core.types import Chunk


class ChromaStore:
    """VectorStore port impl over an ephemeral, per-question Chroma collection."""

    def __init__(self, collection_name: str) -> None:
        self._client = chromadb.EphemeralClient()
        self._name = collection_name
        self._col = self._client.create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self._col.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(self, embedding: list[float], k: int) -> list[Chunk]:
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            Chunk(
                title=meta.get("title", ""),
                text=doc,
                sent_id=meta.get("sent_id"),
                article_idx=meta.get("article_idx"),
                score=1.0 - dist,  # cosine space: distance in [0,2] -> similarity
            )
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def get_embeddings(self) -> list[list[float]]:
        res = self._col.get(include=["embeddings"])
        return [list(vec) for vec in res["embeddings"]]

    def drop(self) -> None:
        self._client.delete_collection(self._name)
