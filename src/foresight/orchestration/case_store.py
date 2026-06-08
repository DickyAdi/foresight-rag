"""Populate a per-question vector store from a HotPotQA case (design/06).

One document per sentence; embedded text is "{title}: {sentence}" when
embed_title_prefix is on (fixes coreference findability), while the stored/displayed
document stays the raw sentence and the title lives in metadata only. The whole pool
is embedded in one batch so the vectors can be reused for the validator's relevance
distribution (via store.get_embeddings()).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresight.config import Config
    from foresight.ports import Embedder, VectorStore


def build_case_store(case: dict, embedder: Embedder, store: VectorStore, cfg: Config) -> None:
    """Embed the case's context pool (one doc per sentence) and add it to `store`."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    to_embed: list[str] = []

    for article_idx, article in enumerate(case["context"]):
        title = article["title"]
        for sent_id, sentence in enumerate(article["sentences"]):
            ids.append(f"{article_idx}-{sent_id}")
            documents.append(sentence)  # display = raw sentence
            metadatas.append({
                "title": title, "sent_id": sent_id,
                "article_idx": article_idx, "raw_sentence": sentence,
            })
            to_embed.append(f"{title}: {sentence}" if cfg.embed_title_prefix else sentence)

    embeddings = embedder.embed_documents(to_embed)
    store.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
