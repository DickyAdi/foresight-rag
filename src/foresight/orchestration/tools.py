"""LangChain tools for the BASELINE greedy ReAct loop (design/02).

`finish` is a real bound tool only in the baseline; the framework terminates via the
planner->answer node instead. Both tools close over a per-question VectorStore +
Embedder (the shared embedding space). The canonical STRIPS prose lives in
prompts/strips_tools_system.jinja2; these docstrings stay minimal to avoid drift.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from foresight.config import Config
    from foresight.ports import Embedder, VectorStore


def build_tools(store: VectorStore, embedder: Embedder, cfg: Config) -> list:
    """Build the `retrieve`/`finish` tools bound to one question's store."""

    @tool
    def retrieve(query: str) -> str:
        """Semantic search for chunks relevant to an information need."""
        query_emb = embedder.embed_query(query)
        chunks = store.query(query_emb, k=cfg.top_k)
        seen: set[tuple[str, int | None]] = set()
        lines: list[str] = []
        for c in chunks:  # de-dupe by (title, sent_id) per design/06
            key = (c.title, c.sent_id)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"[{c.title}] {c.text}")
        return "\n".join(lines) if lines else "No chunks retrieved."

    @tool
    def finish(answer: str) -> str:
        """Commit the final answer and stop. Precondition: enough context to answer."""
        return answer

    return [retrieve, finish]
