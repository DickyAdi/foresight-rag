"""LangChain tools for the BASELINE greedy ReAct loop (design/02).

`finish` is a real bound tool only in the baseline; the framework terminates via the
planner->answer node instead. Both tools are bound to a per-question VectorStore.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresight.ports import Embedder, VectorStore

# TODO: build_tools(store: VectorStore, embedder: Embedder) -> list[tool]
#   retrieve(query): embed query -> store.query(k) -> chunks (dedup by title,sent_id)
#   finish(answer):  signal stop with the answer
#   STRIPS prose stays in prompts/strips_tools_system.jinja2; docstrings minimal.
