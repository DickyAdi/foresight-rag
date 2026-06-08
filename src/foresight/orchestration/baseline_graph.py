"""Baseline greedy ReAct graph (design/01) — the control arm.

agent <-> tools loop with `retrieve` + `finish` bound; no planner/validator/reflector.
Same tools, store, and executor model as the framework — only the architecture differs.
Built on langgraph's prebuilt ReAct agent; non-termination is capped by the caller via
the graph's recursion_limit (derive from cfg.baseline_max_steps).
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from langchain.agents import create_agent


from foresight.core.prompts import render
from foresight.orchestration.tools import build_tools

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

    from foresight.config import Config
    from foresight.ports import Embedder, VectorStore


def build_baseline_graph(
    chat_raw: ChatOpenAI, store: VectorStore, embedder: Embedder, cfg: Config
):
    """Compile the greedy ReAct baseline. `chat_raw` is a tool-calling-capable model
    (adapters.factory.make_raw_chat); `store` is the populated per-case store."""
    tools = build_tools(store, embedder, cfg)
    return create_agent(chat_raw, tools, prompt=render("baseline_system"))
