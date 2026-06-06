"""Baseline greedy ReAct graph (design/01) — the control arm.

agent <-> tools loop, finish tool to stop. No planner/validator/reflector.
Same tools, store, and executor model as the framework — only the architecture differs.
BASELINE_MAX_STEPS guards non-termination.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresight.config import Config
    from foresight.ports import ChatModel, Embedder, VectorStore

# TODO: build_baseline_graph(chat, store, embedder, cfg) -> CompiledGraph
#   nodes: agent_node (greedy, bound tools), tool_node
#   loop until finish() or cfg.baseline_max_steps reached (then force-answer)
