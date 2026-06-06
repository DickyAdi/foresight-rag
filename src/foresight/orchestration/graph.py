"""Framework StateGraph (design/01) — the ONLY place langgraph is wired.

planner -> executor -> validator -> (router) -> answer -> reflector -> END
Router exits: advance (next hop) / re-plan (failure, budget left) / finalize.
Guards: MAX_HOPS (non-termination), MAX_REPLANS (failure loop) -> force finalize.
Nodes are thin: each calls a core layer (Planner/Validator/Reflector) and maps
to/from AgentState. Core layers stay orchestration-agnostic.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresight.config import Config
    from foresight.core.planner import Planner
    from foresight.core.reflector import Reflector
    from foresight.core.validator import Validator
    from foresight.ports import Embedder, VectorStore

# TODO: build_framework_graph(planner, validator, reflector, store, embedder, cfg) -> CompiledGraph
#   nodes: planner_node, executor_node (retrieve), validator_node, answer_node, reflector_node
#   conditional edge after validator_node: route(state) -> "advance" | "replan" | "finalize"
