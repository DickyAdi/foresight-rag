"""LangGraph StateGraph definitions.

build_framework_graph() — 4-layer pipeline with re-plan loop:
  planner → executor → validator → (route) → answer → reflector → END
                           ↑                              │
                           └──── re-plan if budget left ──┘

build_baseline_graph() — greedy ReAct loop, no planner/validator/reflector:
  agent ⇄ tools
  Same executor model and tools as the framework graph.
  The only variable between the two graphs is the architecture.
"""
# TODO: implement build_framework_graph(store) -> CompiledStateGraph
#   nodes: planner_node, executor_node, validator_node, answer_node, reflector_node
#   conditional edge after validator: route to answer_node or back to planner_node
#   replan_count guard: if >= MAX_REPLANS → force to answer_node

# TODO: implement build_baseline_graph(store) -> CompiledStateGraph
#   nodes: agent_node (greedy ReAct), tool_node
#   loop: agent_node → tool_node → agent_node until finish() called
