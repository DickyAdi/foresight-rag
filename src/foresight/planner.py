"""Layer 2 — Query Planner (cheap model, shallow lookahead).

Process:
  1. Precondition-check each candidate action (free pruning, do first)
  2. Simulate short sequences (max MAX_LOOKAHEAD_STEPS) → predicted post-state
  3. Score viability; prune below threshold
  4. Beam search: keep top BEAM_WIDTH paths
  5. Commit highest-viability path as state.plan + state.predicted_post_state
"""
# TODO: implement run_planner(state: AgentState, llm) -> dict (partial AgentState update)
#   - Render prompts.render("planner", question=..., working_memory=..., available_tools=...)
#   - Parse LLM output into ranked action paths with predicted post-states
#   - Apply beam pruning
#   - Return {"plan": [...], "predicted_post_state": {...}, "trace": [...]}
