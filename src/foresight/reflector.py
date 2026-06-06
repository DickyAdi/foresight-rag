"""Layer 4 — LLM Reflector (strong model).

Checks semantic alignment between the original question and the final answer.
This is NOT fact-checking — it only asks: did we stay on goal?
"""
# TODO: implement run_reflector(state: AgentState, llm) -> dict (partial AgentState update)
#   - Render prompts.render("reflector", question=..., final_answer=..., working_memory=...)
#   - Parse LLM output: {"alignment_score": 0.0-1.0, "reasoning": "..."}
#   - Return {"reflection": {...}, "trace": [...]}
