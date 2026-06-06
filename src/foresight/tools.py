"""LangChain tool definitions for the executor (Layer 1 — STRIPS tools).

Tools are bound to a per-question Chroma store. The store is passed in at
graph construction time so each question gets an isolated retrieval scope.
"""
# TODO: implement retrieve(query: str) -> str
#   - STRIPS: precondition=non-empty query, postcondition=chunks in working_memory
#   - Calls store.similarity_search(query, k=TOP_K)
#   - Returns formatted string of title+snippet pairs; graph node appends to state.working_memory

# TODO: implement finish(answer: str) -> str
#   - STRIPS: precondition=working_memory has enough context, postcondition=final_answer set
#   - Simply returns the answer; graph node writes to state.final_answer and halts tool loop
