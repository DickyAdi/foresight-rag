"""Framework StateGraph (design/01) — the ONLY place langgraph is wired.

planner -> executor -> validator -> (router) -> answer -> reflector -> END
Router exits: advance (next hop) / re-plan (failure, budget left) / finalize.
Guards: MAX_HOPS (in Planner -> terminal answer), MAX_REPLANS (router -> finalize).
Nodes are thin: each calls a core layer (Planner/Validator/Reflector) and maps
to/from AgentState. Core layers stay orchestration-agnostic.

Relevance is scored against the QUESTION (design/04 "current need"): the validator
re-embeds the just-retrieved chunks with the same title-prefix as the pool and asks
RelevanceScorer for their quantile position in the question's pool distribution.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from foresight.core.prompts import render
from foresight.orchestration.state import AgentState

if TYPE_CHECKING:
    from foresight.config import Config
    from foresight.core.planner import Planner
    from foresight.core.reflector import Reflector
    from foresight.core.validator import Validator
    from foresight.ports import ChatModel, Embedder, VectorStore


def route_after_plan(state: AgentState) -> str:
    """Terminal answer (elective finalize or max_hops) skips retrieve/validate."""
    return "answer" if state["decision"].action == "answer" else "execute"


def route_after_validate(state: AgentState, max_replans: int) -> str:
    """advance on success; re-plan while budget remains; else finalize."""
    validation = state["validation"]
    if validation.ok:
        return "advance"
    if state.get("replan_count", 0) < max_replans:
        return "replan"
    return "finalize"


def build_framework_graph(planner: Planner, validator: Validator, reflector: Reflector,
                          store: VectorStore, embedder: Embedder,
                          executor_chat: ChatModel, cfg: Config):
    """Compile the per-question framework graph. `store` is the populated per-case store."""

    def _prefixed(chunk) -> str:
        return f"{chunk.title}: {chunk.text}" if cfg.embed_title_prefix else chunk.text

    def planner_node(state: AgentState) -> dict:
        decision = planner.plan(
            question=state["question"],
            working_memory=state.get("working_memory", []),
            feedback=state.get("validation_feedback"),
            hop=state.get("hop", 0),
        )
        return {"decision": decision}

    def executor_node(state: AgentState) -> dict:
        query = state["decision"].query or state["question"]
        query_emb = embedder.embed_query(query)
        retrieved = store.query(query_emb, k=cfg.top_k)
        working_memory = list(state.get("working_memory", []))
        seen = {(c.title, c.sent_id) for c in working_memory}
        for chunk in retrieved:  # de-dupe accumulated memory by (title, sent_id)
            key = (chunk.title, chunk.sent_id)
            if key not in seen:
                seen.add(key)
                working_memory.append(chunk)
        return {"working_memory": working_memory, "last_retrieved": retrieved}

    def validator_node(state: AgentState) -> dict:
        retrieved = state.get("last_retrieved", [])
        question_emb = embedder.embed_query(state["question"])
        pool_embs = store.get_embeddings()
        retrieved_embs = (
            embedder.embed_documents([_prefixed(c) for c in retrieved]) if retrieved else []
        )
        result = validator.validate(
            query_emb=question_emb, retrieved_embs=retrieved_embs,
            pool_embs=pool_embs, accumulated=state.get("working_memory", []),
        )
        return {"validation": result}

    def answer_node(state: AgentState) -> dict:
        prompt = render("answer", question=state["question"],
                        working_memory=state.get("working_memory", []))
        message = executor_chat.complete([{"role": "user", "content": prompt}])
        content = getattr(message, "content", message)
        return {"final_answer": content if isinstance(content, str) else str(content)}

    def reflector_node(state: AgentState) -> dict:
        reflection = reflector.reflect(
            question=state["question"],
            final_answer=state.get("final_answer", ""),
            working_memory=state.get("working_memory", []),
        )
        return {"reflection": reflection}

    def advance_node(state: AgentState) -> dict:
        return {"hop": state.get("hop", 0) + 1, "replan_count": 0, "validation_feedback": None}

    def replan_node(state: AgentState) -> dict:
        errors = state["validation"].errors
        return {
            "replan_count": state.get("replan_count", 0) + 1,
            "validation_feedback": "; ".join(errors) if errors else "retrieval flagged",
        }

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("validator", validator_node)
    graph.add_node("answer", answer_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("advance", advance_node)
    graph.add_node("replan", replan_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", route_after_plan,
                                {"execute": "executor", "answer": "answer"})
    graph.add_edge("executor", "validator")
    graph.add_conditional_edges(
        "validator", lambda s: route_after_validate(s, cfg.max_replans),
        {"advance": "advance", "replan": "replan", "finalize": "answer"},
    )
    graph.add_edge("advance", "planner")
    graph.add_edge("replan", "planner")
    graph.add_edge("answer", "reflector")
    graph.add_edge("reflector", END)
    return graph.compile()
