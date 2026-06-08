"""Offline tests for the framework graph: router logic + topology compiles.

Full execution needs live LLM + embeddings (an API key), so here we unit-test the
pure routing functions and that the graph wires up / compiles with dummy deps.
"""
from foresight.config import Config
from foresight.core.types import PlanDecision, ValidationResult
from foresight.orchestration.graph import (
    build_framework_graph,
    route_after_plan,
    route_after_validate,
)


# --- route_after_plan ---

def test_route_after_plan_answer():
    assert route_after_plan({"decision": PlanDecision(action="answer")}) == "answer"


def test_route_after_plan_execute():
    assert route_after_plan({"decision": PlanDecision(action="retrieve", query="x")}) == "execute"


# --- route_after_validate ---

def test_route_advance_on_success():
    state = {"validation": ValidationResult(ok=True)}
    assert route_after_validate(state, max_replans=2) == "advance"


def test_route_replan_while_budget_remains():
    state = {"validation": ValidationResult(ok=False, errors=["LOW_RELEVANCE"]),
             "replan_count": 0}
    assert route_after_validate(state, max_replans=2) == "replan"


def test_route_finalize_when_budget_exhausted():
    state = {"validation": ValidationResult(ok=False, errors=["LOW_RELEVANCE"]),
             "replan_count": 2}
    assert route_after_validate(state, max_replans=2) == "finalize"


# --- topology ---

def test_framework_graph_compiles():
    # compile() validates node/edge wiring without invoking any node.
    graph = build_framework_graph(
        planner=object(), validator=object(), reflector=object(),
        store=object(), embedder=object(), executor_chat=object(), cfg=Config(),
    )
    assert graph is not None
