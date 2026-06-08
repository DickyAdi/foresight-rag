"""LangGraph shared state. The langgraph-specific TypedDict lives here (orchestration),
separate from the portable domain types in core/types.py.
"""
from __future__ import annotations
from typing import Any, TypedDict
from foresight.core.types import Chunk, PlanDecision, ValidationResult, Reflection


class AgentState(TypedDict, total=False):
    # Input
    question: str
    gt_answer: str                  # gold — eval only, never read in-loop
    supporting_facts: list[dict]    # gold — eval only
    context_pool: list[dict]

    # Per-hop planner/executor
    decision: PlanDecision
    working_memory: list[Chunk]     # accumulated, deduped by (title, sent_id)
    last_retrieved: list[Chunk]

    # Validator
    validation: ValidationResult
    validation_feedback: str | None   # delta fed back to the planner on re-plan

    # Guards
    hop: int
    replan_count: int

    # Output
    final_answer: str
    reflection: Reflection

    # Debug
    trace: list[dict[str, Any]]
