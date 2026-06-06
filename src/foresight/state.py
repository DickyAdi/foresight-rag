"""LangGraph shared state definition for the framework pipeline."""
from __future__ import annotations
from typing import Any, TypedDict


class Chunk(TypedDict):
    title: str
    text: str


class PredictedPostState(TypedDict):
    required_sources: list[str]   # titles the planner expects to be retrieved
    min_chunks: int


class ValidationResult(TypedDict):
    ok: bool
    errors: list[str]


class TraceEntry(TypedDict):
    node: str
    detail: Any


class AgentState(TypedDict):
    # Input
    question: str
    gt_answer: str
    supporting_facts: list[dict]  # [{"title": ..., "sent_id": ...}]
    context_pool: list[dict]      # [{"title": ..., "sentences": [...]}]

    # Planner output (Layer 2)
    plan: list[str]               # ordered list of tool calls the planner committed to
    predicted_post_state: PredictedPostState

    # Executor output (Layer 1 tools)
    working_memory: list[Chunk]   # chunks accumulated across retrieve() calls
    actual_post_state: dict       # {chunks: [...], ...}

    # Validator output (Layer 3)
    validation: ValidationResult

    # Re-plan tracking
    replan_count: int

    # Answer + reflection (Layer 4)
    final_answer: str
    reflection: dict              # {alignment_score: float, reasoning: str}

    # Full execution trace for debugging
    trace: list[TraceEntry]
