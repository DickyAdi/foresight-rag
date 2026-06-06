"""Domain types — pure pydantic/dataclass, no langchain. Shared across core layers.

These are the types that flow between layers and would ship with the extracted
library. The langgraph-specific AgentState lives in orchestration/state.py instead.
"""
from __future__ import annotations
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrieved unit. `text` is the displayed sentence; title/sent_id from metadata."""
    title: str
    text: str
    sent_id: int | None = None
    article_idx: int | None = None
    score: float | None = None


class PredictedPostState(BaseModel):
    """Planner's prediction for the committed action (light in gold-free mode)."""
    min_chunks: int = 2
    expected_topic: str | None = None   # trace/delta log only


class PlanDecision(BaseModel):
    """Planner output (structured). Either a retrieve query or a terminal answer."""
    action: str = Field(description="'retrieve' or 'answer'")
    query: str | None = None
    predicted_post_state: PredictedPostState = Field(default_factory=PredictedPostState)
    viability: float = 0.0
    reasoning: str = ""


class ValidationResult(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    relevance_score: float | None = None   # continuous; also used to rank grounded rollouts


class Reflection(BaseModel):
    alignment_score: float
    reasoning: str
