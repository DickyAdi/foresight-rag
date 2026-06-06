"""Config dataclass — single source of truth for all knobs.

Defaults live here; layers receive a `Config` (or a focused sub-config) in their
constructor rather than importing globals, so they stay testable and override-able.
See design/09-models-and-config.md and design/12-implementation-architecture.md.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Config:
    # --- Models (OpenRouter) ---
    openrouter_base_url: str = OPENROUTER_BASE_URL
    planner_model: str = "google/gemini-2.0-flash-001"      # cheap/fast, speculative
    executor_model: str = "anthropic/claude-sonnet-4"        # strong (executor/answer)
    reflector_model: str = "anthropic/claude-sonnet-4"       # strong
    embedding_model: str = "openai/text-embedding-3-large"   # shared: store + validator
    temperature: float = 0.0
    max_retries: int = 3

    # --- Retrieval ---
    top_k: int = 5
    embed_title_prefix: bool = True            # embed "{title}: {sentence}" (design/06)

    # --- Planner / foresight (design/03) ---
    planning_strategy: Literal["imagined", "grounded"] = "imagined"
    beam_width: int = 3                         # K candidate queries
    max_lookahead_steps: int = 3
    max_hops: int = 3                           # non-termination guard
    allow_direct_answer: bool = False           # step-0 answer gate

    # --- Validator (design/04) ---
    max_replans: int = 2                        # failure-loop guard
    min_chunks: int = 2
    relevance_quantile_q: float = 0.7           # per-question relative threshold
    validator_mode: Literal["self", "oracle"] = "self"   # oracle = labelled upper-bound ablation

    # --- Baseline ---
    baseline_max_steps: int = 6                 # non-termination guard for greedy ReAct
