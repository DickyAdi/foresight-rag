# Foresight RAG Framework

A research framework that asks a simple question: **does adding structured *foresight* to an agentic RAG system actually help?**

Most agentic systems are **greedy and stateless** — they pick the next best tool call locally, with no lookahead and no verified state tracking. This project implements a 4-layer alternative and **measures it head-to-head against a naive greedy agent** on multi-hop question answering (HotPotQA, `bridge`/`hard` cases).

## The four layers

1. **STRIPS-style tool descriptions** — tools described by *preconditions/postconditions*, so valid call sequences can be inferred and invalid ones pruned early.
2. **Query planner (foresight)** — a lightweight lookahead pass that simulates candidate action sequences and commits the most viable next step, instead of acting greedily.
3. **State validator** — pure deterministic code (no LLM) that checks each retrieval's actual result against what was expected, and triggers a re-plan when it falls short.
4. **Reflector** — a final LLM pass that checks the answer stayed aligned with the original question (catching quiet goal drift).

## How it's evaluated

Three arms run on the same HotPotQA cases so the only variable is the architecture:

- `baseline` — a greedy ReAct agent (the control)
- `framework + cheap foresight`
- `framework + deep foresight`

Metrics: answer Exact Match / F1, retrieval recall & precision against the gold supporting facts, and process signals (re-plans, validator flags, reflector alignment).

## Stack

Python 3.12 · [`uv`](https://docs.astral.sh/uv/) · LangGraph · OpenRouter (chat + embeddings) · ChromaDB · HotPotQA via `datasets`.

## Quickstart

```bash
uv sync                        # install dependencies
cp .env.example .env           # then set OPENROUTER_API_KEY
uv run pytest                  # run the test suite
```

Running the full evaluation harness:

```bash
uv run python -m foresight.eval.run_eval --n 3     # smoke test
uv run python -m foresight.eval.run_eval --n 20    # full A/B
```

## Project layout

```
src/foresight/
  core/          framework logic (planner, validator, reflector, types, prompts)
  ports/         interfaces (ChatModel, Embedder, VectorStore)
  adapters/      concrete model/store implementations
  orchestration/ LangGraph wiring (framework + baseline graphs)
  data/          HotPotQA loading
  eval/          metrics + evaluation harness
```

The `core` package depends only on the `ports` interfaces (not on any specific LLM/vector-store SDK), so it can later be extracted as a minimal-dependency library.

## Status

Early — scaffolding and architecture are in place; layer implementations are in progress.

## Data & license

Uses the [HotPotQA](https://hotpotqa.github.io/) distractor dataset (CC BY-SA 4.0).
