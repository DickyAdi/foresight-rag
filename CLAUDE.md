# CLAUDE.md

Guidance for humans and coding agents working in this repository.

## What this project is

**Foresight RAG Framework** — a research project that tests whether a 4-layer *agentic planning* architecture (STRIPS-style tool descriptions → lookahead query planner → deterministic state validator → LLM reflector) beats a naive greedy RAG agent on multi-hop question answering (HotPotQA, bridge/hard). The goal is a **measurement**, not a product — though the core is structured so it can later be extracted as a minimal-dependency library.

See [README.md](README.md) for the overview, and [`.claude/setup-and-plan.md`](.claude/setup-and-plan.md) (plan/build order) and [`.claude/agentic-framework.md`](.claude/agentic-framework.md) (the underlying theory) for deeper context.

## Stack

Python 3.12 · `uv` · LangGraph + langchain-core/langchain-openai · raw `openai` (embeddings) · raw `chromadb` · OpenRouter (models) · `datasets` (HotPotQA) · jinja2 (prompts) · pydantic · numpy.

## Repository layout

```
src/foresight/
  core/          pure framework logic — imports ONLY ports + pydantic/numpy/jinja2
  ports/         ChatModel · Embedder · VectorStore  (typing.Protocol interfaces)
  adapters/      concrete impls (langchain-openai / openai / chromadb) — wired to ports
  orchestration/ langgraph graphs + state + tools (the ONLY place langgraph appears)
  data/          HotPotQA load/adapt/cache
  eval/          metrics + run_eval harness
  config.py      Config dataclass (all knobs), injected into layers
tests/
```

## Architecture rule (do not break)

**`core/` must never import LangChain, chromadb, or the openai SDK directly.** Those live behind `ports/` and are implemented only in `adapters/` and `orchestration/`. This dependency direction is what keeps the framework extractable as a lean library — preserve it.

## Working in the repo

- Run anything with `uv run …` (e.g. `uv run pytest`, `uv run python -m foresight.eval.run_eval`).
- Secrets via `.env` (copy from `.env.example`); set `OPENROUTER_API_KEY`. Never hardcode keys.
- The deterministic validator is unit-tested and must stay offline (no network) in its tests.

## Reserved workspace — for both humans and agents

`.claude/developer/` is **reserved per-contributor scratch space** and is **already gitignored**. Humans and coding agents may freely put personal plans, scratch notes, throwaway prompts, local agent configs, and experiment logs there — nothing in it is committed. Keep personal/agent working files **inside `.claude/developer/`**, not loose in the committed tree. See [`.claude/CLAUDE.md`](.claude/CLAUDE.md).
