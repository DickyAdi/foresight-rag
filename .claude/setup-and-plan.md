# Setup & Plan — Foresight RAG Framework

> High-level plan, architecture, and build order. Self-contained: the decisions below are the spec to build against.

## Context

The underlying theory ([agentic-framework.md](agentic-framework.md)) argues that current agentic RAG is **greedy and stateless** — next-best-action locally, no lookahead, no verified state. This project implements the proposed 4-layer fix and **measures whether it beats a naive greedy agent** on multi-hop retrieval (HotPotQA `bridge`/`hard`). It is a measurement, not a product.

## Confirmed high-level decisions

- **Test flow:** batch Python eval-script harness + pytest for the deterministic validator. Importable library so a FastAPI wrapper can come later (deferred).
- **Three eval arms:** greedy `baseline` vs `framework + cheap (imagined) foresight` vs `framework + deep (grounded) foresight`.
- **Models (OpenRouter, mixed tier):** cheap planner (`google/gemini-2.0-flash-001`), strong executor/reflector (`anthropic/claude-sonnet-4`), embeddings `openai/text-embedding-3-large`.
- **Fairness:** gold (`answer`, `supporting_facts`) used **only** in metrics; in-loop validator is **gold-free**; both arms share one retrieval substrate.

## Architecture (one LangGraph, iterative MPC)

```
        ┌──────────── router: re-plan (validation failed, budget left) ────────────┐
        ▼                                                                           │
[planner] ──► [executor] ──► [validator] ──► (router) ──► [answer] ──► [reflector] ──► END
 Layer 2       Layer 1        Layer 3          │
 (foresight)   (tools)        (pure code)      ├─► advance ──► [planner]
                                               └─► finalize ──► [answer]
```

Iterative MPC: the planner looks a few steps ahead but commits **one action at a time**, so the hop-2 query is formed after hop-1 results exist. Router exits after the validator: **advance** (plan next hop), **re-plan** (this hop failed, budget remains), **finalize**. Guards: `MAX_HOPS` (non-termination), `MAX_REPLANS` (failure loop) → force finalize.

Baseline graph = greedy `agent ⇄ tools` ReAct loop, no planner/validator/reflector — same tools, store, and executor model, so the only variable is the architecture.

### The four layers

- **Layer 1 — STRIPS tools** (`retrieve`, `finish`): described by precondition/postcondition in the system prompt. `finish` is a real tool only in the baseline; the framework terminates via the planner→answer node. Preconditions are code-enforced during planning.
- **Layer 2 — Planner / foresight:** two pluggable strategies. *Imagined* (cheap) — one cheap-model call returns K candidate queries with self-predicted post-states, prune + commit top-1. *Grounded* (deep) — actually retrieve each candidate and rank by the validator's relevance score, then commit the best. `allow_direct_answer` (default off) forces retrieval at step 0.
- **Layer 3 — State validator (pure code, no LLM):** gold-free. Structural checks (`EMPTY_RETRIEVAL`, `INSUFFICIENT_CHUNKS`) then a **per-question relative quantile** relevance check (flag if the best retrieved chunk doesn't beat quantile `Q≈0.7` of the query↔pool distribution). Exposes a continuous relevance score reused by the grounded planner. On a flag, feeds the delta back to the planner for re-planning.
- **Layer 4 — Reflector (strong model):** gold-blind alignment check (did the answer stay on goal?), observational in v1.

## Implementation architecture (ports & adapters)

Class-based with constructor injection. **`core/` must never import LangChain, chromadb, or the openai SDK directly** — those sit behind `ports/` (typing.Protocol) and are implemented only in `adapters/` and `orchestration/`. This dependency direction is what keeps the framework extractable as a lean library later.

- **Footprint (lean):** LangGraph + langchain-core/langchain-openai for chat/orchestration; **raw `openai`** for embeddings; **raw `chromadb`** for the vector store (so we can add precomputed embeddings and read them back for vector reuse); numpy for cosine.
- **Config:** a `Config` dataclass holds all knobs (models, `top_k`, `beam_width`, `max_hops`, `max_replans`, `min_chunks`, `relevance_quantile_q`, `planning_strategy`, `allow_direct_answer`, `embed_title_prefix`, `temperature`), injected into layers.
- **Prompts:** jinja2 templates, one per role, loaded via `PackageLoader`; `{% include %}` for the shared STRIPS block; JSON schema owned by Pydantic (structured output), not the prompt.

## Retrieval & data

- **Per-question store:** a fresh ephemeral ChromaDB collection per case from that row's `context` (distractors included), dropped after the case. One document **per sentence**, metadata `{title, sent_id, article_idx}`. Embedded text = `"{title}: {sentence}"` (toggle `embed_title_prefix`, default on) to fix HotPotQA coreference; retrieval is semantic over content, title is metadata only.
- **Embeddings reuse:** embed the pool once; reuse those vectors for the validator's relevance distribution; cache query→embedding.
- **Dataset:** `load_dataset("hotpotqa/hotpot_qa", "distractor")`, validation split → filter `bridge`+`hard` → first 20 → adapt the HF **column-oriented** `supporting_facts`/`context` to row-of-dicts → cache to a committed `cases.json` (HF load path should be verified first; datasets 5.x is parquet-based).

## Project structure

```
src/foresight/
  core/          types.py · relevance.py · validator.py · strategies.py · planner.py · reflector.py
                 prompts/ (Jinja2: strips_tools_system, planner, executor, answer, reflector, baseline_system + render())
                 # pure logic — imports ONLY ports + pydantic/numpy/jinja2
  ports/         ChatModel · Embedder · VectorStore (typing.Protocol)
  adapters/      openrouter_chat.py · openrouter_embeddings.py · chroma_store.py · factory.py
  orchestration/ state.py · tools.py · graph.py · baseline_graph.py   # only place langgraph appears
  data/          load_hotpot.py · cases.json (cached)
  eval/          metrics.py · run_eval.py · results/ (gitignored)
  config.py      # Config dataclass, injected into layers
tests/  test_validator.py · test_load_hotpot.py · test_prompts.py
.env.example   # OPENROUTER_API_KEY=
```

**Deps (lean):** `langgraph, langchain-core, langchain-openai, openai, chromadb, numpy, jinja2, datasets, python-dotenv, pydantic` (+ dev `pytest`).

## Build order (incremental, testable)

1. ✅ Scaffold + ports/adapters layout in place; deps lean; prompts done.
2. ✅ `config.py` (done) + `adapters/factory.py` + `openrouter_chat.py` + `openrouter_embeddings.py` (+ `_secrets.py` helper) written. ⏳ Live-verify only (key needed solely to *run* the check, not to develop): one OpenRouter chat round-trip **and** the `/embeddings` endpoint.
3. ✅ `data/load_hotpot.py` + test written (cache-gated, offline; tests skip until `cases.json` exists). ⏳ HF load path unverified — needs one online run to build/commit `cases.json`.
4. ✅ `adapters/chroma_store.py` (offline add/query/get_embeddings/drop round-trip verified, cosine space) + `orchestration/tools.py` (`retrieve` dedup by (title,sent_id); `finish` baseline-only) — per-question Chroma (per-sentence, `title:sentence` prefix), `retrieve`/`finish`, vector reuse.
5. ✅ `core/relevance.py` (empirical per-question quantile, numpy cosine) + `core/validator.py` (structural→relevance gate, gold-free, no `MISSING_SOURCE`) + `test_validator.py` — **fully unit-tested offline (9 tests), no graph wiring.**
6. ✅ `core/strategies.py` (Imagined + Grounded) + `core/planner.py` (max_hops / allow_direct_answer guards) + `core/reflector.py` — offline-tested with fake ChatModel/ports (8 tests). Added `CandidateQuery`/`PlannerCandidates` to `types.py`; rewrote `planner.jinja2` to MPC K-candidate (Pydantic owns schema). Grounded ranks by real top retrieved cosine (see note below).
7. ✅ `orchestration/graph.py` (MPC loop; advance/replan/finalize routers; max_hops via Planner, max_replans via router; elective finalize) + `baseline_graph.py` (prebuilt ReAct) + `case_store.py` (per-question pool population) + `factory.make_raw_chat`. Relevance scored vs the QUESTION (design/04 "one function, two consumers" — Option A; confirmed required, not optional). Offline-tested: routers + graph compiles (6 tests). Live run pending key. NB: pass `recursion_limit` ≈ max_hops·(max_replans+1)·nodes at invoke (step 8).
8. ✅ `eval/metrics.py` (EM/F1 SQuAD-normalized + retrieval recall/precision; offline-tested, 6 tests) + `eval/run_eval.py` (`--n/--mode/--strategy`; per-case store build→arms→metrics→drop; aggregates + console table + `results/<ts>.json`; per-arm try/except). Code complete; **live run pending key** (smoke `--n 3` first).

## Verification

- `uv run pytest` → validator + loader + prompt tests pass (validator tests offline).
- `uv run python -m foresight.eval.run_eval --n 3 --mode framework` → smoke-test full graph; eyeball trace (STRIPS prompt present, planner candidates + committed query, validator delta, reflector score).
- `uv run python -m foresight.eval.run_eval --n 20` → full 3-arm A/B → comparison table + results JSON.
- **Success:** framework > baseline on retrieval recall and/or answer F1; validator flags correlate with baseline failures; deep ≥ cheap foresight.

## Out of scope (deferred)

FastAPI/UI; LLM response caching; scaling past ~20 cases or hyperparameter sweeps; the `extract_entity` tool; turning on `allow_direct_answer` (needs a mixed-difficulty set).
