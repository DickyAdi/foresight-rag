# Setup & Plan — Foresight RAG Framework

> Implementation plan for the 4-layer Agentic Planning Framework (STRIPS tools → query planner → state validator → reflector) over HotPotQA, using LangGraph + OpenRouter.

## Context

The project ([agentic-framework.md](agentic-framework.md)) proposes that current agentic RAG systems are **greedy and stateless** — they pick the next best tool locally with no lookahead and no verified state tracking. The framework adds four structural layers to mitigate this. The goal of this implementation is **not** to ship a product but to **measure whether the 4-layer architecture beats a naive greedy agent** on multi-hop retrieval (HotPotQA bridge/hard cases).

**Confirmed decisions:**
- **Test flow:** Python eval-script harness (batch, reproducible) + pytest for the deterministic validator. Core is an importable library so a FastAPI wrapper can be added later — deferred for now.
- **Baseline:** Build a greedy ReAct-style baseline agent and A/B it against the full framework on the same cases. This is what makes the results meaningful.
- **Models (OpenRouter):** Mixed tier — cheap model for the planner (speculative), strong model for executor + reflector. Exact IDs live in `config.py` and are trivially swappable.

---

## Architecture → LangGraph mapping

The 4 layers map onto a single `StateGraph` with a re-plan loop:

```
        ┌─────────────────────────── re-plan (if validation fails & budget left)
        ▼                                                              │
[planner] ──► [executor] ──► [validator] ──► (route) ──► [answer] ──► [reflector] ──► END
 Layer 2       (tools/      Layer 3         │
 cheap LLM      Layer 1)    pure code       └─► next step OR re-plan OR finalize
                strong LLM
```

- **Layer 1 — STRIPS tools:** tool descriptions written as `goal / precondition / postcondition / mechanism`, injected into the **system prompt** (not user prompt). The retrieval tool operates over a per-question embedded ChromaDB collection.
- **Layer 2 — Planner (cheap model):** shallow lookahead (max 2–3 steps). Precondition-check first (free pruning), then simulate → predicted post-state → viability score → beam (top-K) → commit highest-viability path.
- **Layer 3 — Validator (pure code, no LLM):** structural check first (cheap), semantic embedding check only if structural passes. Computes delta between predicted and actual post-state; significant delta → flag → re-plan or escalate.
- **Layer 4 — Reflector (strong model):** compares final answer vs original question for goal alignment (NOT fact-checking). Emits alignment score + reasoning.

**Baseline graph:** `agent ⇄ tools` greedy ReAct loop, no planner/validator/reflector. Same tools, same vector store, same executor model. Shared retrieval + answer code so the only variable is the architecture.

---

## Tools (Layer 1 — STRIPS descriptions)

For HotPotQA bridge questions (hop 1 finds entity X → hop 2 uses X), the tool set must support chaining:

- `retrieve(query: str)` — search the per-question vector store; appends relevant chunks (with `title`) to working memory.
  - *precondition:* a non-empty search query exists.
  - *postcondition:* working memory contains chunks relevant to `query`, each tagged with source title.
- `finish(answer: str)` — commit the final answer.
  - *precondition:* working memory contains enough context to answer.
  - *postcondition:* final answer is set; run terminates.

STRIPS text for these is injected into the system prompt via the Jinja2 `strips_tools_system` template.

---

## Validator detail (Layer 3) — the deterministic core

This is the only layer with real unit tests. Maps directly to dataset fields:

- `predicted_postcondition.required_sources` ← derived from `supporting_facts` titles.
- **Structural** (`validate_state_structural`): EMPTY_RETRIEVAL, MISSING_SOURCE (required title not in retrieved titles), INSUFFICIENT_CHUNKS (< `min_chunks`).
- **Semantic** (`validate_state_semantic`): cosine similarity (query vs retrieved chunks) via local `all-MiniLM-L6-v2`; LOW_RELEVANCE (max < threshold), POOR_RETRIEVAL_QUALITY (avg < threshold·0.75). No LLM.
- `validate_state` runs structural first, returns early on hard fail, else runs semantic. (Matches the doc's recommended composition.)

---

## Project structure

All application source lives under `src/`. Tests stay at repo root (standard pytest convention).

```
src/foresight/
  config.py            # OpenRouter base_url, model IDs per role, thresholds, K, max_replans
  llm.py               # ChatOpenAI factory: get_planner_llm() / get_executor_llm() / get_reflector_llm()
  state.py             # LangGraph TypedDict: question, gt_answer, supporting_facts, context_pool,
                       #   plan, predicted_post_state, working_memory(chunks), actual_post_state,
                       #   validation, replan_count, final_answer, reflection, trace
  retrieval.py         # build_store(context) -> per-question Chroma collection; HuggingFaceEmbeddings(all-MiniLM-L6-v2)
  tools.py             # retrieve / finish tools bound to a per-question store
  planner.py           # Layer 2: precondition prune -> simulate -> score -> beam -> commit
  validator.py         # Layer 3: structural + semantic (pure code)
  reflector.py         # Layer 4: alignment score + reasoning
  graph.py             # build_framework_graph() + build_baseline_graph(); conditional re-plan edges
  prompts/             # Layer 1 + LLM prompts — Jinja2 templates, one file per role (see below)
    __init__.py        #   render(name, **vars) loader: jinja2 Environment(FileSystemLoader) over this dir
    strips_tools_system.jinja2  # STRIPS goal/precondition/postcondition/mechanism tool block (system prompt)
    planner.jinja2              # Layer 2 lookahead/beam prompt
    executor.jinja2            # executor (tool-calling) prompt
    answer.jinja2             # final answer synthesis from working memory
    reflector.jinja2        # Layer 4 alignment check prompt
    baseline_system.jinja2 # greedy ReAct baseline system prompt (no planner/validator)
  data/
    load_hotpot.py   # load_dataset("hotpotqa/hotpot_qa","distractor"); filter type==bridge & level==hard; first 20
                     # + adapter for HF column-oriented supporting_facts/context (see gotcha below)
  eval/
    metrics.py       # answer EM/F1 (HotPotQA normalization), retrieval recall/precision vs supporting_facts
    run_eval.py      # runs framework + baseline on N cases, aggregates, prints table, writes results JSON
tests/
  test_validator.py    # unit tests for structural + semantic validator (deterministic)
  test_load_hotpot.py  # adapter/shape sanity
  test_prompts.py      # Jinja2 templates render with expected vars
.env.example           # OPENROUTER_API_KEY=
README.md              # how to run
```

---

## Dependencies (added via `uv add`)

Runtime: `langgraph`, `langchain`, `langchain-core`, `langchain-openai`, `langchain-huggingface`, `langchain-chroma`, `chromadb`, `sentence-transformers`, `scikit-learn`, `numpy`, `datasets`, `jinja2`, `python-dotenv`.
Dev (`uv add --dev`): `pytest`.

Notes:
- OpenRouter is OpenAI-compatible → `ChatOpenAI(model=..., base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)`.
- **Vector store: ChromaDB embedded** (`langchain-chroma` + `chromadb`), in-memory `EphemeralClient` (no persistence dir). A fresh, uniquely-named collection per question (see gotcha #3).
- `sentence-transformers` pulls `torch` (heavy, first run downloads the MiniLM model ~80MB). This is local, used by both the Chroma embeddings and the validator — no API cost.
- **Prompts: Jinja2.** All prompt text lives in `src/foresight/prompts/*.jinja2`, one file per role. A small `render(name, **vars)` loader in `prompts/__init__.py` uses `jinja2.Environment(loader=FileSystemLoader(prompt_dir))`. No prompt strings are hardcoded in Python — code only passes variables.
- Default model IDs in `config.py` (verify current OpenRouter IDs at implementation time, swap freely):
  - planner: `google/gemini-2.0-flash-001` (cheap/fast)
  - executor + reflector: `anthropic/claude-sonnet-4` (strong)

---

## Known gotchas to handle

1. **HF schema is column-oriented, not row-of-dicts.** The doc shows `supporting_facts` as a list of `{title, sent_id}` and `context` as a list of `{title, sentences}`, but the HuggingFace `hotpot_qa` dataset stores them as parallel arrays: `supporting_facts = {"title": [...], "sent_id": [...]}` and `context = {"title": [...], "sentences": [[...], ...]}`. `load_hotpot.py` must include an adapter that zips these into the conceptual shape the rest of the code expects.
2. **`trust_remote_code`** may be required by current `datasets` versions for `hotpot_qa`; handle with a clear error message if so.
3. **Vector store is per-question.** Build a fresh embedded Chroma collection (unique collection name per case, `EphemeralClient`) from that row's `context` only (distractors included on purpose) — never a global/shared collection, or the test leaks across cases. Drop the collection after each case.
4. **API key** read from `.env` via `python-dotenv`; never hardcode. `.env` is already covered by `.gitignore` patterns (verify; add if missing).

---

## Metrics captured per run (framework vs baseline)

- **Answer quality:** Exact Match + token F1 (standard HotPotQA normalization).
- **Retrieval quality:** recall & precision of retrieved titles vs `supporting_facts` titles (does the agent hit the needed sources, avoid distractors?).
- **Process signals:** avg re-plan count, validator flag rate, reflector alignment score, tool-call count.
- Output: console summary table + `eval/results/<timestamp>.json`.

---

## Build order (incremental, testable at each step)

1. **Write this plan to `.claude/setup-and-plan.md`** (the requested deliverable).
2. `uv add` all dependencies; create `.env.example`; confirm `.env` is gitignored.
3. `config.py`, `llm.py` — verify a single OpenRouter chat round-trip works.
4. `data/load_hotpot.py` + `test_load_hotpot.py` — load, filter, adapt; assert shape on a sample.
5. `retrieval.py` + `tools.py` — per-question embedded Chroma collection, retrieve/finish tools.
6. `validator.py` + `test_validator.py` — **pure code, fully unit-tested before any graph wiring.**
7. `prompts/` Jinja2 templates + `render()` loader + `test_prompts.py`; then `planner.py`, `reflector.py`.
8. `graph.py` — framework graph (with re-plan loop) + baseline graph.
9. `eval/metrics.py`, `eval/run_eval.py` — wire it together.

---

## Verification (end-to-end)

- `uv run pytest` → validator + loader unit tests pass (no API/network needed for the validator tests).
- `uv run python -m foresight.eval.run_eval --n 3 --mode framework` → smoke-test the full graph on 3 cases (real OpenRouter calls); inspect trace, predicted vs actual post-states, re-plan triggers.
- `uv run python -m foresight.eval.run_eval --n 20 --mode both` → full A/B on 20 bridge/hard cases; produces the comparison table. **Success = framework shows higher retrieval recall and/or answer F1 than baseline, with validator flags correlating to baseline failures.**
- Manually eyeball one full trace to confirm: STRIPS prompt present, planner produces ranked paths with predicted post-states, validator computes a real delta, reflector emits an alignment score.

---

## Out of scope (deferred)

- FastAPI `/query` endpoint and any UI (library is structured to allow it later).
- Persisting/caching LLM responses across runs.
- Scaling past ~20 cases or hyperparameter sweeps on K / thresholds / lookahead depth.
