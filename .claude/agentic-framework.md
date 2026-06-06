# Agentic Planning Framework

> A theory-to-implementation guide for structured LLM tool sequencing

> **Note:** This is the canonical *theory*. The actual build follows the plan in [`setup-and-plan.md`](setup-and-plan.md), which **refines or diverges** from this theory in a few places — notably a **gold-free** in-loop validator (gold is used only for evaluation metrics, never inside the loop), a **per-question relative quantile** relevance threshold instead of the absolute `0.5` shown below, and a **grounded-rollout** foresight strategy that scores candidate queries against real retrievals. Where they differ, the plan governs the implementation.

---

## Core Idea

Current agentic systems are **greedy and stateless** — they pick the next best action locally with no lookahead and no verified state tracking. This framework adds structure around tool descriptions, planning, and validation to mitigate that.

---

## The 4-Layer Architecture

### Layer 1 — STRIPS-Style Tool Descriptions

Instead of describing tools by _when_ to use them, describe them by **preconditions and postconditions**.

```
tool_name:
  goal: what problem this tool solves
  precondition: what must be true before this tool runs usefully
  postcondition: what will be true in the context/state after this tool runs
  mechanism: briefly how it achieves the goal
```

**Why:** The model can infer valid sequences automatically. If tool B's precondition requires something tool A's postcondition produces, the model reasons the order without being explicitly told. Preconditions also act as free pruning — invalid paths are eliminated before simulation.

---

### Layer 2 — Query Planner (Shallow Lookahead)

Before executing anything, run a lightweight planning pass that simulates possible action sequences and their predicted post-states.

```
Input: current state + goal + available tools
Output: ranked action paths with predicted post-states and viability scores

Process:
  for each possible action (or short sequence):
    simulate → predicted post-state
    score viability
    prune if precondition not met OR viability below threshold
  commit to highest viability path
```

**Pruning strategies to implement:**

- Beam search — keep only top K paths, drop the rest early
- Confidence gating — drop paths below a viability threshold
- Precondition checking — eliminate structurally invalid paths immediately (cheapest, do this first)

**Why:** Even 1-2 steps of lookahead prevents locally good actions that make the overall goal harder to reach.

---

### Layer 3 — External State Validator

After each tool call, an external validator (not an LLM, not a human) checks whether the actual post-state matches the predicted post-state from the planner. This is pure deterministic code — just structured comparison functions.

```
predicted_post_state (from planner)
        ↓
  tool executes
        ↓
actual_post_state
        ↓
validator: delta = actual vs predicted
        ↓
if delta is significant → flag, re-plan or escalate
```

**Why:** The LLM cannot verify its own outputs. This layer needs to be deterministic code, not another model call. The delta between predicted and actual becomes a calibration signal for the planner within the same session.

#### Option A — Structural Validation (deterministic, no ML)

Checks fields, lengths, presence of expected keys. Fast and cheap. Use this first.

```python
def validate_state_structural(predicted_postcondition, actual_output):
    errors = []

    # did retrieval return anything?
    if not actual_output["chunks"] or len(actual_output["chunks"]) == 0:
        errors.append("EMPTY_RETRIEVAL")

    # did it hit the expected sources?
    retrieved_titles = [c["title"] for c in actual_output["chunks"]]
    for expected in predicted_postcondition["required_sources"]:
        if expected not in retrieved_titles:
            errors.append(f"MISSING_SOURCE: {expected}")

    # did it return enough context?
    if len(actual_output["chunks"]) < predicted_postcondition["min_chunks"]:
        errors.append("INSUFFICIENT_CHUNKS")

    return len(errors) == 0, errors
```

With HotPotQA, `supporting_facts` maps directly to `predicted_postcondition["required_sources"]` — you already know which sources should have been retrieved.

#### Option B — Embedding Similarity Validation (semantic, no LLM)

For when postconditions are semantic rather than structural — e.g. "retrieved chunks are relevant to the query." Uses cosine similarity between query and retrieved chunks. Still no LLM involved.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, good enough

def validate_state_semantic(query, actual_output, threshold=0.5):
    errors = []

    if not actual_output["chunks"]:
        return False, ["EMPTY_RETRIEVAL"]

    query_embedding = model.encode([query])
    chunk_texts = [c["text"] for c in actual_output["chunks"]]
    chunk_embeddings = model.encode(chunk_texts)

    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    max_similarity = np.max(similarities)
    avg_similarity = np.mean(similarities)

    if max_similarity < threshold:
        errors.append(f"LOW_RELEVANCE: best chunk similarity {max_similarity:.2f}")

    if avg_similarity < (threshold * 0.75):
        errors.append(f"POOR_RETRIEVAL_QUALITY: avg similarity {avg_similarity:.2f}")

    return len(errors) == 0, errors
```

#### Recommended approach

Run structural first — it's free. Only run semantic if structural passes. Structural catches hard failures (wrong source, empty retrieval), semantic catches soft failures (retrieved something but it's irrelevant).

```python
def validate_state(predicted_postcondition, actual_output, query):
    ok, errors = validate_state_structural(predicted_postcondition, actual_output)
    if not ok:
        return False, errors  # hard fail, no point checking semantics

    return validate_state_semantic(query, actual_output)
```

---

### Layer 4 — LLM Reflector

A separate LLM call that compares the **final output** against the **original user intent**.

```
Input: original goal + final output
Task: did we drift? is the output actually answering what was asked?
Output: alignment score + reasoning
```

**Important distinction:**

- Reflector = semantic alignment checker (did we stay on goal?)
- NOT a fact checker or correctness verifier (it's still an LLM, it can't do that reliably)

**Why:** Long tool chains drift quietly. The reflector catches goal misalignment, not execution errors — that's layer 3's job.

---

## Failure Modes to Keep in Mind

| Layer               | What it solves             | What it doesn't solve                           |
| ------------------- | -------------------------- | ----------------------------------------------- |
| STRIPS descriptions | Tool selection, sequencing | Probabilistic/partial postconditions            |
| Query planner       | Greedy local mistakes      | Planner shares LLM's blind spots                |
| State validator     | Execution failures         | Needs you to define what valid state looks like |
| Reflector           | Semantic drift             | Can confidently miss its own errors             |

---

## Implementation Notes for OpenRouter

- Use a cheap/fast model for the **query planner pass** (e.g. mistral-small, gemini-flash) — it's speculative, not final
- Use a stronger model for the **executor** and **reflector**
- State validator should be **pure code** — no LLM call, just structured comparison of expected vs actual fields
- Keep the planner's lookahead shallow (**max 2-3 steps**) to avoid exponential token cost
- STRIPS tool descriptions go in the **system prompt**, not user prompt

---

## Dataset: HotPotQA (distractor config)

```python
from datasets import load_dataset
ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
```

### Schema

```
{
  "id":               unique string
  "question":         the multi-hop question
  "answer":           ground truth answer
  "type":             "bridge" | "comparison"
  "level":            "easy" | "medium" | "hard"
  "supporting_facts": [
    { "title": "Wikipedia article title", "sent_id": 0 }
    // which sentences are actually needed to answer
  ],
  "context": [
    { "title": "Article A", "sentences": ["sent1", "sent2", ...] },
    // includes distractor articles NOT needed to answer
  ]
}
```

### Fields that matter for this framework

| Field                  | Why it matters                                                                                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type == "bridge"`     | Requires chaining — hop 1 finds entity X, hop 2 uses X to find answer. This is your cross-join retrieval case                                                                   |
| `type == "comparison"` | Just compares two entities, less useful for testing chaining                                                                                                                    |
| `supporting_facts`     | Ground truth for your **state validator (layer 3)** — tells you exactly which articles and sentences were needed. Check if your planner's retrieval sequence actually hit these |
| `context`              | Includes distractor articles on purpose — tests if your planner avoids retrieving irrelevant documents                                                                          |
| `level`                | Filter to `hard` for meaningful stress testing                                                                                                                                  |

### Recommended test set filter

```python
test_cases = [
    row for row in ds["validation"]
    if row["type"] == "bridge" and row["level"] == "hard"
]
# Start with 20 cases
test_cases = test_cases[:20]
```

### How to map dataset fields to your framework layers

- `question` → user query input to the planner
- `supporting_facts` → expected postcondition (what the state validator checks against)
- `context` → your mock vector DB (feed this as the retrieval pool)
- `answer` → reflector ground truth (did final output match this?)

---

## Reference Papers to Read

- **Reflexion** — Shinn et al. 2023 (LLM self-reflection pattern)
- **Tree of Thoughts** — Yao et al. 2023 (tree search over reasoning steps)
- **ReAct** — Yao et al. 2022 (observe-reason-act loop, closest to this architecture)
- **STRIPS planning** — Fikes & Nilsson 1971 (the OG precondition/postcondition formalism)

---

## The Honest Ceiling

None of this is a world model. LeCun's criticism still stands — the planner is simulating consequences using the same LLM that will execute them, so they share the same blind spots. What this framework does is make failures **more structured, more detectable, and more recoverable**. That's enough to be useful in practice even if it's not a theoretical solution.
