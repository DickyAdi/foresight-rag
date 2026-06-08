"""Eval harness: run framework and/or baseline on N HotPotQA bridge/hard cases.

Usage:
  uv run python -m foresight.eval.run_eval --n 3 --mode framework --strategy imagined
  uv run python -m foresight.eval.run_eval --n 20 --mode all      # baseline + both foresights

Modes:
  baseline   greedy ReAct only
  framework  framework only (uses --strategy: imagined | grounded)
  both       baseline + framework(--strategy)
  all        baseline + framework(imagined) + framework(grounded)  — the 3 arms

Output:
  - Console summary table (arms side-by-side)
  - JSON results at src/foresight/eval/results/<timestamp>.json  (gitignored)

Gold (answer, supporting_facts) is used ONLY for metrics, equally across arms.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from foresight.adapters import factory
from foresight.config import Config
from foresight.core.planner import Planner
from foresight.core.reflector import Reflector
from foresight.core.relevance import RelevanceScorer
from foresight.core.strategies import GroundedStrategy, ImaginedStrategy
from foresight.core.validator import Validator
from foresight.data.load_hotpot import load_hotpot
from foresight.eval import metrics as M
from foresight.orchestration.baseline_graph import build_baseline_graph
from foresight.orchestration.case_store import build_case_store
from foresight.orchestration.graph import build_framework_graph

RESULTS_DIR = Path(__file__).parent / "results"
_TITLE_RE = re.compile(r"\[([^\]]+)\]")


# --------------------------------------------------------------------------- arms

def _framework_recursion_limit(cfg: Config) -> int:
    # planner/executor/validator/advance|replan per cycle; bound by hops x replan budget.
    return cfg.max_hops * (cfg.max_replans + 1) * 4 + 10


def _build_framework_graph(case_store, embedder, chats, cfg: Config, strategy_name: str):
    relevance = RelevanceScorer(embedder, cfg.relevance_quantile_q)
    validator = Validator(relevance, cfg.min_chunks)
    if strategy_name == "grounded":
        strategy = GroundedStrategy(chats["planner"], embedder, case_store, relevance, cfg)
    else:
        strategy = ImaginedStrategy(chats["planner"], cfg)
    planner = Planner(strategy, cfg)
    reflector = Reflector(chats["reflector"])
    return build_framework_graph(
        planner, validator, reflector, case_store, embedder, chats["executor"], cfg)


def _run_framework(case: dict, case_store, embedder, chats, cfg: Config, strategy_name: str) -> dict:
    graph = _build_framework_graph(case_store, embedder, chats, cfg, strategy_name)
    state = {
        "question": case["question"], "working_memory": [], "last_retrieved": [],
        "hop": 0, "replan_count": 0, "validation_feedback": None, "trace": [],
    }
    result = graph.invoke(state, config={"recursion_limit": _framework_recursion_limit(cfg)})

    working_memory = result.get("working_memory", [])
    titles = list(dict.fromkeys(c.title for c in working_memory))
    reflection = result.get("reflection")
    validation = result.get("validation")
    replan_count = result.get("replan_count", 0)
    return _score(
        case,
        answer=result.get("final_answer", ""),
        retrieved_titles=titles,
        replan_count=replan_count,
        hops=result.get("hop", 0),
        alignment=getattr(reflection, "alignment_score", None),
        exhausted=bool(validation is not None and not validation.ok
                       and replan_count >= cfg.max_replans),
    )


def _run_baseline(case: dict, case_store, embedder, chat_raw, cfg: Config) -> dict:
    graph = build_baseline_graph(chat_raw, case_store, embedder, cfg)
    result = graph.invoke(
        {"messages": [("user", case["question"])]},
        config={"recursion_limit": cfg.baseline_max_steps * 2 + 5},
    )
    messages = result.get("messages", [])
    answer = getattr(messages[-1], "content", "") if messages else ""

    titles: list[str] = []
    tool_calls = 0
    for m in messages:
        if getattr(m, "type", None) == "tool" and getattr(m, "name", None) == "retrieve":
            titles.extend(_TITLE_RE.findall(getattr(m, "content", "") or ""))
            tool_calls += 1
    titles = list(dict.fromkeys(titles))
    return _score(case, answer=answer, retrieved_titles=titles,
                  replan_count=0, hops=tool_calls, alignment=None, exhausted=False)


# ------------------------------------------------------------------------ scoring

def _score(case: dict, *, answer, retrieved_titles, replan_count, hops,
           alignment, exhausted) -> dict:
    required = list(dict.fromkeys(sf["title"] for sf in case["supporting_facts"]))
    answer = answer or ""
    return {
        "answer": answer,
        "em": M.exact_match(answer, case["answer"]),
        "f1": M.token_f1(answer, case["answer"]),
        "recall": M.retrieval_recall(retrieved_titles, required),
        "precision": M.retrieval_precision(retrieved_titles, required),
        "replan_count": replan_count,
        "hops": hops,
        "alignment": alignment,
        "exhausted": exhausted,
        "retrieved_titles": retrieved_titles,
    }


def _aggregate(records: list[dict]) -> dict:
    ok = [r for r in records if "error" not in r]
    if not ok:
        return {"n": 0, "errors": len(records)}
    aligns = [r["alignment"] for r in ok if r["alignment"] is not None]
    return {
        "n": len(ok),
        "errors": len(records) - len(ok),
        "em": mean(r["em"] for r in ok),
        "f1": mean(r["f1"] for r in ok),
        "recall": mean(r["recall"] for r in ok),
        "precision": mean(r["precision"] for r in ok),
        "avg_replans": mean(r["replan_count"] for r in ok),
        "avg_hops": mean(r["hops"] for r in ok),
        "avg_alignment": mean(aligns) if aligns else None,
        "exhausted_rate": mean(1.0 if r["exhausted"] else 0.0 for r in ok),
    }


# --------------------------------------------------------------------------- main

def _arms_for(mode: str, strategy: str) -> list[str]:
    if mode == "baseline":
        return ["baseline"]
    if mode == "framework":
        return [f"framework:{strategy}"]
    if mode == "both":
        return ["baseline", f"framework:{strategy}"]
    if mode == "all":
        return ["baseline", "framework:imagined", "framework:grounded"]
    raise ValueError(f"unknown mode: {mode}")


def _print_table(aggregates: dict[str, dict]) -> None:
    cols = ["em", "f1", "recall", "precision", "avg_replans", "avg_hops",
            "avg_alignment", "exhausted_rate"]
    header = f"{'arm':<22}" + "".join(f"{c:>14}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for arm, agg in aggregates.items():
        if agg.get("n", 0) == 0:
            print(f"{arm:<22}{'(all cases errored)':>14}")
            continue
        cells = []
        for c in cols:
            v = agg.get(c)
            cells.append(f"{'-':>14}" if v is None else f"{v:>14.3f}")
        print(f"{arm:<22}" + "".join(cells) + f"   (n={agg['n']}, err={agg['errors']})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Foresight RAG eval harness")
    parser.add_argument("--n", type=int, default=3, help="number of cases")
    parser.add_argument("--mode", choices=["baseline", "framework", "both", "all"],
                        default="both")
    parser.add_argument("--strategy", choices=["imagined", "grounded"], default="imagined")
    args = parser.parse_args()

    cfg = Config()
    arms = _arms_for(args.mode, args.strategy)
    cases = load_hotpot(args.n)

    # Shared deps (one embedding space; chats reused across cases).
    embedder = factory.make_embedder(cfg)
    chats = {
        "planner": factory.make_planner_chat(cfg),
        "executor": factory.make_executor_chat(cfg),
        "reflector": factory.make_reflector_chat(cfg),
    }
    chat_raw = factory.make_raw_chat(cfg.executor_model, cfg)

    records: dict[str, list[dict]] = {arm: [] for arm in arms}

    for i, case in enumerate(cases):
        collection = f"case-{i}-{case['id']}"
        store = factory.make_store(collection)
        build_case_store(case, embedder, store, cfg)
        try:
            for arm in arms:
                try:
                    if arm == "baseline":
                        rec = _run_baseline(case, store, embedder, chat_raw, cfg)
                    else:
                        strategy_name = arm.split(":", 1)[1]
                        rec = _run_framework(case, store, embedder, chats, cfg, strategy_name)
                    rec["case_id"] = case["id"]
                    records[arm].append(rec)
                    print(f"[{i + 1}/{len(cases)}] {arm:<20} "
                          f"EM={rec['em']:.0f} F1={rec['f1']:.2f} "
                          f"recall={rec['recall']:.2f} prec={rec['precision']:.2f}")
                except Exception as exc:  # per-arm isolation (design/08 #9)
                    traceback.print_exc()
                    records[arm].append({"case_id": case["id"], "error": str(exc)})
        finally:
            if hasattr(store, "drop"):
                store.drop()

    aggregates = {arm: _aggregate(recs) for arm, recs in records.items()}
    _print_table(aggregates)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps({
        "n": args.n, "mode": args.mode, "strategy": args.strategy,
        "config": asdict(cfg),
        "aggregates": aggregates,
        "records": records,
    }, indent=2, ensure_ascii=False))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
