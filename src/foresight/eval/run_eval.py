"""Eval harness: run framework and/or baseline on N HotPotQA bridge/hard cases.

Usage:
  uv run python -m foresight.eval.run_eval --n 3 --mode framework
  uv run python -m foresight.eval.run_eval --n 20 --mode both

Output:
  - Console summary table (framework vs baseline side-by-side)
  - JSON results file at src/foresight/eval/results/<timestamp>.json
"""
# TODO: implement main()
#   1. Parse --n, --mode (framework | baseline | both) via argparse
#   2. Load n cases via data.load_hotpot(n)
#   3. For each case:
#      a. Build per-question Chroma store from case["context"]
#      b. If mode includes "framework": run build_framework_graph(store).invoke(initial_state)
#      c. If mode includes "baseline": run build_baseline_graph(store).invoke(initial_state)
#      d. Compute metrics for each: EM, F1, retrieval recall/precision, replan_count, validator flags, reflector score
#      e. Drop the Chroma collection
#   4. Aggregate and print comparison table
#   5. Write results JSON to eval/results/<timestamp>.json
