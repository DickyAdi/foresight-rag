"""Load and filter HotPotQA (distractor config) for eval.

HuggingFace stores supporting_facts and context as column-oriented parallel arrays,
not as lists of dicts. This module adapts them to the row-oriented shape the rest
of the codebase expects.

Expected output shape per row:
  {
    "id": str,
    "question": str,
    "answer": str,
    "type": "bridge" | "comparison",
    "level": "easy" | "medium" | "hard",
    "supporting_facts": [{"title": str, "sent_id": int}, ...],
    "context": [{"title": str, "sentences": [str, ...]}, ...],
  }
"""
# TODO: implement load_hotpot(n: int = 20) -> list[dict]
#   1. load_dataset("hotpotqa/hotpot_qa", "distractor")
#   2. filter ds["validation"] where type == "bridge" and level == "hard"
#   3. take first n rows
#   4. adapt each row:
#      - supporting_facts: zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"])
#      - context: zip(row["context"]["title"], row["context"]["sentences"])
#   5. return list of adapted dicts
