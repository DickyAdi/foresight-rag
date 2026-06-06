# OpenRouter base URL and model IDs per role, thresholds, and eval knobs.
# All values here are the single source of truth — never hardcode elsewhere.

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Mixed-tier model strategy: cheap for speculative planning, strong for execution/reflection.
PLANNER_MODEL = "google/gemini-2.0-flash-001"
EXECUTOR_MODEL = "anthropic/claude-sonnet-4"
REFLECTOR_MODEL = "anthropic/claude-sonnet-4"

# Retrieval
TOP_K = 5

# Planner beam search
BEAM_WIDTH = 3
MAX_LOOKAHEAD_STEPS = 3

# Re-plan budget per question
MAX_REPLANS = 2

# Validator thresholds
MIN_CHUNKS = 2
SEMANTIC_SIMILARITY_THRESHOLD = 0.5
