"""Unit tests for Layer 2 — strategies + planner (offline, fake ChatModel/ports)."""
from foresight.config import Config
from foresight.core.planner import Planner
from foresight.core.relevance import RelevanceScorer
from foresight.core.strategies import GroundedStrategy, ImaginedStrategy
from foresight.core.types import CandidateQuery, Chunk, PlanDecision, PlannerCandidates


class FakeChat:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def complete(self, messages, *, schema=None):
        self.calls.append({"messages": messages, "schema": schema})
        return self.response


class FakeEmbedder:
    def __init__(self, mapping):
        self._mapping = mapping

    def embed_documents(self, texts):
        return [self._mapping[t] for t in texts]

    def embed_query(self, text):
        return self._mapping[text]


class FakeStore:
    def __init__(self, results, pool=None):
        self._results = results  # tuple(embedding) -> list[Chunk]
        self._pool = pool or []

    def query(self, embedding, k):
        return self._results[tuple(embedding)]

    def get_embeddings(self):
        return self._pool


class FakeStrategy:
    def __init__(self, decision):
        self.decision = decision
        self.called = False

    def decide(self, question, working_memory, feedback=None):
        self.called = True
        return self.decision


# --- ImaginedStrategy ---

def test_imagined_commits_highest_viability():
    candidates = PlannerCandidates(candidates=[
        CandidateQuery(query="low", viability=0.3),
        CandidateQuery(query="high", viability=0.9),
    ])
    dec = ImaginedStrategy(FakeChat(candidates), Config()).decide("q", [], None)
    assert dec.action == "retrieve"
    assert dec.query == "high"
    assert dec.viability == 0.9


def test_imagined_prunes_empty_query_then_falls_back():
    candidates = PlannerCandidates(candidates=[CandidateQuery(query="   ", viability=0.9)])
    dec = ImaginedStrategy(FakeChat(candidates), Config()).decide("the question", [], None)
    assert dec.query == "the question"
    assert "fallback" in dec.reasoning


def test_imagined_elective_finalize():
    candidates = PlannerCandidates(
        candidates=[CandidateQuery(query="x", viability=0.9)], ready_to_answer=True)
    memory = [Chunk(title="A", text="t", sent_id=0)]
    dec = ImaginedStrategy(FakeChat(candidates), Config()).decide("q", memory, None)
    assert dec.action == "answer"


# --- GroundedStrategy (scores candidates against the QUESTION, design/04 Option A) ---

def test_grounded_ranks_by_question_relevance_overriding_imagined():
    # "bad" has the higher imagined viability, but "good" retrieves a chunk that is
    # more relevant to the QUESTION, so grounded commits "good".
    candidates = PlannerCandidates(candidates=[
        CandidateQuery(query="good", viability=0.1),
        CandidateQuery(query="bad", viability=0.9),
    ])
    embedder = FakeEmbedder({
        "the question": [1.0, 0.0],
        "good": [1.0, 0.0],
        "bad": [0.0, 1.0],
        "A: ga": [1.0, 0.0],   # good's retrieved chunk: aligned with the question
        "B: bb": [0.0, 1.0],   # bad's retrieved chunk: orthogonal to the question
    })
    store = FakeStore(
        results={
            (1.0, 0.0): [Chunk(title="A", text="ga", sent_id=0)],
            (0.0, 1.0): [Chunk(title="B", text="bb", sent_id=1)],
        },
        pool=[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],  # 1 question-aligned, 2 distractors
    )
    relevance = RelevanceScorer(embedder=None, quantile_q=0.7)
    strat = GroundedStrategy(FakeChat(candidates), embedder, store, relevance, Config())
    dec = strat.decide("the question", [], None)
    assert dec.query == "good"
    assert dec.viability == 1.0  # beats the whole pool on question-relevance


# --- Planner guards ---

def test_planner_forces_answer_at_max_hops():
    strat = FakeStrategy(PlanDecision(action="retrieve", query="x"))
    dec = Planner(strat, Config(max_hops=3)).plan("q", [], hop=3)
    assert dec.action == "answer"
    assert not strat.called  # guard short-circuits before the strategy


def test_planner_forbids_answer_from_empty_memory():
    strat = FakeStrategy(PlanDecision(action="answer", query=None))
    dec = Planner(strat, Config(allow_direct_answer=False)).plan("the q", [], hop=0)
    assert dec.action == "retrieve"
    assert dec.query == "the q"


def test_planner_allows_answer_when_configured():
    strat = FakeStrategy(PlanDecision(action="answer"))
    dec = Planner(strat, Config(allow_direct_answer=True)).plan("q", [], hop=0)
    assert dec.action == "answer"


def test_planner_delegates_normally():
    strat = FakeStrategy(PlanDecision(action="retrieve", query="x"))
    dec = Planner(strat, Config()).plan("q", [Chunk(title="A", text="t", sent_id=0)], hop=1)
    assert dec.action == "retrieve"
    assert dec.query == "x"
    assert strat.called
