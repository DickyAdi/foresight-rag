"""Unit test for Layer 4 — the reflector (offline, fake ChatModel)."""
from foresight.core.reflector import Reflector
from foresight.core.types import Reflection


class FakeChat:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def complete(self, messages, *, schema=None):
        self.calls.append({"messages": messages, "schema": schema})
        return self.response


def test_reflector_returns_structured_reflection():
    chat = FakeChat(Reflection(alignment_score=0.8, reasoning="on goal"))
    out = Reflector(chat).reflect("Who founded X?", "Ada Lovelace", [])

    assert out.alignment_score == 0.8
    assert chat.calls[0]["schema"] is Reflection
    # gold-blind by construction: reflect() has no gold parameter; the prompt carries
    # only question + final answer (+ working memory).
    rendered = chat.calls[0]["messages"][0]["content"]
    assert "Ada Lovelace" in rendered
    assert "Who founded X?" in rendered
