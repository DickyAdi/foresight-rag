"""Ports — the three genuine external boundaries, as typing.Protocol interfaces.

Zero runtime deps. `core` depends on these; `adapters` implement them. This is the
seam that keeps LangChain/chromadb/OpenRouter out of the framework logic so the
core can be extracted as a minimal-dependency library later.

See design/12-implementation-architecture.md.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from foresight.core.types import Chunk


@runtime_checkable
class ChatModel(Protocol):
    """An LLM chat interface. `schema` (a pydantic model) requests structured output."""
    def complete(self, messages: list[dict], *, schema: type | None = None) -> object: ...


@runtime_checkable
class Embedder(Protocol):
    """Text embedding interface (hosted or local). Same space used by store + validator."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Per-question vector store. Accepts precomputed embeddings and can return them
    back (for the validator's vector-reuse — see design/06)."""
    def add(self, ids: list[str], embeddings: list[list[float]],
            documents: list[str], metadatas: list[dict]) -> None: ...
    def query(self, embedding: list[float], k: int) -> list["Chunk"]: ...
    def get_embeddings(self) -> list[list[float]]: ...
