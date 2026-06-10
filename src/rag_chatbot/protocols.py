from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Document:
    """Raw document loaded from data sources."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    """Chunked unit used for embedding and retrieval."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_metadata(self) -> None:
        """Validate required metadata fields used across the pipeline."""
        required_fields = {"doc_id", "chunk_id", "source"}
        missing = sorted(field for field in required_fields if field not in self.metadata)
        if missing:
            raise ValueError(f"Missing required chunk metadata fields: {missing}")


class BaseLoader(ABC):
    """Interface for loading files into normalized documents."""

    @abstractmethod
    def load(self, path: str) -> list[Document]:
        """Load documents from path."""


class BaseChunker(ABC):
    """Interface for splitting documents into chunks."""

    @abstractmethod
    def split(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into chunks."""


class BaseEmbedder(ABC):
    """Interface for transforming text into embeddings."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts."""


class BaseVectorStore(ABC):
    """Interface for vector storage and similarity search."""

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert chunks with embeddings."""

    @abstractmethod
    def query(self, query_vector: list[float], top_k: int = 5) -> list[Chunk]:
        """Retrieve top-k chunks for query embedding."""


class BaseLLM(ABC):
    """Interface for answer generation from retrieved context."""

    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        """Generate answer from question and retrieved context."""
