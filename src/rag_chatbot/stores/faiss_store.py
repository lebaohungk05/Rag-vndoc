import json
from pathlib import Path

import faiss
import numpy as np

from rag_chatbot.protocols import BaseVectorStore, Chunk


class FaissStore(BaseVectorStore):
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "index.faiss"
        self.meta_path = self.index_dir / "chunks.json"
        self._chunks: list[Chunk] = []
        self._index: faiss.Index | None = None

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            raise ValueError("No chunks to add.")
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        # Thử load index cũ nếu tồn tại để hỗ trợ incremental indexing
        try:
            self._load_if_needed()
        except FileNotFoundError:
            pass

        matrix = np.array(embeddings, dtype=np.float32)
        dimension = matrix.shape[1]

        if self._index is None:
            self._index = faiss.IndexFlatIP(dimension)
            self._chunks = chunks
        else:
            if self._index.d != dimension:
                raise ValueError(f"Embedding dimension mismatch: {dimension} vs {self._index.d}")
            self._chunks.extend(chunks)

        self._index.add(matrix)
        self._save()

    def query(self, query_vector: list[float], top_k: int = 5) -> list[Chunk]:
        self._load_if_needed()
        query = np.array([query_vector], dtype=np.float32)
        _, indices = self._index.search(query, top_k)
        return [self._chunks[idx] for idx in indices[0] if idx >= 0]

    def _save(self) -> None:
        if self._index is None:
            return
        faiss.write_index(self._index, str(self.index_path))
        serialized = [{"text": c.text, "metadata": c.metadata} for c in self._chunks]
        self.meta_path.write_text(json.dumps(serialized, ensure_ascii=False), encoding="utf-8")

    def _load_if_needed(self) -> None:
        if self._index is not None:
            return
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("FAISS index not found. Run ingestion first.")
        self._index = faiss.read_index(str(self.index_path))
        raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self._chunks = [Chunk(text=item["text"], metadata=item["metadata"]) for item in raw]

