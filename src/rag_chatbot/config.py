from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
    vector_store: str = os.getenv("VECTOR_STORE", "faiss")
    top_k: int = int(os.getenv("TOP_K", "5"))
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:3b")
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))


def get_settings() -> Settings:
    return Settings()

