from rag_chatbot.protocols import BaseEmbedder


class SbertEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def embed_query(self, text: str) -> list[float]:
        self._ensure_model()
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

