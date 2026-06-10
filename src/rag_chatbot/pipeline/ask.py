from rag_chatbot.config import get_settings
from rag_chatbot.embedders.sbert_embedder import SbertEmbedder
from rag_chatbot.models.ollama_llm import OllamaLLM
from rag_chatbot.stores.hybrid_store import HybridStore


def run_ask(question: str, index_dir: str, alpha: float = 0.5) -> dict:
    settings = get_settings()
    embedder = SbertEmbedder(settings.embedding_model)
    store = HybridStore(index_dir=index_dir, alpha=alpha)
    query_vector = embedder.embed_query(question)
    chunks = store.query(query_vector, top_k=settings.top_k, query_text=question)
    context = "\n\n".join(chunk.text for chunk in chunks[: settings.max_context_chunks])

    llm = OllamaLLM(model=settings.llm_model)

    answer = llm.generate(question=question, context=context)
    return {
        "question": question,
        "answer": answer,
        "sources": [chunk.metadata.get("source") for chunk in chunks],
        "doc_ids": [chunk.metadata.get("doc_id") for chunk in chunks],
    }

