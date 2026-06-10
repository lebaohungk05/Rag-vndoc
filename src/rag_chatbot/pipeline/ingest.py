import json
from pathlib import Path

from rag_chatbot.chunkers.recursive_chunker import RecursiveChunker
from rag_chatbot.config import get_settings
from rag_chatbot.embedders.sbert_embedder import SbertEmbedder
from rag_chatbot.loaders.txt_loader import TxtLoader
from rag_chatbot.loaders.pdf_loader import PdfLoader
from rag_chatbot.protocols import Chunk, Document
from rag_chatbot.stores.hybrid_store import HybridStore


def run_load(raw_dir: str, output_path: str) -> list[Document]:
    loaders = [TxtLoader(), PdfLoader()]
    all_documents: list[Document] = []
    
    for loader in loaders:
        try:
            docs = loader.load(raw_dir)
            all_documents.extend(docs)
        except NotImplementedError:
            continue
            
    payload = [{"text": doc.text, "metadata": doc.metadata} for doc in all_documents]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return all_documents


def run_index(raw_dir: str, index_dir: str, chunks_output_path: str, use_preprocessed: bool = False, docs_path: str = "") -> list[Chunk]:
    settings = get_settings()
    
    if use_preprocessed and Path(docs_path).exists():
        raw_data = json.loads(Path(docs_path).read_text(encoding="utf-8"))
        documents = [Document(text=d["text"], metadata=d["metadata"]) for d in raw_data]
    else:
        documents = run_load(raw_dir, docs_path if docs_path else "data/processed/documents.json")
        
    chunker = RecursiveChunker(settings.chunk_size, settings.chunk_overlap)
    chunks = chunker.split(documents)
    
    embedder = SbertEmbedder(settings.embedding_model)
    embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
    
    store = HybridStore(index_dir=index_dir, alpha=0.5)
    store.add_chunks(chunks, embeddings)

    payload = [{"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks]
    Path(chunks_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(chunks_output_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return chunks

