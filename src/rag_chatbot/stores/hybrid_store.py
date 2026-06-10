import json
import pickle
from pathlib import Path
import numpy as np

from rank_bm25 import BM25Okapi
import underthesea

from rag_chatbot.protocols import BaseVectorStore, Chunk
from rag_chatbot.stores.faiss_store import FaissStore


def tokenize_vi(text: str) -> list[str]:
    """Tách từ tiếng Việt bằng underthesea và chuyển thành lowercase."""
    return underthesea.word_tokenize(text.lower())


class HybridStore(BaseVectorStore):
    def __init__(self, index_dir: str, alpha: float = 0.5):
        """
        alpha: Trọng số của Semantic (FAISS). 
               alpha = 0.5 nghĩa là 50% FAISS + 50% BM25.
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_path = self.index_dir / "bm25_index.pkl"
        
        self.faiss_store = FaissStore(index_dir=index_dir)
        self.bm25: BM25Okapi | None = None
        self.alpha = alpha
        
        # Thử load BM25 nếu đã có
        try:
            self._load_bm25()
        except FileNotFoundError:
            pass

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        # 1. Nạp vào FAISS
        self.faiss_store.add_chunks(chunks, embeddings)
        
        # 2. Xây dựng index BM25
        # Lưu ý: Vì FAISS store nối thêm dữ liệu vào self._chunks, 
        # ta cần lấy toàn bộ chunks từ FAISS để build lại BM25 cho đồng bộ.
        all_chunks = self.faiss_store._chunks
        
        print("Đang tiến hành tách từ Tiếng Việt (Word Segmentation) để tạo index BM25...")
        tokenized_corpus = [tokenize_vi(chunk.text) for chunk in all_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Lưu BM25 ra file
        with open(self.bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)

    def query(self, query_vector: list[float], top_k: int = 5, query_text: str = "") -> list[Chunk]:
        self.faiss_store._load_if_needed()
        self._load_bm25()
        
        if not self.bm25 or not self.faiss_store._chunks:
            return []

        if not query_text:
            raise ValueError("HybridStore yêu cầu query_text để chạy thuật toán BM25.")

        # --- 1. Lấy điểm FAISS (Semantic) ---
        query_np = np.array([query_vector], dtype=np.float32)
        # Tìm top_k * 2 để có tệp ứng viên rộng hơn trước khi rerank bằng công thức lai
        candidate_k = min(top_k * 10, len(self.faiss_store._chunks)) 
        faiss_distances, faiss_indices = self.faiss_store._index.search(query_np, candidate_k)
        
        # --- 2. Lấy điểm BM25 (Keyword) ---
        tokenized_query = tokenize_vi(query_text)
        bm25_all_scores = self.bm25.get_scores(tokenized_query)
        
        # --- 2b. Lọc độ liên quan bằng điểm thô (Relevance Filter) ---
        best_faiss = faiss_distances[0][0] if len(faiss_distances[0]) > 0 else 0.0
        best_bm25 = max(bm25_all_scores) if len(bm25_all_scores) > 0 else 0.0
        if best_faiss < 0.35 and best_bm25 < 9.0:
            return []
        
        # Chuẩn hóa điểm FAISS về [0, 1]
        faiss_scores = faiss_distances[0]
        if max(faiss_scores) > min(faiss_scores):
            faiss_scores = (faiss_scores - min(faiss_scores)) / (max(faiss_scores) - min(faiss_scores) + 1e-9)
        
        # Chuẩn hóa điểm BM25 về [0, 1]
        if max(bm25_all_scores) > min(bm25_all_scores):
            bm25_norm_scores = (bm25_all_scores - min(bm25_all_scores)) / (max(bm25_all_scores) - min(bm25_all_scores) + 1e-9)
        else:
            bm25_norm_scores = bm25_all_scores

        # --- 3. Kết hợp điểm (Reciprocal Rank Fusion hoặc Trọng số tuyến tính) ---
        # Ta dùng trọng số tuyến tính: Score = alpha * Semantic + (1 - alpha) * Keyword
        final_scores = []
        all_chunks = self.faiss_store._chunks
        
        for idx in range(len(all_chunks)):
            # Tìm điểm faiss của idx này, nếu không nằm trong candidate_k thì cho điểm = 0
            f_score = 0.0
            if idx in faiss_indices[0]:
                pos = np.where(faiss_indices[0] == idx)[0][0]
                f_score = faiss_scores[pos]
                
            b_score = bm25_norm_scores[idx]
            
            combined_score = self.alpha * f_score + (1.0 - self.alpha) * b_score
            final_scores.append((idx, combined_score))
            
        # Sắp xếp lại theo điểm kết hợp giảm dần
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Lấy top_k kết quả
        top_indices = [idx for idx, score in final_scores[:top_k]]
        return [all_chunks[idx] for idx in top_indices]

    def _load_bm25(self):
        if self.bm25 is None:
            if not self.bm25_path.exists():
                raise FileNotFoundError("Chưa tìm thấy file index BM25. Hãy chạy lại ingest.")
            with open(self.bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
