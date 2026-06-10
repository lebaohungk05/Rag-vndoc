from rag_chatbot.protocols import BaseChunker, Chunk, Document


class RecursiveChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Ưu tiên ngắt theo đoạn (\n\n), dòng (\n), rồi mới đến câu (.) và từ ( )
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split(self, documents: list[Document]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in documents:
            text_chunks = self._recursive_split(doc.text, self.separators)
            
            # Gộp các mẩu nhỏ lại thành chunk đủ size
            merged_chunks = self._merge_splits(text_chunks)
            
            for i, text in enumerate(merged_chunks):
                metadata = {
                    **doc.metadata,
                    "chunk_id": f"{doc.metadata.get('doc_id', 'doc')}_{i}",
                }
                chunk = Chunk(text=text, metadata=metadata)
                chunk.validate_metadata()
                all_chunks.append(chunk)
        return all_chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Chia nhỏ văn bản dựa trên danh sách các ký tự phân tách ưu tiên."""
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)
        
        for s in splits:
            if len(s) <= self.chunk_size:
                final_chunks.append(s)
            else:
                if not new_separators:
                    final_chunks.append(s)
                else:
                    final_chunks.extend(self._recursive_split(s, new_separators))
        return final_chunks

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Gộp các đoạn nhỏ lại sao cho tối ưu hóa chunk_size và chunk_overlap."""
        merged = []
        current_chunk = []
        current_length = 0

        for s in splits:
            if current_length + len(s) > self.chunk_size and current_chunk:
                merged.append(" ".join(current_chunk))
                # Giữ lại một phần để overlap
                overlap_chunk = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) <= self.chunk_overlap:
                        overlap_chunk.insert(0, item)
                        overlap_len += len(item)
                    else:
                        break
                current_chunk = overlap_chunk
                current_length = overlap_len

            current_chunk.append(s)
            current_length += len(s)

        if current_chunk:
            merged.append(" ".join(current_chunk))
        return merged

