from pathlib import Path
from rag_chatbot.protocols import BaseLoader, Document

class PdfLoader(BaseLoader):
    def load(self, path: str) -> list[Document]:
        root = Path(path)
        documents: list[Document] = []
        
        # Kiểm tra xem fitz (PyMuPDF) đã được cài đặt chưa
        try:
            import fitz
        except ImportError:
            print("Cảnh báo: Thư viện 'pymupdf' chưa được cài đặt. Bỏ qua các file PDF.")
            return []

        for file_path in root.rglob("*.pdf"):
            try:
                text = ""
                with fitz.open(str(file_path)) as doc:
                    for page in doc:
                        text += page.get_text()
                
                if text.strip():
                    documents.append(
                        Document(
                            text=text,
                            metadata={
                                "doc_id": file_path.stem,
                                "source": str(file_path),
                                "title": file_path.name,
                                "domain": "unknown",
                            },
                        )
                    )
            except Exception as e:
                print(f"Lỗi khi đọc file PDF {file_path}: {e}")
                
        return documents

