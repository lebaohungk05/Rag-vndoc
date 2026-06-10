from pathlib import Path

from rag_chatbot.protocols import BaseLoader, Document


class TxtLoader(BaseLoader):
    def load(self, path: str) -> list[Document]:
        root = Path(path)
        documents: list[Document] = []
        for file_path in root.rglob("*.txt"):
            text = file_path.read_text(encoding="utf-8")
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
        return documents

