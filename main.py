import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Thêm src vào sys.path để import các module local
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from rag_chatbot.pipeline.ask import run_ask
from rag_chatbot.pipeline.ingest import run_index

app = FastAPI(title="Vietnamese RAG Chatbot API")

class QuestionRequest(BaseModel):
    question: str
    index_dir: str = "artifacts/faiss_index"

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    doc_ids: List[str]

class IngestRequest(BaseModel):
    raw_dir: str = "data/raw"
    index_dir: str = "artifacts/faiss_index"
    chunks_output: str = "data/processed/chunks.json"

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        result = run_ask(question=request.question, index_dir=request.index_dir)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_data(request: IngestRequest):
    try:
        chunks = run_index(
            raw_dir=request.raw_dir,
            index_dir=request.index_dir,
            chunks_output_path=request.chunks_output
        )
        return {"message": f"Successfully indexed {len(chunks)} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
