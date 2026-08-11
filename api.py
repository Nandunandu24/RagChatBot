from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from config import settings
from rag_graph import run_rag_pipeline
from ingest import run_ingestion

app = FastAPI(
    title="Agentic AI RAG API",
    description="FastAPI service for RAG chatbot grounded in Ebook-Agentic-AI.pdf",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

class ContextChunk(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: List[ContextChunk]
    confidence_score: float

class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    message: str

SAMPLE_QUERIES = [
    "What is Agentic AI according to the ebook?",
    "What are the types of agents based on functional versatility?",
    "What are the main components of an Anatomy of an Agentic AI System?",
    "How do Multi-Agent Systems orchestrate decision making?",
    "What is the difference between traditional automation and Agentic AI?",
    "What factors determine an organization's readiness for Agentic AI?"
]

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "vector_db": settings.VECTOR_DB_TYPE,
        "llm_model": settings.GEMINI_MODEL,
        "pdf_source": settings.PDF_URL
    }

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat_endpoint(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    try:
        result = run_rag_pipeline(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_endpoint(force_reingest: bool = False):
    try:
        count = run_ingestion(force_reingest=force_reingest)
        return IngestResponse(
            status="success",
            chunks_indexed=count,
            message="Ingestion completed successfully."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/sample-queries", tags=["Sample Queries"])
def get_sample_queries():
    return {"sample_queries": SAMPLE_QUERIES}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
