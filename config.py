import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Agentic AI RAG Chatbot"
    DEBUG: bool = False
    
    # PDF Source
    PDF_URL: str = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
    PDF_PATH: Path = BASE_DIR / "Ebook-Agentic-AI.pdf"
    
    # Embeddings Config (HuggingFace)
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Vector DB Config
    VECTOR_DB_TYPE: str = "auto"  # "pinecone", "chroma", or "auto" (pinecone if key exists, else chroma)
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "chroma_db"
    
    # Pinecone Config
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "agentic-ai-rag"
    PINECONE_ENVIRONMENT: str = "us-east-1"
    
    # Google Gemini Config
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"  # Supports gemini-1.5-flash, gemini-2.0-flash, etc.
    
    # RAG Search Settings
    TOP_K_RETRIEVAL: int = 6
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 120

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
