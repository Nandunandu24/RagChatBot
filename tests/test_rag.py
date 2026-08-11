import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from config import settings
from embeddings import get_embedding_model
from vector_store import get_vector_store_manager
from rag_graph import run_rag_pipeline
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_embedding_model():
    """Verify HuggingFace embedding model produces correct vector dimension (384)."""
    model = get_embedding_model()
    vec = model.embed_query("Test query for HuggingFace embeddings")
    assert isinstance(vec, list)
    assert len(vec) == settings.EMBEDDING_DIMENSION

def test_api_health_endpoint():
    """Test FastAPI GET / health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "embedding_model" in data

def test_api_sample_queries_endpoint():
    """Test FastAPI GET /sample-queries endpoint."""
    response = client.get("/sample-queries")
    assert response.status_code == 200
    data = response.json()
    assert "sample_queries" in data
    assert len(data["sample_queries"]) >= 5

def test_rag_pipeline_response_structure():
    """Verify LangGraph RAG pipeline returns exact required response structure."""
    result = run_rag_pipeline("What is Agentic AI?")
    
    assert "query" in result
    assert "final_answer" in result
    assert "retrieved_context_chunks" in result
    assert "confidence_score" in result
    
    assert isinstance(result["final_answer"], str)
    assert isinstance(result["confidence_score"], float)
    assert isinstance(result["retrieved_context_chunks"], list)
    assert 0.0 <= result["confidence_score"] <= 1.0

def test_anti_hallucination_out_of_scope():
    """Verify anti-hallucination fallback when query is out of scope."""
    result = run_rag_pipeline("What is the recipe for cooking lasagna at 200 degrees?")
    
    assert "final_answer" in result
    # Out of scope query should either state insufficient information or have low confidence score
    is_refusal = "insufficient information" in result["final_answer"].lower() or result["confidence_score"] < 0.25
    assert is_refusal
