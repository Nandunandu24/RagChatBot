import os
from typing import TypedDict, List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from config import settings
from vector_store import get_vector_store_manager

# System prompt forcing factual grounding in the provided PDF context
SYSTEM_PROMPT = """Answer the question based only on the provided context chunks from the Agentic AI eBook.

Guidelines:
- Rely strictly on facts present in the context. Do not extrapolate or assume outside information.
- If the context does not contain enough information to answer the question, state:
  "Based on the provided Agentic AI eBook context, there is insufficient information to answer this question."
- Organize the response clearly with an Overview, Detailed Insights, and page citations.

Context:
{context}

Question: {question}
"""

class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Tuple[Document, float]]
    context_text: str
    final_answer: str
    confidence_score: float
    is_grounded: bool

def get_llm():
    """Initialize Gemini LLM instance using API key from settings or env."""
    api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY", "")
    
    if api_key and api_key.strip():
        models_to_try = [settings.GEMINI_MODEL, "gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
        from langchain_google_genai import ChatGoogleGenerativeAI

        for model_name in dict.fromkeys(models_to_try):
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=0.1,
                    max_output_tokens=1024
                )
                return llm
            except Exception as e:
                pass
    return None

def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """Retrieve top-k relevant document chunks from the vector store."""
    query = state["query"]
    vector_mgr = get_vector_store_manager()
    results = vector_mgr.similarity_search_with_score(query, k=settings.TOP_K_RETRIEVAL)
    return {"retrieved_docs": results}

def grade_context_node(state: RAGState) -> Dict[str, Any]:
    """Format context string and calculate initial similarity score."""
    retrieved_docs = state["retrieved_docs"]
    
    if not retrieved_docs:
        return {
            "context_text": "",
            "confidence_score": 0.0
        }

    formatted_chunks = []
    scores = []
    
    for idx, (doc, score) in enumerate(retrieved_docs):
        page_num = doc.metadata.get("page_number", "Unknown")
        formatted_chunks.append(f"[Page {page_num} | Score: {score:.2f}]\n{doc.page_content}")
        scores.append(score)

    context_str = "\n\n---\n\n".join(formatted_chunks)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    top_score = max(scores) if scores else 0.0
    composite_confidence = round(0.7 * top_score + 0.3 * avg_score, 4)

    return {
        "context_text": context_str,
        "confidence_score": composite_confidence
    }

def generate_answer_node(state: RAGState) -> Dict[str, Any]:
    """Generate answer using Gemini or local context summary."""
    query = state["query"]
    context_text = state["context_text"]
    confidence_score = state["confidence_score"]
    
    if not context_text or confidence_score < 0.10:
        return {
            "final_answer": "Based on the provided Agentic AI eBook context, there is insufficient information to answer this question.",
            "confidence_score": 0.0,
            "is_grounded": False
        }

    llm = get_llm()
    
    if llm:
        try:
            prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
            chain = prompt | llm
            response = chain.invoke({"context": context_text, "question": query})
            
            if isinstance(response.content, list):
                raw_text = "".join(
                    [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in response.content]
                )
            else:
                raw_text = str(response.content)
                
            answer = raw_text.strip()
        except Exception as e:
            answer = f"Error generating answer with Gemini: {str(e)}"
    else:
        retrieved_docs = state.get("retrieved_docs", [])
        chunk_details = []
        pages_cited = set()
        for idx, (doc, score) in enumerate(retrieved_docs):
            page_num = doc.metadata.get("page_number", "N/A")
            pages_cited.add(str(page_num))
            chunk_details.append(f"- **Insight {idx+1} (Page {page_num})**: {doc.page_content.strip()}")

        pages_str = ", ".join(sorted(pages_cited))
        insights_str = "\n\n".join(chunk_details)

        answer = (
            f"### Overview\n"
            f"Grounded answer extracted from eBook context:\n\n"
            f"### Key Insights\n"
            f"{insights_str}\n\n"
            f"### Source Pages\n"
            f"Pages: {pages_str}"
        )

    return {
        "final_answer": answer,
        "is_grounded": "insufficient information" not in answer.lower()
    }

def evaluate_grounding_node(state: RAGState) -> Dict[str, Any]:
    """Finalize confidence score based on grounding check."""
    final_answer = state["final_answer"]
    is_grounded = state.get("is_grounded", True)
    
    if "insufficient information" in final_answer.lower() or not is_grounded:
        final_confidence = 0.0
    else:
        final_confidence = 1.0

    return {"confidence_score": round(final_confidence, 4)}

def build_rag_graph():
    """Assemble LangGraph StateGraph workflow."""
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_context", grade_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("evaluate_grounding", evaluate_grounding_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_context")
    workflow.add_edge("grade_context", "generate_answer")
    workflow.add_edge("generate_answer", "evaluate_grounding")
    workflow.add_edge("evaluate_grounding", END)

    return workflow.compile()

rag_app = build_rag_graph()

def run_rag_pipeline(query: str) -> Dict[str, Any]:
    """Execute RAG graph pipeline and return formatted payload."""
    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "context_text": "",
        "final_answer": "",
        "confidence_score": 0.0,
        "is_grounded": True
    }
    
    final_state = rag_app.invoke(initial_state)
    
    chunks_output = []
    for doc, score in final_state.get("retrieved_docs", []):
        chunks_output.append({
            "text": doc.page_content,
            "score": round(score, 4),
            "metadata": {
                "page_number": doc.metadata.get("page_number"),
                "source": doc.metadata.get("source", "Ebook-Agentic-AI.pdf"),
                "chunk_id": doc.metadata.get("chunk_id")
            }
        })
        
    return {
        "query": query,
        "final_answer": final_state.get("final_answer", ""),
        "retrieved_context_chunks": chunks_output,
        "confidence_score": final_state.get("confidence_score", 0.0)
    }

if __name__ == "__main__":
    test_result = run_rag_pipeline("What is Agentic AI according to the book?")
    print("Answer:\n", test_result["final_answer"])
    print("Confidence:", test_result["confidence_score"])
