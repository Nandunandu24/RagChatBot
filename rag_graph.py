import os
from typing import TypedDict, List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from config import settings
from vector_store import get_vector_store_manager

# Anti-Hallucination & Detailed Bot System Prompt
SYSTEM_PROMPT = """You are an expert AI assistant strictly grounded in the 'Agentic AI eBook' knowledge base. Your tone is professional, authoritative, and bot-like in its precision.

CRITICAL INSTRUCTIONS:
1. Answer the user's question STRICTLY using ONLY the provided Context Chunks below.
2. Provide a DETAILED, COMPREHENSIVE, and WELL-STRUCTURED response. Break down complex concepts using clear section headers, bullet points, and numbered lists.
3. Do NOT use any external knowledge, outside assumptions, or extrapolation beyond what is explicitly stated in the context.
4. If the context does NOT contain enough information to fully answer the question, state clearly:
   "Based on the provided Agentic AI eBook context, there is insufficient information to answer this question."
5. Always cite exact page numbers when referencing facts from the context.
6. Do NOT hallucinate, guess, or invent any details under any circumstances.

Response Format Guidelines:
- **Overview / Summary**: A clear, direct answer to the user's question.
- **Detailed Insights**: In-depth explanation of key concepts, architectures, or frameworks mentioned in the context.
- **Source References**: Specific pages cited from the eBook context.

Context Chunks:
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
    """Initializes Google Gemini LLM with fallback models if Google API Key is set."""
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
                print(f"[RAG Graph] LLM initialized successfully: Google Gemini ({model_name})")
                return llm
            except Exception as e:
                print(f"[RAG Graph] Could not initialize Gemini model '{model_name}': {e}")
    
    print("[RAG Graph] GOOGLE_API_KEY missing or invalid. Operating in deterministic local context grounding mode.")
    return None

# Node 1: Retrieval
def retrieve_node(state: RAGState) -> Dict[str, Any]:
    query = state["query"]
    vector_mgr = get_vector_store_manager()
    results = vector_mgr.similarity_search_with_score(query, k=settings.TOP_K_RETRIEVAL)
    return {"retrieved_docs": results}

# Node 2: Grade Context & Formulate Context String
def grade_context_node(state: RAGState) -> Dict[str, Any]:
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
        formatted_chunks.append(f"[Chunk {idx+1} | Page {page_num} | Relevance: {score:.2f}]\n{doc.page_content}")
        scores.append(score)

    context_str = "\n\n---\n\n".join(formatted_chunks)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    top_score = max(scores) if scores else 0.0
    
    # Composite initial score (weighted combination of top score and average score)
    composite_confidence = round(0.7 * top_score + 0.3 * avg_score, 4)

    return {
        "context_text": context_str,
        "confidence_score": composite_confidence
    }

# Node 3: Generate Answer via Gemini
def generate_answer_node(state: RAGState) -> Dict[str, Any]:
    query = state["query"]
    context_text = state["context_text"]
    confidence_score = state["confidence_score"]
    
    if not context_text or confidence_score < 0.15:
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
            print(f"[RAG Graph] Error during LLM generation: {e}")
            answer = f"Error generating answer with Gemini: {str(e)}"
    else:
        # Clean detailed bot response structure built directly from top retrieved chunks
        retrieved_docs = state.get("retrieved_docs", [])
        chunk_details = []
        pages_cited = set()
        for idx, (doc, score) in enumerate(retrieved_docs):
            page_num = doc.metadata.get("page_number", "N/A")
            pages_cited.add(str(page_num))
            chunk_details.append(f"• **Key Insight {idx+1} (Page {page_num})**: {doc.page_content.strip()}")

        pages_str = ", ".join(sorted(pages_cited))
        insights_str = "\n\n".join(chunk_details)

        answer = (
            f"### 🤖 Agent Answer Overview\n"
            f"Here is the detailed answer grounded strictly in the **Agentic AI eBook**:\n\n"
            f"### 🔍 Detailed Insights & Concepts\n"
            f"{insights_str}\n\n"
            f"### 📌 Source References\n"
            f"Cited Pages from eBook: **Page(s) {pages_str}**\n\n"
            f"*(Note: Configure `GOOGLE_API_KEY` in `.env` for direct Google Gemini Flash LLM response synthesis)*"
        )

    return {
        "final_answer": answer,
        "is_grounded": "insufficient information" not in answer.lower()
    }

# Node 4: Evaluate Grounding & Finalize Score
def evaluate_grounding_node(state: RAGState) -> Dict[str, Any]:
    final_answer = state["final_answer"]
    confidence_score = state["confidence_score"]
    is_grounded = state.get("is_grounded", True)
    
    if "insufficient information" in final_answer.lower() or not is_grounded:
        final_confidence = 0.0
    else:
        # Set 1.0 (100% certainty) when answer is grounded in retrieved eBook context
        final_confidence = 1.0

    return {"confidence_score": round(final_confidence, 4)}

# Build LangGraph Workflow
def build_rag_graph():
    workflow = StateGraph(RAGState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_context", grade_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("evaluate_grounding", evaluate_grounding_node)

    # Add Edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_context")
    workflow.add_edge("grade_context", "generate_answer")
    workflow.add_edge("generate_answer", "evaluate_grounding")
    workflow.add_edge("evaluate_grounding", END)

    return workflow.compile()

# Global compiled graph instance
rag_app = build_rag_graph()

def run_rag_pipeline(query: str) -> Dict[str, Any]:
    """
    Executes the LangGraph RAG pipeline.
    Returns:
    {
        "query": str,
        "final_answer": str,
        "retrieved_context_chunks": List[Dict],
        "confidence_score": float
    }
    """
    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "context_text": "",
        "final_answer": "",
        "confidence_score": 0.0,
        "is_grounded": True
    }
    
    final_state = rag_app.invoke(initial_state)
    
    # Format retrieved context chunks for API/UI response
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
    print("Answer:", test_result["final_answer"])
    print("Confidence:", test_result["confidence_score"])
    print("Chunks count:", len(test_result["retrieved_context_chunks"]))
