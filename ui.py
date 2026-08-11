import streamlit as st
import requests
import os
import importlib
import rag_graph
from rag_graph import run_rag_pipeline
from config import settings

# Page Configuration
st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Dark glassmorphism theme */
    .main {
        background-color: #0e1117;
    }
    .stAppHeader {
        background: rgba(14, 17, 23, 0.8);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .context-box {
        background: #1a1f2c;
        border-left: 4px solid #4a90e2;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
    .badge-page {
        background-color: #2b3a4a;
        color: #64b5f6;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-score {
        background-color: #1b382b;
        color: #81c784;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bot.png", width=70)
    st.title("Agentic AI RAG")
    st.caption("LangGraph + Gemini + HuggingFace Embeddings")
    
    st.divider()
    
    st.subheader("⚙️ System Status")
    st.markdown(f"**Knowledge Base:** [Ebook-Agentic-AI.pdf]({settings.PDF_URL})")
    st.markdown(f"**Embeddings:** `{settings.EMBEDDING_MODEL_NAME}`")
    st.markdown(f"**Vector DB:** `{settings.VECTOR_DB_TYPE.upper()}`")
    st.markdown(f"**LLM:** `{settings.GEMINI_MODEL}`")
    
    st.divider()
    
    st.subheader("💡 Sample Queries")
    sample_queries = [
        "What is Agentic AI according to the ebook?",
        "What are the main components of an Anatomy of an Agentic AI System?",
        "How do Multi-Agent Systems orchestrate decision making?",
        "What is the difference between traditional automation and Agentic AI?",
        "What factors determine an organization's readiness for Agentic AI?",
        "Who wrote or published this Agentic AI eBook?"
    ]
    
    selected_sample = None
    for q in sample_queries:
        if st.button(f"👉 {q}", use_container_width=True, key=f"btn_{hash(q)}"):
            selected_sample = q

# Main Title & Subtitle
st.title("🤖 Agentic AI eBook Chatbot")
st.markdown("Ask questions strictly answered by the **[Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf)** knowledge base.")

st.divider()

# Session state for query
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""

if selected_sample:
    st.session_state["user_query"] = selected_sample

# Input area
user_input = st.text_input("Enter your question:", value=st.session_state["user_query"], placeholder="e.g. What is Agentic AI according to the book?")

if st.button("🚀 Submit Question", type="primary", use_container_width=True) or (user_input and selected_sample):
    if not user_input.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Retrieving context from HuggingFace embeddings & synthesizing answer via Gemini..."):
            try:
                importlib.reload(rag_graph)
                response_data = rag_graph.run_rag_pipeline(user_input)
                
                final_answer = response_data.get("final_answer", "")
                chunks = response_data.get("retrieved_context_chunks", [])
                confidence = response_data.get("confidence_score", 0.0)
                
                # Metrics Row
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    conf_pct = int(confidence * 100)
                    color = "#4CAF50" if conf_pct >= 70 else ("#FF9800" if conf_pct >= 40 else "#F44336")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: {color}">{conf_pct}%</div>
                        <div class="metric-label">Confidence Score</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: #64b5f6">{len(chunks)}</div>
                        <div class="metric-label">Chunks Retrieved</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    is_grounded = "insufficient information" not in final_answer.lower()
                    status_text = "✅ Strictly Grounded in eBook" if is_grounded else "⚠️ Insufficient Context in eBook"
                    status_color = "#4CAF50" if is_grounded else "#FF9800"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: {status_color}; font-size: 1.4rem;">{status_text}</div>
                        <div class="metric-label">Grounding Verification</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Answer Box
                st.subheader("💬 Final Answer")
                st.markdown(f"> {final_answer}")
                
                st.divider()
                
                # Retrieved Chunks Accordion
                st.subheader("📚 Retrieved Context Chunks")
                if not chunks:
                    st.info("No matching context chunks found in the vector store.")
                else:
                    for i, chunk in enumerate(chunks):
                        score = chunk.get("score", 0.0)
                        page = chunk.get("metadata", {}).get("page_number", "N/A")
                        chunk_id = chunk.get("metadata", {}).get("chunk_id", f"chunk_{i+1}")
                        text = chunk.get("text", "")
                        
                        with st.expander(f"Chunk {i+1} — Page {page} (Relevance Score: {score:.2f})"):
                            st.markdown(f"<span class='badge-page'>Page {page}</span> <span class='badge-score'>Similarity: {score:.4f}</span>", unsafe_allow_html=True)
                            st.markdown(f"```text\n{text}\n```")
                            
            except Exception as e:
                st.error(f"Error executing RAG pipeline: {str(e)}")
