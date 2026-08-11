import streamlit as st
import importlib
import rag_graph
from config import settings

st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .badge-success {
        background-color: rgba(76, 175, 80, 0.15);
        color: #4CAF50;
        border: 1px solid #4CAF50;
    }
    .badge-warning {
        background-color: rgba(255, 152, 0, 0.15);
        color: #FF9800;
        border: 1px solid #FF9800;
    }
    .page-tag {
        background-color: #2b3a4a;
        color: #64b5f6;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .score-tag {
        background-color: #1b382b;
        color: #81c784;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Welcome! I can answer your questions about the [Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf). Feel free to ask about agent architectures, multi-agent systems, or readiness frameworks.",
            "confidence": 1.0,
            "chunks": []
        }
    ]

with st.sidebar:
    st.title("Agentic AI RAG")
    st.caption("LangGraph + Gemini + HuggingFace Embeddings")
    
    st.divider()
    
    st.subheader("System Info")
    st.markdown(f"**Document:** [Ebook-Agentic-AI.pdf]({settings.PDF_URL})")
    st.markdown(f"**Embeddings:** `{settings.EMBEDDING_MODEL_NAME}`")
    st.markdown(f"**Vector Store:** `{settings.VECTOR_DB_TYPE.upper()}`")
    st.markdown(f"**LLM Model:** `{settings.GEMINI_MODEL}`")
    st.markdown(f"**Retrieval Top-K:** `{settings.TOP_K_RETRIEVAL}`")
    
    st.divider()
    
    st.subheader("Sample Questions")
    sample_queries = [
        "What is Agentic AI according to the ebook?",
        "What are the types of agents based on functional versatility?",
        "What are the main components of an Anatomy of an Agentic AI System?",
        "How do Multi-Agent Systems orchestrate decision making?",
        "What is the difference between traditional automation and Agentic AI?",
        "What factors determine an organization's readiness for Agentic AI?"
    ]
    
    selected_sample = None
    for q in sample_queries:
        if st.button(q, use_container_width=True, key=f"btn_{hash(q)}"):
            selected_sample = q

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Chat reset. How can I help you with the Agentic AI eBook?",
                "confidence": 1.0,
                "chunks": []
            }
        ]
        st.rerun()

st.title("Agentic AI Assistant")
st.caption("Retrieval-Augmented Generation grounded in `Ebook-Agentic-AI.pdf`")

st.divider()

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "confidence" in msg and msg.get("chunks"):
            confidence = msg.get("confidence", 0.0)
            conf_pct = int(confidence * 100)
            is_grounded = "insufficient information" not in msg["content"].lower()
            
            badge_style = "badge-success" if is_grounded else "badge-warning"
            status_text = f"Grounded ({conf_pct}% Confidence)" if is_grounded else f"Insufficient Context ({conf_pct}%)"
            
            st.markdown(f"<div class='badge {badge_style}'>{status_text}</div>", unsafe_allow_html=True)
            
            chunks = msg.get("chunks", [])
            with st.expander(f"Retrieved Context ({len(chunks)} Chunks)"):
                for i, chunk in enumerate(chunks):
                    score = chunk.get("score", 0.0)
                    page = chunk.get("metadata", {}).get("page_number", "N/A")
                    text = chunk.get("text", "")
                    st.markdown(f"<span class='page-tag'>Page {page}</span> <span class='score-tag'>Score: {score:.4f}</span>", unsafe_allow_html=True)
                    st.markdown(f"```text\n{text}\n```")

user_prompt = st.chat_input("Ask a question about the eBook...")

if selected_sample:
    user_prompt = selected_sample

if user_prompt:
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching document context..."):
            try:
                importlib.reload(rag_graph)
                response_data = rag_graph.run_rag_pipeline(user_prompt)
                
                final_answer = response_data.get("final_answer", "")
                confidence = response_data.get("confidence_score", 0.0)
                chunks = response_data.get("retrieved_context_chunks", [])
                
                st.markdown(final_answer)
                
                conf_pct = int(confidence * 100)
                is_grounded = "insufficient information" not in final_answer.lower()
                badge_style = "badge-success" if is_grounded else "badge-warning"
                status_text = f"Grounded ({conf_pct}% Confidence)" if is_grounded else f"Insufficient Context ({conf_pct}%)"
                
                st.markdown(f"<div class='badge {badge_style}'>{status_text}</div>", unsafe_allow_html=True)
                
                if chunks:
                    with st.expander(f"Retrieved Context ({len(chunks)} Chunks)"):
                        for i, chunk in enumerate(chunks):
                            score = chunk.get("score", 0.0)
                            page = chunk.get("metadata", {}).get("page_number", "N/A")
                            text = chunk.get("text", "")
                            st.markdown(f"<span class='page-tag'>Page {page}</span> <span class='score-tag'>Score: {score:.4f}</span>", unsafe_allow_html=True)
                            st.markdown(f"```text\n{text}\n```")
                
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": final_answer,
                    "confidence": confidence,
                    "chunks": chunks
                })
                
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                st.error(error_msg)
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": error_msg,
                    "confidence": 0.0,
                    "chunks": []
                })
