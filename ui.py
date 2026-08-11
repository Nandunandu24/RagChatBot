import streamlit as st
import importlib
import rag_graph
from config import settings

# Page Configuration
st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Dark Glassmorphism Design
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stAppHeader {
        background: rgba(14, 17, 23, 0.8);
    }
    .chat-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .badge-grounded {
        background-color: rgba(76, 175, 80, 0.15);
        color: #4CAF50;
        border: 1px solid #4CAF50;
    }
    .badge-insufficient {
        background-color: rgba(255, 152, 0, 0.15);
        color: #FF9800;
        border: 1px solid #FF9800;
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
    .stChatMessage {
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I am your **Agentic AI Assistant**, strictly grounded in the **[Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf)**. \n\nAsk me anything about Agentic AI architectures, Multi-Agent Systems, or organizational readiness!",
            "confidence": 1.0,
            "chunks": []
        }
    ]

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
    st.markdown(f"**Top-K Chunks:** `{settings.TOP_K_RETRIEVAL}`")
    
    st.divider()
    
    st.subheader("💡 Sample Questions")
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
        if st.button(f"👉 {q}", use_container_width=True, key=f"btn_{hash(q)}"):
            selected_sample = q

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Chat history cleared! How can I assist you with the Agentic AI eBook?",
                "confidence": 1.0,
                "chunks": []
            }
        ]
        st.rerun()

# Main Interface Title
st.title("💬 Agentic AI Interactive Chat")
st.caption("A conversational RAG assistant providing answers strictly grounded in `Ebook-Agentic-AI.pdf`.")

st.divider()

# Display Chat History
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display metadata for assistant messages
        if msg["role"] == "assistant" and "confidence" in msg and msg.get("chunks"):
            confidence = msg.get("confidence", 0.0)
            conf_pct = int(confidence * 100)
            is_grounded = "insufficient information" not in msg["content"].lower()
            
            badge_class = "badge-grounded" if is_grounded else "badge-insufficient"
            status_label = f"✅ Grounded ({conf_pct}% Confidence)" if is_grounded else f"⚠️ Insufficient Context ({conf_pct}%)"
            
            st.markdown(f"<div class='chat-badge {badge_class}'>{status_label}</div>", unsafe_allow_html=True)
            
            chunks = msg.get("chunks", [])
            with st.expander(f"📚 View Retrieved Context ({len(chunks)} Chunks)"):
                for i, chunk in enumerate(chunks):
                    score = chunk.get("score", 0.0)
                    page = chunk.get("metadata", {}).get("page_number", "N/A")
                    text = chunk.get("text", "")
                    st.markdown(f"<span class='badge-page'>Page {page}</span> <span class='badge-score'>Similarity: {score:.4f}</span>", unsafe_allow_html=True)
                    st.markdown(f"```text\n{text}\n```")

# Determine prompt input (chat_input or sidebar sample button)
user_prompt = st.chat_input("Ask a question about Agentic AI...")

if selected_sample:
    user_prompt = selected_sample

# Handle New User Message
if user_prompt:
    # 1. Render User Message
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Render Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing eBook context & generating grounded answer..."):
            try:
                importlib.reload(rag_graph)
                response_data = rag_graph.run_rag_pipeline(user_prompt)
                
                final_answer = response_data.get("final_answer", "")
                confidence = response_data.get("confidence_score", 0.0)
                chunks = response_data.get("retrieved_context_chunks", [])
                
                st.markdown(final_answer)
                
                conf_pct = int(confidence * 100)
                is_grounded = "insufficient information" not in final_answer.lower()
                badge_class = "badge-grounded" if is_grounded else "badge-insufficient"
                status_label = f"✅ Grounded ({conf_pct}% Confidence)" if is_grounded else f"⚠️ Insufficient Context ({conf_pct}%)"
                
                st.markdown(f"<div class='chat-badge {badge_class}'>{status_label}</div>", unsafe_allow_html=True)
                
                if chunks:
                    with st.expander(f"📚 View Retrieved Context ({len(chunks)} Chunks)"):
                        for i, chunk in enumerate(chunks):
                            score = chunk.get("score", 0.0)
                            page = chunk.get("metadata", {}).get("page_number", "N/A")
                            text = chunk.get("text", "")
                            st.markdown(f"<span class='badge-page'>Page {page}</span> <span class='badge-score'>Similarity: {score:.4f}</span>", unsafe_allow_html=True)
                            st.markdown(f"```text\n{text}\n```")
                
                # Append Assistant Message to History
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
