import os
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config import settings
from embeddings import get_embedding_model

class VectorStoreManager:
    """
    Unified Vector Store Manager supporting Pinecone and ChromaDB.
    Auto-detects Pinecone API Key or defaults to local ChromaDB.
    """
    def __init__(self):
        self.embeddings = get_embedding_model()
        self.db_type = self._determine_db_type()
        self.vector_store = None
        self._init_vector_store()

    def _determine_db_type(self) -> str:
        if settings.VECTOR_DB_TYPE.lower() == "pinecone":
            return "pinecone"
        elif settings.VECTOR_DB_TYPE.lower() == "chroma":
            return "chroma"
        else:
            # Auto-detect: if PINECONE_API_KEY is provided and non-empty, use pinecone, else chroma
            api_key = settings.PINECONE_API_KEY or os.getenv("PINECONE_API_KEY", "")
            if api_key and api_key.strip():
                return "pinecone"
            return "chroma"

    def _init_vector_store(self):
        if self.db_type == "pinecone":
            try:
                from pinecone import Pinecone, ServerlessSpec
                from langchain_pinecone import PineconeVectorStore

                api_key = settings.PINECONE_API_KEY or os.getenv("PINECONE_API_KEY", "")
                pc = Pinecone(api_key=api_key)
                index_name = settings.PINECONE_INDEX_NAME

                # Create index if it doesn't exist
                existing_indexes = [idx.name for idx in pc.list_indexes()]
                if index_name not in existing_indexes:
                    pc.create_index(
                        name=index_name,
                        dimension=settings.EMBEDDING_DIMENSION,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENVIRONMENT)
                    )

                self.vector_store = PineconeVectorStore(
                    index_name=index_name,
                    embedding=self.embeddings,
                    pinecone_api_key=api_key
                )
                print(f"[VectorStoreManager] Initialized Pinecone Vector Store (Index: {index_name})")
            except Exception as e:
                print(f"[VectorStoreManager] Pinecone initialization failed: {e}. Falling back to local ChromaDB.")
                self.db_type = "chroma"
                self._init_chroma()
        else:
            self._init_chroma()

    def _init_chroma(self):
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        self.vector_store = Chroma(
            collection_name="agentic_ai_ebook",
            embedding_function=self.embeddings,
            persist_directory=str(settings.CHROMA_PERSIST_DIR)
        )
        print(f"[VectorStoreManager] Initialized ChromaDB at {settings.CHROMA_PERSIST_DIR}")

    def add_documents(self, documents: List[Document]) -> int:
        """Add documents/chunks to the vector store."""
        if not documents:
            return 0
        self.vector_store.add_documents(documents)
        print(f"[VectorStoreManager] Successfully indexed {len(documents)} document chunks into {self.db_type.upper()}")
        return len(documents)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Performs similarity search returning documents with score.
        Scores are normalized to [0.0, 1.0] where 1.0 is highest similarity.
        Auto-triggers ingestion if vector store is empty, or falls back to ChromaDB.
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"[VectorStoreManager] Search on {self.db_type} failed ({e}). Falling back to local ChromaDB.")
            results = []

        # If primary DB returned 0 results, attempt local ChromaDB fallback
        if not results and self.db_type == "pinecone":
            print("[VectorStoreManager] Pinecone returned 0 results. Checking local ChromaDB fallback...")
            try:
                self._init_chroma()
                results = self.vector_store.similarity_search_with_score(query, k=k)
                if results:
                    self.db_type = "chroma"
                    print(f"[VectorStoreManager] Successfully retrieved {len(results)} chunks from local ChromaDB fallback!")
            except Exception as ex:
                print(f"[VectorStoreManager] Fallback search error: {ex}")

        # If still 0 results, trigger auto-ingestion to populate DB
        if not results:
            print("[VectorStoreManager] Vector Store appears empty. Triggering automatic PDF ingestion...")
            from ingest import run_ingestion
            run_ingestion()
            try:
                results = self.vector_store.similarity_search_with_score(query, k=k)
            except Exception as ex:
                print(f"[VectorStoreManager] Post-ingestion search error: {ex}")

        normalized_results = []
        for doc, raw_score in results:
            if self.db_type == "chroma":
                # Convert Chroma distance (lower is better) to similarity score (higher is better)
                similarity = max(0.0, min(1.0, 1.0 - (raw_score / 2.0)))
            else:
                similarity = max(0.0, min(1.0, float(raw_score)))
            
            normalized_results.append((doc, round(similarity, 4)))
            
        return sorted(normalized_results, key=lambda x: x[1], reverse=True)

    def clear(self):
        """Clears existing collection data if needed."""
        if self.db_type == "chroma" and hasattr(self.vector_store, "_collection"):
            try:
                self.vector_store.delete_collection()
                self._init_chroma()
            except Exception as e:
                print(f"[VectorStoreManager] Error clearing Chroma collection: {e}")

_vector_store_instance = None

def get_vector_store_manager() -> VectorStoreManager:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreManager()
    return _vector_store_instance
