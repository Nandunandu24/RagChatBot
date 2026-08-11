from langchain_huggingface import HuggingFaceEmbeddings
from config import settings

def get_embedding_model():
    """
    Returns HuggingFaceEmbeddings using sentence-transformers model.
    Default: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional).
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embedding_model
