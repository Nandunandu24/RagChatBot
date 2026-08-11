import os
import urllib.request
from pathlib import Path
from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from vector_store import get_vector_store_manager

def download_pdf(url: str, save_path: Path) -> Path:
    """Downloads the PDF from URL if it does not already exist locally."""
    if save_path.exists() and save_path.stat().st_size > 0:
        print(f"[Ingest] PDF already exists at {save_path}")
        return save_path

    print(f"[Ingest] Downloading PDF from {url} ...")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
        out_file.write(response.read())

    print(f"[Ingest] Downloaded PDF ({save_path.stat().st_size / 1024:.1f} KB) to {save_path}")
    return save_path

def load_pdf_pages(pdf_path: Path) -> List[Document]:
    """Extracts text page by page from the PDF file with page metadata."""
    reader = PdfReader(pdf_path)
    documents = []
    
    print(f"[Ingest] Extracting text from {len(reader.pages)} pages...")
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            doc = Document(
                page_content=text,
                metadata={
                    "page_number": idx + 1,
                    "source": pdf_path.name,
                    "total_pages": len(reader.pages)
                }
            )
            documents.append(doc)
            
    print(f"[Ingest] Extracted {len(documents)} non-empty pages.")
    return documents

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Splits document pages into smaller contextual chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Assign unique chunk_id to each chunk
    for i, chunk in enumerate(chunks):
        page_num = chunk.metadata.get("page_number", "N/A")
        chunk.metadata["chunk_id"] = f"page_{page_num}_chunk_{i+1}"

    print(f"[Ingest] Created {len(chunks)} text chunks (chunk_size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})")
    return chunks

def run_ingestion(force_reingest: bool = False) -> int:
    """Runs the full ingestion pipeline: download -> extract -> chunk -> embed -> index."""
    pdf_path = download_pdf(settings.PDF_URL, settings.PDF_PATH)
    
    vector_mgr = get_vector_store_manager()
    
    if force_reingest:
        print("[Ingest] Force re-ingestion requested. Clearing existing vector store...")
        vector_mgr.clear()

    documents = load_pdf_pages(pdf_path)
    chunks = chunk_documents(documents)
    
    print("[Ingest] Generating HuggingFace embeddings and indexing chunks into Vector DB...")
    total_indexed = vector_mgr.add_documents(chunks)
    print(f"[Ingest] Ingestion completed! Total chunks indexed: {total_indexed}")
    return total_indexed

if __name__ == "__main__":
    run_ingestion()
