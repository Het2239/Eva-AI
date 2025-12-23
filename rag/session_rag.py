#!/usr/bin/env python3
"""
EVA RAG - Session RAG
=====================
Ephemeral vector store for session-scoped documents.
Documents are deleted when session ends.
"""

import os
import sys
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
_this_dir = Path(__file__).parent
if str(_this_dir.parent) not in sys.path:
    sys.path.insert(0, str(_this_dir.parent))

from langchain_core.documents import Document


class SessionRAG:
    """
    Ephemeral vector store for session-scoped documents.
    
    Features:
        - In-memory ChromaDB collection per session
        - Document summarization for intent detection
        - Automatic cleanup on session end
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
    ):
        """
        Initialize session RAG.
        
        Args:
            session_id: Unique session identifier (auto-generated if None)
            embedding_model: HuggingFace embedding model name
        """
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.embedding_model = embedding_model
        
        # Lazy-loaded components
        self._embeddings = None
        self._vectorstore = None
        self._documents: List[Document] = []
        self.doc_summary: Optional[str] = None
        
        # Temp directory for ChromaDB
        self._temp_dir = None
    
    def _get_embeddings(self):
        """Get or create embeddings."""
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings
    
    def _get_vectorstore(self):
        """Get or create ephemeral ChromaDB vectorstore."""
        if self._vectorstore is None:
            from langchain_chroma import Chroma
            
            # Create temp directory for this session
            self._temp_dir = tempfile.mkdtemp(prefix=f"eva_session_{self.session_id}_")
            
            self._vectorstore = Chroma(
                collection_name=f"session_{self.session_id}",
                embedding_function=self._get_embeddings(),
                persist_directory=self._temp_dir,
            )
        return self._vectorstore
    
    def _generate_summary(self, documents: List[Document]) -> str:
        """Generate a summary of ingested documents."""
        # Combine first parts of documents
        text_parts = []
        total_chars = 0
        max_chars = 2000
        
        for doc in documents:
            content = doc.page_content[:500]
            if total_chars + len(content) > max_chars:
                break
            text_parts.append(content)
            total_chars += len(content)
        
        combined = "\n---\n".join(text_parts)
        
        # Try to use LLM for summarization
        try:
            from models import get_chat_response
            
            prompt = f"""Summarize the following document content in 2-3 sentences. 
Focus on the main topics and key information.

Content:
{combined}

Summary:"""
            summary = get_chat_response(prompt)
            return summary.strip()
        except:
            # Fallback: extract key phrases
            words = combined.split()[:50]
            return " ".join(words) + "..."
    
    def ingest(
        self,
        path: str,
        verbose: bool = True,
    ) -> str:
        """
        Ingest documents into session RAG.
        
        Args:
            path: Path to document or directory
            verbose: Whether to print progress
            
        Returns:
            Document summary
        """
        try:
            from rag.document_loader import load_and_split
        except ImportError:
            from document_loader import load_and_split
        
        try:
            from rag.chunk_processor import ChunkProcessor
        except ImportError:
            from chunk_processor import ChunkProcessor
        
        path = os.path.expanduser(path)
        
        if verbose:
            print(f"📄 Loading to session: {path}")
        
        # Load and split
        chunks = load_and_split(path, chunk_size=500, semantic=False, verbose=verbose)
        
        if not chunks:
            return "No content extracted"
        
        # Process chunks
        processor = ChunkProcessor()
        processed = processor.process(chunks)
        
        if not processed:
            return "No valid chunks after processing"
        
        # Sanitize metadata for ChromaDB (lists → strings)
        sanitized = []
        for doc in processed:
            new_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, list):
                    new_metadata[key] = ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    continue
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    new_metadata[key] = value
                else:
                    new_metadata[key] = str(value)
            sanitized.append(Document(page_content=doc.page_content, metadata=new_metadata))
        
        # Store documents
        self._documents.extend(sanitized)
        
        # Add to vectorstore
        vectorstore = self._get_vectorstore()
        vectorstore.add_documents(sanitized)
        
        if verbose:
            print(f"✓ Added {len(sanitized)} chunks to session")
        
        # Generate and store summary
        self.doc_summary = self._generate_summary(sanitized)
        
        if verbose:
            print(f"📋 Summary: {self.doc_summary[:100]}...")
        
        return self.doc_summary
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        """
        Retrieve relevant documents from session.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of relevant documents
        """
        if not self._documents:
            return []
        
        vectorstore = self._get_vectorstore()
        return vectorstore.similarity_search(query, k=k)
    
    def clear(self) -> None:
        """Clear all session data (end session)."""
        # Clear documents
        self._documents.clear()
        self.doc_summary = None
        
        # Delete vectorstore
        if self._vectorstore is not None:
            del self._vectorstore
            self._vectorstore = None
        
        # Delete temp directory
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except:
                pass
            self._temp_dir = None
        
        print(f"✓ Session {self.session_id} cleared")
    
    @property
    def document_count(self) -> int:
        """Return number of documents in session."""
        return len(self._documents)
    
    @property
    def has_documents(self) -> bool:
        """Check if session has documents."""
        return len(self._documents) > 0
    
    def __del__(self):
        """Cleanup on garbage collection."""
        self.clear()


# ============================================================
# CLI INTERFACE
# ============================================================

if __name__ == "__main__":
    print("SessionRAG Test")
    print("=" * 50)
    
    # Create session
    session = SessionRAG()
    print(f"Session ID: {session.session_id}")
    
    # Test without documents
    print(f"Has documents: {session.has_documents}")
    
    # Test clear
    session.clear()
    print("Test complete")
