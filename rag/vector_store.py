#!/usr/bin/env python3
"""
EVA RAG - Vector Store
======================
Hybrid vector storage using ChromaDB (dense) + BM25 (sparse).
"""

import os
import pickle
from typing import List, Optional, Tuple
from pathlib import Path

from langchain_core.documents import Document


class VectorStore:
    """
    Hybrid vector store combining ChromaDB (dense) and BM25 (sparse).
    
    Features:
        - Dense retrieval via ChromaDB with HuggingFace embeddings
        - Sparse retrieval via BM25
        - Persistence to disk
        - Hybrid search with score fusion
    """
    
    def __init__(
        self,
        collection_name: str = "eva_rag",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name for the ChromaDB collection
            embedding_model: HuggingFace embedding model name
            persist_directory: Directory to persist data (None for in-memory)
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        
        # Lazy-load components
        self._embeddings = None
        self._chroma_store = None
        self._bm25_retriever = None
        self._documents: List[Document] = []
    
    def _get_embeddings(self):
        """Get or create HuggingFace embeddings."""
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings
    
    def _get_chroma_store(self):
        """Get or create ChromaDB store."""
        if self._chroma_store is None:
            import chromadb
            from langchain_chroma import Chroma
            
            embeddings = self._get_embeddings()
            
            if self.persist_directory:
                # Persistent storage
                self._chroma_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=embeddings,
                    persist_directory=self.persist_directory,
                )
            else:
                # In-memory storage
                self._chroma_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=embeddings,
                )
        return self._chroma_store
    
    def _build_bm25_retriever(self):
        """Build BM25 retriever from documents."""
        if not self._documents:
            return None
        
        from langchain_community.retrievers import BM25Retriever
        
        self._bm25_retriever = BM25Retriever.from_documents(
            self._documents,
            k=10,  # Will be overridden in search
        )
        return self._bm25_retriever
    
    def _sanitize_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Sanitize metadata for ChromaDB compatibility.
        
        ChromaDB only accepts str, int, float, bool, or None values.
        Lists are converted to comma-separated strings.
        """
        sanitized = []
        for doc in documents:
            new_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, list):
                    # Convert list to comma-separated string
                    new_metadata[key] = ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    # Skip nested dicts
                    continue
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    new_metadata[key] = value
                else:
                    # Convert other types to string
                    new_metadata[key] = str(value)
            
            sanitized.append(Document(
                page_content=doc.page_content,
                metadata=new_metadata,
            ))
        return sanitized
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to both dense and sparse stores.
        
        Args:
            documents: List of Document objects to add
        """
        if not documents:
            return
        
        # Store original documents for BM25
        self._documents.extend(documents)
        
        # Sanitize metadata for ChromaDB
        sanitized_docs = self._sanitize_metadata(documents)
        
        # Add to ChromaDB
        chroma = self._get_chroma_store()
        chroma.add_documents(sanitized_docs)
        
        # Rebuild BM25 retriever
        self._build_bm25_retriever()
    
    def dense_search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Dense retrieval using ChromaDB embeddings.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        chroma = self._get_chroma_store()
        results = chroma.similarity_search_with_score(query, k=k)
        return results
    
    def sparse_search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        """
        Sparse retrieval using BM25.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of Documents
        """
        if self._bm25_retriever is None:
            self._build_bm25_retriever()
        
        if self._bm25_retriever is None:
            return []
        
        self._bm25_retriever.k = k
        return self._bm25_retriever.invoke(query)
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        dense_weight: float = 0.5,
    ) -> List[Document]:
        """
        Hybrid search combining dense and sparse retrieval.
        
        Uses Reciprocal Rank Fusion (RRF) for score combination.
        
        Args:
            query: Search query
            k: Number of final results
            dense_weight: Weight for dense results (0-1)
            
        Returns:
            List of Documents
        """
        # Get more candidates for fusion
        dense_k = min(k * 3, 20)
        sparse_k = min(k * 3, 20)
        
        # Dense retrieval
        dense_results = self.dense_search(query, k=dense_k)
        
        # Sparse retrieval
        sparse_results = self.sparse_search(query, k=sparse_k)
        
        # Reciprocal Rank Fusion
        rrf_scores = {}
        rrf_constant = 60  # Standard RRF constant
        
        # Score dense results
        for rank, (doc, _) in enumerate(dense_results):
            doc_id = doc.page_content[:100]  # Use content prefix as ID
            rrf_score = dense_weight / (rrf_constant + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
        
        # Score sparse results
        sparse_weight = 1 - dense_weight
        for rank, doc in enumerate(sparse_results):
            doc_id = doc.page_content[:100]
            rrf_score = sparse_weight / (rrf_constant + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
        
        # Combine all documents
        all_docs = {doc.page_content[:100]: doc for doc, _ in dense_results}
        all_docs.update({doc.page_content[:100]: doc for doc in sparse_results})
        
        # Sort by RRF score and return top k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])[:k]
        
        results = []
        for doc_id in sorted_ids:
            if doc_id in all_docs:
                doc = all_docs[doc_id]
                # Add score to metadata
                doc.metadata['rrf_score'] = round(rrf_scores[doc_id], 4)
                results.append(doc)
        
        return results
    
    def save(self, path: str) -> None:
        """
        Save store to disk.
        
        Args:
            path: Directory path to save data
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save documents for BM25
        docs_path = path / "documents.pkl"
        with open(docs_path, 'wb') as f:
            pickle.dump(self._documents, f)
        
        # ChromaDB persists automatically if persist_directory is set
        if self.persist_directory and self._chroma_store:
            pass  # Already persisted
        
        # Save config
        config = {
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model,
            'num_documents': len(self._documents),
        }
        config_path = path / "config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"✓ Saved {len(self._documents)} documents to {path}")
    
    def load(self, path: str) -> None:
        """
        Load store from disk.
        
        Args:
            path: Directory path to load from
        """
        path = Path(path)
        
        # Load documents
        docs_path = path / "documents.pkl"
        if docs_path.exists():
            with open(docs_path, 'rb') as f:
                self._documents = pickle.load(f)
        
        # Rebuild BM25
        self._build_bm25_retriever()
        
        # ChromaDB loads automatically if persist_directory matches
        
        print(f"✓ Loaded {len(self._documents)} documents from {path}")
    
    @property
    def document_count(self) -> int:
        """Return number of indexed documents."""
        return len(self._documents)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_vector_store(
    documents: List[Document],
    collection_name: str = "eva_rag",
    persist_path: Optional[str] = None,
) -> VectorStore:
    """
    Create and populate a vector store.
    
    Args:
        documents: Documents to index
        collection_name: Collection name
        persist_path: Path for persistence
        
    Returns:
        Populated VectorStore
    """
    store = VectorStore(
        collection_name=collection_name,
        persist_directory=persist_path,
    )
    store.add_documents(documents)
    return store


# ============================================================
# CLI INTERFACE
# ============================================================

if __name__ == "__main__":
    import sys
    
    # Test with sample documents
    test_docs = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"source": "ml.txt"}),
        Document(page_content="Deep learning uses neural networks with many layers.", metadata={"source": "dl.txt"}),
        Document(page_content="Natural language processing handles text and speech.", metadata={"source": "nlp.txt"}),
        Document(page_content="Computer vision processes images and video.", metadata={"source": "cv.txt"}),
        Document(page_content="Reinforcement learning learns from rewards and penalties.", metadata={"source": "rl.txt"}),
    ]
    
    print("VectorStore Test")
    print("=" * 50)
    
    store = VectorStore(collection_name="test_collection")
    print(f"Adding {len(test_docs)} documents...")
    store.add_documents(test_docs)
    print(f"Document count: {store.document_count}")
    
    query = "What is deep learning?"
    print(f"\nQuery: '{query}'")
    
    print("\n--- Dense Search ---")
    dense_results = store.dense_search(query, k=3)
    for doc, score in dense_results:
        print(f"  [{score:.3f}] {doc.page_content[:60]}...")
    
    print("\n--- Sparse Search ---")
    sparse_results = store.sparse_search(query, k=3)
    for doc in sparse_results:
        print(f"  {doc.page_content[:60]}...")
    
    print("\n--- Hybrid Search ---")
    hybrid_results = store.hybrid_search(query, k=3)
    for doc in hybrid_results:
        print(f"  [{doc.metadata.get('rrf_score', 0):.4f}] {doc.page_content[:60]}...")
