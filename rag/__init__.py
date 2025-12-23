"""
EVA RAG Module
==============
Retrieval-Augmented Generation for EVA AI Assistant.
"""

from .document_loader import (
    DoclingLoader,
    load_document,
    load_pdf,
    load_documents_from_directory,
    # Hybrid loaders (confidence-based routing)
    HybridDocumentLoader,
    load_document_smart,
    load_documents_smart,
    # Load and split
    load_and_split,
)

from .extraction_quality import ExtractionQuality, QualityMetrics
from .unstructured_loader import UnstructuredLoader
from .semantic_splitter import (
    SemanticSplitter,
    semantic_split,
    semantic_split_documents,
)
from .chunk_processor import ChunkProcessor, process_chunks
from .vector_store import VectorStore, create_vector_store
from .retriever import (
    RAGRetriever,
    MathQueryRewriter,
    Reranker,
    RetrievalResult,
    create_retriever,
)
from .rag_pipeline import (
    RAGPipeline,
    RAGResponse,
    create_rag_pipeline,
    ask,
)

__all__ = [
    # Original loaders
    "DoclingLoader",
    "load_document",
    "load_pdf",
    "load_documents_from_directory",
    # Hybrid loaders
    "HybridDocumentLoader",
    "load_document_smart",
    "load_documents_smart",
    # Load and split
    "load_and_split",
    # Quality assessment
    "ExtractionQuality",
    "QualityMetrics",
    # Fast loader
    "UnstructuredLoader",
    # Semantic splitter
    "SemanticSplitter",
    "semantic_split",
    "semantic_split_documents",
    # Chunk processing
    "ChunkProcessor",
    "process_chunks",
    # Vector store
    "VectorStore",
    "create_vector_store",
    # Retriever
    "RAGRetriever",
    "MathQueryRewriter",
    "Reranker",
    "RetrievalResult",
    "create_retriever",
    # RAG Pipeline
    "RAGPipeline",
    "RAGResponse",
    "create_rag_pipeline",
    "ask",
]
