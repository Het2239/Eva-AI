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
    HybridDocumentLoader,
    load_document_smart,
    load_documents_smart,
    load_and_split,
)

from .extraction_quality import ExtractionQuality, QualityMetrics
from .unstructured_loader import UnstructuredLoader
from .semantic_splitter import SemanticSplitter, semantic_split, semantic_split_documents
from .chunk_processor import ChunkProcessor, process_chunks
from .vector_store import VectorStore, create_vector_store
from .retriever import RAGRetriever, MathQueryRewriter, Reranker, RetrievalResult, create_retriever
from .rag_pipeline import RAGPipeline, RAGResponse, create_rag_pipeline, ask

# Session-based agent
from .session_rag import SessionRAG
from .conversation_memory import ConversationMemory
from .intent_classifier import needs_rag
from .agent import EVAAgent, AgentResponse, create_agent

__all__ = [
    # Loaders
    "DoclingLoader", "load_document", "load_pdf", "load_documents_from_directory",
    "HybridDocumentLoader", "load_document_smart", "load_documents_smart", "load_and_split",
    # Quality
    "ExtractionQuality", "QualityMetrics", "UnstructuredLoader",
    # Splitters
    "SemanticSplitter", "semantic_split", "semantic_split_documents",
    # Processing
    "ChunkProcessor", "process_chunks",
    # Vector store
    "VectorStore", "create_vector_store",
    # Retriever
    "RAGRetriever", "MathQueryRewriter", "Reranker", "RetrievalResult", "create_retriever",
    # Pipeline
    "RAGPipeline", "RAGResponse", "create_rag_pipeline", "ask",
    # Session Agent
    "SessionRAG", "ConversationMemory", "needs_rag",
    "EVAAgent", "AgentResponse", "create_agent",
]

