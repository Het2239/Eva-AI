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
)

__all__ = [
    "DoclingLoader",
    "load_document",
    "load_pdf",
    "load_documents_from_directory",
]
