#!/usr/bin/env python3
"""
EVA RAG - Semantic Text Splitter
================================
Split documents at semantic boundaries using embeddings.
Uses LangChain's experimental SemanticChunker.
"""

import os
from typing import List, Optional, Literal

from langchain_core.documents import Document


# Breakpoint types for semantic chunking
BreakpointType = Literal["percentile", "standard_deviation", "interquartile", "gradient"]


class SemanticSplitter:
    """
    Split documents at semantic boundaries using embeddings.
    
    Uses LangChain's SemanticChunker which:
    1. Splits text into sentences
    2. Computes embeddings for each sentence
    3. Calculates cosine distances between adjacent sentences
    4. Splits where distance exceeds threshold (semantic boundary)
    
    Args:
        embedding_model: Model name for sentence-transformers
        breakpoint_type: How to detect breakpoints
            - "percentile": Split at Nth percentile of distances (default)
            - "standard_deviation": Split when distance > mean + N*std
            - "interquartile": Split when distance > Q3 + 1.5*IQR
            - "gradient": Split at steepest gradient changes
        breakpoint_threshold: Threshold value (percentile=95, std=3.0)
        min_chunk_size: Minimum characters per chunk (merges small chunks)
    """
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        breakpoint_type: BreakpointType = "percentile",
        breakpoint_threshold: float = 95,
        min_chunk_size: int = 100,
    ):
        """Initialize the semantic splitter."""
        self.embedding_model = embedding_model
        self.breakpoint_type = breakpoint_type
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size
        
        # Lazy-load the chunker and embeddings
        self._chunker = None
        self._embeddings = None
    
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
    
    def _get_chunker(self):
        """Get or create the SemanticChunker."""
        if self._chunker is None:
            from langchain_experimental.text_splitter import SemanticChunker
            
            embeddings = self._get_embeddings()
            
            self._chunker = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=self.breakpoint_type,
                breakpoint_threshold_amount=self.breakpoint_threshold,
            )
        return self._chunker
    
    def split(self, text: str) -> List[str]:
        """
        Split text into semantic chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        chunker = self._get_chunker()
        
        # Create documents and split
        docs = chunker.create_documents([text])
        
        # Extract text and merge small chunks
        chunks = []
        current_chunk = ""
        
        for doc in docs:
            chunk_text = doc.page_content.strip()
            
            if len(current_chunk) + len(chunk_text) < self.min_chunk_size:
                # Merge with current chunk
                current_chunk = (current_chunk + " " + chunk_text).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = chunk_text
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split LangChain Documents, preserving metadata.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of split Document objects with preserved metadata
        """
        if not documents:
            return []
        
        chunker = self._get_chunker()
        all_chunks = []
        
        for doc in documents:
            if not doc.page_content.strip():
                continue
            
            # Split the document text
            split_docs = chunker.create_documents([doc.page_content])
            
            # Merge small chunks and preserve metadata
            current_chunk = ""
            chunk_idx = 0
            
            for split_doc in split_docs:
                chunk_text = split_doc.page_content.strip()
                
                if len(current_chunk) + len(chunk_text) < self.min_chunk_size:
                    current_chunk = (current_chunk + " " + chunk_text).strip()
                else:
                    if current_chunk:
                        # Create document with metadata
                        chunk_metadata = doc.metadata.copy()
                        chunk_metadata["chunk_index"] = chunk_idx
                        chunk_metadata["chunk_type"] = "semantic"
                        chunk_metadata["original_length"] = len(doc.page_content)
                        
                        all_chunks.append(Document(
                            page_content=current_chunk,
                            metadata=chunk_metadata,
                        ))
                        chunk_idx += 1
                    
                    current_chunk = chunk_text
            
            # Last chunk
            if current_chunk:
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_index"] = chunk_idx
                chunk_metadata["chunk_type"] = "semantic"
                chunk_metadata["original_length"] = len(doc.page_content)
                
                all_chunks.append(Document(
                    page_content=current_chunk,
                    metadata=chunk_metadata,
                ))
        
        return all_chunks


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def semantic_split(
    text: str,
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    breakpoint_threshold: float = 95,
    min_chunk_size: int = 100,
) -> List[str]:
    """
    Split text into semantic chunks.
    
    Args:
        text: Text to split
        embedding_model: Embedding model name
        breakpoint_threshold: Percentile for breakpoint detection
        min_chunk_size: Minimum chunk size in characters
        
    Returns:
        List of text chunks
    """
    splitter = SemanticSplitter(
        embedding_model=embedding_model,
        breakpoint_threshold=breakpoint_threshold,
        min_chunk_size=min_chunk_size,
    )
    return splitter.split(text)


def semantic_split_documents(
    documents: List[Document],
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    breakpoint_threshold: float = 95,
    min_chunk_size: int = 100,
) -> List[Document]:
    """
    Split documents into semantic chunks.
    
    Args:
        documents: List of Document objects
        embedding_model: Embedding model name
        breakpoint_threshold: Percentile for breakpoint detection
        min_chunk_size: Minimum chunk size in characters
        
    Returns:
        List of split Document objects
    """
    splitter = SemanticSplitter(
        embedding_model=embedding_model,
        breakpoint_threshold=breakpoint_threshold,
        min_chunk_size=min_chunk_size,
    )
    return splitter.split_documents(documents)


# ============================================================
# CLI INTERFACE
# ============================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Semantic text splitter using embeddings",
    )
    parser.add_argument("text_file", nargs="?", help="Text file to split")
    parser.add_argument(
        "--threshold",
        type=float,
        default=95,
        help="Breakpoint threshold percentile (default: 95)"
    )
    parser.add_argument(
        "--min-chunk",
        type=int,
        default=100,
        help="Minimum chunk size in characters (default: 100)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run with test text"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_text = """
        Introduction to Machine Learning
        
        Machine learning is a subset of artificial intelligence that focuses on building 
        systems that can learn from data. These systems improve their performance over 
        time without being explicitly programmed.
        
        Supervised Learning
        
        In supervised learning, the algorithm learns from labeled training data. The 
        model makes predictions based on this training and is corrected when those 
        predictions are wrong. Common examples include classification and regression.
        
        Unsupervised Learning
        
        Unsupervised learning deals with unlabeled data. The algorithm tries to find 
        hidden patterns or intrinsic structures in the input data. Clustering and 
        dimensionality reduction are typical unsupervised learning tasks.
        
        Conclusion
        
        Machine learning continues to evolve and find new applications in various 
        fields including healthcare, finance, and autonomous vehicles.
        """
        
        print("🔬 Semantic Splitter Test")
        print("=" * 50)
        print(f"Threshold: {args.threshold} percentile")
        print(f"Min chunk size: {args.min_chunk} chars")
        print()
        
        splitter = SemanticSplitter(
            breakpoint_threshold=args.threshold,
            min_chunk_size=args.min_chunk,
        )
        
        print("Loading embeddings model...")
        chunks = splitter.split(test_text)
        
        print(f"\n✓ Created {len(chunks)} semantic chunks:\n")
        
        for i, chunk in enumerate(chunks):
            print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
            print()
    
    elif args.text_file:
        with open(args.text_file, 'r') as f:
            text = f.read()
        
        print(f"📄 Splitting: {args.text_file}")
        print(f"   Original length: {len(text)} chars")
        print()
        
        splitter = SemanticSplitter(
            breakpoint_threshold=args.threshold,
            min_chunk_size=args.min_chunk,
        )
        
        print("Loading embeddings model...")
        chunks = splitter.split(text)
        
        print(f"\n✓ Created {len(chunks)} semantic chunks")
        
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
            print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    else:
        parser.print_help()
