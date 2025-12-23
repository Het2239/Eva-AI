#!/usr/bin/env python3
"""
EVA RAG - Chunk Processor
=========================
Filter and enrich document chunks for optimal retrieval.
"""

import re
import hashlib
from typing import List, Optional, Set
from collections import Counter

from langchain_core.documents import Document


class ChunkProcessor:
    """
    Process document chunks: filter noise and enrich with metadata.
    
    Features:
        - Duplicate detection (content hash + similarity)
        - Quality filtering (length, garbage ratio)
        - Metadata enrichment (keywords, categories, flags)
    """
    
    def __init__(
        self,
        min_chunk_length: int = 30,
        max_chunk_length: int = 15000,
        max_garbage_ratio: float = 0.7,  # Relaxed for math/technical content
        remove_duplicates: bool = True,
        extract_keywords: bool = True,
        detect_content_type: bool = True,
    ):
        """
        Initialize the chunk processor.
        
        Args:
            min_chunk_length: Minimum characters per chunk
            max_chunk_length: Maximum characters per chunk
            max_garbage_ratio: Maximum ratio of non-alphabetic content
            remove_duplicates: Whether to remove duplicate chunks
            extract_keywords: Whether to extract keywords
            detect_content_type: Whether to detect math/code content
        """
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.max_garbage_ratio = max_garbage_ratio
        self.remove_duplicates = remove_duplicates
        self.extract_keywords = extract_keywords
        self.detect_content_type = detect_content_type
    
    # ========================================
    # FILTERING
    # ========================================
    
    def _compute_hash(self, text: str) -> str:
        """Compute content hash for duplicate detection."""
        # Normalize text before hashing
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _calculate_garbage_ratio(self, text: str) -> float:
        """Calculate ratio of non-alphabetic characters."""
        if not text:
            return 1.0
        non_whitespace = [c for c in text if not c.isspace()]
        if not non_whitespace:
            return 1.0
        garbage = sum(1 for c in non_whitespace if not c.isalpha())
        return garbage / len(non_whitespace)
    
    def _is_valid_chunk(self, chunk: Document) -> bool:
        """Check if chunk passes quality filters."""
        text = chunk.page_content
        
        # Length check
        if len(text) < self.min_chunk_length:
            return False
        if len(text) > self.max_chunk_length:
            return False
        
        # Garbage ratio check
        garbage_ratio = self._calculate_garbage_ratio(text)
        if garbage_ratio > self.max_garbage_ratio:
            return False
        
        return True
    
    def filter(self, chunks: List[Document]) -> List[Document]:
        """
        Filter chunks to remove noise and duplicates.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Filtered list of chunks
        """
        if not chunks:
            return []
        
        filtered = []
        seen_hashes: Set[str] = set()
        
        for chunk in chunks:
            # Quality filter
            if not self._is_valid_chunk(chunk):
                continue
            
            # Duplicate filter
            if self.remove_duplicates:
                content_hash = self._compute_hash(chunk.page_content)
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)
            
            filtered.append(chunk)
        
        return filtered
    
    # ========================================
    # ENRICHMENT
    # ========================================
    
    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract top keywords using simple TF-IDF-like scoring."""
        # Tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
            'was', 'one', 'our', 'out', 'has', 'have', 'been', 'were', 'they',
            'this', 'that', 'with', 'from', 'will', 'would', 'there', 'their',
            'what', 'about', 'which', 'when', 'make', 'like', 'time', 'just',
            'know', 'take', 'come', 'could', 'than', 'into', 'also', 'after',
        }
        
        # Filter and count
        word_counts = Counter(w for w in words if w not in stop_words)
        
        # Weight longer words higher
        weighted = {w: c * (1 + len(w) / 10) for w, c in word_counts.items()}
        
        # Return top keywords
        return [w for w, _ in sorted(weighted.items(), key=lambda x: -x[1])[:top_k]]
    
    def _detect_math_content(self, text: str) -> bool:
        """Detect if chunk contains mathematical content."""
        math_patterns = [
            r'[∑∏∫∂∇√∞≈≠≤≥±×÷]',  # Math symbols
            r'\$.*?\$',  # LaTeX inline
            r'\\\[.*?\\\]',  # LaTeX display
            r'\b(equation|theorem|proof|lemma|corollary)\b',
            r'\b(derivative|integral|function|variable)\b',
            r'[a-z]\s*=\s*[a-z0-9]',  # Simple equations
            r'\b(sin|cos|tan|log|ln|exp)\s*\(',  # Math functions
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _detect_code_content(self, text: str) -> bool:
        """Detect if chunk contains code."""
        code_patterns = [
            r'```',  # Code blocks
            r'def\s+\w+\s*\(',  # Python function
            r'function\s+\w+\s*\(',  # JavaScript function
            r'class\s+\w+',  # Class definition
            r'import\s+\w+',  # Import statement
            r'\{\s*\n',  # Code block start
            r'=>',  # Arrow function
            r'if\s*\(.*\)\s*\{',  # If statement
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _detect_category(self, text: str, keywords: List[str]) -> str:
        """Detect content category based on keywords and patterns."""
        text_lower = text.lower()
        
        # Check for specific domains
        if any(kw in text_lower for kw in ['neural', 'learning', 'model', 'training', 'dataset']):
            return 'machine_learning'
        if any(kw in text_lower for kw in ['algorithm', 'complexity', 'runtime', 'data structure']):
            return 'computer_science'
        if self._detect_math_content(text):
            return 'mathematics'
        if self._detect_code_content(text):
            return 'programming'
        
        return 'general'
    
    def enrich(self, chunks: List[Document]) -> List[Document]:
        """
        Enrich chunks with additional metadata.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Enriched chunks with metadata
        """
        enriched = []
        
        for i, chunk in enumerate(chunks):
            text = chunk.page_content
            metadata = chunk.metadata.copy()
            
            # Add chunk index
            metadata['chunk_id'] = i
            metadata['chunk_length'] = len(text)
            
            # Extract keywords
            if self.extract_keywords:
                keywords = self._extract_keywords(text)
                metadata['keywords'] = keywords
            else:
                keywords = []
            
            # Detect content type
            if self.detect_content_type:
                metadata['has_math'] = self._detect_math_content(text)
                metadata['has_code'] = self._detect_code_content(text)
                metadata['category'] = self._detect_category(text, keywords)
            
            enriched.append(Document(
                page_content=text,
                metadata=metadata,
            ))
        
        return enriched
    
    def process(self, chunks: List[Document]) -> List[Document]:
        """
        Full processing: filter + enrich.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Processed chunks
        """
        filtered = self.filter(chunks)
        enriched = self.enrich(filtered)
        return enriched


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def process_chunks(
    chunks: List[Document],
    min_length: int = 50,
    remove_duplicates: bool = True,
) -> List[Document]:
    """
    Process chunks with default settings.
    
    Args:
        chunks: List of document chunks
        min_length: Minimum chunk length
        remove_duplicates: Whether to remove duplicates
        
    Returns:
        Processed chunks
    """
    processor = ChunkProcessor(
        min_chunk_length=min_length,
        remove_duplicates=remove_duplicates,
    )
    return processor.process(chunks)


# ============================================================
# CLI INTERFACE
# ============================================================

if __name__ == "__main__":
    import sys
    
    # Test with sample chunks
    test_chunks = [
        Document(page_content="This is a test document about machine learning and neural networks.", metadata={"source": "test1.txt"}),
        Document(page_content="This is a test document about machine learning and neural networks.", metadata={"source": "test2.txt"}),  # Duplicate
        Document(page_content="Short", metadata={"source": "test3.txt"}),  # Too short
        Document(page_content="The derivative of x^2 is 2x. This follows from the power rule.", metadata={"source": "test4.txt"}),
        Document(page_content="def hello_world():\n    print('Hello, World!')", metadata={"source": "test5.txt"}),
        Document(page_content="!@#$%^&*()!@#$%^&*()", metadata={"source": "test6.txt"}),  # Garbage
    ]
    
    print("ChunkProcessor Test")
    print("=" * 50)
    print(f"Input: {len(test_chunks)} chunks")
    
    processor = ChunkProcessor()
    processed = processor.process(test_chunks)
    
    print(f"Output: {len(processed)} chunks")
    print()
    
    for chunk in processed:
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"  Length: {chunk.metadata.get('chunk_length')}")
        print(f"  Keywords: {chunk.metadata.get('keywords')}")
        print(f"  Math: {chunk.metadata.get('has_math')}, Code: {chunk.metadata.get('has_code')}")
        print(f"  Category: {chunk.metadata.get('category')}")
        print()
