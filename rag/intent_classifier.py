#!/usr/bin/env python3
"""
EVA RAG - Intent Classifier
============================
Determine whether to use RAG or normal chat.
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
_this_dir = Path(__file__).parent
if str(_this_dir.parent) not in sys.path:
    sys.path.insert(0, str(_this_dir.parent))


# Keywords that suggest document reference
RAG_KEYWORDS = {
    # Direct references
    'document', 'documents', 'file', 'files', 'pdf', 'uploaded',
    'notes', 'paper', 'article', 'text', 'content',
    # Contextual references
    'above', 'earlier', 'mentioned', 'said', 'stated', 'according',
    'based on', 'from the', 'in the', 'what does', 'what did',
    # Questions about content
    'summarize', 'summary', 'explain', 'describe', 'list',
    'find', 'search', 'look for', 'what is', 'how does',
}


def needs_rag_keywords(query: str, doc_summary: Optional[str] = None) -> bool:
    """
    Keyword-based intent classification.
    
    Args:
        query: User query
        doc_summary: Summary of uploaded documents
        
    Returns:
        True if RAG should be used
    """
    query_lower = query.lower()
    
    # Check for RAG keywords
    for keyword in RAG_KEYWORDS:
        if keyword in query_lower:
            return True
    
    # Check if query mentions topics from document summary
    if doc_summary:
        doc_words = set(doc_summary.lower().split())
        query_words = set(query_lower.split())
        
        # If significant overlap, use RAG
        overlap = doc_words & query_words
        # Filter out common words
        significant_words = {w for w in overlap if len(w) > 4}
        
        if len(significant_words) >= 2:
            return True
    
    return False


def needs_rag_llm(
    query: str,
    doc_summary: Optional[str] = None,
) -> bool:
    """
    LLM-based intent classification.
    
    Args:
        query: User query
        doc_summary: Summary of uploaded documents
        
    Returns:
        True if RAG should be used
    """
    if not doc_summary:
        return False
    
    try:
        from models import get_chat_response
        
        prompt = f"""You are an intent classifier. Determine if the user's question requires information from their uploaded documents.

Document summary: {doc_summary}

User query: {query}

Answer ONLY "YES" if the document should be used to answer, or "NO" if it's a general question.
Answer:"""
        
        response = get_chat_response(prompt).strip().upper()
        return response.startswith("YES")
    except:
        # Fallback to keyword-based
        return needs_rag_keywords(query, doc_summary)


def needs_rag(
    query: str,
    doc_summary: Optional[str] = None,
    has_documents: bool = False,
    use_llm: bool = False,
) -> bool:
    """
    Main intent classification function.
    
    Args:
        query: User query
        doc_summary: Summary of uploaded documents
        has_documents: Whether session has documents
        use_llm: Whether to use LLM for classification
        
    Returns:
        True if RAG should be used
    """
    # No documents = no RAG
    if not has_documents:
        return False
    
    # Use LLM or keyword-based
    if use_llm:
        return needs_rag_llm(query, doc_summary)
    else:
        return needs_rag_keywords(query, doc_summary)


# ============================================================
# CLI INTERFACE
# ============================================================

if __name__ == "__main__":
    print("Intent Classifier Test")
    print("=" * 50)
    
    test_queries = [
        ("What does the document say about Python?", True),
        ("Hello, how are you?", False),
        ("Summarize the uploaded file", True),
        ("What's the weather like?", False),
        ("According to the notes, what is X?", True),
        ("Tell me a joke", False),
        ("Find the main topics in the paper", True),
    ]
    
    doc_summary = "This document covers Python programming, machine learning, and data science."
    
    for query, expected in test_queries:
        result = needs_rag(query, doc_summary, has_documents=True)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{query[:40]}...' → RAG: {result}")
