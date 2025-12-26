#!/usr/bin/env python3
"""
EVA Intent Classifier (Extended)
=================================
Classify user intent: CHAT, RAG, OS_ACTION, FILE_SEARCH
"""

import sys
from enum import Enum
from pathlib import Path
from typing import Optional

_this_dir = Path(__file__).parent
if str(_this_dir.parent) not in sys.path:
    sys.path.insert(0, str(_this_dir.parent))


class Intent(Enum):
    """User intent types."""
    CHAT = "chat"
    RAG = "rag"
    OS_ACTION = "os_action"
    FILE_SEARCH = "file_search"


# Keywords for different intents
RAG_KEYWORDS = {
    'document', 'documents', 'file', 'files', 'pdf', 'uploaded',
    'notes', 'paper', 'article', 'text', 'content',
    'above', 'earlier', 'mentioned', 'said', 'stated', 'according',
    'based on', 'from the', 'in the', 'what does', 'what did',
    'summarize', 'summary',
}

OS_ACTION_KEYWORDS = {
    'open', 'launch', 'start', 'run', 'close', 'quit', 'exit',
    'go to', 'navigate', 'show', 'display',
}

FILE_SEARCH_KEYWORDS = {
    'find', 'search', 'look for', 'where is', 'locate',
    'the file', 'my file', 'that file',
}

APP_KEYWORDS = {
    'firefox', 'chrome', 'browser', 'terminal', 'code', 'vscode',
    'spotify', 'vlc', 'settings', 'files', 'nautilus', 'explorer',
    'notepad', 'calculator', 'calendar', 'mail', 'email',
}

FOLDER_KEYWORDS = {
    'folder', 'directory', 'downloads', 'documents', 'desktop',
    'pictures', 'videos', 'music', 'home', 'this folder',
}


def classify_intent(
    query: str,
    doc_summary: Optional[str] = None,
    has_documents: bool = False,
) -> Intent:
    """
    Classify user intent.
    
    Priority: OS_ACTION > FILE_SEARCH > RAG > CHAT
    """
    query_lower = query.lower()
    
    # Check for OS actions (open/launch/close)
    has_os_action = any(kw in query_lower for kw in OS_ACTION_KEYWORDS)
    has_app = any(kw in query_lower for kw in APP_KEYWORDS)
    has_folder = any(kw in query_lower for kw in FOLDER_KEYWORDS)
    
    if has_os_action and (has_app or has_folder):
        return Intent.OS_ACTION
    
    # Check for file search
    has_file_search = any(kw in query_lower for kw in FILE_SEARCH_KEYWORDS)
    if has_file_search:
        return Intent.FILE_SEARCH
    
    # Check for RAG (if documents available)
    if has_documents:
        has_rag = any(kw in query_lower for kw in RAG_KEYWORDS)
        if has_rag:
            return Intent.RAG
        
        # Check topic overlap with documents
        if doc_summary:
            doc_words = set(doc_summary.lower().split())
            query_words = set(query_lower.split())
            overlap = {w for w in (doc_words & query_words) if len(w) > 4}
            if len(overlap) >= 2:
                return Intent.RAG
    
    return Intent.CHAT


def needs_rag(
    query: str,
    doc_summary: Optional[str] = None,
    has_documents: bool = False,
    use_llm: bool = False,
) -> bool:
    """Legacy function for backward compatibility."""
    intent = classify_intent(query, doc_summary, has_documents)
    return intent == Intent.RAG


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    print("Extended Intent Classifier Test")
    print("=" * 50)
    
    tests = [
        ("Open my downloads folder", Intent.OS_ACTION),
        ("Launch Firefox", Intent.OS_ACTION),
        ("Find the physics notes", Intent.FILE_SEARCH),
        ("Where is my resume?", Intent.FILE_SEARCH),
        ("What does the document say?", Intent.RAG),
        ("Hello, how are you?", Intent.CHAT),
        ("Tell me a joke", Intent.CHAT),
        ("Open the file from yesterday", Intent.FILE_SEARCH),
        ("Start VS Code", Intent.OS_ACTION),
        ("Go to desktop", Intent.OS_ACTION),
    ]
    
    for query, expected in tests:
        result = classify_intent(query, has_documents=True)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{query[:35]}...' → {result.value}")
