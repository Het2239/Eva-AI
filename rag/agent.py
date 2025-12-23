#!/usr/bin/env python3
"""
EVA RAG - Agent
===============
Session-based agent with RAG, conversation memory, and intent classification.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field

# Add parent directory to path
_this_dir = Path(__file__).parent
if str(_this_dir.parent) not in sys.path:
    sys.path.insert(0, str(_this_dir.parent))


@dataclass
class AgentResponse:
    """Response from the agent."""
    answer: str
    used_rag: bool
    sources: List[str] = field(default_factory=list)
    doc_summary: Optional[str] = None


class EVAAgent:
    """
    EVA Agent with session RAG and conversation memory.
    
    Features:
        - Session-scoped document RAG (ephemeral)
        - Persistent conversation summaries
        - Intent-based RAG gating
        - Normal chat fallback
    """
    
    def __init__(
        self,
        user_id: str = "default",
        memory_path: str = "./memory",
        use_llm_intent: bool = False,
    ):
        self.user_id = user_id
        self.use_llm_intent = use_llm_intent
        
        # Initialize components
        try:
            from rag.session_rag import SessionRAG
            from rag.conversation_memory import ConversationMemory
        except ImportError:
            from session_rag import SessionRAG
            from conversation_memory import ConversationMemory
        
        self.session_rag = SessionRAG()
        self.conv_memory = ConversationMemory(storage_path=memory_path)
    
    def ingest(self, path: str, verbose: bool = True) -> str:
        """Ingest documents into session."""
        return self.session_rag.ingest(path, verbose=verbose)
    
    def chat(self, query: str) -> AgentResponse:
        """Main chat interface with automatic RAG/chat routing."""
        try:
            from rag.intent_classifier import needs_rag
        except ImportError:
            from intent_classifier import needs_rag
        
        use_rag = needs_rag(
            query=query,
            doc_summary=self.session_rag.doc_summary,
            has_documents=self.session_rag.has_documents,
            use_llm=self.use_llm_intent,
        )
        
        if use_rag and self.session_rag.has_documents:
            response, sources = self._rag_response(query)
        else:
            response = self._chat_response(query)
            sources = []
        
        self.conv_memory.add_message(self.user_id, "user", query)
        self.conv_memory.add_message(self.user_id, "assistant", response)
        
        return AgentResponse(
            answer=response,
            used_rag=use_rag,
            sources=sources,
            doc_summary=self.session_rag.doc_summary,
        )
    
    def _rag_response(self, query: str) -> tuple:
        """Generate response using RAG."""
        docs = self.session_rag.retrieve(query, k=5)
        
        if not docs:
            return self._chat_response(query), []
        
        context_parts = []
        sources = set()
        
        for doc in docs:
            context_parts.append(doc.page_content)
            source = doc.metadata.get('source_file') or doc.metadata.get('source', 'document')
            sources.add(str(source))
        
        context = "\n\n---\n\n".join(context_parts)
        conv_context = self.conv_memory.get_context(self.user_id)
        
        prompt = f"""You are EVA, a helpful AI assistant.

{conv_context}

Document context:
{context}

User question: {query}

Answer based on the document context. Be helpful and accurate.
Answer:"""
        
        try:
            from models import get_chat_response
            response = get_chat_response(prompt)
        except Exception as e:
            response = f"Error: {e}"
        
        return response, list(sources)
    
    def _chat_response(self, query: str) -> str:
        """Generate normal chat response."""
        conv_context = self.conv_memory.get_context(self.user_id)
        
        if conv_context:
            prompt = f"""You are EVA, a helpful AI assistant.

{conv_context}

User: {query}
EVA:"""
        else:
            prompt = f"You are EVA, a helpful AI assistant.\n\nUser: {query}\nEVA:"
        
        try:
            from models import get_chat_response
            return get_chat_response(prompt)
        except Exception as e:
            return f"Error: {e}"
    
    def end_session(self) -> None:
        """End session and clear ephemeral documents."""
        self.session_rag.clear()
        self.conv_memory.force_summarize(self.user_id)
        print(f"Session ended for {self.user_id}")
    
    @property
    def has_documents(self) -> bool:
        return self.session_rag.has_documents
    
    @property
    def document_count(self) -> int:
        return self.session_rag.document_count


def create_agent(user_id: str = "default") -> EVAAgent:
    """Create an EVA agent."""
    return EVAAgent(user_id=user_id)


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EVA Agent")
    parser.add_argument("--user", default="default", help="User ID")
    subparsers = parser.add_subparsers(dest="command")
    
    ingest_p = subparsers.add_parser("ingest")
    ingest_p.add_argument("path")
    
    chat_p = subparsers.add_parser("chat")
    
    args = parser.parse_args()
    
    agent = EVAAgent(user_id=args.user)
    
    if args.command == "ingest":
        agent.ingest(args.path)
    elif args.command == "chat":
        print("EVA Agent Chat")
        print("=" * 50)
        print("Commands: /ingest <file>, /status, /end, /quit, /help")
        print()
        
        while True:
            try:
                query = input("You: ").strip()
                if not query:
                    continue
                
                # Handle commands
                if query.startswith('/'):
                    parts = query[1:].split(maxsplit=1)
                    cmd = parts[0].lower()
                    arg = parts[1] if len(parts) > 1 else ""
                    
                    if cmd == 'quit' or cmd == 'q':
                        break
                    elif cmd == 'end':
                        agent.end_session()
                    elif cmd == 'ingest' and arg:
                        agent.ingest(arg)
                    elif cmd == 'status':
                        print(f"  Documents: {agent.document_count}")
                        if agent.session_rag.doc_summary:
                            print(f"  Summary: {agent.session_rag.doc_summary[:100]}...")
                    elif cmd == 'help':
                        print("  /ingest <file>  - Load document into session")
                        print("  /status         - Show session status")
                        print("  /end            - End session (clears docs)")
                        print("  /quit           - Exit chat")
                    else:
                        print(f"  Unknown command: /{cmd}")
                    continue
                
                # Normal chat
                response = agent.chat(query)
                print(f"\nEVA: {response.answer}")
                if response.used_rag:
                    print(f"  [RAG: {', '.join(response.sources)}]")
                print()
            except KeyboardInterrupt:
                break
        
        agent.end_session()
    else:
        parser.print_help()

