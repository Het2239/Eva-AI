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
    action_taken: Optional[str] = None
    intent: Optional[str] = None


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
            from eva.os_tools import OSTools
            from eva.speech import SpeechEngine
            from eva.file_search import FileSearchEngine
        except ImportError:
            from session_rag import SessionRAG
            from conversation_memory import ConversationMemory
            # Fallback for eva package imports if running from rag dir
            sys.path.append(str(Path(__file__).parent.parent))
            from eva.os_tools import OSTools
            from eva.speech import SpeechEngine
            from eva.file_search import FileSearchEngine
        
        self.session_rag = SessionRAG()
        self.conv_memory = ConversationMemory(storage_path=memory_path)
        self.os_tools = OSTools()
        self.speech = SpeechEngine()  # For listen/speak capabilities
        self.file_search = FileSearchEngine(cache_path="file_cache.db")
    
    def ingest(self, path: str, verbose: bool = True) -> str:
        """Ingest documents into session."""
        return self.session_rag.ingest(path, verbose=verbose)
    
    def chat(self, query: str) -> AgentResponse:
        """Main chat interface with automatic RAG/chat routing."""
        # Check for web/browser actions first
        web_result = self._handle_web_action(query)
        if web_result:
            return web_result
            
        try:
            from rag.intent_classifier import needs_rag, classify_intent, Intent
        except ImportError:
            from intent_classifier import needs_rag, classify_intent, Intent
        
        # Check for OS/File intents
        intent = classify_intent(
            query,
            doc_summary=self.session_rag.doc_summary,
            has_documents=self.session_rag.has_documents,
        )
        
        if intent == Intent.OS_ACTION:
            return self._handle_os_action(query)
        elif intent == Intent.FILE_SEARCH:
            return self._handle_file_search(query)
        
        # RAG or Chat
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
            intent=intent.value if intent else None
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

    def _handle_web_action(self, query: str) -> Optional[AgentResponse]:
        """Handle web/browser actions."""
        query_lower = query.lower()
        
        # Play music on YouTube Music
        music_triggers = ["play", "play music", "play song", "listen to"]
        youtube_music_triggers = ["youtube music", "on youtube music", "in youtube music"]
        
        is_music_request = any(t in query_lower for t in music_triggers)
        wants_youtube_music = any(t in query_lower for t in youtube_music_triggers)
        
        if is_music_request:
            search_query = query_lower
            for word in ["play", "music", "song", "listen", "to", "on", "in", "youtube", "chrome", "google"]:
                search_query = search_query.replace(f" {word} ", " ")
                search_query = search_query.replace(f"{word} ", "")
            search_query = search_query.strip()
            
            if search_query:
                if wants_youtube_music or "music" in query_lower:
                    result = self.os_tools.play_on_youtube_music(search_query)
                    return AgentResponse(
                        answer=f"Playing {search_query} on YouTube Music",
                        used_rag=False,
                        action_taken=result,
                        intent="web_action"
                    )
                else:
                    result = self.os_tools.search_youtube(search_query)
                    return AgentResponse(
                        answer=f"Searching {search_query} on YouTube",
                        used_rag=False,
                        action_taken=result,
                        intent="web_action"
                    )
        
        # Open website
        web_triggers = ["open", "go to", "visit", "browse"]
        site_keywords = ["site", "website", ".com", ".org", ".io", "youtube", "google", 
                        "gmail", "github", "twitter", "facebook", "instagram", "reddit",
                        "netflix", "spotify", "amazon", "whatsapp", "chatgpt"]
        
        is_web_request = any(t in query_lower for t in web_triggers)
        has_site = any(s in query_lower for s in site_keywords)
        
        if is_web_request and has_site:
            site_name = query_lower
            for word in ["open", "go", "to", "visit", "browse", "the", "website", "site", "in", "chrome", "browser"]:
                site_name = site_name.replace(f" {word} ", " ")
                site_name = site_name.replace(f"{word} ", "")
            site_name = site_name.strip()
            
            if site_name:
                result = self.os_tools.open_website(site_name)
                return AgentResponse(
                    answer=f"Opening {site_name}",
                    used_rag=False,
                    action_taken=result,
                    intent="web_action"
                )
        return None

    def _handle_os_action(self, query: str) -> AgentResponse:
        """Handle OS action requests."""
        query_lower = query.lower()
        
        # Folder mapping
        folder_map = {
            "downloads": "~/Downloads",
            "documents": "~/Documents",
            "desktop": "~/Desktop",
            "pictures": "~/Pictures",
            "videos": "~/Videos",
            "music": "~/Music",
            "home": "~",
            "this folder": self.os_tools.cwd,
        }
        
        # Add mounted drives dynamically
        for drive in self.os_tools.get_mounted_drives():
            drive_name = drive["name"].lower()
            folder_map[drive_name] = drive["path"]
            # Also add common aliases
            folder_map[drive_name.replace(" ", "")] = drive["path"]
        
        # Check if this is a file search with folder context
        file_keywords = ["file", "pdf", "document", "image", "video", "photo"]
        has_file_hint = any(kw in query_lower for kw in file_keywords)
        
        if has_file_hint and "open" in query_lower:
            search_query = query_lower
            for word in ["open", "a", "the", "file", "named", "called", "with", "name", 
                         "pdf", "image", "video", "photo", "document"]:
                search_query = search_query.replace(f" {word} ", " ")
                search_query = search_query.replace(f"{word} ", "")
            search_query = search_query.strip()
            
            matches = self.os_tools.smart_find(search_query)
            if matches:
                best = matches[0]
                if best.score >= 50:
                    result = self.os_tools.open_path(best.path)
                    return AgentResponse(
                        answer=f"Opening {best.name}",
                        used_rag=False,
                        action_taken=result,
                        intent="file_search"
                    )
                else:
                    names = [m.name for m in matches[:3]]
                    return AgentResponse(
                        answer=f"I found: {', '.join(names)}. Which one?",
                        used_rag=False,
                        intent="file_search"
                    )
            else:
                return AgentResponse(
                    answer="I couldn't find that file.",
                    used_rag=False,
                    intent="file_search"
                )
        
        # Open folder
        for name, path in folder_map.items():
            pattern = f"{name} folder" if name != "this folder" else name
            if pattern in query_lower and ("open" in query_lower or "go to" in query_lower):
                if not has_file_hint:
                    result = self.os_tools.open_folder(path)
                    return AgentResponse(
                        answer=f"Opening {name} folder",
                        used_rag=False,
                        action_taken=result,
                        intent="os_action"
                    )
        
        # Open app
        app_keywords = ["launch", "open", "start", "run"]
        if any(kw in query_lower for kw in app_keywords):
            target = None
            for kw in app_keywords:
                if kw in query_lower:
                    parts = query_lower.split(kw, 1)
                    if len(parts) > 1:
                        target = parts[1].strip()
                        break
            
            if target:
                result = self.os_tools.open_application(target)
                return AgentResponse(
                    answer=f"Opening {target}",
                    used_rag=False,
                    action_taken=result,
                    intent="os_action"
                )
        
        # Close app
        if "close" in query_lower or "quit" in query_lower:
            target = None
            for kw in ["close", "quit"]:
                if kw in query_lower:
                    parts = query_lower.split(kw, 1)
                    if len(parts) > 1:
                        target = parts[1].strip()
                        break
            
            if target:
                result = self.os_tools.close_application(target)
                return AgentResponse(
                    answer=f"Closing {target}",
                    used_rag=False,
                    action_taken=result,
                    intent="os_action"
                )
        
        return AgentResponse(
            answer="I'm not sure what action to take.",
            used_rag=False,
            intent="os_action"
        )

    def _handle_file_search(self, query: str) -> AgentResponse:
        """
        Handle file search requests using intelligent search.
        
        Flow:
            Query -> LLM Parse -> Cache Check -> Volume Search -> Results
        """
        # Use the intelligent file search engine
        results = self.file_search.search(query, limit=5, generate_summaries=True)
        
        if not results:
            return AgentResponse(
                answer=f"I couldn't find any files matching your query. Try being more specific.",
                used_rag=False,
                intent="file_search"
            )
        
        best = results[0]
        
        # High confidence - open directly
        if best.score >= 0.7:
            result = self.os_tools.open_path(best.path)
            summary_note = f"\n(Summary: {best.summary[:100]}...)" if best.summary else ""
            return AgentResponse(
                answer=f"Found and opening {best.name}{summary_note}",
                used_rag=False,
                action_taken=result,
                intent="file_search"
            )
        
        # Medium confidence - show options
        else:
            options = []
            for r in results[:3]:
                summary = f" - {r.summary[:50]}..." if r.summary else ""
                options.append(f"• {r.name}{summary}")
            
            return AgentResponse(
                answer=f"I found these files:\n" + "\n".join(options) + "\n\nWhich one would you like to open?",
                used_rag=False,
                intent="file_search"
            )


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
        # Ask for user name
        print("EVA Agent Chat")
        print("=" * 50)
        
        if args.user == "default":
            name = input("Enter your name (or press Enter for guest): ").strip()
            user_id = name if name else "guest"
            # Recreate agent with correct user
            agent = EVAAgent(user_id=user_id)
        else:
            user_id = args.user
        
        print(f"\nWelcome, {user_id}! 👋")
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
                    elif cmd == 'listen':
                        print("🎤 Listening...")
                        try:
                            text = agent.speech.listen()
                            if text:
                                print(f"Heard: {text}")
                                response = agent.chat(text)
                                print(f"\nEVA: {response.answer}")
                                if response.action_taken:
                                    print(f"  [{response.action_taken}]")
                                if response.used_rag:
                                    print(f"  [RAG: {', '.join(response.sources)}]")
                            else:
                                print("  (No speech detected)")
                        except Exception as e:
                            print(f"  Error listening: {e}")
                    elif cmd == 'help':
                        print("  /listen         - Speak a command (no TTS output)")
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
                if response.action_taken:
                    print(f"  [{response.action_taken}]")
                if response.used_rag:
                    print(f"  [RAG: {', '.join(response.sources)}]")
                print()
            except KeyboardInterrupt:
                break
        
        agent.end_session()
    else:
        parser.print_help()

