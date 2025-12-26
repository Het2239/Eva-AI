#!/usr/bin/env python3
"""
EVA Voice Agent
===============
Voice-enabled agent with OS control.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

_this_dir = Path(__file__).parent
if str(_this_dir.parent) not in sys.path:
    sys.path.insert(0, str(_this_dir.parent))


@dataclass
class VoiceResponse:
    """Response from voice agent."""
    text: str
    intent: str
    action_taken: Optional[str] = None


class VoiceAgent:
    """
    Voice-enabled EVA agent.
    
    Combines:
    - Speech recognition (Whisper)
    - Text-to-speech (edge-tts)
    - Session RAG
    - OS tools
    """
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        
        # Import components
        try:
            from rag.agent import EVAAgent
            from rag.intent_classifier import classify_intent, Intent
        except ImportError:
            from agent import EVAAgent
            from intent_classifier import classify_intent, Intent
        
        from eva.os_tools import OSTools
        from eva.speech import SpeechEngine
        
        self.eva = EVAAgent(user_id=user_id)
        self.os_tools = OSTools()
        self.speech = SpeechEngine()
        self.Intent = Intent
        self._classify = classify_intent
    
    def process(self, query: str) -> VoiceResponse:
        """
        Process a query (text or from voice).
        
        Routes to appropriate handler based on intent.
        """
        intent = self._classify(
            query,
            doc_summary=self.eva.session_rag.doc_summary,
            has_documents=self.eva.has_documents,
        )
        
        if intent == self.Intent.OS_ACTION:
            return self._handle_os_action(query)
        elif intent == self.Intent.FILE_SEARCH:
            return self._handle_file_search(query)
        else:
            # RAG or CHAT - use EVA agent
            response = self.eva.chat(query)
            return VoiceResponse(
                text=response.answer,
                intent=intent.value,
            )
    
    def _handle_os_action(self, query: str) -> VoiceResponse:
        """Handle OS action requests."""
        query_lower = query.lower()
        
        # Parse action
        action = None
        target = None
        
        # Open folder
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
        
        for name, path in folder_map.items():
            if name in query_lower and ("open" in query_lower or "go to" in query_lower):
                result = self.os_tools.open_folder(path)
                return VoiceResponse(
                    text=f"Opening {name} folder",
                    intent="os_action",
                    action_taken=result,
                )
        
        # Open app
        app_keywords = ["launch", "open", "start", "run"]
        if any(kw in query_lower for kw in app_keywords):
            # Extract app name (simple heuristic)
            for kw in app_keywords:
                if kw in query_lower:
                    parts = query_lower.split(kw, 1)
                    if len(parts) > 1:
                        target = parts[1].strip()
                        break
            
            if target:
                result = self.os_tools.open_application(target)
                return VoiceResponse(
                    text=f"Opening {target}",
                    intent="os_action",
                    action_taken=result,
                )
        
        # Close app
        if "close" in query_lower or "quit" in query_lower:
            for kw in ["close", "quit"]:
                if kw in query_lower:
                    parts = query_lower.split(kw, 1)
                    if len(parts) > 1:
                        target = parts[1].strip()
                        break
            
            if target:
                result = self.os_tools.close_application(target)
                return VoiceResponse(
                    text=f"Closing {target}",
                    intent="os_action",
                    action_taken=result,
                )
        
        return VoiceResponse(
            text="I'm not sure what action to take. Could you be more specific?",
            intent="os_action",
        )
    
    def _handle_file_search(self, query: str) -> VoiceResponse:
        """Handle file search requests."""
        # Extract search terms
        query_lower = query.lower()
        
        # Remove common prefixes
        for prefix in ["find", "search for", "look for", "where is", "locate"]:
            if query_lower.startswith(prefix):
                query_lower = query_lower[len(prefix):].strip()
                break
        
        # Remove "the", "my", "a"
        for word in ["the", "my", "a"]:
            query_lower = query_lower.replace(f" {word} ", " ")
        
        search_term = query_lower.strip()
        
        if not search_term:
            return VoiceResponse(
                text="What file would you like me to find?",
                intent="file_search",
            )
        
        # Search
        matches = self.os_tools.resolve_file(search_term)
        
        if not matches:
            return VoiceResponse(
                text=f"I couldn't find any files matching '{search_term}'",
                intent="file_search",
            )
        
        best = matches[0]
        
        if best.score >= 70:
            result = self.os_tools.open_path(best.path)
            return VoiceResponse(
                text=f"Found and opening {best.name}",
                intent="file_search",
                action_taken=result,
            )
        else:
            # Low confidence - list options
            names = [m.name for m in matches[:3]]
            return VoiceResponse(
                text=f"I found these files: {', '.join(names)}. Which one?",
                intent="file_search",
            )
    
    def ingest(self, path: str) -> str:
        """Ingest document to session."""
        return self.eva.ingest(path)
    
    def end_session(self) -> None:
        """End session."""
        self.eva.end_session()
    
    def voice_loop(self) -> None:
        """
        Main voice interaction loop.
        
        Listens for wake word "EVA", processes commands.
        """
        print("EVA Voice Mode")
        print("=" * 50)
        print("Say 'EVA' followed by your command")
        print("Say 'EVA quit' to exit")
        print()
        
        self.speech.speak("Hello! I'm EVA, your voice assistant. How can I help?")
        
        while True:
            try:
                # Listen
                command = self.speech.listen_for_command()
                
                if command is None:
                    continue
                
                if not command:
                    self.speech.speak("Yes?")
                    continue
                
                # Check for quit
                if command.lower() in ["quit", "exit", "goodbye", "bye"]:
                    self.speech.speak("Goodbye!")
                    break
                
                print(f"You: {command}")
                
                # Process
                response = self.process(command)
                
                print(f"EVA: {response.text}")
                if response.action_taken:
                    print(f"  [{response.action_taken}]")
                
                # Speak response
                self.speech.speak(response.text)
                
            except KeyboardInterrupt:
                print("\nInterrupted")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        self.end_session()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EVA Voice Agent")
    parser.add_argument("--user", default="default", help="User ID")
    subparsers = parser.add_subparsers(dest="command")
    
    # voice - continuous listening
    voice_p = subparsers.add_parser("voice")
    
    # text - single text command
    text_p = subparsers.add_parser("text")
    text_p.add_argument("query")
    
    # listen - single voice command
    listen_p = subparsers.add_parser("listen")
    
    args = parser.parse_args()
    
    if args.command == "voice":
        agent = VoiceAgent(user_id=args.user)
        agent.voice_loop()
    
    elif args.command == "text":
        agent = VoiceAgent(user_id=args.user)
        response = agent.process(args.query)
        print(f"Intent: {response.intent}")
        print(f"Response: {response.text}")
        if response.action_taken:
            print(f"Action: {response.action_taken}")
    
    elif args.command == "listen":
        agent = VoiceAgent(user_id=args.user)
        print("Listening for command...")
        from eva.speech import SpeechEngine
        engine = SpeechEngine()
        command = engine.listen()
        if command:
            print(f"Heard: {command}")
            response = agent.process(command)
            print(f"Response: {response.text}")
            engine.speak(response.text)
    
    else:
        parser.print_help()
