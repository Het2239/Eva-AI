#!/usr/bin/env python3
"""
EVA RAG - Conversation Memory
=============================
Persistent conversation summaries per user.
Stores running summaries, not full chat history.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

# Add parent directory to path
_this_dir = Path(__file__).parent
if str(_this_dir.parent) not in sys.path:
    sys.path.insert(0, str(_this_dir.parent))


class ConversationMemory:
    """
    Persistent conversation memory using summarization.
    
    Features:
        - Per-user conversation summaries
        - Rolling summarization (not full history)
        - JSON file storage
    """
    
    def __init__(
        self,
        storage_path: str = "./memory",
        max_buffer_messages: int = 10,
    ):
        """
        Initialize conversation memory.
        
        Args:
            storage_path: Directory for storing memory files
            max_buffer_messages: Number of recent messages before summarizing
        """
        self.storage_path = Path(os.path.expanduser(storage_path))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_buffer_messages = max_buffer_messages
        
        # In-memory cache
        self._summaries: Dict[str, dict] = {}
        self._buffers: Dict[str, List[dict]] = {}
        
        # Load existing data
        self._load()
    
    def _get_file_path(self) -> Path:
        """Get path to memory file."""
        return self.storage_path / "conversation_memory.json"
    
    def _load(self) -> None:
        """Load memory from disk."""
        file_path = self._get_file_path()
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self._summaries = data.get('summaries', {})
                    self._buffers = data.get('buffers', {})
            except:
                self._summaries = {}
                self._buffers = {}
    
    def _save(self) -> None:
        """Save memory to disk."""
        file_path = self._get_file_path()
        data = {
            'summaries': self._summaries,
            'buffers': self._buffers,
            'last_updated': datetime.now().isoformat(),
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_summary(self, user_id: str = "default") -> str:
        """
        Get conversation summary for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Conversation summary or empty string
        """
        if user_id in self._summaries:
            return self._summaries[user_id].get('summary', '')
        return ''
    
    def get_context(self, user_id: str = "default") -> str:
        """
        Get full context for LLM (summary + recent messages).
        
        Args:
            user_id: User identifier
            
        Returns:
            Combined context string
        """
        parts = []
        
        # Add summary
        summary = self.get_summary(user_id)
        if summary:
            parts.append(f"Previous conversation summary:\n{summary}")
        
        # Add recent messages
        buffer = self._buffers.get(user_id, [])
        if buffer:
            recent = "\n".join([
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in buffer[-5:]  # Last 5 messages
            ])
            parts.append(f"Recent messages:\n{recent}")
        
        return "\n\n".join(parts) if parts else ""
    
    def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Add a message to the conversation buffer.
        
        Args:
            user_id: User identifier
            role: 'user' or 'assistant'
            content: Message content
        """
        if user_id not in self._buffers:
            self._buffers[user_id] = []
        
        self._buffers[user_id].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
        })
        
        # Check if we need to summarize
        if len(self._buffers[user_id]) >= self.max_buffer_messages:
            self._summarize_buffer(user_id)
        
        self._save()
    
    def _summarize_buffer(self, user_id: str) -> None:
        """Summarize buffer and update summary."""
        buffer = self._buffers.get(user_id, [])
        if not buffer:
            return
        
        # Build conversation text
        conv_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in buffer
        ])
        
        # Get existing summary
        existing_summary = self.get_summary(user_id)
        
        # Generate new summary
        try:
            from models import get_chat_response
            
            if existing_summary:
                prompt = f"""Update this conversation summary with the new messages.

Current summary:
{existing_summary}

New messages:
{conv_text}

Updated summary (2-3 sentences):"""
            else:
                prompt = f"""Summarize this conversation in 2-3 sentences.

Conversation:
{conv_text}

Summary:"""
            
            new_summary = get_chat_response(prompt).strip()
        except:
            # Fallback: just keep last summary + note about new messages
            new_summary = existing_summary + f" [+{len(buffer)} messages]" if existing_summary else f"[{len(buffer)} messages exchanged]"
        
        # Update summary
        self._summaries[user_id] = {
            'summary': new_summary,
            'updated': datetime.now().isoformat(),
            'message_count': self._summaries.get(user_id, {}).get('message_count', 0) + len(buffer),
        }
        
        # Clear buffer (keep last 2 for context continuity)
        self._buffers[user_id] = buffer[-2:]
        
        self._save()
    
    def force_summarize(self, user_id: str = "default") -> str:
        """
        Force summarization of current buffer.
        
        Args:
            user_id: User identifier
            
        Returns:
            New summary
        """
        self._summarize_buffer(user_id)
        return self.get_summary(user_id)
    
    def clear(self, user_id: str = "default") -> None:
        """
        Clear memory for a user.
        
        Args:
            user_id: User identifier
        """
        if user_id in self._summaries:
            del self._summaries[user_id]
        if user_id in self._buffers:
            del self._buffers[user_id]
        self._save()
        print(f"✓ Memory cleared for {user_id}")
    
    def clear_all(self) -> None:
        """Clear all memory."""
        self._summaries.clear()
        self._buffers.clear()
        self._save()
        print("✓ All memory cleared")


# ============================================================
# CLI INTERFACE
# ============================================================

if __name__ == "__main__":
    print("ConversationMemory Test")
    print("=" * 50)
    
    memory = ConversationMemory(storage_path="./test_memory")
    
    # Add some messages
    memory.add_message("test_user", "user", "Hello, how are you?")
    memory.add_message("test_user", "assistant", "I'm doing great! How can I help?")
    memory.add_message("test_user", "user", "Tell me about Python")
    
    # Get context
    print(f"Context:\n{memory.get_context('test_user')}")
    
    # Clean up
    memory.clear("test_user")
    print("Test complete")
