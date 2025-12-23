#!/usr/bin/env python3
"""
EVA Model Configuration
=======================
Centralized configuration for all LLM models used in EVA.
Switch between local (Ollama) and API providers from one place.
"""

import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional

# ============================================================
# CONFIGURATION - Edit these to change models globally
# ============================================================

class Provider(Enum):
    OLLAMA = "ollama"           # Local Ollama models
    OPENAI = "openai"           # OpenAI API
    ANTHROPIC = "anthropic"     # Claude API
    HUGGINGFACE = "huggingface" # HuggingFace Inference API
    GROQ = "groq"               # Groq API (fast inference)


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    provider: Provider
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048


# ============================================================
# ACTIVE MODELS - Change these to switch models globally
# ============================================================

# Primary chat/completion model
CHAT_MODEL = ModelConfig(
    provider=Provider.GROQ,
    model_name="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=2048
)

# Embedding model (for RAG) - Best open-source model
EMBEDDING_MODEL = ModelConfig(
    provider=Provider.HUGGINGFACE,
    model_name="BAAI/bge-large-en-v1.5",  # Top MTEB performer, 1024 dims
)

# Reranking model (optional, for RAG)
RERANK_MODEL = ModelConfig(
    provider=Provider.HUGGINGFACE,
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
)


# ============================================================
# API KEYS - Loaded from .env file
# ============================================================

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
    "groq": os.getenv("GROQ_API_KEY"),
}


# ============================================================
# MODEL INTERFACE - Unified API for all providers
# ============================================================

def get_chat_response(prompt: str, model: ModelConfig = None) -> str:
    """
    Get a chat response using the configured model.
    Automatically routes to the correct provider.
    """
    if model is None:
        model = CHAT_MODEL
    
    if model.provider == Provider.OLLAMA:
        return _ollama_chat(prompt, model)
    elif model.provider == Provider.OPENAI:
        return _openai_chat(prompt, model)
    elif model.provider == Provider.ANTHROPIC:
        return _anthropic_chat(prompt, model)
    elif model.provider == Provider.HUGGINGFACE:
        return _huggingface_chat(prompt, model)
    elif model.provider == Provider.GROQ:
        return _groq_chat(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {model.provider}")


def get_embeddings(texts: list[str], model: ModelConfig = None) -> list[list[float]]:
    """
    Get embeddings for a list of texts.
    """
    if model is None:
        model = EMBEDDING_MODEL
    
    if model.provider == Provider.OLLAMA:
        return _ollama_embeddings(texts, model)
    elif model.provider == Provider.OPENAI:
        return _openai_embeddings(texts, model)
    elif model.provider == Provider.HUGGINGFACE:
        return _huggingface_embeddings(texts, model)
    else:
        raise ValueError(f"Embeddings not supported for provider: {model.provider}")


# ============================================================
# PROVIDER IMPLEMENTATIONS
# ============================================================

def _ollama_chat(prompt: str, model: ModelConfig) -> str:
    """Chat using local Ollama."""
    try:
        from ollama import chat
        response = chat(
            model=model.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content
        elif isinstance(response, dict):
            return response.get("message", {}).get("content", str(response))
        return str(response)
    except ImportError:
        # Fallback to subprocess
        import subprocess
        proc = subprocess.run(
            ["ollama", "run", model.model_name],
            input=prompt,
            text=True,
            capture_output=True
        )
        return proc.stdout or proc.stderr or "Model failed."


def _openai_chat(prompt: str, model: ModelConfig) -> str:
    """Chat using OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEYS["openai"])
    response = client.chat.completions.create(
        model=model.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=model.temperature,
        max_tokens=model.max_tokens
    )
    return response.choices[0].message.content


def _anthropic_chat(prompt: str, model: ModelConfig) -> str:
    """Chat using Anthropic Claude API."""
    import anthropic
    client = anthropic.Anthropic(api_key=API_KEYS["anthropic"])
    response = client.messages.create(
        model=model.model_name,
        max_tokens=model.max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def _huggingface_chat(prompt: str, model: ModelConfig) -> str:
    """Chat using HuggingFace Inference API."""
    import requests
    api_url = f"https://api-inference.huggingface.co/models/{model.model_name}"
    headers = {"Authorization": f"Bearer {API_KEYS['huggingface']}"}
    response = requests.post(api_url, headers=headers, json={"inputs": prompt})
    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", str(result))
    return str(result)


def _groq_chat(prompt: str, model: ModelConfig) -> str:
    """Chat using Groq API."""
    from groq import Groq
    client = Groq(api_key=API_KEYS["groq"])
    response = client.chat.completions.create(
        model=model.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=model.temperature,
        max_completion_tokens=model.max_tokens,
        top_p=1,
        stream=False,  # Non-streaming for simple response
    )
    return response.choices[0].message.content


def _ollama_embeddings(texts: list[str], model: ModelConfig) -> list[list[float]]:
    """Get embeddings using local Ollama."""
    from ollama import embeddings
    results = []
    for text in texts:
        resp = embeddings(model=model.model_name, prompt=text)
        results.append(resp["embedding"])
    return results


def _openai_embeddings(texts: list[str], model: ModelConfig) -> list[list[float]]:
    """Get embeddings using OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEYS["openai"])
    response = client.embeddings.create(
        model=model.model_name,
        input=texts
    )
    return [item.embedding for item in response.data]


def _huggingface_embeddings(texts: list[str], model: ModelConfig) -> list[list[float]]:
    """Get embeddings using HuggingFace (local sentence-transformers)."""
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(model.model_name, device="cpu")
    return encoder.encode(texts, convert_to_numpy=True).tolist()


# ============================================================
# QUICK SWITCH PRESETS
# ============================================================

PRESETS = {
    "local": ModelConfig(Provider.OLLAMA, "phi3:mini"),
    "local-fast": ModelConfig(Provider.OLLAMA, "phi3:mini"),
    "local-smart": ModelConfig(Provider.OLLAMA, "llama3.1:8b"),
    "gpt4": ModelConfig(Provider.OPENAI, "gpt-4-turbo-preview"),
    "gpt3": ModelConfig(Provider.OPENAI, "gpt-3.5-turbo"),
    "claude": ModelConfig(Provider.ANTHROPIC, "claude-3-sonnet-20240229"),
    "groq-fast": ModelConfig(Provider.GROQ, "llama-3.1-70b-versatile"),
}


def use_preset(preset_name: str):
    """Switch to a preset model configuration."""
    global CHAT_MODEL
    if preset_name in PRESETS:
        CHAT_MODEL = PRESETS[preset_name]
        print(f"Switched to preset: {preset_name} ({CHAT_MODEL.model_name})")
    else:
        print(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print(f"Current chat model: {CHAT_MODEL.provider.value} / {CHAT_MODEL.model_name}")
    print(f"Current embedding model: {EMBEDDING_MODEL.provider.value} / {EMBEDDING_MODEL.model_name}")
    
    # Test chat
    response = get_chat_response("Say hello in one sentence.")
    print(f"\nTest response: {response}")
