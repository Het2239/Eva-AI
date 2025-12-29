# EVA Configuration Guide

Complete configuration reference for all EVA components.

---

## Table of Contents

1. [Environment Variables](#environment-variables)
2. [LLM Configuration](#llm-configuration)
3. [RAG Settings](#rag-settings)
4. [Voice Settings](#voice-settings)
5. [OS Tools Settings](#os-tools-settings)

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Required: At least one LLM provider
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx

# Optional: Additional providers
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxx

# Optional: Ollama (no key needed, just URL)
OLLAMA_BASE_URL=http://localhost:11434
```

---

## LLM Configuration

**File:** `models.py`

### Provider Selection

```python
from models import ModelConfig, Provider

# Available providers
Provider.GROQ       # Fast, cloud-based
Provider.OPENAI     # GPT models
Provider.ANTHROPIC  # Claude models
Provider.OLLAMA     # Local inference
Provider.HUGGINGFACE # HuggingFace Inference API
```

### Chat Model

```python
# Line ~50 in models.py
CHAT_MODEL = ModelConfig(
    provider=Provider.GROQ,
    model_name="llama-3.1-70b-versatile",
    temperature=0.7,      # 0.0 = deterministic, 1.0 = creative
    max_tokens=2048,      # Max response length
)
```

### Available Models by Provider

| Provider | Models |
|----------|--------|
| GROQ | `llama-3.1-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768` |
| OPENAI | `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo` |
| ANTHROPIC | `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku` |
| OLLAMA | Any model you have pulled locally |

### Embedding Model

```python
# Line ~80 in models.py
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Alternatives:
# "BAAI/bge-base-en-v1.5"     - Faster, less accurate
# "BAAI/bge-m3"               - Multilingual
# "sentence-transformers/all-MiniLM-L6-v2"  - Fast, general purpose
```

---

## RAG Settings

### Chunking

**File:** `rag/semantic_splitter.py`

```python
# Line ~25
class SemanticSplitter:
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        chunk_size: int = 500,           # Target chunk size (chars)
        chunk_overlap: int = 50,          # Overlap between chunks
        similarity_threshold: float = 0.5, # Split when similarity drops below
    ):
```

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `chunk_size` | 500 | 200-1500 | Smaller = precise retrieval, Larger = more context |
| `chunk_overlap` | 50 | 0-200 | Higher = better continuity, more storage |
| `similarity_threshold` | 0.5 | 0.3-0.8 | Lower = more aggressive splitting |

### Chunk Processing

**File:** `rag/chunk_processor.py`

```python
# Line ~26
class ChunkProcessor:
    def __init__(
        self,
        min_chunk_length: int = 30,       # Filter short chunks
        max_chunk_length: int = 15000,    # Filter very long chunks
        max_garbage_ratio: float = 0.7,   # Max non-alphabetic content
        remove_duplicates: bool = True,
        extract_keywords: bool = True,
        detect_content_type: bool = True,
    ):
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `min_chunk_length` | 30 | Increase to filter more noise |
| `max_garbage_ratio` | 0.7 | Decrease for cleaner text (may filter code/math) |

### Retrieval

**File:** `rag/retriever.py`

```python
# Line ~262
class RAGRetriever:
    def __init__(
        self,
        vector_store,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        dense_weight: float = 0.5,       # Dense vs sparse balance
        use_reranker: bool = True,       # Cross-encoder reranking
        use_math_rewriter: bool = True,  # Math query expansion
    ):
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `dense_weight` | 0.5 | 1.0 = dense only, 0.0 = sparse only |
| `use_reranker` | True | Better precision, slower |
| `rerank_model` | ms-marco-MiniLM | Change for quality/speed tradeoff |

### Query Settings

```python
# In retriever.py, Line ~289
def retrieve(
    self,
    query: str,
    top_k: int = 5,              # Final results returned
    rerank_candidates: int = 20,  # Candidates for reranking
):
```

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `top_k` | 5 | 3-20 | More = better recall, longer context |
| `rerank_candidates` | 20 | 10-50 | More = better precision, slower |

### Vector Store

**File:** `rag/vector_store.py`

```python
# Line ~50
class VectorStore:
    def __init__(
        self,
        collection_name: str = "eva_rag",
        persist_directory: str = "./rag_data",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
    ):
```

---

## Voice Settings

**File:** `eva/speech.py`

### Speech Configuration

```python
# Line ~48
@dataclass
class SpeechConfig:
    wake_word: str = "eva"              # Wake word (if enabled)
    whisper_model: str = "base"         # STT model size
    tts_engine: str = "piper"           # TTS engine
    tts_voice: str = "en-US-AriaNeural" # Edge TTS voice (fallback)
    piper_model: str = "en_US-amy-medium.onnx"  # Piper voice
    sample_rate: int = 16000            # Audio sample rate
    silence_threshold: float = 0.01     # Mic sensitivity
    silence_duration: float = 1.5       # Seconds before processing
```

### Whisper Model Sizes

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | 39M | Fastest | Low |
| `base` | 74M | Fast | Medium |
| `small` | 244M | Medium | Good |
| `medium` | 769M | Slow | Very Good |
| `large` | 1.5G | Slowest | Best |

### Piper Voices

Download from [Piper Voices](https://huggingface.co/rhasspy/piper-voices):

```bash
# Example: Download Ryan (male) voice
cd eva/models/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json
```

Then update `speech.py`:
```python
piper_model: str = "en_US-ryan-medium.onnx"
```

### Silence Detection

| Parameter | Default | Effect |
|-----------|---------|--------|
| `silence_threshold` | 0.01 | Lower = more sensitive mic |
| `silence_duration` | 1.5 | Seconds of silence before processing |

---

## OS Tools Settings

**File:** `eva/os_tools.py`

### File Search

```python
# Line ~161
def resolve_file(
    self,
    filename: str,
    search_dirs: List[str] = None,  # Directories to search
    recursive: bool = True,          # Search subdirectories
    fuzzy: bool = True,              # Fuzzy matching
    threshold: int = 70,             # Fuzzy match threshold (0-100)
):
```

### Default Search Directories

```python
# Line ~180
DEFAULT_SEARCH_DIRS = [
    str(Path.home() / "Downloads"),
    str(Path.home() / "Documents"),
    str(Path.home() / "Desktop"),
]
```

### Whitelisted Applications

```python
# Line ~350
# Apps that can be opened via voice/text command
WHITELISTED_APPS = [
    "code", "firefox", "chrome", "terminal",
    "nautilus", "gimp", "vlc", "spotify", ...
]
```

---

## Quick Configuration Examples

### High Accuracy RAG

```python
# models.py
CHAT_MODEL = ModelConfig(
    provider=Provider.GROQ,
    model_name="llama-3.1-70b-versatile",
    temperature=0.3,  # More deterministic
)

# retriever.py - RAGRetriever.__init__()
dense_weight = 0.6        # Favor dense
use_reranker = True
rerank_candidates = 50    # More candidates

# chunk_processor.py - ChunkProcessor.__init__()
min_chunk_length = 50     # Filter noise
chunk_size = 300          # Smaller chunks
```

### Fast Response

```python
# models.py
CHAT_MODEL = ModelConfig(
    provider=Provider.GROQ,
    model_name="llama-3.1-8b-instant",  # Smaller model
    temperature=0.7,
)

# retriever.py
use_reranker = False      # Skip reranking
top_k = 3                 # Fewer results
```

### Low Resource Voice

```python
# speech.py - SpeechConfig
whisper_model = "tiny"    # Smallest STT model
silence_duration = 2.0    # Less frequent processing
```

---

## Author

Het Patel ([@Het2239](https://github.com/Het2239))
