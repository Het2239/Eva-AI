# EVA - AI Assistant with RAG & Voice

A modular AI assistant with multi-provider LLM support, session-based RAG, voice control, and intelligent conversation memory.

## Features

- 🤖 **Multi-Provider LLM** - Ollama, OpenAI, Anthropic, Groq, HuggingFace
- 📄 **Document Ingestion** - PDF, DOCX, images (OCR), and 20+ formats
- 🧠 **3-Memory Architecture** - Session RAG + Conversation Summaries + LLM Context
- 🔍 **Hybrid Retrieval** - Dense (ChromaDB) + Sparse (BM25) + Reranking
- 🎯 **Intent Classification** - Auto-routes between RAG and normal chat
- 🎤 **Voice Mode** - Hands-free with Whisper STT + Piper TTS
- 💻 **OS Control** - Open apps, search files, browse web via voice/text

## Quick Start

```bash
# Clone & setup
git clone https://github.com/Het2239/Eva-AI.git
cd eva_rag

# Install dependencies
pip install -r requirements.txt

# Setup voice (optional)
./setup_voice.sh

# Add API key
echo "GROQ_API_KEY=gsk_..." > .env

# Start chat
eva chat
```

## Usage

### Chat Mode

```bash
eva chat
```

```
Commands: /ingest <file>, /listen, /status, /end, /quit

You: /ingest document.pdf
✓ Added 12 chunks to session

You: What does the document say about X?
EVA: Based on the document... [RAG: document.pdf]

You: /listen
🎤 Listening... (speak your command)
```

### Voice Mode

```bash
eva voice
```

Hands-free mode with continuous listening and spoken responses.

### OS & Web Commands

Works in both chat and voice mode:
- "Open VS Code"
- "Play Shape of You on YouTube Music"
- "Find physics quiz PDF in Downloads"
- "Open GitHub"

## Project Structure

```
eva_rag/
├── models.py              # LLM configuration
├── setup_voice.sh         # Voice setup script
├── eva/
│   ├── speech.py          # STT (Whisper) + TTS (Piper)
│   ├── voice_agent.py     # Voice control loop
│   └── os_tools.py        # OS/Web actions
└── rag/
    ├── document_loader.py # Docling + Unstructured
    ├── semantic_splitter.py
    ├── chunk_processor.py
    ├── vector_store.py    # ChromaDB + BM25
    ├── retriever.py       # Hybrid + Reranking
    ├── intent_classifier.py
    └── agent.py           # EVAAgent
```

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System overview, memory model, API reference |
| [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) | Pipeline flowcharts, code locations, tunable parameters |

## Configuration

### LLM (models.py)

```python
CHAT_MODEL = ModelConfig(
    provider=Provider.GROQ,
    model_name="llama-3.1-70b-versatile",
    temperature=0.7,
)
```

### Voice (eva/speech.py)

Edit `SpeechConfig` to change:
- Whisper model: `base` / `small` / `medium` / `large`
- Piper voice: Download from [Piper Voices](https://huggingface.co/rhasspy/piper-voices)

## Roadmap

- [x] Multi-provider LLM support
- [x] Document parsing (Docling + Unstructured)
- [x] Semantic chunking (BGE embeddings)
- [x] Hybrid retrieval (ChromaDB + BM25)
- [x] Cross-encoder reranking
- [x] Session-based agent
- [x] Voice mode (Whisper + Piper)
- [x] OS control & web actions
- [ ] Web UI
- [ ] Plugin system

## License

MIT

## Author

Het Patel ([@Het2239](https://github.com/Het2239))

