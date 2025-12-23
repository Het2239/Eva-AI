# EVA - AI Assistant with RAG

A modular AI assistant with multi-provider LLM support, session-based RAG, and intelligent conversation memory.

## Features

- 🤖 **Multi-Provider LLM** - Ollama, OpenAI, Anthropic, Groq, HuggingFace
- 📄 **Document Ingestion** - PDF, DOCX, images (OCR), and 20+ formats
- 🧠 **3-Memory Architecture** - Session RAG + Conversation Summaries + LLM Context
- 🔍 **Hybrid Retrieval** - Dense (ChromaDB) + Sparse (BM25) + Reranking
- 🎯 **Intent Classification** - Auto-routes between RAG and normal chat
- 💬 **Interactive Chat** - CLI with document ingestion on-the-fly

## Quick Start

```bash
# Clone & setup
git clone https://github.com/Het2239/Eva-AI.git
cd eva_rag

# Install dependencies
pip install -r requirements.txt

# Add API key
echo "GROQ_API_KEY=gsk_..." > .env

# Start chat
python3 rag/agent.py chat
```

## Usage

### Session Agent (Recommended)

```bash
python3 rag/agent.py --user het chat
```

```
Commands: /ingest <file>, /status, /end, /quit, /help

You: /ingest document.pdf
✓ Added 12 chunks to session

You: What does the document say about X?
EVA: Based on the document... [RAG: document.pdf]

You: Hello!
EVA: Hi there! (normal chat - no RAG)

You: /end
✓ Session cleared
```

### Persistent RAG Pipeline

```bash
# Ingest to knowledge base
eva ingest document.pdf

# Query
eva ask "What is machine learning?"

# Interactive
eva chat
```

### Python API

```python
from rag import EVAAgent

agent = EVAAgent(user_id="het")
agent.ingest("document.pdf")  # Session-scoped

response = agent.chat("What does the document say?")
print(response.answer)
print(response.sources)  # ["document.pdf"]

agent.end_session()  # Clears documents
```

## Memory Model

| Memory | Scope | Lifecycle |
|--------|-------|-----------|
| Session RAG | Per session | Ephemeral (cleared on `/end`) |
| Conversation Summary | Per user | Persistent (JSON) |
| LLM Context | Per request | Transient |

## Project Structure

```
eva_rag/
├── models.py              # LLM configuration
├── ARCHITECTURE.md        # Detailed technical docs
├── eva/
│   └── eva_pro.sh         # CLI interface
└── rag/
    ├── document_loader.py # Hybrid Unstructured + Docling
    ├── semantic_splitter.py
    ├── chunk_processor.py
    ├── vector_store.py    # ChromaDB + BM25
    ├── retriever.py       # Hybrid search + reranking
    ├── rag_pipeline.py    # Persistent RAG
    ├── session_rag.py     # Ephemeral session store
    ├── conversation_memory.py
    ├── intent_classifier.py
    └── agent.py           # EVAAgent
```

## Supported Formats

| Category | Formats |
|----------|---------|
| Documents | PDF, DOCX, PPTX, XLSX, HTML, MD |
| OpenDocument | ODT, ODP, ODS |
| Images (OCR) | PNG, JPG, TIFF, BMP, WebP |
| Text | TXT, CSV, JSON, XML, YAML |

## Configuration

Edit `models.py`:

```python
CHAT_MODEL = ModelConfig(
    provider=Provider.GROQ,
    model_name="llama-3.1-70b-versatile",
)
```

## Roadmap

- [x] Multi-provider LLM support
- [x] Document parsing (Docling + Unstructured)
- [x] Semantic chunking (BGE embeddings)
- [x] Hybrid retrieval (ChromaDB + BM25)
- [x] Cross-encoder reranking
- [x] Session-based agent
- [x] Conversation memory
- [x] Intent classification
- [ ] Web scraping & URL ingestion
- [ ] Web UI

## Documentation

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

## License

MIT

## Author

Het Patel ([@Het2239](https://github.com/Het2239))
