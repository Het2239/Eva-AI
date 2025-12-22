# EVA - AI Assistant with RAG

A modular AI assistant with multi-provider LLM support and RAG (Retrieval-Augmented Generation) capabilities.

## Features

- 🤖 **Multi-Provider LLM Support** - Seamlessly switch between:
  - Ollama (local)
  - OpenAI (GPT-4, GPT-3.5)
  - Anthropic (Claude)
  - Groq (fast inference)
  - HuggingFace

- 💬 **CLI Chat Interface** - Interactive command-line chat with EVA
- 🧠 **Persistent Memory** - Remember context across sessions
- 🔧 **Centralized Configuration** - Change models globally from one file

- 📄 **Multi-Format Document Loading** - Parse and extract text from:
  - Documents: PDF, DOCX, PPTX, XLSX, HTML, Markdown
  - OpenDocument: ODT, ODP, ODS
  - Images (OCR): PNG, JPG, TIFF, BMP, WebP
  - Text: TXT, CSV, JSON, XML, YAML, and more

## Project Structure

```
eva_rag/
├── models.py           # Centralized LLM configuration & API
├── .env                # API keys (local only, gitignored)
├── .env.example        # Template for API keys
├── requirements.txt    # Python dependencies
├── eva/
│   ├── eva_pro.sh      # Main CLI interface
│   ├── eva_backend.py  # Ollama integration
│   ├── prompt_template.txt
│   ├── memory.py       # Persistent memory system
│   └── memory.json     # Memory storage
└── rag/
    ├── __init__.py     # RAG module exports
    └── document_loader.py  # Multi-format document loader (Docling)
```

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Het2239/Eva-AI.git
cd eva_rag

# Copy environment template and add your API keys
cp .env.example .env
nano .env  # Add your API keys
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add to Bash (Optional)

Add this to your `~/.bashrc`:
```bash
eva() {
    ~/eva_rag/eva/eva_pro.sh "$@"
}
```

Then reload: `source ~/.bashrc`

### 4. Use EVA

```bash
eva "Hello, how are you?"
eva "Explain quantum computing"
```

## Configuration

### Switching Models

Edit `models.py` to change the default model:

```python
CHAT_MODEL = ModelConfig(
    provider=Provider.GROQ,           # Change provider
    model_name="openai/gpt-oss-120b", # Change model
    temperature=0.7,
    max_tokens=2048
)
```

### Available Providers

| Provider | Example Models |
|----------|---------------|
| `OLLAMA` | `phi3:mini`, `llama3.1:8b` |
| `OPENAI` | `gpt-4-turbo`, `gpt-3.5-turbo` |
| `ANTHROPIC` | `claude-3-sonnet-20240229` |
| `GROQ` | `openai/gpt-oss-120b`, `llama-3.1-70b-versatile` |
| `HUGGINGFACE` | `mistralai/Mistral-7B-Instruct-v0.2` |

### Using from Python

```python
from models import get_chat_response, use_preset

# Use default model
response = get_chat_response("Hello!")

# Switch to a preset
use_preset("groq-fast")
response = get_chat_response("Hello!")
```

## Environment Variables

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HUGGINGFACE_API_KEY=hf_...
GROQ_API_KEY=gsk_...
```

## Document Loading

### Supported Formats

| Category | Formats |
|----------|--------|
| Documents | PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, HTML, MD |
| OpenDocument | ODT, ODP, ODS |
| Images (OCR) | PNG, JPG, JPEG, TIFF, BMP, WebP |
| Text Files | TXT, CSV, JSON, XML, YAML, RST, TEX, LOG |

### Usage

```bash
# Load any document
python3 rag/document_loader.py document.pdf
python3 rag/document_loader.py spreadsheet.xlsx

# List all supported formats
python3 rag/document_loader.py . --list-formats

# Disable OCR (faster for digital docs)
python3 rag/document_loader.py document.pdf --no-ocr
```

```python
from rag import load_document, load_documents_from_directory

# Load any supported format
docs = load_document("document.pdf")
docs = load_document("spreadsheet.xlsx")
docs = load_document("image.png")  # With OCR

# Load all documents from a directory
docs = load_documents_from_directory("~/Documents/")
```

## Roadmap

- [x] Multi-provider LLM support
- [x] CLI interface
- [x] Centralized model configuration
- [x] Multi-format document parsing (Docling - IBM Research)
- [ ] Text chunking & splitting
- [ ] Vector database (ChromaDB/FAISS)
- [ ] RAG query pipeline
- [ ] Web scraping & URL ingestion
- [ ] Web UI

## License

MIT

## Author

Het Patel ([@Het2239](https://github.com/Het2239))
