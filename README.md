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

## Project Structure

```
eva_rag/
├── models.py           # Centralized LLM configuration & API
├── .env                # API keys (local only, gitignored)
├── .env.example        # Template for API keys
└── eva/
    ├── eva_pro.sh      # Main CLI interface
    ├── eva_backend.py  # Ollama integration
    ├── prompt_template.txt
    ├── memory.py       # Persistent memory system
    └── memory.json     # Memory storage
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

## Roadmap

- [x] Multi-provider LLM support
- [x] CLI interface
- [x] Centralized model configuration
- [ ] LangChain RAG integration
- [ ] Web scraping & document ingestion
- [ ] Vector database (ChromaDB/FAISS)
- [ ] Web UI

## License

MIT

## Author

Het Patel ([@Het2239](https://github.com/Het2239))
