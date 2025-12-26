#!/usr/bin/env bash

# EVA PRO — Voice-Enabled AI Assistant with RAG
# ==============================================
# Commands:
#   eva voice              - Voice mode (continuous listening)
#   eva chat               - Interactive text chat
#   eva ingest <file|dir>  - Ingest to persistent knowledge base
#   eva ask "question"     - Query persistent knowledge base
#   eva status             - Show RAG status
#   eva do "task" [files]  - Generate shell commands
#   eva "question"         - Direct LLM chat
#   eva "question" [files] - Chat with file context

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_DIR="$SCRIPT_DIR/../rag"

# ======================================================
# VOICE MODE
# ======================================================

if [ "$1" = "voice" ]; then
    shift
    python3 "$SCRIPT_DIR/voice_agent.py" voice
    exit 0
fi

if [ "$1" = "listen" ]; then
    shift
    python3 "$SCRIPT_DIR/voice_agent.py" listen
    exit 0
fi

# ======================================================
# AGENT CHAT (Session-based with /ingest, /status, etc.)
# ======================================================

if [ "$1" = "chat" ]; then
    shift
    python3 "$RAG_DIR/agent.py" chat
    exit 0
fi

# ======================================================
# PERSISTENT RAG COMMANDS
# ======================================================

# Ingest documents to persistent RAG
if [ "$1" = "ingest" ]; then
    shift
    python3 "$RAG_DIR/rag_pipeline.py" ingest "$@"
    exit 0
fi

# Ask RAG query (persistent)
if [ "$1" = "ask" ]; then
    shift
    python3 "$RAG_DIR/rag_pipeline.py" ask "$@"
    exit 0
fi

# RAG status
if [ "$1" = "status" ]; then
    python3 "$RAG_DIR/rag_pipeline.py" status
    exit 0
fi

# ======================================================
# OS TOOLS
# ======================================================

if [ "$1" = "open" ]; then
    shift
    python3 "$SCRIPT_DIR/os_tools.py" open "$@"
    exit 0
fi

if [ "$1" = "find" ]; then
    shift
    python3 "$SCRIPT_DIR/os_tools.py" find "$@"
    exit 0
fi

if [ "$1" = "apps" ]; then
    python3 "$SCRIPT_DIR/os_tools.py" apps
    exit 0
fi

# ======================================================
# FILE EXTRACTION UTILITY
# ======================================================

temp_output=$(mktemp)

extract_file() {
    file="$1"
    echo "---- File: $file ----" >> "$temp_output"

    if [ ! -f "$file" ]; then
        echo "[ERROR] File not found" >> "$temp_output"
        return
    fi

    mime=$(file --mime-type -b "$file")

    case "$mime" in
        text/*)
            cat "$file" >> "$temp_output"
            ;;
        application/json)
            jq . "$file" 2>/dev/null >> "$temp_output" || cat "$file" >> "$temp_output"
            ;;
        application/pdf)
            python3 -c "
from rag.document_loader import load_document
docs = load_document('$file')
for doc in docs:
    print(doc.page_content)
" >> "$temp_output" 2>/dev/null || echo "[PDF - use 'eva ingest' for full parsing]" >> "$temp_output"
            ;;
        *)
            head -c 1000 "$file" >> "$temp_output"
            ;;
    esac

    echo -e "\n" >> "$temp_output"
}

# ======================================================
# DO MODE (EVA generates commands)
# ======================================================

if [ "$1" = "do" ]; then
    shift
    task="$1"
    shift

    for f in "$@"; do
        extract_file "$f"
    done

    full_prompt="Task: $task\n\nAttached file data:\n$(cat "$temp_output")\n\nReturn ONLY the final shell commands with no explanation."

    cmd=$(echo "$full_prompt" | python3 "$SCRIPT_DIR/eva_backend.py")

    echo -e "\nEVA recommends:\n$cmd"
    echo -en "\nRun these commands? (y/n): "
    read ans

    if [ "$ans" = "y" ]; then
        eval "$cmd"
    else
        echo "Cancelled."
    fi

    rm "$temp_output"
    exit 0
fi

# ======================================================
# HELP
# ======================================================

if [ "$1" = "help" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "EVA - Voice-Enabled AI Assistant"
    echo ""
    echo "Voice Commands:"
    echo "  eva voice              - Continuous voice mode (say 'EVA' to activate)"
    echo "  eva listen             - Single voice command"
    echo ""
    echo "Chat Commands:"
    echo "  eva chat               - Interactive text chat"
    echo ""
    echo "RAG Commands:"
    echo "  eva ingest <file|dir>  - Add to knowledge base"
    echo "  eva ask \"question\"     - Query knowledge base"
    echo "  eva status             - Show status"
    echo ""
    echo "OS Commands:"
    echo "  eva open <path>        - Open file/folder"
    echo "  eva find <query>       - Find files"
    echo "  eva apps               - List installed apps"
    echo ""
    echo "Other:"
    echo "  eva do \"task\" [files]  - Generate shell commands"
    echo "  eva \"question\"         - Direct LLM chat"
    echo ""
    exit 0
fi

# ======================================================
# NORMAL CHAT (NO FILES)
# ======================================================

question="$1"
shift

if [ "$#" -eq 0 ]; then
    echo "$question" | python3 "$SCRIPT_DIR/eva_backend.py"
    exit 0
fi

# ======================================================
# CHAT WITH FILES
# ======================================================

for f in "$@"; do
    extract_file "$f"
done

combined_prompt="$question\n\nAttached data:\n$(cat "$temp_output")"
echo "$combined_prompt" | python3 "$SCRIPT_DIR/eva_backend.py"

rm "$temp_output"
exit 0
