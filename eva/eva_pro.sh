#!/usr/bin/env bash

# EVA PRO — AI Assistant with RAG
# ================================
# Commands:
#   eva ingest <file|dir>  - Ingest documents to RAG
#   eva ask "question"     - Query RAG knowledge base
#   eva chat               - Interactive RAG chat
#   eva status             - Show RAG status
#   eva remember <text>    - Add to memory
#   eva memory             - Show memory
#   eva forget             - Clear memory
#   eva do "task" [files]  - Generate commands
#   eva "question"         - Direct LLM chat
#   eva "question" [files] - Chat with file context

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_DIR="$SCRIPT_DIR/../rag"

# ======================================================
# RAG COMMANDS
# ======================================================

# Ingest documents to RAG
if [ "$1" = "ingest" ]; then
    shift
    python3 "$RAG_DIR/rag_pipeline.py" ingest "$@"
    exit 0
fi

# Ask RAG query
if [ "$1" = "ask" ]; then
    shift
    python3 "$RAG_DIR/rag_pipeline.py" ask "$@"
    exit 0
fi

# Interactive RAG chat
if [ "$1" = "chat" ]; then
    python3 "$RAG_DIR/rag_pipeline.py" chat
    exit 0
fi

# RAG status
if [ "$1" = "status" ]; then
    python3 "$RAG_DIR/rag_pipeline.py" status
    exit 0
fi

# ======================================================
# MEMORY COMMANDS
# ======================================================

if [ "$1" = "remember" ]; then
    shift
    python3 "$SCRIPT_DIR/memory.py" add "$@"
    exit 0
fi

if [ "$1" = "memory" ]; then
    python3 "$SCRIPT_DIR/memory.py" read
    exit 0
fi

if [ "$1" = "forget" ]; then
    python3 "$SCRIPT_DIR/memory.py" clear
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
    echo "EVA - AI Assistant with RAG"
    echo ""
    echo "Usage:"
    echo "  eva ingest <file|dir>  - Ingest documents to knowledge base"
    echo "  eva ask \"question\"     - Query knowledge base"
    echo "  eva chat               - Interactive RAG chat"
    echo "  eva status             - Show RAG status"
    echo "  eva remember <text>    - Add to memory"
    echo "  eva memory             - Show memory"
    echo "  eva forget             - Clear memory"
    echo "  eva do \"task\" [files]  - Generate shell commands"
    echo "  eva \"question\"         - Direct LLM chat"
    echo "  eva \"question\" [files] - Chat with file context"
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
