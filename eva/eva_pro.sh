#!/usr/bin/env bash

# EVA PRO MAX — Local AI agent using Ollama + phi3:mini
# -----------------------------------------------------
# FINAL VERSION — Synchronized with new local RAG system

# ======================================================
# 1. COMMAND ROUTING  (no backend folder required)
# ======================================================

# PDF / File Ingestion → Local RAG
if [ "$1" = "ingest" ]; then
    shift
    python3 ~/eva_rag/pdf_ingest.py "$@"
    exit 0
fi

# URL Scrape → Local RAG
if [ "$1" = "scrape" ]; then
    shift
    python3 ~/eva_rag/scrape_ingest.py "$@"
    exit 0
fi

# Ask RAG Query
if [ "$1" = "ask" ]; then
    shift
    python3 ~/eva_rag/rag_query.py "$@"
    exit 0
fi

# Memory commands
if [ "$1" = "remember" ]; then
    shift
    ~/eva_rag/eva/memory.py add "$@"
    exit 0
fi

if [ "$1" = "memory" ]; then
    ~/eva_rag/eva/memory.py read
    exit 0
fi

if [ "$1" = "forget" ]; then
    ~/eva_rag/eva/memory.py clear
    exit 0
fi

# Web Search (local or API-based)
if [ "$1" = "search" ]; then
    shift
    ~/eva_rag/eva/web_search.py "$@"
    exit 0
fi

# Project Analyzer
if [ "$1" = "project" ]; then
    shift
    ~/eva_rag/eva/project_agent.sh "$@"
    exit 0
fi

# ======================================================
# 2. FILE EXTRACTION UTILITY
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
            echo "[TEXT]" >> "$temp_output"
            cat "$file" >> "$temp_output"
            ;;
        application/json)
            echo "[JSON]" >> "$temp_output"
            jq . "$file" >> "$temp_output"
            ;;
        application/zip)
            echo "[ZIP CONTENTS]" >> "$temp_output"
            unzip -l "$file" >> "$temp_output"
            ;;
        application/pdf)
            echo "[PDF EXTRACTED]" >> "$temp_output"
            python3 - <<EOF >> "$temp_output"
import pdfplumber
with pdfplumber.open("$file") as pdf:
    for page in pdf.pages:
        print(page.extract_text() or "")
EOF
            ;;
        image/*)
            echo "[IMAGE OCR]" >> "$temp_output"
            python3 - <<EOF >> "$temp_output"
from PIL import Image
import pytesseract
print(pytesseract.image_to_string(Image.open("$file")))
EOF
            ;;
        application/vnd.openxmlformats-officedocument.spreadsheetml.sheet)
            echo "[EXCEL]" >> "$temp_output"
            python3 - <<EOF >> "$temp_output"
import pandas as pd
df = pd.read_excel("$file")
print(df.to_string())
EOF
            ;;
        text/csv)
            echo "[CSV]" >> "$temp_output"
            python3 - <<EOF >> "$temp_output"
import pandas as pd
df = pd.read_csv("$file")
print(df.to_string())
EOF
            ;;
        application/x-ipynb+json)
            echo "[IPYNB]" >> "$temp_output"
            python3 - <<EOF >> "$temp_output"
import nbformat
nb = nbformat.read("$file", as_version=4)
for i, cell in enumerate(nb.cells):
    print(f"\n--- Cell {i} ({cell.cell_type}) ---")
    print(cell.source)
EOF
            ;;
        *)
            echo "[UNKNOWN FORMAT - HEX PREVIEW]" >> "$temp_output"
            head -c 256 "$file" | hexdump -C >> "$temp_output"
            ;;
    esac

    echo -e "\n\n" >> "$temp_output"
}

# ======================================================
# 3. DO MODE (EVA generates commands)
# ======================================================

if [ "$1" = "do" ]; then
    shift
    task="$1"
    shift

    for f in "$@"; do
        extract_file "$f"
    done

    full_prompt="Task: $task\n\nAttached file data:\n$(cat "$temp_output")\n\nReturn ONLY the final shell commands with no explanation."

    cmd=$(echo "$full_prompt" | ~/eva_rag/eva/eva_backend.py)

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
# 4. NORMAL CHAT (NO FILES)
# ======================================================

question="$1"
shift

if [ "$#" -eq 0 ]; then
    echo "$question" | ~/eva_rag/eva/eva_backend.py
    exit 0
fi

# ======================================================
# 5. NORMAL CHAT WITH FILES
# ======================================================

for f in "$@"; do
    extract_file "$f"
done

combined_prompt="$question\n\nAttached data:\n$(cat "$temp_output")"
echo "$combined_prompt" | ~/eva_rag/eva/eva_backend.py

rm "$temp_output"
exit 0
