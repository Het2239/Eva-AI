#!/usr/bin/env python3
import sys
import subprocess
import os
# Read the prompt from stdin
template_file = os.path.expanduser("~/eva_rag/eva/prompt_template.txt")
template = ""

if os.path.exists(template_file):
    with open(template_file) as f:
        template = f.read() + "\n\n"

prompt = template + sys.stdin.read().strip()

# Send prompt to phi3:mini through Ollama
process = subprocess.Popen(
    ["ollama", "run", "phi3:mini"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

out, err = process.communicate(prompt)
print(out)
