#!/usr/bin/env python3
import os, sys, subprocess

backend = os.path.expanduser("~/eva_rag/eva/eva_rag_query.py")

print("EVA RAG Chat Mode — type 'exit' to stop\n")

while True:
    msg = input("You: ")
    if msg.lower().strip() in ["exit", "quit"]:
        break

    process = subprocess.Popen(
        [backend, msg],
        stdout=subprocess.PIPE,
        text=True
    )
    answer = process.communicate()[0]
    print("\nEVA:", answer, "\n")
