#!/usr/bin/env python3
import sys, requests, subprocess

query = " ".join(sys.argv[1:])
url = f"https://google.com/?q={query}&ia=web"

results = requests.get(
    "https://api.google.com/",
    params={"q": query, "format": "json"}
).json()

summary = ""

if "RelatedTopics" in results:
    for r in results["RelatedTopics"][:5]:
        if "Text" in r:
            summary += "- " + r["Text"] + "\n"

process = subprocess.Popen(
    ["ollama", "run", "phi3:mini"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True
)

prompt = f"Summarize these search results:\n\n{summary}"
out = process.communicate(prompt)[0]

print(out)
