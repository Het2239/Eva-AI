#!/usr/bin/env python3
import requests, sys, subprocess
from bs4 import BeautifulSoup
from markdownify import markdownify as md

url = sys.argv[1]
r = requests.get(url, timeout=8)

soup = BeautifulSoup(r.text, "html.parser")
for tag in soup(["script", "style", "noscript"]):
    tag.decompose()

markdown = md(r.text)

prompt = f"Read and summarize this webpage in detail:\nURL: {url}\n\nContent:\n{markdown[:50000]}"

process = subprocess.Popen(
    ["ollama", "run", "phi3:mini"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True
)

print(process.communicate(prompt)[0])
