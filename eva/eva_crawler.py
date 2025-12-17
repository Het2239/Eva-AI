#!/usr/bin/env python3
import sys
import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# add backend modules
backend_path = os.path.expanduser("~/eva_rag/backend")
sys.path.append(backend_path)
from rag import ingest_url

visited = set()

def crawl(url, depth=2):
    if depth == 0 or url in visited:
        return

    visited.add(url)
    print(f"Crawling: {url}")
    try:
        ingest_url(url)
    except:
        pass

    try:
        r = requests.get(url, timeout=5)
    except:
        return

    soup = BeautifulSoup(r.text, "html.parser")
    base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(url))

    for link in soup.find_all('a', href=True):
        new = urljoin(base, link['href'])
        if base in new:
            crawl(new, depth - 1)

if __name__ == "__main__":
    crawl(sys.argv[1], depth=3)
    print("✓ Auto-crawl complete.")
