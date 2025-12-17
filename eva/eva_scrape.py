#!/usr/bin/env python3
import sys, os

path = os.path.expanduser("~/eva_rag/scrape_ingest.py")
os.execv("/usr/bin/python3", ["python3", path] + sys.argv[1:])
