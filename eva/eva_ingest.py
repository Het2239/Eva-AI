#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.expanduser("~/eva_rag/backend/llamaindex"))

from rag_engine import ingest_path

for p in sys.argv[1:]:
    ingest_path(os.path.abspath(p))
