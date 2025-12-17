#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.expanduser("~/eva_rag/backend/llamaindex"))

from rag_engine import ask_query

query = " ".join(sys.argv[1:])
print(ask_query(query))
