#!/usr/bin/env python3
import json, os, sys

memory_file = os.path.expanduser("~/eva_rag/eva/memory.json")

# create file if missing
if not os.path.exists(memory_file):
    with open(memory_file, "w") as f:
        json.dump({"memory": []}, f)

with open(memory_file, "r") as f:
    data = json.load(f)

cmd = sys.argv[1]

if cmd == "add":
    text = " ".join(sys.argv[2:])
    data["memory"].append(text)
    with open(memory_file, "w") as f:
        json.dump(data, f, indent=2)
    print("✓ Added to EVA memory.")

elif cmd == "read":
    print("\n".join(data["memory"]))

elif cmd == "clear":
    data["memory"] = []
    with open(memory_file, "w") as f:
        json.dump(data, f, indent=2)
    print("✓ Memory cleared.")
