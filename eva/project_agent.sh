#!/usr/bin/env bash

project_dir="$1"

if [ ! -d "$project_dir" ]; then
    echo "Folder not found."
    exit 1
fi

temp=$(mktemp)

echo "---- PROJECT STRUCTURE ----" >> "$temp"
tree "$project_dir" >> "$temp"

echo -e "\n\n---- IMPORTANT FILES ----" >> "$temp"

# capture key files safely
for f in $(find "$project_dir" -maxdepth 3 -type f \
          -name "*.py" -o -name "*.js" -o -name "*.json" -o -name "*.md" \
          -o -name "Dockerfile" -o -name "requirements.txt"); do
    echo -e "\n---- $f ----" >> "$temp"
    head -c 3000 "$f" >> "$temp"
done

cat "$temp" | ~/eva_rag/eva/eva_backend.py

rm "$temp"
