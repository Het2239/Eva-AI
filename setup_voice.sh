#!/bin/bash
# EVA Voice Setup Script
# Downloads Piper TTS binary and voice model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVA_DIR="$SCRIPT_DIR/eva"

echo "EVA Voice Setup"
echo "==============="

# Create directories
mkdir -p "$EVA_DIR/bin"
mkdir -p "$EVA_DIR/models/piper"

# Download Piper binary if not exists
if [ ! -f "$EVA_DIR/bin/piper" ]; then
    echo "Downloading Piper TTS binary..."
    wget -q -O /tmp/piper.tar.gz https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz
    tar -xzf /tmp/piper.tar.gz -C "$EVA_DIR/bin/"
    mv "$EVA_DIR/bin/piper/"* "$EVA_DIR/bin/"
    rmdir "$EVA_DIR/bin/piper"
    rm /tmp/piper.tar.gz
    chmod +x "$EVA_DIR/bin/piper"
    echo "✓ Piper binary installed"
else
    echo "✓ Piper binary already exists"
fi

# Download voice model if not exists
MODEL_NAME="en_US-amy-medium"
if [ ! -f "$EVA_DIR/models/piper/$MODEL_NAME.onnx" ]; then
    echo "Downloading voice model ($MODEL_NAME)..."
    wget -q -O "$EVA_DIR/models/piper/$MODEL_NAME.onnx" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/$MODEL_NAME.onnx"
    wget -q -O "$EVA_DIR/models/piper/$MODEL_NAME.onnx.json" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/$MODEL_NAME.onnx.json"
    echo "✓ Voice model installed"
else
    echo "✓ Voice model already exists"
fi

echo ""
echo "Setup complete! You can now use 'eva voice'"
