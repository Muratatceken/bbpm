#!/bin/bash
# Download Hugging Face models for LLM experiments

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "Downloading Hugging Face models for exp07..."

# Check if transformers is available
if ! python -c "import transformers" 2>/dev/null; then
    echo "Error: transformers not installed"
    echo "Install with: pip install transformers"
    exit 1
fi

# Download tiny-gpt2 model
MODEL_NAME="sshleifer/tiny-gpt2"

echo "Downloading model: $MODEL_NAME"
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = '$MODEL_NAME'
print(f'Downloading {model_name}...')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f'Model downloaded successfully!')
print(f'Cache location: {os.path.expanduser(\"~/.cache/huggingface\")}')
"

echo ""
echo "Model download complete!"
