import platform
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import MODEL_ID


def main():
    print("=" * 80)
    print("MAIA SFT VALIDATION - PREFLIGHT")
    print("=" * 80)
    print("Python:", sys.version.split()[0])
    print("macOS:", platform.mac_ver()[0] or "not macOS")
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    params = sum(p.numel() for p in model.parameters())
    print("Model:", MODEL_ID)
    print("Parameters:", f"{params:,}")
    print("Vocabulary:", len(tokenizer))
    print("Context:", model.config.n_positions)
    assert params == 354_823_168, f"Unexpected parameter count: {params}"
    assert len(tokenizer) == 50_257
    print("PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
