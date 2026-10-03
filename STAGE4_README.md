# Stage 4 — Full Assistant SFT & Deployment Control

This is the end-to-end rehearsal for Maia Lite Instruct using official GPT-2 Medium (~355M).

Data sources:
- OpenAssistant OASST2: multilingual/multi-turn assistant conversations (Apache-2.0).
- Databricks Dolly 15K: human-written instruction following (CC-BY-SA-3.0).

Design:
- EN/PT/ES where available.
- Exact prompt leakage audit.
- Full-parameter response-only SFT; every assistant turn is supervised.
- Best checkpoint by validation loss.
- Frozen SHA-256 benchmark: 180 objective + 120 qualitative prompts.
- Baseline and post-SFT evaluation.
- F16 GGUF reference plus Q8_0 and Q4_K_M.
- Ollama Modelfile and deployment smoke test.

Run:
```bash
chmod +x scripts/run_stage4.sh scripts/convert_stage4_gguf.sh scripts/install_ollama_stage4.sh
./scripts/run_stage4.sh
```

Review `outputs/stage4-metrics.json` and especially `outputs/stage4-post-results.jsonl`.

Then:
```bash
./scripts/convert_stage4_gguf.sh
./scripts/install_ollama_stage4.sh
```

The GGUF conversion uses current llama.cpp. If its Python dependencies conflict with the Stage 4 venv, create a dedicated llama.cpp Python environment and set `PYTHON=/path/to/python`.

For the first definitive Maia Lite SFT, keep this pipeline frozen and replace the base model/tokenizer and identity/deployment names only.
