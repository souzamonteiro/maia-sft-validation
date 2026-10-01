# Stage 1 V2 — Generalization Control

Corrected replacement for the discarded Stage 1 dataset.

- 1,000 unique examples
- 800 train / 100 validation / 100 test
- deterministic per-category split
- zero duplicate IDs across splits
- zero duplicate instructions across splits
- zero duplicate instruction/response pairs across splits
- response-only masking audit before training
- GPT-2 Medium baseline on frozen test set
- full-parameter SFT on Apple MPS
- post-SFT evaluation on the identical frozen test set
- automatic archive using the repository's existing archive_experiment.py

Run:

    ./scripts/run_stage1.sh
