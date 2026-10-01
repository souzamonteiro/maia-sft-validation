# Stage 3 — Robust Generalization Control

Tests separately:
- in-family generalization on 100 unseen instances;
- transfer to 200 examples from held-out task variants never seen in training/validation.

Held-out variants: addition→subtraction, less-than→greater-than, upper/lower→title case, string length→vowel count.

Run:
```bash
chmod +x scripts/run_stage3.sh
./scripts/run_stage3.sh
```

Interpret aggregate PASS cautiously; per-family held-out metrics are the main scientific result.
