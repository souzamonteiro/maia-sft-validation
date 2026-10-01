# Stage 2 — Multilingual Control

Purpose: test whether the validated response-only SFT pipeline can teach GPT-2 Medium
controlled instruction following in English, Portuguese, and Spanish.

This stage deliberately tests in-family multilingual generalization, not held-out-family
reasoning. Stage 3 is reserved for robust/held-out-family generalization.

## Dataset
1,500 examples total; 500 per language. Five task families per language:
uppercase, lowercase, parity, addition, comparison.

Split: 1,200 train / 150 validation / 150 test. Each split is stratified by language
and family. Responses vary; binary tasks are balanced by construction.

## Run
Overlay this package on the existing `maia-sft-validation` repository, then:

```bash
chmod +x scripts/run_stage2.sh
./scripts/run_stage2.sh
```

The runner audits leakage/masking, evaluates the untouched GPT-2 Medium baseline,
trains for three epochs, selects the best checkpoint by validation loss, evaluates
that checkpoint, and archives the experiment when the existing archiver is present.
