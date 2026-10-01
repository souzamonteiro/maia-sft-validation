# Maia SFT Validation

A controlled supervised fine-tuning validation project for the **GPT-2 Medium (355M)** architecture, designed as a positive-control experiment before applying the same SFT methodology to **Maia Lite 355M**.

## Scientific objective

The project answers one question before Maia Lite instruction tuning begins:

> Can our SFT pipeline reliably transform a known GPT-2 Medium causal language model into a response-following model while computing loss only on assistant response tokens?

The first experiment intentionally overfits a tiny deterministic dataset. This is a pipeline validation test, not a model-quality benchmark.

## Reference model

- `openai-community/gpt2-medium`
- 354,823,168 parameters
- 24 Transformer blocks
- 16 attention heads
- hidden size 1024
- context length 1024
- vocabulary 50,257

## Design principles

1. Use the official GPT-2 Medium model and tokenizer as the positive control.
2. Perform full-parameter SFT; no LoRA or QLoRA in the baseline.
3. Compute loss only on assistant response tokens.
4. Mask prompt and padding labels with `-100`.
5. Append EOS to every supervised response.
6. Use fixed-length tensors to keep Apple MPS graph shapes stable.
7. Audit tokenization and labels before training.
8. Require a tiny-dataset overfit before scaling to a real SFT corpus.
9. Keep the pipeline simple enough to port to Maia Lite with minimal changes.

## Repository layout

```text
maia-sft-validation/
├── data/
│   ├── micro_train.jsonl
│   └── micro_eval.jsonl
├── outputs/
├── scripts/
│   ├── setup_mac.sh
│   └── run_micro_sft.sh
├── src/
│   ├── audit.py
│   ├── common.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── preflight.py
│   └── train.py
├── tests/
│   └── test_masking.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## macOS / Apple Silicon setup

Requires an Apple Silicon Mac with a PyTorch build that supports MPS.

```bash
git clone <YOUR-REPOSITORY-URL>
cd maia-sft-validation
./scripts/setup_mac.sh
```

The preflight downloads GPT-2 Medium and verifies the exact expected parameter count and vocabulary size.

## Experiment 1 — response-only micro-overfit

Run:

```bash
./scripts/run_micro_sft.sh
```

This performs, in order:

1. response-only label audit;
2. automated masking test;
3. full-parameter GPT-2 Medium SFT on 64 deterministic examples;
4. deterministic generation-based evaluation on the training set.

The training examples intentionally associate numbered test items with unique strings such as `MAIA-0007`. A successful run should show a substantial training-loss decrease and progressively learn these arbitrary associations.

## Manual label audit

Run independently with:

```bash
python -m src.audit --data data/micro_train.jsonl --examples 5
```

For every example, the following invariant must hold:

```text
prompt tokens   -> label = -100
response tokens -> label = token id
EOS             -> label = EOS token id
padding          -> label = -100
```

Training must not proceed if this invariant fails.

## Manual generation

```bash
python -m src.generate \
  --model outputs/gpt2-medium-micro-sft \
  --prompt "Return the validation code associated with test item 7."
```

## Interpretation

### PASS

The experiment passes when the model can deliberately overfit the tiny dataset: loss falls substantially and trained prompts produce their target responses with high reliability.

This establishes a positive control for the SFT mechanics: tokenization, prompt formatting, response masking, gradient updates, EOS handling, checkpoint serialization, and generation.

### FAIL

A failure is scientifically useful. Do not compensate by immediately changing many hyperparameters. Inspect, in this order:

1. label mask;
2. prompt/response boundary;
3. EOS handling;
4. decoded supervised tokens;
5. gradients and optimizer updates;
6. MPS-specific errors or CPU fallback;
7. learning rate and number of updates.

## Experiment sequence

```text
GPT-2 Medium official
        |
        v
64-example response-only overfit
        |
        +---- FAIL -> debug SFT pipeline
        |
        v PASS
1k-5k clean instruction SFT
        |
        v
freeze validated SFT pipeline
        |
        v
Maia Lite 355M Base
        |
        v
same micro-overfit protocol
        |
        v
Maia Lite 355M Instruct SFT
```

## Why not Axolotl yet?

The goal of this repository is not convenience; it is to establish a transparent reference implementation. Once this baseline is validated, an Axolotl experiment can be added as an independent implementation and compared against the same dataset and evaluation protocol.

## Notes on Apple MPS

The scripts enable `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported MPS operations may fall back to CPU. Fixed-length sequences are used deliberately to avoid producing many different tensor shapes during training.

Start conservatively with sequence length 256 and micro-batch size 1. The experiment is designed for correctness before throughput.

## License

Apache License 2.0.
