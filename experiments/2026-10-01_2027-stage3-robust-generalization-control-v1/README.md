# stage3-robust-generalization-control-v1

## Result

**PASS**

## Metrics

- **stage:** stage3-robust-generalization-control-v1
- **model:** openai-community/gpt2-medium
- **epochs:** 3
- **optimizer_updates:** 300
- **initial_validation_loss:** 3.854910583496094
- **best_validation_loss:** 0.5687138036408578
- **history:** [{'epoch': 1, 'update': 100, 'validation_loss': 0.5758585763943848}, {'epoch': 2, 'update': 200, 'validation_loss': 0.5687138036408578}, {'epoch': 3, 'update': 300, 'validation_loss': 0.570687946218386}]
- **baseline_core:** {'model': 'openai-community/gpt2-medium', 'split': 'test_core', 'overall': {'examples': 100, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}, 'by_family': {'arithmetic': {'examples': 25, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}, 'comparison': {'examples': 25, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}, 'string_length': {'examples': 25, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}, 'string_transform': {'examples': 25, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}}, 'by_variant': {'addition': {'examples': 25, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}, 'length': {'examples': 25, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}, 'less_than': {'examples': 25, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}, 'lowercase': {'examples': 13, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}, 'uppercase': {'examples': 12, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}}}
- **post_core:** {'model': 'outputs/gpt2-medium-stage3-best', 'split': 'test_core', 'overall': {'examples': 100, 'exact_match': 45, 'exact_match_rate': 0.45, 'eos_count': 100, 'eos_rate': 1.0}, 'by_family': {'arithmetic': {'examples': 25, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 25, 'eos_rate': 1.0}, 'comparison': {'examples': 25, 'exact_match': 12, 'exact_match_rate': 0.48, 'eos_count': 25, 'eos_rate': 1.0}, 'string_length': {'examples': 25, 'exact_match': 8, 'exact_match_rate': 0.32, 'eos_count': 25, 'eos_rate': 1.0}, 'string_transform': {'examples': 25, 'exact_match': 25, 'exact_match_rate': 1.0, 'eos_count': 25, 'eos_rate': 1.0}}, 'by_variant': {'addition': {'examples': 25, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 25, 'eos_rate': 1.0}, 'length': {'examples': 25, 'exact_match': 8, 'exact_match_rate': 0.32, 'eos_count': 25, 'eos_rate': 1.0}, 'less_than': {'examples': 25, 'exact_match': 12, 'exact_match_rate': 0.48, 'eos_count': 25, 'eos_rate': 1.0}, 'lowercase': {'examples': 13, 'exact_match': 13, 'exact_match_rate': 1.0, 'eos_count': 13, 'eos_rate': 1.0}, 'uppercase': {'examples': 12, 'exact_match': 12, 'exact_match_rate': 1.0, 'eos_count': 12, 'eos_rate': 1.0}}}
- **baseline_heldout:** {'model': 'openai-community/gpt2-medium', 'split': 'test_heldout', 'overall': {'examples': 200, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}, 'by_family': {'arithmetic': {'examples': 50, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}, 'comparison': {'examples': 50, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}, 'string_length': {'examples': 50, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}, 'string_transform': {'examples': 50, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}}, 'by_variant': {'arithmetic_subtraction': {'examples': 50, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}, 'comparison_greater_than': {'examples': 50, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}, 'string_title_case': {'examples': 50, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}, 'string_vowel_count': {'examples': 50, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 0, 'eos_rate': 0.0}}}
- **post_heldout:** {'model': 'outputs/gpt2-medium-stage3-best', 'split': 'test_heldout', 'overall': {'examples': 200, 'exact_match': 27, 'exact_match_rate': 0.135, 'eos_count': 200, 'eos_rate': 1.0}, 'by_family': {'arithmetic': {'examples': 50, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 50, 'eos_rate': 1.0}, 'comparison': {'examples': 50, 'exact_match': 25, 'exact_match_rate': 0.5, 'eos_count': 50, 'eos_rate': 1.0}, 'string_length': {'examples': 50, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 50, 'eos_rate': 1.0}, 'string_transform': {'examples': 50, 'exact_match': 2, 'exact_match_rate': 0.04, 'eos_count': 50, 'eos_rate': 1.0}}, 'by_variant': {'arithmetic_subtraction': {'examples': 50, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 50, 'eos_rate': 1.0}, 'comparison_greater_than': {'examples': 50, 'exact_match': 25, 'exact_match_rate': 0.5, 'eos_count': 50, 'eos_rate': 1.0}, 'string_title_case': {'examples': 50, 'exact_match': 2, 'exact_match_rate': 0.04, 'eos_count': 50, 'eos_rate': 1.0}, 'string_vowel_count': {'examples': 50, 'exact_match': 0, 'exact_match_rate': 0.0, 'eos_count': 50, 'eos_rate': 1.0}}}
- **core_exact_match_gain:** 0.45
- **heldout_exact_match_gain:** 0.135
- **result:** PASS
- **interpretation_note:** Inspect per-family metrics; PASS alone is not proof of robust reasoning.

## Environment

- Python: `3.14.5`
- PyTorch: `2.14.1`
- Transformers: `5.17.0`
- Platform: `macOS-27.0.1-arm64-arm-64bit-Mach-O`
- Machine: `arm64`
- MPS built: `True`
- MPS available: `True`

## Git

- Commit: `d7953bd999bfc1f2b94e5914692bea5b63b437ad`
- Branch: `main`
- Dirty working tree: `True`

## Reproducibility

All copied artifacts are listed in `manifest.json` with SHA-256 hashes.
