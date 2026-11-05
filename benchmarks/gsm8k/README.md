# GSM8K (vendored)

`test.jsonl` (all 1,319 test rows) and `train_few_shot.jsonl` (the first 8
train rows, the few-shot pool) are the `openai/gsm8k` `main` config, converted
unmodified from the parquet revision and files recorded in `SOURCE.json`.

Used by `scripts/spec_acceptance_gsm8k.py`, the project's standard speculative
acceptance test (see AGENTS.md and docs/QUALIFICATION.md).

License: MIT, Copyright (c) 2021 OpenAI
(<https://huggingface.co/datasets/openai/gsm8k>, Cobbe et al., "Training
Verifiers to Solve Math Word Problems", 2021).
