### IMO AnswerBench + LLM-as-a-judge (boxed extraction → equivalence judge)

This example mirrors the common pattern:

- **Step 1**: extract the final `\\boxed{...}` answer locally (cheap / deterministic)
- **Step 2**: call an LLM judge to decide whether the extracted answer is mathematically equivalent to the ground truth

#### Prepare dataset

Convert your Tinker AnswerBench JSONL into SkyRL parquet format:

```bash
uv run examples/imo_llm_judge/imo_dataset_judge.py \
  --train_jsonl /accounts/projects/berkeleynlp/windsey/projects/tinker-cookbook/training_data_imo/answerbench-hard32-trainset.jsonl \
  --val_jsonl   /accounts/projects/berkeleynlp/windsey/projects/tinker-cookbook/training_data_imo/answerbench-hard32.jsonl \
  --output_dir  $HOME/data/imo_llm_judge
```

#### Run training

Set `OPENAI_API_KEY` in your environment (and `WANDB_API_KEY` if using W&B), then:

```bash
bash examples/imo_llm_judge/run_imo_llm_judge.sh
```

#### Removing the default GPT-OSS system prompt

Some GPT-OSS tokenizer chat templates inject a default system message. To remove it, override the chat template:

```bash
bash examples/imo_llm_judge/run_imo_llm_judge.sh \
  generator.chat_template.source=file \
  generator.chat_template.name_or_path=/data/windsey/SkyRL/skyrl-train/skyrl_train/utils/templates/gpt_oss_no_system.jinja2
```

#### Notes

- The env is registered at runtime by `examples/imo_llm_judge/main_imo_llm_judge.py` (no changes to `skyrl-gym` required).
- The environment id is `imo_llm_judge`.
