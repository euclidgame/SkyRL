set -x

# IMO AnswerBench training with "boxed extraction -> LLM judge equivalence".
#
# 1) Prepare data:
#   uv run examples/imo_llm_judge/imo_dataset_judge.py \
#     --train_jsonl /accounts/projects/berkeleynlp/windsey/projects/tinker-cookbook/training_data_imo/answerbench-hard32-trainset.jsonl \
#     --val_jsonl   /accounts/projects/berkeleynlp/windsey/projects/tinker-cookbook/training_data_imo/answerbench-hard32.jsonl \
#     --output_dir  $HOME/data/imo_llm_judge
#
# 2) Add OPENAI_API_KEY + WANDB_API_KEY to .env.imo_judge (optional: base_url).
# 3) Run:
#   bash examples/imo_llm_judge/run_imo_llm_judge.sh

# =============================================================================
# Configuration Variables
# =============================================================================
DATA_DIR="$HOME/data/imo_llm_judge"
CKPT_PATH="./ckpts/imo_llm_judge_ckpt"

NUM_GPUS=4
NUM_INFERENCE_ENGINES=1
TP_SIZE=4
LOGGER=wandb

# =============================================================================
# Environment File (for API keys)
# =============================================================================
ENV_FILE=".env.imo_judge"
ENV_FILE_ARGS=()
if [[ -f "$ENV_FILE" ]]; then
  ENV_FILE_ARGS=(--env-file "$ENV_FILE")
else
  echo "Warning: $ENV_FILE not found; using exported environment variables instead." 1>&2
fi

# =============================================================================
# Run Training
# =============================================================================
# GPT-OSS: flash-attn is incompatible with attention sinks; SkyRL patches GPT-OSS
# to use FlexAttention automatically when transformers>=4.56.2.
#
# Arguments are grouped as follows:
#   - Data
#   - Trainer: Model (path, LoRA, optimizer, flash_attn)
#   - Trainer: Placement & Strategy (FSDP, GPUs)
#   - Trainer: Algorithm (GRPO, KL loss)
#   - Trainer: Training Params (epochs, batch sizes)
#   - Trainer: Evaluation
#   - Trainer: Checkpointing & Logging
#   - Generator: Engine Config (vLLM, TP, memory)
#   - Generator: Sampling Params (temperature, top_p)
#   - Generator: Debug
#   - Environment (LLM judge config)
# =============================================================================

uv run --isolated --extra vllm "${ENV_FILE_ARGS[@]}" -m examples.imo_llm_judge.main_imo_llm_judge \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.policy.model.path="unsloth/gpt-oss-20b-BF16" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.flash_attn=false \
  trainer.use_sample_packing=false \
  trainer.strategy=fsdp2 \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=true \
  trainer.epochs=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=8 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=2048 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.eval_batch_size=32 \
  trainer.ckpt_interval=10 \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.export_path="./exports" \
  trainer.resume_mode=null \
  trainer.logger="$LOGGER" \
  trainer.project_name="imo_llm_judge" \
  trainer.run_name="gptoss_20b_imo_judge" \
  trainer.dump_data_batch=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  generator.gpu_memory_utilization=0.6 \
  generator.async_engine=true \
  generator.batched=false \
  generator.enforce_eager=true \
  +generator.chat_template_kwargs={reasoning_effort:'low'} \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=8 \
  generator.sampling_params.max_generate_length=8192 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.debug_log_rendered_prompt=true \
  generator.debug_log_rendered_prompt_every_n_steps=1 \
  generator.debug_log_rendered_prompt_max_per_step=1 \
  generator.debug_log_rendered_prompt_max_chars=2000 \
  environment.env_class=imo_llm_judge \
  environment.skyrl_gym.imo_llm_judge.model="gpt-4o" \
  environment.skyrl_gym.imo_llm_judge.require_boxed=true \
  $@