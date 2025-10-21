#!/bin/bash
set -e

# ==========================
# KG-Infused RAG DPO Training
# ==========================

# export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Model
export LLM_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
# export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
LLM_MODEL_NAME=$(basename "$LLM_MODEL")


# Training data files
INPUT_FILES=(
  # from Llama
  "../data/dpo_training_data/2wikimqa/target_kg_guided_knowledge_augmentation_round_0_to_1/LLaMa3.1-8B_num_samples_6_score_model_gpt-4o-mini-2024-07-18_index_0_1500_0_1500.json"
)


for INPUT_FILE in "${INPUT_FILES[@]}"; do

  # Build a tidy run name by stripping a known prefix/suffix
  TRIMMED_PATH="${INPUT_FILE#*/dpo_training_data/}"
  filename="${TRIMMED_PATH%.json}"
  echo "filename: $filename"

  # Output directories
  OUTPUT_DIR="../models/$LLM_MODEL_NAME/$filename"
  mkdir -p "$OUTPUT_DIR"

  # Logs
  LOG_DIR="../logs/running_log/$LLM_MODEL_NAME/kg_infused_rag/${filename}"
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/$(date +'%Y-%m-%d_%H-%M-%S').txt"
  echo "LOG_FILE: $LOG_FILE"
  exec > "$LOG_FILE" 2>&1

  TRAINING_LOG_DIR="../logs/training_log/kg_infused_rag/$LLM_MODEL_NAME/$filename"
  mkdir -p "$TRAINING_LOG_DIR"

  # Training
  echo "==========================="
  echo "Start training on $INPUT_FILE"
  echo "==========================="

  master_port=7428
  torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=$master_port \
    --node_rank=0 \
    ./train.py \
      --model_name_or_path $LLM_MODEL \
      --dataset_name $INPUT_FILE \
      --trust_remote_code \
      --max_length 8192 \
      --max_prompt_length 8000 \
      --output_dir $OUTPUT_DIR \
      --save_steps 50 \
      --save_total_limit 2 \
      --gradient_accumulation_steps 2 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --learning_rate 5e-7 \
      --logging_strategy steps \
      --logging_steps 20 \
      --logging_dir $TRAINING_LOG_DIR \
      --bf16 True \
      --num_train_epochs 1 \
      --report_to "tensorboard" \
      --save_only_model \
      --gradient_checkpointing \
      --deepspeed ./config/ds_config_zero3.json

  echo "Finished training on $INPUT_FILE"
  echo ""
done
