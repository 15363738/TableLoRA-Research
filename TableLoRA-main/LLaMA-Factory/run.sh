#!/bin/bash

# === 配置区域 ===
export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="/model_and_data/model_data/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR="saves/llama3-8b/lora/table_lora_wikitq"

# === 启动训练 ===
echo "Starting TableLoRA training..."

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path "$MODEL_PATH" \
    --dataset wikitq_train \
    --template llama3 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 200 \
    --learning_rate 5e-6 \
    --num_train_epochs 3.0 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --cutoff_len 1000 \
    --fp16 \
    --emb_lora True \
    --packing False \
    --remove_unused_columns False
    # --max_samples 1000
