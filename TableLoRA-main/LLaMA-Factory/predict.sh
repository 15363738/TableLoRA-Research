#!/bin/bash

# === 配置区域 ===
export CUDA_VISIBLE_DEVICES=0

# 模型和权重路径 (请根据您实际训练保存的路径修改)
MODEL_PATH="/model_and_data/model_data/Meta-Llama-3-8B-Instruct"
# 如果您训练时使用了 saves/llama3-8b/lora/table_lora_wikitq，请确保这里一致
ADAPTER_PATH="/论文1lab/TableLoRA-main/LLaMA-Factory/saves/llama3-8b/lora/table_lora_wikitq/checkpoint-2121"

# 输出目录
OUTPUT_DIR="saves/test/llama3-8b/lora/table_lora_wikitq/prediction"

# === 启动预测 ===
echo "Starting TableLoRA prediction..."

llamafactory-cli train \
    --stage sft \
    --do_predict \
    --model_name_or_path "$MODEL_PATH" \
    --adapter_name_or_path "$ADAPTER_PATH" \
    --dataset wikitq_test \
    --eval_dataset wikitq_test \
    --template llama3 \
    --finetuning_type lora \
    --output_dir "$OUTPUT_DIR" \
    --per_device_eval_batch_size 2 \
    --predict_with_generate \
    --fp16 \
    --emb_lora True \
    --packing False \
    --remove_unused_columns False

echo "Prediction finished! Results saved to $OUTPUT_DIR"