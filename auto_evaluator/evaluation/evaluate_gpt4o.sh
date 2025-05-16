#!/bin/bash

# 定义变量
CAPTIONING_FILE="/neurips_mmearth_benchmark/benchmark_data/captioning_sample.jsonl"
OPEN_FILE="/neurips_mmearth_benchmark/benchmark/results/msearth_open.json"
MCQ_FILE="/neurips_mmearth_benchmark/benchmark/results/msearth_mcq.json"
SAVE_DIR="/neurips_mmearth_benchmark/evaluation_result_closed_models"

# 定义模型列表
# MODELS=("gpt-4o-2024-11-20", "claude-3-5-haiku-latest", "claude-3-7-sonnet-latest","gpt-4o-mini", "gemini-2.5-flash-thinking","gemini-2.5-pro-thinking")
MODELS=("gemini-2.5-pro-thinking")
# 遍历模型并并行运行任务
for MODEL in "${MODELS[@]}"; do
    echo "Processing model: $MODEL"

    # Captioning 任务
    python evaluate_closed_models.py \
        --captioning-file "$CAPTIONING_FILE" \
        --open-file "$OPEN_FILE" \
        --mcq-file "$MCQ_FILE" \
        --save-dir "$SAVE_DIR" \
        --model "$MODEL" &

done

# 等待所有后台任务完成
wait

echo "All tasks completed!"
