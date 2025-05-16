#!/bin/bash

# 定义 Python 脚本路径
PYTHON_SCRIPT="llm_metric.py"

# 定义 Captioning 任务文件列表
CAPTIONING_FILES=(
    "/evaluation_result_closed_models/captioning_results_claude-3-5-haiku-latest.jsonl"
    "/evaluation_result_closed_models/captioning_results_claude-3-7-sonnet-latest.jsonl"
    "/evaluation_result_closed_models/captioning_results_deepseek-vl2.jsonl"
    "/evaluation_result_closed_models/captioning_results_gemini-2.5-flash-thinking.jsonl"
    "/evaluation_result_closed_models/captioning_results_gemini-2.5-pro-thinking.jsonl"
    "/evaluation_result_closed_models/captioning_results_gpt-4o-2024-11-20.jsonl"
    "/evaluation_result_closed_models/captioning_results_gpt-4o-mini.jsonl"
)

# 定义 QA 任务文件列表
QA_FILES=(
    "/evaluation_result_closed_models/open_results_claude-3-7-sonnet-latest.jsonl"
    "/evaluation_result_closed_models/open_results_deepseek-vl2.jsonl"
    "/evaluation_result_closed_models/open_results_gemini-2.5-flash-thinking.jsonl"
    "/evaluation_result_closed_models/open_results_gemini-2.5-pro-thinking.jsonl"
    "/evaluation_result_closed_models/open_results_gpt-4o-2024-11-20.jsonl"
    "/evaluation_result_closed_models/open_results_gpt-4o-mini.jsonl"
    "/evaluate_open_source/evaluation_result_intervl78B/openended_result_intervl78B.jsonl"
    "/evaluate_open_source/evaluation_result_qwen72B/openended_result_qwen72B.jsonl"
    "/evaluate_open_source/evaluation_result_qwen32B/openended_result_qwen32B.json"
    "/evaluate_open_source/evaluation_result_qwen7B/openended_result_qwen7B.jsonl"
    "/evaluate_open_source/evaluation_result_llavaone72B/openended_result_llavaone72B.jsonl"
    "/evaluate_open_source/evaluation_result_internvl8B/openended_result_internvl8B.jsonl"
)

# 定义输出目录
OUTPUT_DIR="/evaluation_results_output"
mkdir -p "$OUTPUT_DIR"

# 定义模型名称
MODEL_NAME="Qwen2.5-VL-72B-Instruct"

# 处理 Captioning 任务
echo "Processing Captioning tasks..."
for INPUT_FILE in "${CAPTIONING_FILES[@]}"; do
    # 提取文件名并生成输出文件路径
    FILE_NAME=$(basename "$INPUT_FILE")
    OUTPUT_FILE="$OUTPUT_DIR/captioning_evaluation_${FILE_NAME}"

    # 运行 Python 脚本
    echo "Processing $INPUT_FILE..."
    python "$PYTHON_SCRIPT" --task "captioning" --input_file "$INPUT_FILE" --output_file "$OUTPUT_FILE" --model_name "$MODEL_NAME"
done

# 处理 QA 任务
echo "Processing QA tasks..."
for INPUT_FILE in "${QA_FILES[@]}"; do
    # 提取文件名并生成输出文件路径
    FILE_NAME=$(basename "$INPUT_FILE")
    OUTPUT_FILE="$OUTPUT_DIR/qa_evaluation_${FILE_NAME}"

    # 运行 Python 脚本
    echo "Processing $INPUT_FILE..."
    python "$PYTHON_SCRIPT" --task "qa" --input_file "$INPUT_FILE" --output_file "$OUTPUT_FILE" --model_name "$MODEL_NAME"
done

echo "All tasks completed. Results saved in $OUTPUT_DIR."
