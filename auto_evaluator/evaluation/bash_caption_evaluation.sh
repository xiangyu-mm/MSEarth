#!/bin/bash

# 设置代理环境变量（如有需要，可以注释掉）
export http_proxy=
export https_proxy=
export HF_HOME=

# 输入文件夹路径
INPUT_FOLDER="/neurips_mmearth_benchmark/evaluation_result_closed_models"

# Python 脚本路径
SCRIPT1_PATH="/neurips_mmearth_benchmark/evaluation/evaluate_caption_metric.py"
SCRIPT2_PATH="/neurips_mmearth_benchmark/evaluation/evaluate_openended_metric.py"

# 检查输入文件夹是否存在
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder '$INPUT_FOLDER' does not exist."
    exit 1
fi

# 定义 JSON 文件和输出文件路径
CAPTIONING_FILE="$INPUT_FOLDER/captioning_results_gemini-2.5-flash-preview-04-17.jsonl"
OPENENDED_FILE="$INPUT_FOLDER/open_results_gemini-2.5-flash-preview-04-17.jsonl"
CAPTIONING_OUTPUT="$INPUT_FOLDER/gemini-2.5-flash-preview-04-17_captioning_metrics.json"
OPENENDED_OUTPUT="$INPUT_FOLDER/gemini-2.5-flash-preview-04-17_openended_metrics.json"

# 检查 Captioning JSON 文件是否存在
if [ -f "$CAPTIONING_FILE" ]; then
    echo "Processing Captioning File: $CAPTIONING_FILE"
    python $SCRIPT1_PATH --input-file "$CAPTIONING_FILE" --output-file "$CAPTIONING_OUTPUT"
    echo "Captioning metrics saved to: $CAPTIONING_OUTPUT"
else
    echo "Warning: Captioning file '$CAPTIONING_FILE' not found."
fi

# 检查 Open-ended JSON 文件是否存在
if [ -f "$OPENENDED_FILE" ]; then
    echo "Processing Open-ended File: $OPENENDED_FILE"
    python $SCRIPT2_PATH --input-file "$OPENENDED_FILE" --output-file "$OPENENDED_OUTPUT"
    echo "Open-ended metrics saved to: $OPENENDED_OUTPUT"
else
    echo "Warning: Open-ended file '$OPENENDED_FILE' not found."
fi