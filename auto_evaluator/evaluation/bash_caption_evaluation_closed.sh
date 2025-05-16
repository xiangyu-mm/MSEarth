#!/bin/bash

# 设置代理环境变量（如有需要，可以注释掉）
export http_proxy=
export https_proxy=
export HF_HOME=

# 输入文件夹路径
INPUT_FOLDER="/fs-computility/ai4sData/zhaoxiangyu1//evaluation_result_closed_models"

# Python 脚本路径
SCRIPT1_PATH="/fs-computility/ai4sData/zhaoxiangyu1//evaluation/evaluate_caption_metric.py"
SCRIPT2_PATH="/fs-computility/ai4sData/zhaoxiangyu1//evaluation/evaluate_openended_metric.py"

# 检查输入文件夹是否存在
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder '$INPUT_FOLDER' does not exist."
    exit 1
fi

# 遍历文件夹中的所有 captioning JSON 文件
# for CAPTIONING_FILE in "$INPUT_FOLDER"/captioning_results_gemini-2.5-flash-preview-*.jsonl; do
#     # 检查文件是否存在
#     if [ -f "$CAPTIONING_FILE" ]; then
#         # 提取模型名称（去掉路径和前缀，只保留模型相关部分）
#         MODEL_NAME=$(basename "$CAPTIONING_FILE" | sed -E 's/^captioning_results_(.*)\.jsonl$/\1/')
        
#         # 定义输出文件路径，包含模型名称
#         CAPTIONING_OUTPUT="$INPUT_FOLDER/captioning_metrics_${MODEL_NAME}.json"
        
#         echo "Processing Captioning File: $CAPTIONING_FILE for Model: $MODEL_NAME"
        
#         # 调用 Python 脚本处理文件
#         python $SCRIPT1_PATH --input-file "$CAPTIONING_FILE" --output-file "$CAPTIONING_OUTPUT"
        
#         echo "Captioning metrics for $MODEL_NAME saved to: $CAPTIONING_OUTPUT"
#     else
#         echo "Warning: No captioning files found in '$INPUT_FOLDER'."
#     fi
# done

# 遍历文件夹中的所有 open-ended JSON 文件（如果需要处理 open-ended 文件，可以取消注释以下代码）
for OPENENDED_FILE in "$INPUT_FOLDER"/open_results_gemini-2.5-pro-thinking*.jsonl; do
    if [ -f "$OPENENDED_FILE" ]; then
        MODEL_NAME=$(basename "$OPENENDED_FILE" | sed -E 's/^openended_results_(.*)\.jsonl$/\1/')
        OPENENDED_OUTPUT="$INPUT_FOLDER/openended_metrics_${MODEL_NAME}.json"
        echo "Processing Open-ended File: $OPENENDED_FILE for Model: $MODEL_NAME"
        python $SCRIPT2_PATH --input-file "$OPENENDED_FILE" --output-file "$OPENENDED_OUTPUT"
        echo "Open-ended metrics for $MODEL_NAME saved to: $OPENENDED_OUTPUT"
    else
        echo "Warning: No open-ended files found in '$INPUT_FOLDER'."
    fi
done
