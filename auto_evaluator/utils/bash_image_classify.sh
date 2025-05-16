#!/bin/bash

# 设置输入文件路径
INPUT_FILE_1="/neurips_mmearth_benchmark/benchmark/classification_results_merged_mcq.json"
INPUT_FILE_2="/neurips_mmearth_benchmark/benchmark/classification_results_merge_open.json"

# 设置输出目录
OUTPUT_DIR="/neurips_mmearth_benchmark/benchmark/results/"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 设置模型名称
MODEL_NAME="Qwen2.5-VL-72B-Instruct"

# 处理第一个文件
echo "Processing file: $INPUT_FILE_1"
python image_qa_classify.py --test "$INPUT_FILE_1" --save-dir "$OUTPUT_DIR" --model-name "$MODEL_NAME"

# 处理第二个文件
echo "Processing file: $INPUT_FILE_2"
python image_qa_classify.py --test "$INPUT_FILE_2" --save-dir "$OUTPUT_DIR" --model-name "$MODEL_NAME"

echo "Processing complete. Results saved to $OUTPUT_DIR"
