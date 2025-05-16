#!/bin/bash

# 设置输入文件路径
INPUT_FILE1="/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/captioning_sample.json"
INPUT_FILE2="/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/merge_open.json"
INPUT_FILE3="/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/merged_mcq.json"

# 设置输出目录
OUTPUT_DIR="/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/"

# 设置模型名称
MODEL_NAME="Qwen2.5-VL-72B-Instruct"

# 处理第一个文件
echo "Processing file: $INPUT_FILE1"
python subject_classify.py \
    --test "$INPUT_FILE1" \
    --save-dir "$OUTPUT_DIR" \
    --model-name "$MODEL_NAME"

# 处理第二个文件
echo "Processing file: $INPUT_FILE2"
python subject_classify.py \
    --test "$INPUT_FILE2" \
    --save-dir "$OUTPUT_DIR" \
    --model-name "$MODEL_NAME"

# 处理第三个文件
echo "Processing file: $INPUT_FILE3"
python subject_classify.py \
    --test "$INPUT_FILE3" \
    --save-dir "$OUTPUT_DIR" \
    --model-name "$MODEL_NAME"

echo "Processing completed. Results saved in $OUTPUT_DIR"