#!/bin/bash

# python filter_hard_data47B.py \
# --save-dir /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/hard_mcq_qwen7B \
# --test /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark_data/easy_data_v2.jsonl \
# --model-name Qwen2.5-VL-7B-Instruct

python validate_hard47B.py \
--save-dir /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/hard_mcq_qwen7B \
--test /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/filtered_questions.jsonl \
--model-name Qwen2.5-VL-72B-Instruct