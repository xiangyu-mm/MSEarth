#!/bin/bash

# # 第一个命令
# python evaluate_captioning.py \
# --save-dir /neurips_mmearth_benchmark/evaluation_result_qwen32B \
# --test /neurips_mmearth_benchmark/benchmark_data/captioning_sample.jsonl \
# --model-name Qwen2.5-VL-32B-Instruct

# # 第二个命令
# python evaluation_open.py \
# --save-dir /neurips_mmearth_benchmark/evaluation_result_qwen32B \
# --test /neurips_mmearth_benchmark/benchmark_data/merge_open_caption_no_caption.jsonl \
# --model-name Qwen2.5-VL-32B-Instruct

python evaluation_mcq.py \
--save-dir /neurips_mmearth_benchmark/evaluation_result_qwen32B \
--test /neurips_mmearth_benchmark/benchmark_data/merged_mcq_data.jsonl \
--model-name Qwen2.5-VL-32B-Instruct