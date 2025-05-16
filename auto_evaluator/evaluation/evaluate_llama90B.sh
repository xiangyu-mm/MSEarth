#!/bin/bash

# # 第一个命令
# python evaluate_captioning.py \
# --save-dir /neurips_mmearth_benchmark/evaluation_result_llama90B \
# --test /neurips_mmearth_benchmark/benchmark_data/captioning_sample.jsonl \
# --model-name Llama-3.2-90B-Vision-Instruct

# # 第二个命令
# python evaluation_open.py \
# --save-dir /neurips_mmearth_benchmark/evaluation_result_llama90B \
# --test /neurips_mmearth_benchmark/benchmark_data/merge_open_caption_no_caption.jsonl \
# --model-name Llama-3.2-90B-Vision-Instruct

python evaluation_mcq.py \
--save-dir /neurips_mmearth_benchmark/evaluation_result_llama90B \
--test /neurips_mmearth_benchmark/benchmark_data/merged_mcq_data.jsonl \
--model-name Llama-3.2-90B-Vision-Instruct