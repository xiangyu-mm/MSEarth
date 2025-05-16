#!/bin/bash

# # # 第一个命令
# python evaluate_captioning.py \
# --save-dir /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/evaluation_result_llamavl11B \
# --test /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark_data/captioning_sample.jsonl \
# --model-name Llama-3.2-11B-Vision-Instruct

# # 第二个命令
# python evaluation_open.py \
# --save-dir /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/evaluation_result_llamavl11B \
# --test /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark_data/merge_open_caption_no_caption.jsonl \
# --model-name Llama-3.2-11B-Vision-Instruct

python evaluation_mcq.py \
--save-dir /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/evaluate_open_source/evaluation_result_llamavl11B \
--test /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark_data/merged_mcq_data.jsonl \
--model-name Llama-3.2-11B-Vision-Instruct