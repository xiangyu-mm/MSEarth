#!/bin/bash

python evaluation_mcq.py \
--save-dir /neurips_mmearth_benchmark/evaluation_result_intervl3-78B \
--test /neurips_mmearth_benchmark/benchmark/results/msearth_mcq.json \
--model-name InternVL3-78B

# 第一个命令
python evaluate_captioning.py \
--save-dir /neurips_mmearth_benchmark/evaluation_result_intervl3-78B \
--test /neurips_mmearth_benchmark/benchmark_data/captioning_sample.jsonl \
--model-name InternVL3-78B

# 第二个命令
python evaluation_open.py \
--save-dir /neurips_mmearth_benchmark/evaluation_result_intervl3-78B \
--test /neurips_mmearth_benchmark/benchmark/results/msearth_open.json \
--model-name InternVL3-78B