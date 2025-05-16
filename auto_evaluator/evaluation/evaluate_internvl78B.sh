#!/bin/bash

python evaluation_mcq.py \
--save-dir /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/evaluation_result_intervl3-78B \
--test /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/results/msearth_mcq.json \
--model-name InternVL3-78B

# 第一个命令
python evaluate_captioning.py \
--save-dir /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/evaluation_result_intervl3-78B \
--test /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark_data/captioning_sample.jsonl \
--model-name InternVL3-78B

# 第二个命令
python evaluation_open.py \
--save-dir /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/evaluation_result_intervl3-78B \
--test /fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/results/msearth_open.json \
--model-name InternVL3-78B