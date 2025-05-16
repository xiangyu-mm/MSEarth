# MSEarth Evaluation Guide

This guide provides instructions for evaluating both open-sourced and proprietary Multimodal Large Language Models (MLLMs) using the MSEarth framework.

## Evaluation for Open-Sourced MLLMs

### Setup

1. Navigate to the deployment script directory:
    ```bash
    cd MSEarth/auto_evaluator/deploy_script
    ```
2. Choose your target model, e.g. Qwen2.5-VL-72B

    ```
    bash deploy_qwen72B.sh
    ```

### Evaluation
1. Navigate to the evaluation directory:
    ```
    cd MSEarth/auto_evaluator/evaluation
    ```
2. Run the evaluation script:
    ```
    bash evaulate_qwen72B.sh
    ```
### Customization

You can customize the save directory, test dataset, and model name by modifying the script:
```

#!/bin/bash

python evaluate_captioning.py \
--save-dir /fs-computility/ai4sData/ \
--test MSEarth/auto_evaluator/results/captioning_sample.json \
--model-name Qwen2.5-VL-72B-Instruct

python evaluation_open.py \
--save-dir /fs-computility/ai4sData/ \
--test MSEarth/auto_evaluator/results/msearth_open.json \
--model-name Qwen2.5-VL-72B-Instruct

python evaluation_mcq.py \
--save-dir your dir \
--test MSEarth/auto_evaluator/results/msearth_mcq.json \
--model-name Qwen2.5-VL-72B-Instruct
```
The generated answers will be saved in your specified directory.

## Evaluation for Proprietary MLLMs

### Execution

Run the following script to evaluate proprietary models:
`bash MSEarth/evaluation/bash_gpt4o.sh`
### Model Selection

You can choose different models from the list provided in the script:
```
#!/bin/bash

CAPTIONING_FILE="/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark_data/captioning_sample.jsonl"
OPEN_FILE="/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/results/msearth_open.json"
MCQ_FILE="/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/results/msearth_mcq.json"
SAVE_DIR="/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/evaluation_result_closed_models"

MODELS=("gpt-4o-2024-11-20", "claude-3-5-haiku-latest", "claude-3-7-sonnet-latest","gpt-4o-mini", "gemini-2.5-flash-thinking","gemini-2.5-pro-thinking")

for MODEL in "${MODELS[@]}"; do
    echo "Processing model: $MODEL"

    python evaluate_closed_models.py \
        --captioning-file "$CAPTIONING_FILE" \
        --open-file "$OPEN_FILE" \
        --mcq-file "$MCQ_FILE" \
        --save-dir "$SAVE_DIR" \
        --model "$MODEL" &

done

wait

echo "All tasks completed!"
```

## Metrics

### Captioning and Open-Ended Results

To evaluate captioning and open-ended results using metrics such as ROUGE1, ROUGE2, ROUGEL, METEOR, BLEU, and BERTSCORE, run:

```
bash MSEarth/evaluation/bash_llm_metrics.sh
```

### MCQ Results

The evaluation metrics for MCQ results are automatically generated. You can count the number of entries where the "is_correct" field is true. If the evaluated model has poor instruction-following capabilities, you can run:

`python MSEarth/utils/post_process4llama_mcq.py`

This guide should help you effectively evaluate MLLMs using the MSEarth framework. Adjust paths and parameters as needed for your specific setup.
# MSEarth
