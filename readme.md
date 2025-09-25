<div align="center">

# MSEarth-Bench: A Benchmark for Multimodal Scientific Comprehension of Earth Science

<a href="https://huggingface.co/MSEarth-Data">
  <img src="https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface&logoColor=white" alt="HuggingFace">
</a>

</div>

---

### 🆕 News

MSEarthMCQ is now integrated into [**VLMEvalKit**](https://github.com/open-compass/VLMEvalKit/tree/b01d484ddc316ce2ce7f7f77f7b0c19f5f705f3b), enabling users to easily evaluate various models on the platform. Special thanks to [**OpenCompass**](https://github.com/open-compass) for their support!

## Abstract

The rapid advancement of multimodal large language models (MLLMs) has unlocked new opportunities to tackle complex scientific challenges. Despite this progress, their application in addressing earth science problems, especially at the graduate level, remains underexplored. A significant barrier is the absence of benchmarks that capture the depth and contextual complexity of geoscientific reasoning. Current benchmarks often rely on synthetic datasets or simplistic figure-caption pairs, which do not adequately reflect the intricate reasoning and domain-specific insights required for real-world scientific applications. To address these gaps, we introduce MSEarth, a multimodal scientific benchmark curated from high-quality, open-access scientific publications. MSEarth encompasses the five major spheres of Earth science: atmosphere, cryosphere, hydrosphere, lithosphere, and biosphere, featuring over 7K figures with refined captions. These captions are crafted from the original figure captions and enriched with discussions and reasoning from the papers, ensuring the benchmark captures the nuanced reasoning and knowledge-intensive content essential for advanced scientific tasks. MSEarth supports a variety of tasks, including scientific figure captioning, multiple choice questions, and open-ended reasoning challenges. By bridging the gap in graduate-level benchmarks, MSEarth provides a scalable and high-fidelity resource to enhance the development and evaluation of MLLMs in scientific reasoning. The benchmark is publicly available to foster further research and innovation in this field.

## MSEarth Evaluation Guide

This guide provides instructions for evaluating both open-sourced and proprietary Multimodal Large Language Models (MLLMs) using the MSEarth framework. You can download the benchmark dataset and images form https://huggingface.co/MSEarth.

### Evaluation for Open-Sourced MLLMs

#### Setup

1. Navigate to the deployment script directory:
    ```bash
    cd MSEarth/auto_evaluator/deploy_script
    ```
2. Choose your target model, e.g. Qwen2.5-VL-72B

    ```
    bash deploy_qwen72B.sh
    ```

#### Evaluation
1. Navigate to the evaluation directory:
    ```
    cd MSEarth/auto_evaluator/evaluation
    ```
2. Run the evaluation script:
    ```
    bash evaulate_qwen72B.sh
    ```
#### Customization

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

### Evaluation for Proprietary MLLMs

#### Execution

Run the following script to evaluate proprietary models:
`bash MSEarth/evaluation/bash_gpt4o.sh`
#### Model Selection

You can choose different models from the list provided in the script:
```
#!/bin/bash

CAPTIONING_FILE="MSEarth/evaluation/results/captioning_sample.jsonl"
OPEN_FILE="MSEarth/evaluation/results/msearth_open.json"
MCQ_FILE="MSEarth/evaluation/results/msearth_mcq.json"
SAVE_DIR="/evaluation_result_closed_models"

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

### Metrics

#### Captioning and Open-Ended Results

To evaluate captioning and open-ended results using metrics such as ROUGE1, ROUGE2, ROUGEL, METEOR, BLEU, and BERTSCORE, run:

```
bash MSEarth/evaluation/bash_llm_metrics.sh
```

#### MCQ Results

The evaluation metrics for MCQ results are automatically generated. You can count the number of entries where the "is_correct" field is true. If the evaluated model has poor instruction-following capabilities, you can run:

`python MSEarth/utils/post_process4llama_mcq.py`

This guide should help you effectively evaluate MLLMs using the MSEarth framework. Adjust paths and parameters as needed for your specific setup.
